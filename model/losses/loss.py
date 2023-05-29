import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from model.losses.desc_loss import DescriptionLoss
from model.losses.location_loss import LocationLoss


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()
    
    def forward_orig(self, grad_fake, grad_real):
        prod = ( grad_fake[:,:,None,:] @ grad_real[:,:,:,None] ).squeeze(-1).squeeze(-1)
        fake_norm = torch.sqrt( torch.sum( grad_fake**2, dim=-1 ) )
        real_norm = torch.sqrt( torch.sum( grad_real**2, dim=-1 ) )
        return 1 - torch.mean( prod/(fake_norm*real_norm))

    def forward(self, grad_fake, grad_real, valid_mask):
        grad_fake = grad_fake[valid_mask]
        grad_real = grad_real[valid_mask]
        prod = ( grad_fake[None,:] @ grad_real[:,None])
        fake_norm = torch.sqrt(torch.sum( grad_fake**2, dim=-1 ) )
        real_norm = torch.sqrt(torch.sum( grad_real**2, dim=-1 ) )
        return 1 - torch.mean(prod/(fake_norm*real_norm))

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
    
    def imgrad(self, img):
        img = torch.mean(img, 1, True)
        fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv1.weight = nn.Parameter(weight)
        grad_x = conv1(img)

        fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv2.weight = nn.Parameter(weight)
        grad_y = conv2(img)

    #     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))
        
        return grad_y, grad_x

    def imgrad_yx(self, img):
        N,C,_,_ = img.size()
        grad_y, grad_x = self.imgrad(img)
        return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)

    # L1 norm
    def forward(self, grad_fake, grad_real, valid_mask):
        return torch.sum(torch.mean(torch.abs((grad_real - grad_fake)[valid_mask])))

class BerHu(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold
    
    def forward(self, fake, real):
        mask = real>0
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        fake = fake * mask
        diff = torch.abs(real-fake)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()[0]

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
        part2 = part2 / (2.*delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class PlainLoss(torch.nn.Module):
    def __init__(self, config):
        super(PlainLoss, self).__init__()
        self.MSELoss = nn.MSELoss()
        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.MaskedL1Loss = MaskedL1Loss()
        self.BCELoss = nn.BCELoss(reduction='none')
        self.FocalLoss = FocalLoss(2, 1 - 0.001)
        self.GradLoss = GradLoss()
        self.NormalLoss = NormalLoss()
        self.DescriptorLoss = DescriptionLoss()
        self.LocationLoss = LocationLoss()
        # self.negative_weight_factor = 5.0
        self.config = config
        self.depth_pred_loss_weight = self.get_weights('depth_pred_loss_weight')
        self.heatmap_loss_weight = self.get_weights('heatmap_loss_weight')
        self.heatmap_rgb_loss_weight = self.get_weights('heatmap_rgb_loss_weight')
        self.heatmap_pcd_loss_weight = self.get_weights('heatmap_pcd_loss_weight')
        self.heatmap_consistency_loss = self.get_weights('heatmap_consistency_loss')
        self.coord_projection_loss = self.get_weights('coord_projection_loss')
        self.chamfer_loss_weight = self.get_weights('chamfer_loss_weight')
        self.descriptor_loss_weight = self.get_weights('descriptor_loss_weight')
    
    def get_weights(self, name, default_val=0):
        return self.config['loss'][name] if name in self.config['loss'] else default_val

    def forward(self, loss_input):
        epoch, output_depth, output_heatmap, target_depth, target_heatmap = \
            loss_input['epoch'], loss_input['output_depth'], loss_input['output_heatmap'], \
                loss_input['target_depth'], loss_input['target_heatmap']
        chamfer_offset_map = loss_input.get('chamfer_offset_map')
        output_descriptor = loss_input.get('output_descriptor')
        projection_pts_info = loss_input.get('projection_pts_info')
        loss_dict = {}
        losses = []

        # depth pred loss
        if (self.depth_pred_loss_weight > 0) and (not output_depth is None):
            # mask out zero value
            # mask = torch.where(target_depth == 0, torch.full_like(target_depth, 0), torch.full_like(target_depth, 1))
            # depth_pred_loss = self.depth_pred_loss_weight * (mask * self.MSELoss(output_depth, target_depth)).mean()
            # MaskedL1Loss
            depth_pred_loss = self.MaskedL1Loss(output_depth, target_depth)
            # grad loss
            output_grad, target_grad = self.GradLoss.imgrad_yx(output_depth), self.GradLoss.imgrad_yx(target_depth)
            # valid_mask contain two channel (y,x)
            N,_,_ = output_grad.size()
            valid_mask = (target_depth>0).detach()
            valid_mask_yx = torch.cat([valid_mask, valid_mask], dim=1)
            valid_mask_yx = valid_mask_yx.view(N,2,-1)
            # if epoch > 3:
            #     depth_pred_loss += self.GradLoss(output_grad, target_grad, valid_mask_yx)
            # # normal loss
            # if epoch > 5:
            #     depth_pred_loss += self.NormalLoss(output_grad, target_grad, valid_mask_yx)
            # add weight
            loss_dict['depth_pred_loss'] = depth_pred_loss.item()
            depth_pred_loss *= self.depth_pred_loss_weight
            losses.append(depth_pred_loss)

        # keypoint heatmap loss
        if self.heatmap_loss_weight > 0:
            mask = target_depth > 0
            # negative_weight = 1e4 / (640 * 480)
            negative_weight = self.config['loss']['heatmap_negative_weight']
            positive_weight = 1. - negative_weight
            balance_weight = torch.where(target_heatmap > 0, torch.full_like(target_heatmap, positive_weight), torch.full_like(target_heatmap, negative_weight))
            B, _, H, W = target_depth.shape
            if output_heatmap.shape[2] != H and output_heatmap.shape[3] != W:
                # resize if need
                output_heatmap = torch.nn.functional.interpolate(output_heatmap, size=(H, W), mode='nearest')
            # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486/2
            heatmap_loss = (self.BCELoss(output_heatmap[mask], target_heatmap[mask]) * balance_weight[mask]).mean()
            # heatmap_loss = self.heatmap_loss_weight * (self.FocalLoss(output_heatmap, target_heatmap)).mean()
            loss_dict['heatmap_loss'] = heatmap_loss.item()
            heatmap_loss *= self.heatmap_loss_weight
            losses.append(heatmap_loss)
        
        # rgb tower keypoint heatmap loss
        if self.heatmap_rgb_loss_weight > 0 and loss_input.get('output_rgb_heatmap') != None:
            curr_heatmap = loss_input.get('output_rgb_heatmap')
            if curr_heatmap.shape[2] != H and curr_heatmap.shape[3] != W:
                # resize if need
                curr_heatmap = torch.nn.functional.interpolate(curr_heatmap, size=(H, W), mode='nearest')
            # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486/2
            rgb_heatmap_loss = (self.BCELoss(curr_heatmap[mask], target_heatmap[mask]) * balance_weight[mask]).mean()
            loss_dict['rgb_heatmap_loss'] = rgb_heatmap_loss.item()
            rgb_heatmap_loss *= self.heatmap_rgb_loss_weight
            losses.append(rgb_heatmap_loss)
        
        # pcd tower keypoint heatmap loss
        if self.heatmap_pcd_loss_weight > 0 and loss_input.get('output_pcd_heatmap') != None:
            curr_heatmap = loss_input.get('output_pcd_heatmap')
            if curr_heatmap.shape[2] != H and curr_heatmap.shape[3] != W:
                # resize if need
                curr_heatmap = torch.nn.functional.interpolate(curr_heatmap, size=(H, W), mode='nearest')
            # https://discuss.pytorch.org/t/unclear-about-weighted-bce-loss/21486/2
            pcd_target_heatmap = target_heatmap
            # use fuse target to boost up pcd feature learning
            # pcd_target_heatmap = output_heatmap.detach()
            pcd_heatmap_loss = (self.BCELoss(curr_heatmap[mask], pcd_target_heatmap[mask]) * balance_weight[mask]).mean()
            loss_dict['pcd_heatmap_loss'] = pcd_heatmap_loss.item()
            pcd_heatmap_loss *= self.heatmap_pcd_loss_weight
            losses.append(pcd_heatmap_loss)

        # keypoint location loss
        if self.heatmap_consistency_loss > 0 or self.coord_projection_loss > 0:
            location_loss = self.LocationLoss(output_heatmap, loss_input['output_coord'], projection_pts_info)
            if location_loss != None:
                loss_dict.update(location_loss)
                if 'score_consistency_loss' in location_loss:
                    losses.append(self.heatmap_consistency_loss * location_loss['score_consistency_loss'])
                if 'rpe_loss' in location_loss:
                    losses.append(self.coord_projection_loss * location_loss['rpe_loss'])

        # chamfer loss
        if self.chamfer_loss_weight > 0 and (not chamfer_offset_map is None):
            # chamfer_loss = self.chamfer_loss_weight * (output_heatmap * chamfer_offset_map).mean()
            mask = chamfer_offset_map > 0
            if mask.sum() > 0:
                zero_heatmap = torch.full_like(target_heatmap, 0)
                chamfer_loss = (self.BCELoss(output_heatmap[mask], zero_heatmap[mask]) * torch.pow(chamfer_offset_map[mask], 2)).mean()
                loss_dict['chamfer_loss'] = chamfer_loss.item()
                chamfer_loss *= self.chamfer_loss_weight
                losses.append(chamfer_loss)

        # descriptor loss
        if self.descriptor_loss_weight > 0 and output_descriptor != None and projection_pts_info != None:
            desc_loss = self.DescriptorLoss(output_descriptor, projection_pts_info)
            if desc_loss != None:
                loss_dict['desc_loss'] = desc_loss.item()
                desc_loss = self.descriptor_loss_weight * desc_loss
                losses.append(desc_loss)

        loss = sum(losses)
        return loss, loss_dict
