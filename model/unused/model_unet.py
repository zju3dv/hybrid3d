import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.unet.unet_parts import *

# adapt from d2-net
class SoftDetectionModule(nn.Module):
    def __init__(self, soft_local_max_size=3):
        super(SoftDetectionModule, self).__init__()

        self.soft_local_max_size = soft_local_max_size

        self.pad = self.soft_local_max_size // 2

    def forward(self, batch):
        b = batch.size(0)

        batch = F.relu(batch)

        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
        # normalize before perform soft local-max
        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
        sum_exp = (
            self.soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
                self.soft_local_max_size, stride=1
            )
        )
        local_max_score = exp / sum_exp

        depth_wise_max = torch.max(batch, dim=1)[0]
        depth_wise_max_score = batch / depth_wise_max.unsqueeze(1)

        all_scores = local_max_score * depth_wise_max_score
        score = torch.max(all_scores, dim=1)[0]

        score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1)

        return score

# assume some channel of positive features
class SoftDetectionModule_v2(nn.Module):
    def __init__(self, soft_local_max_size=3, n_positive_features=64):
        super(SoftDetectionModule_v2, self).__init__()

        self.soft_local_max_size = soft_local_max_size

        self.pad = self.soft_local_max_size // 2
        
        self.n_positive_features = n_positive_features

    def forward(self, batch):
        b = batch.size(0)

        batch = F.relu(batch)

        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
        # normalize before perform soft local-max
        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
        sum_exp = (
            self.soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [self.pad] * 4, mode='replicate'),
                self.soft_local_max_size, stride=1
            )
        )
        sum_exp = torch.sum(sum_exp, dim=1, keepdim=True)
        local_max_score = exp / sum_exp

        # all_scores = local_max_score * batch
        # score = torchmax(all_scores, dim=1)[0]

        # score = score / torch.sum(score.view(b, -1), dim=1).view(b, 1, 1)
        score = torch.sum(local_max_score[:, :self.n_positive_features, :, :], dim=1, keepdim=True)

        return score

# just remove normalization in last step (sum to 1)
class SoftDetectionModule_v3(nn.Module):
    def __init__(self, soft_local_max_size=3):
        super(SoftDetectionModule_v3, self).__init__()

        self.soft_local_max_size = soft_local_max_size

        self.pad = self.soft_local_max_size // 2

    def forward(self, batch):
        b = batch.size(0)

        batch = F.relu(batch)

        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
        # normalize before perform soft local-max
        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
        sum_exp = (
            self.soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
                self.soft_local_max_size, stride=1
            )
        )
        local_max_score = exp / sum_exp

        depth_wise_max = torch.max(batch, dim=1)[0]
        depth_wise_max_score = batch / depth_wise_max.unsqueeze(1)

        all_scores = local_max_score * depth_wise_max_score
        score = torch.max(all_scores, dim=1)[0]

        return score

# assume 1 detect and 63 dustbin
class SoftDetectionModule_v4(nn.Module):
    def __init__(self, soft_local_max_size=3, in_features=64):
        super(SoftDetectionModule_v4, self).__init__()

        self.soft_local_max_size = soft_local_max_size

        self.pad = self.soft_local_max_size // 2
        
        self.out_detect_conv = nn.Conv2d(in_features, 2, 1) # detection and dustbin

    def forward(self, batch):

        batch = self.out_detect_conv(batch)
        exp = torch.exp(batch)
        sum_exp = (
            self.soft_local_max_size ** 2 *
            F.avg_pool2d(
                F.pad(exp, [self.pad] * 4, mode='replicate'),
                self.soft_local_max_size, stride=1
            )
        )
        sum_exp = torch.sum(sum_exp, dim=1, keepdim=True)
        local_max_score = exp / sum_exp

        score = local_max_score[:, 0, :, :]

        return score

class SigmoidDetection(nn.Module):
    def __init__(self, n_positive_features=64):
        super(SigmoidDetection, self).__init__()
        self.Sigmoid = nn.Sigmoid()
        self.n_positive_features = n_positive_features

    def forward(self, x):
        # x = torch.sum(x[:, 0 : self.n_positive_features, :, :], dim=1)
        x, _ = torch.max(x[:, 0 : self.n_positive_features, :, :], dim=1, keepdim=True)
        x = self.Sigmoid(x)
        return x

class ConvSigmoid(nn.Module):
    def __init__(self, n_features=64):
        super(ConvSigmoid, self).__init__()
        self.Sigmoid = nn.Sigmoid()
        self.outconv = outconv(n_features, 1)

    def forward(self, x):
        x = self.outconv(x)
        x = self.Sigmoid(x)
        return x


class HybridPlainModel(BaseModel):
    def __init__(self, n_features=64, fix_depth_pred_param=False):
        super().__init__()
        self.inc = inconv(4, n_features)
        self.down1 = down(n_features, n_features*2)
        self.down2 = down(n_features*2, n_features*4)
        self.down3 = down(n_features*4, n_features*8)
        self.down4 = down(n_features*8, n_features*8)

        self.up_smooth_1 = up(n_features*16, n_features*4)
        self.up_smooth_2 = up(n_features*8, n_features*2)
        self.up_smooth_3 = up(n_features*4, n_features)
        self.up_smooth_4 = up(n_features*2, n_features)
        self.outc_smooth = outconv(n_features, 1)

        self.up_detect_1 = up(n_features*16, n_features*4)
        self.up_detect_2 = up(n_features*8, n_features*2)
        self.up_detect_3 = up(n_features*4, n_features)
        self.up_detect_4 = up(n_features*2, n_features)

        # self.outc_detect = SoftDetectionModule(soft_local_max_size=3)
        # self.outc_detect = SoftDetectionModule_v3(soft_local_max_size=9)
        # self.outc_detect = SigmoidDetection(n_positive_features=64)
        self.outc_detect = ConvSigmoid(n_features=n_features)
        # self.outc_detect = SoftDetectionModule_v2(soft_local_max_size=9, n_positive_features=60)
        # self.outc_detect = SoftDetectionModule_v4(soft_local_max_size=5, in_features=n_features)

        if fix_depth_pred_param:
            depth_pred_params = list(self.inc.parameters())
            depth_pred_params += list(self.down1.parameters())
            depth_pred_params += list(self.down2.parameters())
            depth_pred_params += list(self.down3.parameters())
            depth_pred_params += list(self.down4.parameters())
            depth_pred_params += list(self.up_smooth_1.parameters())
            depth_pred_params += list(self.up_smooth_2.parameters())
            depth_pred_params += list(self.up_smooth_3.parameters())
            depth_pred_params += list(self.up_smooth_4.parameters())
            depth_pred_params += list(self.outc_smooth.parameters())
            for param in depth_pred_params:
                param.requires_grad = False
            print('Model : fix_depth_pred_param!')

    def forward(self, rgb, depth):
        #TODO separate rgb and depth

        x = torch.cat([rgb, depth], dim=1)

        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # smooth decoder
        x_s = self.up_smooth_1(x5, x4)
        x_s = self.up_smooth_2(x_s, x3)
        x_s = self.up_smooth_3(x_s, x2)
        x_s = self.up_smooth_4(x_s, x1)
        x_s = self.outc_smooth(x_s)
        # x_s = torch.squeeze(x_s, dim=1)

        # detection decoder
        x_d = self.up_detect_1(x5, x4)
        x_d = self.up_detect_2(x_d, x3)
        x_d = self.up_detect_3(x_d, x2)
        x_d = self.up_detect_4(x_d, x1)
        x_d = self.outc_detect(x_d)
        # x_d = torch.squeeze(x_d, dim=1)

        return x_s, x_d
