import numpy as np
import torch
import torch.nn as nn
from model.losses.ap_loss import APLoss
from model.losses.circle_loss import CircleLoss, convert_label_to_similarity
from model.losses.siamese_triplet_loss import *
from utils.utils_algo import nms_fast

class DescriptionLoss(nn.Module):
    def __init__(self):
        super(DescriptionLoss, self).__init__()
        self.aploss = APLoss(nq=20)
        self.circleloss = CircleLoss(m=0.25, gamma=64)
        self.contrastive_loss = OnlineContrastiveLoss(1.0, HardNegativePairSelector(cpu=False))
        self.loss_type = 'APLoss'
        # self.loss_type = 'CircleLoss'
        # self.loss_type = 'ContrastiveLoss'
        self.safe_radius = 32
    
    def forward(self, feats, back_projection_info):
        """
            - feats: dense features BxDxHxW
            - back_projection_info: list B x [ Nx2 coordinates(X, Y); nx2 labals(labelIdx), H, W ]
        """
        B, C, _, _ = feats.shape
        total_N = sum([len(x) for _, x, _, _ in back_projection_info])
        if total_N <= 2:
            return None

        kpt_features = []
        kpt_labels = []
        too_close_labels = np.empty((0), dtype=int)
        for idx, (coords, labels, H, W) in enumerate(back_projection_info):
            # safe radius
            if self.safe_radius > 0:
                # find if label exist in current too_close_labels list
                mask_too_close = np.in1d(labels, too_close_labels)
                # remove such exist too close labels
                coords = coords[~mask_too_close, ...]
                labels = labels[~mask_too_close]
                # find too close labels via nms
                in_corners = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
                _, remain_ind = nms_fast(in_corners.transpose(1, 0), H, W, self.safe_radius)
                coords = coords[remain_ind, ...]
                labels_orig = labels.copy()
                labels = labels[remain_ind]
                # append new close label to too_close_labels
                too_close_labels = np.concatenate([np.setdiff1d(labels_orig, labels), too_close_labels])

            grids = torch.from_numpy(coords).float().reshape(1, -1, 1, 2)
            grids[:,:,:,0] *= 2/(W-1)
            grids[:,:,:,1] *= 2/(H-1)
            grids -= 1
            single_feats = nn.functional.grid_sample(feats[idx, None, ...], grids.to(feats.device), mode='bilinear', padding_mode='border', align_corners=True)
            kpt_features.append(single_feats.reshape(C, -1))
            kpt_labels.append(torch.from_numpy(labels))
        kpt_features = nn.functional.normalize(torch.cat(kpt_features, dim=1).transpose(1, 0), dim=1)
        kpt_labels = torch.cat(kpt_labels)
        
        # update total_N
        if self.safe_radius > 0: total_N = kpt_labels.shape[0]
        if total_N == 0:
            return None
        if self.loss_type == 'APLoss':
            scores = torch.matmul(kpt_features, kpt_features.transpose(1, 0))
            gt = kpt_labels[:, None].expand(-1, total_N) == kpt_labels[:, None].expand(-1, total_N).transpose(1, 0)
            gt = gt.to(scores.device)
            return (1 - self.aploss(scores, gt)).mean()
        elif self.loss_type == 'CircleLoss':
            inp_sp, inp_sn = convert_label_to_similarity(kpt_features, kpt_labels)
            return self.circleloss(inp_sp, inp_sn) / total_N
        elif self.loss_type == 'ContrastiveLoss':
            return self.contrastive_loss(kpt_features, kpt_labels)
