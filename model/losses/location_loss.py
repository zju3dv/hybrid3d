import numpy as np
import torch
import torch.nn as nn

class LocationLoss(nn.Module):
    def __init__(self):
        super(LocationLoss, self).__init__()
    
    def forward(self, heatmap, coord, back_projection_info):
        """
            - feats: dense features BxDxHxW
            - back_projection_info: list B x [ Nx2 coordinates(X, Y); nx2 labals(labelIdx), H, W ]
        """
        total_N = sum([len(x) for _, x, _, _ in back_projection_info])
        if total_N <= 2:
            return None
        kpt_scores = []
        kpt_proj_errors = []
        kpt_labels = []
        for idx, (coords, labels, H, W) in enumerate(back_projection_info):
            if coords.shape[0] == 0:
                continue
            grids_norm = torch.from_numpy(coords).float().reshape(1, -1, 1, 2)
            grids_norm[:,:,:,0] *= 2/(W-1)
            grids_norm[:,:,:,1] *= 2/(H-1)
            grids_norm -= 1
            # collect scores(heats)
            single_scores = nn.functional.grid_sample(heatmap[idx, None, ...],\
                grids_norm.to(heatmap.device), mode='nearest', padding_mode='border', align_corners=True)
            # compute reprojection error
            single_proj_errors = nn.functional.grid_sample(coord[idx, None, ...],\
                grids_norm.to(coord.device), mode='nearest', padding_mode='border', align_corners=True)
            single_proj_errors = single_proj_errors[0, :, :, 0].transpose(1, 0)
            single_proj_errors = torch.norm(single_proj_errors - torch.from_numpy(coords).to(single_proj_errors.device), dim=1)
            single_scores = single_scores.squeeze()
            single_proj_errors = single_proj_errors.squeeze()
            if coords.shape[0] == 1: # avoid zero dimentional problem in cat
                single_scores = single_scores.unsqueeze(0)
                single_proj_errors = single_proj_errors.unsqueeze(0)
            kpt_scores.append(single_scores)
            kpt_proj_errors.append(single_proj_errors)
            kpt_labels.append(torch.from_numpy(labels))
        if len(kpt_scores) == 0:
            return None
        kpt_scores = torch.cat(kpt_scores)
        kpt_proj_errors = torch.cat(kpt_proj_errors)
        kpt_labels = torch.cat(kpt_labels)

        unique_labels = torch.unique(kpt_labels)
        if unique_labels.numel() < 2:
            return None

        # minize reprojection error
        mean_proj_error = torch.mean(kpt_proj_errors)

        # return mean_proj_error

        score_losses = []
        for label in unique_labels:
            mask = kpt_labels == label
            if torch.sum(mask) < 2:
                continue
            selected_scores = kpt_scores[mask]
            selected_proj_errors = kpt_proj_errors[mask]
            # 1. minimize scores variance with same label
            # 2. good points have smaller reprojection error
            score_losses.append(torch.var(selected_scores) + 
                torch.mean(selected_scores) * (torch.mean(selected_proj_errors) - mean_proj_error))
            # score_losses.append(torch.var(selected_scores))
        if len(score_losses) == 0:
            return {'rpe_loss': mean_proj_error}
        else:
            return {
                'score_consistency_loss': sum(score_losses) / len(score_losses),
                'rpe_loss': mean_proj_error
            }
            
