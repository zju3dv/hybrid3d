import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('.')

from model.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule, PointnetSAModuleBalanced
from model.pointnet2.pointnet2_utils import ball_query

class VoteFusionModuleFCGFOnly(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # pcd conv module
        self.config = config
        self.out_dim = config['arch']['vote_out_dim']

        self.conv_model_type = config['arch']['vote_pcd_conv_model']

        if self.conv_model_type == 'fcgf':
            from model.model_pcd import MinokowskiLayer
            self.pcd_conv = MinokowskiLayer(fix_weight=config['arch'].get('fix_pcd_conv_weight', True))
        elif self.conv_model_type == 'D3Feat':
            from model.d3feat.model_d3feat import D3FeatModule
            self.pcd_conv = D3FeatModule(fix_weight=config['arch'].get('fix_pcd_conv_weight', True), dataset_type='3dmatch_0.03')
        elif self.conv_model_type == 'fcgf_official':
            from model.fcgf.model_fcgf import FCGFModule
            self.pcd_conv = FCGFModule(fix_weight=config['arch'].get('fix_pcd_conv_weight', True))

        self.balance_weight = config['arch'].get('balance_weight', 0.0) if config != None else 0.0

        print('VoteFusionModuleFCGFOnly start with balanced weight: ', self.balance_weight)

        self.multi_fuse_layer = self.config['arch']['multi_fuse_layer']
        self.extra_feature_header = self.config['arch']['extra_feature_header']
        self.nms_radius = self.config['arch'].get('score_nms_radius', 0)

        print('VoteFusionModuleFCGFOnly nms radius: ', self.nms_radius)

        self.N_anchor = self.config['arch']['fuse_anchor']
        self.N_anchor_extra = self.config['arch']['fuse_anchor_extra']
        self.radius = 0.1

        if self.balance_weight > 0:
            self.fuse_sa1 = PointnetSAModuleBalanced(
                npoint=self.N_anchor,
                radius=self.radius,
                nsample=32,
                mlp=[32+128, 256, 256, 256],
                use_xyz=True,
                normalize_xyz=True,
                balance_weight=self.balance_weight,
                npoints_extra=self.N_anchor_extra,
            )
        else:
            self.fuse_sa1 = PointnetSAModuleVotes(
                npoint=self.N_anchor,
                radius=self.radius,
                nsample=32,
                mlp=[32+128, 256, 256, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fuse_layer_out_dim = 128 if self.extra_feature_header else self.out_dim
        self.fuse_fp1 = PointnetFPModule(mlp=[32+128+256,256,self.fuse_layer_out_dim])

        if self.multi_fuse_layer:
            self.fuse_sa2 = PointnetSAModuleVotes(
                npoint=self.N_anchor // 2,
                radius=self.radius * 2,
                nsample=32,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )
            # self.fuse_sa3 = PointnetSAModuleVotes(
            #     npoint=512,
            #     radius=self.radius * 4,
            #     nsample=16,
            #     mlp=[256, 128, 128, 256],
            #     use_xyz=True,
            #     normalize_xyz=True
            # )
            self.fuse_fp2 = PointnetFPModule(mlp=[256+256,256,256])
            # self.fuse_fp3 = PointnetFPModule(mlp=[256+256,256,256])

        # vote and aggregate feature
        vote_c = self.fuse_layer_out_dim
        self.relu = torch.nn.ReLU(inplace=True)
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

        self.score_type = self.config['arch']['score_type']
        if self.score_type == 'header':
            self.vote_score_header = nn.Sequential(*list(filter(None.__ne__, [
                nn.Conv1d(vote_c, vote_c, 1),
                nn.BatchNorm1d(vote_c),
                self.relu,
                nn.Conv1d(vote_c, 1, 1),
                nn.Sigmoid(),
                # nn.Softplus(),
            ])))
        
        if self.config['arch']['extra_feature_header']:
            self.vote_feature_header = nn.Sequential(*list(filter(None.__ne__, [
                nn.Conv1d(vote_c, vote_c, 1),
                nn.BatchNorm1d(vote_c),
                self.relu,
                nn.Conv1d(vote_c, vote_c, 1),
                nn.BatchNorm1d(vote_c),
                self.relu,
                nn.Conv1d(vote_c, self.out_dim, 1),
            ])))

        self.ball_query = ball_query
        self.neighbor_radius = config['arch']['neighbor_radius']
        self.neighbor_max_sample = config['arch']['neighbor_max_sample']

    def forward(self, input):
        # pcd conv
        pcd_xyz = input['pcd_xyz']
        if type(pcd_xyz) is torch.Tensor or self.conv_model_type == 'fcgf_official':
            pcd_xyz = pcd_xyz.contiguous()
            pcd_end_points = self.pcd_conv(pcd_xyz)
            pcd_seed_features = pcd_end_points['features'].unsqueeze(0).transpose(1, 2) # [B, C, N]
            pcd_seed_xyz = pcd_end_points['xyz'].unsqueeze(0) if 'xyz' in pcd_end_points else pcd_xyz.squeeze().unsqueeze(0) # [B, N, 3]
        else:
            pcd_seed_features = self.pcd_conv(pcd_xyz).unsqueeze(0).transpose(1, 2) # [B, C, N]
            pcd_seed_xyz = pcd_xyz.pos.unsqueeze(0).cuda() # [B, N, 3]

        # if balance sampling has been enabled
        if self.balance_weight > 0:
            num_pcd = pcd_seed_xyz.shape[1]
        else:
            num_pcd_full = pcd_seed_xyz.shape[1]
            num_pcd = min(num_pcd_full, 5000)
            # downsample pcd featues, so that it has more chance to be grouped with rgb features
            # maybe a better solution would be weighted sampling?
            rand_idx = torch.randperm(num_pcd_full)[:num_pcd]
            pcd_seed_features = pcd_seed_features[:,:,rand_idx]
            pcd_seed_xyz = pcd_seed_xyz[:,rand_idx,:]
        
        rgb_xyz = input['rgb_xyz'].squeeze().unsqueeze(0).contiguous()
        rgb_features = input['rgb_features'].squeeze().unsqueeze(0)

        # expand rgb and pcd features with zero, we do not directly concat features
        C_pcd, N_pcd = pcd_seed_features.shape[1:]
        C_rgb, N_rgb = rgb_features.shape[1:]

        device = pcd_seed_features.device
        pcd_seed_features_orig = pcd_seed_features
        
        # test FCGF with our score header
        vote_features = pcd_seed_features_orig.transpose(2, 1)

        fp_xyz = pcd_seed_xyz
        fp_features = vote_features.transpose(2, 1)
        
        if self.score_type == 'header':
            vote_scores = self.vote_score_header(fp_features).squeeze() # (B, N)
        elif self.score_type == 'D3Feat':
            vote_scores = self.detection_scores(fp_xyz, vote_features).squeeze()
        # vote_scores = self.detection_scores(fp_xyz, pcd_seed_features.transpose(2, 1)).squeeze()

        if not self.training and self.nms_radius > 0:
            vote_scores = self.nms_score(fp_xyz, vote_scores, self.nms_radius, 64)

        # assert(vote_features.shape[2] == self.out_dim)
        vote_features = F.normalize(vote_features.contiguous(), p=2, dim=2)

        # test original raw feature
        # vote_features = pcd_seed_features_orig.transpose(2, 1)

        return {
            'vote_xyz': fp_xyz.squeeze(),
            'vote_scores': vote_scores.squeeze(),
            'vote_features': vote_features.squeeze(),
        }
    
    def nms_score(self, xyz, scores, radius, max_sample):
        # we adopt score computing from D3Feat
        xyz = xyz.squeeze()
        scores = scores.squeeze().unsqueeze(1)

        shadow_scores = torch.zeros_like(scores[:1, :])
        scores_with_shadow = torch.cat([scores, shadow_scores], dim=0)

        neighbor = self.ball_query(radius, max_sample, xyz.unsqueeze(0), xyz.unsqueeze(0), False)[0].long()
        neighbor_scores = scores_with_shadow[neighbor, :] # [n_points, n_neighbors, 1]
        max_scores, _ = torch.max(neighbor_scores, dim=1)  # [n_points, 1]

        # scores = scores * (max_scores == scores).float()

        is_max_score = (max_scores == scores).float()
        scores = scores * is_max_score + scores * (1-is_max_score) * 1e-4

        return scores

    def detection_scores(self, xyz, features):
        # we adopt score computing from D3Feat
        xyz = xyz.squeeze()
        features = features.squeeze()

        assert(len(xyz.shape) == 2) # we assume only have one batch

        # neighbor = self.ball_query(0.2, 48, xyz.unsqueeze(0), xyz.unsqueeze(0), False)[0].long()
        neighbor = self.ball_query(self.neighbor_radius, self.neighbor_max_sample, xyz.unsqueeze(0), xyz.unsqueeze(0), False)[0].long()
        # print('neighbor mean', torch.sum((neighbor >= 0).float(), dim=1).mean())
        # print('neighbor count', (neighbor < 0).sum(), (neighbor == 0).sum(), (neighbor > 0).sum())

        # neighbor = inputs['neighbors'][0]  # [n_points, n_neighbors]
        N = xyz.shape[0]

        # add a fake point in the last row for shadow neighbors
        # ball query return -1 when no sufficient points in radius, which indicates the last points
        shadow_features = torch.zeros_like(features[:1, :])
        features = torch.cat([features, shadow_features], dim=0)
        shadow_neighbor = torch.ones_like(neighbor[:1, :]) * N
        neighbor = torch.cat([neighbor, shadow_neighbor], dim=0)

        # normalize the feature to avoid overflow
        features = features / (torch.max(features) + 1e-6)

        # local max score (saliency score)
        neighbor_features = features[neighbor, :] # [n_points, n_neighbors, C]
        # neighbor_features_sum = torch.sum(neighbor_features, dim=-1)  # [n_points, n_neighbors]
        # neighbor_num = (neighbor_features_sum != 0).sum(dim=-1, keepdims=True)  # [n_points, 1]
        neighbor_num = torch.sum((neighbor >= 0).float(), dim=1, keepdims=True) # [n_points, 1]
        neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num)) # at least one
        mean_features = torch.sum(neighbor_features, dim=1) / neighbor_num  # [n_points, C]
        local_max_score = F.softplus(features - mean_features)  # [n_points, C]

        # calculate the depth-wise max score
        depth_wise_max = torch.max(features, dim=1, keepdims=True)[0]  # [n_points, 1]
        depth_wise_max_score = features / (1e-6 + depth_wise_max)  # [n_points, C]

        all_scores = local_max_score * depth_wise_max_score
        # use the max score among channel to be the score of a single point. 
        scores = torch.max(all_scores, dim=1, keepdims=True)[0]  # [n_points, 1]

        # hard selection (used during test)
        # if self.training is False:
            # check if any channel is local max among its neighbors
            # something like nms
            # local_max = torch.max(neighbor_features, dim=1)[0]
            # is_local_max = (features == local_max)
            # # print(f"Local Max Num: {float(is_local_max.sum().detach().cpu())}")
            # detected = torch.max(is_local_max.float(), dim=1, keepdims=True)[0]
            # scores = scores * detected
        return scores[:-1, :]


if __name__=='__main__':
    backbone_net = VoteFusionModuleFCGFOnly(out_dim=64, config={
        'arch': {
            'balance_weight' : 0.5,
            'neighbor_radius' : 0.075,
            'neighbor_max_sample' : 48,
            }}).cuda()
    # print(backbone_net)
    backbone_net.eval()
    from torch_points3d.core.data_transform import GridSampling3D, AddOnes, AddFeatByKey
    from torch_geometric.transforms import Compose
    from torch_geometric.data import Batch
    input = {
        'pcd_xyz': torch.rand(1,20000,3).cuda() * 5,
        'rgb_xyz': torch.rand(1,1234,3).cuda() * 5,
        'rgb_features': torch.rand(1,128,1234).cuda(),
    }
    out = backbone_net(input)
    
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
