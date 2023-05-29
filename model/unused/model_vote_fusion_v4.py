import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('.')

from model.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule, PointnetSAModuleBalanced
from model.model_pcd import MinokowskiLayer
from model.pointnet2.pointnet2_utils import ball_query

# def save_pointcloud(pcd_np, pcd_path):
#     import open3d as o3d
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pcd_np)
#     o3d.io.write_point_cloud(pcd_path, pcd)

class VoteFusionModuleV3(torch.nn.Module):
    def __init__(self, in_dim=None, out_dim=64, config=None):
        super().__init__()
        # pcd conv module
        self.pcd_conv = MinokowskiLayer(fix_weight=config['arch'].get('fix_pcd_conv_weight', True))

        self.out_dim = out_dim

        self.balance_weight = config['arch'].get('balance_weight', 0.0) if config != None else 0.0

        print('VoteFusionModuleV3 start with balanced weight: ', self.balance_weight)

        self.N_anchor = 4096
        self.sample_radius = 0.1

        if self.balance_weight > 0:
            self.fuse_sa1 = PointnetSAModuleBalanced(
                npoint=self.N_anchor,
                radius=self.sample_radius,
                nsample=32,
                mlp=[32+128, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True,
                balance_weight=self.balance_weight,
            )
        else:
            self.fuse_sa1 = PointnetSAModuleVotes(
                npoint=self.N_anchor,
                radius=self.sample_radius,
                nsample=32,
                mlp=[32+128, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
        self.fuse_fp1 = PointnetFPModule(mlp=[32+128+128,128,32])

        # vote and aggregate feature
        vote_c = 32
        self.relu = torch.nn.ReLU(inplace=True)
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

        self.vote_score_header = nn.Sequential(*list(filter(None.__ne__, [
            nn.Conv1d(vote_c, vote_c, 1),
            nn.BatchNorm1d(vote_c),
            self.relu,
            nn.Conv1d(vote_c, 1, 1),
            nn.Sigmoid(),
        ])))

        self.vote_feature_header = nn.Sequential(*list(filter(None.__ne__, [
            nn.Conv1d(vote_c, vote_c, 1),
            nn.BatchNorm1d(vote_c),
            self.relu,
            nn.Conv1d(vote_c, vote_c, 1),
            nn.BatchNorm1d(vote_c),
            self.relu,
            nn.Conv1d(vote_c, out_dim, 1),
        ])))

        self.ball_query = ball_query
        self.neighbor_radius = config['arch']['neighbor_radius']
        self.neighbor_max_sample = config['arch']['neighbor_max_sample']

        self.only_fuse_overlap = True

    def find_overlap_point_idx(self, src_xyz, dst_xyz, radius):
        pc_src_input = src_xyz.squeeze().transpose(1, 0) # 3xM
        pc_dst_input = dst_xyz.squeeze().transpose(1, 0) # 3xN
        _, M = pc_src_input.shape
        _, N = pc_dst_input.shape
        pc_src_input_expanded = pc_src_input.unsqueeze(2).expand(3, M, N)
        pc_dst_input_expanded = pc_dst_input.unsqueeze(1).expand(3, M, N)
        dist_diff = torch.norm(pc_src_input_expanded - pc_dst_input_expanded, dim=0, keepdim=False)  # MxN
        valid_idx_mask, _ = torch.max((dist_diff<=radius).float(), dim=1)
        return torch.where(valid_idx_mask > 0)

    def forward(self, input):
        # pcd conv
        pcd_xyz = input['pcd_xyz']
        if type(pcd_xyz) is torch.Tensor:
            pcd_xyz = pcd_xyz.contiguous()
            pcd_end_points = self.pcd_conv(pcd_xyz)
            pcd_seed_features = pcd_end_points['features'][None, :, :].transpose(1, 2) # [B, C, N]
            pcd_seed_xyz = pcd_end_points['xyz'][None, ...] # [B, N, 3]
        else:
            pcd_seed_features = self.pcd_conv(pcd_xyz)[None, ...].transpose(1, 2) # [B, C, N]
            pcd_seed_xyz = pcd_xyz.pos[None, ...].cuda() # [B, N, 3]

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
        
        rgb_xyz = input['rgb_xyz'].contiguous()
        rgb_features = input['rgb_features']

        # expand rgb and pcd features with zero, we do not directly concat features
        C_pcd, N_pcd = pcd_seed_features.shape[1:]
        C_rgb, N_rgb = rgb_features.shape[1:]

        device = pcd_seed_features.device

        pcd_seed_features_expand = torch.cat([pcd_seed_features, torch.zeros((1, C_rgb, N_pcd), device=device)], dim=1) # [B, Cpcd, Npcd]->[B, Cpcd+Crgb, Npcd]
        rgb_features_expand = torch.cat([torch.zeros((1, C_pcd, N_rgb), device=device), rgb_features], dim=1) # [B, Crgb, Nrgb]->[B, Cpcd+Crgb, Nrgb]

        if self.balance_weight > 0:
            if self.only_fuse_overlap:
                # find overlap points
                overlap_idx = self.find_overlap_point_idx(pcd_seed_xyz, rgb_xyz, self.sample_radius)[0]
                fp_features = pcd_seed_features
                if overlap_idx.numel() > 0:
                    pcd_overlap_xyz = pcd_seed_xyz[:, overlap_idx, :].contiguous()
                    pcd_overlap_feature = pcd_seed_features_expand[:, :, overlap_idx]
                    xyz_sa1, features_sa1, fps_inds_sa1 = self.fuse_sa1(pcd_overlap_xyz, pcd_overlap_feature, rgb_xyz, rgb_features_expand)
                    overlap_features = self.fuse_fp1(pcd_overlap_xyz, xyz_sa1, pcd_overlap_feature, features_sa1)
                    fp_features[:, :, overlap_idx] = overlap_features
                    # save_pointcloud(pcd_overlap_xyz.squeeze().detach().cpu().numpy(), 'pcd_radius_0.02.ply')
                    # save_pointcloud(rgb_xyz.squeeze().detach().cpu().numpy(), 'rgb_radius_0.02.ply')
                else:
                    print('Warning: no overlap between rgb candidates and pcd!')
            else:
                # we let balanced set abstraction layer to fuse features
                xyz_sa1, features_sa1, fps_inds_sa1 = self.fuse_sa1(pcd_seed_xyz, pcd_seed_features_expand, rgb_xyz, rgb_features_expand)
                fp_features = self.fuse_fp1(pcd_seed_xyz, xyz_sa1, pcd_seed_features_expand, features_sa1)
        else:
            # assemble for rgb and pcd feature aggregation
            full_xyz = torch.cat([pcd_seed_xyz, rgb_xyz], dim=1).contiguous()
            full_features = torch.cat([pcd_seed_features_expand, rgb_features_expand], dim=2).contiguous()
            xyz_sa1, features_sa1, fps_inds_sa1 = self.fuse_sa1(full_xyz, full_features)
            fp_features = self.fuse_fp1(full_xyz, xyz_sa1, full_features, features_sa1)
        
        # only remain features at pcd location
        fp_xyz = pcd_seed_xyz
        fp_features = fp_features[:, :, :num_pcd]

        # vote_features = self.vote_feature_header(fp_features).transpose(2, 1) # (B, N, out_dim)
        vote_features = fp_features.transpose(2, 1) # (B, N, out_dim)
        # vote_featuers = pcd_seed_features.transpose(2, 1)
        
        # vote_scores = self.vote_score_header(fp_features).squeeze() # (B, N)
        vote_scores = self.detection_scores(fp_xyz, vote_features).squeeze()
        # vote_scores = self.detection_scores(fp_xyz, pcd_seed_features.transpose(2, 1)).squeeze()

        vote_features = F.normalize(vote_features.contiguous(), p=2, dim=2)

        return {
            'vote_xyz': fp_xyz.squeeze(),
            'vote_scores': vote_scores.squeeze(),
            'vote_features': vote_features.squeeze(),
        }

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
    backbone_net = VoteFusionModuleV3(out_dim=32, config={
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
