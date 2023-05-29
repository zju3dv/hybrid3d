import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('.')

from model.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# from VoteNet
class PointConvModule(torch.nn.Module):
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
                npoint=4096,
                radius=0.1,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[256, 256, 256, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        # self.sa3 = PointnetSAModuleVotes(
        #         npoint=1024,
        #         radius=0.4,
        #         nsample=32,
        #         mlp=[256, 128, 128, 256],
        #         use_xyz=True,
        #         normalize_xyz=True
        #     )

        # self.sa4 = PointnetSAModuleVotes(
        #         npoint=512,
        #         radius=0.8,
        #         nsample=16,
        #         mlp=[256, 128, 128, 256],
        #         use_xyz=True,
        #         normalize_xyz=True
        #     )

        # self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        # self.fp2 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp3 = PointnetFPModule(mlp=[128+256,256,256])
        self.fp4 = PointnetFPModule(mlp=[256,256,256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        end_points = {}
        pointcloud = input['pcd_xyz']
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        # add rgb features
        rgb_xyz = input['rgb_xyz']
        rgb_features = input['rgb_features']
        # expand rgb and pcd features with zero, we do not directly concat features
        pcd_seed_features = torch.cat([features, torch.zeros_like(features)], dim=1) # [B, 128, Npcd]->[B, 256, Npcd]
        rgb_features = torch.cat([torch.zeros_like(rgb_features), rgb_features], dim=1) # [B, 128, Nrgb]->[B, 256, Nrgb]

        xyz = torch.cat([xyz, rgb_xyz], dim=1).contiguous()
        features = torch.cat([pcd_seed_features, rgb_features], dim=2).contiguous()

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        # xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        # end_points['sa3_xyz'] = xyz
        # end_points['sa3_features'] = features

        # xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        # end_points['sa4_xyz'] = xyz
        # end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = end_points['sa2_features']
        # features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        # features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        features = self.fp3(end_points['sa1_xyz'], end_points['sa2_xyz'], end_points['sa1_features'], features)
        features = self.fp4(input['pcd_xyz'], end_points['sa1_xyz'], None, features)

        end_points['out_features'] = features
        end_points['out_xyz'] = input['pcd_xyz']
        # num_seed = end_points['out_xyz'].shape[1]
        # end_points['fp3_inds'] = end_points['sa1_inds'] # indices among the entire input point clouds
        return end_points


class VoteFusionModule(torch.nn.Module):
    def __init__(self, in_dim=None, out_dim=128, config=None):
        super().__init__()
        # pcd conv module
        self.pcd_conv = PointConvModule(input_feature_dim=0)

        self.out_dim = out_dim

        # vote and aggregate feature
        vote_c = 256
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

    def forward(self, input):
        # pcd conv
        # out_features     torch.Size([2, 128, 2048])
        # fp3_inds         torch.Size([2, 2048])
        # out_xyz          torch.Size([2, 2048, 3])
        pcd_end_points = self.pcd_conv(input)
        pcd_seed_features = pcd_end_points['out_features'] # [B, C, N]
        pcd_seed_xyz = pcd_end_points['out_xyz'] # [B, N, 3]

        fp_features = pcd_seed_features

        vote_scores = self.vote_score_header(fp_features).squeeze() # (B, N)

        vote_features = self.vote_feature_header(fp_features).transpose(2, 1) # (B, N, out_dim)

        vote_features = F.normalize(vote_features.contiguous(), p=2, dim=2)
        # print(vote_features.shape)

        return {
            'vote_xyz': pcd_seed_xyz.squeeze(),
            'vote_scores': vote_scores.squeeze(),
            'vote_features': vote_features.squeeze(),
        }


if __name__=='__main__':
    backbone_net = VoteFusionModule(out_dim=128).cuda()
    # print(backbone_net)
    backbone_net.eval()
    out = backbone_net({
        'pcd_xyz': torch.rand(1,20000,3).cuda(),
        'rgb_xyz': torch.rand(1,1234,3).cuda(),
        'rgb_features': torch.rand(1,128,1234).cuda(),
    })
    for key in sorted(out.keys()):
        print(key, '\t', out[key].shape)
