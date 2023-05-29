import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class CandidateVoteModule_V3(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # propagate features
        self.sa1 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.2,
            nsample=32,
            mlp=[in_dim, 256, 256, 256],
            use_xyz=True,
            normalize_xyz=True
        )
        self.fp1 = PointnetFPModule(mlp=[in_dim+256,256,256])

        self.in_dim = in_dim
        self.out_dim = out_dim

        # vote and aggregate feature
        vote_c = 256
        self.relu = torch.nn.ReLU(inplace=True)
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

        self.vote_offset_header = nn.Sequential(*list(filter(None.__ne__, [
            nn.Conv1d(vote_c, vote_c, 1),
            nn.BatchNorm1d(vote_c),
            self.relu,
            nn.Conv1d(vote_c, vote_c, 1),
            nn.BatchNorm1d(vote_c),
            self.relu,
            nn.Conv1d(vote_c, 3, 1),
        ])))

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


    def break_up_coord_and_features(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, input):
        xyz, features = input['xyz'], input['features']
        # xyz, features = self.break_up_coord_and_features(x)
        xyz, features = xyz.contiguous(), features.contiguous()
        xyz_sa1, features_sa1, fps_inds_sa1 = self.sa1(xyz, features)
        fp_features = self.fp1(xyz, xyz_sa1, features, features_sa1)

        # currently, we assume batch_size==1
        offset = self.vote_offset_header(fp_features).transpose(2, 1) # (1, N, 3)
        vote_xyz = xyz + offset # (1, N, 3)

        vote_scores = self.vote_score_header(fp_features).squeeze() # (1, N)

        vote_features = self.vote_feature_header(fp_features).transpose(2, 1) # (1, N, out_dim)

        vote_features = F.normalize(vote_features.contiguous(), p=2, dim=2)
        # print(vote_features.shape)

        return {
            'vote_xyz': vote_xyz.squeeze(),
            'vote_xyz_offset': offset.squeeze(),
            'vote_scores': vote_scores.squeeze(),
            'vote_features': vote_features.squeeze(),
        }
