import os
import torch
import torch.nn.functional as F
import numpy as np
from model.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class CandidateVoteModule_V2(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # propagate features
        self.sa1 = PointnetSAModuleVotes(
            npoint=2048,
            radius=0.2,
            nsample=32,
            mlp=[in_dim, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )
        self.sa2 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.4,
            nsample=32,
            mlp=[256, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )
        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[in_dim+256,256,128])

        self.in_dim = in_dim
        self.out_dim = out_dim

        # vote and aggregate feature
        vote_c = 256
        self.conv1 = torch.nn.Conv1d(128, vote_c, 1)
        self.conv2 = torch.nn.Conv1d(vote_c, vote_c, 1)
        self.conv3 = torch.nn.Conv1d(vote_c, 3 + 1 + out_dim, 1) # vote_xyz, score, features
        self.bn1 = torch.nn.BatchNorm1d(vote_c)
        self.bn2 = torch.nn.BatchNorm1d(vote_c)
        self.relu = torch.nn.ReLU(inplace=True)
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

    def break_up_coord_and_features(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, xyz, features):
        # xyz, features = self.break_up_coord_and_features(x)
        xyz, features = xyz.contiguous(), features.contiguous()
        xyz_sa1, features_sa1, fps_inds_sa1 = self.sa1(xyz, features)
        xyz_sa2, features_sa2, fps_inds_sa2 = self.sa2(xyz_sa1, features_sa1)
        fp_features = self.fp1(xyz_sa1, xyz_sa2, features_sa1, features_sa2)
        fp_features = self.fp2(xyz, xyz_sa1, features, fp_features)

        seed_xyz = xyz
        seed_features = fp_features

        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed
        net = self.relu(self.bn1(self.conv1(seed_features)))
        net = self.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (batch_size, (4+out_dim), num_seed)
        net = net.transpose(2, 1).view(batch_size, num_seed, -1) # (batch_size, num_seed, (4+out_dim))
        offset = net[:, :, 0:3]
        vote_xyz = seed_xyz + offset # (batch_size, num_seed, 3)

        vote_scores = self.sigmoid(net[:, :, 3])

        residual_features = net[:, :, 4:]  # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2, 1) + residual_features
        vote_features = F.normalize(vote_features.contiguous(), p=2, dim=2)

        return {
            'vote_xyz': vote_xyz.squeeze(),
            'vote_xyz_offset': offset.squeeze(),
            'vote_scores': vote_scores.squeeze(),
            'vote_features': vote_features.squeeze(),
        }
