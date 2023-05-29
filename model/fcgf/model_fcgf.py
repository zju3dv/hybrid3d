import torch
import numpy as np
from easydict import EasyDict as edict
import json
import os
import sys
import contextlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from module_fcgf import load_model
from module_fcgf.resunet import ResUNetBN2C

import MinkowskiEngine as ME

class FCGFModule(torch.nn.Module):
    def __init__(self, fix_weight=True):
        super().__init__()
        self.fix_weight = fix_weight
        # config_path = f'{BASE_DIR}/pretrained/config.json'
        # self.config = json.load(open(config_path, 'r'))
        # self.config = edict(self.config)

        # create model 
        self.voxel_size = 0.025
        self.model = ResUNetBN2C(1, 32, normalize_feature=True, conv1_kernel_size=7, D=3)
        self.model.load_state_dict(torch.load(f'{BASE_DIR}/pretrained/voxel_0.025_dim_32_2019-08-19_06-17-41.pth')['state_dict'])
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def forward(self, data):
        if self.fix_weight:
            self.model.eval()
        # simulate list_data
        if type(data) is torch.Tensor:
            data = data.squeeze().detach().cpu().numpy()
        elif hasattr(data, 'pos'):
            data = data.pos.squeeze().detach().cpu().numpy()
        xyz = data # [N, 3]
        feats = []
        feats.append(np.ones((len(xyz), 1)))
        feats = np.hstack(feats)
        # feats = np.ones_like(pts[:, :1]).astype(np.float32) # [N, 1]

        # Voxelize xyz and feats
        coords = np.floor(xyz / self.voxel_size)
        # print('1', coords.shape)
        _, unique_map, inverse_map = ME.utils.sparse_quantize(coords, return_index=True, return_inverse=True)
        inds = unique_map
        coords = coords[inds]
        # print('2', coords.shape)
        # Convert to batched coords compatible with ME
        return_coords = xyz[inds]
        # return_coords = xyz
        coords = ME.utils.batched_coordinates([coords])

        feats = feats[inds]

        feats = torch.tensor(feats, dtype=torch.float32)
        coords = coords.to(dtype=torch.int32)
        with (torch.no_grad() if self.fix_weight else contextlib.nullcontext()):
            stensor = ME.SparseTensor(feats, coordinates=coords, device=self.device)
            out_feat = self.model(stensor).F
        # print(out_feat.shape[0], xyz.shape[0])
        out_feat_keep_num = out_feat[inverse_map]
        # return_coords = return_coords[inverse_map]
        return_coords_keep_num = return_coords[inverse_map]

        return {
            'features': out_feat.squeeze(),
            'xyz': torch.from_numpy(return_coords).squeeze().cuda(),
            'features_keep_num': out_feat_keep_num.squeeze(),
            'xyz_keep_num': torch.from_numpy(return_coords_keep_num).squeeze().cuda(),
        }
