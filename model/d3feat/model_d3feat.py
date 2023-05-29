import torch
import numpy as np
from easydict import EasyDict as edict
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from architectures import KPFCNN
from d3feat_dataloader import collate_fn_descriptor

class D3FeatModule(torch.nn.Module):
    def __init__(self, fix_weight=True, dataset_type='3dmatch_0.03'):
        super().__init__()
        if dataset_type == '3dmatch_0.03':
            self.neighborhood_limits = [39, 35, 36, 36, 35] # hard code for 3DMatch in 0.03 voxel size
        else:
            print('Warning, dataset_type: {} not valid.'.format(dataset_type))
            self.neighborhood_limits = [39, 35, 36, 36, 35]
        self.fix_weight = fix_weight
        config_path = f'{BASE_DIR}/pretrained/config_d3feat.json'
        self.config = json.load(open(config_path, 'r'))
        self.config = edict(self.config)
        self.config.output_score = False

        # create model 
        self.config.architecture = [
            'simple',
            'resnetb',
        ]
        for i in range(self.config.num_layers-1):
            self.config.architecture.append('resnetb_strided')
            self.config.architecture.append('resnetb')
            self.config.architecture.append('resnetb')
        for i in range(self.config.num_layers-2):
            self.config.architecture.append('nearest_upsample')
            self.config.architecture.append('unary')
        self.config.architecture.append('nearest_upsample')
        self.config.architecture.append('last_unary')
        self.model = KPFCNN(self.config)
        self.model.load_state_dict(torch.load(f'{BASE_DIR}/pretrained/model_best_acc.pth')['state_dict'])
        print(f"D3FeatModule load weight from: {BASE_DIR}/pretrained/model_best_acc.pth.")
        self.model.eval()

    def get_d3feat_config(self):
        return self.config

    def forward(self, data):
        if self.fix_weight:
            self.model.eval()
        # simulate list_data
        if type(data) is torch.Tensor:
            data = data.squeeze().detach().cpu().numpy()
        elif hasattr(data, 'pos'):
            data = data.pos.squeeze().detach().cpu().numpy()
        pts = data # [N, 3]
        feat = np.ones_like(pts[:, :1]).astype(np.float32) # [N, 1]
        list_data = [[pts, feat]]
        dict_input = collate_fn_descriptor(list_data, self.config, self.neighborhood_limits)

        for k, v in dict_input.items():
            if type(v) is torch.Tensor:
                dict_input[k] = v.cuda()
            elif type(v) is list:
                dict_input[k] = [x.cuda() for x in v]

        out_feat, _ = self.model(dict_input)
        return {
            'features': out_feat,
        }
