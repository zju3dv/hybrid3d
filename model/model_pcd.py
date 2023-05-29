from torch_points3d.applications.kpconv import KPConv
from torch_points3d.applications.pointnet2 import PointNet2
from omegaconf import OmegaConf
import os
import torch
import numpy as np
from torch_geometric.data import Data
import contextlib

CUR_FILE = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class KPConvLayer(torch.nn.Module):
    def __init__(self, in_out_feat=32):
        super().__init__()
        PATH_TO_CONFIG = os.path.join(DIR_PATH, "kpconv.yaml")
        model_config = OmegaConf.load(PATH_TO_CONFIG)
        self.unet = KPConv(
            architecture="unet",
            input_nc=0,
            num_layers=4,
            in_grid_size=0.02,
            in_feat = in_out_feat,
            config = model_config
            # output_nc=64
            )

    def forward(self, data):
        # Forward through unet and classifier
        data_features = self.unet(data)
        self.output = data_features
        return self.output

    def get_spatial_ops(self):
        return self.unet.get_spatial_ops()

class PointNet2Layer(torch.nn.Module):
    def __init__(self, in_out_feat=32):
        super().__init__()
        self.unet = PointNet2(
            architecture="unet",
            input_nc=1,
            num_layers=3,
            # in_grid_size=0.02,
            multiscale=True,
            output_nc=in_out_feat
            # in_feat = in_out_feat
            # output_nc=64
            )

    def forward(self, data):
        # Forward through unet and classifier
        if len(data.pos.shape) < 3:
            data.pos = data.pos.unsqueeze(0)
        if len(data.x.shape) < 3:
            data.x = data.x.unsqueeze(0)
        data_features = self.unet(data)
        self.output = data_features
        return self.output

    # def get_spatial_ops(self):
    #     return self.unet.get_spatial_ops()

from torch_points3d.core.data_transform import GridSampling3D, AddOnes, AddFeatByKey
from torch_geometric.transforms import Compose
from torch_geometric.data import Batch
import time
class MinokowskiLayer(torch.nn.Module):
    def __init__(self, fix_weight=True, dataset='3dmatch'):
        super().__init__()
        self.fix_weight = fix_weight
        from torch_points3d.applications.pretrained_api import PretainedRegistry
        try:
            self.model = PretainedRegistry.from_pretrained("minkowski-registration-{}".format(dataset)).cuda()
        except Exception:
            # the model has a different transform, which may be failed to load
            pass
        self.transform = Compose([GridSampling3D(mode='last', size=0.02, quantize_coords=True), AddOnes(), AddFeatByKey(add_to_x=True, feat_name="ones")])
        self.time_accu = []

    def forward(self, x):
        t1 = time.time()
        if self.fix_weight:
            self.model.eval()
        else:
            self.model.train()
        if type(x) is torch.Tensor:
            x = x.squeeze()
            data_s = self.transform(Batch(pos=x.cpu().float(), batch=torch.zeros(x.shape[0]).long()))
            self.model.set_input(data_s, "cuda")
            with (torch.no_grad() if self.fix_weight else contextlib.nullcontext()):
                ret_dict = {
                    'xyz': data_s.pos.cuda(),
                    'features': self.model.forward(), 
                }
        else:
            x.batch = torch.zeros(x.pos.shape[0]).long()
            self.model.set_input(x, "cuda")
            with (torch.no_grad() if self.fix_weight else contextlib.nullcontext()):
                ret = self.model.forward()
                t2 = time.time()
                self.time_accu.append(t2-t1)
                # if len(self.time_accu) % 100 == 0 or len(self.time_accu) > 1000:
                # print('pcd time', sum(self.time_accu), sum(self.time_accu) / len(self.time_accu))
                return ret
