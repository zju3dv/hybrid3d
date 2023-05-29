import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision
from model.model_helper import *
from model.model_vote_factory import get_vote_model_by_name
import math
import time
import copy

def double_conv(in_channel, out_channel, relu, bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        relu,
        nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        relu
    )

class H3DCoordRGB(BaseModel):
    def __init__(self, config=None):
        super(H3DCoordRGB, self).__init__()
        use_leaky_relu = True
        use_dropout_for_encoder = False
        use_dropout_for_decoder = False

        self.bn_momentum = 0.1
        self.cross_ratio = 2.0
        self.cell = 8

        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout_encoder = nn.Dropout2d(0.2) if use_dropout_for_encoder else None
        self.dropout_decoder = nn.Dropout2d(0.2) if use_dropout_for_decoder else None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shuffle_up = torch.nn.PixelShuffle(upscale_factor=2)

        rgb_c = [32, 64, 128, 256]
        self.layer_rgb_conv_1 = nn.Sequential(*list(filter(None.__ne__, [
            double_conv(3, rgb_c[0], self.relu),
            self.dropout_encoder,
            self.pool,
            double_conv(rgb_c[0], rgb_c[1], self.relu),
            self.dropout_encoder,
            self.pool,
            double_conv(rgb_c[1], rgb_c[2], self.relu),
            self.dropout_encoder,
        ])))

        self.layer_rgb_conv_2 = nn.Sequential(*list(filter(None.__ne__, [
            self.pool,
            double_conv(rgb_c[2], rgb_c[3], self.relu),
            self.dropout_encoder,
        ])))

        fuse_c = 256
        # heatmap
        self.heatmap_header = nn.Sequential(*list(filter(None.__ne__, [
            nn.Conv2d(fuse_c, fuse_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fuse_c),
            self.relu,
            self.dropout_decoder,
            nn.Conv2d(fuse_c, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        ])))

        # location
        self.location_header = nn.Sequential(*list(filter(None.__ne__, [
            nn.Conv2d(fuse_c, fuse_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fuse_c),
            self.relu,
            self.dropout_decoder,
            nn.Conv2d(fuse_c, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        ])))

        # descriptor
        self.desc_header_1 = nn.Sequential(*list(filter(None.__ne__, [
            nn.Conv2d(fuse_c, fuse_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fuse_c),
            self.relu,
            self.dropout_decoder,
            nn.Conv2d(fuse_c, fuse_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fuse_c),
            self.shuffle_up,
        ])))


        self.desc_header_2 = nn.Sequential(*list(filter(None.__ne__, [
            nn.Conv2d(fuse_c//4 + rgb_c[-2], fuse_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fuse_c),
            self.relu,
            nn.Conv2d(fuse_c, fuse_c//2, kernel_size=3, stride=1, padding=1),
        ])))

        self.layer_rgb_conv_1.apply(weights_init)
        self.layer_rgb_conv_2.apply(weights_init)
        self.location_header.apply(weights_init)
        self.heatmap_header.apply(weights_init)
        self.desc_header_1.apply(weights_init)
        self.desc_header_2.apply(weights_init)
        
        # add vote model for ablation study
        # assert(config != None)
        # self.votenet = getattr(
        #     get_vote_model_by_name(config['arch']['vote_model']), config['arch']['vote_model'])(config=config)
        # self.pcd_method = self.votenet.conv_model_type
        # self.pcd_conv = self.votenet.pcd_conv
    
    def get_spatial_ops(self):
        return None
    
    # def forward_vote(self, input):
    #     return self.votenet.forward(input)

    def forward(self, data, cache_pcd_inference=False):
        rgb = data['rgb']
        pcd = data['pcd']
        pcd_crsp_idx = data['pcd_crsp_idx']
        fragment_key = data['fragment_key']
        B, _, H, W = rgb.shape
        skip_rgb = self.layer_rgb_conv_1(rgb)
        x_rgb = self.layer_rgb_conv_2(skip_rgb)

        x = x_rgb

        B, _, Hc, Wc = x.shape

        border_mask = torch.ones(B, Hc, Wc)
        border_mask[:, 0] = 0
        border_mask[:, Hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, Wc - 1] = 0
        border_mask = border_mask.unsqueeze(1).to(x.device)

        # x_desc = F.normalize(self.desc_header(x), p=2, dim=1)

        x_heatmap = self.heatmap_header(x)
        x_heatmap = x_heatmap * border_mask

        x_center_shift = self.location_header(x)

        step = (self.cell-1) / 2.
        center_base = image_grid(B, Hc, Wc,
                                 dtype=x_center_shift.dtype,
                                 device=x_center_shift.device,
                                 ones=False, normalized=False).mul(self.cell) + step

        coord_un = center_base.add(x_center_shift.mul(self.cross_ratio * step))
        coord = coord_un.clone()
        coord[:, 0] = torch.clamp(coord_un[:, 0], min=0, max=W-1)
        coord[:, 1] = torch.clamp(coord_un[:, 1], min=0, max=H-1)

        x_desc = self.desc_header_1(x)
        x_desc = torch.cat([x_desc, skip_rgb], dim=1)
        x_desc = self.desc_header_2(x_desc)
        x_desc = F.normalize(x_desc, p=2, dim=1)

        return {
            'depth' : None,
            'heatmap' : x_heatmap,
            'descriptor': x_desc,
            'coord': coord, # order: x y
        }
