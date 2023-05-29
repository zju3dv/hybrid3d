import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision
from model.model_resnet import *
from model.model_pcd import *
from model.unet.unet_parts import down, inconv
from model.decoder_part import make_deconv_layer
import math
import time
import copy

class H3DNet_v8(BaseModel):
    def __init__(self, config=None):
        super(H3DNet_v8, self).__init__()
        use_leaky_relu = True
        use_dropout = False
        use_dropout_for_heat = True
        rgb_c = [32, 64, 128, 256]
        self.layer_rgb_conv = nn.Sequential(*list(filter(None.__ne__, [
            inconv(3, rgb_c[0], leaky_relu=use_leaky_relu),
            nn.Dropout2d(0.2) if use_dropout else None,
            down(rgb_c[0], rgb_c[1], leaky_relu=use_leaky_relu),
            nn.Dropout2d(0.2) if use_dropout else None,
            down(rgb_c[1], rgb_c[2], leaky_relu=use_leaky_relu),
            nn.Dropout2d(0.2) if use_dropout else None,
            down(rgb_c[2], rgb_c[3], leaky_relu=use_leaky_relu),
            nn.Dropout2d(0.2) if use_dropout else None,
        ])))

        # pcd sparse feature conv
        pcd_feat_channels = 32
        pcd_c = [32, 64, 128]
        self.layer_pcd_plain_conv = nn.Sequential(*list(filter(None.__ne__, [
            inconv(pcd_feat_channels, pcd_c[0], leaky_relu=use_leaky_relu),
            nn.Dropout2d(0.2) if use_dropout else None,
            down(pcd_c[0], pcd_c[1], leaky_relu=use_leaky_relu),
            nn.Dropout2d(0.2) if use_dropout else None,
            down(pcd_c[1], pcd_c[2], leaky_relu=use_leaky_relu),
            nn.Dropout2d(0.2) if use_dropout else None,
            # down(pcd_c[2], pcd_c[3], leaky_relu=use_leaky_relu),
        ])))

        # fuse conv
        fuse_c = 256
        self.fuse_conv = nn.Sequential(*list(filter(None.__ne__, [
            nn.Conv2d(rgb_c[-1] + pcd_c[-1], fuse_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(fuse_c),
            nn.LeakyReLU(inplace=True) if use_leaky_relu else nn.ReLU(inplace=True),
            nn.Dropout2d(0.2) if use_dropout else None,
            make_deconv_layer(1, [fuse_c//2], [4], fuse_c, leaky_relu=use_leaky_relu),
            nn.Dropout2d(0.2) if use_dropout else None,
        ])))

        # heatmap
        self.heatmap_head = nn.Sequential(*list(filter(None.__ne__, [
            make_deconv_layer(2, [fuse_c//4, fuse_c//4], [4, 4], fuse_c//2, leaky_relu=use_leaky_relu),
            nn.Dropout2d(0.2) if use_dropout_for_heat else None,
            nn.Conv2d(fuse_c//4, fuse_c//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(fuse_c//4),
            nn.LeakyReLU(inplace=True) if use_leaky_relu else nn.ReLU(inplace=True),
            nn.Dropout2d(0.2) if use_dropout_for_heat else None,
            nn.Conv2d(fuse_c//4, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        ])))

        # descriptor
        self.desc_header = nn.Sequential(*list(filter(None.__ne__, [
            nn.Conv2d(fuse_c//2, fuse_c//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(fuse_c//2),
            nn.LeakyReLU(inplace=True) if use_leaky_relu else nn.ReLU(inplace=True),
            nn.Dropout2d(0.2) if use_dropout else None,
            nn.Conv2d(fuse_c//2, fuse_c//2, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(fuse_c//2),
        ])))

        self.layer_rgb_conv.apply(weights_init)
        self.layer_pcd_plain_conv.apply(weights_init)
        self.fuse_conv.apply(weights_init)
        self.heatmap_head.apply(weights_init)
        self.desc_header.apply(weights_init)

        # pcd_conv layer
        # self.pcd_method = 'pointnet2'
        self.pcd_method = 'kpconv'
        if self.pcd_method == 'pointnet2':
            self.pcd_conv = PointNet2Layer(pcd_feat_channels)
        elif self.pcd_method == 'kpconv':
            self.pcd_conv = KPConvLayer(pcd_feat_channels)
        self.last_fragment_key = ''
    
    def get_spatial_ops(self):
        if self.pcd_method == 'kpconv':
            return self.pcd_conv.get_spatial_ops()
        return None

    def forward(self, data, cache_pcd_inference=False):
        rgb = data['rgb']
        pcd = data['pcd']
        pcd_crsp_idx = data['pcd_crsp_idx']
        fragment_key = data['fragment_key']

        x = self.layer_rgb_conv(rgb)

        # pointcloud conv
        if (not cache_pcd_inference) or self.last_fragment_key != fragment_key:
            self.out_pcd_feat = self.pcd_conv(pcd)
        self.last_fragment_key = fragment_key

        # feature project to plane
        if self.pcd_method == 'pointnet2':
            pcd_3d_ftr = torch.squeeze(self.out_pcd_feat.x).transpose(1, 0)
        elif self.pcd_method == 'kpconv':
            pcd_3d_ftr = self.out_pcd_feat.x

        # pcd_3d_ftr = F.normalize(pcd_3d_ftr, p=2, dim=1) # NxC

        C = pcd_3d_ftr.shape[1]
        B, H, W = pcd_crsp_idx.shape
        pcd_proj_ftr = torch.zeros((B, C, H, W), device=pcd_crsp_idx.device)
        bs, ys, xs = torch.where(pcd_crsp_idx >= 0)
        pcd_proj_ftr[bs, :, ys, xs] = pcd_3d_ftr[pcd_crsp_idx[bs, ys, xs], ...]

        # apply conv to sparse pcd features
        pcd_proj_ftr = F.interpolate(pcd_proj_ftr, scale_factor=0.5, mode='nearest')
        x_pcd = self.layer_pcd_plain_conv(pcd_proj_ftr)
        
        # concat features
        # print(x.shape, x_pcd.shape)
        x = torch.cat([x, x_pcd], dim=1)

        x = self.fuse_conv(x)

        x_desc = F.normalize(self.desc_header(x), p=2, dim=1)

        x_heatmap = self.heatmap_head(x)


        return {
            # 'depth' : x_depth,
            'depth' : None,
            'heatmap' : x_heatmap,
            'descriptor': x_desc
        }
