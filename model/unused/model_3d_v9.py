import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision
from model.model_resnet import *
from model.model_pcd import *
from model.unet.unet_parts import down, inconv
from model.decoder_part import make_deconv_layer
from model.model_helper import image_grid
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

class H3DNet_v9(BaseModel):
    def __init__(self):
        super(H3DNet_v9, self).__init__()
        use_leaky_relu = True
        use_dropout_for_encoder = True
        use_dropout_for_decoder = True

        self.bn_momentum = 0.1
        self.cross_ratio = 2.0
        self.cell = 8

        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout_encoder = nn.Dropout2d(0.2) if use_dropout_for_encoder else None
        self.dropout_decoder = nn.Dropout2d(0.2) if use_dropout_for_decoder else None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shuffle_up = torch.nn.PixelShuffle(upscale_factor=2)

        rgb_c = [32, 64, 128, 256]
        self.layer_rgb_conv = nn.Sequential(*list(filter(None.__ne__, [
            double_conv(3, rgb_c[0], self.relu),
            self.dropout_encoder,
            self.pool,
            double_conv(rgb_c[0], rgb_c[1], self.relu),
            self.dropout_encoder,
            self.pool,
            double_conv(rgb_c[1], rgb_c[2], self.relu),
            self.dropout_encoder,
            self.pool,
            double_conv(rgb_c[2], rgb_c[3], self.relu),
            self.dropout_encoder,
        ])))

        # pcd sparse feature conv
        pcd_feat_channels = 32
        pcd_c = [32, 64, 128]
        self.layer_pcd_plain_conv = nn.Sequential(*list(filter(None.__ne__, [
            double_conv(pcd_feat_channels, pcd_c[0], self.relu),
            self.dropout_encoder,
            self.pool,
            double_conv(pcd_c[0], pcd_c[1], self.relu),
            self.dropout_encoder,
            self.pool,
            double_conv(pcd_c[1], pcd_c[2], self.relu),
            self.dropout_encoder
        ])))

        # fuse conv
        fuse_c = 256
        self.fuse_conv = nn.Sequential(*list(filter(None.__ne__, [
            nn.Conv2d(rgb_c[-1] + pcd_c[-1], fuse_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fuse_c),
            self.relu,
            self.dropout_encoder,
            nn.Conv2d(fuse_c, fuse_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fuse_c),
            self.relu,
            self.dropout_encoder,
        ])))

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
        self.desc_header = nn.Sequential(*list(filter(None.__ne__, [
            nn.Conv2d(fuse_c, fuse_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fuse_c),
            self.relu,
            self.dropout_decoder,
            nn.Conv2d(fuse_c, fuse_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fuse_c),
            self.shuffle_up,
            nn.Conv2d(fuse_c//4, fuse_c//4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fuse_c//4),
            self.relu,
            nn.Conv2d(fuse_c//4, fuse_c//4, kernel_size=3, stride=1, padding=1),
        ])))

        self.layer_rgb_conv.apply(weights_init)
        self.layer_pcd_plain_conv.apply(weights_init)
        self.fuse_conv.apply(weights_init)
        self.location_header.apply(weights_init)
        self.heatmap_header.apply(weights_init)
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

        B, _, Hc, Wc = x.shape

        border_mask = torch.ones(B, Hc, Wc)
        border_mask[:, 0] = 0
        border_mask[:, Hc - 1] = 0
        border_mask[:, :, 0] = 0
        border_mask[:, :, Wc - 1] = 0
        border_mask = border_mask.unsqueeze(1)

        # x_desc = F.normalize(self.desc_header(x), p=2, dim=1)

        x_heatmap = self.heatmap_header(x)
        x_heatmap = x_heatmap * border_mask.to(x_heatmap.device)

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

        x_desc = F.normalize(self.desc_header(x), p=2, dim=1)


        return {
            'depth' : None,
            'heatmap' : x_heatmap,
            'descriptor': x_desc,
            'coord': coord, # order: x y
        }
