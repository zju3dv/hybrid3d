import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision
from model.model_resnet import *
from model.model_pcd import *
from model.unet.unet_parts import down
from model.decoder_part import make_deconv_layer
import math
import time
import copy

class Hybrid3DNetPN2Only(BaseModel):
    def __init__(self, pretrained=True):
        super(Hybrid3DNetPN2Only, self).__init__()

        self.last_activation = 'sigmoid'

        # pcd sparse feature conv
        pcd_feat_channels = 32
        c1, c2, c3, c4 = 64, 128, 256, 512
        self.layer1_pcd = down(32, c1)
        self.layer2_pcd = down(c1, c2)
        self.layer3_pcd = down(c2, c3)
        self.layer4_pcd = down(c3, c4)

        self.decoder = make_deconv_layer(
            4,
            [c4//2, c4//4, c4//8, c4//8],
            [4, 4, 4, 4],
            c4
        )
        # remove relu at last layer
        self.decoder = nn.Sequential(*list(self.decoder.children())[:-1])
        self.conv3_heatmap = nn.Conv2d(c4//8,1,kernel_size=1)
        self.Sigmoid = nn.Sigmoid()

        self.layer1_pcd.apply(weights_init)
        self.layer2_pcd.apply(weights_init)
        self.layer3_pcd.apply(weights_init)
        self.layer4_pcd.apply(weights_init)
        self.decoder.apply(weights_init)
        self.conv3_heatmap.apply(weights_init)

        # pcd_conv layer
        # self.pcd_method = 'pointnet2'
        self.pcd_method = 'kpconv'
        if self.pcd_method == 'pointnet2':
            self.pcd_conv = PointNet2Layer(pcd_feat_channels)
        elif self.pcd_method == 'kpconv':
            self.pcd_conv = KPConvLayer(pcd_feat_channels)
        self.last_fragment_key = ''

    def softmax(self, ux):
        return F.softmax(ux, dim=1)[:,1:2]

    def softplus(self, ux):
        x = F.softplus(ux)
        return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
    
    def get_spatial_ops(self):
        if self.pcd_method == 'kpconv':
            return self.pcd_conv.get_spatial_ops()
        return None

    def forward(self, data):
        rgb = data['rgb']
        depth = data['depth']
        pcd = data['pcd']
        pcd_crsp_idx = data['pcd_crsp_idx']
        fragment_key = data['fragment_key']

        # pointcloud conv
        if self.training or self.last_fragment_key != fragment_key:
            self.out_pcd_feat = self.pcd_conv(pcd)
        self.last_fragment_key = fragment_key

        # feature project to plane
        if self.pcd_method == 'pointnet2':
            pcd_3d_ftr = torch.squeeze(self.out_pcd_feat.x).transpose(1, 0)
        elif self.pcd_method == 'kpconv':
            pcd_3d_ftr = self.out_pcd_feat.x

        pcd_3d_ftr = F.normalize(pcd_3d_ftr, p=2, dim=1)

        C = pcd_3d_ftr.shape[1]
        B, H, W = pcd_crsp_idx.shape
        pcd_proj_ftr = torch.zeros((B, C, H, W)).float().to(pcd_crsp_idx.device)
        bs, ys, xs = torch.where(pcd_crsp_idx >= 0)
        pcd_proj_ftr[bs, :, ys, xs] = pcd_3d_ftr[pcd_crsp_idx[bs, ys, xs], ...]

        # apply conv to sparse pcd features
        x_pcd = self.layer4_pcd(self.layer3_pcd(self.layer2_pcd(self.layer1_pcd(pcd_proj_ftr))))
        
        # concat features
        # print(x.shape, x_pcd.shape)
        x = x_pcd

        x = self.decoder(x)
        x_desc = F.normalize(x, p=2, dim=1)
        # print(x)
        # print(x_desc)

        if self.last_activation == 'sigmoid':
            x_heatmap = self.conv3_heatmap(x)
            x_heatmap = self.Sigmoid(x_heatmap)
        elif self.last_activation == 'softplus':
            x_heatmap = self.conv3_heatmap(x ** 2)
            x_heatmap = self.softplus(x_heatmap)

        return {
            # 'depth' : x_depth,
            'depth' : None,
            'heatmap' : x_heatmap,
            'descriptor': x_desc
        }
