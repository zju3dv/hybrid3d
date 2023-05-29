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

class Hybrid3DNetPN2(BaseModel):
    def __init__(self, layers=50, in_channels=4, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(Hybrid3DNetPN2, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        # fix pretrained model parameters
        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.with_sparse_d = False

        if self.with_sparse_d:
            in_channels = 4
        else:
            in_channels = 3

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        # self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # pcd sparse feature conv
        pcd_feat_channels = 32
        c1, c2, c3, c4 = 64, 128, 256, 512
        self.layer1_pcd = down(32, c1)
        self.layer2_pcd = down(c1, c2)
        self.layer3_pcd = down(c2, c3)
        self.layer4_pcd = down(c3, c4)
        
        # unlock conv_4_6
        for param in self.layer3[5].parameters():
            param.requires_grad = True

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        # fuse conv
        # self.conv2 = nn.Conv2d(num_channels//2 + c4,num_channels//2,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(num_channels//2 + c4,num_channels//2,kernel_size=1)
        self.bn2 = nn.BatchNorm2d(num_channels//2)

        # heatmap
        # decoder='deconv3'
        # self.decoder_heatmap = choose_decoder(decoder, num_channels//2)
        self.decoder_heatmap = make_deconv_layer(
            4,
            [num_channels//4, num_channels//8, num_channels//16, num_channels//32],
            [4, 4, 4, 4],
            num_channels//2
        )
        self.conv3_heatmap = nn.Conv2d(num_channels//32,1,kernel_size=1)
        self.Sigmoid = nn.Sigmoid()

        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.layer1_pcd.apply(weights_init)
        self.layer2_pcd.apply(weights_init)
        self.layer3_pcd.apply(weights_init)
        self.layer4_pcd.apply(weights_init)
        self.decoder_heatmap.apply(weights_init)
        self.conv3_heatmap.apply(weights_init)

        # pcd_conv layer
        self.pcd_conv = PointNet2Layer(pcd_feat_channels)
        self.last_fragment_key = ''

    def softmax(self, ux):
        return F.softmax(ux, dim=1)[:,1:2]
    
    # def get_spatial_ops(self):
    #     return self.pcd_conv.get_spatial_ops()

    def forward(self, data):
        rgb = data['rgb']
        depth = data['depth']
        pcd = data['pcd']
        pcd_crsp_idx = data['pcd_crsp_idx']
        fragment_key = data['fragment_key']
        if self.with_sparse_d:
            depth = data['depth']
            x = torch.cat([rgb, depth], dim=1)
        else:
            x = rgb
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        # pointcloud conv
        if self.training or self.last_fragment_key != fragment_key:
            self.out_pcd_feat = self.pcd_conv(pcd)
        self.last_fragment_key = fragment_key

        # feature project to plane
        pcd_3d_ftr = torch.squeeze(self.out_pcd_feat.x).transpose(1, 0)
        C = pcd_3d_ftr.shape[1]
        B, H, W = pcd_crsp_idx.shape
        pcd_proj_ftr = torch.zeros((B, C, H, W)).float().to(pcd_crsp_idx.device)
        bs, ys, xs = torch.where(pcd_crsp_idx >= 0)
        pcd_proj_ftr[bs, :, ys, xs] = pcd_3d_ftr[pcd_crsp_idx[bs, ys, xs], ...]

        # apply conv to sparse pcd features
        x_pcd = self.layer4_pcd(self.layer3_pcd(self.layer2_pcd(self.layer1_pcd(pcd_proj_ftr))))
        
        # concat features
        # print(x.shape, x_pcd.shape)
        x = torch.cat([x, x_pcd], dim=1)

        x = self.relu(self.bn2(self.conv2(x)))

        x_heatmap = self.decoder_heatmap(x)
        x_heatmap = self.conv3_heatmap(x_heatmap)
        x_heatmap = self.Sigmoid(x_heatmap)

        return {
            # 'depth' : x_depth,
            'depth' : None,
            'heatmap' : x_heatmap
        }
