import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision
from model.model_resnet import *
from model.model_pcd import *
import math
import time

class Hybrid3DNet(BaseModel):
    def __init__(self, layers=34, decoder='deconv3', output_size=(480, 640), in_channels=4, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(Hybrid3DNet, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)

        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)

        self.output_size = output_size

        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels//2)

        # depth predictor
        self.decoder_depth = choose_decoder(decoder, num_channels//2)
        self.conv3_depth = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=True) # setting bias=true doesn't improve accuracy
        self.bilinear_depth = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # heatmap
        self.decoder_heatmap = choose_decoder(decoder, num_channels//2)
        # self.conv3_heatmap = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=True)
        self.conv3_heatmap = nn.Conv2d(32, num_channels//32,kernel_size=3,stride=1,padding=1,bias=True)
        self.bn3 = nn.BatchNorm2d(num_channels//32)
        self.conv4_heatmap = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=True)
        self.bilinear_heatmap = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.Sigmoid = nn.Sigmoid()

        # weight init
        self.conv2.apply(weights_init)
        self.bn2.apply(weights_init)
        self.decoder_depth.apply(weights_init)
        self.conv3_depth.apply(weights_init)
        self.decoder_heatmap.apply(weights_init)
        self.conv3_heatmap.apply(weights_init)
        self.bn3.apply(weights_init)
        self.conv4_heatmap.apply(weights_init)

        # kpconv layer
        self.kpconv = KPConvLayer()
        self.last_fragment_key = ''
    
    def get_spatial_ops(self):
        return self.kpconv.get_spatial_ops()

    def forward(self, data):
        rgb = data['rgb']
        depth = data['depth']
        pcd = data['pcd']
        pcd_crsp_idx = data['pcd_crsp_idx']
        fragment_key = data['fragment_key']
        x = torch.cat([rgb, depth], dim=1)
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # pointcloud conv
        if self.training or self.last_fragment_key != fragment_key:
            t1 = time.time()
            self.out_pcd_feat = self.kpconv(pcd)
            t2 = time.time()
            # print(type(self.out_pcd_feat), self.out_pcd_feat, 'time {}s'.format(t2 - t1))
        self.last_fragment_key = fragment_key

        # feature project to plane
        pcd_3d_ftr = self.out_pcd_feat.x
        C = pcd_3d_ftr.shape[1]
        B, H, W = pcd_crsp_idx.shape
        pcd_proj_ftr = torch.zeros((B, C, H, W)).float().to(pcd_crsp_idx.device)
        bs, ys, xs = torch.where(pcd_crsp_idx >= 0)
        pcd_proj_ftr[bs, :, ys, xs] = pcd_3d_ftr[pcd_crsp_idx[bs, ys, xs], ...]

        # decoder
        x_depth = self.decoder_depth(x)
        x_depth = self.conv3_depth(x_depth)
        x_depth = self.bilinear_depth(x_depth)

        x_heatmap = self.decoder_heatmap(x)
        x_heatmap = self.bilinear_heatmap(x_heatmap)
        # print(x_heatmap.shape, pcd_proj_ftr.shape)
        x_heatmap = self.conv3_heatmap(torch.cat([x_heatmap, pcd_proj_ftr], dim=1))
        x_heatmap = self.relu(self.bn3(x_heatmap))
        x_heatmap = self.conv4_heatmap(x_heatmap)
        x_heatmap = self.Sigmoid(x_heatmap)

        return x_depth, x_heatmap
