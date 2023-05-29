import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math

from model.decoder_part import make_deconv_layer

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class HybridPlainResNet_v3(BaseModel):
    def __init__(self, layers=50, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(HybridPlainResNet_v3, self).__init__()
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

        # unlock conv_4_6
        for param in self.layer3[5].parameters():
            param.requires_grad = True

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048

        # depth predictor
        self.decoder_depth = make_deconv_layer(
            4,
            [num_channels//4, num_channels//8, num_channels//16, num_channels//32],
            [4, 4, 4, 4],
            num_channels//2
        )
        self.conv3_depth = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=True) # setting bias=true doesn't improve accuracy

        # heatmap
        self.decoder_heatmap = make_deconv_layer(
            4,
            [num_channels//4, num_channels//8, num_channels//16, num_channels//32],
            [4, 4, 4, 4],
            num_channels//2
        )
        self.conv3_heatmap = nn.Conv2d(num_channels//32,1,kernel_size=1)
        self.Sigmoid = nn.Sigmoid()

        # weight init
        self.decoder_depth.apply(weights_init)
        self.conv3_depth.apply(weights_init)
        self.decoder_heatmap.apply(weights_init)
        self.conv3_heatmap.apply(weights_init)

    def forward(self, data):
        rgb = data['rgb']
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


        # decoder
        if self.training:
            x_depth = self.decoder_depth(x)
            x_depth = self.conv3_depth(x_depth)
        else:
            x_depth = None

        x_heatmap = self.decoder_heatmap(x)
        x_heatmap = self.conv3_heatmap(x_heatmap)
        x_heatmap = self.Sigmoid(x_heatmap)

        return {
            'depth' : x_depth,
            'heatmap' : x_heatmap
        }
