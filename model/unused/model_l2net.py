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


class PlainQuadL2Net(BaseModel):
    def __init__(self, in_channels=3, dim=128, mchan=4, dilated=True, dilation=1, bn=True, bn_affine=False, relu22=False):
        super(PlainQuadL2Net, self).__init__()

        self.with_sparse_d = False
        if self.with_sparse_d:
            in_channels = 4
        else:
            in_channels = 3

        self.inchan = in_channels
        self.curchan = in_channels
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine

        self.ops = nn.ModuleList([])

        self._add_conv(  8*mchan)
        self._add_conv(  8*mchan)
        self._add_conv( 16*mchan, stride=2)
        self._add_conv( 16*mchan)
        self._add_conv( 32*mchan, stride=2)
        self._add_conv( 32*mchan)
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = dim

        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:,1:2]

    def normalize(self, x, ureliability, urepeatability):
        return dict(descriptors = F.normalize(x, p=2, dim=1),
                    repeatability = self.softmax( urepeatability ),
                    reliability = self.softmax( ureliability ))

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True):
        d = self.dilation * dilation
        if self.dilated: 
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=1)
            self.dilation *= stride
        else:
            conv_params = dict(padding=((k-1)*d)//2, dilation=d, stride=stride)
        self.ops.append( nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params) )
        if bn and self.bn: self.ops.append( self._make_bn(outd) )
        if relu: self.ops.append( nn.ReLU(inplace=True) )
        self.curchan = outd

    def forward(self, data):
        rgb = data['rgb']
        if self.with_sparse_d:
            depth = data['depth']
            x = torch.cat([rgb, depth], dim=1)
        else:
            x = rgb
        # print(x.shape)
        for op in self.ops:
            x = op(x)
        heatmap = self.clf(x**2)
        heatmap = self.softmax(heatmap)
        return {
            'depth': None,
            'heatmap': heatmap
        }
