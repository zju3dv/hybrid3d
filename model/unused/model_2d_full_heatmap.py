import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision
from model.model_resnet import *
from model.model_helper import image_grid
from model.decoder_part import make_deconv_layer
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

class H2DFullHeatmap(BaseModel):
    def __init__(self, config=None):
        super(H2DFullHeatmap, self).__init__()
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
        self.rgb_heatmap_header = nn.Sequential(*list(filter(None.__ne__, [
            make_deconv_layer(3, [fuse_c//2, fuse_c//4, fuse_c//8], [4, 4, 4], fuse_c, leaky_relu=use_leaky_relu),
            self.dropout_decoder,
            nn.Conv2d(fuse_c//8, fuse_c//8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fuse_c//8),
            self.relu,
            self.dropout_decoder,
            nn.Conv2d(fuse_c//8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
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
        self.rgb_heatmap_header.apply(weights_init)
        self.desc_header_1.apply(weights_init)
        self.desc_header_2.apply(weights_init)

    def forward(self, data, cache_pcd_inference=False):
        rgb = data['rgb']
        B, _, H, W = rgb.shape
        skip_rgb = self.layer_rgb_conv_1(rgb)
        x_rgb = self.layer_rgb_conv_2(skip_rgb)
        x = x_rgb

        x_rgb_heatmap = self.rgb_heatmap_header(x_rgb)

        x_desc = self.desc_header_1(x)
        x_desc = torch.cat([x_desc, skip_rgb], dim=1)
        x_desc = self.desc_header_2(x_desc)
        x_desc = F.normalize(x_desc, p=2, dim=1)

        return {
            'depth' : None,
            'heatmap' : x_rgb_heatmap,
            'descriptor': x_desc,
        }
