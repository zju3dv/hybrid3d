import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torchvision
from model.model_pcd import *
from model.model_helper import *
from model.model_vote_factory import get_vote_model_by_name
# from model.model_vote import CandidateVoteModule
# from model.model_vote_v3 import CandidateVoteModule_V3
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

class H3DNetCoordVote(BaseModel):
    def __init__(self, config=None):
        super(H3DNetCoordVote, self).__init__()
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

        # pcd sparse feature conv
        pcd_feat_channels = 32
        pcd_c = [32, 64, 128]
        self.layer_pcd_plain_conv_1 = nn.Sequential(*list(filter(None.__ne__, [
            double_conv(pcd_feat_channels, pcd_c[0], self.relu),
            self.dropout_encoder,
            self.pool,
            double_conv(pcd_c[0], pcd_c[1], self.relu),
            self.dropout_encoder,
        ])))

        self.layer_pcd_plain_conv_2 = nn.Sequential(*list(filter(None.__ne__, [
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
        desc_out_dim = fuse_c//2
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
            nn.Conv2d(fuse_c//4 + rgb_c[-2] + pcd_c[-2], fuse_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(fuse_c),
            self.relu,
            nn.Conv2d(fuse_c, desc_out_dim, kernel_size=3, stride=1, padding=1),
        ])))

        # self.layer_rgb_conv_1.apply(weights_init)
        # self.layer_rgb_conv_2.apply(weights_init)
        # self.layer_pcd_plain_conv_1.apply(weights_init)
        # self.layer_pcd_plain_conv_2.apply(weights_init)
        # self.fuse_conv.apply(weights_init)
        # self.location_header.apply(weights_init)
        # self.heatmap_header.apply(weights_init)
        # self.desc_header_1.apply(weights_init)
        self.apply(weights_init)

        # candidate voting and aggregation
        # input feature+weight
        # self.votenet = CandidateVoteModule(desc_out_dim + 1, desc_out_dim)
        # self.votenet = CandidateVoteModule(desc_out_dim + 1, desc_out_dim)
        # self.votenet = CandidateVoteModule_V3(desc_out_dim + 1, desc_out_dim)
        assert(config != None)
        self.votenet = getattr(
            get_vote_model_by_name(config['arch']['vote_model']), config['arch']['vote_model'])(config=config)

        # pcd_conv layer
        # self.pcd_method = 'pointnet2'
        # self.pcd_method = 'kpconv'
        self.pcd_method = self.votenet.conv_model_type
        # if self.pcd_method == 'pointnet2':
        #     self.pcd_conv = PointNet2Layer(pcd_feat_channels)
        # elif self.pcd_method == 'kpconv':
        #     self.pcd_conv = KPConvLayer(pcd_feat_channels)
        # elif self.pcd_method == 'fcgf':
        #     self.pcd_conv = MinokowskiLayer()
        self.pcd_conv = self.votenet.pcd_conv

        self.last_fragment_key = ''
    
    def get_spatial_ops(self):
        # if self.pcd_method == 'kpconv':
        #     return self.pcd_conv.get_spatial_ops()
        return None

    def forward_vote(self, input, use_cache=False):
        if use_cache:
            input['pcd_conv_cache'] = self.out_pcd_feat
        return self.votenet.forward(input)

    def forward(self, data, cache_pcd_inference=False):
        rgb = data['rgb']
        pcd = data['pcd']
        pcd_crsp_idx = data['pcd_crsp_idx']
        fragment_key = data['fragment_key']

        skip_rgb = self.layer_rgb_conv_1(rgb)
        x_rgb = self.layer_rgb_conv_2(skip_rgb)

        # pointcloud conv
        if (not cache_pcd_inference) or self.last_fragment_key != fragment_key:
            self.out_pcd_feat = self.pcd_conv(pcd)
        self.last_fragment_key = fragment_key

        # feature project to plane
        if self.pcd_method == 'pointnet2':
            pcd_3d_ftr = torch.squeeze(self.out_pcd_feat.x).transpose(1, 0)
        elif self.pcd_method == 'kpconv':
            pcd_3d_ftr = self.out_pcd_feat.x
        elif self.pcd_method == 'fcgf':
            pcd_3d_ftr = self.out_pcd_feat
        elif self.pcd_method == 'fcgf_official':
            pcd_3d_ftr = self.out_pcd_feat['features']
        elif self.pcd_method == 'D3Feat':
            pcd_3d_ftr = self.out_pcd_feat['features']

        # pcd_3d_ftr = F.normalize(pcd_3d_ftr, p=2, dim=1) # NxC

        C = pcd_3d_ftr.shape[1]
        B, H, W = pcd_crsp_idx.shape
        pcd_proj_ftr = torch.zeros((B, C, H, W), device=pcd_crsp_idx.device)
        bs, ys, xs = torch.where(pcd_crsp_idx >= 0)
        if xs.numel() > 0:
            pcd_proj_ftr[bs, :, ys, xs] = pcd_3d_ftr[pcd_crsp_idx[bs, ys, xs], ...]
        else:
            print('Warning: pcd_proj_ftr empty!')

        # apply conv to sparse pcd features
        pcd_proj_ftr = F.interpolate(pcd_proj_ftr, scale_factor=0.5, mode='nearest')
        skip_pcd = self.layer_pcd_plain_conv_1(pcd_proj_ftr)
        x_pcd = self.layer_pcd_plain_conv_2(skip_pcd)
        
        # concat features
        # print(x.shape, x_pcd.shape)
        x = torch.cat([x_rgb, x_pcd], dim=1)

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

        x_desc = self.desc_header_1(x)
        x_desc = torch.cat([x_desc, skip_rgb, skip_pcd], dim=1)
        x_desc = self.desc_header_2(x_desc)
        x_desc = F.normalize(x_desc, p=2, dim=1)

        return {
            'depth' : None,
            'heatmap' : x_heatmap,
            'descriptor': x_desc,
            'coord': coord, # order: x y
        }
