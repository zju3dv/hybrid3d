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

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Decoder(nn.Module):
    # Decoder is the base class for all decoders
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class DeConv(Decoder):
    def __init__(self, in_channels, kernel_size):
        assert kernel_size>=2, "kernel_size out of range: {}".format(kernel_size)
        super(DeConv, self).__init__()

        def convt(in_channels):
            stride = 2
            padding = (kernel_size - 1) // 2
            output_padding = kernel_size % 2
            assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"

            module_name = "deconv{}".format(kernel_size)
            return nn.Sequential(collections.OrderedDict([
                  (module_name, nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size,
                        stride,padding,output_padding,bias=False)),
                  ('batchnorm', nn.BatchNorm2d(in_channels//2)),
                  ('relu',      nn.ReLU(inplace=True)),
                ]))

        self.layer1 = convt(in_channels)
        self.layer2 = convt(in_channels // 2)
        self.layer3 = convt(in_channels // (2 ** 2))
        self.layer4 = convt(in_channels // (2 ** 3))


def choose_decoder(decoder, in_channels):
    # iheight, iwidth = 10, 8
    if decoder[:6] == 'deconv':
        assert len(decoder)==7
        kernel_size = int(decoder[6])
        return DeConv(in_channels, kernel_size)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)


class HybridPlainResNet_v2(BaseModel):
    def __init__(self, layers=50, decoder='deconv3', output_size=(480, 640), in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))

        super(HybridPlainResNet_v2, self).__init__()
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

        self.output_size = output_size

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

        # self.conv2 = nn.Conv2d(num_channels,num_channels//2,kernel_size=1,bias=False)
        # self.bn2 = nn.BatchNorm2d(num_channels//2)

        # depth predictor
        self.decoder_depth = choose_decoder(decoder, num_channels//2)
        self.conv3_depth = nn.Conv2d(num_channels//32,1,kernel_size=3,stride=1,padding=1,bias=True) # setting bias=true doesn't improve accuracy
        # self.bilinear_depth = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)

        # heatmap
        self.decoder_heatmap = choose_decoder(decoder, num_channels//2)
        self.conv3_heatmap = nn.Conv2d(num_channels//32,1,kernel_size=1)
        # self.bilinear_heatmap = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
        self.Sigmoid = nn.Sigmoid()

        # weight init
        # self.conv2.apply(weights_init)
        # self.bn2.apply(weights_init)
        # self.decoder_depth.apply(weights_init)
        # self.conv3_depth.apply(weights_init)
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
        # x = self.layer4(x)

        # x = self.conv2(x)
        # x = self.bn2(x)

        # decoder
        if self.training:
            x_depth = self.decoder_depth(x)
            x_depth = self.conv3_depth(x_depth)
            # x_depth = self.bilinear_depth(x_depth)
        else:
            x_depth = None

        x_heatmap = self.decoder_heatmap(x)
        x_heatmap = self.conv3_heatmap(x_heatmap)
        # x_heatmap = self.bilinear_heatmap(x_heatmap)
        x_heatmap = self.Sigmoid(x_heatmap)

        return {
            'depth' : x_depth,
            'heatmap' : x_heatmap
        }
