import torch
import torch.nn as nn

def get_deconv_cfg(deconv_kernel, index):
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0

    return deconv_kernel, padding, output_padding

def make_deconv_layer(num_layers, num_filters, num_kernels, in_channels, deconv_with_bias=False, leaky_relu=False):
    assert num_layers == len(num_filters), \
        'ERROR: num_deconv_layers is different len(num_deconv_filters)'
    assert num_layers == len(num_kernels), \
        'ERROR: num_deconv_layers is different len(num_deconv_filters)'
    BN_MOMENTUM = 0.1
    layers = []
    inplanes = in_channels
    for i in range(num_layers):
        kernel, padding, output_padding = \
            get_deconv_cfg(num_kernels[i], i)

        planes = num_filters[i]
        layers.append(
            nn.ConvTranspose2d(
                in_channels=inplanes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=deconv_with_bias))
        layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
        layers.append(nn.LeakyReLU(inplace=True) if leaky_relu else nn.ReLU(inplace=True))
        inplanes = planes
    return nn.Sequential(*layers)
