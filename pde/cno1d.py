# CNO1d adapted from the authors' reference implementation:
# Raonić et al., "Convolutional Neural Operators for robust and accurate learning of PDEs",
# NeurIPS 2023; github.com/camlab-ethz/ConvolutionalNeuralOperator, CNO1d_simplified/CNO1d.py
# (MIT License, (c) 2024 Computational and Applied Mathematics Laboratory @ ETH Zurich).
# Changes: batch norm off by default (a stepper's Jacobian must not depend on batch statistics),
# class and argument names unchanged otherwise.

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNO_LReLu(nn.Module):
    """Activation on a 2x upsampled grid, antialiased back to out_size."""

    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.interpolate(x.unsqueeze(2), size=(1, 2 * self.in_size), mode="bicubic", antialias=True)
        x = self.act(x)
        x = F.interpolate(x, size=(1, self.out_size), mode="bicubic", antialias=True)
        return x[:, :, 0]


class CNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_size, out_size, use_bn=False):
        super().__init__()
        self.convolution = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.act = CNO_LReLu(in_size, out_size)

    def forward(self, x):
        return self.act(self.batch_norm(self.convolution(x)))


class LiftProjectBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, latent_dim=64):
        super().__init__()
        self.inter_CNOBlock = CNOBlock(in_channels, latent_dim, size, size, use_bn=False)
        self.convolution = nn.Conv1d(latent_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.convolution(self.inter_CNOBlock(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels, size, use_bn=False):
        super().__init__()
        self.convolution1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.convolution2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(channels) if use_bn else nn.Identity()
        self.batch_norm2 = nn.BatchNorm1d(channels) if use_bn else nn.Identity()
        self.act = CNO_LReLu(size, size)

    def forward(self, x):
        out = self.act(self.batch_norm1(self.convolution1(x)))
        return x + self.batch_norm2(self.convolution2(out))


class ResNet(nn.Module):
    def __init__(self, channels, size, num_blocks, use_bn=False):
        super().__init__()
        self.res_nets = nn.Sequential(*[ResidualBlock(channels, size, use_bn) for _ in range(num_blocks)])

    def forward(self, x):
        return self.res_nets(x)


class CNO1d(nn.Module):
    """Operator U-Net: lift, N_layers down blocks with residual skips, neck, up blocks, project.
    Input and output have the fixed spatial size `size` (divisible by 2**N_layers)."""

    def __init__(self, in_dim, out_dim, size, N_layers, N_res=4, N_res_neck=4,
                 channel_multiplier=16, use_bn=False):
        super().__init__()
        self.N_layers = int(N_layers)
        self.lift_dim = channel_multiplier // 2
        self.encoder_features = [self.lift_dim] + [2 ** i * channel_multiplier for i in range(self.N_layers)]
        self.decoder_features_in = self.encoder_features[1:][::-1]
        self.decoder_features_out = self.encoder_features[:-1][::-1]
        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2 * self.decoder_features_in[i]
        self.encoder_sizes = [size // 2 ** i for i in range(self.N_layers + 1)]
        self.decoder_sizes = [size // 2 ** (self.N_layers - i) for i in range(self.N_layers + 1)]

        self.lift = LiftProjectBlock(in_dim, self.encoder_features[0], size)
        self.project = LiftProjectBlock(self.encoder_features[0] + self.decoder_features_out[-1], out_dim, size)
        self.encoder = nn.ModuleList([CNOBlock(self.encoder_features[i], self.encoder_features[i + 1],
                                               self.encoder_sizes[i], self.encoder_sizes[i + 1], use_bn)
                                      for i in range(self.N_layers)])
        self.ED_expansion = nn.ModuleList([CNOBlock(self.encoder_features[i], self.encoder_features[i],
                                                    self.encoder_sizes[i], self.decoder_sizes[self.N_layers - i], use_bn)
                                           for i in range(self.N_layers + 1)])
        self.decoder = nn.ModuleList([CNOBlock(self.decoder_features_in[i], self.decoder_features_out[i],
                                               self.decoder_sizes[i], self.decoder_sizes[i + 1], use_bn)
                                      for i in range(self.N_layers)])
        self.res_nets = nn.Sequential(*[ResNet(self.encoder_features[l], self.encoder_sizes[l], int(N_res), use_bn)
                                        for l in range(self.N_layers)])
        self.res_net_neck = ResNet(self.encoder_features[self.N_layers], self.encoder_sizes[self.N_layers],
                                   int(N_res_neck), use_bn)

    def forward(self, x):
        x = self.lift(x)
        skip = []
        for i in range(self.N_layers):
            skip.append(self.res_nets[i](x))
            x = self.encoder[i](x)
        x = self.res_net_neck(x)
        for i in range(self.N_layers):
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x)
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i])), 1)
            x = self.decoder[i](x)
        x = torch.cat((x, self.ED_expansion[0](skip[0])), 1)
        return self.project(x)
