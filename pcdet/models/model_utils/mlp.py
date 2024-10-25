import torch
import torch.nn as nn

from typing import Tuple
from ...ops.pointnet2.pointnet2_batch import pointnet2_utils

class MLP(nn.Module):
    def __init__(self,
                 in_channel=128,
                 conv_channels=(256,256),
                 bias=True):
        super().__init__()
        self.mlp = nn.Sequential()
        prev_channels = in_channel
        for i, conv_channel in enumerate(conv_channels):
            self.mlp.add_module(
                f'layer{i}',
                BasicBlock1D(
                    prev_channels,
                    conv_channels[i],
                    kernel_size=1,
                    padding=0,
                    bias=bias
                )
            )
            prev_channels = conv_channels[i]

    def forward(self, img_features):
        return self.mlp(img_features)


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(
            self.conv.weight, a=0, mode='fan_out', nonlinearity='relu'
        )
        if hasattr(self.conv, 'bias') and self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, features):
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return  x