import torch
import torch.nn as nn
import math

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)
        for params in self.parameters():
            params.requires_grad = False

class phase(nn.Module):
    def phasing_filters(self, kernal_size):
        phases = kernal_size * kernal_size
        tensor = torch.eye(phases, kernal_size, kernal_size).view(phases, kernal_size, kernal_size)
        tensor = tensor[None, :, :, :].permute(1, 0, 2, 3)
        return tensor

    def __init__(self, size):
        super(phase, self).__init__()
        self.phases = size * size
        weight = self.phasing_filters(size)
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.phases, kernel_size=2, bias=False, stride=2)
        self.conv.to(torch.device('cuda'))
        for params in self.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = x.to(torch.device('cuda'))
        conv_x = self.conv(x)
        results = conv_x.permute(1, 0, 2, 3)[None, :, :, :, :]
        return results

class _Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu
