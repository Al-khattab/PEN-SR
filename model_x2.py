import torch
import torch.nn as nn
import math

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) # W H C N
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)
        for params in self.parameters():
            params.requires_grad = False

class phase(nn.Module):
    def phasing_filters(self, kernal_size):
        phases = kernal_size * kernal_size
        tensor = torch.zeros((phases, kernal_size, kernal_size), dtype=torch.float, device='cuda', requires_grad=False)
        x = 0
        while (x < phases):
            for y in range(kernal_size):
                for z in range(kernal_size):
                    tensor[x][y][z] = 1.
                    x = x + 1
        tensor = tensor[None, :, :, :]
        tensor = tensor.permute(1, 0, 2, 3)
        return tensor

    def __init__(self, size):
        self.phases = size * size
        weight = self.phasing_filters(size)
        super(phase, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.phases, kernel_size=(2, 2), bias=False, stride=(2, 2))
        self.conv.weight = torch.nn.Parameter(weight)
        self.conv.weight.to(device='cuda', dtype=torch.float)
        for params in self.parameters():
            params.requires_grad = False

    def forward(self, x):
        phases = [self.phases(out) for out in x]
        results = [self.conv(x_p) for x_p in phases]
        results = torch.cat(results, dim=1)
        results = results.permute(1, 0, 2, 3)
        results = results[None, :, :, :, :]
        return results

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        identity_data = x
        output = self.model(x)
        output += identity_data
        return output

class _Residual_Block_phase(nn.Module):
    def __init__(self):
        super(_Residual_Block_phase, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        identity_data = x
        output = self.model(x)
        output += identity_data
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = MeanShift(rgb_mean, -1)
        self.phases = phase(2)

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.input_phases = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(_Residual_Block, 16)
        self.phase_residual = self.make_layer(_Residual_Block_phase, 4)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_mid_phases = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

        self.upscale2x_phases = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16 * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=16, out_channels=16 * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

        self.conv_output = nn.Conv2d(in_channels=64 * 2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill
