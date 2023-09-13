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

class Phase(nn.Module):
    def __init__(self, size):
        super(Phase, self).__init__()
        self.phases = size * size
        weight = torch.ones((self.phases, 1, 2, 2), dtype=torch.float, device='cuda', requires_grad=False)
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.phases, kernel_size=2, stride=2, bias=False)
        self.conv.weight = nn.Parameter(weight)
        self.conv.weight.to(device='cuda', dtype=torch.float)
        for params in self.parameters():
            params.requires_grad = False

    def forward(self, x):
        phases = [self.conv(x[:, i:i+1, :, :]) for i in range(x.size(1))]
        return torch.cat(phases, dim=1)

class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output

class _Residual_Block_Phase(nn.Module):
    def __init__(self):
        super(_Residual_Block_Phase, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = MeanShift(rgb_mean, -1)
        self.phases = Phase(2)

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.input_phases = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(_Residual_Block, 16)
        self.phase_residual = self.make_layer(_Residual_Block_Phase, 4)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_mid_phases = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

        self.upscale4x_phases = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=16, out_channels=16*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=16, out_channels=16*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

        self.conv_output = nn.Conv2d(in_channels=64*2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.add_mean = MeanShift(rgb_mean, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layers):
        return nn.Sequential(*[block() for _ in range(num_of_layers)])

    def forward(self, x):
        out = self.sub_mean(x)
        phases = self.phases(out)
        
        out = self.conv_input(out)
        phases = self.input_phases(phases)
        
        residual = out
        out = self.conv_mid(self.residual(out))
        phases = self.conv_mid_phases(self.phase_residual(phases))
        
        out += residual
        t_list = [out, phases]
        
        out = self.upscale4x(out)
        phases = self.upscale4x_phases(phases)
        
        t_list.extend([phases] * 4)
        
        final = torch.cat(t_list, dim=1)
        final = self.conv_output(final)
        out = self.add_mean(final)
        
        return out