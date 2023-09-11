import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
    
class phase(nn.Module):
    def phasing_filters (self,kernal_size):
        phases = kernal_size * kernal_size
        tensor = torch.zeros((phases,kernal_size,kernal_size), dtype=torch.float, device = 'cuda',requires_grad=False)
        x = 0
        while (x<phases):
            for y in range(kernal_size):
                for z in range(kernal_size):
                    tensor[x][y][z] = 1.;
                    x = x+1
        tensor = tensor[None,:,:,:]
        tensor = tensor.permute(1,0,2,3)
        return tensor
    
    def __init__(self, size):
        self.phases = size*size
        weight = self.phasing_filters(size)
        super(phase, self).__init__()
        self.conv = nn.Conv2d(in_channels = 1, out_channels = self.phases, kernel_size=(2, 2), bias=False ,stride= (2,2))
        self.conv.weight = torch.nn.Parameter(weight)
        self.conv.weight.to(device='cuda', dtype=torch.float)
        for params in self.parameters():
            params.requires_grad = False
            
    def forward(self, x):
        final = []
        for i in range(len(x[:])):
            x1 = x[None,i,0,:,:]
            x2 = x[None,i,1,:,:]
            x3 = x[None,i,2,:,:]
            x1 = torch.unsqueeze(x1, dim=0).to('cuda')
            x2 = torch.unsqueeze(x2, dim=0).to('cuda')
            x3 = torch.unsqueeze(x3, dim=0).to('cuda')
            conv_x1 = self.conv(x1)
            conv_x2 = self.conv(x2)
            conv_x3 = self.conv(x3)
            outputs = []
            for c in range(self.phases):
                F = torch.cat((conv_x1[None,:,c,:,:],conv_x2[None,:,c,:,:],conv_x3[None,:,c,:,:]),0)
                outputs.append(F)
                results = torch.cat(outputs, dim=1)
                results = results.permute(1,0,2,3)
                results = results[None,:,:,:,:]
            final.append(results)
        final = torch.cat(final, dim=0)
        final = final.permute(1,0,2,3,4)
        return final

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
        
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class Upsampler_phase(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2)+1)):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler_phase, self).__init__(*m)
