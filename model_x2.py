from torchsummary import summary
import torch
import torch.nn as nn
import math

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) # W H C N
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False
            
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
        # Freeze the MeanShift layer
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
        output *= 1
        output = torch.add(output,identity_data)
        return output 
    
class _Residual_Block_phase(nn.Module): 
    def __init__(self):
        super(_Residual_Block_phase, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x): 
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 1
        output = torch.add(output,identity_data)
        return output 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = MeanShift(rgb_mean, -1)
        
        self.phases = phase(2)
        
        #input conv
        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.input_phases = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        
        #residual block
        self.residual = self.make_layer(_Residual_Block, 16)
        self.phase_residual = self.make_layer(_Residual_Block_phase, 4)
        
        #mid convolution
        self.conv_mid = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_mid_phases = nn.Conv2d(in_channels = 16,  out_channels = 16,  kernel_size=3, stride=1, padding=1, bias=False)

        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),)
        
        self.upscale2x_phases = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 16*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels = 16, out_channels = 16*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),)

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

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.sub_mean(x)
        
        p1  = self.phases(out)[0]
        p2  = self.phases(out)[1]
        p3  = self.phases(out)[2]
        p4  = self.phases(out)[3]
        
        out = self.conv_input(out)
        p1  = self.input_phases(p1)
        p2  = self.input_phases(p2)
        p3  = self.input_phases(p3)
        p4  = self.input_phases(p4)
        
        residual = out
        rp1 = p1
        rp2 = p2
        rp3 = p3
        rp4 = p4
        
        out = self.conv_mid(self.residual(out))
        p1  = self.conv_mid_phases(self.phase_residual(p1))
        p2  = self.conv_mid_phases(self.phase_residual(p2))
        p3  = self.conv_mid_phases(self.phase_residual(p3))
        p4  = self.conv_mid_phases(self.phase_residual(p4))
        
        out = torch.add(out,residual)
        p1  = torch.add(p1,rp1)
        p2  = torch.add(p2,rp2)
        p3  = torch.add(p3,rp3)
        p4  = torch.add(p4,rp4)
        
        out = self.upscale2x(out)
        p1  = self.upscale2x_phases(p1)
        p2  = self.upscale2x_phases(p2)
        p3  = self.upscale2x_phases(p3)
        p4  = self.upscale2x_phases(p4)
        
        final = torch.cat((out,p1,p2,p3,p4),dim = 1)
        final = self.conv_output(final)
        out = self.add_mean(final)
        
        
        return out
