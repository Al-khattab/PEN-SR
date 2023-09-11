import torch.nn as nn
from torchsummary import summary
from common_blocks import *
class PENsr(nn.Module):
    def __init__(self, conv=default_conv):
        super(PENsr, self).__init__()
        n_resblocks = 16
        n_feats = 64
        kernel_size = 3 
        scale = 4
        n_colors = 3
        rgb_range = 255
        resi_scale=1
        
        n_p_resblocks = 4
        n_p_feats = 16
        act = nn.ReLU(True)
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)
        self.phases = phase(2)
################################################################################
        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]
        # define head module for phases
        m_p_head = [conv(n_colors, n_p_feats, kernel_size)]
################################################################################
        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale = resi_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define body module
        m_p_body = [
            ResBlock(
                conv, n_p_feats, kernel_size, act=act, res_scale = resi_scale
            ) for _ in range(n_p_resblocks)
        ]
        m_p_body.append(conv(n_p_feats, n_p_feats, kernel_size))
################################################################################
        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            #conv(n_feats, n_colors, kernel_size)
        ]

        # define tail module
        m_p_tail = [
            Upsampler_phase(conv, scale, n_p_feats, act=False),
            #conv(n_p_feats, n_colors, kernel_size)
        ]
################################################################################
        # define recon module 
        m_recon = [conv(n_feats*2, n_colors, kernel_size, bias=True)]
################################################################################
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        self.p_head = nn.Sequential(*m_p_head)
        self.p_body = nn.Sequential(*m_p_body)
        self.p_tail = nn.Sequential(*m_p_tail)

        self.recn = nn.Sequential(*m_recon)
################################################################################
    def forward(self, x):
        out = self.sub_mean(x)
        
        #########################
        p1  = self.phases(out)[0]
        p2  = self.phases(out)[1]
        p3  = self.phases(out)[2]
        p4  = self.phases(out)[3]
        #########################
        
        ####################
        out = self.head(out)
        p1 = self.p_head(p1)
        p2 = self.p_head(p2)
        p3 = self.p_head(p3)
        p4 = self.p_head(p4)
        ####################

        ########################
        res = self.body(out)
        res += out
        res_p1 = self.p_body(p1)
        res_p1 += p1
        res_p2 = self.p_body(p2)
        res_p2 += p2
        res_p3 = self.p_body(p3)
        res_p3 += p3
        res_p4 = self.p_body(p4)
        res_p4 += p4
        ########################

        ########################
        out = self.tail(res)
        p1 = self.p_tail(res_p1)
        p2 = self.p_tail(res_p2)
        p3 = self.p_tail(res_p3)
        p4 = self.p_tail(res_p4)
        ########################

        out = torch.cat((out,p1,p2,p3,p4),dim = 1)
        out = self.recn(out)
        out = self.add_mean(out)

        return out
#############################################################################################
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError(f'While copying the parameter named {name}, '
                                           f'whose dimensions in the model are {own_state[name].size()} and '
                                           f'whose dimensions in the checkpoint are {param.size()}.')
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError(f'unexpected key "{name}" in state_dict')
