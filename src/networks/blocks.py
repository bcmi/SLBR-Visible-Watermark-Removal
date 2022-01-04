import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import functools
import math
import numbers

from src.utils.model_init import *
from torch import nn, cuda
from torch.autograd import Variable



class ECABlock(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECABlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)

def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1, dilation=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
        dilation=dilation)


def up_conv3x3(in_channels, out_channels, transpose=True):
    if transpose:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv3x3(in_channels, out_channels))




      

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, residual=True,norm=nn.BatchNorm2d, 
        act=F.relu, concat=True,use_att=False, use_mask=False, dilations=[], out_fuse=False):
        super(UpConv, self).__init__()
        self.concat = concat
        self.residual = residual
        self.conv2 = []
        self.use_att = use_att
        self.use_mask = use_mask
        
        self.out_fuse = out_fuse
        self.up_conv = up_conv3x3(in_channels, out_channels, transpose=False)
        if isinstance(norm, str):
            if norm == 'bn':
                norm = nn.BatchNorm2d
            elif norm == 'in':
                norm = nn.InstanceNorm2d
            else:
                raise TypeError("Unknown Type:\t{}".format(norm))
        self.norm0 = norm(out_channels)
        if len(dilations) == 0: dilations = [1] * blocks

        if self.concat:
            self.conv1 = conv3x3(2 * out_channels + int(use_mask), out_channels)
            self.norm1 = norm(out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
            self.norm1 = norm(out_channels)
        for i in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels, dilation=dilations[i], padding=dilations[i]))
        
        self.bn = []
        for _ in range(blocks):
            self.bn.append(norm(out_channels))
        self.bn = nn.ModuleList(self.bn)
        self.conv2 = nn.ModuleList(self.conv2)
        self.act = act

    def forward(self, from_up, from_down, mask=None,se=None):
        from_up = self.act(self.norm0(self.up_conv(from_up)))
        if self.concat:
            if self.use_mask:
                x1 = torch.cat((from_up, from_down, mask), 1)
            else:
                x1 = torch.cat((from_up, from_down), 1)
        else:
            if from_down is not None:
                x1 = from_up + from_down
            else:
                x1 = from_up
        
        xfuse = x1 = self.act(self.norm1(self.conv1(x1)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            x2 = self.bn[idx](x2)
            
            if (se is not None) and (idx == len(self.conv2) - 1): # last 
                x2 = se(x2)

            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        if self.out_fuse:
            return x2, xfuse
        else:
            return x2


class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, pooling=True, norm=nn.BatchNorm2d,act=F.relu,residual=True, dilations=[]):
        super(DownConv, self).__init__()
        self.pooling = pooling
        self.residual = residual
        self.pool = None
        self.conv1 = conv3x3(in_channels, out_channels)
        if isinstance(norm, str):
            if norm == 'bn':
                norm = nn.BatchNorm2d
            elif norm == 'in':
                norm = nn.InstanceNorm2d
            else:
                raise TypeError("Unknown Type:\t{}".format(norm))
        self.norm1 = norm(out_channels)
        if len(dilations) == 0: dilations = [1] * blocks
        self.conv2 = []
        for i in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels, dilation=dilations[i], padding=dilations[i]))
       
        self.bn = []
        for _ in range(blocks):
            self.bn.append(norm(out_channels))
        self.bn = nn.ModuleList(self.bn)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ModuleList(self.conv2)
        self.act = act

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x1 = self.act(self.norm1(self.conv1(x)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        before_pool = x2
        if self.pooling:
            x2 = self.pool(x2)
        return x2, before_pool




class MBEBlock(nn.Module):
    def __init__(self, mode='res_mask', in_channels=512, out_channels=3, norm=nn.BatchNorm2d,act=F.relu, blocks=1, residual=True,
                  concat=True, is_final=True):
        super(MBEBlock, self).__init__()
        self.concat = concat
        self.residual = residual
        self.mode = mode # vanilla, res_mask

        self.up_conv = up_conv3x3(in_channels, out_channels, transpose=False)
        if isinstance(norm, str):
            if norm == 'bn':
                norm = nn.BatchNorm2d
            elif norm == 'in':
                norm = nn.InstanceNorm2d
            else:
                raise TypeError("Unknown Type:\t{}".format(norm))
        self.norm0 = norm(out_channels)

        if self.concat:
            conv1_in = 2*out_channels
        else:
            conv1_in = out_channels
        self.conv1 = conv3x3(conv1_in, out_channels)
        self.norm1 = norm(out_channels)

        # residual structure
        self.conv2 = []
        self.conv3 = []
        for i in range(blocks):
            self.conv2.append(
                nn.Sequential(*[
                    nn.Conv2d(out_channels // 2 + 1, out_channels // 4, 5, 1, 2),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels // 4, 1, 5, 1, 2),
                    nn.Sigmoid()
                ])
            )
            self.conv3.append(conv3x3(out_channels // 2, out_channels))
        
        self.bn = []
        for _ in range(blocks):
            self.bn.append(norm(out_channels))
        self.bn = nn.ModuleList(self.bn)
        self.conv2 = nn.ModuleList(self.conv2)
        self.conv3 = nn.ModuleList(self.conv3)
        self.act = act

    def forward(self, from_up, from_down, mask=None):
        from_up = self.act(self.norm0(self.up_conv(from_up)))
        if self.concat:
            x1 = torch.cat((from_up, from_down), 1)
        else:
            if from_down is not None:
                x1 = from_up + from_down
            else:
                x1 = from_up
        x1 = self.act(self.norm1(self.conv1(x1)))

        # residual structure
        _,C,H,W = x1.shape
        for idx, convs in enumerate(zip(self.conv2, self.conv3)):
            mask = convs[0](torch.cat([x1[:,:C//2], mask], dim=1))
            x2_actv = x1[:,C//2:] * mask
            x2 = convs[1](x1[:,C//2:] + x2_actv)
            x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2        
        return x2
        

  
class SMRBlock(nn.Module):
    def __init__(self, args, ins, outs, norm=nn.BatchNorm2d,act=F.relu, blocks=1, residual=True, concat=True):
        super(SMRBlock, self).__init__()
        self.args = args
        self.mode = args.mask_mode
        self.threshold = 0.5
        self.upconv = UpConv(ins, outs, blocks, residual=residual, concat=concat, norm=norm, act=act, out_fuse=True)
        self.primary_mask = nn.Sequential(*[
            nn.Conv2d(outs,1,1,1,0),
            nn.Sigmoid()
        ])
        self.refine_branch = nn.Sequential(*[
            nn.Conv2d(outs,1,1,1,0),
            nn.Sigmoid()
        ])
        self.self_calibrated = SelfAttentionSimple(self.mode, outs, sim_metric=args.sim_metric, k_center=args.k_center, project_mode=args.project_mode)

    def forward(self, input, encoder_outs=None):
        # upconv features
        mask_x, fuse_x = self.upconv(input, encoder_outs)
        primary_mask = self.primary_mask(mask_x)
        mask_x, self_calibrated_mask = self.self_calibrated(mask_x, mask_x, primary_mask)
        return {"feats":[mask_x], "attn_maps":[primary_mask, self_calibrated_mask]}

class SelfAttentionSimple(nn.Module):
    def __init__(self, mode, in_channel, k_center=1, sim_metric='fc', project_mode='linear'):
        super(SelfAttentionSimple, self).__init__()
        self.k_center = k_center # 1 for foreground, 2 for background & foreground
        self.reduction = 1
        self.mode = mode
        self.project_mode = project_mode
        self.q_conv = nn.Conv2d(in_channel, in_channel, 1,1,0)
        self.k_conv = nn.Conv2d(in_channel, in_channel*k_center, 1,1,0)
        self.v_conv = nn.Conv2d(in_channel, in_channel*k_center, 1,1,0)
   
        self.min_area = 100
        self.threshold = 0.5
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//8, 3,1,1),
            nn.ReLU(True),
            nn.Conv2d(in_channel//8, 1, 3,1,1)
        ) 
        
        
        self.sim_func = nn.Conv2d(in_channel + in_channel, 1, 1,1,0)
        self.k_weight = nn.Parameter(torch.full((1,k_center,1,1), fill_value=1,dtype=torch.float32), requires_grad=True)
        
    def compute_attention(self, query, key, mask, eps=1):  # in: [B, C:128, 64, 64]
        b,c,h,w = query.shape
        query_org = query
        query = self.q_conv(query)
        key_in = key
        key = self.k_conv(key_in)
        keys = list(key.split(c,dim=1))
        
        importance_map = torch.where(mask >= self.threshold, torch.ones_like(mask), torch.zeros_like(mask)).to(mask.device)
        s_area = torch.clamp_min(torch.sum(importance_map, dim=[2,3]), self.min_area)[:,0:1]
        if self.k_center != 2:
            keys = [torch.sum(k*importance_map, dim=[2,3]) / s_area for k in keys] # b,c * k
        else:
            keys = [
                torch.sum(keys[0]*importance_map, dim=[2,3]) / s_area,
                torch.sum(keys[1]*(1-importance_map), dim=[2,3]) / (keys[1].shape[2]*keys[1].shape[3] - s_area + eps)
            ]

        f_query = query # b, c, h, w
        f_key = [k.reshape(b,c,1,1).repeat(1, 1, f_query.size(2),f_query.size(3)) for k in keys]
        attention_scores = []
        for k in f_key:
            combine_qk = torch.cat([f_query, k],dim=1).tanh() # tanh
            sk = self.sim_func(combine_qk)
            attention_scores.append(sk)
        s = ascore = torch.cat(attention_scores, dim=1) # b,k,h,w
        
        s = s.permute(0,2,3,1) # b,h,w,k
        v = self.v_conv(key_in)
        if self.k_center == 2:
            v_fg = torch.sum(v[:,:c]*importance_map, dim=[2,3]) / s_area
            v_bg = torch.sum(v[:,c:]*(1-importance_map), dim=[2,3]) / (v.shape[2]*v.shape[3] - s_area + eps)
            v = torch.cat([v_fg, v_bg],dim=1)
        else:
            v = torch.sum(v*importance_map, dim=[2,3]) / s_area # b, c*k
        v = v.reshape(b, self.k_center, c) # b, k, c
        attn = torch.bmm(s.reshape(b,h*w,self.k_center), v).reshape(b,h,w,c).permute(0,3,1,2)
        s = self.out_conv(attn + query)
        return s


    def forward(self, xin, xout, xmask):
        b_num,c,h,w = xin.shape
        attention_score = self.compute_attention(xin, xout, xmask) # b,h*w,k
        attention_score = attention_score.reshape(b_num,1,h,w)
        return xout, attention_score.sigmoid()
        
        





## Refinement Stage
class ResDownNew(nn.Module):
    def __init__(self, in_size, out_size, pooling=True, use_att=False, dilation=False):
        super(ResDownNew, self).__init__()
        self.model = DownConv(in_size, out_size, 3, pooling=pooling, norm=nn.InstanceNorm2d, act=F.leaky_relu, dilations=[1,2,5] if dilation else [])

    def forward(self, x):
        return self.model(x)

class ResUpNew(nn.Module):
    def __init__(self, in_size, out_size, use_att=False):
        super(ResUpNew, self).__init__()
        self.model = UpConv(in_size, out_size, 3, use_att=use_att, norm=nn.InstanceNorm2d)

    def forward(self, x, skip_input, mask=None):
        return self.model(x,skip_input,mask)


        

class CFFBlock(nn.Module):
    def __init__(self, down=ResDownNew, up=ResUpNew, ngf = 32):
        super(CFFBlock, self).__init__()
        self.down1 = down(ngf, ngf)
        self.down2 = down(ngf, ngf*2)
        self.down3 = down(ngf*2, ngf*4, pooling=False, dilation=True)

        self.conv22 = nn.Sequential(*[
            nn.Conv2d(ngf*2, ngf, 3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf, ngf, 3,1,1),
            nn.LeakyReLU(0.2)
        ])

        self.conv33 = nn.Sequential(*[
            nn.Conv2d(ngf*4,ngf*2, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf*2, ngf*2, 3,1,1),
            nn.LeakyReLU(0.2)
        ])

        self.up32 = nn.Sequential(*[
            nn.Conv2d(ngf*4, ngf*1, 3,1,1),
            nn.LeakyReLU(0.2),
        ])

        self.up31 = nn.Sequential(*[
            nn.Conv2d(ngf*4, ngf*1, 3,1,1),
            nn.LeakyReLU(0.2),
        ])

    def forward(self, inputs):
        x1,x2,x3 = inputs
        x32 = F.interpolate(x3, size=x2.shape[2:][::-1], mode='bilinear')
        x32 = self.up32(x32)
        x31 = F.interpolate(x3, size=x1.shape[2:][::-1], mode='bilinear')
        x31 = self.up31(x31)

        # cross-connection
        x,d1 = self.down1(x1 + x31)
        x,d2 = self.down2(x + self.conv22(x2) + x32)
        d3,_ = self.down3(x + self.conv33(x3))
        return [d1,d2,d3]






