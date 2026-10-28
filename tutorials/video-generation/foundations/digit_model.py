"""A small optionally class-conditioned MNIST U-Net with a global 7x7 attention bottleneck."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies', torch.exp(torch.linspace(0, math.log(100), 32)))
        self.mlp = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))

    def forward(self, t):
        phase = t[:, None] * self.frequencies[None]
        return self.mlp(torch.cat((phase.sin(), phase.cos()), 1))


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time = nn.Linear(128, 2*cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(F.silu(t)).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:,:,None,None]) + shift[:,:,None,None]
        h = self.conv2(F.silu(h))
        return (self.skip(x) + h) / math.sqrt(2)


class Attention(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, 3*channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        b,c,h,w = x.shape
        q,k,v = self.qkv(self.norm(x).flatten(2)).chunk(3,dim=1)
        def heads(a): return a.reshape(b,4,c//4,h*w).transpose(-1,-2)
        y = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        y = self.proj(y.transpose(-1,-2).reshape(b,c,h*w)).reshape_as(x)
        return (x+y)/math.sqrt(2)


class DigitUNet(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()
        self.time = TimeFeatures()
        self.class_embedding = nn.Embedding(num_classes,128) if num_classes is not None else None
        self.input = nn.Conv2d(1,32,3,padding=1)
        self.enc1 = Block(32,32)
        self.down1 = nn.Conv2d(32,64,3,stride=2,padding=1)  # 28 -> 14
        self.enc2 = Block(64,64)
        self.down2 = nn.Conv2d(64,128,3,stride=2,padding=1)  # 14 -> 7
        self.mid1 = Block(128,128)
        self.attn = Attention(128)
        self.mid2 = Block(128,128)
        self.up2 = nn.Conv2d(128,64,3,padding=1)
        self.dec2 = Block(128,64)
        self.up1 = nn.Conv2d(64,32,3,padding=1)
        self.dec1 = Block(64,32)
        self.output = nn.Sequential(nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,1,3,padding=1))

    def forward(self,x,t,y=None):
        t=self.time(t)
        if self.class_embedding is not None:
            if y is None:
                raise ValueError('Class labels are required by this conditional model.')
            t=t+self.class_embedding(y)
        s1=self.enc1(self.input(x),t)
        s2=self.enc2(self.down1(s1),t)
        h=self.mid2(self.attn(self.mid1(self.down2(s2),t)),t)
        h=self.up2(F.interpolate(h,size=s2.shape[-2:],mode='nearest'))
        h=self.dec2(torch.cat((h,s2),1),t)
        h=self.up1(F.interpolate(h,size=s1.shape[-2:],mode='nearest'))
        return self.output(self.dec1(torch.cat((h,s1),1),t))
