# DWCBS + GhostELAN + CASimAM blocks (custom lightweight backbone/head building blocks)
import math

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, autopad

__all__ = ("DWCBS", "GhostELAN", "CASimAM")


class DWCBS(nn.Module):
    """Depthwise separable conv + BN + SiLU."""

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, k, s, autopad(k), groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))


class _GhostConvRatio(nn.Module):
    """Ghost-style conv with primary + cheap branches (ratio split)."""

    def __init__(self, c1, c2, ratio=2):
        super().__init__()
        init = math.ceil(c2 / ratio)
        new = init * (ratio - 1)
        self.primary = Conv(c1, init, 1)
        self.cheap = Conv(init, new, 3, g=init)
        self.out = c2

    def forward(self, x):
        y = self.primary(x)
        z = self.cheap(y)
        return torch.cat([y, z], 1)[:, : self.out]


class GhostELAN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.m = nn.Sequential(_GhostConvRatio(c_, c_), _GhostConvRatio(c_, c_))
        self.cv3 = Conv(c_ * 2, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.cv1(x), self.m(self.cv2(x))), 1))


class _SimAM(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = ((x - mean) ** 2).mean(dim=[2, 3], keepdim=True)
        y = (x - mean) ** 2 / (4 * (var + 1e-4)) + 0.5
        return x * torch.sigmoid(y)


class _CoordAtt(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c // 4, 1)
        self.conv_h = nn.Conv2d(c // 4, c, 1)
        self.conv_w = nn.Conv2d(c // 4, c, 1)

    def forward(self, x):
        h = x.mean(dim=3, keepdim=True)
        w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([h, w], 2)
        y = self.conv1(y)
        h, w = torch.split(y, [h.shape[2], w.shape[2]], dim=2)
        w = w.permute(0, 1, 3, 2)
        return x * torch.sigmoid(self.conv_h(h)) * torch.sigmoid(self.conv_w(w))


class CASimAM(nn.Module):
    """SimAM + coordinate attention (user variant); parse_model passes (c, *yaml_args)."""

    def __init__(self, c, *_):
        super().__init__()
        self.simam = _SimAM()
        self.ca = _CoordAtt(c)

    def forward(self, x):
        return self.ca(self.simam(x))
