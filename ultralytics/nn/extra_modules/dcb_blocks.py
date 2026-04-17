from __future__ import annotations

import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class RepVGGDW_DCB(nn.Module):
    """RepVGG-style depthwise block used by DCB (kept separate to avoid name collisions)."""

    def __init__(self, c: int):
        super().__init__()
        self.dw3 = Conv(c, c, 3, 1, g=c, act=False)
        self.dw1 = Conv(c, c, 1, 1, g=c, act=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.dw3(x) + self.dw1(x))


class DCB(nn.Module):
    """DCB block from user snippet."""

    def __init__(self, c: int):
        super().__init__()
        self.cv1 = Conv(c, c, 3, 1)
        self.dw = Conv(c, c, 3, 1, g=c)
        self.pw = Conv(c, c, 1, 1)
        self.rep = RepVGGDW_DCB(c)
        self.cv2 = Conv(c, c, 1, 1, act=False)

    def forward(self, x):
        y = self.cv1(x)
        y = self.dw(y)
        y = self.pw(y)
        y = self.rep(y)
        y = self.cv2(y)
        return x + y


class C2fDCB(nn.Module):
    """A lightweight C2f-like wrapper that stacks DCB blocks."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1)
        self.m = nn.Sequential(*[DCB(c2) for _ in range(n)])
        self.cv2 = Conv(c2, c2, 1)

    def forward(self, x):
        return self.cv2(self.m(self.cv1(x)))


class SCDown_DCB(nn.Module):
    """
    Downsample block from user snippet.

    Note: Ultralytics already has `SCDown` but its depthwise conv uses `act=False` in this repo.
    We keep this variant to match the user's definition without overriding the built-in name.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k, s, g=c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))

