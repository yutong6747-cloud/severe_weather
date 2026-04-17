import torch
import torch.nn as nn

from .attention import CoordAtt


class _ELFirePartialConv3(nn.Module):
    """Partial 3x3 conv used by EL-Fire FasterBlock (local to avoid name conflicts)."""

    def __init__(self, dim: int, n_div: int = 4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), dim=1)


class FasterBlock(nn.Module):
    """FasterNet-style block with Coordinate Attention."""

    def __init__(self, inc: int, dim: int, n_div: int = 4):
        super().__init__()
        self.pw_expand = nn.Conv2d(inc, dim, 1, 1, 0)
        self.pconv = _ELFirePartialConv3(dim, n_div=n_div)
        self.act = nn.GELU()
        self.pw_reduce = nn.Conv2d(dim, inc, 1, 1, 0)
        self.ca = CoordAtt(inc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.pw_expand(x)
        x = self.pconv(x)
        x = self.act(x)
        x = self.pw_reduce(x)
        x = self.ca(x)
        return x + shortcut


class LCACSP(nn.Module):
    """Lightweight Coordinate Attention CSP (EL-Fire)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        if c1 != c2:
            raise ValueError(f"LCACSP requires c1 == c2, got c1={c1}, c2={c2}")
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d(2 * self.c + n * self.c, c2, 1)
        self.m = nn.ModuleList(FasterBlock(self.c, self.c) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for module in self.m:
            y.append(module(y[-1]))
        return self.cv2(torch.cat(y, 1))


class DualConv(nn.Module):
    """
    DualConv (EL-Fire): 1/G output channels use (3x3 + 1x1), remaining use 1x1 only.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, groups: int = 4):
        super().__init__()
        if groups <= 0:
            raise ValueError(f"groups must be > 0, got {groups}")
        self.combined_filters = out_channels // groups
        self.pointwise_filters = out_channels - self.combined_filters

        self.combined_3x3 = nn.Conv2d(in_channels, self.combined_filters, 3, stride, 1, bias=False)
        self.combined_1x1 = nn.Conv2d(in_channels, self.combined_filters, 1, stride, 0, bias=False)
        self.pointwise_only = nn.Conv2d(in_channels, self.pointwise_filters, 1, stride, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_combined = self.combined_3x3(x) + self.combined_1x1(x)
        out_pointwise = self.pointwise_only(x)
        return torch.cat([out_combined, out_pointwise], dim=1)


class DualConv_C2f(nn.Module):
    """C2f-like module using DualConv blocks (EL-Fire)."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(DualConv(self.c, self.c, groups=4) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for module in self.m:
            y.append(module(y[-1]))
        return self.cv2(torch.cat(y, 1))


__all__ = ["LCACSP", "DualConv", "DualConv_C2f", "FasterBlock"]

