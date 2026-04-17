import math

import torch
import torch.nn as nn

from .block import DFL
from .conv import Conv

__all__ = (
    "ChannelAttention",
    "SobelConv",
    "FGM",
    "OmniKernel",
    "EIEStem",
    "SPDConv",
    "ECABlock",
    "CSPOmniKernel",
    "Scale",
    "Conv_GN",
    "DEConv_GN",
    "Detect_LSDECD",
)


class ChannelAttention(nn.Module):
    """Channel attention module similar to SE."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class SobelConv(nn.Module):
    """Learnable Sobel-like edge extraction."""

    def __init__(self, c: int):
        super().__init__()
        self.sobel_x = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=True)
        self.sobel_y = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=True)
        self.alpha = nn.Parameter(torch.ones(1, c, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c, 1, 1))
        self._init_sobel()

    def _init_sobel(self):
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        for conv, kernel in ((self.sobel_x, kx), (self.sobel_y, ky)):
            weight = torch.zeros_like(conv.weight.data)
            for i in range(weight.shape[0]):
                weight[i, 0] = kernel
            conv.weight.data.copy_(weight)

    def forward(self, x):
        gx = self.sobel_x(x)
        gy = self.sobel_y(x)
        return self.alpha * gx + self.beta * gy


class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x2_fft = torch.fft.fft2(x2, norm="backward")
        out = x1 * x2_fft
        out = torch.fft.ifft2(out, dim=(-2, -1), norm="backward")
        out = torch.abs(out)
        return out * self.alpha + x * self.beta


class OmniKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm="backward")
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm="backward")
        x_fca = torch.abs(x_fca)
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        x_sca = self.fgm(x_sca)
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)


class EIEStem(nn.Module):
    """Lightweight EIEStem with attention and residual shortcut."""

    def __init__(self, inc: int, hidc: int, ouc: int) -> None:
        super().__init__()
        self.conv1 = Conv(inc, hidc, 3, 2)
        self.sobel_branch = SobelConv(hidc)
        self.pool_branch = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True),
        )
        self.ca = ChannelAttention(hidc * 2)
        self.conv2 = Conv(hidc * 2, hidc, 3, 2)
        self.conv3 = Conv(hidc, ouc, 1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(hidc, ouc, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(ouc),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        shortcut = self.shortcut(x)
        x_sobel = self.sobel_branch(x)
        x_pool = self.pool_branch(x)
        x = torch.cat([x_sobel, x_pool], dim=1)
        x = self.ca(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + shortcut


class SPDConv(nn.Module):
    """SPDConv with optional high-frequency edge enhancement."""

    def __init__(self, inc, ouc, dimension=1, use_highfreq=True):
        super().__init__()
        self.d = dimension
        self.use_highfreq = use_highfreq

        if self.use_highfreq:
            self.lap = nn.Conv2d(inc, inc, kernel_size=3, padding=1, groups=inc, bias=False)
            lap_kernel = torch.tensor(
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
            )
            with torch.no_grad():
                weight = lap_kernel.view(1, 1, 3, 3).repeat(inc, 1, 1, 1)
                self.lap.weight.copy_(weight)
            self.hf_pool = nn.AvgPool2d(kernel_size=2, stride=2)
            in_channels_conv = inc * 4 + inc
        else:
            self.lap = None
            self.hf_pool = None
            in_channels_conv = inc * 4

        self.conv = Conv(in_channels_conv, ouc, k=3)

    def forward(self, x):
        spd = torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2],
            ],
            dim=1,
        )

        if self.use_highfreq:
            hf = self.lap(x)
            hf = self.hf_pool(hf)
            out = torch.cat([spd, hf], 1)
        else:
            out = spd

        return self.conv(out)


class ECABlock(nn.Module):
    """Lightweight channel attention (ECA)."""

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y


class CSPOmniKernel(nn.Module):
    """CSP-style OmniKernel block with optional ECA attention."""

    def __init__(self, dim, e=0.25, use_eca=True):
        super().__init__()
        self.e = e
        self.ok_channels = max(int(dim * self.e), 1)
        self.id_channels = dim - self.ok_channels
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(self.ok_channels)
        self.use_eca = use_eca
        self.attn = ECABlock(dim) if self.use_eca else None

    def forward(self, x):
        feat = self.cv1(x)
        ok_branch, identity = torch.split(feat, [self.ok_channels, self.id_channels], dim=1)
        out = torch.cat((self.m(ok_branch), identity), 1)
        out = self.cv2(out)
        if self.attn is not None:
            out = self.attn(out)
        return out


class Scale(nn.Module):
    """A learnable scale parameter."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class Conv_GN(nn.Module):
    """Conv + GroupNorm + activation."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.gn = nn.GroupNorm(16 if c2 % 16 == 0 else 1, c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class DEConv_GN(nn.Module):
    """Detail-enhanced convolution with GroupNorm."""

    def __init__(self, channels, kernel_size=3, stride=1, num_groups=16):
        super().__init__()
        padding = kernel_size // 2
        output_padding = 0 if stride == 1 else stride - 1
        self.deconv = nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )
        self.gn = nn.GroupNorm(num_groups if channels % num_groups == 0 else 1, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.deconv(x)
        x = self.gn(x)
        x = self.act(x)
        return x


class Detect_LSDECD(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, hidc=256, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        self.conv = nn.ModuleList(nn.Sequential(Conv_GN(x, hidc, 1)) for x in ch)
        self.share_conv = nn.Sequential(DEConv_GN(hidc), DEConv_GN(hidc))
        self.cv2 = nn.Conv2d(hidc // 2, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc // 2, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for _ in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            xi = self.conv[i](x[i])
            detail = self.share_conv(xi)
            xi = xi + detail
            c = xi.shape[1] // 2
            xi_reg = xi[:, :c]
            xi_cls = xi[:, c:]
            box = self.scale[i](self.cv2(xi_reg))
            cls = self.cv3(xi_cls)
            x[i] = torch.cat((box, cls), 1)

        if self.training:
            return x

        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.dynamic or self.shape != shape:
            from ultralytics.utils.tal import make_anchors

            self.anchors, self.strides = (t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        dbox = self.decode_bboxes(box)
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        m = self
        m.cv2.bias.data[:] = 1.0
        for s in self.stride:
            prior = 5 / m.nc / (640 / s) ** 2
            m.cv3.bias.data[: m.nc] = math.log(prior)

    def decode_bboxes(self, bboxes):
        from ultralytics.utils.tal import dist2bbox

        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides



# The first improvement module is an optimized design based on EIEStem, incorporating Channel Attention, SobelConv, and residual connections.
# The second improvement module consists of SPDConv, OmniKernel, FGM, and CSPOmniKernel.
# The third improvement module is designed based on the concepts of shared convolution, detail-enhanced deconvolution, decoupled classification and regression, and the DFL-based bounding box regression mechanism.
