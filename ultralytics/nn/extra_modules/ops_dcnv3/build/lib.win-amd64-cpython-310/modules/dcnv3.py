import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import DCNv3Function
from ....modules.conv import Conv

def _is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0

class DCNv3(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        center_feature_scale=False,
        remove_center=False,
    ):
        """
        DCNv3 Module
        """
        super().__init__()
        self.stride = (stride, stride)
        self.padding = (pad, pad)
        self.dilation = (dilation, dilation)
        self.kernel_size = (kernel_size, kernel_size)

        self.groups = 1
        self.deformable_groups = 1
        self.mask_groups = 1
        self.im2col_step = 1.0

        self.offset_scale = 1.0  # 或设置为可配置参数
        self.remove_center = remove_center



        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, got {channels} and {group}')
        _d_per_group = channels // group
        if not _is_power_of_2(_d_per_group):
            warnings.warn("You'd better set channels to make each group dimension a power of 2 for efficiency.")

        if remove_center and kernel_size % 2 == 0:
            raise ValueError('remove_center only works with odd kernel size.')


        self.groups = 1
        self.deformable_groups = 1
        self.mask_groups = 1
        self.im2col_step = 1.0

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.group = group
        self.group_channels = channels // group
        self.remove_center = int(remove_center)

        self.dw_conv = Conv(channels, channels, kernel_size, g=channels)
        self.offset = nn.Linear(channels, group * (kernel_size * kernel_size - self.remove_center) * 2)
        self.mask = nn.Linear(channels, group * (kernel_size * kernel_size - self.remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, x):
        N, H, W, _ = x.shape
        x_proj = self.input_proj(x)

        x1 = x.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1).permute(0, 2, 3, 1)

        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, dim=-1).reshape(N, H, W, -1)

        out = DCNv3Function.apply(
            x_proj.float(),  # ✅ 强制转为 float32
            offset.float(),  # ✅ 强制转为 float32
            mask.float(),  # ✅ 强制转为 float32
            None, None,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels, self.offset_scale,
            self.deformable_groups, self.mask_groups,
            self.kernel_size, self.kernel_size,
            self.remove_center
        )

        out = self.output_proj(out)
        return out


class DCNv3_DyHead(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        remove_center=False,
    ):
        """
        DCNv3 DyHead Module
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, got {channels} and {group}')
        _d_per_group = channels // group
        if not _is_power_of_2(_d_per_group):
            warnings.warn("You'd better set channels to make each group dimension a power of 2 for efficiency.")

        if remove_center and kernel_size % 2 == 0:
            raise ValueError('remove_center only works with odd kernel size.')

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.group = group
        self.group_channels = channels // group
        self.remove_center = int(remove_center)

        self.output_proj = nn.Linear(channels, channels)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, x, offset, mask):
        """
        :param x: (N, C, H, W)
        :param offset: (N, H, W, ?)
        :param mask: (N, H, W, ?)
        :return: (N, C, H, W)
        """
        N, C, H, W = x.shape

        x_in = x.permute(0, 2, 3, 1).contiguous()

        out = DCNv3Function.apply(
            x_in, offset, mask,
            None, None,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, 1, 1,
            self.kernel_size, self.kernel_size,
            self.remove_center
        )
        out = self.output_proj(out).permute(0, 3, 1, 2).contiguous()
        return out
