import torch
import torch.nn as nn

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = min(factor, channels)
        assert channels % self.groups == 0, f"channels={channels}不能整除 groups={self.groups}"

        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.gn = nn.GroupNorm(self.groups, channels)
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        print(f"[DEBUG] Running updated EMA forward. Input shape: {x.shape}")
        # 注意：这里我们不再对池化特征做cat送入conv1x1
        y = self.agp(x).view(x.size(0), x.size(1), 1)
        y_h = self.pool_h(x).view(x.size(0), x.size(1), -1)
        y_w = self.pool_w(x).view(x.size(0), x.size(1), -1)
        y = torch.cat([y, y_h, y_w], dim=-1)
        y = self.softmax(y)

        out = self.conv1x1(x)
        out = self.conv3x3(out)
        return out

