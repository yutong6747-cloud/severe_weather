import torch
import torch.nn as nn
from typing import List

class GlobalEdgeInformationTransfer(nn.Module):
    """
    约定：forward 返回 [P3, P4, P5] 三个尺度的特征列表
    channels_list: 例如 [128, 256, 512]
    """
    def __init__(self, channels_list: List[int]):
        super().__init__()
        assert isinstance(channels_list, (list, tuple)) and len(channels_list) == 3
        c3, c4, c5 = channels_list
        # 这里给出一个最简实现，你可以按需替换为真实的边缘信息转移逻辑
        self.proj3 = nn.Conv2d(c3, c3, 3, padding=1, bias=False)
        self.proj4 = nn.Conv2d(c4, c4, 3, padding=1, bias=False)
        self.proj5 = nn.Conv2d(c5, c5, 3, padding=1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """
        x: 期望是一个包含 3 个尺度特征的可迭代对象，或你也可以在此内部自行构建
        为了与 YAML 写法配合，这里默认 x 是来自 'from' 的单路张量(例如 P3)，
        如果你要融合更多路，可在 YAML 改成 [[from1, from2, from3], ...]
        下面演示一种“简化版”实现：假定 x 已经是 list/tuple(3)；若不是，你应改 forward 逻辑。
        """
        if isinstance(x, (list, tuple)) and len(x) == 3:
            p3, p4, p5 = x
        else:
            # 如果你只从一个尺度来（例如 from=2），请在这里自行构造另外两个尺度
            # 这里做一个安全兜底：把单尺度通过下采样/上采样构造成 3 个尺度
            p3 = x
            p4 = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            p5 = nn.functional.avg_pool2d(p4, kernel_size=2, stride=2)

        p3 = self.act(self.proj3(p3))
        p4 = self.act(self.proj4(p4))
        p5 = self.act(self.proj5(p5))
        return [p3, p4, p5]
