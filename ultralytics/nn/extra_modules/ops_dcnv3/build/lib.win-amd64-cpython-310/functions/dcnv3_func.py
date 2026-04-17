import torch
from torch.autograd import Function

# from ultralytics.nn.extra_modules.ops_dcnv3.functions import dcnv3_ext  # 如果你有 C++ 插件扩展
import ultralytics.nn.extra_modules.ops_dcnv3.dcnv3_ext as dcnv3_ext

class DCNv3Function(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias,
                stride_h, stride_w, pad_h, pad_w,
                dilation_h, dilation_w, groups,
                group_channels, offset_scale,  # ✅ 添加这两个
                deformable_groups, mask_groups,
                kernel_h, kernel_w):
        ctx.save_for_backward(input, offset, mask, weight, bias)
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.mask_groups = mask_groups
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale

        output = dcnv3_ext.dcnv3_forward(
            input, offset, mask,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            groups, group_channels, offset_scale,
            deformable_groups, mask_groups
        )

        return output



    @staticmethod
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors

        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = dcnv3_ext.dcnv3_backward(
            grad_output,
            input, offset, mask, weight, bias,
            ctx.stride_h, ctx.stride_w, ctx.pad_h, ctx.pad_w,
            ctx.dilation_h, ctx.dilation_w, ctx.groups,
            ctx.deformable_groups, ctx.mask_groups,
            ctx.kernel_h, ctx.kernel_w, ctx.remove_center
        )

        # 返回值个数必须和 forward 输入参数个数一致
        # 对于非 Tensor 参数返回 None
        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None,
                None, None, None,
                None, None, None, None)