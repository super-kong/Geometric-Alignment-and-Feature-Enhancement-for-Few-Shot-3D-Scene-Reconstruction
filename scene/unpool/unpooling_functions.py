import torch
import torch.nn.functional as F
from typing import Optional



def max_unpool1d_custom(
        input: torch.Tensor,
        indices: torch.Tensor,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        output_size: Optional[int] = None
) -> torch.Tensor:
    """
    自定义1D最大反池化（适配高斯点的1D特征向量）
    Args:
        input: 池化后的特征张量，形状 [N, C]（N=高斯点数，C=特征维度）
        indices: 池化时记录的最大值索引，形状与池化输出一致
        kernel_size/stride/padding: 与池化层一致
        output_size: 反池化目标特征维度（如池化后C=32，反池化恢复到C=64）
    Returns:
        反池化后的特征 [N, output_size]
    """
    # 1D特征需reshape为[N, 1, C]以适配PyTorch的1D反池化API
    input_reshaped = input.unsqueeze(1)  # [N, 1, C]

    # 处理output_size（需为tuple）
    output_size_tuple = (output_size,) if output_size is not None else None

    # 调用原生1D最大反池化
    unpooled = F.max_unpool1d(
        input=input_reshaped,
        indices=indices,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_size=output_size_tuple
    )

    # 恢复形状 [N, output_size]
    return unpooled.squeeze(1)


def avg_unpool1d_custom(
        input: torch.Tensor,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        output_size: Optional[int] = None
) -> torch.Tensor:
    """
    自定义1D平均反池化（适配高斯点的1D特征向量）
    注：无indices参数，通过转置1D卷积实现
    """
    # 1D特征reshape为[N, 1, C]
    input_reshaped = input.unsqueeze(1)  # [N, 1, C]
    in_channels = 1  # 临时通道维度

    # 计算输出尺寸（若未指定）
    if output_size is None:
        _, _, c = input_reshaped.shape
        output_size = (c - 1) * stride - 2 * padding + kernel_size
    output_size_tuple = (output_size,)

    # 转置1D卷积实现平均反池化（核值平均）
    unpooled = F.conv_transpose1d(
        input=input_reshaped,
        weight=torch.ones(in_channels, in_channels, kernel_size, device=input.device, dtype=input.dtype) / kernel_size,
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=0,
        groups=in_channels
    )

    # 裁剪到目标尺寸+恢复形状
    unpooled = unpooled[:, :, :output_size].squeeze(1)  # [N, output_size]
    return unpooled


