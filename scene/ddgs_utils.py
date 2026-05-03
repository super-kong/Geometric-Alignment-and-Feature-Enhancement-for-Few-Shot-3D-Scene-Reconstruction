# ddgs_utils.py
import torch
import torch.nn.functional as F
from torch import nn

# 注意：distCUDA2需根据你的项目路径导入（原项目中用于计算距离矩阵的函数）
# 若你的项目中distCUDA2在其他文件，需调整导入路径
try:
    from gaussian_renderer import distCUDA2
except ImportError:
    # 备用：若没有distCUDA2，可使用torch.cdist（性能略低，兼容CPU/GPU）
    def distCUDA2(xyz):
        return torch.cdist(xyz, xyz, p=2), None


def estimate_gaussian_density(model, k=16):
    """
    独立的高斯密度估计函数
    :param model: GaussianModel实例（包含xyz等参数）
    :param k: k近邻数量
    :return: 归一化后的局部密度张量 (N,)
    """
    xyz = model.get_xyz  # 从model实例中获取高斯位置 (N, 3)
    N = xyz.shape[0]
    if N < k:
        return torch.ones(N, device=xyz.device)  # 点数不足时返回均匀密度

    # 计算k近邻距离（排除自身）
    dist_matrix, _ = distCUDA2(xyz)
    dist_matrix = dist_matrix.masked_fill(
        torch.eye(N, device=xyz.device, dtype=torch.bool),
        float('inf')
    )
    k_dist, _ = torch.topk(dist_matrix, k=k, dim=1, largest=False)  # (N, k)

    # 密度 = 1 / (平均距离 + 1e-6)（距离越小密度越大），并归一化
    mean_k_dist = k_dist.mean(dim=1)
    density = 1.0 / (mean_k_dist + 1e-6)
    return density / density.max()  # 归一化到[0,1]


def split_high_density_gaussians(model, density, split_threshold=0.7, max_split_ratio=0.2):
    """
    独立的高斯分裂函数（高密度区域）
    :param model: GaussianModel实例
    :param density: 每个高斯的局部密度 (N,)
    :param split_threshold: 高密度阈值
    :param max_split_ratio: 最大分裂比例（避免过度分裂）
    """
    if density.numel() == 0:
        return  # 无高斯时直接返回

    # 从model实例中获取所有高斯参数
    xyz = model.get_xyz
    scaling = model.get_scaling
    rotation = model.get_rotation
    features_dc = model._features_dc
    features_rest = model._features_rest
    opacity = model._opacity

    # 筛选高密度高斯
    high_density_mask = density > split_threshold
    num_split = int(high_density_mask.sum() * max_split_ratio)
    if num_split == 0:
        return

    # 随机选择待分裂的高斯索引
    split_indices = torch.where(high_density_mask)[0]
    split_indices = split_indices[torch.randperm(len(split_indices))[:num_split]]

    # 存储新生成的高斯参数
    new_xyz = []
    new_scaling = []
    new_rotation = []
    new_features_dc = []
    new_features_rest = []
    new_opacity = []

    for idx in split_indices:
        # 提取单个高斯参数
        pos = xyz[idx]
        scale = scaling[idx]
        rot = rotation[idx]
        feat_dc = features_dc[idx]
        feat_rest = features_rest[idx]
        opac = opacity[idx]

        # 每个高斯分裂为2个，添加轻微扰动
        for _ in range(2):
            # 基于尺度的位置噪声（确保局部性）
            noise = torch.randn(3, device=xyz.device) * 0.1 * scale
            new_pos = pos + noise

            # 尺度微调 + 逆激活（model中存储的是log尺度）
            new_scale = scale * (0.8 + 0.4 * torch.rand(3, device=xyz.device))
            new_scale_log = model.scaling_inverse_activation(new_scale)

            # 旋转微调 + 归一化
            new_rot = F.normalize(rot + 0.1 * torch.randn(4, device=xyz.device))

            # 复制特征和不透明度
            new_xyz.append(new_pos)
            new_scaling.append(new_scale_log)
            new_rotation.append(new_rot)
            new_features_dc.append(feat_dc)
            new_features_rest.append(feat_rest)
            new_opacity.append(opac)

    # 合并新高斯参数到model中（直接修改model的属性）
    model._xyz = nn.Parameter(torch.cat([xyz, torch.stack(new_xyz)], dim=0).requires_grad_(True))
    model._scaling = nn.Parameter(torch.cat([model._scaling, torch.stack(new_scaling)], dim=0).requires_grad_(True))
    model._rotation = nn.Parameter(torch.cat([model._rotation, torch.stack(new_rotation)], dim=0).requires_grad_(True))
    model._features_dc = nn.Parameter(
        torch.cat([features_dc, torch.stack(new_features_dc)], dim=0).requires_grad_(True))
    model._features_rest = nn.Parameter(
        torch.cat([features_rest, torch.stack(new_features_rest)], dim=0).requires_grad_(True))
    model._opacity = nn.Parameter(torch.cat([opacity, torch.stack(new_opacity)], dim=0).requires_grad_(True))

    # 更新model的辅助参数
    model.max_radii2D = torch.cat([model.max_radii2D, torch.zeros(len(new_xyz), device=xyz.device)], dim=0)
    model.confidence = torch.cat([model.confidence, torch.ones(len(new_xyz), device=xyz.device)], dim=0)


def prune_low_density_gaussians(model, density, prune_threshold=0.1, min_opacity=0.01):
    """
    独立的高斯修剪函数（低密度区域）
    :param model: GaussianModel实例
    :param density: 每个高斯的局部密度 (N,)
    :param prune_threshold: 低密度阈值
    :param min_opacity: 最小不透明度阈值（保留高可见性高斯）
    """
    if density.numel() == 0:
        return

    # 从model实例中获取参数
    opacity = model.get_opacity.squeeze()  # (N,)
    # 计算保留掩码：密度达标 或 不透明度达标
    keep_mask = (density > prune_threshold) | (opacity > min_opacity)

    # 过滤model的所有参数
    model._xyz = nn.Parameter(model._xyz[keep_mask].requires_grad_(True))
    model._scaling = nn.Parameter(model._scaling[keep_mask].requires_grad_(True))
    model._rotation = nn.Parameter(model._rotation[keep_mask].requires_grad_(True))
    model._features_dc = nn.Parameter(model._features_dc[keep_mask].requires_grad_(True))
    model._features_rest = nn.Parameter(model._features_rest[keep_mask].requires_grad_(True))
    model._opacity = nn.Parameter(model._opacity[keep_mask].requires_grad_(True))

    # 更新辅助参数
    model.max_radii2D = model.max_radii2D[keep_mask]
    model.confidence = model.confidence[keep_mask]
