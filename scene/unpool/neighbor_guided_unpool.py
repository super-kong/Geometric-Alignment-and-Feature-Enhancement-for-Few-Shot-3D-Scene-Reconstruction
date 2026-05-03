# scene/unpool/neighbor_guided_unpool.py
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid


def safe_distCUDA2(points):
    """安全调用distCUDA2，处理返回值异常"""
    # 前置检查：至少2个点，且是二维张量 (N,3)
    if points.dim() != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        return None, None

    try:
        # 调用distCUDA2，兼容不同返回格式
        result = distCUDA2(points)
        if isinstance(result, tuple):
            # 标准返回：(distances, indices)，distances.shape=(N, k_default)
            dists, indices = result if len(result) >= 2 else (result[0], None)
        else:
            # 非标准返回：仅距离矩阵
            dists = result
            indices = None

        # 确保距离矩阵是二维
        if dists.dim() == 1:
            dists = dists.unsqueeze(1)
        return dists, indices
    except Exception as e:
        print(f"[safe_distCUDA2] 调用失败: {e}")
        return None, None


def select_unpool_candidates(gaussian_model, density_threshold=0.01, gradient_threshold=0.001):
    """选择反池化候选点（增加安全校验）"""
    xyz = gaussian_model.get_xyz.detach()  # (N, 3)
    if xyz.dim() != 2 or xyz.shape[0] < 2:
        return torch.zeros(0, dtype=torch.bool, device=xyz.device)

    # 安全调用distCUDA2
    dist2, _ = safe_distCUDA2(xyz)
    if dist2 is None:  # 调用失败则直接返回空掩码
        return torch.zeros(0, dtype=torch.bool, device=xyz.device)

    dist2 = torch.clamp_min(dist2, 1e-7)
    local_density = 1.0 / torch.sqrt(dist2).mean(dim=1)  # (N,)

    # 梯度计算（保留原有逻辑）
    if gaussian_model.xyz_gradient_accum.numel() > 0 and gaussian_model.xyz_gradient_accum.dim() == 2:
        xyz_gradient = gaussian_model.xyz_gradient_accum / (gaussian_model.denom + 1e-7)
        gradient_norm = torch.norm(xyz_gradient, dim=1)
    else:
        gradient_norm = torch.ones_like(local_density)

    opacity = gaussian_model.get_opacity.detach().squeeze(-1)
    candidate_mask = (opacity > 0.01) & (local_density > density_threshold) & (gradient_norm > gradient_threshold)
    return candidate_mask


def neighbor_guided_unpool(gaussian_model, k_neighbors=3, unpool_ratio=2, valid_mask=None, density_threshold=0.01, gradient_threshold=0.001):
    """邻域引导反池化（核心修复tuple/index越界）"""
    # 边界检查1：基础维度/数量校验
    xyz = gaussian_model.get_xyz
    if xyz.dim() != 2 or xyz.shape[0] < 2 or xyz.shape[1] != 3:
        print(f"[Unpool] 无效的高斯点形状: {xyz.shape}，跳过反池化")
        return (
            torch.empty(0, 3, device="cuda"),
            torch.empty(0, gaussian_model.get_features.shape[1] if gaussian_model.get_features.dim() == 2 else 0,
                        device="cuda"),
            torch.empty(0, 3, device="cuda"),
            torch.empty(0, 4, device="cuda"),
            torch.empty(0, 1, device="cuda"),
            torch.empty(0, 1, device="cuda")
        )

    # 边界检查2：有效掩码校验
    if valid_mask is None:
        valid_mask = select_unpool_candidates(gaussian_model, density_threshold, gradient_threshold)
    valid_mask = valid_mask.squeeze() if valid_mask.dim() > 1 else valid_mask
    if valid_mask.sum() < 2:
        print(f"[Unpool] 有效点数量不足: {valid_mask.sum()}，跳过反池化")
        return (
            torch.empty(0, 3, device="cuda"),
            torch.empty(0, gaussian_model.get_features.shape[1] if gaussian_model.get_features.dim() == 2 else 0,
                        device="cuda"),
            torch.empty(0, 3, device="cuda"),
            torch.empty(0, 4, device="cuda"),
            torch.empty(0, 1, device="cuda"),
            torch.empty(0, 1, device="cuda")
        )

    # 提取有效点
    xyz_valid = gaussian_model.get_xyz[valid_mask]  # (N_valid, 3)
    features_valid = gaussian_model.get_features[valid_mask]
    scaling_valid = gaussian_model.get_scaling[valid_mask]
    rotation_valid = gaussian_model.get_rotation[valid_mask]
    opacity_valid = gaussian_model.get_opacity[valid_mask]
    confidence_valid = gaussian_model.confidence[valid_mask]

    # 核心修复：安全计算距离矩阵（解决tuple/index越界）
    dist_matrix, _ = safe_distCUDA2(xyz_valid)
    if dist_matrix is None:
        # 降级方案：用torch.cdist计算全距离矩阵
        dist_matrix = torch.cdist(xyz_valid, xyz_valid, p=2)  # (N_valid, N_valid)
    # 确保dist_matrix是二维
    if dist_matrix.dim() == 1:
        dist_matrix = dist_matrix.unsqueeze(1)
    # 排除自身点（对角线设为无穷大）
    if dist_matrix.shape[0] == dist_matrix.shape[1]:
        dist_matrix.fill_diagonal_(float('inf'))

    # 修复：动态适配k_neighbors（避免shape[1]越界）
    max_available_k = dist_matrix.shape[1] if dist_matrix.dim() >= 2 else 0
    k = min(k_neighbors, max_available_k) if max_available_k > 0 else 1  # 兜底k=1
    if k < 1:
        print(f"[Unpool] 无可用邻域数，跳过反池化")
        return (
            torch.empty(0, 3, device="cuda"),
            torch.empty(0, features_valid.shape[1] if features_valid.dim() == 2 else 0, device="cuda"),
            torch.empty(0, 3, device="cuda"),
            torch.empty(0, 4, device="cuda"),
            torch.empty(0, 1, device="cuda"),
            torch.empty(0, 1, device="cuda")
        )

    # 正常计算邻域索引
    _, nn_indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)

    # 生成新点（保留原有逻辑，仅增加维度校验）
    N_valid = xyz_valid.shape[0]
    new_xyz, new_features, new_scaling, new_rotation, new_opacity, new_confidence = [], [], [], [], [], []

    for i in range(N_valid):
        if nn_indices[i].numel() == 0:
            continue
        nn_xyz = xyz_valid[nn_indices[i]]
        nn_center = nn_xyz.mean(dim=0)
        nn_std = nn_xyz.std(dim=0) + 1e-6

        for _ in range(unpool_ratio):
            noise = torch.randn(3, device="cuda") * nn_std * 0.1
            new_xyz_i = xyz_valid[i] + noise
            new_xyz.append(new_xyz_i)

            # 特征维度兼容
            feat_i = features_valid[i] if features_valid[i].dim() == 1 else features_valid[i].squeeze()
            nn_feat = features_valid[nn_indices[i]].mean(dim=0) if features_valid[nn_indices[i]].dim() == 2 else \
            features_valid[nn_indices[i]].mean(dim=0).squeeze()
            new_feat_i = (feat_i + nn_feat) / 2
            new_features.append(new_feat_i)

            new_scale_i = (scaling_valid[i] + scaling_valid[nn_indices[i]].mean(dim=0)) / 2
            new_scaling.append(new_scale_i)

            new_rot_i = rotation_valid[i]
            new_rotation.append(new_rot_i)

            new_opacity_i = opacity_valid[i]
            new_opacity.append(new_opacity_i)

            new_conf_i = confidence_valid[i] * 0.9
            new_confidence.append(new_conf_i)

    # 转换为张量（处理空列表）
    new_xyz = torch.stack(new_xyz) if new_xyz else torch.empty(0, 3, device="cuda")
    new_features = torch.stack(new_features) if new_features else torch.empty(0, features_valid.shape[
        1] if features_valid.dim() == 2 else 0, device="cuda")
    new_features = new_features.reshape(-1, 1, 3)  # 剔除16维中多余的13维
    new_features = new_features.squeeze(dim=-1) if new_features.dim() > 3 else new_features

    new_scaling = torch.stack(new_scaling) if new_scaling else torch.empty(0, 3, device="cuda")
    new_rotation = torch.stack(new_rotation) if new_rotation else torch.empty(0, 4, device="cuda")
    new_opacity = torch.stack(new_opacity) if new_opacity else torch.empty(0, 1, device="cuda")
    new_confidence = torch.stack(new_confidence) if new_confidence else torch.empty(0, 1, device="cuda")

    return new_xyz, new_features, new_scaling, new_rotation, new_opacity, new_confidence


def generate_neighbor_guided_gaussians(self, candidate_xyz, candidate_mask, k=8):
    """
    基于近邻引导生成新高斯点（作为GaussianModel的方法）
    Args:
        candidate_xyz: 候选点位置 (M, 3)
        candidate_mask: 候选点掩码 (N,)
        k: 近邻数
    Returns:
        new_gaussians: 字典，包含新高斯的xyz/features/scaling/rotation/opacity/confidence
    """
    full_xyz = self.get_xyz.detach()  # 新增：剥离梯度
    num_candidates = candidate_xyz.shape[0]
    device = full_xyz.device

    # 1. 为每个候选点检索k近邻
    # 修复：full_xyz 和 candidate_xyz 都先detach再转numpy
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(full_xyz.cpu().numpy())
    _, neighbor_indices = nbrs.kneighbors(candidate_xyz.detach().cpu().numpy())  # 核心修复：detach()
    neighbor_indices = torch.tensor(neighbor_indices, device=device)  # (M, k)
    neighbor_xyz = full_xyz[neighbor_indices]  # (M, k, 3)

    # 2. 生成新位置：近邻插值 + 随机偏移（保持局部分布）
    neighbor_dist = torch.norm(neighbor_xyz - candidate_xyz.unsqueeze(1), dim=2) + 1e-6
    neighbor_weight = 1.0 / neighbor_dist
    neighbor_weight = neighbor_weight / neighbor_weight.sum(dim=1, keepdim=True)  # (M, k)
    interpolated_xyz = (neighbor_xyz * neighbor_weight.unsqueeze(2)).sum(dim=1)  # (M, 3)

    # 加入小偏移（避免与原有点重合）
    offset = torch.randn_like(interpolated_xyz) * 0.01 * self.spatial_lr_scale
    new_xyz = interpolated_xyz + offset

    # 3. 初始化新高斯的特征（融合近邻SH特征）
    full_features_dc = self._features_dc.detach()  # (N, 1, 3)
    full_features_rest = self._features_rest.detach()  # (N, sh_dim-1, 3)
    neighbor_feat_dc = full_features_dc[neighbor_indices]  # (M, k, 1, 3)
    neighbor_feat_rest = full_features_rest[neighbor_indices]  # (M, k, sh_dim-1, 3)

    # 加权融合近邻特征
    new_feat_dc = (neighbor_feat_dc * neighbor_weight.unsqueeze(2).unsqueeze(3)).sum(dim=1)  # (M, 1, 3)
    new_feat_rest = (neighbor_feat_rest * neighbor_weight.unsqueeze(2).unsqueeze(3)).sum(dim=1)  # (M, sh_dim-1, 3)

    # 4. 初始化缩放（适配近邻几何分布）
    full_scaling = self.get_scaling.detach()  # (N, 3)
    neighbor_scaling = full_scaling[neighbor_indices]  # (M, k, 3)
    new_scaling = self.scaling_inverse_activation(
        (neighbor_scaling * neighbor_weight.unsqueeze(2)).sum(dim=1)
    )

    # 5. 初始化旋转（近邻旋转的平均）
    full_rotation = self._rotation.detach()  # (N, 4) 四元数
    neighbor_rotation = full_rotation[neighbor_indices]  # (M, k, 4)
    new_rotation = F.normalize((neighbor_rotation * neighbor_weight.unsqueeze(2)).sum(dim=1), dim=1)

    # 6. 初始化不透明度（继承近邻均值）
    full_opacity = self.get_opacity.detach()  # (N, 1)
    neighbor_opacity = full_opacity[neighbor_indices]  # (M, k, 1)
    new_opacity = inverse_sigmoid(
        (neighbor_opacity * neighbor_weight.unsqueeze(2)).sum(dim=1)
    )

    # 7. 初始化置信度（与原候选点一致，适配原有二维张量）
    full_confidence = self.confidence.detach()
    new_confidence = full_confidence[candidate_mask].clone()

    return {
        'name': 'xyz',
        "xyz": new_xyz,
        "features_dc": new_feat_dc,
        "features_rest": new_feat_rest,
        "scaling": new_scaling,
        "rotation": new_rotation,
        "opacity": new_opacity,
        "confidence": new_confidence
    }

