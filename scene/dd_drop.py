import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors



class DDDrop(nn.Module):
    def __init__(self,
                 k_neighbors=10,
                 omega_depth=0.4,  # 深度分数权重
                 omega_density=0.6,  # 密度分数权重
                 lambda_middle=0.6,  # 中层衰减因子
                 lambda_far=0.4,  # 远层衰减因子（
                 r_min=0.05,  # 最小dropout率
                 r_max=0.2,  # 最大dropout率
                 total_iterations=10000):
        super(DDDrop, self).__init__()
        self.k_neighbors = k_neighbors
        self.omega_depth = omega_depth
        self.omega_density = omega_density
        self.lambda_middle = lambda_middle
        self.lambda_far = lambda_far
        self.r_min = r_min
        self.r_max = r_max
        self.total_iterations = total_iterations

    def compute_local_scores(self, gaussian_xyz, camera_pos):
        """
        计算局部分数：深度分数（归一化）+ 密度分数（归一化）
        Args:
            gaussian_xyz: 高斯中心坐标 (N, 3)
            camera_pos: 当前相机位置 (3,)
        Returns:
            depth_score: 深度分数 (N,)
            density_score: 密度分数 (N,)
        """
        N = gaussian_xyz.shape[0]
        if N == 0:
            return torch.zeros(0, device=gaussian_xyz.device), torch.zeros(0, device=gaussian_xyz.device)

        # 深度分数：高斯到相机的欧氏距离
        depth = torch.norm(gaussian_xyz - camera_pos.unsqueeze(0), dim=1)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth_score = 1.0 - depth_norm

        # 密度分数：k近邻平均距离的倒数
        if N >= self.k_neighbors:
            xyz_np = gaussian_xyz.detach().cpu().numpy()
            nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(xyz_np)
            distances, _ = nbrs.kneighbors(xyz_np)
            avg_dist = torch.tensor(distances.mean(axis=1), device=gaussian_xyz.device)
        else:
            avg_dist = torch.ones(N, device=gaussian_xyz.device)

        density = 1.0 / (avg_dist + 1e-8)
        density_score = (density - density.min()) / (density.max() - density.min() + 1e-8)

        return depth_score, density_score

    def compute_global_dropout_prob(self, depth_score, density_score, gaussian_xyz, camera_pos):
        """
        计算全局dropout概率：局部分数 + 深度分层衰减
        Args:
            depth_score: 局部深度分数 (N,)
            density_score: 局部密度分数 (N,)
            gaussian_xyz: 高斯中心坐标 (N, 3)
            camera_pos: 当前相机位置 (3,)
        Returns:
            dropout_prob: 每个高斯的dropout概率 (N,)
        """
        N = gaussian_xyz.shape[0]
        if N == 0:
            return torch.zeros(0, device=gaussian_xyz.device)

        # 局部dropout分数 S_i = ω_depth*depth_score + ω_density*density_score
        S_i = self.omega_depth * depth_score + self.omega_density * density_score

        # 全局深度分层：按深度分布的1/3和2/3分位数划分近、中、远层
        depth = torch.norm(gaussian_xyz - camera_pos.unsqueeze(0), dim=1)
        D_near = torch.quantile(depth, 1 / 3)
        D_middle = torch.quantile(depth, 2 / 3)

        # 应用分层衰减因子
        dropout_prob = torch.zeros_like(S_i)
        dropout_prob[depth <= D_near] = S_i[depth <= D_near]  # 近层：无衰减
        dropout_prob[(depth > D_near) & (depth <= D_middle)] = S_i[(depth > D_near) & (
                    depth <= D_middle)] * self.lambda_middle  # 中层：衰减
        dropout_prob[depth > D_middle] = S_i[depth > D_middle] * self.lambda_far  # 远层：强衰减

        return dropout_prob.clamp(0.0, 1.0)

    def dynamic_dropout_rate(self, current_iter):
        t = min(current_iter, self.total_iterations)
        r_t = self.r_min + (self.r_max - self.r_min) * (t / self.total_iterations)
        return r_t

    def forward(self, gaussian_xyz, camera_pos, current_iter):
        """
        前向传播：生成dropout掩码
        Args:
            gaussian_xyz: 高斯中心坐标 (N, 3)
            camera_pos: 当前相机位置 (3,)
            current_iter: 当前训练迭代数
        Returns:
            keep_mask: 保留高斯的掩码（True=保留，False=dropout）(N,)
        """
        N = gaussian_xyz.shape[0]
        if N == 0:
            return torch.ones_like(gaussian_xyz[:, 0], dtype=torch.bool)

        # 1. 计算局部分数（深度+密度）
        depth_score, density_score = self.compute_local_scores(gaussian_xyz, camera_pos)

        # 2. 计算全局dropout概率（分层衰减）
        dropout_prob = self.compute_global_dropout_prob(depth_score, density_score, gaussian_xyz, camera_pos)

        # 3. 动态调整dropout率（随迭代线性增加）
        r_t = self.dynamic_dropout_rate(current_iter)

        # 4. 生成dropout掩码（基于概率采样）
        random_mask = torch.rand_like(dropout_prob)  # 生成0~1随机数
        dropout_mask = random_mask < (dropout_prob * r_t)  # True表示要dropout的高斯
        keep_mask = ~dropout_mask  # True表示保留的高斯

        return keep_mask

