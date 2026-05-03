#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, build_rotation
from torch import nn

import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, chamfer_dist
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from .litept_wrapper import LitePTFeatureExtractor
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster
import math
from skimage.restoration import denoise_bilateral
from sklearn.neighbors import LocalOutlierFactor
from utils.SEBlock import SEBlock, SEBlock_ECA, SEBlock_ImprovedPool, SEBlock_Comprehensive, SEBlock_WeightedPool, SEBlock_HierarchicalPool, SEBlock_DynamicECA, SEBlock_CrossChannel, SEBlock_ImprovedPool_Light
from scene.unpool.unpooling_functions import max_unpool1d_custom, avg_unpool1d_custom
from utils.loss_utils import l1_loss
from torchmetrics.functional.regression import pearson_corrcoef

import warnings
warnings.filterwarnings('ignore')


class GatedFusionNet(nn.Module):
    def __init__(self, sh_dim, litept_dim):
        super().__init__()
        # 输入拼接了 SH特征(取均值) 和 LitePT特征
        input_dim = sh_dim + litept_dim
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1), # 输出一个标量权重
            nn.Sigmoid()      # 归一化到 [0, 1]
        )

    def forward(self, sh_feats, litept_feats):
        # sh_feats: [N, 3, 15] -> 取平均变成 [N, 15] 简化计算
        sh_summary = sh_feats.mean(dim=1)
        combined = torch.cat([sh_summary, litept_feats], dim=1)
        return self.gate(combined)



class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.args = args
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree
        self.init_point = torch.empty(0)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.bg_color = torch.empty(0)
        self.confidence = torch.empty(0) # origin
        # 修复：初始化为二维空张量 [0,1]
        # self.confidence = torch.empty((0, 1), device="cuda")
        self.use_litept = args.use_litept
        self.use_cosine = args.use_cosine # 使用余弦退火
        # self.attn = nn.Linear(64, 64).cuda()  # 为了使用SE模块
        # nn.init.xavier_uniform_(self.attn.weight)
        self.feat_projector = None
        self.use_debug = args.use_debug
        self.use_gaussian_init = args.use_gaussian_init
        self.use_OutlierFactor = args.use_OutlierFactor
        self.use_Dweight =  args.use_Dweight # 动态调整权重
        self.use_multi_frature = args.use_multi_frature
        self.use_SEBlock = args.use_SEBlock
        self.use_dd_drop = args.use_dd_drop
        self.use_multi_scale_depth = args.use_multi_scale_depth
        # self.dd_drop = DDDrop(
        #     k_neighbors=getattr(args, "dd_k", 6),
        #     omega_depth=getattr(args, "dd_omega_depth", 0.5),
        #     omega_density=getattr(args, "dd_omega_density", 0.5),
        #     lambda_middle=getattr(args, "dd_lambda_middle", 0.5),
        #     lambda_far=getattr(args, "dd_lambda_far", 0.1),
        #     r_min=getattr(args, "dd_r_min", 0.05),
        #     r_max=getattr(args, "dd_r_max", 0.3),
        #     total_iterations=10000
        # ).to("cuda")
        self.use_dafe = args.use_dafe

        self.se_switch_config = {
            'base': "base",
            'improved': "improved",
            'eca': "eca",
            'comprehensive': "comprehensive",
            'WeightedPool': "WeightedPool",
            'HierarchicalPool': "HierarchicalPool",
            'DynamicECA': "DynamicECA",
            'ImprovedPool_Light': "ImprovedPool_Light",
            'CrossChannel': "CrossChannel"
        }
        self.current_se_type = args.se_type  # 记录当前使用的SEBlock类型
        self.use_unpool = args.use_unpool # 高斯反池化
        self.use_fusionnet = args.use_fusionnet
        self.fusion_gate = None

        # 用在深度一致性缩放约束
        self.use_scale_constraint = args.use_scale_constraint
        self.init_dist = None
        self.cameras_extent = None

        # SEBlock
    def _create_se_block(self, feat_dim, device, se_type):
        """
        仅根据指定类型创建SEBlock（无任何判断条件）
        Args:
            feat_dim: 特征维度（如LitePT的64维）
            device: 设备（cuda/cpu）
            se_type: 指定的SEBlock类型（base/improved_pool/eca/comprehensive）
        Returns:
            对应类型的SEBlock实例
        """
        if se_type == "base":
            return SEBlock(feat_dim=feat_dim, reduction=4).to(device)
        elif se_type == "improved":
            return SEBlock_ImprovedPool(feat_dim=feat_dim, reduction=4).to(device)
        elif se_type == "eca":
            return SEBlock_ECA(feat_dim=feat_dim, gamma=2, b=1).to(device)
        elif se_type == "comprehensive":
            return SEBlock_Comprehensive(feat_dim=feat_dim, min_mid_dim=8, max_reduction=16).to(device)
        elif se_type == "WeightedPool":
            return SEBlock_WeightedPool(feat_dim=feat_dim, reduction=4).to(device)
        elif se_type == "HierarchicalPool":
            return  SEBlock_HierarchicalPool(feat_dim=feat_dim, reduction=4).to(device)
        elif se_type == "DynamicECA":
            return SEBlock_DynamicECA(feat_dim=feat_dim, gamma=2, b=1).to(device)
        elif se_type == "ImprovedPool_Light":
            return SEBlock_ImprovedPool_Light(feat_dim=feat_dim, reduction=4).to(device)
        elif se_type == "CrossChannel":
            return SEBlock_CrossChannel(feat_dim=feat_dim, reduction=4).to(device)
        else:
            raise ValueError(f"不支持的SEBlock类型: {se_type}，仅支持base/improved_pool/eca/comprehensive")

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.confidence = torch.ones_like(self._opacity, device="cuda") # NOTICE

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        w = self.rotation_activation(self._rotation)
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
    #     self.spatial_lr_scale = spatial_lr_scale
    #     fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
    #     fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    #     if self.use_litept:
    #         print("*"*50)
    #         print("Using Gaussian Model With LitePT")
    #         # 使用LitePT提取特征并归一化（解决方差警告）
    #         litept_extractor = LitePTFeatureExtractor()
    #
    #         # 传入颜色信息
    #         litept_feats = litept_extractor.extract(np.asarray(pcd.points), pcd_colors=np.asarray(pcd.colors))
    #
    #         # # 不传入颜色信息
    #         # litept_feats = litept_extractor.extract(np.asarray(pcd.points))  # (N, C)
    #
    #         # # 改进3： 特征 PCA 降维
    #         # pca = PCA(n_components=32)
    #         # litept_feats = pca.fit_transform(litept_feats)
    #
    #         # 改进4： 注意力机制
    #         litept_feats = torch.tensor(litept_feats).cuda().float()
    #         attn = nn.Linear(64, 64).cuda()
    #         attn_weight = F.softmax(attn(litept_feats), dim=1)
    #         litept_feats = litept_feats * attn_weight
    #
    #         # 特征归一化
    #         litept_feats = F.normalize(litept_feats, dim=-1, p=2)  # L2归一化到单位球
    #         litept_feats = (litept_feats - litept_feats.mean()) / (litept_feats.std() + 1e-6)  # 标准化
    #
    #         # 将LitePT特征映射到原始SH特征维度
    #         # 原始特征结构：3通道（RGB） × SH维度（(max_sh_degree+1)²）
    #         sh_dim = (self.max_sh_degree + 1) ** 2  # 如max_sh_degree=3时，sh_dim=16
    #         num_gaussians = fused_point_cloud.shape[0]
    #
    #         # 初始化原始特征容器
    #         features = torch.zeros((num_gaussians, 3, sh_dim), device="cuda", dtype=torch.float)
    #         if self.args.use_color:
    #             features[:, :3, 0] = fused_color  # 保留原始颜色SH特征（DC分量）
    #
    #         # 构建特征映射层：将LitePT特征（C维）映射到3通道×(sh_dim-1)维度（AC分量）
    #         # 避免直接拼接，而是用线性层适配维度
    #         feat_projector = nn.Linear(litept_feats.shape[1], 3 * (sh_dim - 1)).cuda()
    #         projected_feats = feat_projector(litept_feats)  # (N, 3*(sh_dim-1))
    #         projected_feats = projected_feats.reshape(num_gaussians, 3, sh_dim - 1)  # (N, 3, 15)
    #
    #         # # 改进方法1：针对纹理密度的不同做不同处理
    #         # # 使用颜色方差作为纹理指标
    #         # color = torch.tensor(pcd.colors).cuda().float()
    #         # color_var = color.var(dim=1)  # (N,)，方差大=高纹理，方差小=低纹理
    #         # high_texture_mask = color_var > 0.01  # 阈值可调，根据场景调整
    #         # low_texture_mask = ~high_texture_mask
    #         #
    #         # # 低纹理区融合LitePT
    #         # litept_weight = 0.1
    #         # features[low_texture_mask, :3, 1:] = features[low_texture_mask, :3, 1:] * (1 - litept_weight) + projected_feats[
    #         #     low_texture_mask] * litept_weight
    #         # # 高纹理区完全保留原生特征
    #         # features[high_texture_mask, :3, 1:] = features[high_texture_mask, :3, 1:]
    #
    #         # 改进方法2： 基于 SH 特征的聚类结果，仅在 聚类稀疏区融合 LitePT
    #         sh_feats = features[:, :, 0].detach().cpu().numpy()
    #         kmeans = KMeans(n_clusters=10).fit(sh_feats)
    #         cluster_density = np.bincount(kmeans.labels_)  # 聚类密度
    #         low_density_clusters = np.where(cluster_density < 1000)[0]  # 稀疏聚类
    #         # 2. 仅在稀疏聚类区域融合LitePT（权重0.05）
    #         mask = np.isin(kmeans.labels_, low_density_clusters)
    #         features[mask, :3, 1:] = features[mask, :3, 1:] * 0.95 + projected_feats[mask] * 0.05
    #
    #
    #         # 可视化展示
    #         # LitePT特征
    #         litept_feats_tsne = TSNE(n_components=2).fit_transform(
    #             litept_feats.detach().cpu().numpy()[:10000]
    #         )
    #         # IPSM SH特征
    #         sh_feats_tsne = TSNE(n_components=2).fit_transform(
    #             features[:, :, 0].detach().cpu().numpy()[:10000]
    #         )
    #
    #         # 绘制对比图
    #         plt.subplot(121)
    #         plt.scatter(litept_feats_tsne[:, 0], litept_feats_tsne[:, 1], s=1)
    #         plt.title("LitePT Features")
    #         plt.subplot(122)
    #         plt.scatter(sh_feats_tsne[:, 0], sh_feats_tsne[:, 1], s=1)
    #         plt.title("IPSM SH Features")
    #         plt.savefig("feature_tsne.png")
    #
    #     else:
    #         features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    #         if self.args.use_color:
    #             features[:, :3, 0] = fused_color
    #         features[:, 3:, 1:] = 0.0
    #
    #     # 后续逻辑完全保留（与IPSM原生一致，避免维度冲突）
    #     print("Number of points at initialisation : ", fused_point_cloud.shape[0])
    #     self.init_point = fused_point_cloud
    #
    #     dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
    #     scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    #     rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    #     rots[:, 0] = 1
    #
    #     opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
    #
    #     self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    #     self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._scaling = nn.Parameter(scales.requires_grad_(True))
    #     self._rotation = nn.Parameter(rots.requires_grad_(True))
    #     self._opacity = nn.Parameter(opacities.requires_grad_(True))
    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    #     self.confidence = torch.ones_like(opacities, device="cuda")
    #     if self.args.train_bg:
    #         self.bg_color = nn.Parameter((torch.zeros(3, 1, 1) + 0.).cuda().requires_grad_(True))


    # 计算局部邻域几何直方图
    def compute_local_geometric_histograms(self, points, k=32, num_bins=16):
        """
        计算每个点的局部邻域几何直方图特征
        """
        N = points.shape[0]
        device = points.device

        # 1. 计算全距离矩阵（N, N），确保形状正确
        # 若distCUDA2返回（距离矩阵, 索引矩阵），则直接使用；否则需重新计算全距离
        dist_matrix, _ = distCUDA2(points)  # 假设返回（N, N）的距离矩阵
        if dist_matrix.shape != (N, N):
            # 若distCUDA2返回的是（N, K），则重新计算全距离矩阵（备选方案）
            dist_matrix = torch.cdist(points, points, p=2)  # 计算欧氏距离，确保形状为（N, N）

        # 2. 排除自身点（对角线元素设为无穷大）
        # 生成对角线为False的掩码（排除自身）
        mask = ~torch.eye(N, dtype=torch.bool, device=device)  # 形状（N, N），对角线为False
        dist_matrix = dist_matrix.masked_fill(~mask, float('inf'))  # 自身点距离设为无穷大

        # 3. 取每个点的前k个近邻
        _, topk_indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)  # （N, k）
        nn_points = points[topk_indices]  # （N, k, 3）

        # 后续几何属性计算和直方图构建逻辑不变...
        # （以下代码与原逻辑一致，省略重复部分）
        relative_pos = nn_points - points.unsqueeze(1)  # （N, k, 3）

        distances = torch.norm(relative_pos, dim=2)  # （N, k）
        dist_max = distances.max(dim=1, keepdim=True)[0] + 1e-6
        distances_norm = distances / dist_max

        z_components = relative_pos[..., 2]
        polar_angles = torch.acos(torch.clamp(z_components / (distances + 1e-6), -1.0, 1.0))
        polar_norm = polar_angles / torch.pi

        x_components = relative_pos[..., 0]
        y_components = relative_pos[..., 1]
        azimuth_angles = torch.atan2(y_components, x_components)
        azimuth_norm = (azimuth_angles + torch.pi) / (2 * torch.pi)

        def build_histogram(values, num_bins):
            bins = torch.linspace(0, 1, num_bins + 1, device=device)
            hist = torch.zeros((values.shape[0], num_bins), device=device)
            for i in range(num_bins):
                mask_bin = (values >= bins[i]) & (values < bins[i + 1])
                hist[:, i] = mask_bin.sum(dim=1).float()
            hist = hist / (k + 1e-6)
            return hist

        dist_hist = build_histogram(distances_norm, num_bins)
        polar_hist = build_histogram(polar_norm, num_bins)
        azimuth_hist = build_histogram(azimuth_norm, num_bins)

        hist_feats = torch.cat([dist_hist, polar_hist, azimuth_hist], dim=1)

        if self.use_unpool:
            print(f"Using Unpooling:{self.use_unpool}")
            print("*" * 50)
            # 对低分辨率几何特征（如24维）反池化到48维
            hist_feats_reshaped = hist_feats.unsqueeze(1)  # [N, 1, 48]
            pooled_hist, indices = F.max_pool1d(
                input=hist_feats_reshaped,
                kernel_size=2,
                stride=2,
                return_indices=True
            )
            pooled_hist = pooled_hist.squeeze(1)  # [N, 24]

            # 反池化恢复到48维
            hist_feats = max_unpool1d_custom(
                input=pooled_hist,
                indices=indices,
                kernel_size=2,
                stride=2,
                output_size=48
            )


        return hist_feats

    def rotation_matrix_to_quaternion(self, R):
        """将3x3旋转矩阵转换为四元数 (w, x, y, z)"""
        batch_size = R.shape[0]
        q = torch.zeros(batch_size, 4, device=R.device)
        tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # 矩阵迹

        # 分情况计算四元数
        mask1 = tr > 0
        s1 = torch.sqrt(tr[mask1] + 1.0) * 2.0
        q[mask1, 0] = 0.25 * s1
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1

        mask2 = ~mask1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        s2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2.0
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
        q[mask2, 1] = 0.25 * s2
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

        mask3 = ~mask1 & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
        s3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2.0
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
        q[mask3, 2] = 0.25 * s3
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

        mask4 = ~mask1 & ~mask2 & ~mask3
        s4 = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2.0
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
        q[mask4, 3] = 0.25 * s4

        return F.normalize(q, dim=1)  # 归一化四元数

    # 多尺度特征
    def multi_scale_litept_features(self, litept_feats, sigma_list=[1, 3], device=None):
        """
        多尺度LitePT特征融合（兼容PyTorch张量和NumPy数组，作为类成员方法使用）
        Args:
            self: 类实例本身（必选，第一个参数）
            litept_feats: 输入特征（torch.Tensor 或 numpy.ndarray）
            sigma_list: 高斯滤波标准差列表
            device: 输出张量设备（默认自动识别）
        Returns:
            融合后的多尺度PyTorch张量
        """
        # 步骤1：统一转换为NumPy数组（方便执行高斯滤波）
        if isinstance(litept_feats, torch.Tensor):
            # 若为PyTorch张量，先转CPU再转NumPy
            feats_np = litept_feats.detach().cpu().numpy()
        elif isinstance(litept_feats, np.ndarray):
            # 若已为NumPy数组，直接使用
            feats_np = litept_feats
        else:
            raise TypeError(f"不支持的输入类型：{type(litept_feats)}，仅支持torch.Tensor和numpy.ndarray")

        # 步骤2：提取多尺度特征
        multi_scale_feats = [feats_np]  # 先加入原始特征
        for sigma in sigma_list:
            scale_feat = gaussian_filter1d(feats_np, sigma=sigma, axis=0)
            multi_scale_feats.append(scale_feat)

        # 步骤3：拼接并转换为PyTorch张量
        concat_feats_np = np.concatenate(multi_scale_feats, axis=1)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        concat_feats_tensor = torch.tensor(concat_feats_np, dtype=torch.float32).to(device)

        return concat_feats_tensor

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):

        print("*" * 50)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 不是在这个文件实现
        print(f"Using multi-scale depth:{self.use_multi_scale_depth}")
        print("*" * 50)

        self.spatial_lr_scale = spatial_lr_scale


        print(f"Using Outlier Factor: {self.use_OutlierFactor}")
        print("*" * 50)
        if self.use_OutlierFactor:
            # 离群点过滤
            points_np = np.asarray(pcd.points)
            colors_np = np.asarray(pcd.colors)

            # 对点数>100的点云进行过滤
            if len(points_np) > 100:
                lof = LocalOutlierFactor(n_neighbors=20)
                inlier_mask = lof.fit_predict(points_np) == 1
                points_np = points_np[inlier_mask]
                colors_np = colors_np[inlier_mask]

            fused_point_cloud = torch.tensor(points_np).cuda().float()
            fused_color = RGB2SH(torch.tensor(colors_np).float().cuda())

        else:
            # print(f"pcd point: {pcd.points}")
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
            # print(f"fused point cloud: {fused_point_cloud}")
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())


        if self.use_scale_constraint:
            print(f"Using Scale Constraint : {self.use_scale_constraint}")
            print("*" * 50)
            min_xyz = fused_point_cloud.min(dim=0)[0]
            max_xyz = fused_point_cloud.max(dim=0)[0]
            scene_extent = (max_xyz - min_xyz).norm().item() / 2.0  # 对角线长度/2
            # 方式2：包围盒最大边长（备选，根据你的场景选择）
            # scene_extent = (max_xyz - min_xyz).max().item()
            self.cameras_extent = scene_extent

        print(f"Using Gaussian Cluster : {self.use_gaussian_init}")
        if self.use_gaussian_init:
            print("*" * 50)
            torch.cuda.empty_cache()
            # --------------------------
            # 分层聚类 + 自适应协方差初始化
            # --------------------------
            points_np = fused_point_cloud.detach().cpu().numpy()
            num_points = points_np.shape[0]

            # 分层聚类
            if num_points > 1000:
                # 最近邻
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=50).fit(points_np)
                distances, _ = nbrs.kneighbors(points_np)
                avg_dist = distances.mean()
                # 基于平均距离动态设置聚类阈值
                Z = linkage(points_np, method='ward', metric='euclidean')
                threshold = np.percentile(Z[:, 2], 90)  # 取90%分位的距离作为阈值
            else:
                Z = linkage(points_np, method='ward', metric='euclidean')
                threshold = np.mean(Z[-int(0.1 * num_points):, 2])

            labels = fcluster(Z, t=threshold, criterion='distance')  # 生成聚类标签
            labels = torch.tensor(labels, device=fused_point_cloud.device)
            unique_labels = torch.unique(labels)

            # 为每个聚类计算自适应协方差，分解为缩放和旋转
            scales_list = []
            rots_list = []
            for label in unique_labels:
                mask = (labels == label)
                cluster_points = fused_point_cloud[mask]
                n = cluster_points.shape[0]

                if n < 3:
                    # 若聚类点数太少，直接使用全局尺度初始化
                    global_dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 1e-7)
                    global_scale = torch.log(torch.sqrt(global_dist2)).mean().item()
                    cluster_scales = torch.full((n, 3), global_scale, device=fused_point_cloud.device)
                    cluster_rots = torch.zeros((n, 4), device=fused_point_cloud.device)
                    cluster_rots[:, 0] = 1  # 单位四元数

                else:
                    # 计算聚类内点的协方差矩阵
                    mu = cluster_points.mean(dim=0, keepdim=True)  # 均值
                    centered = cluster_points - mu  # 去中心化
                    cov = torch.matmul(centered.T, centered) / (n - 1)  # 协方差矩阵 (3,3)

                    # SVD分解协方差矩阵：cov = U * S * V^T（U为旋转矩阵，S为特征值）
                    U, S, Vh = torch.svd(cov)
                    # 特征值开平方作为缩放因子（协方差 = 旋转 * 缩放^2 * 旋转^T）
                    scaling = torch.sqrt(S)  # (3,)
                    # 旋转矩阵转四元数
                    quat = self.rotation_matrix_to_quaternion(U.unsqueeze(0))  # (1,4)

                    # 为聚类内所有点分配相同的缩放和旋转（基于聚类分布）
                    cluster_scales = torch.log(scaling).repeat(n, 1)  # 存储log值（后续exp激活）
                    cluster_rots = quat.repeat(n, 1)  # 复制四元数

                scales_list.append(cluster_scales)
                rots_list.append(cluster_rots)

            # 按原始点顺序拼接缩放和旋转参数
            sorted_indices = torch.cat([torch.where(labels == l)[0] for l in unique_labels])
            scales = torch.cat(scales_list, dim=0)[sorted_indices]
            rots = torch.cat(rots_list, dim=0)[sorted_indices]
        else:
            print(f"Don‘t Use Cluster")
            print("*" * 50)



        print(f"Using Gaussian Model With LitePT: {self.use_litept}")
        print("*" * 50)

        print(f"Using multi-feature: {self.use_multi_frature}")
        print("*" * 50)

        if self.use_litept:

            # LitePT特征提取
            litept_extractor = LitePTFeatureExtractor()
            if self.use_OutlierFactor:
                litept_feats = litept_extractor.extract(points_np, pcd_colors=colors_np)
            else:
                litept_feats = litept_extractor.extract(np.asarray(pcd.points), pcd_colors=np.asarray(pcd.colors))
            # litept_feats = torch.tensor(litept_feats).cuda().float()
            if self.use_multi_frature:
                # 多尺度特征提取
                litept_feats = self.multi_scale_litept_features(litept_feats, sigma_list=[1, 3])
            else:
                litept_feats = litept_feats

            # 特征去噪
            litept_feats_np = litept_feats
            if isinstance(litept_feats_np, torch.Tensor):
                litept_feats_np = litept_feats_np.detach().cpu().numpy()


            # # 异常值裁剪（±3σ）
            # mean = litept_feats_np.mean(axis=0, keepdims=True)
            # std = litept_feats_np.std(axis=0, keepdims=True)
            # litept_feats_np = np.clip(litept_feats_np, mean - 3 * std, mean + 3 * std)

            # 双边滤波去噪
            litept_feats_reshaped = litept_feats_np.reshape(1, -1, litept_feats_np.shape[1])
            litept_feats_denoised = denoise_bilateral(
                litept_feats_reshaped,
                sigma_color=0.1,
                sigma_spatial=5,
                channel_axis=1
            )
            litept_feats_np = litept_feats_denoised.reshape(litept_feats_np.shape)

            litept_feats = torch.tensor(litept_feats_np).cuda().float()


            # 注意力层
            # 动态初始化注意力层
            if not hasattr(self, 'attn') or self.attn.in_features != litept_feats.shape[-1]:
                # 获取litept_feats的最后一维（特征维度）
                feat_dim = litept_feats.shape[-1]
                # 初始化注意力层：in_features=feat_dim（匹配输入），out_features=feat_dim（保持特征维度不变）
                self.attn = torch.nn.Linear(feat_dim, feat_dim).to(litept_feats.device)
                # 可选：初始化权重，提升训练效果
                torch.nn.init.xavier_uniform_(self.attn.weight)
                if self.attn.bias is not None:
                    torch.nn.init.constant_(self.attn.bias, 0.0)

            # 此时self.attn维度与litept_feats完全匹配，不会报错
            attn_weight = F.softmax(self.attn(litept_feats), dim=1)
            litept_feats = litept_feats * attn_weight

            print(f"Using SEBlock: {self.use_SEBlock}")
            if self.use_SEBlock:
                if not hasattr(self, 'se_block') or self.se_block.fc[2].out_features != feat_dim:
                    # self.se_block = SEBlock(feat_dim=feat_dim, reduction=4).to(device)
                    # self.se_block = SEBlock_ImprovedPool(feat_dim=feat_dim, reduction=4).to(device)
                    # self.se_block = SEBlock_ECA(feat_dim=feat_dim, gamma=2, b=1).to(device)
                    # self.se_block = SEBlock_Comprehensive(feat_dim=feat_dim, min_mid_dim=8, max_reduction=16).to(device)
                    self.se_block = self._create_se_block(feat_dim=feat_dim, device=device, se_type=self.current_se_type)
                    print(f"Current using SEBlock: {self.current_se_type}")
                    print("*" * 50)


                # SE模块前向传播
                litept_feats = self.se_block(litept_feats)

            # 特征归一化
            litept_feats = F.normalize(litept_feats, dim=-1, p=2)
            litept_feats = (litept_feats - litept_feats.mean()) / (litept_feats.std() + 1e-6)

            # 计算局部邻域几何直方图特征
            geo_hist_feats = self.compute_local_geometric_histograms(
                points=fused_point_cloud,
                k=32,
                num_bins=16
            )

            fused_feats = torch.cat([litept_feats, geo_hist_feats], dim=1)  # (N, 112)

            # 不计算局部邻域几何特征

            # fused_feats = litept_feats

            # 特征映射
            sh_dim = (self.max_sh_degree + 1) ** 2
            num_gaussians = fused_point_cloud.shape[0]
            features = torch.zeros((num_gaussians, 3, sh_dim), device="cuda", dtype=torch.float)
            if self.args.use_color:
                features[:, :3, 0] = fused_color

            if self.feat_projector is None:
                # self.feat_projector = nn.Linear(litept_feats.shape[1], 3 * (sh_dim - 1)).cuda()
                self.feat_projector = nn.Linear(fused_feats.shape[1], 3 * (sh_dim - 1)).cuda()

                nn.init.xavier_uniform_(self.feat_projector.weight)
            projected_feats = self.feat_projector(fused_feats)  # (N, 3*(sh_dim-1))
            projected_feats = projected_feats.reshape(num_gaussians, 3, sh_dim - 1)  # (N, 3, sh_dim-1)

            # 双重筛选融合区域（纹理密度+聚类稀疏度）
            # 纹理密度计算
            if self.use_OutlierFactor:
                color = torch.tensor(colors_np).cuda().float()
            else:
                color = torch.tensor(pcd.colors).cuda().float()
            color_var = color.var(dim=1).cpu().numpy()


            print(f"Using Dweight: {self.use_Dweight}")
            print('*' * 50)
            if self.use_Dweight:
                texture_threshold = np.percentile(color_var, 15)
                low_texture_mask = color_var < texture_threshold # 动态调整阈值
            else:
                low_texture_mask = color_var < 0.05

            # 自适应KMeans聚类
            sh_feats = features[:, :, 0].detach().cpu().numpy()
            num_points = len(sh_feats)
            n_clusters = min(max(num_points // 10000, 5), 20)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(sh_feats)
            cluster_density = np.bincount(kmeans.labels_)
            low_density_threshold = cluster_density.mean() / 4
            low_density_clusters = np.where(cluster_density < low_density_threshold)[0]
            sparse_mask = np.isin(kmeans.labels_, low_density_clusters)

            final_mask = sparse_mask | low_texture_mask
            final_mask = torch.tensor(final_mask, dtype=torch.bool).cuda()

            # 基于特征相似度的动态权重调节机制
            litept_feats_norm = F.normalize(litept_feats, dim=-1)
            sh_feats_tensor = torch.tensor(sh_feats).cuda().float()
            sh_feats_norm = F.normalize(sh_feats_tensor, dim=-1)
            similarity = F.cosine_similarity(litept_feats_norm[:, :3], sh_feats_norm, dim=-1).unsqueeze(1)
            if self.use_Dweight:
                color_var = color.var(dim=1)
                max_color_var = color_var.max()
                texture_complexity = 1 - (color_var / (max_color_var + 1e-6))
                similarity_scaled = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-6)
                litept_weight = 0.2 + 0.6 * (similarity_scaled * texture_complexity.reshape(-1, 1))
            else:
                litept_weight = 0.01 + 0.09 * similarity

            # 特征融合
            if self.use_fusionnet:
                print(f"Using FusionNet:{self.use_fusionnet}")
                print("*" * 50)
                sh_ac_dim = (self.max_sh_degree + 1) ** 2 - 1
                self.fusion_gate = GatedFusionNet(sh_ac_dim, litept_feats.shape[1]).cuda()

                # 预测权重
                fusion_weight = self.fusion_gate(features[:, :, 1:].detach(), litept_feats)

                # 扩展权重维度以匹配 [N, 3, 15]
                weight_expanded = fusion_weight.unsqueeze(1).repeat(1, 3, sh_ac_dim)

                # 融合
                # 增强型特征融合
                features[:, :3, 1:] = features[:, :3, 1:] * (1 - weight_expanded) + \
                                      projected_feats * weight_expanded



            else:
                if torch.any(final_mask):
                    weight_slice = litept_weight[final_mask]
                    if weight_slice.dim() == 1:
                        weight_slice = weight_slice.unsqueeze(1)
                    weight_expanded = weight_slice.unsqueeze(2)
                    weight_expanded = weight_expanded.repeat(1, 3, sh_dim - 1)
                    # if weight_expanded.dim() == 3:
                    #     weight_expanded = weight_expanded.repeat(1, 3, sh_dim - 1)
                    # else:
                    #     raise ValueError(f"weight_expanded维度异常：{weight_expanded.dim()}维，预期3维")
                    features[final_mask, :3, 1:] = features[final_mask, :3, 1:] * (1 - weight_expanded) + \
                                                   projected_feats[final_mask] * weight_expanded
                else:
                    print("Warning: No points match relaxed condition, select 5% points for LitePT fusion")
                    sorted_idx = np.argsort(color_var)
                    select_num = max(100, int(num_points * 0.05))
                    select_idx = sorted_idx[:select_num]
                    final_mask = torch.zeros(num_points, dtype=torch.bool).cuda()
                    final_mask[select_idx] = True
                    weight_slice = litept_weight[final_mask]
                    if weight_slice.dim() == 1:
                        weight_slice = weight_slice.unsqueeze(1)
                    weight_expanded = weight_slice.unsqueeze(2)
                    weight_expanded = weight_expanded.repeat(1, 3, sh_dim - 1)
                    features[final_mask, :3, 1:] = features[final_mask, :3, 1:] * (1 - weight_expanded) + \
                                                   projected_feats[final_mask] * weight_expanded

            if self.use_debug:
                # debug，特征可视化
                sample_size = min(10000, len(litept_feats))
                litept_feats_tsne = TSNE(n_components=2, random_state=42).fit_transform(
                    litept_feats.detach().cpu().numpy()[:sample_size]
                )
                sh_feats_tsne = TSNE(n_components=2, random_state=42).fit_transform(
                    features[:, :, 0].detach().cpu().numpy()[:sample_size]
                )
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.scatter(litept_feats_tsne[:, 0], litept_feats_tsne[:, 1], s=1)
                plt.title("LitePT Features")
                plt.subplot(122)
                plt.scatter(sh_feats_tsne[:, 0], sh_feats_tsne[:, 1], s=1)
                plt.title("IPSM SH Features")
                plt.savefig("orchids_feature_tsne.png")
                plt.close()
        else:
            features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            if self.args.use_color:
                features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0



        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        self.init_point = fused_point_cloud

        if self.use_gaussian_init:
            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))  # 新缩放参数
            self._rotation = nn.Parameter(rots.requires_grad_(True))  # 新旋转参数
            self._opacity = nn.Parameter(inverse_sigmoid(
                0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")).requires_grad_(
                True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            self.confidence = torch.ones_like(self._opacity, device="cuda")
        else:
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)

            self.init_dist = torch.sqrt(dist2).detach()

            scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1

            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            self.confidence = torch.ones_like(opacities, device="cuda")
        if self.args.train_bg:
            self.bg_color = nn.Parameter((torch.zeros(3, 1, 1) + 0.).cuda().requires_grad_(True))
    # use LitePT and cluster
    # def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
    #     self.spatial_lr_scale = spatial_lr_scale
    #     fused_point_cloud = torch.tensor(np.asarray(pcd.points)).cuda().float()
    #     fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    #     num_points = fused_point_cloud.shape[0]
    #     print(f"使用分层聚类: {self.use_gaussian_init}, 使用LitePT: {self.use_litept}")
    #     # --------------------------
    #     # 分层聚类计算缩放和旋转参数
    #     # --------------------------
    #     if self.use_gaussian_init:
    #         torch.cuda.empty_cache()
    #         points_np = fused_point_cloud.detach().cpu().numpy()
    #
    #         # 分层聚类
    #         if num_points > 1000:
    #             nbrs = NearestNeighbors(n_neighbors=50).fit(points_np)
    #             distances, _ = nbrs.kneighbors(points_np)
    #             avg_dist = distances.mean()
    #             Z = linkage(points_np, method='ward', metric='euclidean')
    #             threshold = np.percentile(Z[:, 2], 90)  # 90%分位阈值
    #         else:
    #             Z = linkage(points_np, method='ward', metric='euclidean')
    #             threshold = np.mean(Z[-int(0.1 * num_points):, 2])
    #
    #         labels = fcluster(Z, t=threshold, criterion='distance')  # 聚类标签
    #         labels = torch.tensor(labels, device=fused_point_cloud.device)
    #         unique_labels = torch.unique(labels)
    #
    #         scales_list = []
    #         rots_list = []
    #         for label in unique_labels:
    #             mask = (labels == label)
    #             cluster_points = fused_point_cloud[mask]
    #             n = cluster_points.shape[0]
    #
    #             if n < 3:
    #                 nbrs = NearestNeighbors(n_neighbors=min(5, num_points - 1)).fit(points_np)
    #                 dists, _ = nbrs.kneighbors(cluster_points.cpu().numpy())
    #                 local_scale = torch.log(torch.tensor(dists.mean()).cuda().float())
    #                 cluster_scales = torch.full((n, 3), local_scale, device=fused_point_cloud.device)
    #                 cluster_rots = torch.zeros((n, 4), device=fused_point_cloud.device)
    #                 cluster_rots[:, 0] = 1  # 单位四元数
    #             else:
    #
    #                 mu = cluster_points.mean(dim=0, keepdim=True)
    #                 centered = cluster_points - mu
    #                 cov = torch.matmul(centered.T, centered) / (n - 1)
    #                 U, S, Vh = torch.svd(cov)
    #                 scaling = torch.sqrt(S)
    #                 quat = self.rotation_matrix_to_quaternion(U.unsqueeze(0))
    #                 cluster_scales = torch.log(scaling).repeat(n, 1)
    #                 cluster_rots = quat.repeat(n, 1)
    #
    #             scales_list.append(cluster_scales)
    #             rots_list.append(cluster_rots)
    #
    #         sorted_indices = torch.cat([torch.where(labels == l)[0] for l in unique_labels])
    #         scales = torch.cat(scales_list, dim=0)[sorted_indices]
    #         rots = torch.cat(rots_list, dim=0)[sorted_indices]
    #     else:
    #         dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 1e-7)
    #         scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    #         rots = torch.zeros((num_points, 4), device="cuda")
    #         rots[:, 0] = 1
    #         labels = None
    #
    #     # --------------------------
    #     # LitePT特征提取与融合
    #     # --------------------------
    #     sh_dim = (self.max_sh_degree + 1) ** 2
    #     features = torch.zeros((num_points, 3, sh_dim), device="cuda", dtype=torch.float)
    #     if self.args.use_color:
    #         features[:, :3, 0] = fused_color  # 保留DC分量的颜色信息
    #
    #     if self.use_litept:
    #         print("*" * 50)
    #         print("启用LitePT特征融合")
    #         # 提取LitePT特征
    #         litept_extractor = LitePTFeatureExtractor()
    #         litept_feats = litept_extractor.extract(
    #             np.asarray(pcd.points),
    #             pcd_colors=np.asarray(pcd.colors)
    #         )
    #         litept_feats = torch.tensor(litept_feats).cuda().float()
    #
    #         # 特征注意力与归一化
    #         attn_weight = F.softmax(self.attn(litept_feats), dim=1)
    #         litept_feats = litept_feats * attn_weight
    #         litept_feats = F.normalize(litept_feats, dim=-1)
    #         litept_feats = (litept_feats - litept_feats.mean()) / (litept_feats.std() + 1e-6)
    #
    #         # 投影LitePT特征到SH的AC分量
    #         if self.feat_projector is None:
    #             self.feat_projector = nn.Linear(litept_feats.shape[1], 3 * (sh_dim - 1)).cuda()
    #         projected_feats = self.feat_projector(litept_feats).reshape(num_points, 3, sh_dim - 1)
    #
    #         # 基于分层聚类结果融合特征
    #         if self.use_gaussian_init and labels is not None:
    #
    #             cluster_sizes = torch.bincount(labels)
    #             sparse_threshold = cluster_sizes.float().mean() * 0.5
    #             sparse_clusters = torch.where(cluster_sizes < sparse_threshold)[0]
    #             mask = torch.isin(labels, sparse_clusters)
    #             print(f"稀疏簇数量: {len(sparse_clusters)}, 总簇数: {len(unique_labels)}")
    #
    #             features[mask, :3, 1:] = features[mask, :3, 1:] * 0.95 + projected_feats[mask] * 0.05
    #         else:
    #
    #             features[:, :3, 1:] = features[:, :3, 1:] * 0.9 + projected_feats * 0.1
    #
    #         if self.use_debug and num_points > 1000:
    #             litept_tsne = TSNE(n_components=2).fit_transform(litept_feats[:10000].cpu().numpy())
    #             sh_tsne = TSNE(n_components=2).fit_transform(features[:10000, :, 0].cpu().numpy())
    #             plt.figure(figsize=(10, 5))
    #             plt.subplot(121);
    #             plt.scatter(litept_tsne[:, 0], litept_tsne[:, 1], s=1);
    #             plt.title("LitePT Features")
    #             plt.subplot(122);
    #             plt.scatter(sh_tsne[:, 0], sh_tsne[:, 1], s=1);
    #             plt.title("SH Features (with LitePT)")
    #             plt.savefig("litept_sh_features.png")
    #             plt.close()
    #
    #     # --------------------------
    #     # 初始化高斯参数
    #     # --------------------------
    #     print(f"初始化高斯数量: {num_points}")
    #     self.init_point = fused_point_cloud
    #
    #     opacities = inverse_sigmoid(0.1 * torch.ones((num_points, 1), dtype=torch.float, device="cuda"))
    #
    #     self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    #     self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._scaling = nn.Parameter(scales.requires_grad_(True))
    #     self._rotation = nn.Parameter(rots.requires_grad_(True))
    #     self._opacity = nn.Parameter(opacities.requires_grad_(True))
    #     self.max_radii2D = torch.zeros(num_points, device="cuda")
    #     self.confidence = torch.ones_like(opacities, device="cuda")
    #
    #     if self.args.train_bg:
    #         self.bg_color = nn.Parameter(torch.zeros(3, 1, 1).cuda().requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.training_args = training_args

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
        ]
        if self.args.train_bg:
            l.append({'params': [self.bg_color], 'lr': 0.001, "name": "bg_color"})

        if self.fusion_gate is not None:
            l.append({'params': self.fusion_gate.parameters(), 'lr': 0.001, "name": "fusion_gate"})




        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.use_cosine:
            print("============================use cosine =================================")
            lr_init = self.training_args.position_lr_init * self.spatial_lr_scale
            lr_final = self.training_args.position_lr_final * self.spatial_lr_scale
            max_steps = self.training_args.position_lr_max_steps * 1.5
            iteration = min(iteration, max_steps)
            # 余弦退火
            cosine_decay = 0.5 * (1 + math.cos(math.pi * iteration / max_steps))
            xyz_lr = lr_final + (lr_init - lr_final) * cosine_decay
        else:
            xyz_lr = self.xyz_scheduler_args(iteration)


        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = xyz_lr
                # print(f"lr: {xyz_lr}")
                return xyz_lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, reset_param=0.05):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * reset_param))
        if len(self.optimizer.state.keys()):
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]
    
    def reset_opacity_origin(self, reset_param=0.01):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * reset_param))
        if len(self.optimizer.state.keys()):
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if self.use_fusionnet:
                valid_names = ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]
                if group["name"] not in valid_names:
                    continue
            else:
                if group["name"] in ['bg_color']:
                    continue


            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def dist_prune(self):
        dist = chamfer_dist(self.init_point, self._xyz)
        valid_points_mask = (dist < 3.0)
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def prune_points(self, mask, iter):
        if self.use_dd_drop:
            if iter > self.args.prune_from_iter:
                current_num = self._xyz.shape[0]
                new_prune_mask = (self.get_opacity < self.args.prune_threshold).squeeze()
                if hasattr(self, 'max_radii2D') and self.max_radii2D.numel() > 0:
                    if len(self.max_radii2D) != current_num:
                        self.max_radii2D = self.max_radii2D[:current_num] if len(
                            self.max_radii2D) > current_num else torch.cat([self.max_radii2D, torch.zeros(
                            current_num - len(self.max_radii2D), dtype=self.max_radii2D.dtype,
                            device=self.max_radii2D.device)])
                    big_points_vs = self.max_radii2D > (
                        self.args.max_screen_size if hasattr(self.args, 'max_screen_size') else 0.1)
                    new_prune_mask = torch.logical_or(new_prune_mask, big_points_vs)

                if len(new_prune_mask) != current_num:
                    new_prune_mask = new_prune_mask[:current_num] if len(new_prune_mask) > current_num else torch.cat(
                        [new_prune_mask, torch.zeros(current_num - len(new_prune_mask), dtype=new_prune_mask.dtype,
                                                     device=new_prune_mask.device)])

                valid_points_mask = ~new_prune_mask

                optimizable_tensors = self._prune_optimizer(valid_points_mask)
                self._xyz = optimizable_tensors["xyz"]
                self._features_dc = optimizable_tensors["f_dc"]
                self._features_rest = optimizable_tensors["f_rest"]
                self._opacity = optimizable_tensors["opacity"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]

                self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
                self.denom = self.denom[valid_points_mask]
                self.max_radii2D = self.max_radii2D[valid_points_mask]

                # ========== 核心修复：先统一 confidence 为二维，再用一维掩码索引 ==========
                # 1. 确保 confidence 是二维 [N,1]
                if len(self.confidence.shape) == 1:
                    self.confidence = self.confidence.unsqueeze(-1)
                # 2. 确保 valid_points_mask 是一维（掩码必须是一维才能索引二维张量）
                if len(valid_points_mask.shape) > 1:
                    valid_points_mask = valid_points_mask.squeeze(-1)
                # 3. 一维掩码索引二维 confidence，保留二维维度
                self.confidence = self.confidence[valid_points_mask]

                # 调试日志
                print(
                    f"[Prune] 迭代{iter}：外部掩码长度{len(mask)} → 内部重新生成掩码长度{len(new_prune_mask)} | 当前高斯数{current_num} | 剪枝后{self._xyz.shape[0]} | confidence维度{self.confidence.shape}")
        else:
            if iter > self.args.prune_from_iter:

                valid_points_mask = ~mask
                optimizable_tensors = self._prune_optimizer(valid_points_mask)

                self._xyz = optimizable_tensors["xyz"]
                self._features_dc = optimizable_tensors["f_dc"]
                self._features_rest = optimizable_tensors["f_rest"]
                self._opacity = optimizable_tensors["opacity"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]

                self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

                self.denom = self.denom[valid_points_mask]
                self.max_radii2D = self.max_radii2D[valid_points_mask]
                self.confidence = self.confidence[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if self.use_fusionnet:
                if group["name"] not in tensors_dict:
                    continue
            else:
                if group["name"] in ['bg_color']:
                    continue



            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if self.use_dd_drop:
            # 生成二维的 new_confidence [K,1]
            new_confidence = torch.ones((new_opacities.shape[0], 1), device="cuda")
            # 确保 self.confidence 是二维（防御性处理）
            if len(self.confidence.shape) == 1:
                self.confidence = self.confidence.unsqueeze(-1)
            # 二维拼接（维度0）
            self.confidence = torch.cat([self.confidence, new_confidence], 0)

            self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros_like(new_opacities[:, 0])])
            if not self.xyz_gradient_accum.shape[0] == self._xyz.shape[0]:
                self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 3), device="cuda")
                self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        else:
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            self.confidence = torch.cat([self.confidence, torch.ones(new_opacities.shape, device="cuda")], 0)

    def proximity(self, scene_extent, N = 3):
        dist, nearest_indices = distCUDA2(self.get_xyz)
        selected_pts_mask = torch.logical_and(dist > (5. * scene_extent),
                                              torch.max(self.get_scaling, dim=1).values > (scene_extent))

        new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        source_xyz = self._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
        target_xyz = self._xyz[new_indices]
        new_xyz = (source_xyz + target_xyz) / 2
        new_scaling = self._scaling[new_indices]
        new_rotation = torch.zeros_like(self._rotation[new_indices])
        new_rotation[:, 0] = 1
        new_features_dc = torch.zeros_like(self._features_dc[new_indices])
        new_features_rest = torch.zeros_like(self._features_rest[new_indices])
        new_opacity = self._opacity[new_indices]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def densify_and_split(self, grads, grad_threshold, scene_extent, iter, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        dist, _ = distCUDA2(self.get_xyz)
        selected_pts_mask2 = torch.logical_and(dist > (self.args.dist_thres * scene_extent),
                                               torch.max(self.get_scaling, dim=1).values > ( scene_extent))
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask2)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, iter)
        
    def densify_and_split_origin(self, grads, grad_threshold, scene_extent, iter, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        if self.use_dd_drop:
            pass
        else:
            self.prune_points(prune_filter, iter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, iter):
        if self.use_dd_drop:
            current_num_gaussians = self._xyz.shape[0]
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0

            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent, iter)
            if iter < 2000:
                self.proximity(extent)

            prune_mask = (self.get_opacity < min_opacity).squeeze()
            # 对齐掩码长度（双重保险）
            if len(prune_mask) != current_num_gaussians:
                prune_mask = prune_mask[:current_num_gaussians] if len(
                    prune_mask) > current_num_gaussians else torch.cat([prune_mask, torch.zeros(
                    current_num_gaussians - len(prune_mask), dtype=prune_mask.dtype, device=prune_mask.device)])

            if max_screen_size:
                # 校验max_radii2D长度
                if len(self.max_radii2D) != current_num_gaussians:
                    self.max_radii2D = self.max_radii2D[:current_num_gaussians] if len(
                        self.max_radii2D) > current_num_gaussians else torch.cat([self.max_radii2D, torch.zeros(
                        current_num_gaussians - len(self.max_radii2D), dtype=self.max_radii2D.dtype,
                        device=self.max_radii2D.device)])

                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

                # 对齐big_points_ws长度
                if len(big_points_ws) != current_num_gaussians:
                    big_points_ws = big_points_ws[:current_num_gaussians] if len(
                        big_points_ws) > current_num_gaussians else torch.cat([big_points_ws, torch.zeros(
                        current_num_gaussians - len(big_points_ws), dtype=big_points_ws.dtype,
                        device=big_points_ws.device)])

                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

            self.prune_points(prune_mask, iter)
            torch.cuda.empty_cache()
        else:
            grads = self.xyz_gradient_accum / self.denom
            grads[grads.isnan()] = 0.0

            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent, iter)
            if iter < 2000:
                self.proximity(extent)

            prune_mask = (self.get_opacity < min_opacity).squeeze()
            if max_screen_size:
                big_points_vs = self.max_radii2D > max_screen_size
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
                prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

            self.prune_points(prune_mask, iter)
            torch.cuda.empty_cache()

    def densify_and_prune_origin(self, max_grad, min_opacity, extent, max_screen_size, iter):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split_origin(grads, max_grad, extent, iter)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask, iter)

        torch.cuda.empty_cache()
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.denom[update_filter] += 1

    def regularization_loss(self, lambda_opacity=0.001, lambda_scale=0.001):
        """
        计算正则化损失：
        1. Opacity Entropy: 鼓励 opacity 趋向 0 或 1
        2. Scale Isotropy: 鼓励高斯球体不要过度拉伸（可选）
        """
        losses = {}

        # --- 1. Opacity Entropy Loss ---
        opacity = self.get_opacity  # 获取激活后的 opacity [0, 1]
        # 添加 epsilon 避免 log(0)
        opacity = torch.clamp(opacity, 1e-6, 1 - 1e-6)

        # 二进制熵公式: -p*log(p) - (1-p)*log(1-p)
        # 我们希望熵最小（确定性最高）
        entropy = -opacity * torch.log(opacity) - (1 - opacity) * torch.log(1 - opacity)
        losses['opacity_entropy'] = entropy.mean() * lambda_opacity

        # --- 2. Scale Regularization (防止极度扁平的伪影) ---
        scaling = self.get_scaling  # (N, 3)
        # 计算最大尺度与最小尺度的比率
        max_scale = scaling.max(dim=1).values
        min_scale = scaling.min(dim=1).values
        anisotropy = max_scale / (min_scale + 1e-6)

        # 惩罚各向异性过大的高斯 (例如长宽比超过 20)
        scale_loss = torch.relu(anisotropy - 20.0).mean()
        losses['scale_reg'] = scale_loss * lambda_scale

        total_loss = losses['opacity_entropy'] + losses['scale_reg']
        return total_loss


    # 深度一致性缩放约束
    def apply_depth_scale_constraint(self, iteration):
        # 1. 初始化/校验 cameras_extent（之前修复的属性）
        if not hasattr(self, "cameras_extent") or self.cameras_extent is None:
            min_xyz = self._xyz.detach().min(dim=0)[0]
            max_xyz = self._xyz.detach().max(dim=0)[0]
            self.cameras_extent = (max_xyz - min_xyz).norm().item() / 2.0

        # 2. 定义尺度约束的因子（根据你的业务逻辑调整，示例值）
        extent_factor = 0.1  # 可替换为迭代相关的动态因子，如随iteration衰减

        # 3. 计算全局缩放上限（匹配self._scaling的维度）
        upper_limit_global = self.cameras_extent * extent_factor
        # 生成[N]的上限，再扩展为[N,3]匹配self._scaling（shape: [N,3]）
        max_limit = torch.full((self._scaling.shape[0],), upper_limit_global, device=self._scaling.device)
        log_max_limit = torch.log(max_limit)  # [N]
        log_max_limit = log_max_limit.unsqueeze(1).repeat(1, 3)  # [N,3]

        # 4. 核心修复：避免叶子张量的in-place操作
        # 方式1：使用torch.no_grad()包裹非原地赋值（推荐）
        with torch.no_grad():  # 临时禁用梯度追踪，安全修改Parameter
            # 计算裁剪后的新scaling值
            new_scaling = torch.min(self._scaling, log_max_limit)
            # 非原地赋值（替换copy_的in-place操作）
            self._scaling.copy_(new_scaling)  # 此时在no_grad中，允许in-place

    def multi_view_consistency_loss(self, cameras, lambda_mv=0.05):
        """
        多视图投影一致性正则化：约束高斯点在不同相机视角下的投影位置/深度一致
        Args:
            cameras: 相机列表（至少2个）
            lambda_mv: 正则化权重
        Returns:
            mv_loss: 多视图一致性损失
        """
        if len(cameras) < 2:
            return torch.tensor(0.0, device="cuda")

        # 1. 获取高斯点的3D坐标
        gauss_xyz = self.get_xyz  # [N, 3]

        # 2. 选第一个相机作为参考视角，投影高斯点到参考图像平面
        ref_cam = cameras[0]
        # 参考相机的投影矩阵（内参@外参的投影部分，需根据你的Camera类调整）
        ref_proj_mat = ref_cam.full_proj_transform  # 需确保是 [4,4] 投影矩阵（CUDA）
        # 高斯点齐次坐标 [N, 4]
        gauss_xyz_homo = torch.cat([gauss_xyz, torch.ones_like(gauss_xyz[:, :1])], dim=-1)
        # 投影到参考图像平面 [N, 3] (u, v, depth)
        ref_proj = (ref_proj_mat @ gauss_xyz_homo.T).T  # [N, 4]
        ref_proj_uv = ref_proj[:, :2] / ref_proj[:, 2:3]  # 归一化到图像平面 [N, 2]
        ref_depth = ref_proj[:, 2:3]  # 参考深度 [N, 1]

        # 3. 遍历其他相机，计算投影一致性损失
        mv_loss = 0.0
        for cam in cameras[1:]:
            # 当前相机的投影矩阵
            curr_proj_mat = cam.full_proj_transform
            # 投影高斯点到当前相机图像平面
            curr_proj = (curr_proj_mat @ gauss_xyz_homo.T).T
            curr_proj_uv = curr_proj[:, :1] / curr_proj[:, 2:3]
            curr_depth = curr_proj[:, 2:3]

            # 损失1：投影坐标一致性（L1）
            uv_loss = l1_loss(ref_proj_uv, curr_proj_uv)
            # 损失2：深度一致性（Pearson相关系数）
            depth_corr = 1 - pearson_corrcoef(ref_depth, curr_depth)
            # 总视图损失
            mv_loss += (uv_loss + depth_corr)

        # 平均所有视图对的损失，并乘以正则化权重
        loss = lambda_mv * (mv_loss / (len(cameras) - 1))
        return loss





