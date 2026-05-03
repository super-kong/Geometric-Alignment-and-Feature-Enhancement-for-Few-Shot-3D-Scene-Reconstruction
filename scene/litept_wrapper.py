from collections import OrderedDict
import os
import torch
from torch import nn
import numpy as np
from dataset.datatools.transform import Compose
from LitePT.litept.model import LitePT
import open3d as o3d


class LitePTFeatureExtractor:
    def __init__(self):
        self.model = LitePT()
        self.model.cuda()

        ckpt_local_path = "/home/sari/PycharmProjects/IPSM/pretrained_models/litept_model_best.pth"
        if not os.path.exists(ckpt_local_path):
            raise FileNotFoundError(
                f"LitePT权重文件未找到！请检查路径：{ckpt_local_path}\n"
                "建议将权重文件放在IPSM项目根目录的weights文件夹下，命名为litept_model_best.pth"
            )

        # 加载本地权重文件
        ckpt = torch.load(ckpt_local_path, map_location="cpu")
        weight = OrderedDict()
        prefix = "module.backbone."


        for key, value in ckpt["state_dict"].items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                weight[new_key] = value

        # 加载权重到模型
        self.model.load_state_dict(weight, strict=True)
        self.model.eval()  # 设为评估模式，禁用Dropout/BatchNorm

        # 点云预处理
        self.transform = Compose([
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "inverse"),
                feat_keys=("coord", "strength"),
            ),
        ])

    # def extract(self, points):
    #     coord = points[:, :3].astype(np.float32)
    #     # strength = np.zeros((len(coord), 1), dtype=np.float32)
    #     coord_min = coord.min(axis=0)  # 按列求min
    #     coord_max = coord.max(axis=0)  # 按列求max
    #
    #
    #     x_coord = coord[:, 0:1]
    #     strength = (x_coord - coord_min[0]) / (coord_max[0] - coord_min[0] + 1e-6)
    #     # strength = (coord - coord.min()) / (coord.max() - coord.min() + 1e-6)[:, :1]
    #
    #     # 动态调整网格大小
    #     bbox_diag = np.linalg.norm(coord_max - coord_min)  # 包围盒对角线长度
    #     grid_size = max(bbox_diag / 200, 0.005)  # 最小0.005m，避免网格过小
    #     # if grid_size < 0.5:
    #     #     grid_size = grid_size
    #     # else:
    #     #     grid_size = 0.05
    #     # 修复1：迭代transform实例，通过类名判断是否为GridSample
    #     for t in self.transform.transforms:
    #         # 判断当前transform是否是GridSample类的实例（通过类名）
    #         if t.__class__.__name__ == "GridSample":
    #             # 修复2：访问实例属性修改grid_size（而非字典下标）
    #             t.grid_size = grid_size
    #
    #     point = dict(coord=coord, strength=strength)
    #     point = self.transform(point)
    #
    #     with torch.no_grad():
    #         for key in point.keys():
    #             if isinstance(point[key], torch.Tensor):
    #                 point[key] = point[key].cuda(non_blocking=True)
    #         point = self.model(point)
    #         dense_feat = point.feat[point.inverse]
    #
    #
    #     if dense_feat.shape[1] != 64:
    #         proj = nn.Linear(dense_feat.shape[1], 64).cuda()
    #         proj.eval()
    #
    #         dense_feat = proj(dense_feat)
    #
    #     return dense_feat.detach().cpu().numpy()


    def extract(self, points, pcd_colors=None):
        coord = points[:, :3].astype(np.float32)
        # strength = np.zeros((len(coord), 1), dtype=np.float32)
        coord_min = coord.min(axis=0)
        coord_max = coord.max(axis=0)

        # x_coord = coord[:, 0:1]
        # strength = (x_coord - coord_min[0]) / (coord_max[0] - coord_min[0] + 1e-6)

        # 将点云法线+颜色灰度值作为伪强度
        # 融入颜色灰度值+点云法线
        # 点云法线
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=16))
        normals = np.asarray(pcd_o3d.normals).astype(np.float32)

        if pcd_colors is not None:
            gray = np.mean(pcd_colors, axis=1, keepdims=True)
        else:
            gray = np.zeros((len(coord), 1), dtype=np.float32)

        # 合并为伪强度，几何+纹理双信息
        strength = (gray + normals[:, 0:1]) / 2  # 法线x轴区分朝向
        strength = (strength - strength.min()) / (strength.max() - strength.min() + 1e-6)
        # strength = (coord - coord.min()) / (coord.max() - coord.min() + 1e-6)[:, :1]

        # 动态调整网格大小
        bbox_diag = np.linalg.norm(coord_max - coord_min)
        grid_size = max(bbox_diag / 200, 0.005)
        # if grid_size < 0.5:
        #     grid_size = grid_size
        # else:
        #     grid_size = 0.05
        for t in self.transform.transforms:
            if t.__class__.__name__ == "GridSample":
                t.grid_size = grid_size

        point = dict(coord=coord, strength=strength)
        point = self.transform(point)

        with torch.no_grad():
            for key in point.keys():
                if isinstance(point[key], torch.Tensor):
                    point[key] = point[key].cuda(non_blocking=True)
            point = self.model(point)
            dense_feat = point.feat[point.inverse]


        if dense_feat.shape[1] != 64:
            proj = nn.Linear(dense_feat.shape[1], 64).cuda()
            proj.eval()

            dense_feat = proj(dense_feat)

        return dense_feat.detach().cpu().numpy()