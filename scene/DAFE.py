import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn

class DAFE(nn.Module):
    def __init__(self,
                 depth_estimator,  # 单目深度估计模型（如MiDaS）
                 tau=0.1,  # 远场阈值（τ*D_max，文献最优5%-15%）
                 lambda_dafe=1.0,  # DAFE损失权重（文献最优1.0）
                 image_size=(512, 512)):  # 深度估计输入尺寸
        super(DAFE, self).__init__()
        self.depth_estimator = depth_estimator
        self.tau = tau
        self.lambda_dafe = lambda_dafe
        self.image_size = image_size

        # 图像预处理（适配深度估计模型输入）
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def generate_far_field_mask(self, rendered_img):
        """
        生成远场掩码 M_dis：深度 > τ*D_max 的区域设为1
        Args:
            rendered_img: 渲染图像 (3, H, W) 或 (B, 3, H, W)
        Returns:
            far_mask: 远场掩码 (H, W) 或 (B, H, W)
        """
        # 处理单张图像或批次图像
        is_batch = len(rendered_img.shape) == 4
        if not is_batch:
            rendered_img = rendered_img.unsqueeze(0)  # (1, 3, H, W)

        B, C, H, W = rendered_img.shape

        # 1. 预处理图像（适配深度估计模型）
        img_resized = F.interpolate(rendered_img, size=self.image_size, mode='bilinear', align_corners=False)
        img_normalized = self.preprocess(img_resized)

        # 2. 单目深度估计（关闭梯度计算，避免影响主模型）
        with torch.no_grad():
            depth_map = self.depth_estimator(img_normalized)  # (B, 1, H', W')

        # 3. 调整深度图尺寸与渲染图一致
        depth_map = F.interpolate(depth_map, size=(H, W), mode='bilinear', align_corners=False).squeeze(1)  # (B, H, W)

        # 4. 计算远场阈值 τ*D_max（每张图独立计算）
        D_max = depth_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # (B, 1, 1)
        far_threshold = self.tau * D_max

        # 5. 生成二值掩码（远场=1，近场=0）
        far_mask = (depth_map > far_threshold).float()  # (B, H, W)

        # 恢复单张图像格式
        if not is_batch:
            far_mask = far_mask.squeeze(0)  # (H, W)

        return far_mask

    def forward(self, rendered_img, gt_img):
        """
        前向传播：计算DAFE损失（文献公式5）
        Args:
            rendered_img: 渲染图像 (3, H, W) 或 (B, 3, H, W)
            gt_img: 真实图像 (3, H, W) 或 (B, 3, H, W)
        Returns:
            dafe_loss: DAFE损失（标量）
        """
        # 1. 生成远场掩码
        far_mask = self.generate_far_field_mask(rendered_img)  # (H, W) 或 (B, H, W)

        # 2. 扩展掩码维度（匹配图像通道数）
        if len(far_mask.shape) == 2:
            far_mask = far_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        else:
            far_mask = far_mask.unsqueeze(1)  # (B, 1, H, W)

        # 3. 计算掩码区域的L1损失（仅远场区域）
        masked_rendered = rendered_img * far_mask
        masked_gt = gt_img * far_mask

        # 避免掩码全为0导致的除以0
        mask_sum = far_mask.sum() + 1e-8
        dafe_loss = (masked_rendered - masked_gt).abs().sum() / mask_sum

        # 4. 应用损失权重
        return self.lambda_dafe * dafe_loss
