import torch
import torch.nn as nn
import torch.nn.functional as F


# SE通道注意力模块
class SEBlock(nn.Module):
    def __init__(self, feat_dim, reduction=4):
        """
        Args:
            feat_dim: 输入特征的通道数（即litept_feats的最后一维维度）
            reduction: 通道压缩比，通常设为4（平衡性能与计算量）
        """
        super(SEBlock, self).__init__()
        # 全局平均池化，聚合每个通道的全局信息
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层+非线性激活，生成通道权重
        self.fc = nn.Sequential(
            # 压缩通道数
            nn.Linear(feat_dim, feat_dim // reduction, bias=False),
            nn.ReLU(inplace=True),  # 非线性激活，增强表达能力
            # 恢复通道数
            nn.Linear(feat_dim // reduction, feat_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播：输入x shape (N, feat_dim)，输出shape (N, feat_dim)
        """
        b, c = x.shape

        x_reshaped = x.unsqueeze(2)

        # Squeeze：全局平均池化
        y = self.avg_pool(x_reshaped).view(b, c)

        y = self.fc(y)

        out = x * y

        return out

# 改进池化
class SEBlock_ImprovedPool(nn.Module):
    def __init__(self, feat_dim, reduction=4):
        super().__init__()  # 简化super调用（PyTorch>=3.5推荐写法）
        self.feat_dim = feat_dim
        # 限制压缩比，避免feat_dim较小时出现mid_dim=0的错误
        self.reduction = max(reduction, 2) if feat_dim < 8 else reduction
        mid_dim = feat_dim // self.reduction

        # 定义全连接层（直接用mid_dim，避免重复计算feat_dim//reduction）
        self.fc1 = nn.Linear(2 * feat_dim, mid_dim, bias=False)
        self.fc2 = nn.Linear(mid_dim, feat_dim, bias=False)

        # 可选：添加轻量归一化，提升小批量训练稳定性（二维特征适配LayerNorm）
        self.ln = nn.LayerNorm(mid_dim)

    def forward(self, x):
        """
        前向传播：输入x shape (N, feat_dim)，输出shape (N, feat_dim)
        真正融合 二维特征的GAP+GMP，提升通道权重判别性，适配小批量/低维特征
        """
        b, c = x.shape  # b:样本数，c:特征维度/通道数
        # 核心修复：实现二维特征的全局平均池化(GAP)和全局最大池化(GMP)
        # keepdim=True + expand_as(x)：保证池化后形状与原特征一致，方便拼接
        x_avg = torch.mean(x, dim=1, keepdim=True).expand_as(x)  # (N, c)，全局通道均值广播
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_max = x_max.expand_as(x)  # (N, c)，全局通道最大值广播

        # 融合两种池化特征：拼接得到 (N, 2c)，保留GAP+GMP的互补信息
        x_fused = torch.cat([x_avg, x_max], dim=1)

        # 生成通道权重：激活+归一化组合，提升梯度稳定性（替换原生relu，可选inplace）
        y = self.fc1(x_fused)
        y = self.ln(y)  # 归一化在激活前，符合最优实践
        y = F.relu(y, inplace=False)  # 小模块建议关闭inplace，避免梯度覆盖
        y = torch.sigmoid(self.fc2(y))  # 0-1权重，对原特征逐通道加权

        # 加权原特征，输出与输入形状一致
        out = x * y

        return out


# 轻量化池化
class SEBlock_ImprovedPool_Light(nn.Module):
    def __init__(self, feat_dim, reduction=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.reduction = max(reduction, 2) if feat_dim < 8 else reduction
        mid_dim = feat_dim // self.reduction

        # 加权融合无需维度膨胀，fc1输入维度仍为c，计算量减少50%
        self.fc1 = nn.Linear(feat_dim, mid_dim, bias=False)
        self.fc2 = nn.Linear(mid_dim, feat_dim, bias=False)
        self.ln = nn.LayerNorm(mid_dim)
        # 可学习池化权重，动态平衡GAP和GMP的贡献
        self.pool_weight = nn.Parameter(torch.ones(2) / 2, requires_grad=True)

    def forward(self, x):
        b, c = x.shape
        # 计算GAP和GMP
        x_avg = torch.mean(x, dim=1, keepdim=True).expand_as(x)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_max = x_max.expand_as(x)

        # 加权融合（替代拼接）：可学习权重，无维度膨胀
        x_fused = self.pool_weight[0] * x_avg + self.pool_weight[1] * x_max

        # 生成权重（与基础版一致）
        y = self.ln(self.fc1(x_fused))
        y = F.relu(y)
        y = torch.sigmoid(self.fc2(y))

        out = x * y
        return out


# 基于 ECA-Net 思想的改进 SEBlock
class SEBlock_ECA(nn.Module):
    def __init__(self, feat_dim, gamma=2, b=1):
        super(SEBlock_ECA, self).__init__()
        self.k_size = int(abs((torch.log2(torch.tensor(feat_dim, dtype=torch.float32)) + b) / gamma))
        self.k_size = self.k_size if self.k_size % 2 == 1 else self.k_size + 1

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k_size, padding=(self.k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播：无通道降维，捕捉局部通道依赖，适配 (N, feat_dim) 二维特征
        """
        b, c = x.shape
        x_reshaped = x.unsqueeze(1)

        y = self.avg_pool(x_reshaped.transpose(1, 2)).transpose(1, 2)

        y = self.conv(y)
        y = self.sigmoid(y).squeeze(1)  # 还原为 (N, c) 形状

        out = x * y

        return out


class SEBlock_Comprehensive(nn.Module):
    def __init__(self, feat_dim, min_mid_dim=8, max_reduction=16):
        super(SEBlock_Comprehensive, self).__init__()
        # 自适应压缩比
        self.reduction = min(max(2, feat_dim // min_mid_dim), max_reduction)
        mid_dim = max(min_mid_dim, feat_dim // self.reduction)

        self.fc1 = nn.Linear(2 * feat_dim, mid_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_dim)
        self.fc2 = nn.Linear(mid_dim, feat_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(feat_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.hard_sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        b, c = x.shape

        # 池化融合（平均+最大）
        x_fused = torch.cat([x, x], dim=1)


        y = self.leaky_relu(self.bn1(self.fc1(x_fused)))
        y = self.hard_sigmoid(self.bn2(self.fc2(y)))

        out = x * y

        return out


#加权融合池化
class SEBlock_WeightedPool(nn.Module):
    def __init__(self, feat_dim, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, feat_dim // reduction, bias=False)
        self.fc2 = nn.Linear(feat_dim // reduction, feat_dim, bias=False)
        self.pool_weight = nn.Parameter(torch.ones(2) / 2, requires_grad=True)

    def forward(self, x):
        b, c = x.shape
        x_avg = torch.mean(x, dim=1, keepdim=True).expand_as(x)  # 全局均值广播
        x_max = torch.max(x, dim=1, keepdim=True)[0].expand_as(x)  # 全局最大值广播

        x_fused = self.pool_weight[0] * x_avg + self.pool_weight[1] * x_max

        y = F.relu(self.fc1(x_fused), inplace=True)
        y = torch.sigmoid(self.fc2(y))

        return x * y


# 分层池化
class SEBlock_HierarchicalPool(nn.Module):
    def __init__(self, feat_dim, reduction=4):
        super().__init__()
        self.reduction = reduction
        self.fc1 = nn.Linear(3 * feat_dim, feat_dim // reduction, bias=False)
        self.fc2 = nn.Linear(feat_dim // reduction, feat_dim, bias=False)

    def forward(self, x):
        b, c = x.shape
        # 全局平均池化（GAP）
        x_avg = torch.mean(x, dim=1, keepdim=True).expand_as(x)
        # 全局最大池化（GMP）
        x_max = torch.max(x, dim=1, keepdim=True)[0].expand_as(x)
        # 全局标准差池化（GSP）：捕捉特征分布的离散度
        x_std = torch.std(x, dim=1, keepdim=True).expand_as(x)

        x_fused = torch.cat([x_avg, x_max, x_std], dim=1)

        y = F.relu(self.fc1(x_fused), inplace=True)
        y = torch.sigmoid(self.fc2(y))

        return x * y

# 动态卷积核的 ECA
class SEBlock_DynamicECA(nn.Module):
    def __init__(self, feat_dim, gamma=2, b=1):
        super().__init__()
        self.gamma = gamma
        self.b = b
        # 可学习的卷积缩放因子
        self.conv_scale = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        b, c = x.shape
        k_size = int(abs((torch.log2(torch.tensor(c, dtype=torch.float32)) + self.b) / self.gamma))
        k_size = k_size if k_size % 2 == 1 else k_size + 1

        x_reshaped = x.unsqueeze(1)
        # 全局平均池化
        y = torch.mean(x_reshaped, dim=2, keepdim=True)

        # 动态创建 1D 卷积
        conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False).to(x.device)
        # 卷积捕捉局部通道依赖 + 缩放因子
        y = conv(y) * self.conv_scale
        y = torch.sigmoid(y).squeeze(1)

        return x * y


# 跨通道注意力
class SEBlock_CrossChannel(nn.Module):
    def __init__(self, feat_dim, reduction=4):
        super().__init__()
        self.mid_dim = feat_dim // reduction
        self.q = nn.Linear(feat_dim, self.mid_dim, bias=False)
        self.k = nn.Linear(feat_dim, self.mid_dim, bias=False)
        self.v = nn.Linear(feat_dim, feat_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c = x.shape
        Q = self.q(x)  # (N, mid_dim)
        K = self.k(x)  # (N, mid_dim)
        V = self.v(x)  # (N, c)


        attn = self.softmax(torch.bmm(Q.unsqueeze(2), K.unsqueeze(1)) / (self.mid_dim ** 0.5))  # (N, mid_dim, mid_dim)
        # 加权 Value 生成通道权重
        y = torch.bmm(V.unsqueeze(1), attn).squeeze(1)  # (N, c)
        y = torch.sigmoid(y)

        return x * y
