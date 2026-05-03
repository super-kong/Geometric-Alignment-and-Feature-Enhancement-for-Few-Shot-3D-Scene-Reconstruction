import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ====================== 【仅修改这2个路径即可】 ======================
# 1. 你的DTU数据集根目录（解压后的dtu_training文件夹路径）

# 2. 你想要读取的图片完整路径（直接复制粘贴）
TARGET_IMAGE_PATH = r"/home/sari/PycharmProjects/IPSM/dataset/dtu/scan21/images/rect_026_3_r5000.png"


# =====================================================================

def read_pfm(pfm_path):
    """读取DTU深度图（PFM格式），返回深度numpy数组（单位：毫米）"""
    with open(pfm_path, 'rb') as f:
        header = f.readline().decode().strip()
        w, h = map(int, f.readline().split())
        scale = float(f.readline().strip())

        data = np.fromfile(f, '<f' if scale < 0 else '>f')
        data = data.reshape(h, w)
        depth = np.flipud(data)  # 修正坐标，匹配原图
    return depth


# ---------------------- 自动匹配深度图（无需手动计算） ----------------------
# 解析图片文件名，自动提取场景号(scan)、视角号(view)
img_name = os.path.basename(TARGET_IMAGE_PATH)  # 例：rect_001_0_r5000.png
scan_id = img_name.split("_")[1]  # 例：001 → scan1
view_id = img_name.split("_")[2]  # 例：0 → 视角0

# 拼接对应深度图路径（标准DTU目录结构）
depth_path = r"/home/sari/PycharmProjects/IPSM/dataset/dtu/scan21/depth_maps/depth_rect_026_3_r5000.pfm"

# ---------------------- 读取数据 ----------------------
# 读取目标图片
img = cv2.imread(TARGET_IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 读取对应深度数据（核心！直接拿到深度数值）
depth_data = read_pfm(depth_path)
print(f"✅ 成功读取深度数据！")
print(f"深度尺寸: {depth_data.shape}")
print(f"深度范围: {np.min(depth_data):.2f} ~ {np.max(depth_data):.2f} 毫米")

# ---------------------- 生成彩色深度图 ----------------------
# 归一化深度值，转为彩色图
depth_norm = (depth_data - np.min(depth_data)) / (np.max(depth_data) - np.min(depth_data))
depth_norm = (depth_norm * 255).astype(np.uint8)
depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)  # 彩色深度图

# ---------------------- 显示 + 保存 ----------------------
plt.figure(figsize=(12, 5))
plt.subplot(121), plt.imshow(img), plt.title("原始图片"), plt.axis('off')
plt.subplot(122), plt.imshow(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)), plt.title("深度图"), plt.axis('off')
plt.show()

# 保存结果（自动保存在代码同级目录）
cv2.imwrite("my_depth_map.png", depth_color)
np.save("my_depth_data.npy", depth_data)  # 保存原始深度数据（可用于后续计算）
print(f"\n✅ 深度图已保存：my_depth_map.png")
print(f"✅ 原始深度数据已保存：my_depth_data.npy")
