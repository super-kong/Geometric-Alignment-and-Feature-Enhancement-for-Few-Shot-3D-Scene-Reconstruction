from PIL import Image, ImageDraw
import numpy as np

def draw_dashed_rect(draw, region, dash_len=10, gap_len=5, outline="red", width=2):
    """
    绘制虚线矩形（辅助函数）
    :param draw: ImageDraw对象
    :param region: 矩形坐标 (左, 上, 右, 下)
    :param dash_len: 虚线每段长度
    :param gap_len: 虚线间隔长度
    :param outline: 虚线颜色
    :param width: 虚线宽度
    """
    left, top, right, bottom = region

    # 绘制顶边
    x = left
    while x < right:
        x_end = min(x + dash_len, right)
        draw.line([(x, top), (x_end, top)], fill=outline, width=width)
        x += dash_len + gap_len

    # 绘制底边
    x = left
    while x < right:
        x_end = min(x + dash_len, right)
        draw.line([(x, bottom), (x_end, bottom)], fill=outline, width=width)
        x += dash_len + gap_len

    # 绘制左边
    y = top
    while y < bottom:
        y_end = min(y + dash_len, bottom)
        draw.line([(left, y), (left, y_end)], fill=outline, width=width)
        y += dash_len + gap_len

    # 绘制右边
    y = top
    while y < bottom:
        y_end = min(y + dash_len, bottom)
        draw.line([(right, y), (right, y_end)], fill=outline, width=width)
        y += dash_len + gap_len

def weighted_blend_custom_region(
    img1_path,
    img2_path,
    region,
    alpha=0.5,
    dash_color="red",    # 虚线颜色
    dash_len=12,         # 虚线每段长度
    dash_gap=6,          # 虚线间隔
    dash_width=3,        # 虚线粗细
    output_path="blended_result.jpg"
):
    """
    两张图片 指定小区域 加权叠加 + 虚线轮廓标注
    """
    # 1. 打开并统一图片尺寸
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    img2 = img2.resize(img1.size)
    width, height = img1.size

    # 2. 校验叠加区域
    left, top, right, bottom = region
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if left >= right or top >= bottom:
        raise ValueError("叠加区域坐标无效！")
    safe_region = (left, top, right, bottom)

    # 3. 像素级加权混合（核心）
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    arr1[top:bottom, left:right] = arr1[top:bottom, left:right] * alpha + arr2[top:bottom, left:right] * (1 - alpha)
    blended_img = Image.fromarray(np.uint8(arr1))

    # 4. 【新增】绘制虚线轮廓
    draw = ImageDraw.Draw(blended_img)
    draw_dashed_rect(
        draw,
        safe_region,
        dash_len=dash_len,
        gap_len=dash_gap,
        outline=dash_color,
        width=dash_width
    )


    blended_img.save(output_path)
    blended_img.show()


#  region=(285, 115, 395, 275), 右边盒子
#  region=(5, 160, 100, 275),  左边盒子
#  region=(246, 120, 278, 170), 南瓜上面
#  region=(200, 90, 275, 170),



if __name__ == "__main__":
    weighted_blend_custom_region(
        img1_path="/home/sari/桌面/Vis_ablation/nothing2.png",
        img2_path="/home/sari/桌面/Vis_ablation/worstbase.png",
        # 自定义叠加区域：(左, 上, 右, 下) 像素坐标
        region=(200, 90, 275, 170),
        alpha=0,                # 加权比例
        dash_color="blue",         # 虚线颜色（支持 white/blue/green 等）
        dash_len=12,              # 虚线线段长度
        dash_gap=6,               # 虚线间隔大小
        dash_width=3,             # 虚线粗细
        output_path="/home/sari/桌面/Vis_ablation/nothing3.png"
    )
