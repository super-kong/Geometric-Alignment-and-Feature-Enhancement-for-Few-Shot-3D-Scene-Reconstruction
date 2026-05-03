# Author: Xavier-Kong
# Date: 2025-01-01
# Email: 2534867633@qq.com


from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import torch
import json


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_VIEWS = 3
DATASET_DIR = ''
MODEL_PATH = ""
BLIP_RST_FILENAME = 'blip_rst.txt'


torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

processor = BlipProcessor.from_pretrained(MODEL_PATH)
model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()


scene_list = [
    name for name in sorted(os.listdir(DATASET_DIR))
    if os.path.isdir(os.path.join(DATASET_DIR, name))
]

if not scene_list:
    print(f"{DATASET_DIR} 下未找到任何场景文件夹")
else:
    print(f"共找到 {len(scene_list)} 个场景")

for idx, scene_name in enumerate(scene_list, 1):
    scene_dir = os.path.join(DATASET_DIR, scene_name)
    n_views_dir = os.path.join(scene_dir, f"{N_VIEWS}_views", "images")

    if not os.path.exists(n_views_dir):
        print(f"[{idx}/{len(scene_list)}] 跳过 {scene_name}：{n_views_dir} 不存在")
        continue


    img_list = [
        name for name in sorted(os.listdir(n_views_dir))
        if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]

    if len(img_list) < N_VIEWS:
        print(f"[{idx}/{len(scene_list)}] 警告 {scene_name}：图片数量({len(img_list)}) < {N_VIEWS}")
        continue


    random_img_idx = torch.randint(low=0, high=len(img_list), size=(1,), device=DEVICE)[0].item()
    img_filename = img_list[random_img_idx]
    img_path = os.path.join(n_views_dir, img_filename)

    print(f"\n[{idx}/{len(scene_list)}] 处理 {scene_name} | 随机选图：{img_filename}")

    try:
        raw_image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"跳过 {img_path}：图片加载失败，错误：{e}")
        continue

    text_prompt = "a photography of"
    inputs = processor(
        images=raw_image,
        text=text_prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=3,
            early_stopping=True
        )


    blip_rst = processor.decode(out[0], skip_special_tokens=True)
    img_name = os.path.splitext(img_filename)[0]

    blip_rst_path = os.path.join(scene_dir, BLIP_RST_FILENAME)
    try:
        with open(blip_rst_path, 'w', encoding='utf-8') as f:
            writing_content = f"random select {img_name} blip result:{blip_rst}"
            f.write(writing_content)
            print(f"结果已保存：{blip_rst_path} | 内容：{writing_content}")
    except Exception as e:
        print(f"保存 {blip_rst_path} 失败，错误：{e}")

print("\n所有场景处理完成！")
