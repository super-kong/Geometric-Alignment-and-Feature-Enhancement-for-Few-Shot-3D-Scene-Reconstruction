# Author: Xavier-Kong
# Date: 2025-01-01
# Email: 2534867633@qq.com


import os
import re
import numpy as np
import json
import subprocess
import yaml


def parse_metrics_from_scene(scene_dir, model_path, iteration, cuda_idx):
    """
    重新运行metrics.py获取单个场景的指标
    """
    metrics_code = (
        f'CUDA_VISIBLE_DEVICES={str(cuda_idx)} python metrics.py '
        f'--source_path {scene_dir}  --model_path {model_path} --iteration {str(iteration)}'
    )
    try:

        metrics_result = subprocess.check_output(
            metrics_code, shell=True, stderr=subprocess.STDOUT, text=True
        )

        print(f"\n【{scene_name}】metrics.py原始输出：")
        print("-" * 40)
        print(metrics_result)
        print("-" * 40)


        metrics = {
            'PSNR': None,
            'SSIM': None,
            'LPIPS': None
        }


        psnr_patterns = [
            r'PSNR[:=：]\s*([\d\.]+)',
            r'Average\s+PSNR[:=：]\s*([\d\.]+)',
            r'psnr\s*:\s*([\d\.]+)',
            r'峰值信噪比[:=：]\s*([\d\.]+)',
            r'PSNR\s+=\s*([\d\.]+)'
        ]

        for idx, pattern in enumerate(psnr_patterns):
            psnr_match = re.search(pattern, metrics_result, re.IGNORECASE)
            if psnr_match:
                try:
                    metrics['PSNR'] = float(psnr_match.group(1))
                    print(f"匹配到PSNR: {metrics['PSNR']}（规则{idx + 1}：{pattern}）")
                    break
                except ValueError:
                    continue
        if metrics['PSNR'] is None:
            print(f"未匹配到PSNR")


        ssim_patterns = [
            r'SSIM[:=：]\s*([\d\.]+)',
            r'Average\s+SSIM[:=：]\s*([\d\.]+)',
            r'SSIM\s*\(RGB\)[:=：]\s*([\d\.]+)',
            r'ssim\s*:\s*([\d\.]+)',
            r'结构相似性[:=：]\s*([\d\.]+)',
            r'SSIM\s+=\s*([\d\.]+)'
        ]

        for idx, pattern in enumerate(ssim_patterns):
            ssim_match = re.search(pattern, metrics_result, re.IGNORECASE)
            if ssim_match:
                try:
                    metrics['SSIM'] = float(ssim_match.group(1))
                    print(f"匹配到SSIM: {metrics['SSIM']}（规则{idx + 1}：{pattern}）")
                    break
                except ValueError:
                    continue
        if metrics['SSIM'] is None:
            print(f"未匹配到SSIM")


        lpips_patterns = [
            r'LPIPS[:=：]\s*([\d\.]+)',
            r'LPIPS\s+distance[:=：]\s*([\d\.]+)',
            r'lpips\s*:\s*([\d\.]+)',
            r'LPIPS\s+=\s*([\d\.]+)'
        ]
        for idx, pattern in enumerate(lpips_patterns):
            lpips_match = re.search(pattern, metrics_result, re.IGNORECASE)
            if lpips_match:
                try:
                    metrics['LPIPS'] = float(lpips_match.group(1))
                    print(f"匹配到LPIPS: {metrics['LPIPS']}（规则{idx + 1}：{pattern}）")
                    break
                except ValueError:
                    continue
        if metrics['LPIPS'] is None:
            print(f"未匹配到LPIPS")

        return metrics, "成功"

    except subprocess.CalledProcessError as e:
        error_info = f"执行失败: {e.returncode} | 输出: {e.output}"
        print(f"场景 {scene_name} metrics.py执行失败：{error_info}")
        return {"PSNR": None, "SSIM": None, "LPIPS": None}, error_info
    except Exception as e:
        error_info = f"未知错误: {str(e)}"
        print(f"场景 {scene_name} 解析失败：{error_info}")
        return {"PSNR": None, "SSIM": None, "LPIPS": None}, error_info


def calculate_overall_average_from_scenes():
    """
    遍历所有场景目录，计算总体平均指标
    """

    exp_setting = ''
    with open(exp_setting, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)


    dataset_dir = result['dateset_dir']
    output_parent_dir = result['output_parent_dir']
    exp_date =  result['exp_date']
    exp_name = result['dataset']
    exp_batch = result['exp_batch']
    output_parent_dir = os.path.join(
        output_parent_dir,
        f"{exp_date}",
        f"{exp_batch}",
    )


    scene_names = ['trex', 'fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room']
    iteration = result['iterations']
    cuda_idx = 0


    psnr_list = []
    ssim_list = []
    lpips_list = []
    scene_details = {}


    global scene_name
    for scene_name in scene_names:
        print(f"\n{'=' * 50}")
        print(f"处理场景: {scene_name}")
        print(f"{'=' * 50}")

        scene_dir = os.path.join(dataset_dir, scene_name)
        model_path = os.path.join(output_parent_dir, f"{exp_name}_20{exp_date}_{scene_name}")

        if not os.path.exists(model_path):
            scene_details[scene_name] = {
                "状态": "模型目录不存在",
                "PSNR": None,
                "SSIM": None,
                "LPIPS": None,
                "备注": f"路径 {model_path} 不存在"
            }
            print(f"场景 {scene_name} 模型目录不存在")
            continue


        metrics, status = parse_metrics_from_scene(scene_dir, model_path, iteration, cuda_idx)
        scene_details[scene_name] = {
            "状态": status,
            "PSNR": metrics['PSNR'],
            "SSIM": metrics['SSIM'],
            "LPIPS": metrics['LPIPS'],
            "备注": "解析成功" if "成功" in status else "解析失败"
        }


        if isinstance(metrics['PSNR'], (int, float)) and not np.isnan(metrics['PSNR']):
            psnr_list.append(metrics['PSNR'])
            print(f"场景 {scene_name} PSNR加入统计: {metrics['PSNR']}")
        if isinstance(metrics['SSIM'], (int, float)) and not np.isnan(metrics['SSIM']):
            ssim_list.append(metrics['SSIM'])
            print(f"场景 {scene_name} SSIM加入统计: {metrics['SSIM']}")
        if isinstance(metrics['LPIPS'], (int, float)) and not np.isnan(metrics['LPIPS']):
            lpips_list.append(metrics['LPIPS'])
            print(f"场景 {scene_name} LPIPS加入统计: {metrics['LPIPS']}")


    success_scenes = [name for name, detail in scene_details.items() if "成功" in detail["状态"]]
    # 统计各指标有效数据数
    psnr_valid_count = len(psnr_list)
    ssim_valid_count = len(ssim_list)
    lpips_valid_count = len(lpips_list)

    overall_metrics = {
        "所有场景数": len(scene_names),
        "成功执行metrics的场景数": len(success_scenes),
        "PSNR有效数据数": psnr_valid_count,
        "SSIM有效数据数": ssim_valid_count,
        "LPIPS有效数据数": lpips_valid_count,
        "总体平均PSNR": np.mean(psnr_list) if psnr_valid_count > 0 else None,
        "总体平均SSIM": np.mean(ssim_list) if ssim_valid_count > 0 else None,
        "总体平均LPIPS": np.mean(lpips_list) if lpips_valid_count > 0 else None,
        "各场景详细指标": scene_details,
        "各指标原始数据": {
            "PSNR": psnr_list,
            "SSIM": ssim_list,
            "LPIPS": lpips_list
        }
    }


    print("\n" + "=" * 60)
    print("          所有场景总体平均指标计算结果")
    print("=" * 60)
    print(f"总场景数: {overall_metrics['所有场景数']}")
    print(f"成功执行metrics的场景数: {overall_metrics['成功执行metrics的场景数']}")
    print(f"PSNR有效数据数: {psnr_valid_count} / {len(success_scenes)}")
    print(f"SSIM有效数据数: {ssim_valid_count} / {len(success_scenes)}")
    print(f"LPIPS有效数据数: {lpips_valid_count} / {len(success_scenes)}")
    print("-" * 60)


    if overall_metrics["总体平均PSNR"] is not None:
        print(f"总体平均PSNR: {overall_metrics['总体平均PSNR']:.2f} (±{np.std(psnr_list):.2f})")
    else:
        print("总体平均PSNR: 无有效数据")

    if overall_metrics["总体平均SSIM"] is not None:
        print(f"总体平均SSIM: {overall_metrics['总体平均SSIM']:.4f} (±{np.std(ssim_list):.4f})")
    else:
        print("总体平均SSIM: 无有效数据")

    if overall_metrics["总体平均LPIPS"] is not None:
        print(f"总体平均LPIPS: {overall_metrics['总体平均LPIPS']:.4f} (±{np.std(lpips_list):.4f})")
    else:
        print("总体平均LPIPS: 无有效数据")
    print("=" * 60)


    save_dir = os.path.join(output_parent_dir, f"{exp_name}_20{exp_date}_summary")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "overall_average_metrics.json")


    def format_float(obj):
        if isinstance(obj, float):
            return round(obj, 4)
        elif isinstance(obj, list):
            return [format_float(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: format_float(v) for k, v in obj.items()}
        else:
            return obj

    formatted_metrics = format_float(overall_metrics)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_metrics, f, indent=4, ensure_ascii=False)

    csv_path = os.path.join(save_dir, "scene_metrics_summary.csv")
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        writer.writerow(["场景名", "状态", "PSNR", "SSIM", "LPIPS", "备注"])

        for name, detail in scene_details.items():
            writer.writerow([
                name,
                detail["状态"],
                detail["PSNR"] if detail["PSNR"] is not None else "",
                detail["SSIM"] if detail["SSIM"] is not None else "",
                detail["LPIPS"] if detail["LPIPS"] is not None else "",
                detail["备注"]
            ])

    print(f"\n增强版平均指标已保存至: {save_path}")
    print(f"CSV格式汇总已保存至: {csv_path}")

    return overall_metrics


if __name__ == "__main__":

    final_metrics = calculate_overall_average_from_scenes()
