import os
import csv
import json
import yaml
import datetime
import subprocess
import time
from utils.metrics_output import parse_metrics_output
from utils.clean_gpu_memory import clean_gpu_memory


if __name__ == '__main__':
    exp_setting = ''
    with open(exp_setting, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    n_views = result['n_views']
    cuda_idx = result['cuda_idx']
    use_wsds = result['use_wsds']
    use_wsds_2 = result['use_wsds_2']
    use_pixel = result['use_pixel']
    rand_ply = result['rand_ply']
    sh_interval = result['sh_interval']
    reset_param = result['reset_param']
    densify_grad_threshold = result['densify_grad_threshold']
    depth_weight = result['depth_weight']
    depth_pseudo_weight = result['depth_pseudo_weight']
    opacity_reset_interval = result['opacity_reset_interval']
    lambda_dssim = result['lambda_dssim']
    sample_pseudo_interval = result['sample_pseudo_interval']
    pixel_pseudo_weight = result['pixel_pseudo_weight']
    warp_sds_pseudo_weight = result['warp_sds_pseudo_weight']
    warp_sds_pseudo_weight_2 = result['warp_sds_pseudo_weight_2']
    resolution = result['resolution']
    warp_sds_guidance_scale = result['warp_sds_guidance_scale']
    iterations = result['iterations']
    sd_guidance_scale = result['sd_guidance_scale']
    num_inference_steps = result['num_inference_steps']
    lora_start_iter = result['lora_start_iter']
    sd_guidance_start_iter = result['sd_guidance_start_iter']
    warp_sds_guidance_start_iter = result['warp_sds_guidance_start_iter']
    pixel_guidance_start_iter = result['pixel_guidance_start_iter']
    start_sample_pseudo = result['start_sample_pseudo']
    use_litept = result['use_litept']
    use_multi_scale_depth = result['use_multi_scale_depth']
    use_dd_drop = result['use_dd_drop']
    use_dafe = result['use_dafe']
    exp_date = result['exp_date']
    exp_batch = result['exp_batch']
    use_SEBlock = result['use_SEBlock']
    dateset_dir = result['dateset_dir']
    dataset_name = result['dataset']
    output_parent_dir = result['output_parent_dir']
    exp_name = result['dataset']
    use_multi_scale_depth = result['use_multi_scale_depth']
    use_multi_frature = result['use_multi_frature']
    use_cosine = result['use_cosine']
    use_Dweight = result['use_Dweight']
    use_gaussian_init = result['use_gaussian_init']
    use_OutlierFactor = result['use_OutlierFactor']
    use_debug = result['use_debug']
    se_type = result['se_type']
    use_unpool = result['use_unpool']
    use_fusionnet = result['use_fusionnet']
    use_scale_constraint = result['use_scale_constraint']
    use_multi_view_consistency = result['use_multi_view_consistency']


    all_scene_names = ['trex', 'fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room']
    test_scene_names = ['trex'] # 针对特定场景做测试
    scene_names = test_scene_names


    all_scene_metrics = {}
    # exp_date = datetime.datetime.now().strftime('%Y%m%d')
    # exp_date = '20260110'

    # ====================== Batch process scenes ======================
    for scene_name in scene_names:
        print(f"\n===================== Starting to process scene: {scene_name} =====================")

        # if scene_name == 'leaves':
        #     se_type = 'base'
        # else:
        #     se_type = result['se_type']


        scene_dir = os.path.join(dateset_dir, dataset_name, scene_name)

        scene_out_dir = os.path.join(
            output_parent_dir,
            f"{exp_date}",
            f"{exp_batch}",
            f"{exp_name}_20{exp_date}_{scene_name}"
        )
        os.makedirs(scene_out_dir, exist_ok=True)

        print(f"\n[Training] Scene: {scene_name}")

        train_start_time = time.time()

        pycode = (
            f'CUDA_VISIBLE_DEVICES={str(cuda_idx)} python train.py '
            f'-s {str(scene_dir)} -m {str(scene_out_dir)} --eval '
            f'--n_views {str(n_views)} --iterations {str(iterations)} '
            f'--depth_weight {str(depth_weight)} --depth_pseudo_weight {str(depth_pseudo_weight)} '
            f'--opacity_reset_interval {str(opacity_reset_interval)} --start_sample_pseudo {str(start_sample_pseudo)} '
            f'--sample_pseudo_interval {str(sample_pseudo_interval)} --images images_{str(resolution)} '
            f'--lambda_dssim {str(lambda_dssim)} --reset_param {str(reset_param)} '
            f'--sh_interval {str(sh_interval)} --densify_grad_threshold {str(densify_grad_threshold)} --my_debug'
        )
        if use_wsds or use_wsds_2:
            pycode += (
                f' --guidance_scale {warp_sds_guidance_scale} '
                f'--warp_sds_guidance_start_iter {warp_sds_guidance_start_iter}'
            )
            if use_wsds:
                pycode += f' --add_warp_sds_guidance --warp_sds_pseudo_weight {warp_sds_pseudo_weight}'
            if use_wsds_2:
                pycode += f' --add_warp_sds_guidance_2 --warp_sds_pseudo_weight_2 {warp_sds_pseudo_weight_2}'

        if use_pixel:
            pycode += (
                f' --add_pixel_guidance --pixel_guidance_start_iter {pixel_guidance_start_iter} '
                f'--pixel_pseudo_weight {pixel_pseudo_weight}'
            )

        if rand_ply:
            pycode += ' --rand_ply'

        if use_litept:
            pycode += f' --use_litept'

        if use_multi_scale_depth:
            pycode += f' --use_multi_scale_depth'

        if use_dd_drop:
            pycode += f' --use_dd_drop '
        if use_dafe:
            pycode += f' --use_dafe'
        if use_SEBlock:
            pycode += (
                f' --use_SEBlock'
                f' --se_type {se_type}'
            )
        if use_debug:
            pycode += f' --use_debug'

        if use_gaussian_init:
            pycode += f' --use_gaussian_init'

        if use_OutlierFactor:
            pycode += f' --use_OutlierFactor'

        if use_Dweight:
            pycode += f' --use_Dweight'

        if use_cosine:
            pycode += f' --use_cosine'

        if use_unpool:
            pycode += f' --use_unpool'

        if use_fusionnet:
            pycode += f' --use_fusionnet'

        if use_scale_constraint:
            pycode += f' --use_scale_constraint'

        if use_multi_view_consistency:
            pycode += f' --use_multi_view_consistency'


        print(f"Training command: {pycode}")

        train_state = os.system(pycode)

        train_elapsed_time = time.time() - train_start_time
        train_time_formatted = str(datetime.timedelta(seconds=round(train_elapsed_time)))

        print(
            f"[Training Time] Scene {scene_name} took {train_time_formatted} (HH:MM:SS) / {round(train_elapsed_time, 2)} seconds")


        if train_state != 0:
            print(f"Scene {scene_name} training failed, skipping subsequent steps")
            all_scene_metrics[scene_name] = {
                "Status": "Training failed",
                "Training Time (s)": round(train_elapsed_time, 2),
                "Training Time (HH:MM:SS)": train_time_formatted
            }
            continue

        all_scene_metrics[scene_name] = {
            "Status": "Training success",
            "Training Time (s)": round(train_elapsed_time, 2),
            "Training Time (HH:MM:SS)": train_time_formatted,
            "Use LitePT": use_litept,
            "Use Multi-Scale Depth": use_multi_scale_depth
        }

        print(f"\n[Rendering] Scene: {scene_name}")
        render_code = (
            f'CUDA_VISIBLE_DEVICES={str(cuda_idx)} python render.py '
            f'--eval --source_path {scene_dir} --model_path {scene_out_dir} '
            f'--iteration {str(iterations)} --images images_{str(resolution)} --n_views {str(n_views)}'
        )
        print(f"Rendering command: {render_code}")
        render_state = os.system(render_code)
        if render_state != 0:
            print(f"Scene {scene_name} rendering failed, skipping evaluation steps")
            all_scene_metrics[scene_name] = {"Status": "Rendering failed"}
            continue

        print(f"\n[Evaluation] Scene: {scene_name}")
        metrics_code = (
            f'CUDA_VISIBLE_DEVICES={str(cuda_idx)} python metrics.py '
            f'--source_path {scene_dir}  --model_path  {scene_out_dir} --iteration {str(iterations)}'
        )
        print(f"Evaluation command: {metrics_code}")
        try:
            metrics_result = subprocess.check_output(
                metrics_code, shell=True, stderr=subprocess.STDOUT, text=True
            )
            print(f"Evaluation output:\n{metrics_result}")
            scene_metrics = parse_metrics_output(metrics_result)
            scene_metrics["Status"] = "Success"
            # 保存训练耗时
            scene_metrics["Training Time (s)"] = all_scene_metrics[scene_name]["Training Time (s)"]
            scene_metrics["Training Time (HH:MM:SS)"] = all_scene_metrics[scene_name]["Training Time (HH:MM:SS)"]


            all_scene_metrics[scene_name] = scene_metrics
        except subprocess.CalledProcessError as e:
            print(f"Scene {scene_name} evaluation failed: {e.output}")
            all_scene_metrics[scene_name] = {"Status": "Evaluation failed", "Error Message": e.output}
    clean_gpu_memory()

    # ====================== Summarize and save metrics for all scenes ======================
    print("\n===================== Summary of evaluation results for all scenes =====================")
    summary_output_parent_dir = os.path.join(
            output_parent_dir,
            f"{exp_date}",
            f"{exp_batch}"
    )
    summary_dir = os.path.join(summary_output_parent_dir, f"{exp_name}_{exp_date}_summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_json_path = os.path.join(summary_dir, "all_scene_metrics.json")
    summary_csv_path = os.path.join(summary_dir, "all_scene_metrics.csv")

    # Save detailed results for all scenes (JSON)
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_scene_metrics, f, indent=4, ensure_ascii=False)
    print(f"JSON summary file saved to: {summary_json_path}")

    # Save detailed results for all scenes (CSV)
    all_keys = set()
    for metrics in all_scene_metrics.values():
        all_keys.update(metrics.keys())
    all_keys = sorted(list(all_keys))
    with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Scene Name'] + all_keys)
        writer.writeheader()
        for scene_name, metrics in all_scene_metrics.items():
            row = {"Scene Name": scene_name}
            row.update(metrics)
            writer.writerow(row)

        # Add average row in CSV
        successful_scenes = [m for m in all_scene_metrics.values() if m.get("Status") == "Success"]
        if successful_scenes:
            avg_row = {"Scene Name": "Average"}
            for key in all_keys:
                if key in ['PSNR', 'SSIM', 'LPIPS']:  # Only calculate average for numerical metrics
                    avg_value = sum(m.get(key, 0) for m in successful_scenes) / len(successful_scenes)
                    avg_row[key] = round(avg_value, 4)  # Keep 4 decimal places

                elif key == "Training Time (s)":  # 计算成功场景的平均训练时间
                    avg_train_time = sum(m.get(key, 0) for m in successful_scenes) / len(successful_scenes)
                    avg_row[key] = round(avg_train_time, 2)
                    avg_row["Training Time (HH:MM:SS)"] = str(datetime.timedelta(seconds=round(avg_train_time)))

                elif key == "Status":
                    avg_row[key] = f"{len(successful_scenes)}/{len(scene_names)}"  # Successful/Total scenes
                else:
                    avg_row[key] = "-"
            writer.writerow(avg_row)
    print(f"CSV summary file saved to: {summary_csv_path}")

# ===================== Starting to execute eval.py =====================
    print("\n===================== Starting to execute eval.py =====================")
    eval_code = (
        f'CUDA_VISIBLE_DEVICES={str(cuda_idx)} python eval.py '
        f'--summary_dir {summary_dir} '  
        f'--exp_name {exp_name} '  
        f'--exp_date {exp_date}'
    )

    print(f"eval.py execution command: {eval_code}")

    try:
        eval_result = subprocess.check_output(
            eval_code, shell=True, stderr=subprocess.STDOUT, text=True
        )
        print(f"eval.py execution completed successfully! Output:\n{eval_result}")
    except subprocess.CalledProcessError as e:
        print(f"eval.py execution failed with error: {e.output}")
    except FileNotFoundError:
        print("Error: eval.py file not found! Please check the file path.")
