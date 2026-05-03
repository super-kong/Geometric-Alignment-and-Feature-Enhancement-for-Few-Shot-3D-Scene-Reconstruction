# Author: Xavier-Kong
# Date: 2025-01-01
# Email: 2534867633@qq.com


import os
import csv
import json
import yaml

REPEAT_TIMES = 3

PARAM_CHANGES = [
    {"n_views": 2},
    {"n_views": 3},
    {"n_views": 4},
    {"n_views": 5},


]
REPEAT_TIMES = 4

EXP_SETTING = ''



def update_yaml_config(config_path, new_params):

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    config.update(new_params)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"已更新YAML配置：{new_params}")


def run_single_experiment(round_idx):
    """
    执行单次训练+渲染流程
    :param round_idx: 当前训练轮次（用于区分输出目录）
    """
    print(f"\n==================== 开始第 {round_idx + 1} 轮训练 ====================\n")


    with open(EXP_SETTING, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)


    dataset = result['dataset']
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

    dataset_dir = ''
    dataset_name = ''
    scene_name = 'scan55'
    scene_dir = os.path.join(dataset_dir, dataset_name, scene_name)
    print(f"数据集路径：{scene_dir}")

    output_parent_dir = ''
    exp_name = f'DTU_55_round_{round_idx + 1}'
    scene_out_dir = os.path.join(output_parent_dir, exp_name)
    os.makedirs(scene_out_dir, exist_ok=True)
    print(f"输出路径：{scene_out_dir}")

    ''' Training '''
    pycode = (f'CUDA_VISIBLE_DEVICES={cuda_idx} python train_dtu_mask.py '
              f'-s {scene_dir} -m {scene_out_dir} --eval '
              f'--n_views {n_views} --iterations {iterations} '
              f'--depth_weight {depth_weight} --depth_pseudo_weight {depth_pseudo_weight} '
              f'--opacity_reset_interval {opacity_reset_interval} --start_sample_pseudo {start_sample_pseudo} '
              f'--sample_pseudo_interval {sample_pseudo_interval} --images images_4 '
              f'--lambda_dssim {lambda_dssim} --reset_param {reset_param} '
              f'--sh_interval {sh_interval} --densify_grad_threshold {densify_grad_threshold} --my_debug')

    if use_wsds or use_wsds_2:
        pycode += f' --guidance_scale {warp_sds_guidance_scale} --warp_sds_guidance_start_iter {warp_sds_guidance_start_iter}'
        if use_wsds:
            pycode += f' --add_warp_sds_guidance --warp_sds_pseudo_weight {warp_sds_pseudo_weight}'
        if use_wsds_2:
            pycode += f' --add_warp_sds_guidance_2 --warp_sds_pseudo_weight_2 {warp_sds_pseudo_weight_2}'

    if use_pixel:
        pycode += f' --add_pixel_guidance --pixel_guidance_start_iter {pixel_guidance_start_iter} --pixel_pseudo_weight {pixel_pseudo_weight}'

    if rand_ply:
        pycode += ' --rand_ply'

    print(f"训练命令：\n{pycode}")
    os.system(pycode)

    ''' Rendering '''
    pycode = (f'CUDA_VISIBLE_DEVICES={cuda_idx} python render.py --eval '
              f'--source_path {scene_dir} --model_path {scene_out_dir} '
              f'--iteration {iterations} --images images_{resolution} --n_views {n_views}')
    print(f"渲染命令：\n{pycode}")
    state = os.system(pycode)




if __name__ == '__main__':

    for i in range(4):
        update_yaml_config(EXP_SETTING, PARAM_CHANGES[i])
        run_single_experiment(i)
