import os
import csv
import json
import yaml


def process_single_scene(scene_name, config, base_dataset_dir, base_output_dir, exp_name):

    dataset = config['dataset']
    n_views = config['n_views']
    cuda_idx = config['cuda_idx']
    use_wsds = config['use_wsds']
    use_wsds_2 = config['use_wsds_2']
    use_pixel = config['use_pixel']
    rand_ply = config['rand_ply']
    sh_interval = config['sh_interval']
    reset_param = config['reset_param']
    densify_grad_threshold = config['densify_grad_threshold']
    depth_weight = config['depth_weight']
    depth_pseudo_weight = config['depth_pseudo_weight']
    opacity_reset_interval = config['opacity_reset_interval']
    lambda_dssim = config['lambda_dssim']
    sample_pseudo_interval = config['sample_pseudo_interval']
    pixel_pseudo_weight = config['pixel_pseudo_weight']
    warp_sds_pseudo_weight = config['warp_sds_pseudo_weight']
    warp_sds_pseudo_weight_2 = config['warp_sds_pseudo_weight_2']
    resolution = config['resolution']
    warp_sds_guidance_scale = config['warp_sds_guidance_scale']
    iterations = config['iterations']
    sd_guidance_scale = config['sd_guidance_scale']
    num_inference_steps = config['num_inference_steps']
    lora_start_iter = config['lora_start_iter']
    sd_guidance_start_iter = config['sd_guidance_start_iter']
    warp_sds_guidance_start_iter = config['warp_sds_guidance_start_iter']
    pixel_guidance_start_iter = config['pixel_guidance_start_iter']
    start_sample_pseudo = config['start_sample_pseudo']
    exp_date = config['exp_date']
    use_multi_scale_depth = config['use_multi_scale_depth']
    use_multi_frature = config['use_multi_frature']
    use_cosine = config['use_cosine']
    use_Dweight = config['use_Dweight']
    use_gaussian_init = config['use_gaussian_init']
    use_OutlierFactor = config['use_OutlierFactor']
    se_type = config['se_type']
    use_unpool = config['use_unpool']
    use_fusionnet = config['use_fusionnet']
    use_scale_constraint = config['use_scale_constraint']
    use_multi_view_consistency = config['use_multi_view_consistency']
    use_litept = config['use_litept']
    use_multi_scale_depth = config['use_multi_scale_depth']
    use_dd_drop = config['use_dd_drop']
    use_dafe = config['use_dafe']
    use_SEBlock = config['use_SEBlock']

    dataset_name = config['dataset']
    scene_dir = os.path.join(base_dataset_dir, dataset_name, scene_name)
    print(f"\n{'=' * 50}\n当前处理场景: {scene_name}\n数据集路径: {scene_dir}\n{'=' * 50}")

    scene_out_dir = os.path.join(base_output_dir, f"{exp_name}_{exp_date}", scene_name)
    os.makedirs(scene_out_dir, exist_ok=True)
    print(f"输出路径: {scene_out_dir}")

    '''
    Training
    '''
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

    print("\n训练命令:")
    print(pycode)
    os.system(pycode)

    '''
    Rendering
    '''
    pycode = (f'CUDA_VISIBLE_DEVICES={cuda_idx} python render.py '
              f'--eval --source_path {scene_dir} --model_path {scene_out_dir} '
              f'--iteration {iterations} --images images_{resolution} --n_views {n_views}')
    print("\n渲染命令:")
    print(pycode)
    state = os.system(pycode)

    '''
    Evaluation 
    '''
    mask_dir = os.path.join(scene_dir, 'mask')
    pycode = f'CUDA_VISIBLE_DEVICES={cuda_idx} python metrics_dtu.py --model_paths {scene_out_dir} --mask_paths {mask_dir}'
    print("\n评估命令:")
    print(pycode)
    state = os.system(pycode)


if __name__ == '__main__':

    exp_setting = './configs/dtu.yaml'
    base_dataset_dir = ''
    base_output_dir = ''
    exp_name = 'DTU'
    dtu_scene_list = ['scan21',  'scan30',  'scan31', 'scan34', 'scan38', 'scan40', 'scan41', 'scan45', 'scan55', 'scan63', 'scan82', 'scan103', 'scan110', 'scan114']

    with open(exp_setting, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)


    for scene in dtu_scene_list:
        process_single_scene(
            scene_name=scene,
            config=config,
            base_dataset_dir=base_dataset_dir,
            base_output_dir=base_output_dir,
            exp_name=exp_name
        )


