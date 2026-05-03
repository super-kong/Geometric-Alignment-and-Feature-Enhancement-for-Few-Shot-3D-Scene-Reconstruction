import torch
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel
import cv2
from tqdm import tqdm
from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path
from utils.general_utils import vis_depth
from PIL import Image


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    print(f"iteration: {str(iteration)}")

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    print(render_path)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering["render"], os.path.join(render_path, view.image_name + '.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        if args.render_depth:
            print(f"render_depth: {args.render_depth}")
            depth_map = vis_depth(rendering['depth'][0].detach().cpu().numpy())
            np.save(os.path.join(render_path, view.image_name + '_depth.npy'), rendering['depth'][0].detach().cpu().numpy())
            cv2.imwrite(os.path.join(render_path, view.image_name + '_depth.png'), depth_map)
            print(os.path.join(render_path, view.image_name + '_depth.png'))



def render_video(source_path, model_path, iteration, views, gaussians, pipeline, background, fps=30):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    makedirs(render_path, exist_ok=True)
    view = views[0]

    if source_path.find('LLFF') != -1 or source_path.find('LLFF') != -1:
        render_poses = generate_spiral_path(np.load(source_path + '/poses_bounds.npy'))
    elif source_path.find('360') != -1:
        render_poses = generate_ellipse_path(views)

    size = (view.original_image.shape[2], view.original_image.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)

    for idx, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        view.world_view_transform = torch.tensor(getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)).transpose(0, 1).cuda()
        view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        view.camera_center = view.world_view_transform.inverse()[3, :3]
        rendering = render(view, gaussians, pipeline, background)

        img = torch.clamp(rendering["render"], min=0., max=1.)
        torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        video_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
        final_video.write(video_img)

    final_video.release()



def render_sets(dataset : ModelParams, pipeline : PipelineParams, args):

    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
        if args.render_all_imgs:
            scene.my_test(args, gaussians, load_iteration=args.iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if args.video:
            render_video(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTestCameras(),
                         gaussians, pipeline, background, args.fps)

        if args.render_all_imgs:
            render_set(dataset.model_path, "eval", scene.loaded_iter, scene.getEvalCameras(), gaussians, pipeline, background, args)
        if not args.skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)
        if not args.skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)



if __name__ == "__main__":
    torch.manual_seed(0)
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--render_all_imgs", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--render_depth", action="store_true", default=True)
    parser.add_argument("--rand_ply", action="store_true")
    parser.add_argument('--use_litept', action='store_true', default=False,
                        help='Enable LitePT feature fusion in Gaussian Model (default: False)')
    parser.add_argument('--use_multi_scale_depth', action='store_true', default=False, help='Using multi scale depth') # 使用多尺度深度损失
    parser.add_argument('--use_dd_drop', action='store_true', default=False, help='启用DDDrop高斯点丢弃策略')
    parser.add_argument('--use_dafe', action='store_true', default=False, help='远景增强')
    parser.add_argument('--use_SEBlock', action='store_true', default=False, help='通道注意力')
    parser.add_argument('--se_type', type=str, default="base", help='通道注意力机制类型')
    parser.add_argument('--use_multi_frature', action='store_true', default=False, help='多尺度特征')
    parser.add_argument('--use_cosine', action='store_true', default=False, help='余弦退火')
    parser.add_argument('--use_Dweight', action='store_true', default=False, help='动态权重调整')
    parser.add_argument('--use_OutlierFactor', action='store_true', default=False, help='离群点筛选')
    parser.add_argument('--use_gaussian_init', action='store_true', default=False, help='高斯初始化')
    parser.add_argument('--use_unpool', action='store_true', default=False, help='高斯反池化')
    parser.add_argument('--use_fusionnet', action='store_true', default=False, help='可学习的特征融合门控 (Learnable Gated Fusion)' )
    parser.add_argument('--use_scale_constraint', action='store_true', default=False, help='深度一致性缩放约束')
    parser.add_argument('--use_multi_view_consistency', action='store_true', default=False,help='多视图投影一致性正则化')
    parser.add_argument('--use_debug', action='store_true', default=False, help='debug')

    args = parser.parse_args()

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args)