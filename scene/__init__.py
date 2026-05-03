#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model_with_litept import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.pose_utils import generate_random_poses_llff_annealing_view, generate_random_poses_dtu_annealing_view
from scene.cameras import PseudoCamera
import open3d as o3d
import os
from .dd_drop import DDDrop
from .DAFE import DAFE

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}
        self.closest_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # print(args.source_path.find('LLFF'))
            # if args.source_path.find('LLFF') != -1:
            #     scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, 'LLFF', args.n_views, rand_ply=args.rand_ply)
            #     print(scene_info)
            #     # 在 scene/__init__.py 中 Scene.__init__ 方法里，添加调试代码（在读取PLY后）
            #     # 找到读取PLY的代码位置（通常在sceneLoadTypeCallbacks["Colmap"]里）
            #     ply_path = scene_info.ply_path
            #     print(f"PLY文件路径: {ply_path}")
            #     print(f"PLY文件是否存在: {os.path.exists(ply_path)}")
            #
            #     # 读取PLY并打印点云数量
            #     if os.path.exists(ply_path):
            #         pcd = o3d.io.read_point_cloud(ply_path)
            #         points = np.asarray(pcd.points)
            #         print(f"PLY文件中点云数量: {len(points)}")
            #     else:
            #         print("ERROR: PLY文件不存在！")
            #
            # else:
            #     print("LLFF空")
            if args.source_path.find('dtu') != -1:
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, 'dtu', args.n_views, rand_ply=args.rand_ply)
                print(f"source path:{args.source_path}")
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views)
        else:
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

            pseudo_cams = []
            closest_cams = []
            if args.source_path.find('LLFF') != -1:
                pseudo_poses, closest_poses = generate_random_poses_llff_annealing_view(self.train_cameras[resolution_scale])
            elif args.source_path.find('dtu') != -1:
                pseudo_poses, closest_poses = generate_random_poses_dtu_annealing_view(self.train_cameras[resolution_scale])
                
            view = self.train_cameras[resolution_scale][0]
            idx = 0
            for pose in pseudo_poses:
                pseudo_cams.append(PseudoCamera(
                    R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
                    width=view.image_width, height=view.image_height
                ))
                closest_cams.append(closest_poses[idx])
                idx += 1
            
            self.pseudo_cameras[resolution_scale] = pseudo_cams
            self.closest_cameras[resolution_scale] = closest_cams

        if self.loaded_iter:
            print(self.loaded_iter)
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            print(f"model path: {self.model_path}")
            # self.gaussians.load_ply(os.path.join(self.model_path,
            #                                                "point_cloud",
            #                                                "iteration_2000",
            #                                                "point_cloud.ply"))
        else:
            # print(f"scene info:{scene_info.point_cloud}")
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def my_test(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.eval_cameras = {}
        self.pseudo_cameras = {}
        self.closest_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.source_path.find('LLFF') != -1:
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, 'LLFF', args.n_views, rand_ply=args.rand_ply)
            if args.source_path.find('dtu') != -1:
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, 'dtu', args.n_views, rand_ply=args.rand_ply)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.n_views)
        else:
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
                
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(self.cameras_extent, 'cameras_extent')

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Loading Eval Cameras")
            self.eval_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.eval_cameras, resolution_scale, args)

            pseudo_cams = []
            closest_cams = []
            if args.source_path.find('LLFF') != -1:
                pseudo_poses, closest_poses = generate_random_poses_llff_annealing_view(self.train_cameras[resolution_scale])
            elif args.source_path.find('dtu') != -1:
                pseudo_poses, closest_poses = generate_random_poses_dtu_annealing_view(self.train_cameras[resolution_scale])
                
            view = self.train_cameras[resolution_scale][0]
            idx = 0
            for pose in pseudo_poses:
                pseudo_cams.append(PseudoCamera(
                    R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
                    width=view.image_width, height=view.image_height
                ))
                closest_cams.append(closest_poses[idx])
                idx += 1
            
            self.pseudo_cameras[resolution_scale] = pseudo_cams
            self.closest_cameras[resolution_scale] = closest_cams

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getEvalCameras(self, scale=1.0):
        return self.eval_cameras[scale]

    def getPseudoCameras(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale]

    def getPseudoCamerasWithClosestViews(self, scale=1.0):
        if len(self.pseudo_cameras) == 0:
            return [None]
        else:
            return self.pseudo_cameras[scale], self.closest_cameras[scale]
