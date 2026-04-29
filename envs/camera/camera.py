import sapien.core as sapien
import numpy as np
import pdb
from PIL import Image, ImageColor
import open3d as o3d
import json
import transforms3d as t3d
import cv2
import torch
import yaml
import trimesh
import math
from .._GLOBAL_CONFIGS import CONFIGS_PATH
import os
from sapien.sensor import StereoDepthSensor, StereoDepthSensorConfig

try:
    import pytorch3d.ops as torch3d_ops

    def fps(points, num_points=1024, use_cuda=True):
        K = [num_points]
        if use_cuda:
            points = torch.from_numpy(points).cuda()
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.cpu().numpy()
        else:
            points = torch.from_numpy(points)
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.numpy()

        return sampled_points, indices

except:
    print("missing pytorch3d")

    def fps(points, num_points=1024, use_cuda=True):
        print("fps error: missing pytorch3d")
        exit()


class Camera:

    def __init__(self, bias=0, random_head_camera_dis=0, **kwags):
        """ """
        self.pcd_crop = kwags.get("pcd_crop", False)
        self.pcd_down_sample_num = kwags.get("pcd_down_sample_num", 0)
        self.pcd_crop_bbox = kwags.get("bbox", [[-0.6, -0.35, 0.7401], [0.6, 0.35, 2]])
        self.pcd_crop_bbox[0][2] += bias
        self.table_z_bias = bias
        self.random_head_camera_dis = random_head_camera_dis

        self.static_camera_config = []
        self.head_camera_type = kwags["camera"].get("head_camera_type", "D435")
        self.wrist_camera_type = kwags["camera"].get("wrist_camera_type", "D435")

        self.collect_head_camera = kwags["camera"].get("collect_head_camera", True)
        self.collect_wrist_camera = kwags["camera"].get("collect_wrist_camera", True)

        # embodiment = kwags.get('embodiment')
        # embodiment_config_path = os.path.join(CONFIGS_PATH, '_embodiment_config.yml')
        # with open(embodiment_config_path, 'r', encoding='utf-8') as f:
        #     embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)
        # robot_file = embodiment_types[embodiment]['file_path']
        # if robot_file is None:
        #     raise "No embodiment files"

        # robot_config_file = os.path.join(robot_file, 'config.yml')
        # with open(robot_config_file, 'r', encoding='utf-8') as f:
        #     embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
        # TODO
        self.static_camera_info_list = kwags["left_embodiment_config"]["static_camera_list"]
        self.static_camera_num = len(self.static_camera_info_list)

        # Camera random
        '''
        From RoboTwin 2.0-Plus in https://arxiv.org/pdf/2603.22078v2
        C1
        距离缩放 ∈ [0.85, 1.0] x 原始值 (仅head)

        C2
        方位角/俯仰角扰动(±10°) 和 ±10%距离变化（默认禁用）

        C3
        偏航/俯仰/滚转 随机方向上的各自[0°, 5°]
        '''
        camera_kw = kwags.get("camera", {})
        self.enable_head_camera_random = camera_kw.get("enable_head_camera_random", False)
        # C1
        self.head_cam_distance_scale_range = camera_kw.get("head_cam_distance_scale_range", [1.0, 1.0])

        # C2
        self.head_cam_azimuth_deg = float(camera_kw.get("head_cam_azimuth_deg", 0.0))
        self.head_cam_elevation_deg = float(camera_kw.get("head_cam_elevation_deg", 0.0))
        self.head_cam_distance_ratio = float(camera_kw.get("head_cam_distance_ratio", 0.0))

        # C3
        self.head_cam_yaw_deg = float(camera_kw.get("head_cam_yaw_deg", 0.0))
        self.head_cam_pitch_deg = float(camera_kw.get("head_cam_pitch_deg", 0.0))
        self.head_cam_roll_deg = float(camera_kw.get("head_cam_roll_deg", 0.0))

        # # info
        # self.cam_random_params = {}
        # Head camera visibility / occlusion check
        self.enable_head_camera_occlusion_check = bool(camera_kw.get("enable_head_camera_occlusion_check", False))
        self.head_camera_occlusion_valid_ratio_min = float(camera_kw.get("head_camera_occlusion_valid_ratio_min", 0.15))
        self.head_camera_occlusion_near_ratio_max = float(camera_kw.get("head_camera_occlusion_near_ratio_max", 0.60))
        self.head_camera_occlusion_near_depth_m = float(camera_kw.get("head_camera_occlusion_near_depth_m", 0.1))
        
        # Head camera dynamic randomization inside one episode
        self.enable_head_camera_dynamic_random = bool(camera_kw.get("enable_head_camera_dynamic_random", False))
        self.head_cam_dyn_min_interval_frames = int(camera_kw.get("head_cam_dyn_min_interval_frames", 5))
        self.head_cam_dyn_max_interval_frames = int(camera_kw.get("head_cam_dyn_max_interval_frames", 30))
        self.head_cam_dyn_trigger_prob = float(camera_kw.get("head_cam_dyn_trigger_prob", 0.15))

        self.head_cam_dyn_trans_step_max_m = camera_kw.get("head_cam_dyn_trans_step_max_m", [0.01, 0.01, 0.005])
        self.head_cam_dyn_rot_step_max_deg = camera_kw.get("head_cam_dyn_rot_step_max_deg", [2.0, 2.0, 2.0])
        self.head_cam_dyn_trans_abs_bound_m = camera_kw.get("head_cam_dyn_trans_abs_bound_m", [0.03, 0.03, 0.02])
        self.head_cam_dyn_rot_abs_bound_deg = camera_kw.get("head_cam_dyn_rot_abs_bound_deg", [8.0, 8.0, 8.0])

        self.head_cam_dyn_transition_frames = int(camera_kw.get("head_cam_dyn_transition_frames", 4))
        self.head_cam_dyn_smooth_profile = str(camera_kw.get("head_cam_dyn_smooth_profile", "cosine"))
        self.head_cam_dyn_anchor_mode = str(camera_kw.get("head_cam_dyn_anchor_mode", "episode_init"))

        # Dynamic mode: random_interval (legacy) | action_follow (human-like)
        self.head_cam_dyn_mode = str(camera_kw.get("head_cam_dyn_mode", "action_follow"))
        self.head_cam_dyn_follow_alpha_min = float(camera_kw.get("head_cam_dyn_follow_alpha_min", 0.02))
        self.head_cam_dyn_follow_alpha_max = float(camera_kw.get("head_cam_dyn_follow_alpha_max", 0.18))
        self.head_cam_dyn_motion_norm = float(camera_kw.get("head_cam_dyn_motion_norm", 0.015))
        self.head_cam_dyn_focus_z_bias = float(camera_kw.get("head_cam_dyn_focus_z_bias", 0.0))
        self.head_cam_dyn_hand_weight_temperature = float(camera_kw.get("head_cam_dyn_hand_weight_temperature", 0.03))
        self.head_cam_dyn_hand_weight_bias = float(camera_kw.get("head_cam_dyn_hand_weight_bias", 0.15))
        self.head_cam_dyn_hand_weight_smooth = float(camera_kw.get("head_cam_dyn_hand_weight_smooth", 0.35))

        self.head_cam_dyn_noise_trans_max_m = camera_kw.get("head_cam_dyn_noise_trans_max_m", [0.002, 0.002, 0.001])
        self.head_cam_dyn_noise_rot_max_deg = camera_kw.get("head_cam_dyn_noise_rot_max_deg", [0.8, 0.8, 0.8])
        self.head_cam_dyn_noise_update_prob = float(camera_kw.get("head_cam_dyn_noise_update_prob", 0.1))

        self.head_cam_dyn_log_min_trans_m = float(camera_kw.get("head_cam_dyn_log_min_trans_m", 0.0015))
        self.head_cam_dyn_log_min_rot_deg = float(camera_kw.get("head_cam_dyn_log_min_rot_deg", 0.5))
        
        self.head_cam_dyn_occlusion_reject = bool(camera_kw.get("head_cam_dyn_occlusion_reject", True))
        self.head_cam_dyn_max_resample = int(camera_kw.get("head_cam_dyn_max_resample", 5))
        
        # runtime record for dataset logging
        self.head_camera_episode_record = {}
        self.head_camera_dynamic_events = []

        # runtime states for dynamic camera
        self.scene = None
        self._head_dyn_last_trigger_frame = -1
        self._head_dyn_transition = None
        self._head_dyn_base_pose = None
        self._head_dyn_forced_change_count = 0
        self._head_dyn_random_change_count = 0
        self._head_dyn_follow_base_rel = None
        self._head_dyn_follow_noise_pos = np.zeros(3, dtype=np.float64)
        self._head_dyn_follow_noise_rot_deg = np.zeros(3, dtype=np.float64)
        self._head_dyn_last_logged_pose = None
        self._head_dyn_hand_weight_ema = np.array([0.5, 0.5], dtype=np.float64)


    def load_camera(self, scene):
        """
        Add cameras and set camera parameters
            - Including four cameras: left, right, front, head.
        """
        near, far = 0.1, 100
        camera_config_path = os.path.join(CONFIGS_PATH, "_camera_config.yml")

        assert os.path.isfile(camera_config_path), "task config file is missing"

        with open(camera_config_path, "r", encoding="utf-8") as f:
            camera_args = yaml.load(f.read(), Loader=yaml.FullLoader)

        # sensor_mount_actor = scene.create_actor_builder().build_kinematic()

        # camera_args = get_camera_config()

        # def create_camera(camera_info, random_head_camera_dis=0, is_head_camera=False):
        #     if camera_info["type"] not in camera_args.keys():
        #         raise ValueError(f"Camera type {camera_info['type']} not supported")

        #     camera_config = camera_args[camera_info["type"]]
        #     cam_pos = np.array(camera_info["position"])
        #     vector = np.random.randn(3)
        #     random_dir = vector / np.linalg.norm(vector)
        #     cam_pos = cam_pos + random_dir * np.random.uniform(low=0, high=random_head_camera_dis)
        #     cam_forward = np.array(camera_info["forward"]) / np.linalg.norm(np.array(camera_info["forward"]))
        #     cam_left = np.array(camera_info["left"]) / np.linalg.norm(np.array(camera_info["left"]))
        #     up = np.cross(cam_forward, cam_left)
        #     mat44 = np.eye(4)
        #     mat44[:3, :3] = np.stack([cam_forward, cam_left, up], axis=1)
        #     mat44[:3, 3] = cam_pos

        #     # ========================= sensor camera =========================
        #     # sensor_config = StereoDepthSensorConfig()
        #     # sensor_config.rgb_resolution = (camera_config['w'], camera_config['h'])

        #     camera = scene.add_camera(
        #         name=camera_info["name"],
        #         width=camera_config["w"],
        #         height=camera_config["h"],
        #         fovy=np.deg2rad(camera_config["fovy"]),
        #         near=near,
        #         far=far,
        #     )
        #     camera.entity.set_pose(sapien.Pose(mat44))

        #     # ========================= sensor camera =========================
        #     # sensor_camera = StereoDepthSensor(
        #     #     sensor_config,
        #     #     sensor_mount_actor,
        #     #     sapien.Pose(mat44)
        #     # )
        #     # camera.entity.set_pose(sapien.Pose(camera_info['position']))
        #     # return camera, sensor_camera, camera_config
        #     return camera, camera_config

        def create_camera(camera_info, random_head_camera_dis=0, is_head_camera=False):
            if camera_info["type"] not in camera_args.keys():
                raise ValueError(f"Camera type {camera_info['type']} not supported")

            def _norm(v):
                v = np.asarray(v, dtype=np.float64)
                n = np.linalg.norm(v)
                if n < 1e-8:
                    return v
                return v / n

            def _rand_deg(max_deg):
                if max_deg <= 0:
                    return 0.0
                return np.deg2rad(np.random.uniform(-max_deg, max_deg))

            def _orthonormalize(R):
                '''把矩阵近似正交化'''
                U, _, Vt = np.linalg.svd(R)
                R = U @ Vt
                if np.linalg.det(R) < 0:
                    U[:, -1] *= -1
                    R = U @ Vt
                return R

            camera_config = camera_args[camera_info["type"]]

            base_pos = np.array(camera_info["position"])
            base_forward = _norm(np.array(camera_info["forward"]))
            base_left = _norm(np.array(camera_info["left"]))
            base_up = _norm(np.cross(base_forward, base_left))

            # default orientation from config
            R_default = np.stack([base_forward, base_left, base_up], axis=1)
            R_default = _orthonormalize(R_default)

            cam_pos = base_pos.copy()
            R = R_default.copy()

            use_head_random = bool(is_head_camera and self.enable_head_camera_random)
            #record
            sampled_distance_scale = 1.0
            sampled_azimuth_offset_deg = 0.0
            sampled_elevation_offset_deg = 0.0
            sampled_distance_ratio = 0.0
            sampled_yaw_deg = 0.0
            sampled_pitch_deg = 0.0
            sampled_roll_deg = 0.0
            sampled_translation_jitter = 0.0

            if use_head_random:
                # ---------------- C1/C2: position perturbation around auto anchor ----------------
                # auto anchor from default camera pose + viewing ray intersecting table plane
                # 参考 base_task
                table_z = 0.74 + self.table_z_bias 
                #桌面高度table_height 0.74，
                # R(t) = P + t × F为视线，t是距离参数，相机位置 P = (px, py, pz) 出发，方向 F = (fx, fy, fz)
                #R_z(t) = pz + t × fz = 0.74 视线和桌子平面的交点
                if abs(base_forward[2]) > 1e-6:
                    t_hit = (table_z - base_pos[2]) / base_forward[2] 
                    if t_hit > 0:
                        anchor = base_pos + t_hit * base_forward
                    else:
                        anchor = base_pos + 0.8 * base_forward
                else:
                    anchor = base_pos + 0.8 * base_forward

                rel = base_pos - anchor
                rel_norm = np.linalg.norm(rel)
                if rel_norm < 1e-8:
                    rel = np.array([0.0, -0.8, 0.2], dtype=np.float64)
                    rel_norm = np.linalg.norm(rel)

                # spherical coords
                az = np.arctan2(rel[1], rel[0]) #水平面方位角
                el = np.arctan2(rel[2], np.linalg.norm(rel[:2])) #与水平面夹角的俯仰角
                r = rel_norm #相机到锚点的距离

                # C1 distance scaling
                smin, smax = self.head_cam_distance_scale_range
                if smin > smax:
                    smin, smax = smax, smin
                # r = r * np.random.uniform(smin, smax)
                sampled_distance_scale = float(np.random.uniform(smin, smax))
                r = r * sampled_distance_scale

                # C2 azimuth/elevation + distance ratio perturb
                sampled_azimuth_offset_deg = float(
                    np.random.uniform(-self.head_cam_azimuth_deg, self.head_cam_azimuth_deg)
                )
                sampled_elevation_offset_deg = float(
                    np.random.uniform(-self.head_cam_elevation_deg, self.head_cam_elevation_deg)
                )
                az += np.deg2rad(sampled_azimuth_offset_deg)
                el += np.deg2rad(sampled_elevation_offset_deg)
                # az += np.deg2rad(np.random.uniform(-self.head_cam_azimuth_deg, self.head_cam_azimuth_deg))
                # el += np.deg2rad(np.random.uniform(-self.head_cam_elevation_deg, self.head_cam_elevation_deg))
                el = np.clip(el, np.deg2rad(-85.0), np.deg2rad(85.0))

                if self.head_cam_distance_ratio > 0:
                    ratio = np.random.uniform(-self.head_cam_distance_ratio, self.head_cam_distance_ratio)
                    sampled_distance_ratio = float(ratio)
                    r = r * (1.0 + ratio)

                rel_new = np.array([
                    r * np.cos(el) * np.cos(az),
                    r * np.cos(el) * np.sin(az),
                    r * np.sin(el),
                ], dtype=np.float64)
                cam_pos = anchor + rel_new

                # keep old isotropic translation jitter compatibility
                if random_head_camera_dis > 0:
                    vec = _norm(np.random.randn(3))
                    # cam_pos = cam_pos + vec * np.random.uniform(low=0, high=random_head_camera_dis)
                    sampled_translation_jitter = float(np.random.uniform(low=0, high=random_head_camera_dis))
                    cam_pos = cam_pos + vec * sampled_translation_jitter

                # recompute look-at orientation (look at anchor)
                f = _norm(anchor - cam_pos)
                world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                if abs(np.dot(f, world_up)) > 0.99:
                    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                l = _norm(np.cross(world_up, f))
                u = _norm(np.cross(f, l))
                R = np.stack([f, l, u], axis=1)
                R = _orthonormalize(R)

                # ---------------- C3: orientation perturbation ----------------
                yaw = _rand_deg(self.head_cam_yaw_deg)      # around local up（z），左右摇头
                pitch = _rand_deg(self.head_cam_pitch_deg)  # around local left（y），上下俯仰
                roll = _rand_deg(self.head_cam_roll_deg)    # around local forward（x），左右倾斜
                sampled_yaw_deg = float(np.rad2deg(yaw))
                sampled_pitch_deg = float(np.rad2deg(pitch))
                sampled_roll_deg = float(np.rad2deg(roll))

                u_axis = R[:, 2] #Up，z
                l_axis = R[:, 1] #Left，y
                f_axis = R[:, 0] #Forward，x

                R_yaw = t3d.axangles.axangle2mat(u_axis, yaw)
                R_pitch = t3d.axangles.axangle2mat(l_axis, pitch)
                R_roll = t3d.axangles.axangle2mat(f_axis, roll)

                R = R_yaw @ R_pitch @ R_roll @ R  #从右向左，先x，再y，最后z
                R = _orthonormalize(R)

            else:
                # original behavior (position-only random jitter)
                if random_head_camera_dis > 0:
                    vector = np.random.randn(3)
                    random_dir = vector / (np.linalg.norm(vector) + 1e-12)
                    # cam_pos = cam_pos + random_dir * np.random.uniform(low=0, high=random_head_camera_dis)
                    sampled_translation_jitter = float(np.random.uniform(low=0, high=random_head_camera_dis))
                    cam_pos = cam_pos + random_dir * sampled_translation_jitter
                R = R_default

            mat44 = np.eye(4, dtype=np.float64)
            mat44[:3, :3] = R
            mat44[:3, 3] = cam_pos

            camera = scene.add_camera(
                name=camera_info["name"],
                width=camera_config["w"],
                height=camera_config["h"],
                fovy=np.deg2rad(camera_config["fovy"]),
                near=near,
                far=far,
            )
            camera.entity.set_pose(sapien.Pose(mat44))
            
            if is_head_camera:
                camera_pose = camera.entity.get_pose()
                self.head_camera_episode_record = {
                    "head_camera_randomization_sampled": {
                        "used_head_camera_random": bool(use_head_random),
                        "distance_scale": float(sampled_distance_scale),
                        "azimuth_offset_deg": float(sampled_azimuth_offset_deg),
                        "elevation_offset_deg": float(sampled_elevation_offset_deg),
                        "distance_ratio": float(sampled_distance_ratio),
                        "yaw_deg": float(sampled_yaw_deg),
                        "pitch_deg": float(sampled_pitch_deg),
                        "roll_deg": float(sampled_roll_deg),
                        "translation_jitter": float(sampled_translation_jitter),
                    },
                    "head_camera_final_pose": {
                        "position": np.asarray(camera_pose.p, dtype=np.float64).tolist(),
                        "quaternion_wxyz": np.asarray(camera_pose.q, dtype=np.float64).tolist(),
                        "rotation_matrix": np.asarray(mat44[:3, :3], dtype=np.float64).tolist(),
                    },
                }
            return camera, camera_config

        # ================================= wrist camera =================================
        if self.collect_wrist_camera:
            wrist_camera_config = camera_args[self.wrist_camera_type]
            self.left_camera = scene.add_camera(
                name="left_camera",
                width=wrist_camera_config["w"],
                height=wrist_camera_config["h"],
                fovy=np.deg2rad(wrist_camera_config["fovy"]),
                near=near,
                far=far,
            )

            self.right_camera = scene.add_camera(
                name="right_camera",
                width=wrist_camera_config["w"],
                height=wrist_camera_config["h"],
                fovy=np.deg2rad(wrist_camera_config["fovy"]),
                near=near,
                far=far,
            )

        # ================================= sensor camera =================================
        # sensor_config = StereoDepthSensorConfig()
        # sensor_config.rgb_resolution = (wrist_camera_config['w'], wrist_camera_config['h'])
        # self.left_sensor_camera = StereoDepthSensor(
        #     sensor_config,
        #     sensor_mount_actor,
        #     sapien.Pose([0,0,0],[1,0,0,0])
        # )

        # self.right_sensor_camera = StereoDepthSensor(
        #     sensor_config,
        #     sensor_mount_actor,
        #     sapien.Pose([0,0,0],[1,0,0,0])
        # )

        # ================================= static camera =================================
        self.head_camera_id = None
        self.static_camera_list = []
        # self.static_sensor_camera_list = []
        self.static_camera_name = []
        # static camera list
        for i, camera_info in enumerate(self.static_camera_info_list):
            if camera_info.get("forward") == None:
                camera_info["forward"] = (-1 * np.array(camera_info["position"])).tolist()
            if camera_info.get("left") == None:
                camera_info["left"] = [
                    -camera_info["forward"][1],
                    camera_info["forward"][0],
                ] + [0]

            if camera_info["name"] == "head_camera":
                if self.collect_head_camera:
                    self.head_camera_id = i
                    camera_info["type"] = self.head_camera_type
                    # camera, sensor_camera, camera_config = create_camera(camera_info)
                    camera, camera_config = create_camera(camera_info,
                                                          random_head_camera_dis=self.random_head_camera_dis,
                                                          is_head_camera=True)
                    self.static_camera_list.append(camera)
                    self.static_camera_name.append(camera_info["name"])
                    # self.static_sensor_camera_list.append(sensor_camera)
                    self.static_camera_config.append(camera_config)
                    # ================================= sensor camera =================================
                    # camera_config = get_camera_config(camera_info['type'])
                    # cam_pos = np.array(camera_info['position'])
                    # cam_forward = np.array(camera_info['forward']) / np.linalg.norm(np.array(camera_info['forward']))
                    # cam_left = np.array(camera_info['left']) / np.linalg.norm(np.array(camera_info['left']))
                    # up = np.cross(cam_forward, cam_left)
                    # mat44 = np.eye(4)
                    # mat44[:3, :3] = np.stack([cam_forward, cam_left, up], axis=1)
                    # mat44[:3, 3] = cam_pos
                    # sensor_config = StereoDepthSensorConfig()
                    # sensor_config.rgb_resolution = (camera_config['w'], camera_config['h'])

                    # self.head_sensor = StereoDepthSensor(
                    #     sensor_config,
                    #     sensor_mount_actor,
                    #     sapien.Pose(mat44)
                    # )
            else:
                # camera, sensor_camera, camera_config = create_camera(camera_info)
                camera, camera_config = create_camera(camera_info)
                self.static_camera_list.append(camera)
                self.static_camera_name.append(camera_info["name"])
                # self.static_sensor_camera_list.append(sensor_camera)
                self.static_camera_config.append(camera_config)

        # observer camera
        self.observer_camera = scene.add_camera(
            name="observer_camera",
            width=320,
            height=240,
            fovy=np.deg2rad(93),
            near=near,
            far=far,
        )
        observer_cam_pos = np.array([0.0, 0.23, 1.33])
        observer_cam_forward = np.array([0, -1, -1.02])
        # observer_cam_left = np.array([1,-1, 0])
        observer_cam_left = np.array([1, 0, 0])
        observer_up = np.cross(observer_cam_forward, observer_cam_left)
        observer_mat44 = np.eye(4)
        observer_mat44[:3, :3] = np.stack([observer_cam_forward, observer_cam_left, observer_up], axis=1)
        observer_mat44[:3, 3] = observer_cam_pos
        self.observer_camera.entity.set_pose(sapien.Pose(observer_mat44))

        # world pcd camera
        self.world_camera1 = scene.add_camera(
            name="world_camera1",
            width=640,
            height=480,
            fovy=np.deg2rad(50),
            near=near,
            far=far,
        )
        world_cam_pos = np.array([0.4, -0.4, 1.6])
        world_cam_forward = np.array([-1, 1, -1.4])
        world_cam_left = np.array([-1, -1, 0])
        world_cam_up = np.cross(world_cam_forward, world_cam_left)
        world_cam_mat44 = np.eye(4)
        world_cam_mat44[:3, :3] = np.stack([world_cam_forward, world_cam_left, world_cam_up], axis=1)
        world_cam_mat44[:3, 3] = world_cam_pos
        self.world_camera1.entity.set_pose(sapien.Pose(world_cam_mat44))

        self.world_camera2 = scene.add_camera(
            name="world_camera1",
            width=640,
            height=480,
            fovy=np.deg2rad(50),
            near=near,
            far=far,
        )
        world_cam_pos = np.array([-0.4, -0.4, 1.6])
        world_cam_forward = np.array([1, 1, -1.4])
        world_cam_left = np.array([-1, 1, 0])
        world_cam_up = np.cross(world_cam_forward, world_cam_left)
        world_cam_mat44 = np.eye(4)
        world_cam_mat44[:3, :3] = np.stack([world_cam_forward, world_cam_left, world_cam_up], axis=1)
        world_cam_mat44[:3, 3] = world_cam_pos
        self.world_camera2.entity.set_pose(sapien.Pose(world_cam_mat44))
        
        self.reset_head_camera_dynamic_runtime()
        
    def reset_head_camera_dynamic_runtime(self):
        self._head_dyn_last_trigger_frame = -1
        self._head_dyn_transition = None
        self._head_dyn_forced_change_count = 0
        self._head_dyn_random_change_count = 0
        self.head_camera_dynamic_events = []
        self._head_dyn_follow_base_rel = None
        self._head_dyn_follow_noise_pos = np.zeros(3, dtype=np.float64)
        self._head_dyn_follow_noise_rot_deg = np.zeros(3, dtype=np.float64)
        self._head_dyn_last_logged_pose = None
        self._head_dyn_hand_weight_ema = np.array([0.5, 0.5], dtype=np.float64)

        if self.head_camera_id is None or not self.collect_head_camera:
            self._head_dyn_base_pose = None
            return

        head_camera = self.static_camera_list[self.head_camera_id]
        pose = head_camera.entity.get_pose()
        self._head_dyn_base_pose = {
            "p": np.asarray(pose.p, dtype=np.float64).copy(),
            "q": np.asarray(pose.q, dtype=np.float64).copy(),
        }
        self._head_dyn_last_logged_pose = {
            "p": np.asarray(pose.p, dtype=np.float64).copy(),
            "q": np.asarray(pose.q, dtype=np.float64).copy(),
        }
    
    def get_head_camera_info(self):
        return self.head_camera_episode_record

    def _normalize3(self, value, cast=float):
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) >= 3:
                return np.array([cast(value[0]), cast(value[1]), cast(value[2])], dtype=np.float64)
            elif len(value) == 1:
                v = cast(value[0])
                return np.array([v, v, v], dtype=np.float64)
        v = cast(value)
        return np.array([v, v, v], dtype=np.float64)

    def _smooth_alpha(self, t):
        t = float(np.clip(t, 0.0, 1.0))
        profile = self.head_cam_dyn_smooth_profile.lower()
        if profile == "linear":
            return t
        if profile == "cubic":
            return 3.0 * t * t - 2.0 * t * t * t
        # default cosine
        return 0.5 * (1.0 - np.cos(np.pi * t))

    def _slerp_wxyz(self, q0, q1, alpha):
        q0 = np.asarray(q0, dtype=np.float64)
        q1 = np.asarray(q1, dtype=np.float64)
        q0 = q0 / (np.linalg.norm(q0) + 1e-12)
        q1 = q1 / (np.linalg.norm(q1) + 1e-12)

        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        if dot > 0.9995:
            q = q0 + alpha * (q1 - q0)
            return q / (np.linalg.norm(q) + 1e-12)

        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * alpha
        sin_theta = np.sin(theta)
        s0 = np.sin(theta_0 - theta) / (sin_theta_0 + 1e-12)
        s1 = sin_theta / (sin_theta_0 + 1e-12)
        q = s0 * q0 + s1 * q1
        return q / (np.linalg.norm(q) + 1e-12)

    def _quat_angle_deg(self, q_from, q_to):
        R_from = t3d.quaternions.quat2mat(np.asarray(q_from, dtype=np.float64))
        R_to = t3d.quaternions.quat2mat(np.asarray(q_to, dtype=np.float64))
        R_rel = R_to @ R_from.T
        trace_val = np.trace(R_rel)
        cos_theta = np.clip((trace_val - 1.0) * 0.5, -1.0, 1.0)
        return float(np.rad2deg(np.arccos(cos_theta)))

    def _look_at_quat_wxyz(self, cam_pos, target_pos):
        cam_pos = np.asarray(cam_pos, dtype=np.float64)
        target_pos = np.asarray(target_pos, dtype=np.float64)
        f = target_pos - cam_pos
        n = np.linalg.norm(f)
        if n < 1e-8:
            if self._head_dyn_base_pose is not None:
                return np.asarray(self._head_dyn_base_pose["q"], dtype=np.float64)
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        f = f / n
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(f, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        l = np.cross(world_up, f)
        l = l / (np.linalg.norm(l) + 1e-12)
        u = np.cross(f, l)
        u = u / (np.linalg.norm(u) + 1e-12)
        R = np.stack([f, l, u], axis=1)
        return t3d.quaternions.mat2quat(R)

    def _try_apply_dynamic_pose(self, frame_idx, reason, target_p, target_q, sample_meta=None):
        head_camera = self.static_camera_list[self.head_camera_id]
        cur_pose = head_camera.entity.get_pose()
        cur_p = np.asarray(cur_pose.p, dtype=np.float64)
        cur_q = np.asarray(cur_pose.q, dtype=np.float64)

        head_camera.entity.set_pose(sapien.Pose(target_p, target_q))
        if self.scene is not None:
            self.scene.update_render()

        if self.head_cam_dyn_occlusion_reject:
            ok, metrics = self._check_head_camera_visibility_internal()
            if not ok:
                head_camera.entity.set_pose(sapien.Pose(cur_p, cur_q))
                if self.scene is not None:
                    self.scene.update_render()
                self.head_camera_dynamic_events.append({
                    "trigger_frame": int(frame_idx),
                    "reason": str(reason),
                    "accepted": False,
                    "occlusion_check": metrics,
                    "candidate_pose": {
                        "position": np.asarray(target_p, dtype=np.float64).tolist(),
                        "quaternion_wxyz": np.asarray(target_q, dtype=np.float64).tolist(),
                    },
                })
                return False

        # logging sparse keyframes only
        if self._head_dyn_last_logged_pose is None:
            should_log = True
        else:
            dp = np.linalg.norm(np.asarray(target_p, dtype=np.float64) - self._head_dyn_last_logged_pose["p"])
            dq = self._quat_angle_deg(self._head_dyn_last_logged_pose["q"], target_q)
            should_log = (dp >= self.head_cam_dyn_log_min_trans_m) or (dq >= self.head_cam_dyn_log_min_rot_deg)

        if should_log:
            event = {
                "trigger_frame": int(frame_idx),
                "reason": str(reason),
                "accepted": True,
                "final_pose": {
                    "position": np.asarray(target_p, dtype=np.float64).tolist(),
                    "quaternion_wxyz": np.asarray(target_q, dtype=np.float64).tolist(),
                },
            }
            if sample_meta is not None:
                event["sample_meta"] = sample_meta
            self.head_camera_dynamic_events.append(event)
            self._head_dyn_last_logged_pose = {
                "p": np.asarray(target_p, dtype=np.float64).copy(),
                "q": np.asarray(target_q, dtype=np.float64).copy(),
            }
        return True
    
    def _check_head_camera_visibility_internal(self):
        if self.head_camera_id is None or not self.collect_head_camera:
            metrics = {
                "enabled": bool(self.enable_head_camera_occlusion_check),
                "checked": False,
                "ok": True,
                "reason": "head_camera_not_collected",
            }
            return True, metrics

        if not self.enable_head_camera_occlusion_check:
            metrics = {
                "enabled": False,
                "checked": False,
                "ok": True,
                "reason": "disabled_by_config",
            }
            return True, metrics

        head_camera = self.static_camera_list[self.head_camera_id]
        head_camera.take_picture()
        position = head_camera.get_picture("Position")

        valid_mask = position[..., 3] < 1
        depth = -position[..., 2]

        total_pixels = int(valid_mask.size)
        valid_pixels = int(np.count_nonzero(valid_mask))
        valid_ratio = float(valid_pixels / max(total_pixels, 1))

        near_depth = self.head_camera_occlusion_near_depth_m
        near_pixels = int(np.count_nonzero(valid_mask & (depth < near_depth)))
        near_ratio = float(near_pixels / max(valid_pixels, 1))

        valid_ratio_ok = valid_ratio >= self.head_camera_occlusion_valid_ratio_min
        near_ratio_ok = near_ratio <= self.head_camera_occlusion_near_ratio_max
        ok = bool(valid_ratio_ok and near_ratio_ok)

        metrics = {
            "enabled": True,
            "checked": True,
            "ok": ok,
            "threshold": {
                "valid_ratio_min": float(self.head_camera_occlusion_valid_ratio_min),
                "near_ratio_max": float(self.head_camera_occlusion_near_ratio_max),
                "near_depth_m": float(self.head_camera_occlusion_near_depth_m),
            },
            "measured": {
                "total_pixels": total_pixels,
                "valid_pixels": valid_pixels,
                "valid_ratio": valid_ratio,
                "near_pixels": near_pixels,
                "near_ratio": near_ratio,
            },
        }
        return ok, metrics

    def _sample_head_dynamic_target_pose(self):
        head_camera = self.static_camera_list[self.head_camera_id]
        cur_pose = head_camera.entity.get_pose()
        cur_p = np.asarray(cur_pose.p, dtype=np.float64)
        cur_q = np.asarray(cur_pose.q, dtype=np.float64)

        if self._head_dyn_base_pose is None:
            base_p = cur_p.copy()
            base_q = cur_q.copy()
        else:
            base_p = self._head_dyn_base_pose["p"]
            base_q = self._head_dyn_base_pose["q"]

        anchor_mode = self.head_cam_dyn_anchor_mode.lower()
        if anchor_mode == "previous":
            ref_p = cur_p
            ref_q = cur_q
        else:
            ref_p = base_p
            ref_q = base_q

        trans_step_max = self._normalize3(self.head_cam_dyn_trans_step_max_m)
        rot_step_max_deg = self._normalize3(self.head_cam_dyn_rot_step_max_deg)

        delta_p = np.random.uniform(-trans_step_max, trans_step_max)

        R_ref = t3d.quaternions.quat2mat(ref_q)
        yaw = np.deg2rad(np.random.uniform(-rot_step_max_deg[0], rot_step_max_deg[0]))
        pitch = np.deg2rad(np.random.uniform(-rot_step_max_deg[1], rot_step_max_deg[1]))
        roll = np.deg2rad(np.random.uniform(-rot_step_max_deg[2], rot_step_max_deg[2]))

        f_axis = R_ref[:, 0]
        l_axis = R_ref[:, 1]
        u_axis = R_ref[:, 2]

        R_yaw = t3d.axangles.axangle2mat(u_axis, yaw)
        R_pitch = t3d.axangles.axangle2mat(l_axis, pitch)
        R_roll = t3d.axangles.axangle2mat(f_axis, roll)
        R_target = R_yaw @ R_pitch @ R_roll @ R_ref

        p_target = ref_p + delta_p

        # absolute bound around episode init pose
        trans_abs_bound = self._normalize3(self.head_cam_dyn_trans_abs_bound_m)
        p_target = base_p + np.clip(p_target - base_p, -trans_abs_bound, trans_abs_bound)

        rot_abs_bound_deg = self._normalize3(self.head_cam_dyn_rot_abs_bound_deg)
        R_base = t3d.quaternions.quat2mat(base_q)
        R_rel = R_target @ R_base.T
        rx, ry, rz = t3d.euler.mat2euler(R_rel, axes='sxyz')
        rx = np.clip(rx, -np.deg2rad(rot_abs_bound_deg[0]), np.deg2rad(rot_abs_bound_deg[0]))
        ry = np.clip(ry, -np.deg2rad(rot_abs_bound_deg[1]), np.deg2rad(rot_abs_bound_deg[1]))
        rz = np.clip(rz, -np.deg2rad(rot_abs_bound_deg[2]), np.deg2rad(rot_abs_bound_deg[2]))
        R_rel_clamped = t3d.euler.euler2mat(rx, ry, rz, axes='sxyz')
        R_target = R_rel_clamped @ R_base

        q_target = t3d.quaternions.mat2quat(R_target)
        return {
            "p": np.asarray(p_target, dtype=np.float64),
            "q": np.asarray(q_target, dtype=np.float64),
            "delta": {
                "translation_xyz_m": np.asarray(delta_p, dtype=np.float64).tolist(),
                "rotation_yaw_pitch_roll_deg": [float(np.rad2deg(yaw)), float(np.rad2deg(pitch)), float(np.rad2deg(roll))],
            },
        }

    def _try_build_head_dynamic_transition(self, frame_idx, reason):
        head_camera = self.static_camera_list[self.head_camera_id]
        cur_pose = head_camera.entity.get_pose()
        start_p = np.asarray(cur_pose.p, dtype=np.float64)
        start_q = np.asarray(cur_pose.q, dtype=np.float64)

        max_retry = max(1, int(self.head_cam_dyn_max_resample))
        reject_attempts = 0
        reject_metrics = None

        for _ in range(max_retry):
            sampled = self._sample_head_dynamic_target_pose()
            target_p = sampled["p"]
            target_q = sampled["q"]

            head_camera.entity.set_pose(sapien.Pose(target_p, target_q))
            if self.scene is not None:
                self.scene.update_render()

            if self.head_cam_dyn_occlusion_reject:
                ok, metrics = self._check_head_camera_visibility_internal()
                if not ok:
                    reject_attempts += 1
                    reject_metrics = metrics
                    head_camera.entity.set_pose(sapien.Pose(start_p, start_q))
                    if self.scene is not None:
                        self.scene.update_render()
                    continue
            else:
                ok, metrics = True, {
                    "enabled": False,
                    "checked": False,
                    "ok": True,
                    "reason": "dynamic_occlusion_reject_disabled",
                }

            # restore now, will be applied smoothly in transition
            head_camera.entity.set_pose(sapien.Pose(start_p, start_q))
            if self.scene is not None:
                self.scene.update_render()

            duration = max(1, int(self.head_cam_dyn_transition_frames))
            event = {
                "trigger_frame": int(frame_idx),
                "reason": str(reason),
                "transition_frames": duration,
                "start_pose": {
                    "position": start_p.tolist(),
                    "quaternion_wxyz": start_q.tolist(),
                },
                "target_pose": {
                    "position": np.asarray(target_p, dtype=np.float64).tolist(),
                    "quaternion_wxyz": np.asarray(target_q, dtype=np.float64).tolist(),
                },
                "sampled_delta": sampled["delta"],
                "occlusion_check": metrics,
                "accepted": True,
                "rejected_attempts": int(reject_attempts),
            }

            return {
                "start_frame": int(frame_idx),
                "duration": duration,
                "start_p": start_p,
                "start_q": start_q,
                "target_p": np.asarray(target_p, dtype=np.float64),
                "target_q": np.asarray(target_q, dtype=np.float64),
                "event": event,
            }

        # all candidates rejected by occlusion
        event = {
            "trigger_frame": int(frame_idx),
            "reason": str(reason),
            "transition_frames": max(1, int(self.head_cam_dyn_transition_frames)),
            "accepted": False,
            "rejected_attempts": int(reject_attempts),
            "occlusion_check": reject_metrics,
        }
        self.head_camera_dynamic_events.append(event)
        return None

    def _update_head_camera_dynamic_follow(self, frame_idx=0, motion_signal=None):
        if motion_signal is None:
            return

        left_ee_pos = motion_signal.get("left_ee_pos", None)
        right_ee_pos = motion_signal.get("right_ee_pos", None)
        focus_target = motion_signal.get("focus_target", None)
        if left_ee_pos is None or right_ee_pos is None:
            return

        left_ee_pos = np.asarray(left_ee_pos, dtype=np.float64)
        right_ee_pos = np.asarray(right_ee_pos, dtype=np.float64)
        focus_target = np.asarray(focus_target if focus_target is not None else (left_ee_pos + right_ee_pos) * 0.5, dtype=np.float64)
        focus_target[2] += float(self.head_cam_dyn_focus_z_bias)

        motion_intensity = float(motion_signal.get("motion_intensity", 0.0))
        motion_intensity = float(np.clip(motion_intensity, 0.0, 1.0))

        left_motion = float(motion_signal.get("left_motion", 0.0))
        right_motion = float(motion_signal.get("right_motion", 0.0))
        motion_scale = max(float(self.head_cam_dyn_motion_norm), 1e-6)
        left_score = left_motion / motion_scale
        right_score = right_motion / motion_scale
        temp = max(float(self.head_cam_dyn_hand_weight_temperature), 1e-6)
        score_delta = (left_score - right_score) / temp
        dominant_left = 1.0 / (1.0 + np.exp(-score_delta))
        dominant_right = 1.0 - dominant_left

        base_bias = float(np.clip(self.head_cam_dyn_hand_weight_bias, 0.0, 0.49))
        left_weight = dominant_left * (1.0 - 2.0 * base_bias) + base_bias
        right_weight = dominant_right * (1.0 - 2.0 * base_bias) + base_bias
        hand_weights = np.array([left_weight, right_weight], dtype=np.float64)
        hand_weights = hand_weights / (np.sum(hand_weights) + 1e-12)

        smooth = float(np.clip(self.head_cam_dyn_hand_weight_smooth, 0.0, 1.0))
        self._head_dyn_hand_weight_ema = (1.0 - smooth) * self._head_dyn_hand_weight_ema + smooth * hand_weights
        hand_weights = self._head_dyn_hand_weight_ema / (np.sum(self._head_dyn_hand_weight_ema) + 1e-12)

        focus_target = hand_weights[0] * left_ee_pos + hand_weights[1] * right_ee_pos
        focus_target[2] += float(self.head_cam_dyn_focus_z_bias)

        head_camera = self.static_camera_list[self.head_camera_id]
        cur_pose = head_camera.entity.get_pose()
        cur_p = np.asarray(cur_pose.p, dtype=np.float64)
        cur_q = np.asarray(cur_pose.q, dtype=np.float64)

        if self._head_dyn_follow_base_rel is None:
            self._head_dyn_follow_base_rel = cur_p - focus_target

        # low-frequency micro-motion noise
        if np.random.rand() < float(np.clip(self.head_cam_dyn_noise_update_prob, 0.0, 1.0)):
            noise_pos_max = self._normalize3(self.head_cam_dyn_noise_trans_max_m)
            noise_rot_max_deg = self._normalize3(self.head_cam_dyn_noise_rot_max_deg)
            self._head_dyn_follow_noise_pos = np.random.uniform(-noise_pos_max, noise_pos_max)
            self._head_dyn_follow_noise_rot_deg = np.random.uniform(-noise_rot_max_deg, noise_rot_max_deg)

        base_rel = np.asarray(self._head_dyn_follow_base_rel, dtype=np.float64)
        desired_p = focus_target + base_rel + self._head_dyn_follow_noise_pos

        # absolute bound around episode init pose
        if self._head_dyn_base_pose is not None:
            base_p = self._head_dyn_base_pose["p"]
            trans_abs_bound = self._normalize3(self.head_cam_dyn_trans_abs_bound_m)
            desired_p = base_p + np.clip(desired_p - base_p, -trans_abs_bound, trans_abs_bound)

        desired_q = self._look_at_quat_wxyz(desired_p, focus_target)
        R_des = t3d.quaternions.quat2mat(desired_q)
        yaw_n = np.deg2rad(self._head_dyn_follow_noise_rot_deg[0])
        pitch_n = np.deg2rad(self._head_dyn_follow_noise_rot_deg[1])
        roll_n = np.deg2rad(self._head_dyn_follow_noise_rot_deg[2])
        f_axis = R_des[:, 0]
        l_axis = R_des[:, 1]
        u_axis = R_des[:, 2]
        R_des = t3d.axangles.axangle2mat(u_axis, yaw_n) @ t3d.axangles.axangle2mat(l_axis, pitch_n) @ t3d.axangles.axangle2mat(f_axis, roll_n) @ R_des

        if self._head_dyn_base_pose is not None:
            base_q = self._head_dyn_base_pose["q"]
            rot_abs_bound_deg = self._normalize3(self.head_cam_dyn_rot_abs_bound_deg)
            R_base = t3d.quaternions.quat2mat(base_q)
            R_rel = R_des @ R_base.T
            rx, ry, rz = t3d.euler.mat2euler(R_rel, axes='sxyz')
            rx = np.clip(rx, -np.deg2rad(rot_abs_bound_deg[0]), np.deg2rad(rot_abs_bound_deg[0]))
            ry = np.clip(ry, -np.deg2rad(rot_abs_bound_deg[1]), np.deg2rad(rot_abs_bound_deg[1]))
            rz = np.clip(rz, -np.deg2rad(rot_abs_bound_deg[2]), np.deg2rad(rot_abs_bound_deg[2]))
            R_des = t3d.euler.euler2mat(rx, ry, rz, axes='sxyz') @ R_base

        desired_q = t3d.quaternions.mat2quat(R_des)

        alpha_min = float(np.clip(self.head_cam_dyn_follow_alpha_min, 0.0, 1.0))
        alpha_max = float(np.clip(self.head_cam_dyn_follow_alpha_max, alpha_min, 1.0))
        alpha = alpha_min + (alpha_max - alpha_min) * motion_intensity

        new_p = cur_p * (1.0 - alpha) + desired_p * alpha
        new_q = self._slerp_wxyz(cur_q, desired_q, alpha)

        # optional occlusion reject and sparse logging
        sample_meta = {
            "motion_intensity": float(motion_intensity),
            "alpha": float(alpha),
            "focus_target": np.asarray(focus_target, dtype=np.float64).tolist(),
            "left_hand_weight": float(hand_weights[0]),
            "right_hand_weight": float(hand_weights[1]),
            "left_motion": float(left_motion),
            "right_motion": float(right_motion),
            "dominant_hand": "left" if hand_weights[0] >= hand_weights[1] else "right",
            "mode": "action_follow",
        }
        self._try_apply_dynamic_pose(frame_idx=frame_idx, reason="action_follow", target_p=new_p, target_q=new_q, sample_meta=sample_meta)
    
    def update_head_camera_dynamic(self, frame_idx=0, motion_signal=None):
        if self.head_camera_id is None or not self.collect_head_camera:
            return
        if not self.enable_head_camera_dynamic_random:
            return

        frame_idx = int(frame_idx)
        
        mode = self.head_cam_dyn_mode.lower()
        if mode == "action_follow":
            self._update_head_camera_dynamic_follow(frame_idx=frame_idx, motion_signal=motion_signal)
            return
        
        head_camera = self.static_camera_list[self.head_camera_id]

        # ongoing smooth transition
        if self._head_dyn_transition is not None:
            trans = self._head_dyn_transition
            elapsed = frame_idx - trans["start_frame"] + 1
            alpha = self._smooth_alpha(elapsed / max(trans["duration"], 1))
            p_now = trans["start_p"] * (1.0 - alpha) + trans["target_p"] * alpha
            q_now = self._slerp_wxyz(trans["start_q"], trans["target_q"], alpha)
            head_camera.entity.set_pose(sapien.Pose(p_now, q_now))

            if elapsed >= trans["duration"]:
                pose = head_camera.entity.get_pose()
                trans["event"]["end_frame"] = int(frame_idx)
                trans["event"]["final_pose"] = {
                    "position": np.asarray(pose.p, dtype=np.float64).tolist(),
                    "quaternion_wxyz": np.asarray(pose.q, dtype=np.float64).tolist(),
                }
                self.head_camera_dynamic_events.append(trans["event"])
                self._head_dyn_transition = None
            return

        min_itv = max(0, int(self.head_cam_dyn_min_interval_frames))
        max_itv = max(min_itv + 1, int(self.head_cam_dyn_max_interval_frames))
        interval = frame_idx - int(self._head_dyn_last_trigger_frame)

        should_trigger = False
        reason = ""
        if interval >= max_itv:
            should_trigger = True
            reason = "forced_by_max_interval"
            self._head_dyn_forced_change_count += 1
        elif interval >= min_itv:
            if np.random.rand() < float(self.head_cam_dyn_trigger_prob):
                should_trigger = True
                reason = "random_trigger"
                self._head_dyn_random_change_count += 1

        if not should_trigger:
            return

        transition = self._try_build_head_dynamic_transition(frame_idx=frame_idx, reason=reason)
        if transition is not None:
            self._head_dyn_last_trigger_frame = frame_idx
            self._head_dyn_transition = transition

    def update_picture(self):
        # camera
        if self.collect_wrist_camera:
            self.left_camera.take_picture()
            self.right_camera.take_picture()

        for camera in self.static_camera_list:
            camera.take_picture()

        # ================================= sensor camera =================================
        # self.head_sensor.take_picture()
        # self.head_sensor.compute_depth()

    def update_wrist_camera(self, left_pose, right_pose):
        """
        Update rendering to refresh the camera's RGBD information
        (rendering must be updated even when disabled, otherwise data cannot be collected).
        """
        if self.collect_wrist_camera:
            self.left_camera.entity.set_pose(left_pose)
            self.right_camera.entity.set_pose(right_pose)

    def get_config(self) -> dict:
        res = {}

        def _get_config(camera):
            camera_intrinsic_cv = camera.get_intrinsic_matrix()
            camera_extrinsic_cv = camera.get_extrinsic_matrix()
            camera_model_matrix = camera.get_model_matrix()
            return {
                "intrinsic_cv": camera_intrinsic_cv,
                "extrinsic_cv": camera_extrinsic_cv,
                "cam2world_gl": camera_model_matrix,
            }

        if self.collect_wrist_camera:
            res["left_camera"] = _get_config(self.left_camera)
            res["right_camera"] = _get_config(self.right_camera)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = _get_config(camera)
            else:
                res[camera_name] = _get_config(camera)
        # ================================= sensor camera =================================
        # res['head_sensor'] = res['head_camera']
        # print(res)
        return res

    def get_head_camera_episode_record(self) -> dict:
        if self.head_camera_id is None:
            return {}

        result = dict(self.head_camera_episode_record)
        result["head_camera_dynamic_events"] = list(self.head_camera_dynamic_events)
        result["head_camera_dynamic_summary"] = {
            "forced_change_count": int(self._head_dyn_forced_change_count),
            "random_change_count": int(self._head_dyn_random_change_count),
            "accepted_change_count": int(sum(1 for e in self.head_camera_dynamic_events if e.get("accepted", False))),
            "rejected_change_count": int(sum(1 for e in self.head_camera_dynamic_events if not e.get("accepted", False))),
        }
        try:
            head_camera = self.static_camera_list[self.head_camera_id]
            
            result["head_camera_final_params"] = {
                "intrinsic_cv": np.asarray(head_camera.get_intrinsic_matrix(), dtype=np.float64).tolist(),
                "extrinsic_cv": np.asarray(head_camera.get_extrinsic_matrix(), dtype=np.float64).tolist(),
                "cam2world_gl": np.asarray(head_camera.get_model_matrix(), dtype=np.float64).tolist(),
            }
        except Exception:
            result["head_camera_final_params"] = {}

        return result

    def check_head_camera_visibility(self):
        """
        Check whether head camera frustum is severely occluded.
        Returns:
            ok (bool), metrics (dict)
        """
        # if self.head_camera_id is None or not self.collect_head_camera:
        #     metrics = {
        #         "enabled": bool(self.enable_head_camera_occlusion_check),
        #         "checked": False,
        #         "ok": True,
        #         "reason": "head_camera_not_collected",
        #     }
        #     self.head_camera_episode_record["head_camera_visibility_check"] = metrics
        #     return True, metrics

        # if not self.enable_head_camera_occlusion_check:
        #     metrics = {
        #         "enabled": False,
        #         "checked": False,
        #         "ok": True,
        #         "reason": "disabled_by_config",
        #     }
        #     self.head_camera_episode_record["head_camera_visibility_check"] = metrics
        #     return True, metrics

        # head_camera = self.static_camera_list[self.head_camera_id]
        # head_camera.take_picture()
        # position = head_camera.get_picture("Position")

        # valid_mask = position[..., 3] < 1
        # depth = -position[..., 2]

        # total_pixels = int(valid_mask.size)
        # valid_pixels = int(np.count_nonzero(valid_mask))
        # valid_ratio = float(valid_pixels / max(total_pixels, 1))

        # near_depth = self.head_camera_occlusion_near_depth_m
        # near_pixels = int(np.count_nonzero(valid_mask & (depth < near_depth)))
        # near_ratio = float(near_pixels / max(valid_pixels, 1))

        # valid_ratio_ok = valid_ratio >= self.head_camera_occlusion_valid_ratio_min
        # near_ratio_ok = near_ratio <= self.head_camera_occlusion_near_ratio_max
        # ok = bool(valid_ratio_ok and near_ratio_ok)

        # metrics = {
        #     "enabled": True,
        #     "checked": True,
        #     "ok": ok,
        #     "threshold": {
        #         "valid_ratio_min": float(self.head_camera_occlusion_valid_ratio_min),
        #         "near_ratio_max": float(self.head_camera_occlusion_near_ratio_max),
        #         "near_depth_m": float(self.head_camera_occlusion_near_depth_m),
        #     },
        #     "measured": {
        #         "total_pixels": total_pixels,
        #         "valid_pixels": valid_pixels,
        #         "valid_ratio": valid_ratio,
        #         "near_pixels": near_pixels,
        #         "near_ratio": near_ratio,
        #     },
        # }
        ok, metrics = self._check_head_camera_visibility_internal()
        self.head_camera_episode_record["head_camera_visibility_check"] = metrics
        return ok, metrics

    def get_rgb(self) -> dict:
        rgba = self.get_rgba()
        rgb = {}
        for camera_name, camera_data in rgba.items():
            rgb[camera_name] = {}
            rgb[camera_name]["rgb"] = camera_data["rgba"][:, :, :3]  # Exclude alpha channel
        return rgb
    
    # Get Camera RGBA
    def get_rgba(self) -> dict:

        def _get_rgba(camera):
            camera_rgba = camera.get_picture("Color")
            camera_rgba_img = (camera_rgba * 255).clip(0, 255).astype("uint8")
            return camera_rgba_img

        # ================================= sensor camera =================================
        # def _get_sensor_rgba(sensor):
        #     camera_rgba = sensor.get_rgb()
        #     camera_rgba_img = (camera_rgba * 255).clip(0, 255).astype("uint8")[:,:,:3]
        #     return camera_rgba_img

        res = {}

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"]["rgba"] = _get_rgba(self.left_camera)
            res["right_camera"]["rgba"] = _get_rgba(self.right_camera)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name]["rgba"] = _get_rgba(camera)
            else:
                res[camera_name] = {}
                res[camera_name]["rgba"] = _get_rgba(camera)
        # ================================= sensor camera =================================
        # res['head_sensor']['rgb'] = _get_sensor_rgba(self.head_sensor)

        return res

    def get_observer_rgb(self) -> dict:
        self.observer_camera.take_picture()

        def _get_rgb(camera):
            camera_rgba = camera.get_picture("Color")
            camera_rgb_img = (camera_rgba * 255).clip(0, 255).astype("uint8")[:, :, :3]
            return camera_rgb_img

        return _get_rgb(self.observer_camera)

    # Get Camera Segmentation
    def get_segmentation(self, level="mesh") -> dict:

        def _get_segmentation(camera, level="mesh"):
            # visual_id is the unique id of each visual shape
            seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
            colormap = sorted(set(ImageColor.colormap.values()))
            color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
            if level == "mesh":
                label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
            elif level == "actor":
                label0_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
            return color_palette[label0_image]

        res = {
            # 'left_camera':{},
            # 'right_camera':{}
        }

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"][f"{level}_segmentation"] = _get_segmentation(self.left_camera, level=level)
            res["right_camera"][f"{level}_segmentation"] = _get_segmentation(self.right_camera, level=level)

        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name][f"{level}_segmentation"] = _get_segmentation(camera, level=level)
            else:
                res[camera_name] = {}
                res[camera_name][f"{level}_segmentation"] = _get_segmentation(camera, level=level)
        return res

    # Get Camera Depth
    def get_depth(self) -> dict:

        def _get_depth(camera):
            position = camera.get_picture("Position")
            depth = -position[..., 2]
            depth_image = (depth * 1000.0).astype(np.float64)
            return depth_image

        def _get_sensor_depth(sensor):
            depth = sensor.get_depth()
            depth = (depth * 1000.0).astype(np.float64)
            return depth

        res = {}
        rgba = self.get_rgba()

        if self.collect_wrist_camera:
            res["left_camera"] = {}
            res["right_camera"] = {}
            res["left_camera"]["depth"] = _get_depth(self.left_camera)
            res["right_camera"]["depth"] = _get_depth(self.right_camera)
            res["left_camera"]["depth"] *= rgba["left_camera"]["rgba"][:, :, 3] / 255
            res["right_camera"]["depth"] *= rgba["right_camera"]["rgba"][:, :, 3] / 255
        
        for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
            if camera_name == "head_camera":
                if self.collect_head_camera:
                    res[camera_name] = {}
                    res[camera_name]["depth"] = _get_depth(camera)
                    res[camera_name]["depth"] *= rgba[camera_name]["rgba"][:, :, 3] / 255
            else:
                res[camera_name] = {}
                res[camera_name]["depth"] = _get_depth(camera)
                res[camera_name]["depth"] *= rgba[camera_name]["rgba"][:, :, 3] / 255
        # res['head_sensor']['depth'] = _get_sensor_depth(self.head_sensor)

        return res

    # Get World PointCloud
    def get_world_pcd(self):
        self.world_camera1.take_picture()
        self.world_camera2.take_picture()

        def _get_camera_pcd(camera, color=True):
            rgba = camera.get_picture_cuda("Color").torch()  # [H, W, 4]
            position = camera.get_picture_cuda("Position").torch()
            model_matrix = camera.get_model_matrix()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_matrix = torch.tensor(model_matrix, dtype=torch.float32).to(device)

            # Extract valid three-dimensional points and corresponding color data.
            valid_mask = position[..., 3] < 1
            points_opengl = position[..., :3][valid_mask]
            points_color = rgba[valid_mask][:, :3]
            # Transform into the world coordinate system.
            points_world = (torch.bmm(
                points_opengl.view(1, -1, 3),
                model_matrix[:3, :3].transpose(0, 1).view(-1, 3, 3),
            ).squeeze(1) + model_matrix[:3, 3])

            # Format color data.
            points_color = torch.clamp(points_color, 0, 1)
            points_world = points_world.squeeze(0)

            # Convert the tensor back to a NumPy array for use with Open3D.
            points_world_np = points_world.cpu().numpy()
            points_color_np = points_color.cpu().numpy()
            # print(points_world_np.shape, points_color_np.shape)

            res_pcd = (np.hstack((points_world_np, points_color_np)) if color else points_world_np)
            return res_pcd

        pcd1 = _get_camera_pcd(self.world_camera1, color=True)
        pcd2 = _get_camera_pcd(self.world_camera2, color=True)
        res_pcd = np.vstack((pcd1, pcd2))

        return res_pcd
        pcd_array, index = fps(res_pcd[:, :3], 2000)
        index = index.detach().cpu().numpy()[0]

        return pcd_array

    # Get Camera PointCloud
    def get_pcd(self, if_combine=False):

        def _get_camera_pcd(camera, point_num=0):
            rgba = camera.get_picture_cuda("Color").torch()  # [H, W, 4]
            position = camera.get_picture_cuda("Position").torch()
            model_matrix = camera.get_model_matrix()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_matrix = torch.tensor(model_matrix, dtype=torch.float32).to(device)

            # Extract valid three-dimensional points and corresponding color data.
            valid_mask = position[..., 3] < 1
            points_opengl = position[..., :3][valid_mask]
            points_color = rgba[valid_mask][:, :3]
            # Transform into the world coordinate system.
            points_world = (torch.bmm(
                points_opengl.view(1, -1, 3),
                model_matrix[:3, :3].transpose(0, 1).view(-1, 3, 3),
            ).squeeze(1) + model_matrix[:3, 3])

            # Format color data.
            points_color = torch.clamp(points_color, 0, 1)

            points_world = points_world.squeeze(0)

            # If crop is needed
            if self.pcd_crop:
                min_bound = torch.tensor(self.pcd_crop_bbox[0], dtype=torch.float32).to(device)
                max_bound = torch.tensor(self.pcd_crop_bbox[1], dtype=torch.float32).to(device)
                inside_bounds_mask = (points_world.squeeze(0) >= min_bound).all(dim=1) & (points_world.squeeze(0)
                                                                                          <= max_bound).all(dim=1)
                points_world = points_world[inside_bounds_mask]
                points_color = points_color[inside_bounds_mask]

            # Convert the tensor back to a NumPy array for use with Open3D.
            points_world_np = points_world.cpu().numpy()
            points_color_np = points_color.cpu().numpy()

            if point_num > 0:
                # points_world_np,index = fps(points_world_np,point_num)
                index = index.detach().cpu().numpy()[0]
                points_color_np = points_color_np[index, :]

            return np.hstack((points_world_np, points_color_np))

        if self.head_camera_id is None:
            print("No head camera in static camera list, pointcloud save error!")
            return None

        combined_pcd = np.array([])

        # Merge pointcloud
        if if_combine:
            # combined_pcd = np.vstack((head_pcd , left_pcd , right_pcd, front_pcd))
            if self.collect_wrist_camera:
                combined_pcd = np.vstack((
                    _get_camera_pcd(self.left_camera),
                    _get_camera_pcd(self.right_camera),
                ))
            for camera, camera_name in zip(self.static_camera_list, self.static_camera_name):
                if camera_name == "head_camera":
                    if self.collect_head_camera:
                        combined_pcd = np.vstack((combined_pcd, _get_camera_pcd(camera)))
                else:
                    combined_pcd = np.vstack((combined_pcd, _get_camera_pcd(camera)))
        elif self.collect_head_camera:
            combined_pcd = _get_camera_pcd(self.static_camera_list[self.head_camera_id])
        
        def pad_array(numpy_array, target_num):
            current_num = numpy_array.shape[0]
            if current_num < target_num:
                if current_num == 0:
                    numpy_array = np.zeros((target_num, 6))
                else:
                    pad_num = target_num - current_num
                    padding = np.zeros((pad_num, 6))
                    numpy_array = np.vstack([numpy_array, padding])
            return numpy_array

        combined_pcd = pad_array(combined_pcd, self.pcd_down_sample_num)

        pcd_array, index = combined_pcd[:, :3], np.array(range(len(combined_pcd)))

        if self.pcd_down_sample_num > 0:
            pcd_array, index = fps(combined_pcd[:, :3], self.pcd_down_sample_num)
            index = index.detach().cpu().numpy()[0]

        return combined_pcd[index]