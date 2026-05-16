import os
import yaml
import json
import sys

import argparse
import importlib
import numpy as np
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 先把工作目录切到项目根
os.chdir(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import CONFIGS_PATH

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def build_args(task_name, task_config):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["save_data"] = False
    args["collect_data"] = False
    args["need_plan"] = True
    args["render_freq"] = 0  # 不开窗口，只拿观测

    # 确保会返回 RGB observation
    if "data_type" not in args:
        args["data_type"] = {}
    args["data_type"]["rgb"] = True

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(emb):
        robot_file = emb_types[emb]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def to_uint8_rgb(img):
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_obs_rgb(obs, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    cam_keys = ["head_camera", "left_camera", "right_camera"]
    saved = []

    observation = obs.get("observation", {})
    for cam in cam_keys:
        if cam not in observation:
            print(f"[WARN] {cam} not in observation")
            continue
        if "rgb" not in observation[cam]:
            print(f"[WARN] {cam}.rgb not found; check data_type.rgb=True")
            continue

        rgb = to_uint8_rgb(observation[cam]["rgb"])
        save_path = os.path.join(out_dir, f"{prefix}_{cam}.png")
        Image.fromarray(rgb).save(save_path)
        saved.append(save_path)

    # 额外保存相机位姿信息，方便对比你改动前后
    cam_meta = {}
    for cam in cam_keys:
        if cam in observation:
            cam_meta[cam] = {
                "intrinsic_cv": observation[cam].get("intrinsic_cv", None).tolist()
                if isinstance(observation[cam].get("intrinsic_cv", None), np.ndarray) else observation[cam].get("intrinsic_cv", None),
                "extrinsic_cv": observation[cam].get("extrinsic_cv", None).tolist()
                if isinstance(observation[cam].get("extrinsic_cv", None), np.ndarray) else observation[cam].get("extrinsic_cv", None),
                "cam2world_gl": observation[cam].get("cam2world_gl", None).tolist()
                if isinstance(observation[cam].get("cam2world_gl", None), np.ndarray) else observation[cam].get("cam2world_gl", None),
            }

    with open(os.path.join(out_dir, f"{prefix}_camera_meta.json"), "w", encoding="utf-8") as f:
        json.dump(cam_meta, f, ensure_ascii=False, indent=2)

    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="adjust_bottle")
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./debug_cam_obs")
    parser.add_argument("--after_play_once", action="store_true")
    args_cmd = parser.parse_args()

    env = class_decorator(args_cmd.task_name)
    args = build_args(args_cmd.task_name, args_cmd.task_config)
    out_dir = os.path.join(args_cmd.out_dir, args_cmd.task_name, args_cmd.task_config)

    try:
        env.setup_demo(now_ep_num=0, seed=args_cmd.seed, **args)

        # 初始化后先抓一帧，观察“初始视角”
        obs0 = env.get_obs()
        saved0 = save_obs_rgb(obs0, out_dir, "init")
        print("Saved init images:")
        for p in saved0:
            print(" -", p)

        # 可选：执行一次任务动作后再抓一帧
        if args_cmd.after_play_once:
            env.play_once()
            obs1 = env.get_obs()
            saved1 = save_obs_rgb(obs1, out_dir, "after_play_once")
            print("Saved after_play_once images:")
            for p in saved1:
                print(" -", p)

    finally:
        try:
            env.close_env()
        except Exception:
            pass

# python test_cam_play_once.py --task_name adjust_bottle --task_config exp_camera --seed 0 --after_play_once
if __name__ == "__main__":
    main()