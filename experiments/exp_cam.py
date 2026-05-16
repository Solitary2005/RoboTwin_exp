import os
import math
import yaml
import sys
import importlib
import argparse

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
    with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def build_args(task_name, task_config, render_freq):
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["render_freq"] = render_freq
    args["save_data"] = False
    args["collect_data"] = False
    args["eval_mode"] = False
    args["need_plan"] = True

    # 解析 embodiment
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb_map = yaml.load(f.read(), Loader=yaml.FullLoader)

    emb = args.get("embodiment")
    if len(emb) == 1:
        args["left_robot_file"] = emb_map[emb[0]]["file_path"]
        args["right_robot_file"] = emb_map[emb[0]]["file_path"]
        args["dual_arm_embodied"] = True
    elif len(emb) == 3:
        args["left_robot_file"] = emb_map[emb[0]]["file_path"]
        args["right_robot_file"] = emb_map[emb[1]]["file_path"]
        args["embodiment_dis"] = emb[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment error")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def print_camera_info(task):
    # 先刷新渲染再拍图
    task.scene.update_render()

    cameras = []

    if task.cameras.collect_wrist_camera:
        cameras.append(("left_camera", task.cameras.left_camera))
        cameras.append(("right_camera", task.cameras.right_camera))

    for cam, name in zip(task.cameras.static_camera_list, task.cameras.static_camera_name):
        cameras.append((name, cam))

    # cameras.append(("observer_camera", task.cameras.observer_camera))
    # cameras.append(("world_camera1", task.cameras.world_camera1))
    # cameras.append(("world_camera2", task.cameras.world_camera2))

    print("\n========== Camera Info ==========")
    for name, cam in cameras:
        cam.take_picture()
        rgba = cam.get_picture("Color")
        h, w = rgba.shape[0], rgba.shape[1]

        K = cam.get_intrinsic_matrix()
        fx = K[0, 0]
        fy = K[1, 1]
        fovx = 2.0 * math.degrees(math.atan(w / (2.0 * fx)))
        fovy = 2.0 * math.degrees(math.atan(h / (2.0 * fy)))

        pose = cam.entity.get_pose()  # world pose
        print(f"\n[{name}]")
        print(f"  resolution: {w}x{h}")
        print(f"  fovx/fovy (deg): {fovx:.2f} / {fovy:.2f}")
        print(f"  position xyz: {pose.p}")
        print(f"  quaternion wxyz: {pose.q}")
        print(f"  intrinsic K:\n{K}")
        print(f"  extrinsic (cv):\n{cam.get_extrinsic_matrix()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="adjust_bottle")
    parser.add_argument("--task_config", type=str, default="exp_camera")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render_freq", type=int, default=1)
    parser.add_argument("--play_once", action="store_true")
    args_cmd = parser.parse_args()

    task = class_decorator(args_cmd.task_name)
    args = build_args(args_cmd.task_name, args_cmd.task_config, args_cmd.render_freq)

    try:
        task.setup_demo(now_ep_num=0, seed=args_cmd.seed, **args)
        print_camera_info(task)

        if args_cmd.play_once:
            print("\nRunning play_once() ...")
            task.play_once()

        print("\nViewer opened")
        while not task.viewer.closed:
            task.scene.step()
            task._update_render()
            task.viewer.render()

    finally:
        try:
            task.close_env()
        except Exception:
            pass


if __name__ == "__main__":
    main()