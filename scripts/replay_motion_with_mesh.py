import argparse
import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

# Ensure project modules are importable when running from scripts/
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PKG_DIR = os.path.join(PROJECT_ROOT, "open_duck_reference_motion_generator")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from placo_walk_engine import PlacoWalkEngine
from placo_utils.visualization import robot_viz


def main():
    parser = argparse.ArgumentParser(description="Replay a motion JSON with robot mesh (MeshCat)")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to motion JSON file")
    parser.add_argument(
        "--duck",
        choices=["go_bdx", "open_duck_mini", "open_duck_mini_v2"],
        required=True,
        help="Robot type to load",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override playback FPS")
    args = parser.parse_args()

    # Load episode
    with open(args.file, "r") as f:
        episode = json.load(f)

    frames = episode["Frames"]
    frame_offsets = episode["Frame_offset"][0]
    joints_names = episode.get("Joints", [])

    frame_duration = 1.0 / args.fps if args.fps else episode.get("FrameDuration", 1.0 / 50.0)

    # Slices
    root_pos_slice = slice(frame_offsets["root_pos"], frame_offsets["root_quat"])
    root_quat_slice = slice(frame_offsets["root_quat"], frame_offsets["joints_pos"])
    joints_pos_slice = slice(frame_offsets["joints_pos"], frame_offsets["left_toe_pos"])

    # Build robot from URDF using PlacoWalkEngine (for consistent joint naming/order)
    script_path = os.path.dirname(os.path.abspath(__file__))
    asset_path = os.path.join(
        script_path,
        f"../open_duck_reference_motion_generator/robots/{args.duck}",
    )
    robot_urdf = f"{args.duck}.urdf"

    # Load gait parameters (defaults) to initialize Placo
    defaults_path = os.path.join(asset_path, "placo_defaults.json")
    with open(defaults_path, "r") as gpf:
        gait_parameters = json.load(gpf)

    pwe = PlacoWalkEngine(asset_path, robot_urdf, gait_parameters)
    viz = robot_viz(pwe.robot)

    # Replay loop: set base pose and joint angles, then display
    for frame in frames:
        root_position = frame[root_pos_slice]
        root_quat = frame[root_quat_slice]
        joints_pos = frame[joints_pos_slice]

        # Build SE3 from root position + quaternion (xyzw)
        T = np.eye(4)
        T[:3, 3] = root_position
        T[:3, :3] = R.from_quat(root_quat).as_matrix()

        # Update robot base and joints
        try:
            pwe.robot.set_T_world_fbase(T)
        except Exception:
            # If fbase setter is not available, try trunk as fallback
            if hasattr(pwe.robot, "set_T_world_trunk"):
                pwe.robot.set_T_world_trunk(T)
            else:
                raise

        for name, value in zip(joints_names, joints_pos):
            pwe.robot.set_joint(name, value)

        viz.display(pwe.robot.state.q)
        time.sleep(frame_duration)


if __name__ == "__main__":
    main()


