import argparse
import json
import math
import os
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

# Make project modules importable when running from scripts/
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PKG_DIR = os.path.join(PROJECT_ROOT, "open_duck_reference_motion_generator")
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from placo_walk_engine import PlacoWalkEngine


def compute_angular_velocity(quat: List[float], prev_quat: List[float], dt: float) -> List[float]:
    if prev_quat is None:
        prev_quat = quat
    r1 = R.from_quat(quat)
    r0 = R.from_quat(prev_quat)
    r_rel = r0.inv() * r1
    axis_angle = r_rel.as_rotvec()
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-9:
        return [0.0, 0.0, 0.0]
    axis = axis_angle / (angle + 1e-12)
    ang_vel = axis * (angle / dt)
    return ang_vel.tolist()


def build_episode_skeleton(fps: int) -> dict:
    return {
        "LoopMode": "Wrap",
        "FPS": fps,
        "FrameDuration": float(np.around(1 / fps, 4)),
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": False,
        "Joints": [],
        "Frame_offset": [],
        "Frame_size": [],
        "Frames": [],
        "MotionWeight": 1,
    }


def get_asset_path(duck: str) -> str:
    return os.path.join(PROJECT_ROOT, f"open_duck_reference_motion_generator/robots/{duck}")


def lock_feet_tasks_to_current_world(pwe: PlacoWalkEngine, hard_weight: float = 1.0) -> None:
    T_world_left = pwe.robot.get_T_world_left().copy()
    T_world_right = pwe.robot.get_T_world_right().copy()

    lt = pwe.tasks.left_foot_task
    rt = pwe.tasks.right_foot_task

    lt.position().target_world = T_world_left[:3, 3]
    lt.orientation().R_world_frame = T_world_left[:3, :3]
    lt.configure("left_foot", "hard", hard_weight)

    rt.position().target_world = T_world_right[:3, 3]
    rt.orientation().R_world_frame = T_world_right[:3, :3]
    rt.configure("right_foot", "hard", hard_weight)


def set_feet_task_weight(pwe: PlacoWalkEngine, mode: str, weight: float) -> None:
    pwe.tasks.left_foot_task.configure("left_foot", mode, weight)
    pwe.tasks.right_foot_task.configure("right_foot", mode, weight)


def move_to_upright_start(pwe: PlacoWalkEngine, com_height_offset: float = 0.01, trunk_pitch_deg: float = 0.0) -> None:
    # Lock current feet to prevent drift during alignment
    lock_feet_tasks_to_current_world(pwe, hard_weight=1.0)

    T0 = pwe.robot.get_T_world_fbase().copy()
    R0 = T0[:3, :3]
    t0 = T0[:3, 3].copy()
    r0, p0, y0 = R.from_matrix(R0).as_euler("xyz")

    target_pitch = math.radians(trunk_pitch_deg)
    dz = float(max(0.0, com_height_offset))

    # Temporarily relax joint limits to allow near-upright alignment
    for jn in ["left_knee", "right_knee", "left_ankle", "right_ankle", "left_hip_pitch", "right_hip_pitch"]:
        try:
            pwe.robot.set_joint_limits(jn, -0.8, 0.8)
        except Exception:
            pass

    # Capture current joints to blend towards neutral
    def getj(name):
        try:
            return pwe.robot.get_joint(name)
        except Exception:
            return 0.0

    j0 = {
        "left_knee": getj("left_knee"),
        "right_knee": getj("right_knee"),
        "left_ankle": getj("left_ankle"),
        "right_ankle": getj("right_ankle"),
        "left_hip_pitch": getj("left_hip_pitch"),
        "right_hip_pitch": getj("right_hip_pitch"),
    }

    # Gradually align roll->0, pitch->target, keep yaw; lift slightly in z
    steps = 20
    for k in range(steps + 1):
        s = k / steps
        r = (1.0 - s) * r0 + s * 0.0
        p = (1.0 - s) * p0 + s * target_pitch
        y = y0  # keep yaw
        Rw = R.from_euler("xyz", [r, p, y]).as_matrix()
        tw = t0 + np.array([0.0, 0.0, s * dz])
        update_trunk_targets(pwe, Rw, tw, pos_mode="soft", pos_weight=0.3, ori_mode="soft", ori_weight=0.6)
        # Blend leg joints towards neutral
        for name in j0.keys():
            try:
                target = (1.0 - s) * j0[name] + s * 0.0
                pwe.joints_task.set_joint(name, target)
            except Exception:
                pass
        solve_until_converged(pwe, iters=4)


def update_trunk_targets(
    pwe: PlacoWalkEngine,
    R_world: np.ndarray,
    t_world: np.ndarray,
    pos_mode: str = "soft",
    pos_weight: float = 0.2,
    ori_mode: str = "soft",
    ori_weight: float = 0.2,
) -> None:
    pwe.tasks.trunk_task.target_world = t_world
    pwe.tasks.trunk_task.configure("trunk_position", pos_mode, pos_weight)
    pwe.tasks.trunk_orientation_task.R_world_frame = R_world
    pwe.tasks.trunk_orientation_task.configure("trunk_orientation", ori_mode, ori_weight)


def solve_until_converged(pwe: PlacoWalkEngine, iters: int = 16) -> None:
    for _ in range(iters):
        pwe.robot.update_kinematics()
        _ = pwe.solver.solve(True)


def pack_episode(frames: List[List[float]], joints_order: List[str]) -> dict:
    episode = {}
    episode["Frames"] = frames
    # Build offsets from first frame
    f0 = frames[0]
    NJ = len(joints_order)
    root_position = f0[0:3]
    root_quat = f0[3:7]
    joints_pos = f0[7 : 7 + NJ]
    left_toe_pos = f0[7 + NJ : 7 + NJ + 3]
    right_toe_pos = f0[7 + NJ + 3 : 7 + NJ + 6]
    world_linear_vel = f0[7 + NJ + 6 : 7 + NJ + 9]
    world_angular_vel = f0[7 + NJ + 9 : 7 + NJ + 12]
    joints_vel = f0[7 + NJ + 12 : 7 + NJ + 12 + NJ]
    left_toe_vel = f0[7 + NJ + 12 + NJ : 7 + NJ + 15 + NJ]
    right_toe_vel = f0[7 + NJ + 15 + NJ : 7 + NJ + 18 + NJ]

    offsets = {}
    sizes = {}
    offset = 0

    def push(key: str, arr: List[float]):
        nonlocal offset
        offsets[key] = offset
        sizes[key] = len(arr)
        offset += len(arr)

    push("root_pos", root_position)
    push("root_quat", root_quat)
    push("joints_pos", joints_pos)
    push("left_toe_pos", left_toe_pos)
    push("right_toe_pos", right_toe_pos)
    push("world_linear_vel", world_linear_vel)
    push("world_angular_vel", world_angular_vel)
    push("joints_vel", joints_vel)
    push("left_toe_vel", left_toe_vel)
    push("right_toe_vel", right_toe_vel)
    push("foot_contacts", [1, 1])

    return offsets, sizes


def record_loop(pwe: PlacoWalkEngine, duration: float, fps: int, frame_updater) -> dict:
    dt = 1.0 / fps
    frames: List[List[float]] = []
    joints_order = list(pwe.get_angles().keys())

    prev_root_pos = None
    prev_root_quat = None
    prev_joints = None
    prev_left_toe = None
    prev_right_toe = None

    for i in range(int(duration * fps)):
        t = i * dt
        frame_updater(t)
        solve_until_converged(pwe, iters=8)

        T_world_fbase = pwe.robot.get_T_world_fbase()
        root_position = T_world_fbase[:3, 3].tolist()
        root_quat = R.from_matrix(T_world_fbase[:3, :3]).as_quat().tolist()
        joints_pos = [pwe.robot.get_joint(j) for j in joints_order]

        T_world_left = pwe.robot.get_T_world_left()
        T_world_right = pwe.robot.get_T_world_right()
        T_body_left = np.linalg.inv(T_world_fbase) @ T_world_left
        T_body_right = np.linalg.inv(T_world_fbase) @ T_world_right
        left_toe_pos = T_body_left[:3, 3].tolist()
        right_toe_pos = T_body_right[:3, 3].tolist()

        if i == 0:
            world_linear_vel = [0.0, 0.0, 0.0]
            world_angular_vel = [0.0, 0.0, 0.0]
            joints_vel = [0.0 for _ in joints_pos]
            left_toe_vel = [0.0, 0.0, 0.0]
            right_toe_vel = [0.0, 0.0, 0.0]
        else:
            world_linear_vel = ((np.array(root_position) - np.array(prev_root_pos)) / dt).tolist()
            world_angular_vel = compute_angular_velocity(root_quat, prev_root_quat, dt)
            joints_vel = ((np.array(joints_pos) - np.array(prev_joints)) / dt).tolist()
            left_toe_vel = ((np.array(left_toe_pos) - np.array(prev_left_toe)) / dt).tolist()
            right_toe_vel = ((np.array(right_toe_pos) - np.array(prev_right_toe)) / dt).tolist()

        contacts = [1, 1]

        frames.append(
            root_position
            + root_quat
            + joints_pos
            + left_toe_pos
            + right_toe_pos
            + world_linear_vel
            + world_angular_vel
            + joints_vel
            + left_toe_vel
            + right_toe_vel
            + contacts
        )

        prev_root_pos = root_position
        prev_root_quat = root_quat
        prev_joints = joints_pos
        prev_left_toe = left_toe_pos
        prev_right_toe = right_toe_pos

    episode = build_episode_skeleton(fps)
    episode["Joints"] = joints_order
    offsets, sizes = pack_episode(frames, joints_order)
    episode["Frame_offset"].append(offsets)
    episode["Frame_size"].append(sizes)
    episode["Frames"] = frames
    return episode


def generate_body_sway(pwe: PlacoWalkEngine, duration: float, fps: int, params: dict | None) -> dict:
    amp_deg = params.get("sway_amplitude_deg", 6.0) if params else 6.0
    freq = params.get("sway_frequency_hz", 0.6) if params else 0.6
    y_amp = params.get("lateral_shift_m", 0.01) if params else 0.01
    head_gain = params.get("head_roll_gain", 0.3) if params else 0.3

    move_to_upright_start(pwe, com_height_offset=0.02, trunk_pitch_deg=0.0)
    lock_feet_tasks_to_current_world(pwe, hard_weight=1.0)

    T_base0 = pwe.robot.get_T_world_fbase().copy()
    R0 = T_base0[:3, :3]
    t0 = T_base0[:3, 3].copy()

    # Do not force knees; keep current initial pose

    def updater(t: float):
        phi = math.radians(amp_deg) * math.sin(2 * math.pi * freq * t)
        y_shift = y_amp * math.sin(2 * math.pi * freq * t + math.pi / 2)
        R_world = R.from_euler("x", phi).as_matrix() @ R0
        t_world = t0 + np.array([0.0, y_shift, 0.0])
        # Stronger orientation target to create visible sway, still soft to stay feasible
        update_trunk_targets(pwe, R_world, t_world, pos_mode="soft", pos_weight=0.15, ori_mode="soft", ori_weight=0.6)
        if "head_roll" in pwe.get_angles():
            pwe.joints_task.set_joint("head_roll", head_gain * phi)

    return record_loop(pwe, duration, fps, updater)


def generate_head_roll(pwe: PlacoWalkEngine, duration: float, fps: int, params: dict | None) -> dict:
    amp_deg = params.get("amplitude_deg", 18.0) if params else 18.0
    freq = params.get("frequency_hz", 0.7) if params else 0.7

    move_to_upright_start(pwe, com_height_offset=0.02, trunk_pitch_deg=0.0)
    lock_feet_tasks_to_current_world(pwe, hard_weight=1.0)

    T_base0 = pwe.robot.get_T_world_fbase().copy()
    R0 = T_base0[:3, :3]
    t0 = T_base0[:3, 3].copy()

    def updater(t: float):
        phi = math.radians(amp_deg) * math.sin(2 * math.pi * freq * t)
        # Let trunk follow slightly (roly-poly effect)
        R_world = R.from_euler("x", 0.3 * phi).as_matrix() @ R0
        update_trunk_targets(pwe, R_world, t0, pos_mode="soft", pos_weight=0.1, ori_mode="soft", ori_weight=0.5)
        if "head_roll" in pwe.get_angles():
            pwe.joints_task.set_joint("head_roll", phi)

    return record_loop(pwe, duration, fps, updater)


def generate_head_pitch(pwe: PlacoWalkEngine, duration: float, fps: int, params: dict | None) -> dict:
    amp_deg = params.get("amplitude_deg", 16.0) if params else 16.0
    freq = params.get("frequency_hz", 0.5) if params else 0.5

    move_to_upright_start(pwe, com_height_offset=0.02, trunk_pitch_deg=0.0)
    lock_feet_tasks_to_current_world(pwe, hard_weight=1.0)

    T_base0 = pwe.robot.get_T_world_fbase().copy()
    R0 = T_base0[:3, :3]
    t0 = T_base0[:3, 3].copy()

    def updater(t: float):
        theta = math.radians(amp_deg) * math.sin(2 * math.pi * freq * t)
        # Trunk pitches slightly to follow head (soft)
        R_world = R.from_euler("y", 0.25 * theta).as_matrix() @ R0
        update_trunk_targets(pwe, R_world, t0, pos_mode="soft", pos_weight=0.1, ori_mode="soft", ori_weight=0.5)
        if "head_pitch" in pwe.get_angles():
            pwe.joints_task.set_joint("head_pitch", theta)

    return record_loop(pwe, duration, fps, updater)


def generate_jump(pwe: PlacoWalkEngine, duration: float, fps: int, params: dict | None) -> dict:
    cycle_t = (params.get("cycle_t", 1.2) if params else 1.2)
    crouch_t = (params.get("crouch_t", 0.4) if params else 0.4)
    launch_t = (params.get("launch_t", 0.15) if params else 0.15)
    flight_t = (params.get("flight_t", 0.3) if params else 0.3)

    # Keep within conservative ranges to avoid conflicts with locked feet
    max_knee_flex = (params.get("max_knee_flex", -0.2) if params else -0.2)
    max_ankle = (params.get("max_ankle", 0.1) if params else 0.1)
    max_hip = (params.get("max_hip", 0.15) if params else 0.15)
    max_z = (params.get("max_z", 0.0) if params else 0.0)

    move_to_upright_start(pwe, com_height_offset=0.02, trunk_pitch_deg=0.0)
    lock_feet_tasks_to_current_world(pwe, hard_weight=1.0)
    T_base0 = pwe.robot.get_T_world_fbase().copy()
    R0 = T_base0[:3, :3]
    t0 = T_base0[:3, 3].copy()

    def set_legs(knee: float, hip: float, ankle: float):
        for jn in ["left_knee", "right_knee"]:
            if jn in pwe.get_angles():
                pwe.joints_task.set_joint(jn, knee)
        for jn in ["left_hip_pitch", "right_hip_pitch"]:
            if jn in pwe.get_angles():
                pwe.joints_task.set_joint(jn, hip)
        for jn in ["left_ankle", "right_ankle"]:
            if jn in pwe.get_angles():
                pwe.joints_task.set_joint(jn, -ankle)

    dt = 1.0 / fps

    def updater(tabs: float):
        t = tabs % cycle_t
        if t < crouch_t:
            s = t / crouch_t
            set_feet_task_weight(pwe, "hard", 1.0)
            knee = max_knee_flex * s
            hip = max_hip * s
            ankle = max_ankle * s
            z = max_z * 0.2 * s
        elif t < crouch_t + launch_t:
            s = (t - crouch_t) / launch_t
            set_feet_task_weight(pwe, "hard", 1.0)
            knee = max_knee_flex * (1 - s)
            hip = max_hip * (1 - s)
            ankle = max_ankle * (1 - s)
            z = max_z * (0.2 + 0.5 * s)
        elif t < crouch_t + launch_t + flight_t:
            s = (t - crouch_t - launch_t) / flight_t
            set_feet_task_weight(pwe, "hard", 1.0)
            knee = 0.0
            hip = 0.0
            ankle = 0.0
            z = 0.0
        else:
            s = (t - crouch_t - launch_t - flight_t) / (cycle_t - crouch_t - launch_t - flight_t)
            set_feet_task_weight(pwe, "hard", 1.0)
            knee = max_knee_flex * (0.3 * (1 - s))
            hip = max_hip * (0.3 * (1 - s))
            ankle = max_ankle * (0.3 * (1 - s))
            z = 0.0

        set_legs(knee, hip, ankle)
        t_world = t0 + np.array([0.0, 0.0, z])
        # Keep trunk soft and near original
        update_trunk_targets(pwe, R0, t_world, pos_mode="soft", pos_weight=0.2, ori_mode="soft", ori_weight=0.2)

    return record_loop(pwe, duration, fps, updater)


def main():
    parser = argparse.ArgumentParser(description="Generate motions using Placo solver (feet locked, solver-driven)")
    parser.add_argument("--duck", choices=["go_bdx"], default="go_bdx")
    parser.add_argument("--motion", default="all", help="head_roll|head_pitch|body_sway|jump|all or path to JSON config")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_ROOT, "recordings"))
    args = parser.parse_args()

    asset_path = get_asset_path(args.duck)
    robot_urdf = f"{args.duck}.urdf"

    with open(os.path.join(asset_path, "placo_defaults.json"), "r") as f:
        gait_parameters = json.load(f)

    pwe = PlacoWalkEngine(asset_path, robot_urdf, gait_parameters)

    os.makedirs(args.output_dir, exist_ok=True)

    motions = []
    cfg = None
    if os.path.isfile(args.motion) and args.motion.endswith(".json"):
        with open(args.motion, "r") as cf:
            cfg = json.load(cf)
        motion_type = cfg.get("type", "body_sway")
        duration = cfg.get("duration", args.duration)
        fps = cfg.get("fps", args.fps)
        params = cfg.get("params", {})
        name = cfg.get("name", motion_type)
        if motion_type == "body_sway":
            epi = generate_body_sway(pwe, duration, fps, params)
        elif motion_type == "head_roll":
            epi = generate_head_roll(pwe, duration, fps, params)
        elif motion_type == "head_pitch":
            epi = generate_head_pitch(pwe, duration, fps, params)
        elif motion_type == "jump":
            epi = generate_jump(pwe, duration, fps, params)
        else:
            raise ValueError(f"Unknown motion type: {motion_type}")
        motions.append((name, epi))
    else:
        if args.motion in ("body_sway", "all"):
            motions.append(("body_sway_solved", generate_body_sway(pwe, args.duration, args.fps, None)))
        if args.motion in ("head_roll", "all"):
            motions.append(("head_roll_solved", generate_head_roll(pwe, args.duration, args.fps, None)))
        if args.motion in ("head_pitch", "all"):
            motions.append(("head_pitch_solved", generate_head_pitch(pwe, args.duration, args.fps, None)))
        if args.motion in ("jump", "all"):
            motions.append(("jump_solved", generate_jump(pwe, args.duration, args.fps, None)))

    for name, episode in motions:
        episode["Placo"] = {"preset_name": name, "duration": args.duration}
        out_path = os.path.join(args.output_dir, f"custom_{name}.json")
        with open(out_path, "w") as f:
            json.dump(episode, f, indent=2)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()


