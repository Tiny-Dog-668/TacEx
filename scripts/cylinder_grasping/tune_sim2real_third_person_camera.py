"""Interactive third-person camera tuning scene for the sim2real grasp setup.

This script reuses the same Franka / bottle / cap / wooden board / camera configuration
as the current sim2real grasp task, but does not create the RL environment. It launches
only the scene so the third-person camera pose can be tuned manually with the keyboard.

Usage:
    isaaclab -p scripts/cylinder_grasping/tune_sim2real_third_person_camera.py
"""

from __future__ import annotations

import argparse
from collections import deque

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Tune the sim2real third-person camera pose with the keyboard.")
parser.add_argument("--translation_step", type=float, default=0.01, help="World-space translation step in meters.")
parser.add_argument("--lookat_step", type=float, default=0.01, help="Look-at target step in meters.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if getattr(args_cli, "headless", False):
    raise RuntimeError("Keyboard camera tuning requires GUI mode. Remove --headless.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb.input as carb_input
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.appwindow
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from omni.kit.viewport.utility import get_active_viewport

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_grasp.sim2real_grasp_env import Sim2RealGraspEnvCfg


def _tuple3(tensor: torch.Tensor) -> tuple[float, float, float]:
    values = tensor.detach().cpu().tolist()
    return (float(values[0]), float(values[1]), float(values[2]))


def _tuple4(tensor: torch.Tensor) -> tuple[float, float, float, float]:
    values = tensor.detach().cpu().tolist()
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _camera_quat_from_view(eye: torch.Tensor, target: torch.Tensor, device: str) -> torch.Tensor:
    rot_m = math_utils.create_rotation_matrix_from_view(
        eye.unsqueeze(0), target.unsqueeze(0), up_axis="Z", device=device
    )
    return math_utils.quat_from_matrix(rot_m)[0]


def _print_pose(eye: torch.Tensor, target: torch.Tensor, device: str) -> None:
    quat = _camera_quat_from_view(eye, target, device)
    pos = _tuple3(eye)
    lookat = _tuple3(target)
    rot = _tuple4(quat)
    print("[POSE] eye    =", pos)
    print("[POSE] target =", lookat)
    print("[POSE] rot    =", rot)
    print("[POSE] config snippet:")
    print(f"    pos=({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}),")
    print(f"    rot=({rot[0]:.6f}, {rot[1]:.6f}, {rot[2]:.6f}, {rot[3]:.6f}),")


def main():
    cfg = Sim2RealGraspEnvCfg()

    sim = sim_utils.SimulationContext(cfg.sim)
    sim.set_camera_view(cfg.viewer.eye, cfg.viewer.lookat)

    cfg.ground.spawn.func(
        cfg.ground.prim_path,
        cfg.ground.spawn,
        translation=cfg.ground.init_state.pos,
        orientation=cfg.ground.init_state.rot,
    )

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(cfg.robot.replace(prim_path="/World/Robot"))
    cylinder = RigidObject(cfg.cylinder.replace(prim_path="/World/cylinder"))
    cap = RigidObject(cfg.cap.replace(prim_path="/World/cap"))
    plate = RigidObject(cfg.plate.replace(prim_path="/World/floor_panel"))
    camera_cfg = cfg.wrist_camera.replace(prim_path="/World/third_person_camera")
    camera = TiledCamera(camera_cfg)

    sim.reset()

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_pos[:, :7] = torch.tensor(
        [
            -0.30803592681804604,
            -0.1153251832265955,
            0.3018773037895521,
            -2.273053513963647,
            0.04098702578428312,
            2.1623620899157245,
            0.7541767871490952,
        ],
        dtype=torch.float32,
        device=sim.device,
    )

    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7].clone())
    robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:].clone())
    robot.write_joint_state_to_sim(default_joint_pos, robot.data.default_joint_vel.clone())
    robot.reset()

    cylinder.write_root_pose_to_sim(cylinder.data.default_root_state[:, :7].clone())
    cylinder.write_root_velocity_to_sim(cylinder.data.default_root_state[:, 7:].clone())
    cylinder.reset()

    plate.write_root_pose_to_sim(plate.data.default_root_state[:, :7].clone())
    plate.write_root_velocity_to_sim(plate.data.default_root_state[:, 7:].clone())
    plate.reset()

    cap.write_root_pose_to_sim(cap.data.default_root_state[:, :7].clone())
    cap.write_root_velocity_to_sim(cap.data.default_root_state[:, 7:].clone())
    cap.reset()

    device = sim.device
    default_eye = torch.tensor(camera_cfg.offset.pos, dtype=torch.float32, device=device)
    default_quat = torch.tensor(camera_cfg.offset.rot, dtype=torch.float32, device=device).unsqueeze(0)
    forward = math_utils.quat_apply(
        default_quat, torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device)
    )[0]
    viewer_target = torch.tensor(cfg.viewer.lookat, dtype=torch.float32, device=device)
    look_distance = torch.linalg.norm(viewer_target - default_eye).clamp(min=0.1)
    default_target = default_eye + forward * look_distance

    eye = default_eye.clone()
    target = default_target.clone()

    cap_offset_pos = torch.tensor([[0.0, 0.0, cfg.cap_center_offset_z]], dtype=torch.float32, device=device)
    cap_offset_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    def _sync_cap():
        cap_pos, cap_quat = math_utils.combine_frame_transforms(
            cylinder.data.root_pos_w, cylinder.data.root_quat_w, cap_offset_pos, cap_offset_rot
        )
        cap_pose = torch.cat([cap_pos, cap_quat], dim=-1)
        cap_vel = torch.cat([cylinder.data.root_lin_vel_w, cylinder.data.root_ang_vel_w], dim=-1)
        cap.write_root_pose_to_sim(cap_pose)
        cap.write_root_velocity_to_sim(cap_vel)

    def _apply_camera_pose():
        camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0), env_ids=[0])

    _sync_cap()
    _apply_camera_pose()

    viewport = None
    perspective_camera_path = None
    try:
        viewport = get_active_viewport()
        if viewport is not None:
            perspective_camera_path = viewport.get_active_camera()
            viewport.set_active_camera(camera_cfg.prim_path)
    except Exception:
        viewport = None

    step_translation = float(args_cli.translation_step)
    step_lookat = float(args_cli.lookat_step)
    command_queue: deque[tuple[str, object]] = deque()

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window is not None else None
    input_iface = carb_input.acquire_input_interface() if keyboard is not None else None
    kb_sub = None

    def _on_kb_event(event, *args, **kwargs):
        if event.type not in (carb_input.KeyboardEventType.KEY_PRESS, carb_input.KeyboardEventType.KEY_REPEAT):
            return True

        if event.input == carb_input.KeyboardInput.I:
            command_queue.append(("eye", (1.0, 0.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.K:
            command_queue.append(("eye", (-1.0, 0.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.J:
            command_queue.append(("eye", (0.0, 1.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.L:
            command_queue.append(("eye", (0.0, -1.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.U:
            command_queue.append(("eye", (0.0, 0.0, 1.0)))
        elif event.input == carb_input.KeyboardInput.O:
            command_queue.append(("eye", (0.0, 0.0, -1.0)))
        elif event.input == carb_input.KeyboardInput.W:
            command_queue.append(("target", (1.0, 0.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.S:
            command_queue.append(("target", (-1.0, 0.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.A:
            command_queue.append(("target", (0.0, 1.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.D:
            command_queue.append(("target", (0.0, -1.0, 0.0)))
        elif event.input == carb_input.KeyboardInput.Q:
            command_queue.append(("target", (0.0, 0.0, 1.0)))
        elif event.input == carb_input.KeyboardInput.E:
            command_queue.append(("target", (0.0, 0.0, -1.0)))
        elif event.input == carb_input.KeyboardInput.R and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            command_queue.append(("reset", None))
        elif event.input == carb_input.KeyboardInput.P and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            command_queue.append(("print", None))
        elif event.input == carb_input.KeyboardInput.C and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            command_queue.append(("activate_camera", None))
        elif event.input == carb_input.KeyboardInput.MINUS and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            command_queue.append(("step_scale", 0.5))
        elif event.input == carb_input.KeyboardInput.EQUAL and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            command_queue.append(("step_scale", 2.0))
        elif (
            event.input == carb_input.KeyboardInput.ESCAPE
            and event.type == carb_input.KeyboardEventType.KEY_PRESS
            and viewport is not None
            and perspective_camera_path is not None
        ):
            command_queue.append(("perspective_camera", None))
        return True

    if keyboard is not None and input_iface is not None:
        kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_kb_event)

    print("[INFO] Sim2real scene ready.")
    print("[INFO] Click the viewport first, then use keyboard:")
    print("       I/K: eye +/-X, J/L: eye +/-Y, U/O: eye +/-Z")
    print("       W/S: target +/-X, A/D: target +/-Y, Q/E: target +/-Z")
    print("       -/=: halve/double both step sizes")
    print("       R: reset camera, P: print current pose, C: switch viewport to third_person_camera")
    print("       ESC: switch viewport back to perspective camera")
    print(f"[INFO] Current third-person camera prim: {camera_cfg.prim_path}")
    print(f"[INFO] translation_step = {step_translation:.4f} m, lookat_step = {step_lookat:.4f} m")
    _print_pose(eye, target, device)

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        while command_queue:
            cmd, payload = command_queue.popleft()
            if cmd == "eye":
                eye += torch.tensor(payload, dtype=torch.float32, device=device) * step_translation
                _apply_camera_pose()
                _print_pose(eye, target, device)
            elif cmd == "target":
                target += torch.tensor(payload, dtype=torch.float32, device=device) * step_lookat
                _apply_camera_pose()
                _print_pose(eye, target, device)
            elif cmd == "reset":
                eye.copy_(default_eye)
                target.copy_(default_target)
                _apply_camera_pose()
                print("[INFO] Camera reset to the current task defaults.")
                _print_pose(eye, target, device)
            elif cmd == "print":
                _print_pose(eye, target, device)
            elif cmd == "activate_camera":
                if viewport is not None:
                    viewport.set_active_camera(camera_cfg.prim_path)
                    print(f"[INFO] Viewport switched to: {camera_cfg.prim_path}")
            elif cmd == "perspective_camera":
                if viewport is not None and perspective_camera_path is not None:
                    viewport.set_active_camera(perspective_camera_path)
                    print(f"[INFO] Viewport switched to: {perspective_camera_path}")
            elif cmd == "step_scale":
                factor = float(payload)
                step_translation = max(step_translation * factor, 1e-4)
                step_lookat = max(step_lookat * factor, 1e-4)
                print(
                    f"[INFO] translation_step = {step_translation:.4f} m, "
                    f"lookat_step = {step_lookat:.4f} m"
                )

        robot.set_joint_position_target(default_joint_pos)
        robot.write_data_to_sim()

        _sync_cap()
        sim.step()

        robot.update(sim_dt)
        cylinder.update(sim_dt)
        cap.update(sim_dt)
        plate.update(sim_dt)
        camera.update(sim_dt)

    if kb_sub is not None:
        kb_sub = None

    simulation_app.close()


if __name__ == "__main__":
    main()
