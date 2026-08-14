"""Keyboard teleoperation and geometry inspection for the GelSight RMA Teacher.

Run from the repository root:
    ./tacex.sh -p scripts/sim2real_gelsight_rma/gelsight_teacher_keyboard_teleop.py
"""

from __future__ import annotations

import argparse
import traceback
from collections import deque

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--translation_step",
    type=float,
    default=0.004,
    help="End-effector translation per simulation step in metres.",
)
parser.add_argument(
    "--print_interval_steps",
    type=int,
    default=6,
    help="Number of simulation steps between gripper-pose printouts.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=20,
    help="Number of zero-action settle steps after reset.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if getattr(args_cli, "headless", False):
    raise RuntimeError("Keyboard teleoperation requires GUI mode. Remove --headless.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import carb.input as carb_input
import omni.appwindow
import omni.usd
import torch
import isaaclab.utils.math as math_utils
from pxr import Gf, Usd, UsdGeom

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_rma_env import (
    Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv,
    Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg,
)


_GELSIGHT_GRIPPER_PRIM_NAMES = (
    "gelsight_mini_case_left",
    "gelsight_mini_case_right",
    "gelpad_left",
    "gelpad_right",
)


def _format_vec3(values: torch.Tensor | Gf.Vec3d) -> str:
    """Format a world-frame XYZ position in metres."""
    if isinstance(values, torch.Tensor):
        return f"[{values[0].item():.4f}, {values[1].item():.4f}, {values[2].item():.4f}]"
    return f"[{values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}]"


def _bbox_corners(bbox: Gf.BBox3d) -> tuple[Gf.Vec3d, ...]:
    """Return the eight corners expressed in the owning body's local frame."""
    local_range = bbox.GetRange()
    lower = local_range.GetMin()
    upper = local_range.GetMax()
    return tuple(
        Gf.Vec3d(x, y, z)
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    )


def _load_gelsight_local_geometry_bounds(
    env: Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv,
    bbox_cache: UsdGeom.BBoxCache,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    """Load body-local GelSight geometry corners once.

    USD stage transforms are static for this PhysX articulation. The returned
    corners are therefore later transformed by live body poses from PhysX.
    ``local_corners`` has shape ``[bodies=4, corners=8, xyz=3]``.
    """
    stage = omni.usd.get_context().get_stage()
    body_ids, body_names = env._robot.find_bodies(
        list(_GELSIGHT_GRIPPER_PRIM_NAMES), preserve_order=True
    )
    if tuple(body_names) != _GELSIGHT_GRIPPER_PRIM_NAMES:
        raise RuntimeError(
            "GelSight body names do not match the expected case/gelpad profile: "
            f"{body_names}"
        )
    all_corners: list[list[list[float]]] = []
    for name in _GELSIGHT_GRIPPER_PRIM_NAMES:
        path = f"/World/envs/env_0/Robot/{name}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"GelSight gripper prim is missing: {path}")
        # Exclude the body's own static joint transform. Child geometry offsets
        # remain in this body's local frame and are transformed from the live
        # PhysX body pose below.
        local_bound = bbox_cache.ComputeUntransformedBound(prim)
        all_corners.append(
            [[float(point[0]), float(point[1]), float(point[2])] for point in _bbox_corners(local_bound)]
        )
    return (
        torch.tensor(all_corners, device=env.device, dtype=torch.float32),
        torch.tensor(body_ids, device=env.device, dtype=torch.long),
        tuple(body_names),
    )


def _compute_gelsight_gripper_lowest_point_world(
    env: Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv,
    local_corners: torch.Tensor,
    body_ids: torch.Tensor,
    body_names: tuple[str, ...],
) -> tuple[torch.Tensor, str]:
    """Return the lowest live GelSight case/gelpad geometry corner in world frame."""
    # body positions/quaternions: [bodies=4, xyz/quat] in the PhysX world frame.
    body_pos = env._robot.data.body_link_pos_w[0, body_ids]
    body_quat = env._robot.data.body_link_quat_w[0, body_ids]
    # Expand each body pose across its eight geometry corners: [4, 8, xyz].
    corner_count = local_corners.shape[1]
    world_corners = math_utils.quat_apply(
        body_quat[:, None, :].expand(-1, corner_count, -1).reshape(-1, 4),
        local_corners.reshape(-1, 3),
    ).reshape(-1, corner_count, 3) + body_pos[:, None, :]
    lowest_flat_index = world_corners[:, :, 2].reshape(-1).argmin()
    body_index = int((lowest_flat_index // corner_count).item())
    corner_index = int((lowest_flat_index % corner_count).item())
    return world_corners[body_index, corner_index], body_names[body_index]


def _compute_gripper_center_world(
    env: Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv,
) -> torch.Tensor:
    """Return the nominal two-fingertip midpoint in the world frame, shape [3]."""
    return env._compute_reach_center_world()[0].detach().clone()


def _print_gripper_pose(
    env: Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv,
    local_corners: torch.Tensor,
    body_ids: torch.Tensor,
    body_names: tuple[str, ...],
    prefix: str = "[gripper]",
) -> None:
    center = _compute_gripper_center_world(env)
    lowest_point, source_name = _compute_gelsight_gripper_lowest_point_world(
        env, local_corners, body_ids, body_names
    )
    print(
        f"{prefix} center_world_m={_format_vec3(center)}  "
        f"lowest_geometry_world_m={_format_vec3(lowest_point)}  "
        f"lowest_source={source_name}",
        flush=True,
    )


def _step_env(env: Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv, actions: torch.Tensor) -> None:
    """Advance exactly one physics step using the environment's normal action path."""
    env._pre_physics_step(actions)
    env._apply_action()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    env.sim.render()


def _reset_env(
    env: Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv,
    local_corners: torch.Tensor,
    body_ids: torch.Tensor,
    body_names: tuple[str, ...],
    warmup_steps: int,
) -> None:
    env.reset()
    zero_actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    for _ in range(max(int(warmup_steps), 0)):
        _step_env(env, zero_actions)
    _print_gripper_pose(env, local_corners, body_ids, body_names, prefix="[reset]")


def main() -> None:
    env_cfg = Sim2RealCubeRealAlignmentRMAGelSightTeacherEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.reward_print_interval = 0
    env_cfg.wrist_visual_randomization_enabled = False
    env_cfg.light_randomization_enabled = False
    env_cfg.ground_color_randomization_enabled = False
    env_cfg.plate_color_randomization_enabled = False

    env = Sim2RealCubeRealAlignmentRMAGelSightTeacherEnv(env_cfg)
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=True
    )
    local_corners, body_ids, body_names = _load_gelsight_local_geometry_bounds(
        env, bbox_cache
    )
    _reset_env(env, local_corners, body_ids, body_names, args_cli.warmup_steps)

    translation_step_units = float(args_cli.translation_step) / max(float(env.cfg.action_scale), 1.0e-6)
    translation_step_units = max(min(translation_step_units, 1.0), 0.0)
    print_interval_steps = max(int(args_cli.print_interval_steps), 1)

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window is not None else None
    input_iface = carb_input.acquire_input_interface() if keyboard is not None else None
    keyboard_subscription = None
    if keyboard is None or input_iface is None:
        raise RuntimeError("Keyboard device unavailable. Please run with a GUI viewport.")

    pressed = {
        "up": False,
        "down": False,
        "left": False,
        "right": False,
        "forward": False,
        "backward": False,
        "open": False,
        "close": False,
    }
    command_queue: deque[str] = deque()

    def _on_keyboard_event(event, *_args, **_kwargs):
        if event.type not in (
            carb_input.KeyboardEventType.KEY_PRESS,
            carb_input.KeyboardEventType.KEY_REPEAT,
            carb_input.KeyboardEventType.KEY_RELEASE,
        ):
            return True
        is_pressed = event.type in (
            carb_input.KeyboardEventType.KEY_PRESS,
            carb_input.KeyboardEventType.KEY_REPEAT,
        )
        if event.input in (carb_input.KeyboardInput.UP, carb_input.KeyboardInput.W):
            pressed["up"] = is_pressed
        elif event.input in (carb_input.KeyboardInput.DOWN, carb_input.KeyboardInput.S):
            pressed["down"] = is_pressed
        elif event.input in (carb_input.KeyboardInput.LEFT, carb_input.KeyboardInput.A):
            pressed["left"] = is_pressed
        elif event.input in (carb_input.KeyboardInput.RIGHT, carb_input.KeyboardInput.D):
            pressed["right"] = is_pressed
        elif event.input == carb_input.KeyboardInput.E:
            pressed["forward"] = is_pressed
        elif event.input == carb_input.KeyboardInput.Q:
            pressed["backward"] = is_pressed
        elif event.input == carb_input.KeyboardInput.J:
            pressed["open"] = is_pressed
        elif event.input == carb_input.KeyboardInput.K:
            pressed["close"] = is_pressed
        elif event.input == carb_input.KeyboardInput.R and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            command_queue.append("reset")
        elif event.input == carb_input.KeyboardInput.P and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            command_queue.append("print")
        elif event.input == carb_input.KeyboardInput.ESCAPE and event.type == carb_input.KeyboardEventType.KEY_PRESS:
            command_queue.append("quit")
        return True

    keyboard_subscription = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard_event)
    print("[INFO] GelSight Teacher keyboard inspection is ready. Click the viewport first.")
    print("       W/S or Up/Down: Z +/-; A/D or Left/Right: Y +/-; E/Q: X +/-")
    print("       J/K: gripper open/close; R: reset; P: print once; ESC: quit")
    print(
        f"[INFO] translation_step={float(args_cli.translation_step):.4f} m, "
        f"action_units={translation_step_units:.4f}, print_interval_steps={print_interval_steps}"
    )

    step_counter = 0
    try:
        while simulation_app.is_running():
            while command_queue:
                command = command_queue.popleft()
                if command == "reset":
                    _reset_env(env, local_corners, body_ids, body_names, args_cli.warmup_steps)
                    step_counter = 0
                elif command == "print":
                    _print_gripper_pose(env, local_corners, body_ids, body_names)
                elif command == "quit":
                    return

            dx_direction = float(pressed["forward"]) - float(pressed["backward"])
            dy_direction = float(pressed["right"]) - float(pressed["left"])
            dz_direction = float(pressed["up"]) - float(pressed["down"])
            gripper_direction = float(pressed["open"]) - float(pressed["close"])
            actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
            actions[:, 0] = dx_direction * translation_step_units
            actions[:, 1] = dy_direction * translation_step_units
            actions[:, 2] = dz_direction * translation_step_units
            actions[:, 3] = gripper_direction
            _step_env(env, actions)

            if step_counter % print_interval_steps == 0:
                _print_gripper_pose(env, local_corners, body_ids, body_names)
            step_counter += 1
    finally:
        try:
            if keyboard_subscription is not None:
                input_iface.unsubscribe_from_keyboard_events(keyboard, keyboard_subscription)
        finally:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        carb.log_error(error)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
