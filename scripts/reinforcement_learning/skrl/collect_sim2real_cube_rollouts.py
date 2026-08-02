"""Collect batched sim2real Cube rollouts from an exported end-to-end Actor.

This diagnostic entry point intentionally supports historical exported models
when ``--allow_legacy_contract`` is provided. Normal train/play/export paths
remain fail-closed for legacy sim2real contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect batched sim2real Cube action/state traces.")
parser.add_argument("--task", default="TacEx-Sim2Real-Cube-Real-Alignment-v0")
parser.add_argument("--model", required=True, help="Exported end-to-end TorchScript Actor.")
parser.add_argument("--metadata", default=None, help="Export JSON metadata. Defaults to the model's .json sibling.")
parser.add_argument("--output", default=None, help="Output NPZ path.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=450, help="Policy steps to collect.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--camera_warmup_steps", type=int, default=3)
parser.add_argument(
    "--allow_legacy_contract",
    action="store_true",
    help="Allow v1-v3 only for explicitly labelled offline diagnostics.",
)
parser.add_argument(
    "--rgb_env_id",
    type=int,
    default=0,
    help="Environment whose pre-action RGB frames are sampled; use -1 to disable.",
)
parser.add_argument("--rgb_stride", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch

import tacex_tasks  # noqa: F401

from vision_encoder_artifact import POLICY_CONTRACT_VERSION, load_saved_env_config


ACTION_NAMES = ("x", "y", "z", "gripper")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _find_object(base_env):
    for name in ("_cube", "_can", "_cylinder"):
        obj = getattr(base_env, name, None)
        if obj is not None:
            return obj
    raise RuntimeError("Unable to find the task object (_cube/_can/_cylinder).")


def _policy_observation(observations):
    if isinstance(observations, dict) and "policy" in observations:
        return observations["policy"]
    return observations


def _camera_rgb(base_env) -> torch.Tensor:
    camera = getattr(base_env, "wrist_camera", None)
    if camera is None:
        raise RuntimeError("The selected task has no wrist_camera sensor.")
    rgb = camera.data.output.get("rgb")
    if rgb is None:
        raise RuntimeError("wrist_camera did not produce an RGB frame.")
    if rgb.shape[-1] < 3:
        raise RuntimeError(f"Unexpected wrist RGB shape: {tuple(rgb.shape)}")
    return rgb[..., :3].contiguous().to(dtype=torch.uint8)


def _snapshot(base_env) -> dict[str, np.ndarray]:
    robot_data = base_env._robot.data
    obj_data = _find_object(base_env).data
    origins = base_env.scene.env_origins

    if hasattr(base_env, "_compute_fingertip_positions_world"):
        left_tip_w, right_tip_w = base_env._compute_fingertip_positions_world()
        gripper_center_w = 0.5 * (left_tip_w + right_tip_w)
    else:
        left_tip_w = robot_data.body_link_pos_w[:, base_env._left_finger_body_idx]
        right_tip_w = robot_data.body_link_pos_w[:, base_env._right_finger_body_idx]
        gripper_center_w = 0.5 * (left_tip_w + right_tip_w)

    cube_pos_w = obj_data.root_pos_w
    cube_pos_local = cube_pos_w - origins
    gripper_center_local = gripper_center_w - origins
    measured_width = robot_data.joint_pos[:, base_env._finger_joint_ids].sum(dim=-1)
    desired_width = getattr(base_env, "_desired_gripper_width", measured_width)

    reference_height = getattr(base_env, "_cube_lift_reference_height_per_env", None)
    if reference_height is None:
        lift_delta = torch.full_like(cube_pos_w[:, 2], float("nan"))
    else:
        lift_delta = cube_pos_w[:, 2] - reference_height

    if hasattr(base_env, "_compute_cube_upright_cos"):
        upright_cos = base_env._compute_cube_upright_cos(obj_data.root_quat_w)
        tilt_deg = torch.rad2deg(torch.acos(torch.clamp(upright_cos, -1.0, 1.0)))
    else:
        tilt_deg = torch.full_like(cube_pos_w[:, 2], float("nan"))

    table_collision_force = getattr(base_env, "_last_table_collision_force", None)
    if table_collision_force is None:
        table_collision_force = torch.zeros_like(cube_pos_w[:, 2])
    table_collision = getattr(base_env, "_last_table_collision", None)
    if table_collision is None:
        table_collision = torch.zeros_like(cube_pos_w[:, 2], dtype=torch.bool)
    table_collision_penalty = getattr(base_env, "_last_table_collision_penalty", None)
    if table_collision_penalty is None:
        table_collision_penalty = torch.zeros_like(cube_pos_w[:, 2])

    return {
        "joint_pos": _to_numpy(robot_data.joint_pos),
        "joint_vel": _to_numpy(robot_data.joint_vel),
        "joint_pos_target": _to_numpy(robot_data.joint_pos_target),
        "cube_pos_world": _to_numpy(cube_pos_w),
        "cube_pos_local": _to_numpy(cube_pos_local),
        "cube_quat_world": _to_numpy(obj_data.root_quat_w),
        "cube_lin_vel_world": _to_numpy(obj_data.root_lin_vel_w),
        "cube_ang_vel_world": _to_numpy(obj_data.root_ang_vel_w),
        "left_tip_pos_world": _to_numpy(left_tip_w),
        "right_tip_pos_world": _to_numpy(right_tip_w),
        "gripper_center_world": _to_numpy(gripper_center_w),
        "gripper_center_local": _to_numpy(gripper_center_local),
        "cube_to_gripper": _to_numpy(cube_pos_w - gripper_center_w),
        "measured_gripper_width": _to_numpy(measured_width),
        "desired_gripper_width": _to_numpy(desired_width),
        "cube_lift_delta": _to_numpy(lift_delta),
        "cube_tilt_deg": _to_numpy(tilt_deg),
        "table_collision_force_n": _to_numpy(table_collision_force),
        "table_collision": _to_numpy(table_collision),
        "table_collision_penalty": _to_numpy(table_collision_penalty),
    }


def _stack(records: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack(values, axis=0) for key, values in records.items() if values}


def _action_summary(
    action: np.ndarray,
    processed: np.ndarray,
    ik_command: np.ndarray,
    gate_active: np.ndarray,
    snapshots: dict[str, np.ndarray],
    reward: np.ndarray,
    done: np.ndarray,
) -> dict:
    summary = {
        "shape": {
            "action": list(action.shape),
            "processed_action": list(processed.shape),
        },
        "reward_mean": float(np.mean(reward)),
        "reward_max": float(np.max(reward)),
        "done_count": int(np.count_nonzero(done)),
        "privileged_dz_gate_fraction": float(np.mean(gate_active)),
        "table_collision_transition_fraction": float(
            np.mean(snapshots["next_table_collision"])
        ),
        "table_collision_force_max_n": float(
            np.max(snapshots["next_table_collision_force_n"])
        ),
        "max_cube_lift_delta_m": float(np.nanmax(snapshots["next_cube_lift_delta"])),
        "max_cube_tilt_deg": float(np.nanmax(snapshots["next_cube_tilt_deg"])),
        "desired_gripper_width_range_m": [
            float(np.min(snapshots["next_desired_gripper_width"])),
            float(np.max(snapshots["next_desired_gripper_width"])),
        ],
        "measured_gripper_width_range_m": [
            float(np.min(snapshots["next_measured_gripper_width"])),
            float(np.max(snapshots["next_measured_gripper_width"])),
        ],
        "action": {},
    }
    for index, name in enumerate(ACTION_NAMES):
        values = action[..., index]
        processed_values = processed[..., index]
        item = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "abs_mean": float(np.mean(np.abs(values))),
            "fraction_abs_ge_0_95": float(np.mean(np.abs(values) >= 0.95)),
            "processed_mean": float(np.mean(processed_values)),
            "processed_std": float(np.std(processed_values)),
        }
        if name == "z":
            item["ik_command_mean"] = float(np.mean(ik_command[..., 2]))
            item["requested_vs_ik_abs_diff_mean"] = float(
                np.mean(np.abs(processed_values - ik_command[..., 2]))
            )
        summary["action"][name] = item
    return summary


def main() -> None:
    model_path = Path(args_cli.model).expanduser().resolve()
    metadata_path = (
        Path(args_cli.metadata).expanduser().resolve()
        if args_cli.metadata
        else model_path.with_suffix(".json")
    )
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("task") != args_cli.task:
        raise ValueError(f"Metadata task {metadata.get('task')!r} does not match {args_cli.task!r}.")
    expected_hash = metadata.get("torchscript_sha256")
    actual_hash = _sha256(model_path)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError("TorchScript SHA-256 does not match export metadata.")

    contract = metadata.get("policy_contract") or {}
    contract_version = int(contract.get("version", 0))
    if contract_version < int(POLICY_CONTRACT_VERSION) and not args_cli.allow_legacy_contract:
        raise RuntimeError(
            f"Legacy contract v{contract_version} is rejected by default; current is "
            f"v{POLICY_CONTRACT_VERSION}. Re-run only for diagnostics with "
            "--allow_legacy_contract."
        )
    if contract_version < int(POLICY_CONTRACT_VERSION):
        print(
            f"[WARN] Diagnostic replay of legacy contract v{contract_version}; "
            "this does not make the policy compatible with the current environment or real robot."
        )

    params_dir = Path(metadata["env_config_source"]).expanduser().resolve().parent
    env_cfg, env_cfg_path = load_saved_env_config(params_dir)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.action_noise_scale = 0.0
    env_cfg.robot_joint_pos_noise = 0.0
    env_cfg.robot_joint_vel_noise = 0.0
    if hasattr(env_cfg, "cube_position_curriculum_force_full_range"):
        env_cfg.cube_position_curriculum_force_full_range = True
    # v1-v3 pickles predate the explicit config field, although their manifests
    # state that the legacy gate was active. Reconstruct that saved behavior
    # only inside this labelled diagnostic path.
    gate_contract = str(contract.get("privileged_dz_gate", ""))
    env_cfg.privileged_dz_gate_enabled = gate_contract.startswith("applied_")
    print(f"[INFO] Loaded saved environment config: {env_cfg_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    env_device = torch.device(base_env.device)
    # Exported deployment Actors intentionally keep the embedded skrl running
    # scaler and ResNet on CPU. Run inference there exactly as the real-robot
    # program does, then copy only the 4-D action back to the simulator device.
    policy_device = torch.device("cpu")
    model = torch.jit.load(str(model_path), map_location=policy_device).eval()
    print(f"[INFO] Loaded TorchScript Actor on {policy_device}: {model_path}", flush=True)
    observations, _ = env.reset()
    warmup_steps = max(0, int(args_cli.camera_warmup_steps))
    if warmup_steps and base_env.sim.has_rtx_sensors():
        print(f"[INFO] Warming up RTX camera sensors for {warmup_steps} frame(s).")
        for _ in range(warmup_steps):
            base_env.sim.render()
            base_env.scene.update(dt=base_env.physics_dt)
    print("[INFO] Environment reset and camera warm-up completed.", flush=True)

    if args_cli.rgb_env_id >= args_cli.num_envs:
        raise ValueError("--rgb_env_id must be less than --num_envs, or -1.")
    rgb_stride = max(1, int(args_cli.rgb_stride))
    steps = max(1, int(args_cli.steps))
    episode_id = np.zeros(args_cli.num_envs, dtype=np.int64)

    records: dict[str, list[np.ndarray]] = {
        "step": [],
        "episode_id": [],
        "episode_step": [],
        "action": [],
        "processed_action": [],
        "action_history_pre": [],
        "action_history_next": [],
        "requested_arm_command": [],
        "ik_arm_command": [],
        "privileged_dz_gate_active": [],
        "reward": [],
        "terminated": [],
        "truncated": [],
        "done": [],
        "rgb_mean": [],
    }
    snapshot_records: dict[str, list[np.ndarray]] = {}
    rgb_steps: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []

    try:
        for step in range(steps):
            if step == 0:
                print("[INFO] Capturing first rollout transition.", flush=True)
            policy_obs = _policy_observation(observations)
            action_history_pre = policy_obs["action_history"]
            proprio_pre = policy_obs["proprio_obs"]
            rgb = _camera_rgb(base_env)
            pre_snapshot = _snapshot(base_env)
            episode_step = _to_numpy(base_env.episode_length_buf).copy()

            with torch.inference_mode():
                actions_cpu = model(
                    action_history_pre.to(policy_device),
                    proprio_pre.to(policy_device),
                    rgb.to(policy_device),
                )
                actions = actions_cpu.to(env_device)
                observations, rewards, terminated, truncated, _ = env.step(actions)

            dones = torch.logical_or(terminated, truncated)
            next_policy_obs = _policy_observation(observations)
            next_snapshot = _snapshot(base_env)

            records["step"].append(np.full(args_cli.num_envs, step, dtype=np.int64))
            records["episode_id"].append(episode_id.copy())
            records["episode_step"].append(episode_step)
            records["action"].append(_to_numpy(actions))
            records["processed_action"].append(_to_numpy(base_env.processed_actions))
            records["action_history_pre"].append(_to_numpy(action_history_pre))
            records["action_history_next"].append(_to_numpy(next_policy_obs["action_history"]))
            records["requested_arm_command"].append(_to_numpy(base_env._last_requested_arm_command))
            records["ik_arm_command"].append(_to_numpy(base_env._last_ik_arm_command))
            records["privileged_dz_gate_active"].append(
                _to_numpy(base_env._last_privileged_dz_gate_active)
            )
            records["reward"].append(_to_numpy(rewards))
            records["terminated"].append(_to_numpy(terminated))
            records["truncated"].append(_to_numpy(truncated))
            records["done"].append(_to_numpy(dones))
            records["rgb_mean"].append(_to_numpy(rgb.to(torch.float32).mean(dim=(1, 2))))

            for key, value in pre_snapshot.items():
                snapshot_records.setdefault(f"pre_{key}", []).append(value)
            for key, value in next_snapshot.items():
                snapshot_records.setdefault(f"next_{key}", []).append(value)

            if args_cli.rgb_env_id >= 0 and step % rgb_stride == 0:
                rgb_steps.append(np.asarray(step, dtype=np.int64))
                rgb_frames.append(_to_numpy(rgb[args_cli.rgb_env_id]))

            episode_id[_to_numpy(dones).astype(bool)] += 1
    finally:
        env.close()

    arrays = _stack(records)
    arrays.update(_stack(snapshot_records))
    if rgb_frames:
        arrays["rgb_step"] = np.stack(rgb_steps, axis=0)
        arrays["rgb_env_id"] = np.asarray(args_cli.rgb_env_id, dtype=np.int64)
        arrays["rgb"] = np.stack(rgb_frames, axis=0)
    arrays["task"] = np.asarray(args_cli.task)
    arrays["model_path"] = np.asarray(str(model_path))
    arrays["model_sha256"] = np.asarray(actual_hash)
    arrays["contract_version"] = np.asarray(contract_version, dtype=np.int64)
    arrays["policy_frequency_hz"] = np.asarray(contract.get("nominal_policy_frequency_hz", np.nan))
    arrays["action_scales"] = np.asarray(contract.get("action_scales", []), dtype=np.float32)

    output_path = (
        Path(args_cli.output).expanduser().resolve()
        if args_cli.output
        else model_path.parent.parent.parent
        / "metrics"
        / "sim2real_rollouts"
        / f"rollout_v{contract_version}_{args_cli.num_envs}env_{steps}steps.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)

    summary = _action_summary(
        arrays["action"],
        arrays["processed_action"],
        arrays["ik_arm_command"],
        arrays["privileged_dz_gate_active"],
        arrays,
        arrays["reward"],
        arrays["done"],
    )
    summary.update(
        {
            "task": args_cli.task,
            "model": str(model_path),
            "model_sha256": actual_hash,
            "contract_version": contract_version,
            "num_envs": int(args_cli.num_envs),
            "steps": steps,
            "episodes_started": int(args_cli.num_envs + np.count_nonzero(arrays["done"])),
            "npz": str(output_path),
        }
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] Saved rollout NPZ: {output_path}")
    print(f"[INFO] Saved rollout summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
