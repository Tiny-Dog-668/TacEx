# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--stochastic_eval",
    action="store_true",
    default=False,
    help="Use sampled policy actions instead of deterministic mean actions.",
)
parser.add_argument(
    "--camera_warmup_steps",
    type=int,
    default=3,
    help="Number of render/update passes after the initial reset to warm up RTX camera sensors.",
)
parser.add_argument(
    "--show_third_person_env_id",
    type=int,
    default=None,
    help="Open an OpenCV window showing third_person_camera RGB for the selected environment index.",
)
parser.add_argument(
    "--third_person_display_scale",
    type=float,
    default=4.0,
    help="Scale factor for the optional third-person OpenCV preview window.",
)
parser.add_argument(
    "--record_rollout",
    action="store_true",
    default=False,
    help="Record one environment's rollout actions and states to a compressed NPZ file.",
)
parser.add_argument(
    "--record_rollout_env_id",
    type=int,
    default=0,
    help="Environment index to record when --record_rollout is enabled.",
)
parser.add_argument(
    "--record_rollout_steps",
    type=int,
    default=200,
    help="Maximum number of rollout steps to record. Use 0 to record until play exits.",
)
parser.add_argument(
    "--record_rollout_output",
    type=str,
    default=None,
    help="Optional NPZ output path for --record_rollout. Defaults to the checkpoint run's metrics/play_rollout directory.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import copy
import importlib
import os
import pickle
import re
import time
import numpy as np
import torch
import yaml

import skrl
from packaging import version
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
    from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner
    from skrl.utils.spaces.jax import flatten_tensorized_space, tensorize_space

# import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import class_to_dict, print_dict, update_class_from_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import tacex_tasks  # noqa: F401
from vision_encoder_artifact import (
    ENCODER_ARTIFACT_FILENAME,
    ENCODER_MANIFEST_FILENAME,
    STRICT_SIM2REAL_CUBE_TASKS,
    load_saved_env_config,
    load_verified_vision_encoder,
    validate_live_env_against_policy_contract,
    validate_live_vision_contract,
    validate_sim2real_policy_contract,
)

# config shortcuts
algorithm = args_cli.algorithm.lower()


def _to_float_scalar(value):
    """Convert scalar-like values (tensor/number) to float."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().to(dtype=torch.float32).mean().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_metric(container, keys: tuple[str, ...]):
    """Recursively find the first matching metric key in nested dict/list containers."""
    if isinstance(container, dict):
        for key in keys:
            if key in container:
                metric = _to_float_scalar(container[key])
                if metric is not None:
                    return metric
        for value in container.values():
            metric = _find_metric(value, keys)
            if metric is not None:
                return metric
        return None
    if isinstance(container, (list, tuple)):
        for value in container:
            metric = _find_metric(value, keys)
            if metric is not None:
                return metric
    return None


def _extract_env_extras(env_obj):
    """Try to locate IsaacLab-style extras dict across wrapper stacks."""
    seen = set()
    queue = [env_obj]
    while queue:
        obj = queue.pop(0)
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        extras = getattr(obj, "extras", None)
        if isinstance(extras, dict):
            return extras

        for attr in ("unwrapped", "_env", "env", "venv"):
            child = getattr(obj, attr, None)
            if child is not None and id(child) not in seen:
                queue.append(child)
    return None


def _get_base_env(env_obj):
    """Return the unwrapped Isaac Lab environment from common wrapper stacks."""
    return getattr(env_obj, "_unwrapped", getattr(env_obj, "unwrapped", env_obj))


def _show_third_person_frame(env_obj, env_id: int, scale: float, cv2_module):
    """Display one env's third-person RGB camera frame in an OpenCV window."""
    base_env = _get_base_env(env_obj)
    camera = getattr(base_env, "third_person_camera", None)
    if camera is None:
        return
    output = getattr(getattr(camera, "data", None), "output", None)
    if not isinstance(output, dict):
        return
    rgb = output.get("rgb")
    if rgb is None or rgb.numel() == 0:
        return
    if env_id < 0 or env_id >= int(rgb.shape[0]):
        return

    frame = rgb[env_id].detach().cpu()
    if frame.dtype != torch.uint8:
        frame = frame.to(dtype=torch.float32)
        if torch.max(frame).item() <= 1.0 + 1e-6:
            frame = frame * 255.0
        frame = frame.round().clamp(0, 255).to(torch.uint8)
    image = frame.numpy()
    if scale > 0 and abs(scale - 1.0) > 1e-6:
        image = cv2_module.resize(image, None, fx=scale, fy=scale, interpolation=cv2_module.INTER_NEAREST)
    cv2_module.imshow(f"third_person_camera env {env_id}", cv2_module.cvtColor(image, cv2_module.COLOR_RGB2BGR))
    cv2_module.waitKey(1)


def _tensor_slice_to_numpy(value, env_id: int):
    """Return one env slice as a CPU numpy array while preserving the env dimension."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except Exception:
            return None
    if value.ndim == 0:
        return value.detach().reshape(1).cpu().numpy()
    if env_id < 0 or env_id >= int(value.shape[0]):
        return None
    return value[env_id : env_id + 1].detach().cpu().numpy()


def _maybe_squeeze_single_body(value):
    """Normalize body tensors shaped (N, 1, D) to (N, D)."""
    if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[1] == 1:
        return value[:, 0, :]
    return value


def _merge_done_tensors(terminated, truncated):
    """Merge terminated/truncated outputs from torch tensors or tensor-like containers."""
    if not isinstance(terminated, torch.Tensor):
        terminated = torch.as_tensor(terminated)
    if not isinstance(truncated, torch.Tensor):
        truncated = torch.as_tensor(truncated, device=terminated.device)
    else:
        truncated = truncated.to(device=terminated.device)
    return torch.logical_or(terminated.to(dtype=torch.bool), truncated.to(dtype=torch.bool))


class RolloutRecorder:
    """Record a single-env rollout to compressed numpy arrays with shape (T, 1, ...)."""

    def __init__(self, env_id: int, max_steps: int, output_path: str):
        self.env_id = int(env_id)
        self.max_steps = max(0, int(max_steps))
        self.output_path = output_path
        self.saved = False
        self.metadata = {}
        self.data = {
            "step": [],
            "action": [],
            "processed_action": [],
            "joint_pos": [],
            "joint_vel": [],
            "joint_pos_target": [],
            "object_pos": [],
            "object_pos_local": [],
            "object_quat": [],
            "gripper_pos": [],
            "reward": [],
            "done": [],
        }

    def should_record(self) -> bool:
        return self.max_steps == 0 or len(self.data["step"]) < self.max_steps

    def is_complete(self) -> bool:
        return self.max_steps > 0 and len(self.data["step"]) >= self.max_steps

    def capture_initial_state(self, env_obj) -> None:
        base_env = _get_base_env(env_obj)
        obj = getattr(base_env, "_can", None)
        obj_data = getattr(obj, "data", None)
        env_origin = self._get_env_origin(base_env)
        initial_object_pos = _tensor_slice_to_numpy(getattr(obj_data, "root_pos_w", None), self.env_id)
        initial_object_quat = _tensor_slice_to_numpy(getattr(obj_data, "root_quat_w", None), self.env_id)
        self.metadata["env_id"] = np.asarray(self.env_id, dtype=np.int64)
        if env_origin is not None:
            self.metadata["env_origin"] = env_origin
        if initial_object_pos is not None:
            self.metadata["initial_object_pos"] = initial_object_pos
            if env_origin is not None and env_origin.shape == initial_object_pos.shape:
                self.metadata["initial_object_pos_local"] = initial_object_pos - env_origin
        if initial_object_quat is not None:
            self.metadata["initial_object_quat"] = initial_object_quat

    def append(self, step: int, env_obj, actions, rewards, dones) -> None:
        if not self.should_record():
            return
        base_env = _get_base_env(env_obj)
        robot = getattr(base_env, "_robot", None)
        robot_data = getattr(robot, "data", None)
        obj = getattr(base_env, "_can", None)
        obj_data = getattr(obj, "data", None)
        body_idx = getattr(base_env, "_body_idx", None)
        env_origin = self._get_env_origin(base_env)

        self.data["step"].append(np.asarray(step, dtype=np.int64))
        self._append_tensor("action", actions)
        self._append_tensor("processed_action", getattr(base_env, "processed_actions", None))
        self._append_tensor("joint_pos", getattr(robot_data, "joint_pos", None))
        self._append_tensor("joint_vel", getattr(robot_data, "joint_vel", None))
        self._append_tensor("joint_pos_target", getattr(robot_data, "joint_pos_target", None))
        object_pos = getattr(obj_data, "root_pos_w", None)
        self._append_tensor("object_pos", object_pos)
        if object_pos is not None and env_origin is not None:
            self._append_array("object_pos_local", _tensor_slice_to_numpy(object_pos, self.env_id) - env_origin)
        else:
            self._append_array("object_pos_local", None)
        self._append_tensor("object_quat", getattr(obj_data, "root_quat_w", None))

        gripper_pos = None
        body_pos = getattr(robot_data, "body_link_pos_w", None)
        if body_pos is not None and body_idx is not None:
            try:
                gripper_pos = _maybe_squeeze_single_body(body_pos[:, body_idx])
            except Exception:
                gripper_pos = None
        self._append_tensor("gripper_pos", gripper_pos)
        self._append_tensor("reward", rewards)
        self._append_tensor("done", dones)

    def _append_tensor(self, key: str, value) -> None:
        array = _tensor_slice_to_numpy(value, self.env_id)
        self._append_array(key, array)

    def _append_array(self, key: str, array) -> None:
        if array is None:
            array = np.asarray([], dtype=np.float32)
        self.data[key].append(array)

    def _get_env_origin(self, base_env):
        scene = getattr(base_env, "scene", None)
        origins = getattr(scene, "env_origins", None)
        return _tensor_slice_to_numpy(origins, self.env_id)

    def save(self) -> None:
        if self.saved:
            return
        if not self.data["step"]:
            print("[WARN] Rollout recorder did not capture any steps; no NPZ was written.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        arrays = {}
        for key, values in self.data.items():
            if key == "step":
                arrays[key] = np.stack(values, axis=0)
                continue
            first_shape = values[0].shape
            if first_shape and all(value.shape == first_shape for value in values):
                arrays[key] = np.stack(values, axis=0)
            else:
                arrays[key] = np.asarray(values, dtype=object)
        arrays.update(self.metadata)
        np.savez_compressed(self.output_path, **arrays)
        self.saved = True
        print(f"[INFO] Saved rollout NPZ: {self.output_path}")


def _process_cfg(cfg: dict) -> dict:
    """Convert simple types to skrl classes/components."""
    _direct_eval = [
        "learning_rate_scheduler",
        "shared_state_preprocessor",
        "state_preprocessor",
        "value_preprocessor",
    ]

    def reward_shaper_function(scale):
        def reward_shaper(rewards, *args, **kwargs):
            return rewards * scale

        return reward_shaper

    def update_dict(d):
        for key, value in list(d.items()):
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in _direct_eval:
                    if type(d[key]) is str:
                        d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
                elif key in ["rewards_shaper_scale"]:
                    d["rewards_shaper"] = reward_shaper_function(value)
        return d

    return update_dict(copy.deepcopy(cfg))


def _load_component(path_or_name: str):
    """Load component from either 'module:Class' string or bare class name."""
    name = str(path_or_name)
    if ":" in name:
        module_path, class_name = name.split(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    return None


def _load_yaml_if_exists(path: str, *, allow_python_tags: bool = False):
    """Load YAML dict if the file exists, otherwise return None."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.full_load(f) if allow_python_tags else yaml.safe_load(f)
    except Exception as exc:
        print(f"[WARN] Failed to load YAML '{path}': {exc}")
        return None
    return data if isinstance(data, dict) else None


def _load_pickle_if_exists(path: str):
    """Load pickle object if the file exists, otherwise return None."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to load pickle '{path}': {exc}")
        return None


def _refresh_wrapped_observations(env_obj):
    """Recompute observations and update skrl wrapper caches after sensor warm-up."""
    base_env = getattr(env_obj, "_unwrapped", getattr(env_obj, "unwrapped", env_obj))
    if not hasattr(base_env, "_get_observations"):
        return None

    observations = base_env._get_observations()
    info = getattr(base_env, "extras", {})

    if hasattr(env_obj, "_observations") and hasattr(env_obj, "observation_space"):
        policy_obs = observations["policy"] if isinstance(observations, dict) else observations
        env_obj._observations = flatten_tensorized_space(
            tensorize_space(env_obj.observation_space, policy_obs, device=getattr(env_obj, "device", None))
        )
        if hasattr(env_obj, "_info"):
            env_obj._info = info
        if hasattr(env_obj, "_reset_once"):
            env_obj._reset_once = False
        return env_obj._observations, info

    return observations, info


def _warmup_rtx_cameras_after_reset(env_obj, warmup_steps: int):
    """Warm up RTX sensors after reset and refresh the wrapped observations."""
    if warmup_steps <= 0:
        return None

    base_env = getattr(env_obj, "_unwrapped", getattr(env_obj, "unwrapped", env_obj))
    sim = getattr(base_env, "sim", None)
    scene = getattr(base_env, "scene", None)
    physics_dt = getattr(base_env, "physics_dt", None)
    if sim is None or scene is None or physics_dt is None:
        return None

    try:
        has_rtx_sensors = sim.has_rtx_sensors()
    except Exception:
        has_rtx_sensors = False
    if not has_rtx_sensors:
        return None

    print(f"[INFO] Warming up RTX camera sensors for {warmup_steps} frame(s) after initial reset.")
    for _ in range(warmup_steps):
        sim.render()
        scene.update(dt=physics_dt)

    try:
        return _refresh_wrapped_observations(env_obj)
    except Exception as exc:
        print(f"[WARN] Failed to refresh observations after camera warm-up: {exc}")
        return None


def _filter_cfg_overlay(template, data: dict):
    """Drop checkpoint keys that no longer exist in the live config schema."""
    filtered = {}
    for key, value in data.items():
        # Structural config members are tied to the current task code and may
        # legitimately change representation across revisions.
        if key in {"class_type", "func", "ui_window_class_type"}:
            continue
        if key == "spawn":
            continue

        if isinstance(template, dict):
            if key not in template:
                continue
            template_value = template[key]
        else:
            if not hasattr(template, key):
                continue
            template_value = getattr(template, key)

        if isinstance(value, dict) and (isinstance(template_value, dict) or hasattr(template_value, "__dict__")):
            filtered[key] = _filter_cfg_overlay(template_value, value)
        else:
            filtered[key] = value
    return filtered


def _restore_env_cfg_from_data(env_data: dict, source_path: str):
    """Merge checkpoint env data onto the current task config."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_data = copy.deepcopy(env_data)
    checkpoint_seed = env_data.pop("seed", None)
    # These are derived from the current task code and may legitimately drift
    # across revisions. Recompute them from the live config class.
    for key in (
        "observation_space",
        "num_observations",
        "state_space",
        "num_states",
        "action_space",
        "num_actions",
        "observation_noise_model",
        "action_noise_model",
    ):
        env_data.pop(key, None)
    update_class_from_dict(env_cfg, _filter_cfg_overlay(env_cfg, env_data))
    if checkpoint_seed is not None:
        env_cfg.seed = checkpoint_seed
    if hasattr(env_cfg, "__post_init__"):
        env_cfg.__post_init__()
    print(f"[INFO] Loaded environment config from checkpoint run: {source_path}")
    return env_cfg


def _load_env_cfg_from_checkpoint(run_dir: str):
    """Restore env config from a checkpoint run with YAML-first fallback."""
    params_dir = os.path.join(run_dir, "params")
    env_yaml_path = os.path.join(params_dir, "env.yaml")
    env_pkl_path = os.path.join(params_dir, "env.pkl")

    env_yaml = _load_yaml_if_exists(env_yaml_path, allow_python_tags=True)
    if env_yaml is not None:
        try:
            return _restore_env_cfg_from_data(env_yaml, env_yaml_path)
        except Exception as exc:
            print(f"[WARN] Failed to restore environment config from '{env_yaml_path}': {exc}")

    checkpoint_env_cfg = _load_pickle_if_exists(env_pkl_path)
    if checkpoint_env_cfg is not None:
        try:
            env_data = checkpoint_env_cfg if isinstance(checkpoint_env_cfg, dict) else class_to_dict(checkpoint_env_cfg)
            return _restore_env_cfg_from_data(env_data, env_pkl_path)
        except Exception as exc:
            print(f"[WARN] Failed to restore environment config from '{env_pkl_path}': {exc}")
            try:
                env_cfg = parse_env_cfg(
                    args_cli.task,
                    device=args_cli.device,
                    num_envs=args_cli.num_envs,
                    use_fabric=not args_cli.disable_fabric,
                )
                print(f"[INFO] Falling back to live task config defaults for: {env_pkl_path}")
                return env_cfg
            except Exception as parse_exc:
                print(f"[WARN] Failed to build live task config after checkpoint restore failure: {parse_exc}")

    return None


def _resolve_auto_checkpoint_path(log_root_path: str, experiment_cfg: dict) -> str:
    """Resolve the checkpoint to evaluate without explicit CLI input."""
    experiment_name = str(experiment_cfg.get("agent", {}).get("experiment", {}).get("experiment_name", "") or "").strip()
    run_dir_patterns: list[tuple[str, str]] = []
    if experiment_name:
        run_dir_patterns.append(
            (
                rf".*_{algorithm}_{re.escape(args_cli.ml_framework)}_{re.escape(experiment_name)}$",
                f"task-specific experiment '{experiment_name}'",
            )
        )
    run_dir_patterns.append((rf".*_{algorithm}_{re.escape(args_cli.ml_framework)}", "algorithm/framework match"))

    last_error = None
    for run_dir_pattern, label in run_dir_patterns:
        try:
            resume_path = get_checkpoint_path(
                log_root_path,
                run_dir=run_dir_pattern,
                checkpoint=r"best_agent\.pt",
                other_dirs=["checkpoints"],
            )
            print(f"[INFO] Auto-selected best checkpoint from latest {label}: {resume_path}")
            return resume_path
        except ValueError as exc:
            last_error = exc

        try:
            resume_path = get_checkpoint_path(
                log_root_path,
                run_dir=run_dir_pattern,
                other_dirs=["checkpoints"],
            )
            print(f"[INFO] Auto-selected latest checkpoint from latest {label}: {resume_path}")
            return resume_path
        except ValueError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError(f"No checkpoints available under: {log_root_path}")


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = _resolve_auto_checkpoint_path(log_root_path, experiment_cfg)

    env_cfg = None

    # Prefer checkpoint-run configs for backward compatibility and train/play alignment.
    run_dir = os.path.dirname(os.path.dirname(resume_path))
    strict_policy_contract = None
    if args_cli.task in STRICT_SIM2REAL_CUBE_TASKS:
        params_dir = os.path.join(run_dir, "params")
        strict_policy_contract = validate_sim2real_policy_contract(
            os.path.join(params_dir, ENCODER_MANIFEST_FILENAME),
            task=args_cli.task,
            params_dir=params_dir,
        )

    checkpoint_agent_cfg = _load_pickle_if_exists(os.path.join(run_dir, "params", "agent.pkl"))
    if checkpoint_agent_cfg is None:
        checkpoint_agent_cfg = _load_yaml_if_exists(os.path.join(run_dir, "params", "agent.yaml"))
    if checkpoint_agent_cfg is not None:
        experiment_cfg = checkpoint_agent_cfg
        print(f"[INFO] Loaded agent config from checkpoint run: {os.path.join(run_dir, 'params')}")

    if strict_policy_contract is not None:
        env_cfg, strict_env_cfg_path = load_saved_env_config(os.path.join(run_dir, "params"))
        print(f"[INFO] Loaded exact strict environment config: {strict_env_cfg_path}")
    else:
        checkpoint_env_cfg = _load_env_cfg_from_checkpoint(run_dir)
        if checkpoint_env_cfg is not None:
            env_cfg = checkpoint_env_cfg

    if env_cfg is None:
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
    else:
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.disable_fabric:
            env_cfg.sim.use_fabric = False

    # Align evaluation seed with training unless explicitly overridden.
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    elif getattr(env_cfg, "seed", None) is None and isinstance(experiment_cfg, dict):
        cfg_seed = experiment_cfg.get("seed")
        if cfg_seed is not None:
            env_cfg.seed = cfg_seed

    # Disable training-time action noise during evaluation.
    if hasattr(env_cfg, "action_noise_scale"):
        env_cfg.action_noise_scale = 0.0
    if hasattr(env_cfg, "robot_joint_pos_noise"):
        env_cfg.robot_joint_pos_noise = 0.0
    if hasattr(env_cfg, "robot_joint_vel_noise"):
        env_cfg.robot_joint_vel_noise = 0.0

    print(
        f"[INFO] Evaluation env config: num_envs={env_cfg.scene.num_envs}, "
        f"device={env_cfg.sim.device}, use_fabric={env_cfg.sim.use_fabric}, seed={getattr(env_cfg, 'seed', None)}"
    )
    if hasattr(env_cfg, "action_noise_scale"):
        print(f"[INFO] Evaluation override: action_noise_scale={env_cfg.action_noise_scale}")
    if hasattr(env_cfg, "robot_joint_pos_noise"):
        print(f"[INFO] Evaluation override: robot_joint_pos_noise={env_cfg.robot_joint_pos_noise}")
    if hasattr(env_cfg, "robot_joint_vel_noise"):
        print(f"[INFO] Evaluation override: robot_joint_vel_noise={env_cfg.robot_joint_vel_noise}")
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if strict_policy_contract is not None:
        params_dir = os.path.join(run_dir, "params")
        base_env = env.unwrapped
        encoder_manifest = load_verified_vision_encoder(
            base_env._resnet18,
            artifact_path=os.path.join(params_dir, ENCODER_ARTIFACT_FILENAME),
            manifest_path=os.path.join(params_dir, ENCODER_MANIFEST_FILENAME),
        )
        validate_live_vision_contract(base_env, encoder_manifest)
        validate_live_env_against_policy_contract(base_env, strict_policy_contract)
        print(f"[INFO] Verified strict sim2real policy and encoder contract: {params_dir}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo", "ppo_rnn"]:
        env = multi_agent_to_single_agent(env)

    # get environment (physics) dt for real-time evaluation
    try:
        dt = env.physics_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # 根据任务选择默认 Runner 或自定义 LSTM 路径
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

    LSTM_TASK_IDS = {
        "TacEx-Cylinder-Grasping-Four-Tactile-RGB-v0",
    }
    USE_CUSTOM_POLICY = (
        args_cli.task in LSTM_TASK_IDS
        or algorithm == "ppo_rnn"
    )
    # 兜底：如果策略配置请求 recurrent 且观测里有 tactile*_resnet，就强制走自定义 LSTM 路径
    policy_cfg = experiment_cfg.get("models", {}).get("policy", {})
    is_recurrent = bool(policy_cfg.get("recurrent", False))
    tactile_keys = {
        "tactile_left_resnet", "tactile_right_resnet", "tactile_left_down_resnet", "tactile_right_down_resnet",
        "tactile_left_depth_resnet", "tactile_right_depth_resnet", "tactile_left_down_depth_resnet", "tactile_right_down_depth_resnet",
        "tactile_left_rgb", "tactile_right_rgb", "tactile_left_down_depth", "tactile_right_down_depth",
    }
    obs_space = getattr(env, "observation_space", None)
    if hasattr(obs_space, "spaces") and isinstance(obs_space.spaces, dict):
        obs_keys = set(obs_space.spaces.keys())
    else:
        try:
            obs_keys = set(obs_space.keys()) if obs_space is not None else set()
        except Exception:
            obs_keys = set()
    if (not USE_CUSTOM_POLICY) and is_recurrent and obs_keys.intersection(tactile_keys):
        print("[INFO] Enabling custom CylinderFusionLSTM (recurrent policy + tactile_resnet observations).")
        USE_CUSTOM_POLICY = True

    agent_class_spec = str(experiment_cfg.get("agent", {}).get("class", ""))
    USE_CUSTOM_AGENT_CLASS = (not USE_CUSTOM_POLICY) and (":" in agent_class_spec)

    if USE_CUSTOM_AGENT_CLASS:
        print(f"[INFO] Using custom agent class '{agent_class_spec}' (manual eval path)")

        from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
        from skrl.memories.torch import RandomMemory
        from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model

        policy_cfg_local = copy.deepcopy(experiment_cfg["models"]["policy"])
        policy_class_spec = policy_cfg_local.pop("class", "GaussianMixin")
        policy_cls = _load_component(policy_class_spec)
        if policy_cls is None:
            policy_model = gaussian_model(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                **policy_cfg_local,
            )
        else:
            policy_model = policy_cls(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                **policy_cfg_local,
            )

        value_cfg_local = copy.deepcopy(experiment_cfg["models"]["value"])
        value_class_spec = value_cfg_local.pop("class", "DeterministicMixin")
        value_cls = _load_component(value_class_spec)
        if value_cls is None:
            value_model = deterministic_model(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                **value_cfg_local,
            )
        else:
            value_model = value_cls(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                **value_cfg_local,
            )
        models = {"policy": policy_model, "value": value_model}

        memory_cfg_local = copy.deepcopy(experiment_cfg.get("memory", {}))
        memory_class_spec = memory_cfg_local.pop("class", "RandomMemory")
        memory_cls = _load_component(memory_class_spec)
        if memory_cls is None:
            memory_cls = RandomMemory
        if int(memory_cfg_local.get("memory_size", -1)) < 0:
            memory_cfg_local["memory_size"] = int(experiment_cfg["agent"].get("rollouts", 128))
        memory = memory_cls(
            num_envs=env.num_envs,
            device=env.device,
            **_process_cfg(memory_cfg_local),
        )

        agent_runtime_cfg = copy.deepcopy(experiment_cfg["agent"])
        agent_runtime_cfg.pop("class", None)
        if algorithm == "ppo":
            merged_cfg = PPO_DEFAULT_CONFIG.copy()
            merged_cfg.update(_process_cfg(agent_runtime_cfg))
            agent_runtime_cfg = merged_cfg
        else:
            agent_runtime_cfg = _process_cfg(agent_runtime_cfg)

        state_kwargs = agent_runtime_cfg.get("state_preprocessor_kwargs")
        if state_kwargs is None:
            state_kwargs = {}
        if agent_runtime_cfg.get("state_preprocessor", None) is not None:
            state_kwargs.update({"size": env.observation_space, "device": env.device})
        agent_runtime_cfg["state_preprocessor_kwargs"] = state_kwargs

        value_kwargs = agent_runtime_cfg.get("value_preprocessor_kwargs")
        if value_kwargs is None:
            value_kwargs = {}
        if agent_runtime_cfg.get("value_preprocessor", None) is not None:
            value_kwargs.update({"size": 1, "device": env.device})
        agent_runtime_cfg["value_preprocessor_kwargs"] = value_kwargs

        agent_cls = _load_component(agent_class_spec)
        if agent_cls is None:
            raise RuntimeError(f"Failed to resolve custom agent class '{agent_class_spec}'")

        agent = agent_cls(
            models=models,
            memory=memory,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            cfg=agent_runtime_cfg,
        )
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)
        agent.set_running_mode("eval")
    elif not USE_CUSTOM_POLICY:
        # configure and instantiate the skrl runner
        # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
        runner = Runner(env, experiment_cfg)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        # set agent to evaluation mode
        runner.agent.set_running_mode("eval")
        agent = runner.agent
    else:
        # manual PPO_RNN path with custom LSTM policy
        from skrl.utils.model_instantiators.torch import deterministic_model
        from skrl.memories.torch import RandomMemory
        try:
            from skrl.agents.torch.ppo.ppo_rnn import PPO_RNN  # skrl>=1.4.3
        except ImportError:
            from skrl.agents.torch.ppo_rnn import PPO_RNN  # fallback
        from custom_models import CylinderFusionLSTM

        seq_len = experiment_cfg["agent"].get("sequence_length", 64)
        policy_cfg = experiment_cfg["models"]["policy"]
        hidden_size = policy_cfg.get("rnn_units", policy_cfg.get("recurrent_hidden_size", 256))
        num_layers = policy_cfg.get("rnn_layers", policy_cfg.get("recurrent_layers", 1))

        # disable state preprocessor to avoid dict/shape mismatch
        experiment_cfg["agent"]["state_preprocessor"] = None
        experiment_cfg["agent"]["state_preprocessor_kwargs"] = {}
        # value preprocessor: align with training (RunningStandardScaler size=1)
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        experiment_cfg["agent"]["value_preprocessor"] = RunningStandardScaler
        v_kwargs = experiment_cfg["agent"].get("value_preprocessor_kwargs") or {}
        v_kwargs.setdefault("size", 1)
        v_kwargs.setdefault("device", env.device)
        experiment_cfg["agent"]["value_preprocessor_kwargs"] = v_kwargs

        policy_model = CylinderFusionLSTM(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            num_envs=env.num_envs,
            sequence_length=seq_len,
            num_layers=num_layers,
            hidden_size=hidden_size,
            use_vision_placeholder=policy_cfg.get("use_vision_placeholder", True),
            clip_actions=experiment_cfg["models"]["policy"].get("clip_actions", False),
            clip_log_std=experiment_cfg["models"]["policy"].get("clip_log_std", True),
            min_log_std=experiment_cfg["models"]["policy"].get("min_log_std", -20.0),
            max_log_std=experiment_cfg["models"]["policy"].get("max_log_std", 2.0),
            reduction=experiment_cfg["models"]["policy"].get("reduction", "sum"),
            stats_print_every=policy_cfg.get("stats_print_every", 0),
            reset_stats_print=policy_cfg.get("reset_stats_print", False),
        )
        value_model = deterministic_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **experiment_cfg["models"]["value"],
        )
        models = {"policy": policy_model, "value": value_model}
        memory = RandomMemory(
            memory_size=experiment_cfg["agent"]["rollouts"],
            num_envs=env.num_envs,
            device=env.device,
        )
        agent = PPO_RNN(
            models=models,
            memory=memory,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            cfg=experiment_cfg["agent"],
        )
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)
        # ensure RNN hidden states are initialized (PPO_RNN expects _rnn/_rnn_initial_states)
        try:
            spec = policy_model.get_specification().get("rnn", None)
            if spec:
                sizes = spec.get("sizes", [])
                init_states = [torch.zeros(sz, device=env.device) for sz in sizes]
                agent._rnn = True
                agent._rnn_initial_states = {"policy": init_states, "value": init_states}
                agent._rnn_final_states = {"policy": [], "value": []}
        except Exception:
            pass
        agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    warmed_obs = _warmup_rtx_cameras_after_reset(env, args_cli.camera_warmup_steps)
    if warmed_obs is not None:
        obs, _ = warmed_obs
    timestep = 0
    base_env = _get_base_env(env)
    env_cfg = getattr(base_env, "cfg", None)
    third_person_cv2 = None
    if args_cli.show_third_person_env_id is not None:
        if args_cli.show_third_person_env_id < 0 or args_cli.show_third_person_env_id >= env.num_envs:
            raise ValueError(
                f"--show_third_person_env_id must be in [0, {env.num_envs - 1}], "
                f"got {args_cli.show_third_person_env_id}"
            )
        import cv2 as third_person_cv2

        print(f"[INFO] Showing third_person_camera preview for env {args_cli.show_third_person_env_id}")
    record_interval = max(1, int(getattr(env_cfg, "reward_print_interval", 200)))
    metrics_dir = os.path.join(log_dir, "metrics", "play")
    os.makedirs(metrics_dir, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    metrics_csv = os.path.join(metrics_dir, f"play_metrics_{run_tag}.csv")
    metrics_file = open(metrics_csv, "w", encoding="utf-8", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["step", "recent_success_rate", "window_success_rate"])
    metrics_file.flush()
    print(f"[INFO] Streaming play metrics to CSV: {metrics_csv} (every {record_interval} steps)")
    rollout_recorder = None
    if args_cli.record_rollout:
        if args_cli.record_rollout_env_id < 0 or args_cli.record_rollout_env_id >= env.num_envs:
            raise ValueError(
                f"--record_rollout_env_id must be in [0, {env.num_envs - 1}], "
                f"got {args_cli.record_rollout_env_id}"
            )
        rollout_dir = os.path.join(log_dir, "metrics", "play_rollout")
        rollout_output = args_cli.record_rollout_output or os.path.join(rollout_dir, f"rollout_{run_tag}.npz")
        rollout_recorder = RolloutRecorder(
            env_id=args_cli.record_rollout_env_id,
            max_steps=args_cli.record_rollout_steps,
            output_path=os.path.abspath(rollout_output),
        )
        rollout_recorder.capture_initial_state(env)
        step_text = "all steps until play exits" if args_cli.record_rollout_steps == 0 else args_cli.record_rollout_steps
        print(
            f"[INFO] Recording rollout env {args_cli.record_rollout_env_id} "
            f"for {step_text} step(s) to: {rollout_recorder.output_path}"
        )
    if args_cli.stochastic_eval:
        print("[INFO] Evaluation action mode: stochastic (sampled actions)")
    else:
        print("[INFO] Evaluation action mode: deterministic (mean actions)")
    try:
        # simulate environment
        while simulation_app.is_running():
            start_time = time.time()

            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                outputs = agent.act(obs, timestep=timestep, timesteps=timestep)
                actions = outputs[0] if args_cli.stochastic_eval else outputs[-1].get("mean_actions", outputs[0])
                # env stepping
                obs, rewards, terminated, truncated, infos = env.step(actions)

                current_step = timestep + 1
                if rollout_recorder is not None:
                    dones = _merge_done_tensors(terminated, truncated)
                    rollout_recorder.append(current_step, env, actions, rewards, dones)
                    if rollout_recorder.is_complete():
                        rollout_recorder.save()
                if current_step % record_interval == 0:
                    extras = _extract_env_extras(env)
                    success_rate = _find_metric(
                        extras,
                        ("recent_success_rate", "info/recent_success_rate"),
                    )
                    if success_rate is None:
                        success_rate = _find_metric(infos, ("recent_success_rate", "info/recent_success_rate"))
                    window_success_rate = _find_metric(
                        extras,
                        ("episode_success_rate_window", "info/episode_success_rate_window"),
                    )
                    if window_success_rate is None:
                        window_success_rate = _find_metric(
                            infos,
                            ("episode_success_rate_window", "info/episode_success_rate_window"),
                        )
                    success_val = float("nan") if success_rate is None else float(success_rate)
                    window_success_val = (
                        float("nan") if window_success_rate is None else float(window_success_rate)
                    )
                    metrics_writer.writerow([current_step, success_val, window_success_val])
                    metrics_file.flush()
                if third_person_cv2 is not None:
                    _show_third_person_frame(
                        env,
                        args_cli.show_third_person_env_id,
                        args_cli.third_person_display_scale,
                        third_person_cv2,
                    )
            timestep += 1
            if args_cli.video:
                # exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        metrics_file.flush()
        metrics_file.close()
        print(f"[INFO] Saved play metrics CSV: {metrics_csv}")
        if rollout_recorder is not None:
            rollout_recorder.save()

        # close the simulator
        env.close()
        if third_person_cv2 is not None:
            third_person_cv2.destroyAllWindows()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
