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
parser = argparse.ArgumentParser(description="Evaluate a checkpoint over a fixed object-position bucket grid.")
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
parser.add_argument("--bucket_x_min", type=float, default=0.54, help="Minimum fixed object x position.")
parser.add_argument("--bucket_x_max", type=float, default=0.62, help="Maximum fixed object x position.")
parser.add_argument("--bucket_y_min", type=float, default=-0.12, help="Minimum fixed object y position.")
parser.add_argument("--bucket_y_max", type=float, default=0.13, help="Maximum fixed object y position.")
parser.add_argument("--bucket_step", type=float, default=0.005, help="Grid spacing for fixed object positions.")
parser.add_argument(
    "--bucket_include_endpoint",
    action="store_true",
    default=False,
    help="Include x/y max endpoints. Disabled by default, producing 16*50=800 positions for the default bounds.",
)
parser.add_argument(
    "--bucket_trials_per_position",
    type=int,
    default=1,
    help="Deprecated alias for --bucket_rounds. Number of full 800-position rounds to evaluate.",
)
parser.add_argument(
    "--bucket_rounds",
    type=int,
    default=None,
    help="Number of full bucket-grid rounds to evaluate. Each round visits every grid position once in order.",
)
parser.add_argument(
    "--bucket_position_warmup_steps",
    type=int,
    default=3,
    help="RTX render/update warm-up frames after assigning fixed object positions.",
)
parser.add_argument(
    "--bucket_output",
    type=str,
    default=None,
    help="Optional CSV output path. Defaults to the checkpoint run's metrics/play_bucket directory.",
)
parser.add_argument(
    "--bucket_bbox_debug_samples",
    type=int,
    default=0,
    help="Print this many raw bbox prim paths on the first bbox read for debugging.",
)
parser.add_argument(
    "--bucket_missing_occlusion_as",
    type=str,
    default="nan",
    choices=["nan", "one"],
    help="How to record missing bbox occlusion values. 'one' treats missing bbox as fully occluded.",
)
parser.add_argument(
    "--bucket_enable_bbox",
    action="store_true",
    default=False,
    help="Enable per-env bbox occlusion reads. Disabled by default for faster bucket evaluation.",
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
import math
import os
import pickle
import re
import time
import torch
import yaml
from collections.abc import Iterable

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


def _build_bucket_positions(args) -> list[tuple[int, float, float]]:
    """Build a deterministic x/y grid. By default max bounds are excluded."""
    if args.bucket_step <= 0.0:
        raise ValueError(f"--bucket_step must be positive, got {args.bucket_step}")
    if args.bucket_x_max <= args.bucket_x_min:
        raise ValueError("--bucket_x_max must be greater than --bucket_x_min")
    if args.bucket_y_max <= args.bucket_y_min:
        raise ValueError("--bucket_y_max must be greater than --bucket_y_min")
    rounds = args.bucket_rounds if args.bucket_rounds is not None else args.bucket_trials_per_position
    if rounds <= 0:
        raise ValueError(f"bucket rounds must be positive, got {rounds}")

    def axis_values(v_min: float, v_max: float) -> list[float]:
        span = v_max - v_min
        if args.bucket_include_endpoint:
            count = int(math.floor(span / args.bucket_step + 1e-9)) + 1
        else:
            count = int(math.floor(span / args.bucket_step + 1e-9))
        return [round(v_min + i * args.bucket_step, 10) for i in range(count)]

    xs = axis_values(args.bucket_x_min, args.bucket_x_max)
    ys = axis_values(args.bucket_y_min, args.bucket_y_max)
    positions = [(idx, x, y) for idx, (x, y) in enumerate((x, y) for x in xs for y in ys)]
    if not positions:
        raise ValueError("Bucket grid is empty. Check bounds and step.")
    return positions


def _as_1d_bool_tensor(value, device: torch.device, num_envs: int) -> torch.Tensor:
    """Normalize dones/truncated values from wrappers into a [num_envs] bool tensor."""
    if value is None:
        return torch.zeros((num_envs,), dtype=torch.bool, device=device)
    tensor = torch.as_tensor(value, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(num_envs)
    tensor = tensor.reshape(-1)
    if tensor.numel() == 1 and num_envs > 1:
        tensor = tensor.repeat(num_envs)
    if tensor.numel() != num_envs:
        out = torch.zeros((num_envs,), dtype=torch.bool, device=device)
        out[: min(num_envs, tensor.numel())] = tensor[: min(num_envs, tensor.numel())].to(dtype=torch.bool)
        return out
    return tensor.to(dtype=torch.bool)


def _get_bucket_root_z(base_env) -> float:
    """Use the environment's configured object reset height."""
    cfg = getattr(base_env, "cfg", None)
    if cfg is not None and hasattr(cfg, "can_reset_root_z"):
        return float(getattr(cfg, "can_reset_root_z"))
    try:
        from tacex_tasks.occluded_grasping.vt_box import CAN_RESET_ROOT_Z

        return float(CAN_RESET_ROOT_Z)
    except Exception:
        return float(base_env._can.data.default_root_state[0, 2].item())


def _write_bucket_object_positions(env_obj, env_ids: torch.Tensor, xy: torch.Tensor) -> None:
    """Overwrite object root poses for selected envs with deterministic bucket positions."""
    base_env = _get_base_env(env_obj)
    if env_ids.numel() == 0:
        return
    can = getattr(base_env, "_can", None)
    if can is None:
        raise RuntimeError("Base environment does not expose _can; cannot assign bucket object positions")

    env_ids = env_ids.to(device=base_env.device, dtype=torch.long)
    xy = xy.to(device=base_env.device, dtype=torch.float32)
    can_state = can.data.default_root_state[env_ids].clone()
    can_state[:, 0] = xy[:, 0]
    can_state[:, 1] = xy[:, 1]
    can_state[:, 2] = _get_bucket_root_z(base_env)
    can_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=torch.float32, device=base_env.device)
    can_state[:, :3] += base_env.scene.env_origins[env_ids]
    can.write_root_state_to_sim(can_state, env_ids=env_ids)
    can.write_root_velocity_to_sim(torch.zeros((env_ids.numel(), 6), device=base_env.device), env_ids=env_ids)


class _BucketBBoxOcclusionReader:
    """Read per-env bbox occlusion ratios for /World/envs/env_*/can."""

    def __init__(self, env_obj, debug_samples: int = 0):
        self._env = _get_base_env(env_obj)
        self._records: list[tuple[int, object, str]] = []
        self._enabled = False
        self._init_attempted = False
        self._debug_samples = max(0, int(debug_samples))
        self._debug_printed = False

    @staticmethod
    def _unpack_bbox_output(output):
        if isinstance(output, dict):
            data = output.get("data")
            info = output.get("info", {})
            if not info:
                info = {key: value for key, value in output.items() if key != "data"}
            return data, info
        return output, {}

    @staticmethod
    def _extract_env_id_from_path(path: str) -> int | None:
        prefix = "/World/envs/env_"
        if not path.startswith(prefix):
            return None
        suffix = path[len(prefix):]
        digits = []
        for ch in suffix:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            return None
        return int("".join(digits))

    @staticmethod
    def _extract_occlusion_ratio(row) -> float | None:
        value = None
        if isinstance(row, dict):
            value = row.get("occlusionRatio")
        else:
            try:
                value = row["occlusionRatio"]
            except Exception:
                value = getattr(row, "occlusionRatio", None)
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(ratio) or ratio < 0.0:
            return None
        return max(0.0, min(1.0, ratio))

    def _resolve_camera_prim_paths(self) -> list[str]:
        camera = getattr(self._env, "third_person_camera", None)
        if camera is None:
            return []
        view = getattr(camera, "_view", None)
        prim_paths = getattr(view, "prim_paths", None)
        if isinstance(prim_paths, (tuple, list)):
            return [str(path) for path in prim_paths if path is not None and str(path)]
        return []

    def _init(self) -> None:
        self._init_attempted = True
        try:
            import omni.replicator.core as rep  # type: ignore
        except Exception as exc:
            print(f"[WARN] Failed to import omni.replicator.core for bbox occlusion: {exc}")
            return
        try:
            camera = getattr(self._env, "third_person_camera", None)
            camera_prim_paths = self._resolve_camera_prim_paths()
            if not camera_prim_paths:
                print("[WARN] No third-person camera prims found; bbox occlusion will be NaN")
                return
            width = int(getattr(getattr(camera, "cfg", None), "width", 224))
            height = int(getattr(getattr(camera, "cfg", None), "height", 224))
            records = []
            for camera_prim_path in camera_prim_paths:
                env_id = self._extract_env_id_from_path(camera_prim_path)
                if env_id is None:
                    continue
                render_product = rep.create.render_product(camera_prim_path, resolution=(width, height))
                if not isinstance(render_product, str):
                    render_product = render_product.path
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "bounding_box_3d",
                    init_params={"semanticTypes": ["class"]},
                )
                try:
                    annotator.attach(render_product)
                except Exception:
                    annotator.attach([render_product])
                records.append((env_id, annotator, str(render_product)))
            if not records:
                print("[WARN] No per-env bbox render products were created; bbox occlusion will be NaN")
                return
            self._records = records
            self._enabled = True
            print(f"[INFO] BBox annotators attached to {len(records)} per-env third-person render products.")
        except Exception as exc:
            print(f"[WARN] Failed to initialize bbox annotator: {exc}")
            self._records = []
            self._enabled = False

    def ensure_initialized(self) -> None:
        """Attach the bbox annotator before render warm-up frames are produced."""
        if not self._enabled and not self._init_attempted:
            self._init()

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = int(getattr(self._env, "num_envs"))
        device = getattr(self._env, "device")
        occlusion = torch.full((num_envs,), float("nan"), dtype=torch.float32, device=device)
        found = torch.zeros((num_envs,), dtype=torch.bool, device=device)

        if not self._enabled and not self._init_attempted:
            self._init()
        if not self._enabled or not self._records:
            return occlusion, found

        debug_remaining = self._debug_samples if not self._debug_printed else 0
        debug_rows_printed = 0
        for record_env_id, annotator, _render_product in self._records:
            try:
                bbox_data, bbox_info = self._unpack_bbox_output(annotator.get_data())
            except Exception:
                continue
            prim_paths = bbox_info.get("primPaths", [])
            if bbox_data is None or not isinstance(prim_paths, Iterable):
                continue

            prim_paths = [str(path) for path in prim_paths]
            if debug_remaining > 0 and debug_rows_printed == 0:
                print(
                    f"[INFO] BBox debug: per_env_render_products={len(self._records)}, "
                    f"first_render_product_rows={len(prim_paths)}"
                )
            for idx, path in enumerate(prim_paths[:debug_remaining]):
                try:
                    row = bbox_data[idx]
                except Exception:
                    row = None
                print(f"[INFO] BBox debug env {record_env_id} row {idx}: path={path} row={row}")
                debug_rows_printed += 1
            debug_remaining = max(0, debug_remaining - len(prim_paths))

            for row_idx, prim_path in enumerate(prim_paths):
                env_id = self._extract_env_id_from_path(prim_path)
                if env_id is None or env_id < 0 or env_id >= num_envs:
                    continue
                # Each annotator is attached to one env camera. Ignore rows from other envs if Replicator returns any.
                if env_id != record_env_id:
                    continue
                can_prefix = f"/World/envs/env_{env_id}/can"
                if prim_path != can_prefix and not prim_path.startswith(f"{can_prefix}/"):
                    continue
                try:
                    ratio = self._extract_occlusion_ratio(bbox_data[row_idx])
                except Exception:
                    ratio = None
                if ratio is not None:
                    current = occlusion[env_id]
                    occlusion[env_id] = ratio if torch.isnan(current) else max(float(current.item()), ratio)
                    found[env_id] = True
            if debug_rows_printed >= self._debug_samples:
                self._debug_printed = True
        return occlusion, found

    def close(self) -> None:
        for _env_id, annotator, render_product in self._records:
            try:
                annotator.detach([render_product])
            except Exception:
                try:
                    annotator.detach(render_product)
                except Exception:
                    pass
        self._records = []
        self._enabled = False


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

    checkpoint_agent_cfg = _load_pickle_if_exists(os.path.join(run_dir, "params", "agent.pkl"))
    if checkpoint_agent_cfg is None:
        checkpoint_agent_cfg = _load_yaml_if_exists(os.path.join(run_dir, "params", "agent.yaml"))
    if checkpoint_agent_cfg is not None:
        experiment_cfg = checkpoint_agent_cfg
        print(f"[INFO] Loaded agent config from checkpoint run: {os.path.join(run_dir, 'params')}")

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
    if hasattr(env_cfg, "reset_jitter_max_steps"):
        # vt_box.py samples torch.randint(0, reset_jitter_max_steps);
        # 1 disables jitter while keeping the upper bound valid.
        env_cfg.reset_jitter_max_steps = 1
    if hasattr(env_cfg, "aux_alpha_use_bbox3d") and hasattr(env_cfg, "aux_alpha_fallback_occlusion"):
        # VisibleContact policies compute visual_visible_ratio = 1 - occlusion.
        # The legacy tiled-camera bbox path is unreliable for multi-env play, so bucket eval uses a fixed visible prior.
        env_cfg.aux_alpha_use_bbox3d = False
        env_cfg.aux_alpha_fallback_occlusion = 0.0
    try:
        env_cfg.can = env_cfg.can.replace(spawn=env_cfg.can.spawn.replace(semantic_tags=[("class", "can")]))
    except Exception:
        try:
            env_cfg.can.spawn.semantic_tags = [("class", "can")]
        except Exception:
            pass

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
    if hasattr(env_cfg, "reset_jitter_max_steps"):
        print(f"[INFO] Evaluation override: reset_jitter_max_steps={env_cfg.reset_jitter_max_steps}")
    if hasattr(env_cfg, "aux_alpha_use_bbox3d") and hasattr(env_cfg, "aux_alpha_fallback_occlusion"):
        print(f"[INFO] Evaluation override: aux_alpha_use_bbox3d={env_cfg.aux_alpha_use_bbox3d}")
        print(f"[INFO] Evaluation override: aux_alpha_fallback_occlusion={env_cfg.aux_alpha_fallback_occlusion}")
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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

    base_env = _get_base_env(env)
    env_cfg = getattr(base_env, "cfg", None)
    bucket_positions = _build_bucket_positions(args_cli)
    bucket_rounds = int(args_cli.bucket_rounds if args_cli.bucket_rounds is not None else args_cli.bucket_trials_per_position)
    total_trials = len(bucket_positions) * bucket_rounds
    bucket_xy_rows = []
    for round_index in range(bucket_rounds):
        for grid_index, x, y in bucket_positions:
            bucket_xy_rows.append((len(bucket_xy_rows), round_index, grid_index, x, y))

    device = getattr(base_env, "device", env.device)
    num_envs = int(env.num_envs)
    active = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    current_trial_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    current_grid_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    current_xy = torch.full((num_envs, 2), float("nan"), dtype=torch.float32, device=device)
    current_occlusion = torch.full((num_envs,), float("nan"), dtype=torch.float32, device=device)
    current_bbox_found = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    episode_success = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    success_hold = torch.zeros((num_envs,), dtype=torch.long, device=device)
    episode_steps = torch.zeros((num_envs,), dtype=torch.long, device=device)
    next_trial = 0
    completed_trials = 0
    bbox_reader = (
        _BucketBBoxOcclusionReader(env, debug_samples=args_cli.bucket_bbox_debug_samples)
        if args_cli.bucket_enable_bbox
        else None
    )
    grid_counts = [0 for _ in bucket_positions]
    grid_successes = [0 for _ in bucket_positions]
    grid_occlusion_sums = [0.0 for _ in bucket_positions]
    grid_occlusion_counts = [0 for _ in bucket_positions]

    def assign_trials(env_ids: torch.Tensor) -> None:
        nonlocal next_trial
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(device=device, dtype=torch.long)
        assignable = []
        xy_values = []
        trial_values = []
        grid_values = []
        for env_id in env_ids.detach().cpu().tolist():
            if next_trial >= total_trials:
                active[env_id] = False
                current_trial_idx[env_id] = -1
                current_grid_idx[env_id] = -1
                current_xy[env_id] = float("nan")
                current_occlusion[env_id] = float("nan")
                current_bbox_found[env_id] = False
                episode_success[env_id] = False
                success_hold[env_id] = 0
                episode_steps[env_id] = 0
                continue
            trial_id, round_id, grid_id, x, y = bucket_xy_rows[next_trial]
            assignable.append(env_id)
            xy_values.append((x, y))
            trial_values.append(trial_id)
            grid_values.append(grid_id)
            next_trial += 1

        if not assignable:
            return
        assign_env_ids = torch.tensor(assignable, dtype=torch.long, device=device)
        xy_tensor = torch.tensor(xy_values, dtype=torch.float32, device=device)
        _write_bucket_object_positions(env, assign_env_ids, xy_tensor)
        active[assign_env_ids] = True
        current_trial_idx[assign_env_ids] = torch.tensor(trial_values, dtype=torch.long, device=device)
        current_grid_idx[assign_env_ids] = torch.tensor(grid_values, dtype=torch.long, device=device)
        current_xy[assign_env_ids] = xy_tensor
        episode_success[assign_env_ids] = False
        success_hold[assign_env_ids] = 0
        episode_steps[assign_env_ids] = 0

        if bbox_reader is not None:
            bbox_reader.ensure_initialized()
            warmed = _warmup_rtx_cameras_after_reset(env, int(args_cli.bucket_position_warmup_steps))
            occlusion, bbox_found = bbox_reader.read()
            if args_cli.bucket_missing_occlusion_as == "one":
                occlusion = torch.where(bbox_found, occlusion, torch.ones_like(occlusion))
            current_occlusion[assign_env_ids] = occlusion[assign_env_ids]
            current_bbox_found[assign_env_ids] = bbox_found[assign_env_ids]
            return warmed
        current_occlusion[assign_env_ids] = float("nan")
        current_bbox_found[assign_env_ids] = False
        return None

    # reset environment, then overwrite the random object reset with the fixed bucket grid
    obs, _ = env.reset()
    initial_env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    warmed_obs = assign_trials(initial_env_ids)
    if warmed_obs is None:
        warmed_obs = _warmup_rtx_cameras_after_reset(env, args_cli.camera_warmup_steps)
    if warmed_obs is not None:
        obs, _ = warmed_obs
    timestep = 0
    third_person_cv2 = None
    if args_cli.show_third_person_env_id is not None:
        if args_cli.show_third_person_env_id < 0 or args_cli.show_third_person_env_id >= env.num_envs:
            raise ValueError(
                f"--show_third_person_env_id must be in [0, {env.num_envs - 1}], "
                f"got {args_cli.show_third_person_env_id}"
            )
        import cv2 as third_person_cv2

        print(f"[INFO] Showing third_person_camera preview for env {args_cli.show_third_person_env_id}")
    metrics_dir = os.path.join(log_dir, "metrics", "play_bucket")
    if args_cli.bucket_output is None:
        os.makedirs(metrics_dir, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.bucket_output)), exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    metrics_csv = args_cli.bucket_output or os.path.join(metrics_dir, f"play_bucket_{run_tag}.csv")
    metrics_file = open(metrics_csv, "w", encoding="utf-8", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(
        [
            "trial_index",
            "round_index",
            "grid_index",
            "x",
            "y",
            "occlusion_ratio",
            "bbox_found",
            "success",
            "terminated",
            "truncated",
            "episode_steps",
            "env_id",
            "sim_step",
        ]
    )
    metrics_file.flush()
    print(
        f"[INFO] Bucket grid: positions={len(bucket_positions)}, rounds={bucket_rounds}, "
        f"total_trials={total_trials}, assigned_initial={int(active.sum().item())}"
    )
    print(f"[INFO] Streaming bucket evaluation CSV: {metrics_csv}")
    if args_cli.bucket_enable_bbox:
        print("[INFO] Bucket bbox occlusion: enabled")
    else:
        print("[INFO] Bucket bbox occlusion: disabled")
    if args_cli.stochastic_eval:
        print("[INFO] Evaluation action mode: stochastic (sampled actions)")
    else:
        print("[INFO] Evaluation action mode: deterministic (mean actions)")
    try:
        # simulate environment
        while simulation_app.is_running() and completed_trials < total_trials:
            start_time = time.time()

            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                outputs = agent.act(obs, timestep=timestep, timesteps=timestep)
                actions = outputs[0] if args_cli.stochastic_eval else outputs[-1].get("mean_actions", outputs[0])
                # env stepping
                obs, _, terminated, truncated, infos = env.step(actions)

                heights = (
                    base_env._get_object_height_for_success()
                    if hasattr(base_env, "_get_object_height_for_success")
                    else base_env._can.data.root_pos_w[:, 2]
                )
                episode_steps[active] += 1
                success_now = heights > float(getattr(env_cfg, "success_height", 0.135))
                hold_steps = max(1, int(getattr(env_cfg, "success_hold_steps", 1)))
                success_hold = torch.where(
                    active & success_now,
                    torch.clamp(success_hold + 1, max=hold_steps),
                    torch.where(active, torch.zeros_like(success_hold), success_hold),
                )
                episode_success |= active & (success_hold >= hold_steps)

                current_step = timestep + 1
                terminated_tensor = _as_1d_bool_tensor(terminated, device, num_envs)
                truncated_tensor = _as_1d_bool_tensor(truncated, device, num_envs)
                done = active & (terminated_tensor | truncated_tensor)
                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.numel() > 0:
                    for env_id in done_ids.detach().cpu().tolist():
                        trial_id = int(current_trial_idx[env_id].item())
                        if trial_id < 0:
                            continue
                        metrics_writer.writerow(
                            [
                                trial_id,
                                trial_id // len(bucket_positions),
                                int(current_grid_idx[env_id].item()),
                                float(current_xy[env_id, 0].item()),
                                float(current_xy[env_id, 1].item()),
                                float(current_occlusion[env_id].item()),
                                int(bool(current_bbox_found[env_id].item())),
                                int(bool(episode_success[env_id].item())),
                                int(bool(terminated_tensor[env_id].item())),
                                int(bool(truncated_tensor[env_id].item())),
                                int(episode_steps[env_id].item()),
                                env_id,
                                current_step,
                            ]
                        )
                        grid_id = int(current_grid_idx[env_id].item())
                        success_value = int(bool(episode_success[env_id].item()))
                        occlusion_value = float(current_occlusion[env_id].item())
                        grid_counts[grid_id] += 1
                        grid_successes[grid_id] += success_value
                        if math.isfinite(occlusion_value):
                            grid_occlusion_sums[grid_id] += occlusion_value
                            grid_occlusion_counts[grid_id] += 1
                        completed_trials += 1
                    metrics_file.flush()
                    if completed_trials % max(1, min(100, total_trials)) == 0 or completed_trials >= total_trials:
                        print(f"[INFO] Bucket progress: {completed_trials}/{total_trials} trials completed")
                    warmed_obs = assign_trials(done_ids)
                    if warmed_obs is not None:
                        obs, _ = warmed_obs

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
        summary_csv = os.path.splitext(metrics_csv)[0] + "_summary.csv"
        with open(summary_csv, "w", encoding="utf-8", newline="") as summary_file:
            summary_writer = csv.writer(summary_file)
            summary_writer.writerow(
                ["grid_index", "x", "y", "trials", "successes", "success_rate", "occlusion_ratio_mean"]
            )
            for grid_id, x, y in bucket_positions:
                count = grid_counts[grid_id]
                successes = grid_successes[grid_id]
                success_rate = float("nan") if count <= 0 else successes / count
                occ_count = grid_occlusion_counts[grid_id]
                occlusion_mean = float("nan") if occ_count <= 0 else grid_occlusion_sums[grid_id] / occ_count
                summary_writer.writerow([grid_id, x, y, count, successes, success_rate, occlusion_mean])
        if bbox_reader is not None:
            bbox_reader.close()
        print(f"[INFO] Saved bucket evaluation CSV: {metrics_csv}")
        print(f"[INFO] Saved bucket summary CSV: {summary_csv}")

        # close the simulator
        env.close()
        if third_person_cv2 is not None:
            third_person_cv2.destroyAllWindows()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
