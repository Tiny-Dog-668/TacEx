# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path
import importlib

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    """Make local extension packages importable when running from the repo checkout."""
    repo_root = Path(__file__).resolve().parents[3]
    source_root = repo_root / "source"
    package_roots = (
        source_root / "tacex_tasks",
        source_root / "tacex",
        source_root / "tacex_assets",
        source_root / "tacex_uipc",
    )
    for package_root in package_roots:
        package_root_str = str(package_root)
        if package_root.is_dir() and package_root_str not in sys.path:
            sys.path.insert(0, package_root_str)


_extend_repo_pythonpath()

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--save_start_frame",
    action="store_true",
    default=False,
    help="Capture and save training-start RGB frames after env reset.",
)
parser.add_argument(
    "--start_frame_count",
    type=int,
    default=5,
    help="Number of training-start RGB frames to save when start-frame capture is enabled.",
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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
original_argv = list(sys.argv)
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import os
import random
from datetime import datetime

import skrl
import torch
from packaging import version
from PIL import Image
from summary_utils import write_training_summary
from vision_encoder_artifact import (
    ENCODER_ARTIFACT_FILENAME,
    ENCODER_MANIFEST_FILENAME,
    checkpoint_run_dir,
    load_saved_agent_config,
    load_verified_vision_encoder,
    save_training_vision_encoder,
    validate_live_env_against_policy_contract,
    validate_live_vision_contract,
    validate_resume_agent_config,
    validate_sim2real_policy_contract,
)

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
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

# import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from isaaclab_tasks.utils.hydra import hydra_task_config

import tacex_tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"

SIM2REAL_VISION_ENCODER_TASKS = {
    "TacEx-Sim2Real-Grasp-v0",
    "TacEx-Sim2Real-Cube-Grasp-v0",
    "TacEx-Sim2Real-Cube-Real-Alignment-v0",
    "TacEx-Sim2Real-Cube-Real-Alignment-DR-v0",
}
from tacex_tasks.sim2real_grasp.rma_artifacts import RMA_TEACHER_TASKS


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


def _get_rgb_frame_from_sensor(sensor):
    """Return the first available RGB frame from a camera-like sensor."""
    if sensor is None:
        return None
    data = getattr(sensor, "data", None)
    output = getattr(data, "output", None)
    if output is None:
        return None
    rgb = output.get("rgb")
    if rgb is None or getattr(rgb, "numel", lambda: 0)() == 0:
        return None
    return rgb


def _find_preferred_rgb_sensor(base_env):
    """Find the most relevant RGB sensor on the environment."""
    preferred_names = ("wrist_camera", "third_person_camera")

    for name in preferred_names:
        sensor = getattr(base_env, name, None)
        if _get_rgb_frame_from_sensor(sensor) is not None or sensor is not None:
            return name, sensor

    scene = getattr(base_env, "scene", None)
    sensors = getattr(scene, "sensors", None) if scene is not None else None
    if isinstance(sensors, dict):
        for name in preferred_names:
            sensor = sensors.get(name)
            if _get_rgb_frame_from_sensor(sensor) is not None or sensor is not None:
                return name, sensor
        for name, sensor in sensors.items():
            if _get_rgb_frame_from_sensor(sensor) is not None:
                return name, sensor
    return None, None


def _rgb_tensor_to_uint8_image(frame: torch.Tensor) -> torch.Tensor | None:
    """Convert one HWC RGB/RGBA tensor to uint8 RGB on CPU."""
    image = frame.detach().cpu()
    if image.ndim != 3:
        return None
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.dtype.is_floating_point:
        if torch.max(image).item() <= 1.0 + 1e-6:
            image = image * 255.0
        image = image.round().clamp(0, 255).to(torch.uint8)
    else:
        image = image.clamp(0, 255).to(torch.uint8)
    return image


def _save_training_start_camera_frames(env, log_dir: str, frame_count: int) -> None:
    """Capture RGB frames after reset and save them into the run log directory."""
    base_env = env.unwrapped
    sensor_name, sensor = _find_preferred_rgb_sensor(base_env)
    if sensor is None:
        print("[INFO] No RGB camera sensor found. Skipping training-start camera snapshots.")
        return

    frame_count = max(1, int(frame_count))
    output_dir = os.path.join(log_dir, "camera_frames")
    os.makedirs(output_dir, exist_ok=True)

    env.reset()

    saved_count = 0
    attempts = max(frame_count * 3, frame_count + 3)
    for _ in range(attempts):
        if base_env.sim.has_rtx_sensors():
            base_env.sim.render()
        base_env.scene.update(dt=base_env.physics_dt)
        frame = _get_rgb_frame_from_sensor(sensor)

        if frame is None:
            continue

        image = _rgb_tensor_to_uint8_image(frame[0])
        if image is None:
            print(f"[WARN] Unexpected RGB frame shape for sensor '{sensor_name}': {tuple(frame[0].shape)}")
            return

        height, width = image.shape[:2]
        output_path = os.path.join(
            output_dir,
            f"start_{sensor_name}_env0_frame_{saved_count:03d}_{width}x{height}.png",
        )
        Image.fromarray(image.numpy()).save(output_path)
        saved_count += 1
        if saved_count >= frame_count:
            break

    if saved_count == 0:
        print(f"[WARN] RGB camera sensor '{sensor_name}' is present but no frame was available.")
    else:
        print(f"[INFO] Saved {saved_count} training-start camera frame(s) to: {output_dir}")


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    write_training_summary(log_dir, env_cfg, agent_cfg, args_cli, hydra_args, algorithm, original_argv)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    print(f"[INFO] Creating gym environment: task={args_cli.task}, num_envs={env_cfg.scene.num_envs}, device={env_cfg.sim.device}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("[INFO] Gym environment created.")

    if args_cli.task in RMA_TEACHER_TASKS and (
        not args_cli.distributed or app_launcher.local_rank == 0
    ):
        from tacex_tasks.sim2real_grasp.rma_artifacts import (
            load_teacher_manifest,
            validate_live_env_contract,
            write_teacher_manifest,
        )

        if resume_path is not None:
            source_manifest = load_teacher_manifest(resume_path)
            validate_live_env_contract(env.unwrapped.cfg, source_manifest)
        manifest_path = write_teacher_manifest(
            env.unwrapped, os.path.join(log_dir, "params"), agent_cfg
        )
        print(f"[INFO] Saved RMA teacher manifest: {manifest_path}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo", "ppo_rnn"]:
        env = multi_agent_to_single_agent(env)

    # Every task in this inheritance chain receives the BN/history semantic
    # change. Cube additionally requires the tanh contract in the validator.
    if args_cli.task in SIM2REAL_VISION_ENCODER_TASKS and resume_path is not None:
        source_params_dir = checkpoint_run_dir(resume_path) / "params"
        source_manifest_path = source_params_dir / ENCODER_MANIFEST_FILENAME
        source_contract = validate_sim2real_policy_contract(
            source_manifest_path,
            task=args_cli.task,
            params_dir=source_params_dir,
        )
        source_agent_cfg, _ = load_saved_agent_config(source_params_dir)
        validate_resume_agent_config(agent_cfg, source_agent_cfg)
        source_encoder_manifest = load_verified_vision_encoder(
            env.unwrapped._resnet18,
            artifact_path=source_params_dir / ENCODER_ARTIFACT_FILENAME,
            manifest_path=source_manifest_path,
        )
        validate_live_vision_contract(env.unwrapped, source_encoder_manifest)
        validate_live_env_against_policy_contract(env.unwrapped, source_contract)
        print(f"[INFO] Verified strict sim2real resume contract from: {source_params_dir}")

    if args_cli.task in SIM2REAL_VISION_ENCODER_TASKS and (
        not args_cli.distributed or app_launcher.local_rank == 0
    ):
        encoder_manifest = save_training_vision_encoder(
            env.unwrapped,
            os.path.join(log_dir, "params"),
            task=args_cli.task,
            policy_output=agent_cfg.get("models", {}).get("policy", {}).get("output"),
        )
        if encoder_manifest is None:
            raise RuntimeError(f"Task {args_cli.task} requires a frozen _resnet18 vision encoder.")
        print(
            "[INFO] Saved verified training vision encoder: "
            f"state_dict_sha256={encoder_manifest['state_dict_sha256']}"
        )

    auto_save_start_frames = args_cli.task == "TacEx-Sim2Real-Cube-Grasp-v0"
    if args_cli.save_start_frame or auto_save_start_frames:
        print("[INFO] Capturing training-start camera frames...")
        _save_training_start_camera_frames(env, log_dir, args_cli.start_frame_count)
    else:
        print("[INFO] Skip training-start camera frames (use --save_start_frame to enable).")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # print whether the policy config requests an RNN (e.g., LSTM) to verify it is enabled
    policy_cfg = agent_cfg.get("models", {}).get("policy", {})
    is_recurrent = bool(policy_cfg.get("recurrent", False))
    rnn_type = policy_cfg.get("rnn_type") or policy_cfg.get("recurrent_type") or "none"
    rnn_units = policy_cfg.get("rnn_units") or policy_cfg.get("recurrent_hidden_size")
    rnn_layers = policy_cfg.get("rnn_layers") or policy_cfg.get("recurrent_layers")
    seq_len = agent_cfg.get("agent", {}).get("sequence_length")
    if is_recurrent:
        print(f"[INFO] Recurrent policy detected: type={rnn_type}, hidden={rnn_units}, layers={rnn_layers}, sequence_length={seq_len}")
    else:
        print("[INFO] Policy is non-recurrent (no RNN/LSTM enabled).")

    # 如果是 LSTM 版本或显式使用 PPO_RNN，则走自定义 CylinderFusionLSTM 流程
    LSTM_TASK_IDS = {
        "TacEx-Cylinder-Grasping-Four-Tactile-RGB-v0",
    }
    align_cfg = agent_cfg.get("align", {}) or {}
    use_shared_latent = bool(align_cfg.get("enable", False))
    if use_shared_latent and algorithm != "ppo":
        print(f"[WARN] Shared latent alignment is only wired for PPO. Got algorithm={algorithm}.")
        use_shared_latent = False
    USE_CUSTOM_POLICY = (
        args_cli.task in LSTM_TASK_IDS
        or algorithm == "ppo_rnn"
    )
    # 兜底：recurrent 策略 + 触觉 resnet 观测则强制走自定义 LSTM（避免误走 Runner 分支）
    tactile_keys = {
        "tactile_left_resnet", "tactile_right_resnet", "tactile_left_down_resnet", "tactile_right_down_resnet",
        "tactile_left_depth_resnet", "tactile_right_depth_resnet", "tactile_left_down_depth_resnet", "tactile_right_down_depth_resnet",
        "tactile_left_rgb", "tactile_right_rgb", "tactile_left_down_depth", "tactile_right_down_depth",
    }
    obs_space = getattr(env, "observation_space", None)
    if hasattr(obs_space, "spaces"):
        obs_keys = set(obs_space.spaces.keys())
    else:
        try:
            obs_keys = set(obs_space.keys()) if obs_space is not None else set()
        except AttributeError:
            obs_keys = set()
    if (not USE_CUSTOM_POLICY) and is_recurrent and obs_keys.intersection(tactile_keys):
        print("[INFO] Enabling custom CylinderFusionLSTM (recurrent policy + tactile_resnet observations).")
        USE_CUSTOM_POLICY = True
    if use_shared_latent:
        USE_CUSTOM_POLICY = True

    agent_class_spec = str(agent_cfg.get("agent", {}).get("class", ""))
    use_custom_agent_class = ":" in agent_class_spec

    if not USE_CUSTOM_POLICY and not use_custom_agent_class:
        # configure and instantiate the skrl runner
        # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
        runner = Runner(env, agent_cfg)

        # extra proof that an RNN (e.g., LSTM) is wired into the policy
        try:
            policy_model = runner.agent.models.get("policy", None) if hasattr(runner, "agent") else None
            if policy_model is not None and hasattr(policy_model, "get_specification"):
                spec = policy_model.get_specification() or {}
                rnn_spec = spec.get("rnn", None)
                if rnn_spec:
                    print(f"[INFO] Policy model is recurrent with spec: {rnn_spec}")
                else:
                    print("[INFO] Policy model reports no RNN spec (non-recurrent).")
            else:
                print("[WARN] Could not inspect policy model for RNN spec.")
        except Exception as e:
            print(f"[WARN] Failed to inspect policy model for RNN details: {e}")

        # load checkpoint (if specified)
        if resume_path:
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner.agent.load(resume_path)

        # run training
        runner.run()
    else:
        if use_shared_latent:
            print("[INFO] Using custom shared-latent policy with PPO (manual agent/trainer path)")

            from skrl.utils.model_instantiators.torch import deterministic_model
            from skrl.memories.torch import RandomMemory
            from skrl.trainers.torch import SequentialTrainer
            from custom_agents import PPOWithAlignLoss
            from custom_models import VisionTactileSharedLatentPolicy
            from skrl.resources.preprocessors.torch import RunningStandardScaler

            policy_cfg = agent_cfg["models"]["policy"]

            if agent_cfg["agent"].get("state_preprocessor_kwargs") is None:
                agent_cfg["agent"]["state_preprocessor_kwargs"] = {}
            if agent_cfg["agent"].get("value_preprocessor_kwargs") is None:
                agent_cfg["agent"]["value_preprocessor_kwargs"] = {}

            agent_cfg["agent"]["state_preprocessor"] = None
            agent_cfg["agent"]["value_preprocessor"] = RunningStandardScaler

            v_kwargs = agent_cfg["agent"]["value_preprocessor_kwargs"]
            v_kwargs.setdefault("size", 1)
            v_kwargs.setdefault("device", env.device)
            agent_cfg["agent"]["value_preprocessor_kwargs"] = v_kwargs
            agent_cfg["agent"] = _process_cfg(agent_cfg["agent"])

            policy_model = VisionTactileSharedLatentPolicy(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                clip_actions=policy_cfg.get("clip_actions", False),
                clip_log_std=policy_cfg.get("clip_log_std", True),
                min_log_std=policy_cfg.get("min_log_std", -20.0),
                max_log_std=policy_cfg.get("max_log_std", 2.0),
                reduction=policy_cfg.get("reduction", "sum"),
                initial_log_std=policy_cfg.get("initial_log_std", 0.0),
                fixed_log_std=policy_cfg.get("fixed_log_std", False),
                mlp_layers=policy_cfg.get("mlp_layers", [512, 256, 128, 64]),
                mlp_activation=policy_cfg.get("mlp_activation", "elu"),
                latent_dim=align_cfg.get("latent_dim", 128),
                align_cfg=align_cfg,
            )

            value_model = deterministic_model(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                **agent_cfg["models"]["value"],
            )

            models = {"policy": policy_model, "value": value_model}

            memory = RandomMemory(
                memory_size=agent_cfg["agent"]["rollouts"],
                num_envs=env.num_envs,
                device=env.device,
            )

            agent = PPOWithAlignLoss(
                models=models,
                memory=memory,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                cfg=agent_cfg["agent"],
                align_cfg=align_cfg,
            )

            if resume_path:
                print(f"[INFO] Loading model checkpoint from: {resume_path}")
                agent.load(resume_path)

            trainer = SequentialTrainer(cfg=agent_cfg["trainer"], env=env, agents=agent)
            trainer.train()
        elif use_custom_agent_class:
            print(f"[INFO] Using custom agent class '{agent_class_spec}' (manual agent/trainer path)")

            from skrl.utils.model_instantiators.torch import gaussian_model, deterministic_model
            from skrl.memories.torch import RandomMemory
            from skrl.trainers.torch import SequentialTrainer
            from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG

            policy_cfg_local = copy.deepcopy(agent_cfg["models"]["policy"])
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

            value_cfg_local = copy.deepcopy(agent_cfg["models"]["value"])
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

            memory_cfg_local = copy.deepcopy(agent_cfg.get("memory", {}))
            memory_class_spec = memory_cfg_local.pop("class", "RandomMemory")
            memory_cls = _load_component(memory_class_spec)
            if memory_cls is None:
                memory_cls = RandomMemory
            if int(memory_cfg_local.get("memory_size", -1)) < 0:
                memory_cfg_local["memory_size"] = int(agent_cfg["agent"].get("rollouts", 128))
            memory = memory_cls(
                num_envs=env.num_envs,
                device=env.device,
                **_process_cfg(memory_cfg_local),
            )

            agent_runtime_cfg = copy.deepcopy(agent_cfg["agent"])
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

            if resume_path:
                print(f"[INFO] Loading model checkpoint from: {resume_path}")
                agent.load(resume_path)

            trainer_cfg_local = copy.deepcopy(agent_cfg["trainer"])
            trainer_class_spec = str(trainer_cfg_local.pop("class", "SequentialTrainer"))
            trainer_cls = _load_component(trainer_class_spec)
            if trainer_cls is None:
                trainer_cls = SequentialTrainer
            trainer = trainer_cls(cfg=trainer_cfg_local, env=env, agents=agent)
            trainer.train()
        else:
            print("[INFO] Using custom CylinderFusionLSTM policy with PPO_RNN (manual agent/trainer path)")

            from skrl.utils.model_instantiators.torch import deterministic_model
            from skrl.memories.torch import RandomMemory
            try:
                from skrl.agents.torch.ppo.ppo_rnn import PPO_RNN  # skrl>=1.4.3 package layout
            except ImportError:
                from skrl.agents.torch.ppo_rnn import PPO_RNN  # fallback for other layouts
            from skrl.trainers.torch import SequentialTrainer

            from skrl.resources.preprocessors.torch import RunningStandardScaler
            from custom_models import CylinderFusionLSTM

            seq_len = agent_cfg["agent"].get("sequence_length", 64)
            policy_cfg = agent_cfg["models"]["policy"]
            hidden_size = policy_cfg.get("rnn_units", policy_cfg.get("recurrent_hidden_size", 256))
            num_layers = policy_cfg.get("rnn_layers", policy_cfg.get("recurrent_layers", 1))

            if agent_cfg["agent"].get("state_preprocessor_kwargs") is None:
                agent_cfg["agent"]["state_preprocessor_kwargs"] = {}
            if agent_cfg["agent"].get("value_preprocessor_kwargs") is None:
                agent_cfg["agent"]["value_preprocessor_kwargs"] = {}

            agent_cfg["agent"]["state_preprocessor"] = None
            agent_cfg["agent"]["value_preprocessor"] = RunningStandardScaler

            v_kwargs = agent_cfg["agent"]["value_preprocessor_kwargs"]
            v_kwargs.setdefault("size", 1)
            v_kwargs.setdefault("device", env.device)
            agent_cfg["agent"]["value_preprocessor_kwargs"] = v_kwargs
            agent_cfg["agent"] = _process_cfg(agent_cfg["agent"])

            policy_model = CylinderFusionLSTM(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                num_envs=env.num_envs,
                sequence_length=seq_len,
                num_layers=num_layers,
                hidden_size=hidden_size,
                use_vision_placeholder=agent_cfg["models"]["policy"].get("use_vision_placeholder", True),
                clip_actions=agent_cfg["models"]["policy"].get("clip_actions", False),
                clip_log_std=agent_cfg["models"]["policy"].get("clip_log_std", True),
                min_log_std=agent_cfg["models"]["policy"].get("min_log_std", -20.0),
                max_log_std=agent_cfg["models"]["policy"].get("max_log_std", 2.0),
                reduction=agent_cfg["models"]["policy"].get("reduction", "sum"),
                stats_print_every=policy_cfg.get("stats_print_every", 0),
                reset_stats_print=policy_cfg.get("reset_stats_print", False),
            )

            value_model = deterministic_model(
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                **agent_cfg["models"]["value"],
            )

            models = {"policy": policy_model, "value": value_model}

            memory = RandomMemory(
                memory_size=agent_cfg["agent"]["rollouts"],
                num_envs=env.num_envs,
                device=env.device,
            )

            agent = PPO_RNN(
                models=models,
                memory=memory,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device,
                cfg=agent_cfg["agent"],
            )

            try:
                spec = policy_model.get_specification() if hasattr(policy_model, "get_specification") else {}
                rnn_spec = spec.get("rnn", None)
                if rnn_spec:
                    print(f"[INFO] CylinderFusionLSTM recurrent spec: {rnn_spec}")
            except Exception:
                pass

            if resume_path:
                print(f"[INFO] Loading model checkpoint from: {resume_path}")
                agent.load(resume_path)

            trainer = SequentialTrainer(cfg=agent_cfg["trainer"], env=env, agents=agent)
            trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
