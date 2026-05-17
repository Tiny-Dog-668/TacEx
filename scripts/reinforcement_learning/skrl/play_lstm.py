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

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
parser.add_argument("--seed", type=int, default=42, help="Random seed for env/agent; -1 for random.")

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
import os
import time
import torch
import random

import skrl
from packaging import version

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
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import tacex_tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # resolve seed
    seed = args_cli.seed
    if seed == -1:
        seed = random.randint(0, 10000)
    if seed is not None:
        print(f"[INFO] Using seed: {seed}")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
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
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # propagate seed into configs
    if seed is not None:
        try:
            env_cfg.seed = seed
        except Exception:
            pass
        try:
            experiment_cfg["seed"] = seed
            experiment_cfg["agent"]["seed"] = seed
        except Exception:
            pass

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
        print("[INFO] Recording videos during training.")
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

    if not USE_CUSTOM_POLICY:
        # configure and instantiate the skrl runner
        # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
        runner = Runner(env, experiment_cfg)

        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        # set agent to evaluation mode
        runner.agent.set_running_mode("eval")
        agent = runner.agent
    else:
        print("[INFO] Using custom CylinderFusionLSTM policy for play (PPO_RNN path)")
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

        # disable preprocessors to avoid dict/shape mismatch
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
        try:
            spec = policy_model.get_specification().get("rnn", None)
            if spec:
                print(f"[INFO] CylinderFusionLSTM recurrent spec: {spec}")
        except Exception:
            pass
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
    # some wrappers don't accept seed kwarg; try and fall back
    try:
        obs, _ = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = agent.act(obs, timestep=timestep, timesteps=timestep)
            actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
