"""Print per-env occlusion prediction and pseudo-GT for Tactile-Cross-Alpha-Aux."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_root = repo_root / "source"
    for package_root in (
        source_root / "tacex_tasks",
        source_root / "tacex",
        source_root / "tacex_assets",
        source_root / "tacex_uipc",
    ):
        package_root_str = str(package_root)
        if package_root.is_dir() and package_root_str not in sys.path:
            sys.path.insert(0, package_root_str)


parser = argparse.ArgumentParser(description="Test Tactile-Cross-Alpha-Aux occlusion prediction vs pseudo-GT.")
parser.add_argument(
    "--task",
    type=str,
    default="TacEx-Tactile-Cross-Alpha-Aux-Downsample-Drawer-Occlusion-Cylinder",
    help="Task ID to instantiate.",
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--steps", type=int, default=1, help="Number of random-policy-free observation steps to print.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import yaml
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

_extend_repo_pythonpath()
import tacex_tasks  # noqa: F401, E402


def _load_class(class_path: str):
    module_path, class_name = class_path.split(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _select_policy_obs(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    return obs


def _select_policy_space(space):
    if hasattr(space, "spaces") and "policy" in space.spaces:
        return space.spaces["policy"]
    return space


def _to_cpu_1d(value: torch.Tensor) -> list[float]:
    return value.detach().reshape(value.shape[0], -1).squeeze(-1).to("cpu", dtype=torch.float32).tolist()


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
    if isinstance(agent_cfg, str):
        with open(agent_cfg, encoding="utf-8") as f:
            agent_cfg = yaml.safe_load(f)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        obs, _ = env.reset()
        policy_obs = _select_policy_obs(obs)
        observation_space = _select_policy_space(env.observation_space)
        action_space = env.action_space
        device = getattr(env.unwrapped, "device", args_cli.device)

        policy_cfg = dict(agent_cfg["models"]["policy"])
        policy_cls = _load_class(policy_cfg.pop("class"))
        policy = policy_cls(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            **policy_cfg,
        ).to(device)
        policy.eval()

        zero_action = torch.zeros((args_cli.num_envs, action_space.shape[0]), dtype=torch.float32, device=device)

        for step in range(args_cli.steps):
            if step > 0:
                obs, _, _, _, _ = env.step(zero_action)
                policy_obs = _select_policy_obs(obs)

            with torch.no_grad():
                _, _, outputs = policy.compute({"states": policy_obs}, role="policy")

            pred = _to_cpu_1d(outputs["occlusion_pred_values"])
            gt = _to_cpu_1d(outputs["occlusion_gt_values"])
            alpha = _to_cpu_1d(outputs["alpha_values"])
            visible = (
                _to_cpu_1d(policy_obs["pseudo_visible_ratio"])
                if isinstance(policy_obs, dict) and "pseudo_visible_ratio" in policy_obs
                else [float("nan")] * len(pred)
            )
            contact = (
                policy_obs["aux_tactile_contact_gt"].detach().to("cpu", dtype=torch.float32).tolist()
                if isinstance(policy_obs, dict) and "aux_tactile_contact_gt" in policy_obs
                else None
            )

            print(f"\n[step {step}] occlusion prediction vs pseudo-GT")
            print("env_id | pred_occlusion | gt_occlusion | pseudo_visible | alpha | contact_4")
            for env_id, (pred_value, gt_value, visible_value, alpha_value) in enumerate(zip(pred, gt, visible, alpha)):
                contact_text = "-" if contact is None else "[" + ", ".join(f"{v:.3f}" for v in contact[env_id]) + "]"
                print(
                    f"{env_id:>6} | {pred_value:>14.6f} | {gt_value:>12.6f} | "
                    f"{visible_value:>14.6f} | {alpha_value:>5.3f} | {contact_text}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
