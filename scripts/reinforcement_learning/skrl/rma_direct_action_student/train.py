"""Online asymmetric action distillation for the three-input RMA Student."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    root = Path(__file__).resolve().parents[4]
    for name in ("tacex_tasks", "tacex", "tacex_assets", "tacex_uipc"):
        path = root / "source" / name
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_extend_repo_pythonpath()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="TacEx-Sim2Real-Cube-Real-Alignment-RMA-Direct-Action-Student-DR-v0")
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--encoder_init_checkpoint", required=True)
parser.add_argument("--resume", default=None)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--timesteps", type=int, default=100_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--backbone_learning_rate", type=float, default=3.0e-5)
parser.add_argument("--weight_decay", type=float, default=1.0e-5)
parser.add_argument("--action_loss_weight", type=float, default=1.0)
parser.add_argument("--action_smoothness_loss_weight", type=float, default=0.05)
parser.add_argument("--grad_norm_clip", type=float, default=1.0)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--checkpoint_interval", type=int, default=10_000)
parser.add_argument("--log_dir", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn.functional as F
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from torch.utils.tensorboard import SummaryWriter

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_grasp.rma_direct_action_student.artifacts import (
    RMA_DIRECT_ACTION_STUDENT_DR_TASK,
    load_encoder_initialization_checkpoint,
    load_student_checkpoint,
    load_student_model_state,
    load_teacher_manifest,
    make_student_payload,
    sha256_file,
    validate_live_env_contract,
)
from tacex_tasks.sim2real_grasp.rma_direct_action_student.models import RMADirectActionVisualStudent
from tacex_tasks.sim2real_grasp.rma_xy_artifacts import load_teacher_policy_state
from tacex_tasks.sim2real_grasp.rma_xy_models import RMAXYActorCore, extract_xy_actor_core_state_dict


def _atomic_torch_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_dump(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_teacher(checkpoint: Path, device: torch.device) -> RMAXYActorCore:
    state = load_teacher_policy_state(checkpoint, device)
    actor = RMAXYActorCore().to(device)
    actor.load_state_dict(extract_xy_actor_core_state_dict(state), strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def main() -> None:
    if args.task != RMA_DIRECT_ACTION_STUDENT_DR_TASK:
        raise ValueError(f"This trainer only supports {RMA_DIRECT_ACTION_STUDENT_DR_TASK}")
    if min(args.num_envs, args.timesteps, args.log_interval, args.checkpoint_interval) <= 0:
        raise ValueError("num_envs, timesteps, log_interval and checkpoint_interval must be positive")
    if min(args.learning_rate, args.backbone_learning_rate, args.grad_norm_clip) <= 0:
        raise ValueError("learning rates and grad_norm_clip must be positive")
    if min(args.action_loss_weight, args.action_smoothness_loss_weight, args.weight_decay) < 0:
        raise ValueError("loss weights and weight_decay must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    teacher_checkpoint = Path(args.teacher_checkpoint).expanduser().resolve()
    encoder_init_checkpoint = Path(args.encoder_init_checkpoint).expanduser().resolve()
    teacher_manifest = load_teacher_manifest(teacher_checkpoint)
    encoder_state, encoder_init_payload = load_encoder_initialization_checkpoint(encoder_init_checkpoint)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    validate_live_env_contract(env_cfg, teacher_manifest)
    environment_action_scale = torch.tensor(
        [float(env_cfg.action_scale)] * 3 + [float(env_cfg.gripper_width_delta_scale)],
        device=args.device,
        dtype=torch.float32,
    )
    run_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else (
        Path("logs/skrl/sim2real_cube_real_alignment_rma_direct_action_student").resolve()
        / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_distillation")
    )
    params_dir = run_dir / "params"
    _atomic_json_dump(
        {
            **vars(args),
            "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
            "encoder_init_checkpoint_sha256": sha256_file(encoder_init_checkpoint),
            "student_runtime_inputs": ["wrist_rgb", "proprio_obs", "action_history"],
            "training_only_privileged_inputs": ["rma_cube_xy", "rma_contact_force"],
            "environment_action_scales": environment_action_scale.cpu().tolist(),
            "backbone_trainable": "resnet18_layer3_layer4; all_batch_norm_eval",
        },
        params_dir / "direct_action_student_training.json",
    )

    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped
    device = torch.device(base_env.device)
    environment_action_scale = environment_action_scale.to(device)
    teacher = _load_teacher(teacher_checkpoint, device)
    student = RMADirectActionVisualStudent().to(device)
    student.load_vision_encoder_state(encoder_state)
    student.train()
    head_parameters = list(student.action_head.parameters())
    backbone_parameters = [p for p in student.vision_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": args.learning_rate},
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    loss_contract = {
        "action_distillation": "mse_teacher_mean_action",
        "action_weight": float(args.action_loss_weight),
        "action_smoothness": "mse_student_action_vs_previous_requested_normalized_action",
        "action_smoothness_weight": float(args.action_smoothness_loss_weight),
        "position_loss": None,
        "heatmap_loss": None,
        "contact_loss": None,
    }
    optimizer_config = {
        "class": "AdamW",
        "head_learning_rate": float(args.learning_rate),
        "backbone_learning_rate": float(args.backbone_learning_rate),
        "backbone_trainable": "resnet18_layer3_layer4",
        "batch_norm": "eval",
        "weight_decay": float(args.weight_decay),
        "grad_norm_clip": float(args.grad_norm_clip),
    }
    start_step = 0
    if args.resume:
        payload = load_student_checkpoint(
            args.resume, device=device, expected_teacher_checkpoint=teacher_checkpoint, expected_task=args.task
        )
        if payload.get("encoder_init_checkpoint_sha256") != sha256_file(encoder_init_checkpoint):
            raise RuntimeError("Resume checkpoint uses a different encoder initialization checkpoint")
        if payload.get("loss") != loss_contract or payload.get("optimizer_config") != optimizer_config:
            raise RuntimeError("Resume checkpoint loss or optimizer configuration differs")
        load_student_model_state(student, payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["global_step"])
    if start_step >= args.timesteps:
        raise RuntimeError("Resume checkpoint has already reached requested timesteps")

    writer = SummaryWriter(str(run_dir))
    observations, _ = env.reset()
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, args.timesteps + 1):
            obs = observations["policy"]
            proprio = obs["proprio_obs"].to(torch.float32)
            history = obs["action_history"].to(torch.float32)
            # These two tensors are intentionally never passed to Student. They
            # are privileged inputs used only by the frozen Teacher target.
            cube_xy = obs["rma_cube_xy"].to(torch.float32)
            contact_force_n = obs["rma_contact_force"].to(torch.float32)
            student_actions = student(obs["wrist_rgb"], proprio, history)
            with torch.no_grad():
                teacher_actions = teacher(proprio, history, cube_xy, contact_force_n)
                previous_actions = torch.clamp(history / environment_action_scale, -1.0, 1.0)
            action_loss = F.mse_loss(student_actions, teacher_actions)
            smoothness_loss = F.mse_loss(student_actions, previous_actions)
            loss = args.action_loss_weight * action_loss + args.action_smoothness_loss_weight * smoothness_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                head_parameters + backbone_parameters, args.grad_norm_clip
            )
            optimizer.step()
            observations, rewards, _, _, _ = env.step(student_actions.detach())

            action_rmse = torch.sqrt(torch.mean((student_actions.detach() - teacher_actions).square()))
            writer.add_scalar("Loss/total", loss.item(), step)
            writer.add_scalar("Loss/action_distillation", action_loss.item(), step)
            writer.add_scalar("Loss/action_smoothness", smoothness_loss.item(), step)
            writer.add_scalar("Action/teacher_rmse", action_rmse.item(), step)
            writer.add_scalar("Optimization/grad_norm", float(grad_norm), step)
            writer.add_scalar("Reward/mean_step", rewards.mean().item(), step)
            if step % args.log_interval == 0 or step == args.timesteps:
                stats = base_env._episode_success_statistics()
                elapsed = time.perf_counter() - started
                print(
                    f"[Direct-action Student] update={step:,}/{args.timesteps:,} "
                    f"loss(total/action/smooth)={loss.item():.5f}/{action_loss.item():.5f}/{smoothness_loss.item():.5f} "
                    f"action_rmse={action_rmse.item():.5f} "
                    f"success={float(stats['cumulative_rate'].item()):.3f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            if step % args.checkpoint_interval == 0 or step == args.timesteps:
                payload = make_student_payload(
                    model=student, optimizer=optimizer, global_step=step, task=args.task,
                    teacher_checkpoint=teacher_checkpoint, teacher_manifest=teacher_manifest,
                    encoder_init_checkpoint=encoder_init_checkpoint, encoder_init_payload=encoder_init_payload,
                    loss=loss_contract, optimizer_config=optimizer_config,
                )
                checkpoint = run_dir / "checkpoints" / f"student_{step:07d}.pt"
                _atomic_torch_save(payload, checkpoint)
                _atomic_torch_save(payload, checkpoint.parent / "latest.pt")
                print(f"[Direct-action Student] checkpoint saved: {checkpoint}", flush=True)
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
