"""Distill the GelSight X040-DR three-frame Student from its paired Teacher."""

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
    for relative in ("source/tacex_tasks", "source/tacex", "source/tacex_assets"):
        value = str(root / relative)
        if value not in sys.path:
            sys.path.insert(0, value)


_extend_repo_pythonpath()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default=(
        "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-X040-DR-Size-Buckets-"
        "Three-Frame-Direct-Action-Student-DR-v0"
    ),
)
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--resume", default=None)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--timesteps", type=int, default=100_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--backbone_learning_rate", type=float, default=3.0e-5)
parser.add_argument(
    "--train_backbone_after_layer2",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--weight_decay", type=float, default=1.0e-5)
parser.add_argument("--position_loss_weight", type=float, default=1.0)
parser.add_argument("--contact_loss_weight", type=float, default=1.0)
parser.add_argument("--contact_positive_weight", type=float, default=5.0)
parser.add_argument("--action_loss_weight", type=float, default=1.0)
parser.add_argument("--action_smoothness_loss_weight", type=float, default=0.05)
parser.add_argument("--smooth_l1_beta", type=float, default=0.1)
parser.add_argument("--grad_norm_clip", type=float, default=1.0)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--checkpoint_interval", type=int, default=10_000)
parser.add_argument("--log_dir", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
import torch.nn.functional as F
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from torch.utils.tensorboard import SummaryWriter

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_x040_three_frame_artifacts import (
    GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK,
    load_student_checkpoint,
    load_student_model_state,
    load_teacher_manifest,
    load_teacher_policy_state,
    make_student_payload,
    sha256_file,
    state_dict_sha256,
    student_environment_contract,
    validate_live_teacher_contract,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_x040_three_frame_models import (
    RMAGelSightX040ThreeFrameStudent,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_models import (
    RMAGelSightActorCore,
)
from tacex_tasks.sim2real_grasp.rma_models import (
    extract_actor_core_state_dict,
)


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_dump(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _loss_contract() -> dict[str, object]:
    return {
        "position": "smooth_l1_normalized_cube_position_root_xyz",
        "position_weight": float(args.position_loss_weight),
        "smooth_l1_beta": float(args.smooth_l1_beta),
        "contact": "left_right_bce_from_signed_current_minus_reference",
        "contact_weight": float(args.contact_loss_weight),
        "contact_positive_weight": float(args.contact_positive_weight),
        "action": "mse_teacher_deterministic_mean",
        "action_weight": float(args.action_loss_weight),
        "action_smoothness": "mse_to_previous_environment_action",
        "action_smoothness_weight": float(args.action_smoothness_loss_weight),
        "heatmap": "absent",
    }


def _optimizer_contract() -> dict[str, object]:
    return {
        "class": "AdamW",
        "learning_rate": float(args.learning_rate),
        "backbone_learning_rate": float(args.backbone_learning_rate),
        "train_backbone_after_layer2": bool(args.train_backbone_after_layer2),
        "weight_decay": float(args.weight_decay),
        "grad_norm_clip": float(args.grad_norm_clip),
    }


def main() -> None:
    if args.task != GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK:
        raise ValueError(f"This trainer only supports {GELSIGHT_X040_DR_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASK}")
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if min(args.timesteps, args.log_interval, args.checkpoint_interval) <= 0:
        raise ValueError("timesteps, log_interval, and checkpoint_interval must be positive")
    if min(args.learning_rate, args.backbone_learning_rate, args.grad_norm_clip) <= 0:
        raise ValueError("learning rates and grad_norm_clip must be positive")
    if min(
        args.position_loss_weight,
        args.contact_loss_weight,
        args.action_loss_weight,
        args.action_smoothness_loss_weight,
        args.weight_decay,
    ) < 0:
        raise ValueError("loss weights and weight_decay must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    teacher_checkpoint = Path(args.teacher_checkpoint).expanduser().resolve()
    teacher_manifest = load_teacher_manifest(teacher_checkpoint)
    resume_payload = None
    if args.resume:
        resume_payload = load_student_checkpoint(
            args.resume,
            device="cpu",
            expected_teacher_checkpoint=teacher_checkpoint,
        )
        if resume_payload.get("loss") != _loss_contract():
            raise RuntimeError("Resume loss contract differs from current arguments")
        if resume_payload.get("optimizer_config") != _optimizer_contract():
            raise RuntimeError("Resume optimizer contract differs from current arguments")

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    if resume_payload is not None:
        env_cfg.illegal_collision_curriculum_step_offset = int(resume_payload["global_step"])
    validate_live_teacher_contract(env_cfg, teacher_manifest)
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped
    device = torch.device(base_env.device)

    teacher = RMAGelSightActorCore().to(device).eval()
    teacher.load_state_dict(
        extract_actor_core_state_dict(load_teacher_policy_state(teacher_checkpoint, device)),
        strict=True,
    )
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    model = RMAGelSightX040ThreeFrameStudent(pretrained_backbone=True).to(device)
    head_parameters = (
        list(model.temporal_fusion.parameters())
        + list(model.tactile_encoder.parameters())
        + list(model.position_head.parameters())
        + list(model.action_head.parameters())
    )
    trainable = list(head_parameters)
    groups = [{"params": head_parameters, "lr": args.learning_rate}]
    if args.train_backbone_after_layer2:
        model.unfreeze_backbone_after_layer2()
        backbone_parameters = [p for p in model.vision_encoder.parameters() if p.requires_grad]
        trainable += backbone_parameters
        groups.append({"params": backbone_parameters, "lr": args.backbone_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    run_dir = (
        Path(args.log_dir).expanduser().resolve()
        if args.log_dir
        else Path("logs/skrl/sim2real_cube_real_alignment_rma_gelsight_x040_dr_three_frame_student")
        / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_distillation"
    )
    _atomic_json_dump(
        {
            **vars(args),
            "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
            "teacher_actor_state_dict_sha256": state_dict_sha256(teacher.state_dict()),
            "runtime_inputs": [
                "wrist_rgb_history",
                "proprio_obs",
                "action_history",
                "gsmini_left_rgb",
                "gsmini_right_rgb",
                "gsmini_left_reference_rgb",
                "gsmini_right_reference_rgb",
            ],
            "runtime_outputs": [
                "action",
                "left_right_contact_probability",
                "cube_position_root_m",
            ],
            "training_only_labels": ["rma_cube_pos", "rma_contact_state"],
        },
        run_dir / "params" / "training.json",
    )
    writer = SummaryWriter(str(run_dir))
    start_step = 0
    if resume_payload is not None:
        load_student_model_state(model, resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        start_step = int(resume_payload["global_step"])
    model.train()

    observations, _ = env.reset()
    action_scale = torch.tensor(
        [env_cfg.action_scale] * 3 + [env_cfg.gripper_width_delta_scale],
        device=device,
        dtype=torch.float32,
    )
    positive_weight = torch.full((2,), args.contact_positive_weight, device=device)
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, args.timesteps + 1):
            obs = observations["policy"]
            proprio = obs["proprio_obs"].float()
            history = obs["action_history"].float()
            cube_position = obs["rma_cube_pos"].float()
            contact_target = obs["rma_contact_state"].float()
            student_actions, normalized_position, contact_logits = model.forward_with_training_outputs(
                obs["wrist_rgb_history"],
                proprio,
                history,
                obs["gsmini_left_rgb"],
                obs["gsmini_right_rgb"],
                obs["gsmini_left_reference_rgb"],
                obs["gsmini_right_reference_rgb"],
            )
            with torch.no_grad():
                teacher_actions = teacher(proprio, history, cube_position, contact_target)
                previous_actions = torch.clamp(history / action_scale, -1.0, 1.0)
                position_target = model.normalizer.normalize_position(cube_position)
            action_loss = F.mse_loss(student_actions, teacher_actions)
            smoothness_loss = F.mse_loss(student_actions, previous_actions)
            position_loss = F.smooth_l1_loss(
                normalized_position, position_target, beta=args.smooth_l1_beta
            )
            contact_loss = F.binary_cross_entropy_with_logits(
                contact_logits, contact_target, pos_weight=positive_weight
            )
            loss = (
                args.action_loss_weight * action_loss
                + args.action_smoothness_loss_weight * smoothness_loss
                + args.position_loss_weight * position_loss
                + args.contact_loss_weight * contact_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_norm_clip)
            optimizer.step()
            observations, rewards, _, _, _ = env.step(student_actions.detach())

            with torch.no_grad():
                position_rmse = torch.sqrt(
                    torch.mean(
                        (
                            model.normalizer.denormalize_position(normalized_position)
                            - cube_position
                        ).square()
                    )
                )
                contact_probability = torch.sigmoid(contact_logits)
                contact_accuracy = ((contact_probability >= 0.5) == (contact_target >= 0.5)).float().mean()
            writer.add_scalar("Loss/total", loss.item(), step)
            writer.add_scalar("Loss/position", position_loss.item(), step)
            writer.add_scalar("Loss/contact_bce", contact_loss.item(), step)
            writer.add_scalar("Loss/action", action_loss.item(), step)
            writer.add_scalar("Loss/action_smoothness", smoothness_loss.item(), step)
            writer.add_scalar("Position/rmse_m", position_rmse.item(), step)
            writer.add_scalar("Contact/mean_accuracy", contact_accuracy.item(), step)
            writer.add_scalar("Action/teacher_rmse", torch.sqrt(action_loss.detach()).item(), step)
            writer.add_scalar("Optimization/grad_norm", float(grad_norm), step)
            writer.add_scalar("Reward/mean_step", rewards.mean().item(), step)

            if step % args.log_interval == 0 or step == args.timesteps:
                stats = base_env._episode_success_statistics()
                writer.add_scalar("Performance/recent_success_rate", stats["window_rate"].item(), step)
                print(
                    f"[GelSight X040 Three-Frame Student] {step}/{args.timesteps} "
                    f"loss(action/smooth/pos/contact)={action_loss.item():.4f}/"
                    f"{smoothness_loss.item():.4f}/{position_loss.item():.4f}/{contact_loss.item():.4f} "
                    f"position_rmse={1000.0 * position_rmse.item():.1f}mm "
                    f"contact_accuracy={contact_accuracy.item():.3f} "
                    f"success={stats['cumulative_rate'].item():.3f} "
                    f"updates_s={(step - start_step) / max(time.perf_counter() - started, 1.0e-6):.2f}",
                    flush=True,
                )

            if step % args.checkpoint_interval == 0 or step == args.timesteps:
                payload = make_student_payload(
                    model=model,
                    optimizer=optimizer,
                    global_step=step,
                    teacher_checkpoint=teacher_checkpoint,
                    teacher_manifest=teacher_manifest,
                    teacher_actor_state_dict=teacher.state_dict(),
                    student_env_contract=student_environment_contract(env_cfg),
                    loss=_loss_contract(),
                    optimizer_config=_optimizer_contract(),
                )
                checkpoint = run_dir / "checkpoints" / f"student_{step:07d}.pt"
                _atomic_torch_save(payload, checkpoint)
                _atomic_torch_save(payload, checkpoint.parent / "latest.pt")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
