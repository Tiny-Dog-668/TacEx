"""Distill the fixed-size GelSight reference-delta visual Student."""

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
    default="TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Size-Buckets-Student-DR-v0",
)
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--resume", default=None)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--timesteps", type=int, default=100_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--backbone_learning_rate", type=float, default=3.0e-5)
parser.add_argument("--train_backbone_after_layer2", action="store_true")
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
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn.functional as F
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from torch.utils.tensorboard import SummaryWriter

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_size_buckets_artifacts import (
    GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK,
    STUDENT_CHECKPOINT_VERSION,
    STUDENT_MODEL_VERSION,
    load_student_checkpoint,
    load_student_model_state,
    load_teacher_manifest,
    load_teacher_policy_state,
    sha256_file,
    state_dict_sha256,
    validate_live_env_contract,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_size_buckets_models import (
    RMAGelSightReferenceStudent,
)
from tacex_tasks.sim2real_grasp.rma_models import (
    RMAActorCore,
    extract_actor_core_state_dict,
)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _loss_contract() -> dict:
    return {
        "position": "smooth_l1_normalized_xyz",
        "smooth_l1_beta": float(args.smooth_l1_beta),
        "position_weight": float(args.position_loss_weight),
        "contact": "left_right_bce_from_signed_current_minus_reference",
        "contact_positive_weight": float(args.contact_positive_weight),
        "contact_weight": float(args.contact_loss_weight),
        "actor_contact": "hard_binary_logit_greater_than_or_equal_to_zero",
        "action": "mse_teacher_deterministic_mean",
        "action_weight": float(args.action_loss_weight),
        "action_to_tactile_gradient": "none_by_hard_threshold",
        "action_smoothness": "mse_to_previous_environment_action",
        "action_smoothness_weight": float(args.action_smoothness_loss_weight),
    }


def _student_payload(
    model: RMAGelSightReferenceStudent,
    optimizer: torch.optim.Optimizer,
    step: int,
    teacher_checkpoint: Path,
    teacher_manifest: dict,
) -> dict:
    return {
        "kind": "tacex_rma_gelsight_size_buckets_student",
        "version": STUDENT_CHECKPOINT_VERSION,
        "model_version": STUDENT_MODEL_VERSION,
        "task": args.task,
        "global_step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "teacher_manifest": teacher_manifest,
        "normalization": model.actor_core.normalizer.contract(),
        "student_input_contract": {
            "input_order": [
                "wrist_rgb", "proprio_obs", "action_history",
                "gsmini_left_rgb", "gsmini_right_rgb",
                "gsmini_left_reference_rgb", "gsmini_right_reference_rgb",
            ],
            "wrist_rgb": [224, 224, 3],
            "tactile_rgb": [96, 128, 3],
            "tactile_delta": "signed_float32_current_minus_reference_div_255",
            "reference_capture": "first_post_reset_frame_per_environment",
            "contact_output": "left_right_hard_binary_logit_ge_0",
        },
        "vision_encoder_state_dict_sha256": state_dict_sha256(
            model.vision_encoder.state_dict()
        ),
        "tactile_contact_head_state_dict_sha256": state_dict_sha256(
            model.tactile_contact_head.state_dict()
        ),
        "teacher_actor_state_dict_sha256": state_dict_sha256(
            model.actor_core.state_dict()
        ),
        "loss": _loss_contract(),
        "optimizer_config": {
            "class": "AdamW",
            "learning_rate": float(args.learning_rate),
            "backbone_learning_rate": float(args.backbone_learning_rate),
            "train_backbone_after_layer2": bool(args.train_backbone_after_layer2),
            "weight_decay": float(args.weight_decay),
            "grad_norm_clip": float(args.grad_norm_clip),
        },
    }


def main() -> None:
    if args.task != GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK:
        raise RuntimeError(f"This trainer only supports {GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK}")
    if args.num_envs <= 0 or args.num_envs % 8:
        raise ValueError("num_envs must be a positive multiple of 8")
    if args.timesteps <= 0:
        raise ValueError("timesteps must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    teacher_checkpoint = Path(args.teacher_checkpoint).expanduser().resolve()
    teacher_manifest = load_teacher_manifest(teacher_checkpoint)
    resume_payload = None
    if args.resume:
        resume_payload = load_student_checkpoint(
            args.resume, device="cpu", expected_task=args.task
        )
        if resume_payload.get("teacher_checkpoint_sha256") != sha256_file(
            teacher_checkpoint
        ):
            raise RuntimeError("Resume Student was distilled from another Teacher")
        if resume_payload.get("loss") != _loss_contract():
            raise RuntimeError("Resume loss contract differs from current arguments")
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    if resume_payload is not None:
        # DirectRLEnv.common_step_counter restarts with a new process. Carry the
        # curriculum step explicitly so a resumed Student does not jump back
        # from its current threshold to 20 N.
        env_cfg.illegal_collision_curriculum_step_offset = int(
            resume_payload["global_step"]
        )
    validate_live_env_contract(env_cfg, teacher_manifest)
    env = gym.make(args.task, cfg=env_cfg)
    device = torch.device(env.unwrapped.device)

    actor = RMAActorCore().to(device)
    actor.load_state_dict(
        extract_actor_core_state_dict(
            load_teacher_policy_state(teacher_checkpoint, device)
        ),
        strict=True,
    )
    model = RMAGelSightReferenceStudent(actor, pretrained_backbone=True).to(device)
    trainable = list(model.adaptation_head.parameters()) + list(
        model.tactile_contact_head.parameters()
    )
    groups = [{"params": trainable, "lr": args.learning_rate}]
    if args.train_backbone_after_layer2:
        model.unfreeze_backbone_after_layer2()
        backbone = [p for p in model.vision_encoder.parameters() if p.requires_grad]
        trainable += backbone
        groups.append({"params": backbone, "lr": args.backbone_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    run_dir = (
        Path(args.log_dir).expanduser().resolve()
        if args.log_dir
        else Path("logs/skrl/sim2real_cube_real_alignment_rma_gelsight_size_buckets_student")
        / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_distillation")
    ).resolve()
    (run_dir / "params").mkdir(parents=True, exist_ok=True)
    (run_dir / "params" / "student_training.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )
    writer = SummaryWriter(str(run_dir))

    start_step = 0
    if resume_payload is not None:
        load_student_model_state(model, resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        start_step = int(resume_payload["global_step"])

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
            wrist = obs["wrist_rgb"]
            proprio = obs["proprio_obs"].float()
            history = obs["action_history"].float()
            cube_position = obs["rma_cube_pos"].float()
            contact_target = obs["rma_contact_state"].float()
            left = obs["gsmini_left_rgb"]
            right = obs["gsmini_right_rgb"]
            left_ref = obs["gsmini_left_reference_rgb"]
            right_ref = obs["gsmini_right_reference_rgb"]

            normalized_position, contact_logits = model.predict_reference_adaptation(
                wrist, left, right, left_ref, right_ref
            )
            student_actions = model(
                wrist, proprio, history, left, right, left_ref, right_ref
            )
            with torch.no_grad():
                teacher_actions = model.actor_core(
                    proprio, history, cube_position, contact_target
                )
                previous_actions = torch.clamp(history / action_scale, -1.0, 1.0)
            target_position = model.actor_core.normalizer.normalize_position(cube_position)
            position_loss = F.smooth_l1_loss(
                normalized_position, target_position, beta=args.smooth_l1_beta
            )
            contact_loss = F.binary_cross_entropy_with_logits(
                contact_logits, contact_target, pos_weight=positive_weight
            )
            action_loss = F.mse_loss(student_actions, teacher_actions)
            smoothness_loss = F.mse_loss(student_actions, previous_actions)
            loss = (
                args.position_loss_weight * position_loss
                + args.contact_loss_weight * contact_loss
                + args.action_loss_weight * action_loss
                + args.action_smoothness_loss_weight * smoothness_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_norm_clip)
            optimizer.step()
            observations, rewards, _, _, _ = env.step(student_actions.detach())

            predicted = contact_logits.detach() >= 0.0
            target = contact_target.bool()
            accuracy_lr = (predicted == target).float().mean(dim=0)
            bilateral_accuracy = (
                predicted.all(dim=-1) == target.all(dim=-1)
            ).float().mean()
            writer.add_scalar("Loss/total", loss.item(), step)
            writer.add_scalar("Loss/position", position_loss.item(), step)
            writer.add_scalar("Loss/contact_bce", contact_loss.item(), step)
            writer.add_scalar("Loss/action", action_loss.item(), step)
            writer.add_scalar("Loss/action_smoothness", smoothness_loss.item(), step)
            writer.add_scalar("Contact/left_accuracy", accuracy_lr[0].item(), step)
            writer.add_scalar("Contact/right_accuracy", accuracy_lr[1].item(), step)
            writer.add_scalar("Contact/bilateral_accuracy", bilateral_accuracy.item(), step)
            writer.add_scalar("Optimization/grad_norm", float(grad_norm), step)
            writer.add_scalar("Rollout/mean_reward", rewards.mean().item(), step)

            if step % args.log_interval == 0 or step == args.timesteps:
                elapsed = max(time.perf_counter() - started, 1.0e-6)
                print(
                    f"[GelSight Student] {step}/{args.timesteps} "
                    f"loss={loss.item():.4f} contact_acc_lr="
                    f"[{accuracy_lr[0].item():.3f},{accuracy_lr[1].item():.3f}] "
                    f"bilateral={bilateral_accuracy.item():.3f} "
                    f"updates_s={(step-start_step)/elapsed:.2f}",
                    flush=True,
                )
            if step % args.checkpoint_interval == 0 or step == args.timesteps:
                payload = _student_payload(
                    model, optimizer, step, teacher_checkpoint, teacher_manifest
                )
                _atomic_torch_save(payload, run_dir / "checkpoints" / f"student_{step}.pt")
                _atomic_torch_save(payload, run_dir / "checkpoints" / "latest.pt")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
