"""Online visual position/contact and action distillation for the RMA student."""

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
    repo_root = Path(__file__).resolve().parents[3]
    for package_root in (
        repo_root / "source" / "tacex_tasks",
        repo_root / "source" / "tacex",
        repo_root / "source" / "tacex_assets",
        repo_root / "source" / "tacex_uipc",
    ):
        value = str(package_root)
        if package_root.is_dir() and value not in sys.path:
            sys.path.insert(0, value)


_extend_repo_pythonpath()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-v0",
)
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--resume", default=None)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--timesteps", type=int, default=100_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--weight_decay", type=float, default=1.0e-5)
parser.add_argument("--position_loss_weight", type=float, default=1.0)
parser.add_argument("--contact_loss_weight", type=float, default=1.0)
parser.add_argument("--contact_positive_weight", type=float, default=5.0)
parser.add_argument("--action_loss_weight", type=float, default=1.0)
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
from tacex_tasks.sim2real_grasp.rma_artifacts import (
    RMA_STUDENT_CHECKPOINT_VERSION,
    RMA_STUDENT_TASKS,
    load_student_checkpoint,
    load_teacher_manifest,
    load_teacher_policy_state,
    sha256_file,
    state_dict_sha256,
    validate_live_env_contract,
)
from tacex_tasks.sim2real_grasp.rma_models import (
    RMA_MODEL_VERSION,
    RMAActorCore,
    RMAVisualStudent,
    extract_actor_core_state_dict,
)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _format_duration(seconds: float) -> str:
    """Format a non-negative wall-clock duration for compact terminal progress."""
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes:d}m{seconds:02d}s"


def _loss_contract() -> dict:
    return {
        "position": "smooth_l1_normalized_xyz",
        "smooth_l1_beta": float(args.smooth_l1_beta),
        "position_weight": float(args.position_loss_weight),
        "contact": "binary_cross_entropy_with_logits_left_right",
        "contact_weight": float(args.contact_loss_weight),
        "contact_positive_weight": float(args.contact_positive_weight),
        "action": "mse_deterministic_tanh_mean",
        "action_weight": float(args.action_loss_weight),
    }


def _student_payload(
    model: RMAVisualStudent,
    optimizer: torch.optim.Optimizer,
    step: int,
    teacher_checkpoint: Path,
    teacher_manifest: dict,
) -> dict:
    return {
        "kind": "tacex_rma_student",
        "version": RMA_STUDENT_CHECKPOINT_VERSION,
        "task": args.task,
        "model_version": RMA_MODEL_VERSION,
        "global_step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
        "teacher_manifest": teacher_manifest,
        "normalization": model.actor_core.normalizer.contract(),
        "vision_encoder_state_dict_sha256": state_dict_sha256(model.vision_encoder.state_dict()),
        "teacher_actor_state_dict_sha256": state_dict_sha256(model.actor_core.state_dict()),
        "loss": _loss_contract(),
        "optimizer_config": {
            "class": "AdamW",
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "grad_norm_clip": float(args.grad_norm_clip),
        },
    }


def _prepare_model(device: torch.device, teacher_checkpoint: Path) -> RMAVisualStudent:
    policy_state = load_teacher_policy_state(teacher_checkpoint, device)
    actor_core = RMAActorCore().to(device)
    actor_core.load_state_dict(extract_actor_core_state_dict(policy_state), strict=True)
    model = RMAVisualStudent(actor_core, pretrained_backbone=True).to(device)
    model.train()
    return model


def main() -> None:
    if args.task not in RMA_STUDENT_TASKS:
        supported = ", ".join(sorted(RMA_STUDENT_TASKS))
        raise ValueError(f"train_rma_student.py only supports: {supported}")
    if args.timesteps <= 0 or args.num_envs <= 0:
        raise ValueError("timesteps and num_envs must be positive")
    if args.contact_positive_weight <= 0:
        raise ValueError("contact_positive_weight must be positive")
    if min(args.position_loss_weight, args.contact_loss_weight, args.action_loss_weight) < 0:
        raise ValueError("loss weights must be non-negative")

    teacher_checkpoint = Path(args.teacher_checkpoint).expanduser().resolve()
    teacher_manifest = load_teacher_manifest(teacher_checkpoint)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.cube_position_curriculum_force_full_range = True
    validate_live_env_contract(env_cfg, teacher_manifest)
    env = gym.make(args.task, cfg=env_cfg)
    device = torch.device(env.unwrapped.device)

    run_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else Path(
        "logs/skrl/sim2real_cube_real_alignment_rma_student"
    ).resolve() / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_distillation")
    checkpoints_dir = run_dir / "checkpoints"
    params_dir = run_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    (params_dir / "student_training.json").write_text(
        json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8"
    )
    writer = SummaryWriter(str(run_dir))

    model = _prepare_model(device, teacher_checkpoint)
    trainable_parameters = list(model.adaptation_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    if args.resume:
        resume_payload = load_student_checkpoint(
            args.resume,
            device=device,
            expected_teacher_checkpoint=teacher_checkpoint,
            expected_task=args.task,
        )
        if resume_payload.get("loss") != _loss_contract():
            raise RuntimeError("Resume checkpoint loss configuration differs from CLI arguments")
        model.load_state_dict(resume_payload["model"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer"])
        start_step = int(resume_payload["global_step"])
        if start_step >= args.timesteps:
            raise RuntimeError(f"Resume step {start_step} already reached timesteps={args.timesteps}")

    observations, _ = env.reset()
    remaining_updates = args.timesteps - start_step
    remaining_transitions = remaining_updates * args.num_envs
    print(
        "[RMA Student] start | "
        f"updates={start_step + 1}-{args.timesteps}/{args.timesteps} | "
        f"envs={args.num_envs} | remaining_transitions={remaining_transitions:,} | "
        f"log_every={args.log_interval} updates | checkpoint_every={args.checkpoint_interval} updates",
        flush=True,
    )
    training_start_time = time.perf_counter()
    previous_log_time = training_start_time
    previous_log_step = start_step

    try:
        for step in range(start_step + 1, args.timesteps + 1):
            obs = observations["policy"]
            wrist_rgb = obs["wrist_rgb"]
            proprio = obs["proprio_obs"].to(torch.float32)
            history = obs["action_history"].to(torch.float32)
            cube_position = obs["rma_cube_pos"].to(torch.float32)
            contact_state = obs["rma_contact_state"].to(torch.float32)

            predicted_normalized, predicted_contact_logits = model.predict_adaptation(wrist_rgb)
            predicted_contact = torch.sigmoid(predicted_contact_logits)
            target_normalized = model.actor_core.normalizer.normalize_position(cube_position)
            student_actions = model.action_from_normalized_position(
                proprio, history, predicted_normalized, predicted_contact
            )
            with torch.no_grad():
                teacher_actions = model.actor_core(
                    proprio, history, cube_position, contact_state
                )

            position_loss = F.smooth_l1_loss(
                predicted_normalized,
                target_normalized,
                beta=args.smooth_l1_beta,
            )
            positive_weight = torch.full(
                (contact_state.shape[-1],),
                float(args.contact_positive_weight),
                device=device,
            )
            contact_loss = F.binary_cross_entropy_with_logits(
                predicted_contact_logits,
                contact_state,
                pos_weight=positive_weight,
            )
            action_loss = F.mse_loss(student_actions, teacher_actions)
            loss = (
                args.position_loss_weight * position_loss
                + args.contact_loss_weight * contact_loss
                + args.action_loss_weight * action_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_norm_clip)
            optimizer.step()

            observations, rewards, terminated, truncated, _ = env.step(student_actions.detach())

            position_error = model.actor_core.normalizer.denormalize_position(
                predicted_normalized.detach()
            ) - cube_position
            rmse_xyz = torch.sqrt(torch.mean(position_error.square(), dim=0))
            rmse_3d = torch.sqrt(torch.mean(torch.sum(position_error.square(), dim=-1)))
            predicted_contact_binary = predicted_contact.detach() >= 0.5
            contact_accuracy = (
                predicted_contact_binary == contact_state.to(torch.bool)
            ).to(torch.float32).mean()

            writer.add_scalar("Loss/total", loss.item(), step)
            writer.add_scalar("Loss/position", position_loss.item(), step)
            writer.add_scalar("Loss/contact_bce", contact_loss.item(), step)
            writer.add_scalar("Loss/action_distillation", action_loss.item(), step)
            writer.add_scalar("Position/rmse_3d_m", rmse_3d.item(), step)
            writer.add_scalar("Position/rmse_x_m", rmse_xyz[0].item(), step)
            writer.add_scalar("Position/rmse_y_m", rmse_xyz[1].item(), step)
            writer.add_scalar("Position/rmse_z_m", rmse_xyz[2].item(), step)
            writer.add_scalar("Optimization/grad_norm", float(grad_norm), step)
            writer.add_scalar("Contact/accuracy", contact_accuracy.item(), step)
            writer.add_scalar("Contact/true_positive_fraction", contact_state.mean().item(), step)
            writer.add_scalar(
                "Contact/predicted_positive_fraction",
                predicted_contact_binary.to(torch.float32).mean().item(),
                step,
            )

            if step % args.log_interval == 0 or step == args.timesteps:
                now = time.perf_counter()
                interval_updates = step - previous_log_step
                interval_seconds = max(now - previous_log_time, 1e-9)
                updates_per_second = interval_updates / interval_seconds
                transitions_per_second = updates_per_second * args.num_envs
                elapsed_seconds = now - training_start_time
                mean_updates_per_second = (step - start_step) / max(elapsed_seconds, 1e-9)
                eta_seconds = (args.timesteps - step) / max(mean_updates_per_second, 1e-9)
                success_metrics = env.unwrapped._episode_success_statistics()
                completed_episodes = int(success_metrics["completed_count"].item())
                successful_episodes = int(success_metrics["success_count"].item())
                cumulative_success_rate = float(success_metrics["cumulative_rate"].item())
                recent_success_rate = float(success_metrics["window_rate"].item())
                writer.add_scalar("Performance/updates_per_second", updates_per_second, step)
                writer.add_scalar("Performance/transitions_per_second", transitions_per_second, step)
                writer.add_scalar("Performance/cumulative_success_rate", cumulative_success_rate, step)
                writer.add_scalar("Performance/recent_success_rate", recent_success_rate, step)
                print(
                    f"[RMA Student] progress={100.0 * step / args.timesteps:6.2f}% "
                    f"| update={step:,}/{args.timesteps:,} "
                    f"| transitions={step * args.num_envs:,} "
                    f"| speed={updates_per_second:.2f} updates/s ({transitions_per_second:.2f} samples/s) "
                    f"| elapsed={_format_duration(elapsed_seconds)} "
                    f"| eta={_format_duration(eta_seconds)}\n"
                    f"  loss(total/pos/contact/action)={loss.item():.5f}/{position_loss.item():.5f}/"
                    f"{contact_loss.item():.5f}/{action_loss.item():.5f} "
                    f"| position_rmse={1000.0 * rmse_3d.item():.2f} mm "
                    f"| contact_acc={contact_accuracy.item():.3f} "
                    f"| success(cumulative/recent)={cumulative_success_rate:.3f}/{recent_success_rate:.3f} "
                    f"({successful_episodes}/{completed_episodes} completed)",
                    flush=True,
                )
                previous_log_time = now
                previous_log_step = step

            if step % args.checkpoint_interval == 0 or step == args.timesteps:
                payload = _student_payload(
                    model, optimizer, step, teacher_checkpoint, teacher_manifest
                )
                output = checkpoints_dir / f"student_{step:07d}.pt"
                _atomic_torch_save(payload, output)
                _atomic_torch_save(payload, checkpoints_dir / "latest.pt")
                print(f"[RMA Student] checkpoint saved: {output}", flush=True)
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
