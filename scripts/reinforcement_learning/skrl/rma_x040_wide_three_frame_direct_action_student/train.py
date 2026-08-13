"""Train the X040-Wide Size-Buckets three-frame direct-action Student."""

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
parser.add_argument(
    "--task",
    default=(
        "TacEx-Sim2Real-Cube-Real-Alignment-RMA-X040-Wide-Size-Buckets-"
        "Three-Frame-Direct-Action-Student-DR-v0"
    ),
)
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--encoder_init_checkpoint", required=True)
parser.add_argument("--resume", default=None)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--timesteps", type=int, default=100_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=3.0e-4)
parser.add_argument("--backbone_learning_rate", type=float, default=3.0e-5)
parser.add_argument("--weight_decay", type=float, default=1.0e-5)
parser.add_argument("--action_loss_weight", type=float, default=1.0)
parser.add_argument("--action_smoothness_loss_weight", type=float, default=0.05)
parser.add_argument("--position_loss_weight", type=float, default=1.0)
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
from tacex_tasks.sim2real_grasp.rma_x040_wide_models import (
    RMAX040WideActorCore,
    extract_x040_wide_actor_core_state_dict,
)
from tacex_tasks.sim2real_grasp.rma_x040_wide_three_frame_artifacts import (
    RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK,
    RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK,
    RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASKS,
    load_encoder_initialization_checkpoint,
    load_teacher_manifest,
    load_teacher_policy_state,
    load_three_frame_student_checkpoint,
    load_three_frame_student_model_state,
    make_three_frame_student_payload,
    sha256_file,
    three_frame_student_environment_contract,
    validate_live_teacher_contract,
)
from tacex_tasks.sim2real_grasp.rma_x040_wide_three_frame_models import (
    RMAX040WideThreeFrameDirectActionVisualStudent,
)


def _atomic_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    if args.task not in RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASKS:
        raise ValueError(
            "This trainer only supports X040-Wide Size-Buckets three-frame Student tasks: "
            f"{RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_STUDENT_TASKS}"
        )
    if min(
        args.num_envs,
        args.timesteps,
        args.log_interval,
        args.checkpoint_interval,
    ) <= 0:
        raise ValueError(
            "num_envs, timesteps, log_interval and checkpoint_interval must be positive"
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    teacher_checkpoint = Path(args.teacher_checkpoint).expanduser().resolve()
    encoder_checkpoint = Path(args.encoder_init_checkpoint).expanduser().resolve()
    teacher_manifest = load_teacher_manifest(
        teacher_checkpoint,
        expected_task=RMA_X040_WIDE_SIZE_BUCKETS_TEACHER_TASK,
    )
    encoder_state, encoder_payload = load_encoder_initialization_checkpoint(
        encoder_checkpoint
    )
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
    )
    env_cfg.seed = args.seed
    validate_live_teacher_contract(
        env_cfg,
        teacher_manifest,
        allow_static_size_bucket_assignment=bool(
            getattr(env_cfg, "cube_size_assignment", None)
        ),
    )

    default_log_directory = (
        "sim2real_cube_real_alignment_rma_x040_wide_size_buckets_"
        "three_frame_appearance_direct_action_student"
        if args.task
        == RMA_X040_WIDE_SIZE_BUCKETS_THREE_FRAME_APPEARANCE_DIRECT_STUDENT_DR_TASK
        else (
            "sim2real_cube_real_alignment_rma_x040_wide_size_buckets_"
            "three_frame_direct_action_student"
        )
    )
    run_dir = (
        Path(args.log_dir).expanduser().resolve()
        if args.log_dir
        else (Path("logs/skrl") / default_log_directory).resolve()
        / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_distillation"
    )
    _atomic_json(
        {
            **vars(args),
            "teacher_checkpoint_sha256": sha256_file(teacher_checkpoint),
            "encoder_init_checkpoint_sha256": sha256_file(encoder_checkpoint),
            "student_runtime_inputs": [
                "wrist_rgb_history",
                "proprio_obs",
                "action_history",
            ],
            "wrist_rgb_history_contract": {
                "shape": [3, 224, 224, 3],
                "order": "oldest_to_newest",
                "stride_policy_steps": 1,
                "reset_fill": "repeat_first_post_reset_frame",
            },
            "training_only_privileged_inputs": ["rma_cube_pos"],
            "position_target": "normalized_cube_position_root_xyz_at_newest_frame",
        },
        run_dir / "params" / "training.json",
    )

    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped
    device = torch.device(base_env.device)
    action_scale = torch.tensor(
        [env_cfg.action_scale] * 3 + [env_cfg.gripper_width_delta_scale],
        device=device,
    )

    teacher = RMAX040WideActorCore().to(device).eval()
    teacher.load_state_dict(
        extract_x040_wide_actor_core_state_dict(
            load_teacher_policy_state(teacher_checkpoint, device)
        ),
        strict=True,
    )
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = RMAX040WideThreeFrameDirectActionVisualStudent().to(device)
    student.load_vision_encoder_state(encoder_state)
    student.train()
    head_parameters = (
        list(student.temporal_fusion.parameters())
        + list(student.action_head.parameters())
        + list(student.position_head.parameters())
    )
    backbone_parameters = [
        value for value in student.vision_encoder.parameters() if value.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": args.learning_rate},
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    loss_contract = {
        "action_distillation": "mse_teacher_mean_action",
        "action_weight": args.action_loss_weight,
        "action_smoothness": (
            "mse_student_action_vs_previous_requested_normalized_action"
        ),
        "action_smoothness_weight": args.action_smoothness_loss_weight,
        "position": "mse_normalized_current_cube_position_root_xyz",
        "position_weight": args.position_loss_weight,
    }
    optimizer_config = {
        "class": "AdamW",
        "head_learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "backbone_trainable": "shared_resnet18_layer3_layer4",
        "temporal_fusion_trainable": "linear_1536_to_512",
        "batch_norm": "eval",
        "weight_decay": args.weight_decay,
        "grad_norm_clip": args.grad_norm_clip,
    }

    start_step = 0
    if args.resume:
        payload = load_three_frame_student_checkpoint(
            args.resume,
            device=device,
            expected_teacher_checkpoint=teacher_checkpoint,
            expected_task=args.task,
        )
        if (
            payload.get("encoder_init_checkpoint_sha256")
            != sha256_file(encoder_checkpoint)
            or payload.get("loss") != loss_contract
            or payload.get("optimizer_config") != optimizer_config
        ):
            raise RuntimeError("Resume provenance, loss, or optimizer differs")
        load_three_frame_student_model_state(student, payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["global_step"])

    observations, _ = env.reset()
    writer = SummaryWriter(str(run_dir))
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, args.timesteps + 1):
            obs = observations["policy"]
            proprio = obs["proprio_obs"].float()
            history = obs["action_history"].float()
            cube_position = obs["rma_cube_pos"].float()
            actions, normalized_position = student.forward_with_position(
                obs["wrist_rgb_history"],
                proprio,
                history,
            )
            with torch.no_grad():
                teacher_actions = teacher(proprio, history, cube_position)
                previous_actions = torch.clamp(history / action_scale, -1.0, 1.0)
                position_target = student.normalizer.normalize_position(cube_position)

            action_loss = F.mse_loss(actions, teacher_actions)
            smoothness_loss = F.mse_loss(actions, previous_actions)
            position_loss = F.mse_loss(normalized_position, position_target)
            loss = (
                args.action_loss_weight * action_loss
                + args.action_smoothness_loss_weight * smoothness_loss
                + args.position_loss_weight * position_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                head_parameters + backbone_parameters,
                args.grad_norm_clip,
            )
            optimizer.step()
            observations, rewards, _, _, _ = env.step(actions.detach())

            predicted_position = student.normalizer.denormalize_position(
                normalized_position.detach()
            )
            position_rmse_m = torch.sqrt(
                torch.mean((predicted_position - cube_position).square())
            )
            writer.add_scalar("Loss/total", loss.item(), step)
            writer.add_scalar("Loss/action_distillation", action_loss.item(), step)
            writer.add_scalar("Loss/action_smoothness", smoothness_loss.item(), step)
            writer.add_scalar("Loss/position_normalized", position_loss.item(), step)
            writer.add_scalar("Position/rmse_m", position_rmse_m.item(), step)
            writer.add_scalar(
                "Action/teacher_rmse",
                torch.sqrt(torch.mean((actions.detach() - teacher_actions).square())).item(),
                step,
            )
            writer.add_scalar("Optimization/grad_norm", float(grad_norm), step)
            writer.add_scalar("Reward/mean_step", rewards.mean().item(), step)

            if step % args.log_interval == 0 or step == args.timesteps:
                stats = base_env._episode_success_statistics()
                writer.add_scalar(
                    "Performance/recent_success_rate",
                    stats["window_rate"].item(),
                    step,
                )
                print(
                    f"[X040-Wide Three-Frame Student] update={step:,}/{args.timesteps:,} "
                    f"loss(action/smooth/pos)={action_loss.item():.5f}/"
                    f"{smoothness_loss.item():.5f}/{position_loss.item():.5f} "
                    f"pos_rmse_m={position_rmse_m.item():.5f} "
                    f"success={stats['cumulative_rate'].item():.3f} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )

            if step % args.checkpoint_interval == 0 or step == args.timesteps:
                payload = make_three_frame_student_payload(
                    model=student,
                    optimizer=optimizer,
                    global_step=step,
                    task=args.task,
                    teacher_checkpoint=teacher_checkpoint,
                    teacher_manifest=teacher_manifest,
                    encoder_init_checkpoint=encoder_checkpoint,
                    encoder_init_payload=encoder_payload,
                    student_env_contract=three_frame_student_environment_contract(
                        env_cfg
                    ),
                    loss=loss_contract,
                    optimizer_config=optimizer_config,
                )
                checkpoint = (
                    run_dir / "checkpoints" / f"student_{step:07d}.pt"
                )
                _atomic_save(payload, checkpoint)
                _atomic_save(payload, checkpoint.parent / "latest.pt")
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
