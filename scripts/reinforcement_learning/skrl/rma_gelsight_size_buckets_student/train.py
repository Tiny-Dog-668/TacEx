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
parser.add_argument(
    "--train_backbone_after_layer2",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--weight_decay", type=float, default=1.0e-5)
parser.add_argument("--position_loss_weight", type=float, default=1.0)
parser.add_argument("--heatmap_loss_weight", type=float, default=1.0)
parser.add_argument("--heatmap_sigma", type=float, default=1.5)
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
    student_input_contract,
    validate_live_env_contract,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_size_buckets_models import (
    RMAGelSightReferenceStudent,
)
from tacex_tasks.sim2real_grasp.rma_models import (
    RMAActorCore,
    extract_actor_core_state_dict,
    heatmap_soft_argmax,
    make_gaussian_heatmaps,
    project_points_root_to_image,
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
        "heatmap": "masked_mse_projected_cube_center_gaussian_14x14",
        "heatmap_weight": float(args.heatmap_loss_weight),
        "heatmap_sigma_px": float(args.heatmap_sigma),
        "heatmap_validity": "z_forward_and_uv_inside_224x224_student_rgb",
        "contact": "left_right_bce_from_signed_current_minus_reference",
        "contact_positive_weight": float(args.contact_positive_weight),
        "contact_weight": float(args.contact_loss_weight),
        "action": "mse_teacher_deterministic_mean",
        "action_weight": float(args.action_loss_weight),
        "action_gradient": "continuous_to_visual_and_both_tactile_encoders",
        "action_smoothness": "mse_to_previous_environment_action",
        "action_smoothness_weight": float(args.action_smoothness_loss_weight),
    }


def _optimizer_contract() -> dict:
    return {
        "class": "AdamW",
        "learning_rate": float(args.learning_rate),
        "backbone_learning_rate": float(args.backbone_learning_rate),
        "train_backbone_after_layer2": bool(args.train_backbone_after_layer2),
        "weight_decay": float(args.weight_decay),
        "grad_norm_clip": float(args.grad_norm_clip),
    }


def _student_payload(
    model: RMAGelSightReferenceStudent,
    optimizer: torch.optim.Optimizer,
    step: int,
    teacher_checkpoint: Path,
    teacher_manifest: dict,
    teacher_actor: RMAActorCore,
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
        "normalization": model.normalizer.contract(),
        "student_model_contract": model.contract(),
        "student_input_contract": student_input_contract(),
        "vision_encoder_state_dict_sha256": state_dict_sha256(
            model.vision_encoder.state_dict()
        ),
        "tactile_encoder_state_dict_sha256": state_dict_sha256(
            model.tactile_encoder.state_dict()
        ),
        "action_head_state_dict_sha256": state_dict_sha256(
            model.action_head.state_dict()
        ),
        "position_head_state_dict_sha256": state_dict_sha256(
            model.position_head.state_dict()
        ),
        "heatmap_head_state_dict_sha256": state_dict_sha256(
            model.heatmap_head.state_dict()
        ),
        "teacher_actor_state_dict_sha256": state_dict_sha256(
            teacher_actor.state_dict()
        ),
        "loss": _loss_contract(),
        "optimizer_config": _optimizer_contract(),
    }


def _camera_intrinsic_matrix(env_cfg, base_env, device: torch.device) -> torch.Tensor:
    """Return per-environment intrinsics for the final Student RGB frames."""
    intrinsic = torch.tensor(
        env_cfg.camera_model_intrinsic_matrix,
        device=device,
        dtype=torch.float32,
    ).reshape(3, 3)
    intrinsic = intrinsic.unsqueeze(0).repeat(base_env.num_envs, 1, 1)
    if hasattr(base_env, "_dr_focal_scale") and hasattr(
        base_env, "_dr_principal_shift_px"
    ):
        focal_scale = base_env._dr_focal_scale[: base_env.num_envs].to(
            device=device, dtype=torch.float32
        )
        principal_shift = base_env._dr_principal_shift_px[: base_env.num_envs].to(
            device=device, dtype=torch.float32
        )
        intrinsic[:, 0, 0] *= focal_scale
        intrinsic[:, 1, 1] *= focal_scale
        intrinsic[:, 0, 2] += principal_shift[:, 0]
        intrinsic[:, 1, 2] += principal_shift[:, 1]
    return intrinsic


def _camera_pose_root(
    env_cfg, base_env, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return optical camera pose in the robot-root-aligned environment frame."""
    if hasattr(base_env, "_dr_camera_pos_w") and hasattr(base_env, "_dr_camera_quat_w"):
        camera_position_root = (
            base_env._dr_camera_pos_w[: base_env.num_envs]
            - base_env.scene.env_origins[: base_env.num_envs]
        ).to(device=device, dtype=torch.float32)
        camera_quaternion_wxyz = base_env._dr_camera_quat_w[: base_env.num_envs].to(
            device=device, dtype=torch.float32
        )
        return camera_position_root, camera_quaternion_wxyz
    camera_position_root = torch.tensor(
        env_cfg.camera_base_position_m,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0).repeat(base_env.num_envs, 1)
    camera_quaternion_wxyz = torch.tensor(
        env_cfg.wrist_camera.offset.rot,
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0).repeat(base_env.num_envs, 1)
    return camera_position_root, camera_quaternion_wxyz


def _heatmap_targets(
    cube_position_root: torch.Tensor,
    env_cfg,
    base_env,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project cube centers and build Gaussian ``[N,1,14,14]`` targets."""
    device = cube_position_root.device
    camera_position_root, camera_quaternion_wxyz = _camera_pose_root(
        env_cfg, base_env, device
    )
    uv, valid = project_points_root_to_image(
        cube_position_root,
        camera_position_root,
        camera_quaternion_wxyz,
        _camera_intrinsic_matrix(env_cfg, base_env, device),
        image_width=int(env_cfg.wrist_camera.width),
        image_height=int(env_cfg.wrist_camera.height),
    )
    heatmap = make_gaussian_heatmaps(
        uv,
        valid,
        heatmap_height=14,
        heatmap_width=14,
        image_height=int(env_cfg.wrist_camera.height),
        image_width=int(env_cfg.wrist_camera.width),
        sigma=float(args.heatmap_sigma),
    )
    return heatmap, uv, valid


def _masked_heatmap_loss(
    predicted: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    if bool(valid.any().item()):
        return F.mse_loss(predicted[valid], target[valid])
    return predicted.sum() * 0.0


def main() -> None:
    if args.task != GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK:
        raise RuntimeError(f"This trainer only supports {GELSIGHT_SIZE_BUCKETS_STUDENT_DR_TASK}")
    if args.num_envs <= 0 or args.num_envs % 8:
        raise ValueError("num_envs must be a positive multiple of 8")
    if args.timesteps <= 0:
        raise ValueError("timesteps must be positive")
    if min(args.learning_rate, args.backbone_learning_rate, args.grad_norm_clip) <= 0:
        raise ValueError("learning rates and grad_norm_clip must be positive")
    if min(
        args.position_loss_weight,
        args.heatmap_loss_weight,
        args.contact_loss_weight,
        args.action_loss_weight,
        args.action_smoothness_loss_weight,
        args.weight_decay,
    ) < 0:
        raise ValueError("loss weights and weight_decay must be non-negative")
    if args.heatmap_sigma <= 0:
        raise ValueError("heatmap_sigma must be positive")
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
        if resume_payload.get("optimizer_config") != _optimizer_contract():
            raise RuntimeError("Resume optimizer contract differs from current arguments")
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

    teacher_actor = RMAActorCore().to(device)
    teacher_actor.load_state_dict(
        extract_actor_core_state_dict(
            load_teacher_policy_state(teacher_checkpoint, device)
        ),
        strict=True,
    )
    teacher_actor.eval()
    for parameter in teacher_actor.parameters():
        parameter.requires_grad_(False)
    model = RMAGelSightReferenceStudent(pretrained_backbone=True).to(device)
    head_parameters = (
        list(model.position_head.parameters())
        + list(model.heatmap_head.parameters())
        + list(model.tactile_encoder.parameters())
        + list(model.action_head.parameters())
    )
    trainable = list(head_parameters)
    groups = [{"params": head_parameters, "lr": args.learning_rate}]
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
        json.dumps(
            {
                **vars(args),
                "student_actor_feature_dim": 1043,
                "student_actor_feature_order": [
                    "visual_512",
                    "left_tactile_256",
                    "right_tactile_256",
                    "normalized_proprio_15",
                    "normalized_action_history_4",
                ],
                "training_only_privileged_labels": [
                    "rma_cube_pos",
                    "rma_contact_state",
                    "projected_cube_center_heatmap",
                ],
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    writer = SummaryWriter(str(run_dir))
    base_env = env.unwrapped

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
            wrist = obs["wrist_rgb"]
            proprio = obs["proprio_obs"].float()
            history = obs["action_history"].float()
            cube_position = obs["rma_cube_pos"].float()
            contact_target = obs["rma_contact_state"].float()
            left = obs["gsmini_left_rgb"]
            right = obs["gsmini_right_rgb"]
            left_ref = obs["gsmini_left_reference_rgb"]
            right_ref = obs["gsmini_right_reference_rgb"]

            (
                student_actions,
                normalized_position,
                contact_logits,
                predicted_heatmap,
            ) = model.forward_with_auxiliary(
                wrist,
                proprio,
                history,
                left,
                right,
                left_ref,
                right_ref,
            )
            with torch.no_grad():
                teacher_actions = teacher_actor(
                    proprio, history, cube_position, contact_target
                )
                previous_actions = torch.clamp(history / action_scale, -1.0, 1.0)
                target_heatmap, target_uv, heatmap_valid = _heatmap_targets(
                    cube_position, env_cfg, base_env
                )
            target_position = model.normalizer.normalize_position(cube_position)
            position_loss = F.smooth_l1_loss(
                normalized_position, target_position, beta=args.smooth_l1_beta
            )
            heatmap_loss = _masked_heatmap_loss(
                predicted_heatmap, target_heatmap, heatmap_valid
            )
            contact_loss = F.binary_cross_entropy_with_logits(
                contact_logits, contact_target, pos_weight=positive_weight
            )
            action_loss = F.mse_loss(student_actions, teacher_actions)
            smoothness_loss = F.mse_loss(student_actions, previous_actions)
            loss = (
                args.position_loss_weight * position_loss
                + args.heatmap_loss_weight * heatmap_loss
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
            predicted_uv = heatmap_soft_argmax(
                predicted_heatmap.detach(),
                image_height=int(env_cfg.wrist_camera.height),
                image_width=int(env_cfg.wrist_camera.width),
            )
            if bool(heatmap_valid.any().item()):
                heatmap_center_error_px = torch.linalg.vector_norm(
                    predicted_uv[heatmap_valid] - target_uv[heatmap_valid], dim=-1
                ).mean()
            else:
                heatmap_center_error_px = torch.zeros((), device=device)
            position_error = model.normalizer.denormalize_position(
                normalized_position.detach()
            ) - cube_position
            position_rmse_m = torch.sqrt(torch.mean(position_error.square()))
            action_rmse = torch.sqrt(
                torch.mean((student_actions.detach() - teacher_actions).square())
            )
            writer.add_scalar("Loss/total", loss.item(), step)
            writer.add_scalar("Loss/position", position_loss.item(), step)
            writer.add_scalar("Loss/heatmap", heatmap_loss.item(), step)
            writer.add_scalar("Loss/contact_bce", contact_loss.item(), step)
            writer.add_scalar("Loss/action", action_loss.item(), step)
            writer.add_scalar("Loss/action_smoothness", smoothness_loss.item(), step)
            writer.add_scalar("Position/rmse_m", position_rmse_m.item(), step)
            writer.add_scalar(
                "Heatmap/center_error_px", heatmap_center_error_px.item(), step
            )
            writer.add_scalar(
                "Heatmap/valid_fraction", heatmap_valid.float().mean().item(), step
            )
            writer.add_scalar("Action/teacher_rmse", action_rmse.item(), step)
            writer.add_scalar("Contact/left_accuracy", accuracy_lr[0].item(), step)
            writer.add_scalar("Contact/right_accuracy", accuracy_lr[1].item(), step)
            writer.add_scalar("Contact/bilateral_accuracy", bilateral_accuracy.item(), step)
            writer.add_scalar("Optimization/grad_norm", float(grad_norm), step)
            writer.add_scalar("Rollout/mean_reward", rewards.mean().item(), step)

            if step % args.log_interval == 0 or step == args.timesteps:
                elapsed = max(time.perf_counter() - started, 1.0e-6)
                print(
                    f"[GelSight Student] {step}/{args.timesteps} "
                    f"loss(total/pos/heat/contact/action)={loss.item():.4f}/"
                    f"{position_loss.item():.4f}/{heatmap_loss.item():.4f}/"
                    f"{contact_loss.item():.4f}/{action_loss.item():.4f} "
                    f"position_rmse={1000.0 * position_rmse_m.item():.1f}mm "
                    f"heatmap_error={heatmap_center_error_px.item():.1f}px "
                    f"contact_acc_lr="
                    f"[{accuracy_lr[0].item():.3f},{accuracy_lr[1].item():.3f}] "
                    f"bilateral={bilateral_accuracy.item():.3f} "
                    f"updates_s={(step-start_step)/elapsed:.2f}",
                    flush=True,
                )
            if step % args.checkpoint_interval == 0 or step == args.timesteps:
                payload = _student_payload(
                    model,
                    optimizer,
                    step,
                    teacher_checkpoint,
                    teacher_manifest,
                    teacher_actor,
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
