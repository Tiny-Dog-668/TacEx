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
parser.add_argument(
    "--backbone_learning_rate",
    type=float,
    default=3.0e-5,
    help="Learning rate for ResNet18 layer3/layer4 when --train_backbone_after_layer2 is set.",
)
parser.add_argument(
    "--train_backbone_after_layer2",
    action="store_true",
    default=False,
    help="Unfreeze ResNet18 layer3/layer4; keep layer2 and earlier frozen.",
)
parser.add_argument("--weight_decay", type=float, default=1.0e-5)
parser.add_argument("--position_loss_weight", type=float, default=1.0)
parser.add_argument("--contact_loss_weight", type=float, default=1.0)
parser.add_argument("--contact_positive_weight", type=float, default=5.0)
parser.add_argument("--action_loss_weight", type=float, default=1.0)
parser.add_argument("--action_smoothness_loss_weight", type=float, default=0.05)
parser.add_argument(
    "--heatmap_loss_weight",
    type=float,
    default=None,
    help="Cube-center heatmap MSE weight. Defaults to the selected env config.",
)
parser.add_argument(
    "--heatmap_sigma",
    type=float,
    default=None,
    help="GT heatmap Gaussian sigma in 14x14 heatmap pixels. Defaults to env config.",
)
parser.add_argument(
    "--heatmap_debug_interval",
    type=int,
    default=0,
    help="Save heatmap center overlay PNGs every N updates. 0 disables debug images.",
)
parser.add_argument("--heatmap_debug_count", type=int, default=4)
parser.add_argument("--smooth_l1_beta", type=float, default=0.1)
parser.add_argument("--grad_norm_clip", type=float, default=1.0)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--checkpoint_interval", type=int, default=10_000)
parser.add_argument("--log_dir", default=None)
parser.add_argument(
    "--save_start_frame",
    action="store_true",
    default=False,
    help="Save initial Student wrist_rgb frames after the training reset.",
)
parser.add_argument(
    "--start_frame_count",
    type=int,
    default=5,
    help="Number of initial environment frames to save when --save_start_frame is enabled.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn.functional as F
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_grasp.rma_artifacts import (
    RMA_STUDENT_CHECKPOINT_VERSION,
    RMA_STUDENT_TASKS,
    load_student_checkpoint,
    load_student_model_state,
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
    heatmap_soft_argmax,
    make_gaussian_heatmaps,
    project_points_root_to_image,
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


def _save_initial_student_frames(observations: dict, output_dir: Path, frame_count: int) -> int:
    """Save the exact uint8 RGB observations delivered to the Student at reset.

    Each frame comes from a different reset environment.  For the DR task this
    preserves the independently sampled camera and visual perturbation for that
    environment, without taking extra actions or changing training state.
    """
    wrist_rgb = observations["policy"].get("wrist_rgb")
    if wrist_rgb is None:
        raise RuntimeError("Student reset observations do not contain wrist_rgb")
    if wrist_rgb.ndim != 4 or wrist_rgb.shape[-1] < 3:
        raise RuntimeError(
            "Expected Student wrist_rgb shaped [N,H,W,C], got "
            f"{tuple(wrist_rgb.shape)}"
        )

    count = min(max(1, int(frame_count)), int(wrist_rgb.shape[0]))
    output_dir.mkdir(parents=True, exist_ok=True)
    for env_id in range(count):
        image = wrist_rgb[env_id, :, :, :3].detach().to(device="cpu")
        if image.dtype != torch.uint8:
            image = image.to(torch.float32).mul(255.0).round().clamp(0, 255).to(torch.uint8)
        height, width = int(image.shape[0]), int(image.shape[1])
        Image.fromarray(image.contiguous().numpy()).save(
            output_dir / f"start_wrist_rgb_env{env_id:03d}_{width}x{height}.png"
        )
    return count


def _loss_contract(heatmap_loss_weight: float, heatmap_sigma: float) -> dict:
    return {
        "position": "smooth_l1_normalized_xyz",
        "smooth_l1_beta": float(args.smooth_l1_beta),
        "position_weight": float(args.position_loss_weight),
        "contact": "binary_cross_entropy_with_logits_left_right",
        "contact_weight": float(args.contact_loss_weight),
        "contact_positive_weight": float(args.contact_positive_weight),
        "action": "mse_deterministic_tanh_mean",
        "action_weight": float(args.action_loss_weight),
        "action_smoothness": "mse_student_action_to_previous_environment_action",
        "action_smoothness_weight": float(args.action_smoothness_loss_weight),
        "action_smoothness_scales": "environment_action_scales",
        "heatmap": "mse_projected_cube_center_gaussian_14x14",
        "heatmap_weight": float(heatmap_loss_weight),
        "heatmap_sigma_px": float(heatmap_sigma),
        "heatmap_validity": "z_forward_and_uv_inside_224x224_student_rgb",
    }


def _loss_contract_is_resume_compatible(saved: dict | None, current: dict) -> bool:
    if saved == current:
        return True
    if not isinstance(saved, dict):
        return False
    legacy_current = {
        key: value
        for key, value in current.items()
        if not key.startswith("heatmap")
    }
    return saved == legacy_current


def _student_payload(
    model: RMAVisualStudent,
    optimizer: torch.optim.Optimizer,
    step: int,
    teacher_checkpoint: Path,
    teacher_manifest: dict,
    loss_contract: dict,
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
        "loss": loss_contract,
        "optimizer_config": {
            "class": "AdamW",
            "learning_rate": float(args.learning_rate),
            "backbone_learning_rate": float(args.backbone_learning_rate),
            "train_backbone_after_layer2": bool(args.train_backbone_after_layer2),
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


def _camera_intrinsic_matrix(env_cfg, base_env, device: torch.device) -> torch.Tensor:
    intrinsic = torch.tensor(
        env_cfg.camera_model_intrinsic_matrix,
        device=device,
        dtype=torch.float32,
    ).reshape(3, 3)
    intrinsic = intrinsic.unsqueeze(0).repeat(base_env.num_envs, 1, 1)
    if hasattr(base_env, "_dr_focal_scale") and hasattr(base_env, "_dr_principal_shift_px"):
        focal_scale = base_env._dr_focal_scale[: base_env.num_envs].to(device=device, dtype=torch.float32)
        principal_shift = base_env._dr_principal_shift_px[: base_env.num_envs].to(device=device, dtype=torch.float32)
        intrinsic[:, 0, 0] *= focal_scale
        intrinsic[:, 1, 1] *= focal_scale
        intrinsic[:, 0, 2] += principal_shift[:, 0]
        intrinsic[:, 1, 2] += principal_shift[:, 1]
    return intrinsic


def _camera_pose_root(env_cfg, base_env, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(base_env, "_dr_camera_pos_w") and hasattr(base_env, "_dr_camera_quat_w"):
        camera_position_root = (
            base_env._dr_camera_pos_w[: base_env.num_envs]
            - base_env.scene.env_origins[: base_env.num_envs]
        ).to(device=device, dtype=torch.float32)
        camera_quaternion_wxyz = base_env._dr_camera_quat_w[: base_env.num_envs].to(
            device=device,
            dtype=torch.float32,
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
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = cube_position_root.device
    camera_position_root, camera_quaternion_wxyz = _camera_pose_root(env_cfg, base_env, device)
    intrinsic = _camera_intrinsic_matrix(env_cfg, base_env, device)
    uv, valid = project_points_root_to_image(
        cube_position_root,
        camera_position_root,
        camera_quaternion_wxyz,
        intrinsic,
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
        sigma=sigma,
    )
    return heatmap, uv, valid


def _masked_heatmap_loss(
    predicted_heatmap: torch.Tensor,
    target_heatmap: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if bool(valid.any().item()):
        return F.mse_loss(predicted_heatmap[valid], target_heatmap[valid])
    return predicted_heatmap.sum() * 0.0


def _save_heatmap_debug_frames(
    wrist_rgb: torch.Tensor,
    target_uv: torch.Tensor,
    predicted_uv: torch.Tensor,
    valid: torch.Tensor,
    output_dir: Path,
    step: int,
    count: int,
) -> int:
    valid_ids = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_ids.numel() == 0:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for env_id in valid_ids[: max(0, int(count))].detach().cpu().tolist():
        image = wrist_rgb[env_id, :, :, :3].detach().to(device="cpu")
        if image.dtype != torch.uint8:
            image = image.to(torch.float32).mul(255.0).round().clamp(0, 255).to(torch.uint8)
        pil_image = Image.fromarray(image.contiguous().numpy()).convert("RGB")
        draw = ImageDraw.Draw(pil_image)
        gt = target_uv[env_id].detach().cpu().tolist()
        pred = predicted_uv[env_id].detach().cpu().tolist()
        radius = 4
        draw.ellipse((gt[0] - radius, gt[1] - radius, gt[0] + radius, gt[1] + radius), outline=(0, 255, 0), width=2)
        draw.line((gt[0] - radius, gt[1], gt[0] + radius, gt[1]), fill=(0, 255, 0), width=1)
        draw.line((gt[0], gt[1] - radius, gt[0], gt[1] + radius), fill=(0, 255, 0), width=1)
        draw.ellipse((pred[0] - radius, pred[1] - radius, pred[0] + radius, pred[1] + radius), outline=(255, 0, 0), width=2)
        draw.line((pred[0] - radius, pred[1], pred[0] + radius, pred[1]), fill=(255, 0, 0), width=1)
        draw.line((pred[0], pred[1] - radius, pred[0], pred[1] + radius), fill=(255, 0, 0), width=1)
        pil_image.save(output_dir / f"heatmap_step{step:07d}_env{env_id:03d}.png")
        saved += 1
    return saved


def main() -> None:
    if args.task not in RMA_STUDENT_TASKS:
        supported = ", ".join(sorted(RMA_STUDENT_TASKS))
        raise ValueError(f"train_rma_student.py only supports: {supported}")
    if args.timesteps <= 0 or args.num_envs <= 0:
        raise ValueError("timesteps and num_envs must be positive")
    if args.contact_positive_weight <= 0:
        raise ValueError("contact_positive_weight must be positive")
    if min(
        args.position_loss_weight,
        args.contact_loss_weight,
        args.action_loss_weight,
        args.action_smoothness_loss_weight,
    ) < 0:
        raise ValueError("loss weights must be non-negative")
    if args.heatmap_loss_weight is not None and args.heatmap_loss_weight < 0:
        raise ValueError("heatmap_loss_weight must be non-negative")
    if args.heatmap_sigma is not None and args.heatmap_sigma <= 0:
        raise ValueError("heatmap_sigma must be positive")
    if args.heatmap_debug_interval < 0 or args.heatmap_debug_count < 0:
        raise ValueError("heatmap debug arguments must be non-negative")
    if args.train_backbone_after_layer2 and args.backbone_learning_rate <= 0:
        raise ValueError("backbone_learning_rate must be positive when backbone training is enabled")

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
    heatmap_supervision_enabled = bool(
        getattr(env_cfg, "rma_heatmap_supervision_enabled", False)
    )
    heatmap_loss_weight = (
        float(args.heatmap_loss_weight)
        if args.heatmap_loss_weight is not None
        else float(getattr(env_cfg, "rma_heatmap_loss_weight", 0.0))
    )
    if not heatmap_supervision_enabled and args.heatmap_loss_weight is None:
        heatmap_loss_weight = 0.0
    heatmap_sigma = (
        float(args.heatmap_sigma)
        if args.heatmap_sigma is not None
        else float(getattr(env_cfg, "rma_heatmap_sigma_px", 1.5))
    )
    if heatmap_sigma <= 0:
        raise ValueError("effective heatmap sigma must be positive")
    heatmap_loss_enabled = heatmap_loss_weight > 0.0
    loss_contract = _loss_contract(heatmap_loss_weight, heatmap_sigma)
    action_smoothness_scale = torch.tensor(
        [
            float(env_cfg.action_scale),
            float(env_cfg.action_scale),
            float(env_cfg.action_scale),
            float(env_cfg.gripper_width_delta_scale),
        ],
        device=device,
        dtype=torch.float32,
    )

    run_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else Path(
        "logs/skrl/sim2real_cube_real_alignment_rma_student"
    ).resolve() / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_distillation")
    checkpoints_dir = run_dir / "checkpoints"
    params_dir = run_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    training_config = vars(args).copy()
    training_config.update(
        {
            "effective_heatmap_supervision_enabled": heatmap_supervision_enabled,
            "effective_heatmap_loss_weight": heatmap_loss_weight,
            "effective_heatmap_sigma_px": heatmap_sigma,
            "effective_train_backbone": (
                "resnet18_layer3_layer4"
                if args.train_backbone_after_layer2
                else "none"
            ),
        }
    )
    (params_dir / "student_training.json").write_text(
        json.dumps(training_config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    writer = SummaryWriter(str(run_dir))

    model = _prepare_model(device, teacher_checkpoint)
    head_parameters = list(model.adaptation_head.parameters())
    if heatmap_loss_enabled:
        head_parameters += list(model.heatmap_head.parameters())
    parameter_groups = [
        {
            "params": head_parameters,
            "lr": float(args.learning_rate),
        }
    ]
    trainable_parameters = list(head_parameters)
    backbone_parameters = []
    if args.train_backbone_after_layer2:
        model.unfreeze_backbone_after_layer2()
        backbone_parameters = [
            parameter for parameter in model.vision_encoder.parameters()
            if parameter.requires_grad
        ]
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": float(args.backbone_learning_rate),
            }
        )
        trainable_parameters += backbone_parameters
    optimizer = torch.optim.AdamW(
        parameter_groups,
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
        if not _loss_contract_is_resume_compatible(resume_payload.get("loss"), loss_contract):
            raise RuntimeError("Resume checkpoint loss configuration differs from CLI arguments")
        exact_resume_contract = resume_payload.get("loss") == loss_contract
        load_student_model_state(model, resume_payload["model"])
        expected_optimizer_config = {
            "class": "AdamW",
            "learning_rate": float(args.learning_rate),
            "backbone_learning_rate": float(args.backbone_learning_rate),
            "train_backbone_after_layer2": bool(args.train_backbone_after_layer2),
            "weight_decay": float(args.weight_decay),
            "grad_norm_clip": float(args.grad_norm_clip),
        }
        exact_optimizer_config = resume_payload.get("optimizer_config") == expected_optimizer_config
        if exact_resume_contract and exact_optimizer_config:
            optimizer.load_state_dict(resume_payload["optimizer"])
        else:
            print(
                "[WARN] Resume checkpoint predates/differs in heatmap or optimizer contract; "
                "model weights were loaded and optimizer state was reinitialized.",
                flush=True,
            )
        start_step = int(resume_payload["global_step"])
        if start_step >= args.timesteps:
            raise RuntimeError(f"Resume step {start_step} already reached timesteps={args.timesteps}")

    observations, _ = env.reset()
    if args.save_start_frame:
        output_dir = run_dir / "camera_frames"
        saved_count = _save_initial_student_frames(observations, output_dir, args.start_frame_count)
        print(
            f"[INFO] Saved {saved_count} Student input frame(s) to: {output_dir}",
            flush=True,
        )
    remaining_updates = args.timesteps - start_step
    remaining_transitions = remaining_updates * args.num_envs
    print(
        "[RMA Student] start | "
        f"updates={start_step + 1}-{args.timesteps}/{args.timesteps} | "
        f"envs={args.num_envs} | remaining_transitions={remaining_transitions:,} | "
        f"log_every={args.log_interval} updates | checkpoint_every={args.checkpoint_interval} updates | "
        f"heatmap_weight={heatmap_loss_weight:g} | heatmap_sigma={heatmap_sigma:g}px | "
        f"backbone_trainable={'layer3+layer4' if args.train_backbone_after_layer2 else 'none'}",
        flush=True,
    )
    if args.train_backbone_after_layer2:
        print(
            "[RMA Student] ResNet18 trainable: layer3/layer4; frozen: conv1/bn1/layer1/layer2; "
            f"backbone_lr={args.backbone_learning_rate:g}, head_lr={args.learning_rate:g}",
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

            if heatmap_loss_enabled:
                (
                    predicted_normalized,
                    predicted_contact_logits,
                    predicted_heatmap,
                ) = model.predict_adaptation_and_heatmap(wrist_rgb)
                target_heatmap, target_uv, heatmap_valid = _heatmap_targets(
                    cube_position,
                    env_cfg,
                    env.unwrapped,
                    heatmap_sigma,
                )
                heatmap_loss = _masked_heatmap_loss(
                    predicted_heatmap,
                    target_heatmap,
                    heatmap_valid,
                )
                predicted_uv = heatmap_soft_argmax(
                    predicted_heatmap.detach(),
                    image_height=int(env_cfg.wrist_camera.height),
                    image_width=int(env_cfg.wrist_camera.width),
                )
            else:
                predicted_normalized, predicted_contact_logits = model.predict_adaptation(wrist_rgb)
                heatmap_loss = predicted_normalized.sum() * 0.0
                target_uv = torch.zeros((cube_position.shape[0], 2), device=device)
                predicted_uv = torch.zeros_like(target_uv)
                heatmap_valid = torch.zeros(
                    (cube_position.shape[0],), device=device, dtype=torch.bool
                )
            predicted_contact = torch.sigmoid(predicted_contact_logits)
            target_normalized = model.actor_core.normalizer.normalize_position(cube_position)
            student_actions = model.action_from_normalized_position(
                proprio, history, predicted_normalized, predicted_contact
            )
            with torch.no_grad():
                teacher_actions = model.actor_core(
                    proprio, history, cube_position, contact_state
                )
                previous_environment_actions = torch.clamp(
                    history / action_smoothness_scale,
                    min=-1.0,
                    max=1.0,
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
            action_smoothness_loss = F.mse_loss(student_actions, previous_environment_actions)
            loss = (
                args.position_loss_weight * position_loss
                + args.contact_loss_weight * contact_loss
                + args.action_loss_weight * action_loss
                + args.action_smoothness_loss_weight * action_smoothness_loss
                + heatmap_loss_weight * heatmap_loss
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
            mae_xyz = torch.mean(position_error.abs(), dim=0)
            predicted_contact_binary = predicted_contact.detach() >= 0.5
            contact_accuracy = (
                predicted_contact_binary == contact_state.to(torch.bool)
            ).to(torch.float32).mean()
            heatmap_valid_fraction = heatmap_valid.to(torch.float32).mean()
            if bool(heatmap_valid.any().item()):
                heatmap_center_error_px = torch.linalg.norm(
                    predicted_uv[heatmap_valid] - target_uv[heatmap_valid],
                    dim=-1,
                ).mean()
            else:
                heatmap_center_error_px = torch.zeros((), device=device)

            writer.add_scalar("Loss/total", loss.item(), step)
            writer.add_scalar("Loss/position", position_loss.item(), step)
            writer.add_scalar("Loss/xyz", position_loss.item(), step)
            writer.add_scalar("Loss/contact_bce", contact_loss.item(), step)
            writer.add_scalar("Loss/action_distillation", action_loss.item(), step)
            writer.add_scalar("Loss/action_smoothness", action_smoothness_loss.item(), step)
            writer.add_scalar("Loss/heatmap", heatmap_loss.item(), step)
            writer.add_scalar("Position/rmse_3d_m", rmse_3d.item(), step)
            writer.add_scalar("Position/rmse_x_m", rmse_xyz[0].item(), step)
            writer.add_scalar("Position/rmse_y_m", rmse_xyz[1].item(), step)
            writer.add_scalar("Position/rmse_z_m", rmse_xyz[2].item(), step)
            writer.add_scalar("Position/mae_x_m", mae_xyz[0].item(), step)
            writer.add_scalar("Position/mae_y_m", mae_xyz[1].item(), step)
            writer.add_scalar("Position/mae_z_m", mae_xyz[2].item(), step)
            writer.add_scalar("Heatmap/center_error_px", heatmap_center_error_px.item(), step)
            writer.add_scalar("Heatmap/valid_fraction", heatmap_valid_fraction.item(), step)
            writer.add_scalar("Optimization/grad_norm", float(grad_norm), step)
            writer.add_scalar("Contact/accuracy", contact_accuracy.item(), step)
            writer.add_scalar("Contact/true_positive_fraction", contact_state.mean().item(), step)
            writer.add_scalar(
                "Contact/predicted_positive_fraction",
                predicted_contact_binary.to(torch.float32).mean().item(),
                step,
            )
            if (
                heatmap_loss_enabled
                and args.heatmap_debug_interval > 0
                and step % args.heatmap_debug_interval == 0
            ):
                _save_heatmap_debug_frames(
                    wrist_rgb,
                    target_uv,
                    predicted_uv,
                    heatmap_valid,
                    run_dir / "heatmap_debug",
                    step,
                    args.heatmap_debug_count,
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
                    f"  loss(total/xyz/contact/action/smooth/heatmap)={loss.item():.5f}/"
                    f"{position_loss.item():.5f}/{contact_loss.item():.5f}/"
                    f"{action_loss.item():.5f}/{action_smoothness_loss.item():.5f}/"
                    f"{heatmap_loss.item():.5f} "
                    f"| position_rmse={1000.0 * rmse_3d.item():.2f} mm "
                    f"| mae_xyz=({1000.0 * mae_xyz[0].item():.1f},"
                    f"{1000.0 * mae_xyz[1].item():.1f},"
                    f"{1000.0 * mae_xyz[2].item():.1f}) mm "
                    f"| heatmap_uv_err={heatmap_center_error_px.item():.2f}px "
                    f"| heatmap_valid={heatmap_valid_fraction.item():.3f} "
                    f"| contact_acc={contact_accuracy.item():.3f} "
                    f"| success(cumulative/recent)={cumulative_success_rate:.3f}/{recent_success_rate:.3f} "
                    f"({successful_episodes}/{completed_episodes} completed)",
                    flush=True,
                )
                previous_log_time = now
                previous_log_step = step

            if step % args.checkpoint_interval == 0 or step == args.timesteps:
                payload = _student_payload(
                    model,
                    optimizer,
                    step,
                    teacher_checkpoint,
                    teacher_manifest,
                    loss_contract,
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
