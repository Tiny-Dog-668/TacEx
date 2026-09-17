"""Distill a GelSight three-frame Student from its paired Teacher."""

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
parser.add_argument(
    "--encoder_init_checkpoint",
    required=True,
    help="Approved RMA XY Heatmap-DR Student checkpoint used to initialize ResNet18.",
)
parser.add_argument("--resume", default=None)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--timesteps", type=int, default=100_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--environment_seed",
    type=int,
    default=None,
    help="Physical geometry seed. Defaults to the paired Teacher manifest seed.",
)
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
parser.add_argument("--contact_positive_weight", type=float, default=1.0)
parser.add_argument("--occlusion_loss_weight", type=float, default=1.0)
parser.add_argument(
    "--occlusion_predictor_checkpoint",
    default=None,
    help="Required only by the two Alpha-Aux comparison Students.",
)
parser.add_argument("--action_loss_weight", type=float, default=1.0)
parser.add_argument("--action_smoothness_loss_weight", type=float, default=0.05)
parser.add_argument("--smooth_l1_beta", type=float, default=0.1)
parser.add_argument("--grad_norm_clip", type=float, default=1.0)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--checkpoint_interval", type=int, default=10_000)
parser.add_argument("--log_dir", default=None)
parser.add_argument(
    "--save_initial_images",
    action="store_true",
    help="Save one post-reset policy-camera RGB PNG per environment under the run directory.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
import torch.nn.functional as F
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_gelsight_rma import rma_gelsight_pulled_drawer_artifacts as drawer_artifacts
from tacex_tasks.sim2real_gelsight_rma import rma_gelsight_x040_three_frame_artifacts as x040_artifacts
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_pulled_drawer_artifacts import (
    GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
    GELSIGHT_PULLED_DRAWER_STUDENT_TASKS,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_x040_three_frame_artifacts import (
    GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
    GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_DR_TASK,
    GELSIGHT_X040_STUDENT_TASKS,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_x040_binary_tactile_models import (
    RMAGelSightX040BinaryTactileThreeFrameStudent,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_pulled_drawer_models import (
    RMAGelSightPulledDrawerActorCore,
    RMAGelSightPulledDrawerThreeFrameStudent,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_pulled_drawer_binary_tactile_models import (
    RMAGelSightPulledDrawerBinaryTactileThreeFrameStudent,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_pulled_drawer_four_tactile_models import (
    RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent,
    RMAGelSightPulledDrawerFourTactileActorCore,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_large_drawer_fusion_models import (
    TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE,
    TACTILE_CROSS_ALPHA_DOWNSAMPLE,
    VISION_ONLY_DOWNSAMPLE,
    is_aux_variant,
    is_recurrent_variant,
    is_tactile_variant,
)
from tacex_tasks.sim2real_gelsight_rma.large_drawer_occlusion import (
    load_xy_occlusion_predictor,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cube_real_alignment_gelsight_pulled_drawer_four_tactile_env import (
    GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cylinder_real_alignment_gelsight_pulled_drawer_four_tactile_env import (
    GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_env import (
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
)
from tacex_tasks.sim2real_gelsight_rma.sim2real_cylinder_real_alignment_gelsight_large_pulled_drawer_four_tactile_downsample_env import (
    GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
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


BINARY_TACTILE_STUDENT_TASKS = frozenset(
    {
        GELSIGHT_X040_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
        GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK,
        GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
    }
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


def _save_initial_policy_images(observations: dict, output_dir: Path) -> None:
    """Save the first valid post-reset RGB frame for every environment."""
    history = observations["policy"]["wrist_rgb_history"]
    if history.ndim != 5 or history.shape[1] != 3 or history.shape[-1] != 3:
        raise RuntimeError(
            "Expected wrist_rgb_history [N,3,H,W,3], got "
            f"{tuple(history.shape)}"
        )
    images = history[:, -1].detach().to(device="cpu", dtype=torch.uint8).numpy()
    output_dir.mkdir(parents=True, exist_ok=True)
    for env_id, image in enumerate(images):
        Image.fromarray(image, mode="RGB").save(output_dir / f"env_{env_id:04d}.png")
    _atomic_json_dump(
        {
            "source": "policy.wrist_rgb_history[:, -1]",
            "reset_fill": "repeat_first_post_reset_frame",
            "num_envs": int(images.shape[0]),
            "image_shape_hwc": list(images.shape[1:]),
        },
        output_dir / "metadata.json",
    )


def _loss_contract() -> dict[str, object]:
    fusion_variant = drawer_artifacts.LARGE_DRAWER_FUSION_VARIANT_BY_TASK.get(args.task)
    tactile_variant = fusion_variant is None or is_tactile_variant(fusion_variant)
    contact_source = "absent" if not tactile_variant else (
        "four_sensor_bce_from_single_channel_binary_max_abs_rgb_delta_gt_5_u8"
        if fusion_variant is not None or args.task in BINARY_TACTILE_STUDENT_TASKS
        else "left_right_bce_from_signed_current_minus_reference"
    )
    contract = {
        "position": "smooth_l1_normalized_cube_position_root_xyz",
        "position_weight": float(args.position_loss_weight),
        "smooth_l1_beta": float(args.smooth_l1_beta),
        "contact": contact_source,
        "contact_weight": float(args.contact_loss_weight),
        "contact_positive_weight": float(args.contact_positive_weight),
        "action": "mse_teacher_deterministic_mean",
        "action_weight": float(args.action_loss_weight),
        "action_smoothness": "mse_to_previous_environment_action",
        "action_smoothness_weight": float(args.action_smoothness_loss_weight),
        "heatmap": "absent",
    }
    if fusion_variant is not None and is_aux_variant(fusion_variant):
        contract["occlusion"] = "mse_visual_prediction_to_frozen_xy_predictor"
        contract["occlusion_weight"] = float(args.occlusion_loss_weight)
    return contract


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
    if args.task not in {
        *GELSIGHT_X040_STUDENT_TASKS,
        *GELSIGHT_PULLED_DRAWER_STUDENT_TASKS,
    }:
        raise ValueError("This trainer only supports the paired X040 or Pulled-Drawer Student task")

    drawer_profile = args.task in GELSIGHT_PULLED_DRAWER_STUDENT_TASKS
    fusion_spec = drawer_artifacts.LARGE_DRAWER_FUSION_REGISTRY.get(args.task)
    fusion_variant = fusion_spec["variant"] if fusion_spec is not None else None
    fusion_profile = fusion_spec is not None
    fusion_tactile_profile = fusion_profile and is_tactile_variant(fusion_variant)
    fusion_aux_profile = fusion_profile and is_aux_variant(fusion_variant)
    fusion_recurrent_profile = fusion_profile and is_recurrent_variant(fusion_variant)
    binary_tactile_profile = args.task in BINARY_TACTILE_STUDENT_TASKS
    four_tactile_profile = args.task in {
        GELSIGHT_PULLED_DRAWER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK,
        GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK,
    } or fusion_profile
    artifacts = drawer_artifacts if drawer_profile else x040_artifacts
    student_cls = (RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent
        if four_tactile_profile else
        RMAGelSightPulledDrawerBinaryTactileThreeFrameStudent
        if drawer_profile and binary_tactile_profile
        else (
            RMAGelSightPulledDrawerThreeFrameStudent
            if drawer_profile
            else (
                RMAGelSightX040BinaryTactileThreeFrameStudent
                if binary_tactile_profile
                else RMAGelSightX040ThreeFrameStudent
            )
        )
    )
    teacher_cls = (RMAGelSightPulledDrawerFourTactileActorCore if four_tactile_profile
                   else RMAGelSightPulledDrawerActorCore if drawer_profile else RMAGelSightActorCore)
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if min(args.timesteps, args.log_interval, args.checkpoint_interval) <= 0:
        raise ValueError("timesteps, log_interval, and checkpoint_interval must be positive")
    if min(args.learning_rate, args.backbone_learning_rate, args.grad_norm_clip) <= 0:
        raise ValueError("learning rates and grad_norm_clip must be positive")
    if min(
        args.position_loss_weight,
        args.contact_loss_weight,
        args.occlusion_loss_weight,
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
    encoder_checkpoint = Path(args.encoder_init_checkpoint).expanduser().resolve()
    teacher_manifest = artifacts.load_teacher_manifest(teacher_checkpoint)
    encoder_state, encoder_payload = artifacts.load_encoder_initialization_checkpoint(
        encoder_checkpoint
    )
    resume_payload = None
    if args.resume:
        resume_payload = artifacts.load_student_checkpoint(
            args.resume,
            device="cpu",
            expected_teacher_checkpoint=teacher_checkpoint,
        )
        if resume_payload.get("task") != args.task:
            raise RuntimeError("Resume checkpoint task differs from --task")
        if resume_payload.get("encoder_init_checkpoint_sha256") != artifacts.sha256_file(
            encoder_checkpoint
        ):
            raise RuntimeError("Resume checkpoint used a different encoder initialization")
        if resume_payload.get("loss") != _loss_contract():
            raise RuntimeError("Resume loss contract differs from current arguments")
        if resume_payload.get("optimizer_config") != _optimizer_contract():
            raise RuntimeError("Resume optimizer contract differs from current arguments")
        if args.occlusion_predictor_checkpoint and resume_payload.get(
            "occlusion_predictor_checkpoint_sha256"
        ) != artifacts.sha256_file(args.occlusion_predictor_checkpoint):
            raise RuntimeError("Resume checkpoint used a different occlusion predictor")

    if fusion_aux_profile and not args.occlusion_predictor_checkpoint:
        raise ValueError("Aux fusion Students require --occlusion_predictor_checkpoint")
    if not fusion_aux_profile and args.occlusion_predictor_checkpoint:
        raise ValueError("--occlusion_predictor_checkpoint is only valid for Aux Students")
    occlusion_predictor = None
    occlusion_predictor_metadata = None
    if fusion_aux_profile:
        occlusion_predictor, occlusion_predictor_metadata = load_xy_occlusion_predictor(
            args.occlusion_predictor_checkpoint, device=args.device
        )

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if args.environment_seed is not None:
        env_cfg.seed = args.environment_seed
    elif drawer_profile:
        env_cfg.seed = int(teacher_manifest["environment_contract"]["geometry_seed"])
    else:
        env_cfg.seed = args.seed
    if resume_payload is not None:
        env_cfg.illegal_collision_curriculum_step_offset = int(resume_payload["global_step"])
    artifacts.validate_live_teacher_contract(env_cfg, teacher_manifest)
    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped
    device = torch.device(base_env.device)
    if resume_payload is not None:
        if resume_payload.get("student_environment_contract") != artifacts.student_environment_contract(
            env_cfg
        ):
            raise RuntimeError("Resume Student environment contract mismatch")
    if resume_payload is not None and drawer_profile:
        if resume_payload.get("geometry_instance_sha256") != artifacts.geometry_instance_sha256(
            base_env
        ):
            raise RuntimeError("Resume Pulled-Drawer geometry instance mismatch")

    teacher = teacher_cls().to(device).eval()
    teacher.load_state_dict(
        extract_actor_core_state_dict(
            artifacts.load_teacher_policy_state(teacher_checkpoint, device)
        ),
        strict=True,
    )
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    model = (
        artifacts.make_student_model_for_task(args.task, pretrained_backbone=False)
        if fusion_profile
        else student_cls(pretrained_backbone=False)
    ).to(device)
    model.load_vision_encoder_state(encoder_state)
    if fusion_profile:
        head_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("vision_encoder.")
        ]
    else:
        tactile_parameters = (
            list(model.inner_tactile_encoder.parameters()) + list(model.down_tactile_encoder.parameters())
            if four_tactile_profile else list(model.tactile_encoder.parameters())
        )
        head_parameters = (
            list(model.temporal_fusion.parameters())
            + tactile_parameters
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

    if fusion_profile:
        default_log_root = str(fusion_spec["log_directory"])
    elif args.task == GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_DOWNSAMPLE_BINARY_STUDENT_TASK:
        default_log_root = (
            "logs/skrl/sim2real_cylinder_real_alignment_rma_gelsight_large_pulled_drawer_"
            "four_tactile_downsample_progress_three_frame_binary_student"
        )
    elif args.task == GELSIGHT_LARGE_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK:
        default_log_root = (
            "logs/skrl/sim2real_cylinder_real_alignment_rma_gelsight_large_pulled_drawer_"
            "four_tactile_progress_three_frame_binary_student"
        )
    elif args.task == GELSIGHT_PULLED_DRAWER_CYLINDER_PROGRESS_FOUR_TACTILE_BINARY_STUDENT_TASK:
        default_log_root = (
            "logs/skrl/sim2real_cylinder_real_alignment_rma_gelsight_pulled_drawer_"
            "four_tactile_progress_three_frame_binary_student"
        )
    elif four_tactile_profile:
        default_log_root = (
            "logs/skrl/sim2real_cube_real_alignment_rma_gelsight_pulled_drawer_"
            "four_tactile_progress_three_frame_binary_student"
        )
    elif (
        args.task
        == GELSIGHT_PULLED_DRAWER_PROGRESS_BINARY_TACTILE_THREE_FRAME_STUDENT_DR_TASK
    ):
        default_log_root = (
            "logs/skrl/sim2real_cube_real_alignment_rma_gelsight_pulled_drawer_"
            "progress_three_frame_binary_tactile_student"
        )
    elif drawer_profile:
        default_log_root = (
            "logs/skrl/sim2real_cube_real_alignment_rma_gelsight_pulled_drawer_student"
        )
    elif binary_tactile_profile:
        default_log_root = (
            "logs/skrl/sim2real_cube_real_alignment_rma_gelsight_x040_progress_"
            "three_frame_binary_tactile_student"
        )
    elif args.task == GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_DR_TASK:
        default_log_root = (
            "logs/skrl/sim2real_cube_real_alignment_rma_gelsight_x040_progress_"
            "three_frame_student"
        )
    else:
        default_log_root = (
            "logs/skrl/sim2real_cube_real_alignment_rma_gelsight_x040_dr_three_frame_"
            "student_x040_normalized_heatmap_init"
        )
    run_dir = (
        Path(args.log_dir).expanduser().resolve()
        if args.log_dir
        else Path(default_log_root) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_distillation"
    )
    _atomic_json_dump(
        {
            **vars(args),
            "teacher_checkpoint_sha256": artifacts.sha256_file(teacher_checkpoint),
            "teacher_actor_state_dict_sha256": artifacts.state_dict_sha256(teacher.state_dict()),
            "encoder_init_checkpoint_sha256": artifacts.sha256_file(encoder_checkpoint),
            "encoder_init_task": encoder_payload.get("task"),
            "encoder_init_state_dict_sha256": encoder_payload.get(
                "vision_encoder_state_dict_sha256"
            ),
            "position_normalization": model.normalizer.contract(),
            "runtime_inputs": model.contract()["runtime_input_order"],
            "runtime_outputs": model.contract()["runtime_output"],
            "training_only_labels": artifacts.student_input_contract(args.task)[
                "training_only_labels"
            ],
            "fusion_variant": fusion_variant,
            "physical_environment_seed": int(env_cfg.seed),
            "occlusion_predictor_checkpoint_sha256": (
                artifacts.sha256_file(args.occlusion_predictor_checkpoint)
                if fusion_aux_profile else None
            ),
        },
        run_dir / "params" / "training.json",
    )
    writer = SummaryWriter(str(run_dir))
    start_step = 0
    if resume_payload is not None:
        artifacts.load_student_model_state(model, resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        start_step = int(resume_payload["global_step"])
    model.train()

    observations, _ = env.reset()
    if args.save_initial_images:
        initial_image_dir = run_dir / "initial_images"
        _save_initial_policy_images(observations, initial_image_dir)
        print(
            f"[INFO] Saved {args.num_envs} initial policy-camera images to "
            f"{initial_image_dir}"
        )
    action_scale = torch.tensor(
        [env_cfg.action_scale] * 3 + [env_cfg.gripper_width_delta_scale],
        device=device,
        dtype=torch.float32,
    )
    positive_weight = torch.full(
        (4 if four_tactile_profile else 2,),
        args.contact_positive_weight,
        device=device,
    )
    tactile_feature_history = torch.zeros(
        (args.num_envs, 4, 9, 256), dtype=torch.float32, device=device
    )
    reset_mask = torch.ones((args.num_envs,), dtype=torch.bool, device=device)
    started = time.perf_counter()
    try:
        for step in range(start_step + 1, args.timesteps + 1):
            obs = observations["policy"]
            proprio = obs["proprio_obs"].float()
            history = obs["action_history"].float()
            cube_position = obs["rma_cube_pos"].float()
            contact_target = obs["rma_contact_state"].float()
            tactile_inputs = []
            if not fusion_profile or fusion_tactile_profile:
                tactile_inputs = [obs["gsmini_left_rgb"], obs["gsmini_right_rgb"]]
                if four_tactile_profile:
                    tactile_inputs += [obs["gsmini_left_down_rgb"], obs["gsmini_right_down_rgb"]]
                tactile_inputs += [obs["gsmini_left_reference_rgb"], obs["gsmini_right_reference_rgb"]]
                if four_tactile_profile:
                    tactile_inputs += [obs["gsmini_left_down_reference_rgb"], obs["gsmini_right_down_reference_rgb"]]

            alpha = None
            occlusion_prediction = None
            attention = None
            contact_ratios = None
            next_tactile_feature_history = None
            if fusion_variant == VISION_ONLY_DOWNSAMPLE:
                student_actions, normalized_position = model.forward_with_training_outputs(
                    obs["wrist_rgb_history"], proprio, history
                )
                contact_logits = None
            elif fusion_recurrent_profile:
                (
                    student_actions,
                    normalized_position,
                    contact_logits,
                    alpha,
                    occlusion_prediction,
                    next_tactile_feature_history,
                    attention,
                    contact_ratios,
                ) = model.forward_with_recurrent_outputs(
                    obs["wrist_rgb_history"], proprio, history, *tactile_inputs,
                    tactile_feature_history, reset_mask,
                )
            elif fusion_aux_profile:
                (
                    student_actions,
                    normalized_position,
                    contact_logits,
                    alpha,
                    occlusion_prediction,
                    attention,
                    contact_ratios,
                ) = model.forward_with_auxiliary_outputs(
                    obs["wrist_rgb_history"], proprio, history, *tactile_inputs
                )
            elif fusion_variant == TACTILE_CROSS_ALPHA_DOWNSAMPLE:
                (
                    student_actions,
                    normalized_position,
                    contact_logits,
                    alpha,
                    attention,
                ) = model.forward_with_diagnostics(
                    obs["wrist_rgb_history"], proprio, history, *tactile_inputs
                )
            else:
                student_actions, normalized_position, contact_logits = model.forward_with_training_outputs(
                    obs["wrist_rgb_history"], proprio, history, *tactile_inputs
                )
            with torch.no_grad():
                teacher_actions = teacher(proprio, history, cube_position, contact_target)
                previous_actions = torch.clamp(history / action_scale, -1.0, 1.0)
                position_target = model.normalizer.normalize_position(cube_position)
                occlusion_target = (
                    occlusion_predictor(cube_position[:, :2])
                    if occlusion_predictor is not None else None
                )
            action_loss = F.mse_loss(student_actions, teacher_actions)
            smoothness_loss = F.mse_loss(student_actions, previous_actions)
            position_loss = F.smooth_l1_loss(
                normalized_position, position_target, beta=args.smooth_l1_beta
            )
            contact_loss = (
                F.binary_cross_entropy_with_logits(
                    contact_logits, contact_target, pos_weight=positive_weight
                )
                if contact_logits is not None
                else student_actions.sum() * 0.0
            )
            occlusion_loss = (
                F.mse_loss(occlusion_prediction, occlusion_target)
                if occlusion_prediction is not None and occlusion_target is not None
                else student_actions.sum() * 0.0
            )
            loss = (
                args.action_loss_weight * action_loss
                + args.action_smoothness_loss_weight * smoothness_loss
                + args.position_loss_weight * position_loss
                + args.contact_loss_weight * contact_loss
                + args.occlusion_loss_weight * occlusion_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_norm_clip)
            optimizer.step()
            observations, rewards, terminated, truncated, _ = env.step(student_actions.detach())
            if next_tactile_feature_history is not None:
                tactile_feature_history = next_tactile_feature_history.detach()
                reset_mask = torch.logical_or(terminated, truncated)

            with torch.no_grad():
                position_rmse = torch.sqrt(
                    torch.mean(
                        (
                            model.normalizer.denormalize_position(normalized_position)
                            - cube_position
                        ).square()
                    )
                )
                contact_probability = (
                    torch.sigmoid(contact_logits) if contact_logits is not None else None
                )
                contact_accuracy = (
                    ((contact_probability >= 0.5) == (contact_target >= 0.5)).float().mean()
                    if contact_probability is not None else None
                )
            writer.add_scalar("Loss/total", loss.item(), step)
            writer.add_scalar("Loss/position", position_loss.item(), step)
            writer.add_scalar("Loss/contact_bce", contact_loss.item(), step)
            writer.add_scalar("Loss/occlusion", occlusion_loss.item(), step)
            writer.add_scalar("Loss/action", action_loss.item(), step)
            writer.add_scalar("Loss/action_smoothness", smoothness_loss.item(), step)
            writer.add_scalar("Position/rmse_m", position_rmse.item(), step)
            if contact_accuracy is not None:
                writer.add_scalar("Contact/mean_accuracy", contact_accuracy.item(), step)
            if alpha is not None:
                writer.add_scalar("Fusion/alpha_mean", alpha.mean().item(), step)
            if occlusion_prediction is not None and occlusion_target is not None:
                writer.add_scalar("Fusion/occlusion_prediction_mean", occlusion_prediction.mean().item(), step)
                writer.add_scalar("Fusion/occlusion_target_mean", occlusion_target.mean().item(), step)
            if contact_ratios is not None:
                writer.add_scalar("Fusion/tactile_contact_area_mean", contact_ratios.mean().item(), step)
            if attention is not None:
                probabilities = attention.clamp_min(1.0e-8)
                entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
                writer.add_scalar("Fusion/attention_entropy", entropy.item(), step)
            writer.add_scalar("Action/teacher_rmse", torch.sqrt(action_loss.detach()).item(), step)
            writer.add_scalar("Optimization/grad_norm", float(grad_norm), step)
            writer.add_scalar("Reward/mean_step", rewards.mean().item(), step)

            if step % args.log_interval == 0 or step == args.timesteps:
                stats = base_env._episode_success_statistics()
                writer.add_scalar("Performance/recent_success_rate", stats["window_rate"].item(), step)
                print(
                    f"[GelSight Three-Frame Student] {step}/{args.timesteps} "
                    f"loss(action/smooth/pos/contact/occ)={action_loss.item():.4f}/"
                    f"{smoothness_loss.item():.4f}/{position_loss.item():.4f}/"
                    f"{contact_loss.item():.4f}/{occlusion_loss.item():.4f} "
                    f"position_rmse={1000.0 * position_rmse.item():.1f}mm "
                    f"contact_accuracy={(contact_accuracy.item() if contact_accuracy is not None else float('nan')):.3f} "
                    f"success={stats['cumulative_rate'].item():.3f} "
                    f"updates_s={(step - start_step) / max(time.perf_counter() - started, 1.0e-6):.2f}",
                    flush=True,
                )

            if step % args.checkpoint_interval == 0 or step == args.timesteps:
                payload_kwargs = {
                    "student_task": args.task,
                    "model": model,
                    "optimizer": optimizer,
                    "global_step": step,
                    "teacher_checkpoint": teacher_checkpoint,
                    "teacher_manifest": teacher_manifest,
                    "teacher_actor_state_dict": teacher.state_dict(),
                    "encoder_init_checkpoint": encoder_checkpoint,
                    "encoder_init_payload": encoder_payload,
                    "student_env_contract": artifacts.student_environment_contract(env_cfg),
                    "loss": _loss_contract(),
                    "optimizer_config": _optimizer_contract(),
                }
                if drawer_profile:
                    payload_kwargs["geometry_instance_hash"] = artifacts.geometry_instance_sha256(
                        base_env
                    )
                if fusion_aux_profile:
                    payload_kwargs["occlusion_predictor_checkpoint"] = args.occlusion_predictor_checkpoint
                    payload_kwargs["occlusion_predictor_metadata"] = occlusion_predictor_metadata
                payload = artifacts.make_student_payload(**payload_kwargs)
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
