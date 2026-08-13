"""Models for the X040-Wide privileged XYZ Teacher and visual Student."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO
from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space
from torchvision.models import resnet18

from .rma_models import PandaFingertipKinematics
from .rma_direct_action_student.models import RMADirectActionObservationNormalizer


RMA_X040_WIDE_MODEL_VERSION = 1
RMA_X040_WIDE_DIRECT_STUDENT_MODEL_VERSION = 1
RMA_X040_WIDE_ACTOR_FEATURE_DIM = 28


class RMAX040WideObservationNormalizer(RMADirectActionObservationNormalizer):
    """Fixed normalization for cube XYZ in the X040-Wide robot-root frame."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("cube_position_center", torch.tensor([0.40, 0.00, 0.026]))
        self.register_buffer("cube_position_scale", torch.tensor([0.08, 0.10, 0.10]))
        self.register_buffer("gripper_position_center", torch.tensor([0.50, 0.00, 0.175]))
        self.register_buffer("gripper_position_scale", torch.tensor([0.10, 0.10, 0.15]))
        self.register_buffer("target_position_center", torch.tensor([-0.10, 0.00, -0.149]))
        self.register_buffer("target_position_scale", torch.tensor([0.18, 0.10, 0.15]))

    def normalize_position(self, cube_position: torch.Tensor) -> torch.Tensor:
        return (cube_position - self.cube_position_center) / self.cube_position_scale

    def denormalize_position(self, normalized_position: torch.Tensor) -> torch.Tensor:
        return normalized_position * self.cube_position_scale + self.cube_position_center

    def normalize_gripper_position(self, position: torch.Tensor) -> torch.Tensor:
        return (position - self.gripper_position_center) / self.gripper_position_scale

    def normalize_target_position(self, position: torch.Tensor) -> torch.Tensor:
        return (position - self.target_position_center) / self.target_position_scale

    def contract(self) -> dict[str, object]:
        contract = super().contract()
        contract.update({
            "cube_position_center": self.cube_position_center.detach().cpu().tolist(),
            "cube_position_scale": self.cube_position_scale.detach().cpu().tolist(),
            "gripper_position_center": self.gripper_position_center.detach().cpu().tolist(),
            "gripper_position_scale": self.gripper_position_scale.detach().cpu().tolist(),
            "target_position_center": self.target_position_center.detach().cpu().tolist(),
            "target_position_scale": self.target_position_scale.detach().cpu().tolist(),
            "position_frame": "robot_root",
        })
        return contract


class RMAX040WideActorCore(nn.Module):
    """28-feature Teacher Actor: proprio/history/cube XYZ, no contact input."""

    def __init__(self) -> None:
        super().__init__()
        self.normalizer = RMAX040WideObservationNormalizer()
        self.kinematics = PandaFingertipKinematics()
        self.network = nn.Sequential(
            nn.Linear(RMA_X040_WIDE_ACTOR_FEATURE_DIM, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 4),
        )

    def forward(self, proprio_obs: torch.Tensor, action_history: torch.Tensor, cube_position: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting() and cube_position.shape[-1] != 3:
            raise ValueError(f"Expected cube_position[...,3], got {tuple(cube_position.shape)}")
        gripper_position = self.kinematics(proprio_obs[..., :7])
        target_position = cube_position - gripper_position
        features = torch.cat((
            self.normalizer.normalize_proprio(proprio_obs),
            self.normalizer.normalize_history(action_history),
            self.normalizer.normalize_position(cube_position),
            self.normalizer.normalize_gripper_position(gripper_position),
            self.normalizer.normalize_target_position(target_position),
        ), dim=-1)
        return torch.tanh(self.network(features))

    def contract(self) -> dict[str, object]:
        return {
            "feature_dim": RMA_X040_WIDE_ACTOR_FEATURE_DIM,
            "feature_order": [
                "normalized_proprio_obs[15]", "normalized_action_history[4]",
                "normalized_cube_position_root[3]", "normalized_gripper_position_root_from_fk[3]",
                "normalized_cube_minus_gripper_position_root[3]",
            ],
            "object_pose_components": "position_xyz_only",
            "end_effector_pose_components": "position_xyz_only",
            "contact_components": "none",
            "kinematics": self.kinematics.contract(),
        }


class RMAX040WidePrivilegedTeacherPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0, reduction="sum", initial_log_std=0.0,
                 fixed_log_std=False, **_: object) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        if self.num_actions != 4:
            raise ValueError(f"X040-Wide Teacher requires 4 actions, got {self.num_actions}")
        self.actor_core = RMAX040WideActorCore().to(device)
        self.log_std_parameter = nn.Parameter(torch.full((4,), float(initial_log_std), device=device), requires_grad=not fixed_log_std)

    def compute(self, inputs: Mapping[str, torch.Tensor], role: str = ""):
        del role
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        return self.actor_core(states["proprio_obs"], states["action_history"], states["rma_cube_pos"]), self.log_std_parameter, {}


class RMAX040WidePPO(PPO):
    """Named PPO alias for the X040-Wide Teacher."""


class RMAX040WideDirectActionVisualStudent(nn.Module):
    """Three-input deployment policy with an RGB-only cube XYZ auxiliary head."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        self.vision_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.normalizer = RMAX040WideObservationNormalizer()
        self.action_head = nn.Sequential(
            nn.Linear(531, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 4),
        )
        self.position_head = nn.Sequential(
            nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 3),
        )
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.freeze_backbone_after_layer2()

    def freeze_backbone_after_layer2(self) -> None:
        for index, module in enumerate(self.vision_encoder):
            for parameter in module.parameters():
                parameter.requires_grad_(index >= 6)
        self.vision_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_encoder.eval()
        return self

    def load_vision_encoder_state(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        result = self.vision_encoder.load_state_dict(dict(state_dict), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("X040-Wide Student vision encoder state mismatch")

    def encode_visual_features(self, wrist_rgb: torch.Tensor) -> torch.Tensor:
        image = wrist_rgb.to(torch.float32).permute(0, 3, 1, 2) / 255.0
        feature_map = self.vision_encoder((image - self.image_mean) / self.image_std)
        return torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1)), 1)

    def forward_with_position(self, wrist_rgb: torch.Tensor, proprio_obs: torch.Tensor, action_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.encode_visual_features(wrist_rgb)
        action_features = torch.cat((visual, self.normalizer.normalize_proprio(proprio_obs), self.normalizer.normalize_history(action_history)), dim=-1)
        return torch.tanh(self.action_head(action_features)), self.position_head(visual)

    def forward(self, wrist_rgb: torch.Tensor, proprio_obs: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        return self.forward_with_position(wrist_rgb, proprio_obs, action_history)[0]

    def contract(self) -> dict[str, object]:
        return {
            "model_version": RMA_X040_WIDE_DIRECT_STUDENT_MODEL_VERSION,
            "input_order": ["wrist_rgb", "proprio_obs", "action_history"],
            "input_signature": {"wrist_rgb": [224, 224, 3], "proprio_obs": [15], "action_history": [4]},
            "output_signature": {"mean_actions": [4]}, "feature_dim": 531,
            "feature_order": ["resnet18_global_average_pool_visual[512]", "normalized_proprio_obs[15]", "normalized_action_history[4]"],
            "runtime_privileged_inputs": [], "training_only_label": "normalized_cube_position_root[3]",
            "vision_encoder": "resnet18_layer4_global_average_pool", "action_head": [531, 512, 256, 128, 64, 4],
            "position_head": [512, 256, 128, 3], "activation": "ELU_then_action_tanh", "batch_norm": "all_resnet18_batch_norm_eval",
        }


def extract_x040_wide_actor_core_state_dict(policy_state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = {key[len("actor_core."):]: value for key, value in policy_state_dict.items() if key.startswith("actor_core.")}
    if not state:
        raise RuntimeError("Teacher checkpoint policy has no X040-Wide actor_core state")
    return state
