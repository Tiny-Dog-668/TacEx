"""PandaHand RMA models with visual cube-XY localization and force contact input."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO
from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space
from torchvision.models import ResNet18_Weights, resnet18

from .rma_models import (
    CubeCenterHeatmapHead,
    PandaFingertipKinematics,
    heatmap_soft_argmax,
    make_gaussian_heatmaps,
    project_points_root_to_image,
)


RMA_XY_MODEL_VERSION = 2
RMA_XY_PROPRIO_DIM = 15
RMA_XY_HISTORY_DIM = 4
RMA_XY_POSITION_DIM = 2
RMA_XY_CONTACT_DIM = 2
RMA_XY_ACTION_DIM = 4
RMA_XY_ACTOR_FEATURE_DIM = 26


class RMAXYObservationNormalizer(nn.Module):
    """Fixed normalization for the PandaHand XY/force RMA contract."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("joint_lower", torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]))
        self.register_buffer("joint_upper", torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]))
        self.register_buffer("joint_velocity_scale", torch.tensor([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]))
        self.register_buffer("history_scale", torch.tensor([0.025, 0.025, 0.025, 0.005]))
        self.register_buffer("cube_xy_center", torch.tensor([0.50, 0.00]))
        self.register_buffer("cube_xy_scale", torch.tensor([0.05, 0.05]))
        self.register_buffer("gripper_xy_center", torch.tensor([0.50, 0.00]))
        self.register_buffer("gripper_xy_scale", torch.tensor([0.10, 0.10]))
        self.register_buffer("target_xy_center", torch.tensor([0.0, 0.0]))
        self.register_buffer("target_xy_scale", torch.tensor([0.10, 0.10]))

    def normalize_proprio(self, proprio_obs: torch.Tensor) -> torch.Tensor:
        joint_mid = 0.5 * (self.joint_lower + self.joint_upper)
        joint_half_range = 0.5 * (self.joint_upper - self.joint_lower)
        return torch.cat([
            (proprio_obs[..., :7] - joint_mid) / joint_half_range,
            proprio_obs[..., 7:14] / self.joint_velocity_scale,
            (proprio_obs[..., 14:15] - 0.04) / 0.04,
        ], dim=-1)

    def normalize_history(self, action_history: torch.Tensor) -> torch.Tensor:
        return action_history / self.history_scale

    def normalize_position(self, cube_xy: torch.Tensor) -> torch.Tensor:
        return (cube_xy - self.cube_xy_center) / self.cube_xy_scale

    def denormalize_position(self, normalized_xy: torch.Tensor) -> torch.Tensor:
        return normalized_xy * self.cube_xy_scale + self.cube_xy_center

    def normalize_gripper_position(self, gripper_xy: torch.Tensor) -> torch.Tensor:
        return (gripper_xy - self.gripper_xy_center) / self.gripper_xy_scale

    def normalize_target_position(self, target_xy: torch.Tensor) -> torch.Tensor:
        return (target_xy - self.target_xy_center) / self.target_xy_scale

    def contract(self) -> dict[str, object]:
        return {
            "joint_lower": self.joint_lower.detach().cpu().tolist(),
            "joint_upper": self.joint_upper.detach().cpu().tolist(),
            "joint_velocity_scale": self.joint_velocity_scale.detach().cpu().tolist(),
            "gripper_width_center_scale": [0.04, 0.04],
            "history_scale": self.history_scale.detach().cpu().tolist(),
            "cube_xy_center": self.cube_xy_center.detach().cpu().tolist(),
            "cube_xy_scale": self.cube_xy_scale.detach().cpu().tolist(),
            "gripper_xy_center": self.gripper_xy_center.detach().cpu().tolist(),
            "gripper_xy_scale": self.gripper_xy_scale.detach().cpu().tolist(),
            "target_xy_center": self.target_xy_center.detach().cpu().tolist(),
            "target_xy_scale": self.target_xy_scale.detach().cpu().tolist(),
            "position_frame": "robot_root",
        }


class RMAXYActorCore(nn.Module):
    """PandaHand Teacher Actor shared with the visual XY Student."""

    def __init__(self) -> None:
        super().__init__()
        self.normalizer = RMAXYObservationNormalizer()
        self.kinematics = PandaFingertipKinematics()
        self.network = nn.Sequential(
            nn.Linear(RMA_XY_ACTOR_FEATURE_DIM, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, RMA_XY_ACTION_DIM),
        )

    def forward(self, proprio_obs: torch.Tensor, action_history: torch.Tensor, cube_xy: torch.Tensor,
                contact_force_n: torch.Tensor, position_is_normalized: bool = False) -> torch.Tensor:
        if not torch.jit.is_scripting():
            if cube_xy.shape[-1] != RMA_XY_POSITION_DIM:
                raise ValueError(f"Expected cube_xy[...,2], got {tuple(cube_xy.shape)}")
            if contact_force_n.shape[-1] != RMA_XY_CONTACT_DIM:
                raise ValueError(f"Expected contact_force_n[...,2], got {tuple(contact_force_n.shape)}")
        normalized_cube_xy = cube_xy if position_is_normalized else self.normalizer.normalize_position(cube_xy)
        cube_xy_root = self.normalizer.denormalize_position(cube_xy) if position_is_normalized else cube_xy
        gripper_xy_root = self.kinematics(proprio_obs[..., :7])[..., :2]
        target_xy_root = cube_xy_root - gripper_xy_root
        # One feature: grasped only when both finger forces are at least 1 N.
        grasped = torch.all(contact_force_n >= 1.0, dim=-1, keepdim=True).to(dtype=proprio_obs.dtype)
        features = torch.cat([
            self.normalizer.normalize_proprio(proprio_obs),
            self.normalizer.normalize_history(action_history),
            normalized_cube_xy,
            self.normalizer.normalize_gripper_position(gripper_xy_root),
            self.normalizer.normalize_target_position(target_xy_root),
            grasped,
        ], dim=-1)
        return torch.tanh(self.network(features))

    def contract(self) -> dict[str, object]:
        return {
            "feature_dim": RMA_XY_ACTOR_FEATURE_DIM,
            "feature_order": [
                "normalized_proprio_obs[15]", "normalized_action_history[4]",
                "normalized_cube_xy_root[2]", "normalized_gripper_xy_root_from_fk[2]",
                "normalized_cube_minus_gripper_xy_root[2]",
                "grasped_bilateral_cube_finger_force_n_gte_1N[1]",
            ],
            "object_pose_components": "position_xy_only",
            "end_effector_pose_components": "position_xy_only",
            "contact_components": "bilateral_cube_finger_force_n_gte_1N_binary",
            "kinematics": self.kinematics.contract(),
        }


class RMAXYPrivilegedTeacherPolicy(GaussianMixin, Model):
    """skrl Gaussian policy backed by :class:`RMAXYActorCore`."""

    def __init__(self, observation_space, action_space, device, clip_actions: bool = False,
                 clip_log_std: bool = True, min_log_std: float = -20.0, max_log_std: float = 2.0,
                 reduction: str = "sum", initial_log_std: float = 0.0, fixed_log_std: bool = False, **_: object) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        if self.num_actions != RMA_XY_ACTION_DIM:
            raise ValueError(f"RMA XY teacher requires 4 actions, got {self.num_actions}")
        self.actor_core = RMAXYActorCore().to(device)
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std), device=device), requires_grad=not fixed_log_std)

    def compute(self, inputs: Mapping[str, torch.Tensor], role: str = ""):
        del role
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        mean = self.actor_core(states["proprio_obs"], states["action_history"], states["rma_cube_xy"], states["rma_contact_force"])
        return mean, self.log_std_parameter, {}


class RMAXYPPO(PPO):
    """Named PPO alias for the PandaHand XY RMA Teacher."""


class SpatialSoftmaxXYAdaptationHead(nn.Module):
    """Predict normalized cube XY only; never infer contact from images."""

    def __init__(self, channels: int = 32, height: int = 7, width: int = 7) -> None:
        super().__init__()
        self.channels = channels
        self.keypoint_conv = nn.Conv2d(512, channels, kernel_size=1)
        self.register_buffer("x_coords", torch.linspace(-1.0, 1.0, width).repeat(height))
        self.register_buffer("y_coords", torch.linspace(-1.0, 1.0, height).repeat_interleave(width))
        self.trunk = nn.Sequential(nn.Linear(2 * channels, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU())
        self.position_output = nn.Linear(64, RMA_XY_POSITION_DIM)
        nn.init.zeros_(self.position_output.weight)
        nn.init.zeros_(self.position_output.bias)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        attention = torch.softmax(self.keypoint_conv(feature_map).reshape(feature_map.shape[0], self.channels, -1), dim=-1)
        features = self.trunk(torch.cat([
            torch.sum(attention * self.x_coords, dim=-1),
            torch.sum(attention * self.y_coords, dim=-1),
        ], dim=-1))
        return torch.tanh(self.position_output(features))


class RMAXYVisualStudent(nn.Module):
    """Frozen ResNet18 XY localizer plus force-aware frozen Teacher Actor."""

    def __init__(self, actor_core: RMAXYActorCore, *, pretrained_backbone: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        self.vision_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.adaptation_head = SpatialSoftmaxXYAdaptationHead()
        self.heatmap_head = CubeCenterHeatmapHead()
        self.actor_core = actor_core
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.freeze_fixed_modules()

    def freeze_fixed_modules(self) -> None:
        self.vision_encoder.eval(); self.actor_core.eval()
        for module in (self.vision_encoder, self.actor_core):
            for parameter in module.parameters(): parameter.requires_grad_(False)

    @torch.jit.unused
    def unfreeze_backbone_after_layer2(self) -> None:
        for parameter in self.vision_encoder.parameters(): parameter.requires_grad_(False)
        for index, module in enumerate(self.vision_encoder):
            if index >= 6:
                for parameter in module.parameters(): parameter.requires_grad_(True)
        self.vision_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode); self.vision_encoder.eval(); self.actor_core.eval(); return self

    def encode_features(self, wrist_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = wrist_rgb.to(torch.float32).permute(0, 3, 1, 2) / 255.0
        x = (image - self.image_mean) / self.image_std
        layer3 = x
        for index, module in enumerate(self.vision_encoder):
            x = module(x)
            if index == 6: layer3 = x
        return layer3, x

    def predict_adaptation(self, wrist_rgb: torch.Tensor) -> torch.Tensor:
        _, layer4 = self.encode_features(wrist_rgb)
        return self.adaptation_head(layer4)

    def predict_adaptation_and_heatmap(self, wrist_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layer3, layer4 = self.encode_features(wrist_rgb)
        return self.adaptation_head(layer4), self.heatmap_head(layer3)

    def action_from_normalized_position(self, proprio_obs: torch.Tensor, action_history: torch.Tensor,
                                        normalized_xy: torch.Tensor, contact_force_n: torch.Tensor) -> torch.Tensor:
        return self.actor_core(proprio_obs, action_history, normalized_xy, contact_force_n, position_is_normalized=True)

    def forward(self, wrist_rgb: torch.Tensor, proprio_obs: torch.Tensor, action_history: torch.Tensor,
                contact_force_n: torch.Tensor) -> torch.Tensor:
        return self.action_from_normalized_position(proprio_obs, action_history, self.predict_adaptation(wrist_rgb), contact_force_n)


def extract_xy_actor_core_state_dict(policy_state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = {key[len("actor_core."):]: value for key, value in policy_state_dict.items() if key.startswith("actor_core.")}
    if not state:
        raise RuntimeError("Teacher checkpoint policy has no actor_core.* parameters")
    return state
