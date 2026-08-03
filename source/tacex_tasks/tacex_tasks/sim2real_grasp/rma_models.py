"""Shared teacher Actor and visual adaptation models for RMA-style distillation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.agents.torch.ppo import PPO
from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space
from torchvision.models import ResNet18_Weights, resnet18


RMA_MODEL_VERSION = 3
RMA_PROPRIO_DIM = 15
RMA_HISTORY_DIM = 4
RMA_POSITION_DIM = 3
RMA_CONTACT_DIM = 2
RMA_ACTION_DIM = 4
RMA_ACTOR_FEATURE_DIM = 30
RMA_HEATMAP_SIZE = 14


def quaternion_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert batched wxyz quaternions to rotation matrices.

    The returned matrix maps local/camera-frame vectors into the parent frame.
    """
    if quaternion.ndim == 1:
        quaternion = quaternion.unsqueeze(0)
    quaternion = quaternion / torch.linalg.norm(quaternion, dim=-1, keepdim=True).clamp(min=1.0e-9)
    w, x, y, z = quaternion.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        [
            torch.stack([1.0 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w)], dim=-1),
            torch.stack([two * (x * y + z * w), 1.0 - two * (x * x + z * z), two * (y * z - x * w)], dim=-1),
            torch.stack([two * (x * z - y * w), two * (y * z + x * w), 1.0 - two * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def project_points_root_to_image(
    points_root: torch.Tensor,
    camera_position_root: torch.Tensor,
    camera_quaternion_wxyz: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
    image_width: int = 224,
    image_height: int = 224,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project robot-root points into ROS optical image pixels.

    Coordinates:
        points_root: [N,3] in robot-root frame.
        camera_position_root: [3] or [N,3] camera optical origin in the same frame.
        camera_quaternion_wxyz: [4] or [N,4] camera optical orientation in root frame.
        intrinsic_matrix: [3,3] or [N,3,3] for the final Student RGB image.

    ROS optical camera coordinates are x-right, y-down, z-forward.
    Returns uv [N,2] in 224x224 pixel coordinates and a valid [N] mask.
    """
    if points_root.ndim != 2 or points_root.shape[-1] != 3:
        raise ValueError(f"Expected points_root[N,3], got {tuple(points_root.shape)}")
    count = points_root.shape[0]
    if camera_position_root.ndim == 1:
        camera_position_root = camera_position_root.unsqueeze(0).expand(count, -1)
    if camera_quaternion_wxyz.ndim == 1:
        camera_quaternion_wxyz = camera_quaternion_wxyz.unsqueeze(0).expand(count, -1)
    if intrinsic_matrix.ndim == 2:
        intrinsic_matrix = intrinsic_matrix.unsqueeze(0).expand(count, -1, -1)

    rotation_root_camera = quaternion_wxyz_to_matrix(camera_quaternion_wxyz)
    relative_root = points_root - camera_position_root
    points_camera = torch.bmm(rotation_root_camera.transpose(1, 2), relative_root.unsqueeze(-1)).squeeze(-1)
    depth = points_camera[:, 2]
    safe_depth = depth.clamp(min=1.0e-6)
    u = intrinsic_matrix[:, 0, 0] * (points_camera[:, 0] / safe_depth) + intrinsic_matrix[:, 0, 2]
    v = intrinsic_matrix[:, 1, 1] * (points_camera[:, 1] / safe_depth) + intrinsic_matrix[:, 1, 2]
    uv = torch.stack([u, v], dim=-1)
    valid = (
        (depth > 1.0e-6)
        & (u >= 0.0)
        & (u <= float(image_width - 1))
        & (v >= 0.0)
        & (v <= float(image_height - 1))
    )
    return uv, valid


def make_gaussian_heatmaps(
    uv_pixels: torch.Tensor,
    valid: torch.Tensor,
    *,
    heatmap_height: int = RMA_HEATMAP_SIZE,
    heatmap_width: int = RMA_HEATMAP_SIZE,
    image_height: int = 224,
    image_width: int = 224,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Create [N,1,H,W] Gaussian heatmaps centered at image-space uv pixels."""
    if uv_pixels.ndim != 2 or uv_pixels.shape[-1] != 2:
        raise ValueError(f"Expected uv_pixels[N,2], got {tuple(uv_pixels.shape)}")
    device = uv_pixels.device
    dtype = uv_pixels.dtype
    x = torch.arange(heatmap_width, device=device, dtype=dtype).view(1, 1, 1, heatmap_width)
    y = torch.arange(heatmap_height, device=device, dtype=dtype).view(1, 1, heatmap_height, 1)
    center_x = (uv_pixels[:, 0] * float(heatmap_width) / float(image_width)).view(-1, 1, 1, 1)
    center_y = (uv_pixels[:, 1] * float(heatmap_height) / float(image_height)).view(-1, 1, 1, 1)
    sigma_value = max(float(sigma), 1.0e-6)
    heatmap = torch.exp(-((x - center_x).square() + (y - center_y).square()) / (2.0 * sigma_value * sigma_value))
    return heatmap * valid.to(dtype=dtype).view(-1, 1, 1, 1)


def heatmap_soft_argmax(
    heatmap: torch.Tensor,
    *,
    image_height: int = 224,
    image_width: int = 224,
) -> torch.Tensor:
    """Return predicted center uv [N,2] in image pixels from positive [N,1,H,W]."""
    if heatmap.ndim != 4 or heatmap.shape[1] != 1:
        raise ValueError(f"Expected heatmap[N,1,H,W], got {tuple(heatmap.shape)}")
    batch, _, height, width = heatmap.shape
    weights = heatmap.clamp(min=0.0).reshape(batch, height, width)
    probability = weights / weights.sum(dim=(1, 2), keepdim=True).clamp(min=1.0e-6)
    x = torch.arange(width, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, width)
    y = torch.arange(height, device=heatmap.device, dtype=heatmap.dtype).view(1, height, 1)
    u = (probability * x).sum(dim=(1, 2)) * float(image_width) / float(width)
    v = (probability * y).sum(dim=(1, 2)) * float(image_height) / float(height)
    return torch.stack([u, v], dim=-1)


class PandaFingertipKinematics(nn.Module):
    """Deployable Panda FK for the midpoint of the two fingertip centers."""

    def __init__(self) -> None:
        super().__init__()
        # Fixed transforms are copied from the Panda URDF used by the aligned
        # task. Each revolute joint rotates around its child-frame Z axis.
        joint_xyz = torch.tensor(
            [
                [0.0, 0.0, 0.333],
                [0.0, 0.0, 0.0],
                [0.0, -0.316, 0.0],
                [0.0825, 0.0, 0.0],
                [-0.0825, 0.384, 0.0],
                [0.0, 0.0, 0.0],
                [0.088, 0.0, 0.0],
            ]
        )
        joint_roll = torch.tensor(
            [
                0.0,
                -torch.pi / 2,
                torch.pi / 2,
                torch.pi / 2,
                -torch.pi / 2,
                torch.pi / 2,
                torch.pi / 2,
            ]
        )
        transforms = torch.eye(4).repeat(7, 1, 1)
        transforms[:, :3, 3] = joint_xyz
        cosine = torch.cos(joint_roll)
        sine = torch.sin(joint_roll)
        transforms[:, 1, 1] = cosine
        transforms[:, 1, 2] = -sine
        transforms[:, 2, 1] = sine
        transforms[:, 2, 2] = cosine
        self.register_buffer("joint_origin_transforms", transforms)
        # link7->link8 is 0.107 m. The aligned control/reward point is another
        # 0.1034 m along panda_hand Z. The intervening -45 degree hand yaw does
        # not change a point on that Z axis.
        self.register_buffer(
            "link7_to_fingertip_midpoint",
            torch.tensor([0.0, 0.0, 0.2104, 1.0]),
        )

    @staticmethod
    def _rotation_z(angle: torch.Tensor) -> torch.Tensor:
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        zero = torch.zeros_like(angle)
        one = torch.ones_like(angle)
        return torch.stack(
            [
                torch.stack([cosine, -sine, zero, zero], dim=-1),
                torch.stack([sine, cosine, zero, zero], dim=-1),
                torch.stack([zero, zero, one, zero], dim=-1),
                torch.stack([zero, zero, zero, one], dim=-1),
            ],
            dim=-2,
        )

    def forward(self, joint_position: torch.Tensor) -> torch.Tensor:
        # RMA Teacher/Student input is always batched [N, 7]. Keeping this
        # explicit avoids a variadic reshape that TorchScript cannot script.
        if not torch.jit.is_scripting():
            if joint_position.ndim != 2 or joint_position.shape[-1] != 7:
                raise ValueError(f"Expected Panda joint_position[N,7], got {tuple(joint_position.shape)}")
        transform = torch.eye(4, dtype=joint_position.dtype, device=joint_position.device)
        transform = transform.unsqueeze(0).expand(joint_position.shape[0], -1, -1)
        for index in range(7):
            transform = torch.matmul(transform, self.joint_origin_transforms[index])
            transform = torch.matmul(transform, self._rotation_z(joint_position[:, index]))
        point = torch.matmul(transform, self.link7_to_fingertip_midpoint)
        return point[:, :3]

    def contract(self) -> dict[str, object]:
        return {
            "source": "panda_urdf_joint_origins",
            "joint_order": [f"panda_joint{index}" for index in range(1, 8)],
            "joint_axis": "local_z",
            "output": "mean_of_left_and_right_fingertip_centers",
            "output_frame": "robot_root",
            "panda_link7_to_link8_m": 0.107,
            "panda_hand_to_fingertip_midpoint_m": 0.1034,
        }


class RMAObservationNormalizer(nn.Module):
    """Fixed, deployable normalization shared by teacher and student."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "joint_lower",
            torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
        )
        self.register_buffer(
            "joint_upper",
            torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
        )
        self.register_buffer(
            "joint_velocity_scale",
            torch.tensor([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]),
        )
        self.register_buffer("history_scale", torch.tensor([0.025, 0.025, 0.025, 0.005]))
        self.register_buffer("cube_position_center", torch.tensor([0.50, 0.00, 0.026]))
        self.register_buffer("cube_position_scale", torch.tensor([0.05, 0.05, 0.10]))
        self.register_buffer("gripper_position_center", torch.tensor([0.50, 0.00, 0.175]))
        self.register_buffer("gripper_position_scale", torch.tensor([0.10, 0.10, 0.15]))
        self.register_buffer("target_position_center", torch.tensor([0.0, 0.0, -0.15]))
        self.register_buffer("target_position_scale", torch.tensor([0.10, 0.10, 0.15]))

    def normalize_proprio(self, proprio_obs: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            if proprio_obs.shape[-1] != RMA_PROPRIO_DIM:
                raise ValueError(f"Expected proprio_obs[...,15], got {tuple(proprio_obs.shape)}")
        joint_mid = 0.5 * (self.joint_lower + self.joint_upper)
        joint_half_range = 0.5 * (self.joint_upper - self.joint_lower)
        joint_pos = (proprio_obs[..., :7] - joint_mid) / joint_half_range
        joint_vel = proprio_obs[..., 7:14] / self.joint_velocity_scale
        gripper_width = (proprio_obs[..., 14:15] - 0.04) / 0.04
        return torch.cat([joint_pos, joint_vel, gripper_width], dim=-1)

    def normalize_history(self, action_history: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            if action_history.shape[-1] != RMA_HISTORY_DIM:
                raise ValueError(f"Expected action_history[...,4], got {tuple(action_history.shape)}")
        return action_history / self.history_scale

    def normalize_position(self, cube_position: torch.Tensor) -> torch.Tensor:
        return (cube_position - self.cube_position_center) / self.cube_position_scale

    def denormalize_position(self, normalized_position: torch.Tensor) -> torch.Tensor:
        return normalized_position * self.cube_position_scale + self.cube_position_center

    def normalize_gripper_position(self, gripper_position: torch.Tensor) -> torch.Tensor:
        return (gripper_position - self.gripper_position_center) / self.gripper_position_scale

    def normalize_target_position(self, target_position: torch.Tensor) -> torch.Tensor:
        return (target_position - self.target_position_center) / self.target_position_scale

    def contract(self) -> dict[str, list[float] | str]:
        return {
            "joint_lower": self.joint_lower.detach().cpu().tolist(),
            "joint_upper": self.joint_upper.detach().cpu().tolist(),
            "joint_velocity_scale": self.joint_velocity_scale.detach().cpu().tolist(),
            "gripper_width_center_scale": [0.04, 0.04],
            "history_scale": self.history_scale.detach().cpu().tolist(),
            "cube_position_center": self.cube_position_center.detach().cpu().tolist(),
            "cube_position_scale": self.cube_position_scale.detach().cpu().tolist(),
            "gripper_position_center": self.gripper_position_center.detach().cpu().tolist(),
            "gripper_position_scale": self.gripper_position_scale.detach().cpu().tolist(),
            "target_position_center": self.target_position_center.detach().cpu().tolist(),
            "target_position_scale": self.target_position_scale.detach().cpu().tolist(),
            "position_frame": "robot_root",
        }


class RMAActorCore(nn.Module):
    """Teacher Actor mean shared unchanged by visual student inference."""

    def __init__(self) -> None:
        super().__init__()
        self.normalizer = RMAObservationNormalizer()
        self.kinematics = PandaFingertipKinematics()
        self.network = nn.Sequential(
            nn.Linear(RMA_ACTOR_FEATURE_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, RMA_ACTION_DIM),
        )

    def forward(
        self,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        cube_position: torch.Tensor,
        contact_state: torch.Tensor,
        position_is_normalized: bool = False,
    ) -> torch.Tensor:
        if not torch.jit.is_scripting():
            if contact_state.shape[-1] != RMA_CONTACT_DIM:
                raise ValueError(f"Expected contact_state[...,2], got {tuple(contact_state.shape)}")
        if position_is_normalized:
            normalized_cube_position = cube_position
            cube_position_root = self.normalizer.denormalize_position(cube_position)
        else:
            cube_position_root = cube_position
            normalized_cube_position = self.normalizer.normalize_position(cube_position)
        gripper_position_root = self.kinematics(proprio_obs[..., :7])
        target_position_root = cube_position_root - gripper_position_root
        features = torch.cat(
            [
                self.normalizer.normalize_proprio(proprio_obs),
                self.normalizer.normalize_history(action_history),
                normalized_cube_position,
                self.normalizer.normalize_gripper_position(gripper_position_root),
                self.normalizer.normalize_target_position(target_position_root),
                contact_state,
            ],
            dim=-1,
        )
        return torch.tanh(self.network(features))

    def contract(self) -> dict[str, object]:
        return {
            "feature_dim": RMA_ACTOR_FEATURE_DIM,
            "feature_order": [
                "normalized_proprio_obs[15]",
                "normalized_action_history[4]",
                "normalized_cube_position_root[3]",
                "normalized_gripper_position_root_from_fk[3]",
                "normalized_cube_minus_gripper_position_root[3]",
                "left_right_cube_finger_contact[2]",
            ],
            "object_pose_components": "position_xyz_only",
            "end_effector_pose_components": "position_xyz_only",
            "contact_components": "left_right_cube_finger_binary",
            "kinematics": self.kinematics.contract(),
        }


class RMAPrivilegedTeacherPolicy(GaussianMixin, Model):
    """skrl Gaussian policy backed by the deployable RMAActorCore."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
        initial_log_std: float = 0.0,
        fixed_log_std: bool = False,
        **_: object,
    ) -> None:
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )
        if self.num_actions != RMA_ACTION_DIM:
            raise ValueError(f"RMA teacher requires 4 actions, got {self.num_actions}")
        self.actor_core = RMAActorCore().to(device)
        self.log_std_parameter = nn.Parameter(
            torch.full((self.num_actions,), float(initial_log_std), device=device),
            requires_grad=not fixed_log_std,
        )

    def compute(self, inputs: Mapping[str, torch.Tensor], role: str = ""):
        del role
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        mean = self.actor_core(
            states["proprio_obs"],
            states["action_history"],
            states["rma_cube_pos"],
            states["rma_contact_state"],
        )
        return mean, self.log_std_parameter, {}


class RMAPPO(PPO):
    """Named PPO alias that selects the repository's custom-model training path."""


class SpatialSoftmaxAdaptationHead(nn.Module):
    """Predict normalized cube XYZ and left/right contact logits."""

    def __init__(self, channels: int = 32, height: int = 7, width: int = 7) -> None:
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.keypoint_conv = nn.Conv2d(512, channels, kernel_size=1)
        x_coords = torch.linspace(-1.0, 1.0, width).repeat(height)
        y_coords = torch.linspace(-1.0, 1.0, height).repeat_interleave(width)
        self.register_buffer("x_coords", x_coords)
        self.register_buffer("y_coords", y_coords)
        self.trunk = nn.Sequential(
            nn.Linear(2 * channels, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.position_output = nn.Linear(64, RMA_POSITION_DIM)
        self.contact_output = nn.Linear(64, RMA_CONTACT_DIM)
        nn.init.zeros_(self.position_output.weight)
        nn.init.zeros_(self.position_output.bias)
        nn.init.zeros_(self.contact_output.weight)
        nn.init.constant_(self.contact_output.bias, -2.0)

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.keypoint_conv(feature_map).reshape(feature_map.shape[0], self.channels, -1)
        attention = torch.softmax(logits, dim=-1)
        expected_x = torch.sum(attention * self.x_coords, dim=-1)
        expected_y = torch.sum(attention * self.y_coords, dim=-1)
        keypoints = torch.cat([expected_x, expected_y], dim=-1)
        features = self.trunk(keypoints)
        return torch.tanh(self.position_output(features)), self.contact_output(features)


class CubeCenterHeatmapHead(nn.Module):
    """Predict a 14x14 cube-center heatmap from ResNet18 layer3 features."""

    def __init__(self, hidden_channels: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(256, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, layer3_feature: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(layer3_feature))


class RMAVisualStudent(nn.Module):
    """Frozen ResNet18 + trainable localizer + frozen teacher Actor."""

    def __init__(self, actor_core: RMAActorCore, *, pretrained_backbone: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        self.vision_encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.adaptation_head = SpatialSoftmaxAdaptationHead()
        self.heatmap_head = CubeCenterHeatmapHead()
        self.actor_core = actor_core
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.freeze_fixed_modules()

    def freeze_fixed_modules(self) -> None:
        self.vision_encoder.eval()
        self.actor_core.eval()
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.actor_core.parameters():
            parameter.requires_grad_(False)

    @torch.jit.unused
    def unfreeze_backbone_after_layer2(self) -> None:
        """Train ResNet18 layer3/layer4 while keeping layer2 and earlier frozen.

        ``vision_encoder`` is ResNet18 up to layer4:
        0=conv1, 1=bn1, 2=relu, 3=maxpool, 4=layer1, 5=layer2,
        6=layer3, 7=layer4.  BatchNorm modules stay in eval mode through
        ``train()``, so this only enables gradients on convolution weights.
        """
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad_(False)
        for index, module in enumerate(self.vision_encoder):
            if index >= 6:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        self.vision_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_encoder.eval()
        self.actor_core.eval()
        return self

    def encode(self, wrist_rgb: torch.Tensor) -> torch.Tensor:
        _, layer4_feature = self.encode_features(wrist_rgb)
        return layer4_feature

    def encode_features(self, wrist_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.jit.is_scripting():
            if wrist_rgb.ndim != 4 or wrist_rgb.shape[-1] != 3:
                raise ValueError(f"Expected wrist_rgb [N,H,W,3], got {tuple(wrist_rgb.shape)}")
        image = wrist_rgb.to(torch.float32).permute(0, 3, 1, 2) / 255.0
        image = (image - self.image_mean) / self.image_std
        x = image
        layer3_feature = image
        for index, module in enumerate(self.vision_encoder):
            x = module(x)
            if index == 6:
                layer3_feature = x
        return layer3_feature, x

    def predict_normalized_position(self, wrist_rgb: torch.Tensor) -> torch.Tensor:
        normalized_position, _ = self.predict_adaptation(wrist_rgb)
        return normalized_position

    def predict_adaptation(self, wrist_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.adaptation_head(self.encode(wrist_rgb))

    def predict_adaptation_and_heatmap(
        self, wrist_rgb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer3_feature, layer4_feature = self.encode_features(wrist_rgb)
        normalized_position, contact_logits = self.adaptation_head(layer4_feature)
        heatmap = self.heatmap_head(layer3_feature)
        return normalized_position, contact_logits, heatmap

    def predict_contact_probability(self, wrist_rgb: torch.Tensor) -> torch.Tensor:
        _, contact_logits = self.predict_adaptation(wrist_rgb)
        return torch.sigmoid(contact_logits)

    def predict_position(self, wrist_rgb: torch.Tensor) -> torch.Tensor:
        normalized = self.predict_normalized_position(wrist_rgb)
        return self.actor_core.normalizer.denormalize_position(normalized)

    def action_from_normalized_position(
        self,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        normalized_position: torch.Tensor,
        contact_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.actor_core(
            proprio_obs,
            action_history,
            normalized_position,
            contact_state,
            position_is_normalized=True,
        )

    def forward(
        self,
        wrist_rgb: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        normalized_position, contact_logits = self.predict_adaptation(wrist_rgb)
        return self.action_from_normalized_position(
            proprio_obs,
            action_history,
            normalized_position,
            torch.sigmoid(contact_logits),
        )


def extract_actor_core_state_dict(policy_state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract RMAActorCore weights from a skrl policy checkpoint."""
    prefix = "actor_core."
    state = {
        key[len(prefix) :]: value
        for key, value in policy_state_dict.items()
        if key.startswith(prefix)
    }
    if not state:
        raise RuntimeError("Teacher checkpoint policy has no actor_core.* parameters")
    return state
