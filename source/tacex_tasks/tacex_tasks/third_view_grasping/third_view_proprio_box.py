"""Third-view + proprio policy variant of the third-view grasping task."""

from __future__ import annotations

import torch
import torch.nn as nn

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, TiledCamera
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from .privileged_box import OccludedGraspingPrivilegedBoxCfg, OccludedGraspingPrivilegedBoxEnv
from .vt_box import CAN_RESET_ROOT_Z, sample_uniform

try:
    import torchvision

    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


@configclass
class OccludedGraspingThirdViewProprioBoxCfg(OccludedGraspingPrivilegedBoxCfg):
    """Configuration for third-view camera + proprio policy observations."""

    observation_space = {
        "proprio_obs": 18,
        "third_resnet": 512,
        "critic_can_pos": 3,
        "critic_can_quat": 4,
        "critic_can_lin_vel": 3,
        "critic_can_ang_vel": 3,
        "critic_gripper_pos": 3,
        "critic_gripper_quat": 4,
        "critic_gripper_lin_vel": 3,
        "critic_gripper_ang_vel": 3,
        "critic_target_pos": 3,
        "critic_target_distance": 1,
    }


class OccludedGraspingThirdViewProprioBoxEnv(OccludedGraspingPrivilegedBoxEnv):
    """Occluded grasping environment with third-view camera + proprio policy inputs."""

    cfg: OccludedGraspingThirdViewProprioBoxCfg

    def __init__(self, cfg: OccludedGraspingThirdViewProprioBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._use_resnet18 = _HAS_TORCHVISION
        self._third_feature_dim = 512
        if self._use_resnet18:
            try:
                weights = torchvision.models.ResNet18_Weights.DEFAULT
                backbone = torchvision.models.resnet18(weights=weights)
            except Exception:
                backbone = torchvision.models.resnet18(pretrained=True)
            self._third_encoder = nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
            self._imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self._imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        else:
            self._third_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, self._third_feature_dim),
                nn.ReLU(inplace=True),
            ).to(self.device)
        self._third_encoder.eval()
        for p in self._third_encoder.parameters():
            p.requires_grad_(False)

    def _setup_scene(self):
        """Setup a privileged scene with only the third-person camera sensor."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._can = RigidObject(self.cfg.can)
        self.scene.rigid_objects["can"] = self._can

        self._box_floor = RigidObject(self.cfg.box_floor)
        self.scene.rigid_objects["box_floor"] = self._box_floor
        self._box_wall_front = RigidObject(self.cfg.box_wall_front)
        self.scene.rigid_objects["box_wall_front"] = self._box_wall_front
        self._box_wall_back = RigidObject(self.cfg.box_wall_back)
        self.scene.rigid_objects["box_wall_back"] = self._box_wall_back
        self._box_wall_left = RigidObject(self.cfg.box_wall_left)
        self.scene.rigid_objects["box_wall_left"] = self._box_wall_left
        self._box_wall_right = RigidObject(self.cfg.box_wall_right)
        self.scene.rigid_objects["box_wall_right"] = self._box_wall_right

        self.scene.clone_environments(copy_from_source=False)
        self.third_person_camera = TiledCamera(self.cfg.third_person_camera)
        self.scene.sensors["third_person_camera"] = self.third_person_camera

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.11841)),
                ),
            ],
        )
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _initialize_can_positions(self):
        """Initialize can positions inside configured reset bounds."""
        can_state = self._can.data.default_root_state.clone()
        x_min, x_max, y_min, y_max = self._get_can_reset_xy_bounds()
        can_state[:, 0] = sample_uniform(x_min, x_max, (self.num_envs,), self.device)
        can_state[:, 1] = sample_uniform(y_min, y_max, (self.num_envs,), self.device)
        can_state[:, 2] = CAN_RESET_ROOT_Z
        z_jitter = max(float(getattr(self.cfg, "can_reset_z_jitter", 0.0)), 0.0)
        can_state[:, 2] += sample_uniform(-z_jitter, z_jitter, (self.num_envs,), self.device)
        rand_yaw = sample_uniform(-self.cfg.can_rot_range, self.cfg.can_rot_range, (self.num_envs,), self.device)
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        can_state[:, 3:7] = rand_quat
        env_ids = torch.arange(self.num_envs, device=self.device)
        can_state[:, :3] += self.scene.env_origins
        self._can.write_root_state_to_sim(can_state, env_ids)
        self._can.write_root_velocity_to_sim(torch.zeros((self.num_envs, 6), device=self.device), env_ids)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)

        can_pos = self._can.data.root_pos_w
        can_quat = self._can.data.root_quat_w
        can_lin_vel = self._can.data.root_lin_vel_w
        can_ang_vel = self._can.data.root_ang_vel_w

        gripper_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        gripper_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        gripper_lin_vel = self._robot.data.body_link_lin_vel_w[:, self._body_idx]
        gripper_ang_vel = self._robot.data.body_link_ang_vel_w[:, self._body_idx]

        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, self._offset_rot
        )
        target_pos_relative = can_pos - ee_pos
        target_distance = torch.norm(target_pos_relative, dim=-1, keepdim=True)

        third_rgb = self.third_person_camera.data.output.get("rgb")
        if third_rgb is None:
            third_feat = torch.zeros((self.num_envs, self._third_feature_dim), device=self.device)
        else:
            x = third_rgb.to(device=self.device, dtype=torch.float32)
            max_val = x.max()
            if torch.isfinite(max_val) and max_val > 1.5:
                x = x / 255.0
            x = x.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()
            if self._use_resnet18:
                x = (x - self._imagenet_mean) / self._imagenet_std
            with torch.no_grad():
                third_feat = self._third_encoder(x).view(self.num_envs, self._third_feature_dim)

        obs = {
            "proprio_obs": proprio_obs,
            "third_resnet": third_feat,
            "critic_can_pos": can_pos,
            "critic_can_quat": can_quat,
            "critic_can_lin_vel": can_lin_vel,
            "critic_can_ang_vel": can_ang_vel,
            "critic_gripper_pos": gripper_pos,
            "critic_gripper_quat": gripper_quat,
            "critic_gripper_lin_vel": gripper_lin_vel,
            "critic_gripper_ang_vel": gripper_ang_vel,
            "critic_target_pos": target_pos_relative,
            "critic_target_distance": target_distance,
        }
        return {"policy": obs}
