"""Occluded grasping with third-person vision, inner tactile RGB, and down tactile depth."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

from ..sparch_grasp.sparsh_encoder import SparshFrozenEncoder, project_sparsh_features
from .vt_box import OccludedGraspingVisionFourTactileBoxCfg, OccludedGraspingVisionFourTactileBoxEnv
from .vt_sparsh import _resolve_sparsh_encoder_spec


@configclass
class OccludedGraspingVisionMixedTactileBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Third-person vision with inner tactile RGB and down tactile depth."""

    tactile_frame_stack = 5
    sparsh_repo_path = "/home/tinydog/Projects/sparsh"
    sparsh_encoder_name = "dino_vitsmall"
    sparsh_checkpoint_path = ""
    sparsh_model_size = ""
    sparsh_encoder_type = ""
    sparsh_encoder_summary = ""
    sparsh_num_register_tokens = 0
    sparsh_encoder_embed_dim = 384
    sparsh_feature_dim = 256
    depth_feature_dim = 256
    tactile_baseline_diff_threshold = 0.05
    tactile_contact_area_threshold = 0.01
    sparsh_baseline_diff_threshold = 0.02

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.down.usda"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.4693,
                "panda_joint2": -0.1410,
                "panda_joint3": 0.4929,
                "panda_joint4": -2.1960,
                "panda_joint5": 0.0312,
                "panda_joint6": 2.0935,
                "panda_joint7": 0.7937,
                "panda_finger_joint.*": 0.02,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    gsmini_left: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
    )
    gsmini_left.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=0,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_left.data_types = ["tactile_rgb", "camera_depth"]
    gsmini_left.marker_motion_sim_cfg = None
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        tactile_img_res=(128, 96)
    )

    gsmini_right: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )
    gsmini_right.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=0,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_right.data_types = ["tactile_rgb", "camera_depth"]
    gsmini_right.marker_motion_sim_cfg = None
    gsmini_right.optical_sim_cfg = gsmini_right.optical_sim_cfg.replace(
        tactile_img_res=(128, 96)
    )

    gsmini_left_down: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left_down",
    )
    gsmini_left_down.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=0,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_left_down.data_types = ["camera_depth"]
    gsmini_left_down.marker_motion_sim_cfg = None

    gsmini_right_down: GelSightMiniCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right_down",
    )
    gsmini_right_down.sensor_camera_cfg = GelSightMiniCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=0,
        resolution=(128, 96),
        data_types=["depth"],
        clipping_range=(0.024, 0.034),
    )
    gsmini_right_down.data_types = ["camera_depth"]
    gsmini_right_down.marker_motion_sim_cfg = None

    def __post_init__(self):
        super().__post_init__()
        resolved = _resolve_sparsh_encoder_spec(
            encoder_name=self.sparsh_encoder_name,
            checkpoint_path=self.sparsh_checkpoint_path,
            model_size=self.sparsh_model_size,
            encoder_type=self.sparsh_encoder_type,
        )
        self.sparsh_encoder_name = str(resolved["name"])
        self.sparsh_checkpoint_path = str(resolved["checkpoint_path"])
        self.sparsh_model_size = str(resolved["model_size"])
        self.sparsh_encoder_type = str(resolved["encoder_type"])
        self.sparsh_encoder_embed_dim = int(resolved["feature_dim"])
        self.sparsh_feature_dim = int(self.sparsh_feature_dim)
        self.depth_feature_dim = int(self.depth_feature_dim)
        if self.sparsh_feature_dim <= 0:
            raise ValueError(f"sparsh_feature_dim must be positive, got {self.sparsh_feature_dim}")
        if self.depth_feature_dim <= 0:
            raise ValueError(f"depth_feature_dim must be positive, got {self.depth_feature_dim}")
        if self.depth_feature_dim != self.sparsh_feature_dim:
            raise ValueError(
                "Mixed tactile env expects matching inner/down feature dims, "
                f"got sparsh_feature_dim={self.sparsh_feature_dim} and depth_feature_dim={self.depth_feature_dim}"
            )
        self.sparsh_encoder_summary = (
            f"{self.sparsh_encoder_name} | type={self.sparsh_encoder_type} | "
            f"model={self.sparsh_model_size} | encoder_dim={self.sparsh_encoder_embed_dim} | "
            f"out_dim={self.sparsh_feature_dim} | "
            f"ckpt={self.sparsh_checkpoint_path}"
        )
        self.observation_space = {
            "proprio_obs": 18,
            "third_resnet": 256,
            "tactile_left_rgb": self.sparsh_feature_dim * int(self.tactile_frame_stack),
            "tactile_right_rgb": self.sparsh_feature_dim * int(self.tactile_frame_stack),
            "tactile_left_down_depth": self.depth_feature_dim * int(self.tactile_frame_stack),
            "tactile_right_down_depth": self.depth_feature_dim * int(self.tactile_frame_stack),
            "tactile_contact_bits": 4,
            "tactile_valid": 1,
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


class OccludedGraspingVisionMixedTactileBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Box env with RGB inner tactile sensors and depth down tactile sensors."""

    cfg: OccludedGraspingVisionMixedTactileBoxCfg

    def __init__(self, cfg: OccludedGraspingVisionMixedTactileBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if hasattr(self, "_tactile_encoder"):
            del self._tactile_encoder

        self._tactile_feature_dim = int(self.cfg.sparsh_feature_dim)
        self._inner_sparsh_encoder_name = str(self.cfg.sparsh_encoder_name)
        self._inner_sparsh_encoder_summary = str(self.cfg.sparsh_encoder_summary)
        self._inner_sparsh_encoder = SparshFrozenEncoder(
            repo_path=self.cfg.sparsh_repo_path,
            checkpoint_path=self.cfg.sparsh_checkpoint_path,
            img_size_hw=tuple(getattr(self.cfg, "tactile_img_res_hw", (96, 128))),
            model_size=self.cfg.sparsh_model_size,
            encoder_type=self.cfg.sparsh_encoder_type,
            num_register_tokens=int(getattr(self.cfg, "sparsh_num_register_tokens", 0)),
        ).to(self.device)
        self._inner_sparsh_encoder.eval()
        self._inner_sparsh_feature_dim = int(self.cfg.sparsh_feature_dim)
        print(f"[INFO] Mixed inner tactile encoder: {self._inner_sparsh_encoder_summary}")

        self._down_depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, int(self.cfg.depth_feature_dim)),
            nn.ReLU(inplace=True),
        ).to(self.device)
        self._down_depth_encoder.eval()
        for parameter in self._down_depth_encoder.parameters():
            parameter.requires_grad_(False)
        print(f"[INFO] Mixed down tactile encoder: depth-cnn (feature_dim={int(self.cfg.depth_feature_dim)})")

        self._tactile_baseline_depth: dict[str, torch.Tensor | None] = {
            "left": None,
            "right": None,
            "left_down": None,
            "right_down": None,
        }
        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))
        self._inner_sparsh_sensor_keys = ("left", "right")
        self._inner_tactile_baseline_rgb: dict[str, torch.Tensor | None] = {
            key: torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
            for key in self._inner_sparsh_sensor_keys
        }
        self._inner_tactile_prev_rgb: dict[str, torch.Tensor] = {
            key: torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
            for key in self._inner_sparsh_sensor_keys
        }
        self._latest_tactile_contact_ratios: dict[str, torch.Tensor] = {
            "left": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
            "right": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
            "left_down": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
            "right_down": torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device),
        }
        self._tactile_frame_stack = max(1, int(getattr(self.cfg, "tactile_frame_stack", 5)))
        self._tactile_feature_history: dict[str, torch.Tensor] = {
            "left": torch.zeros(
                (self.num_envs, self._tactile_frame_stack, self._tactile_feature_dim),
                dtype=torch.float32,
                device=self.device,
            ),
            "right": torch.zeros(
                (self.num_envs, self._tactile_frame_stack, self._tactile_feature_dim),
                dtype=torch.float32,
                device=self.device,
            ),
            "left_down": torch.zeros(
                (self.num_envs, self._tactile_frame_stack, self._tactile_feature_dim),
                dtype=torch.float32,
                device=self.device,
            ),
            "right_down": torch.zeros(
                (self.num_envs, self._tactile_frame_stack, self._tactile_feature_dim),
                dtype=torch.float32,
                device=self.device,
            ),
        }
        self._tactile_contact_history = torch.zeros(
            (self.num_envs, self._tactile_frame_stack, 4),
            dtype=torch.float32,
            device=self.device,
        )
        self._pending_tactile_baseline_refresh = torch.ones(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._pending_inner_sparsh_baseline_refresh = torch.ones(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )

    def _normalize_depth_01(self, depth_tensor: torch.Tensor | None) -> torch.Tensor | None:
        """Convert camera-depth NHWC1 tensor into float32 [0, 1] with shape NHWC1."""
        if depth_tensor is None:
            return None
        depth = depth_tensor.to(device=self.device, dtype=torch.float32)
        max_val = depth.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            depth = depth / 255.0
        depth = depth.clamp(0.0, 1.0)
        if depth.ndim == 3:
            depth = depth.unsqueeze(-1)
        return depth

    def _maybe_refresh_tactile_baselines(self, sensor_depths: dict[str, torch.Tensor | None]) -> None:
        """Refresh per-env tactile baseline depth images after reset."""
        pending = self._pending_tactile_baseline_refresh
        if not torch.any(pending):
            return
        pending_ids = pending.nonzero(as_tuple=False).squeeze(-1)
        if pending_ids.numel() == 0:
            return

        captured_any = False
        for sensor_name, depth_tensor in sensor_depths.items():
            depth_01 = self._normalize_depth_01(depth_tensor)
            if depth_01 is None:
                continue
            baseline = self._tactile_baseline_depth.get(sensor_name)
            if baseline is None or baseline.shape != depth_01.shape:
                baseline = torch.zeros_like(depth_01)
                self._tactile_baseline_depth[sensor_name] = baseline
            baseline[pending_ids] = depth_01[pending_ids]
            captured_any = True

        if captured_any:
            pending[pending_ids] = False

    def _prep_rgb_01(self, rgb_tensor: torch.Tensor | None) -> torch.Tensor:
        """Convert tactile RGB NHWC3 into normalized NCHW3."""
        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))
        if rgb_tensor is None:
            return torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
        rgb = rgb_tensor.to(device=self.device, dtype=torch.float32)
        max_val = rgb.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            rgb = rgb / 255.0
        rgb = rgb.clamp(0.0, 1.0)
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(-1)
        rgb_nchw = rgb.permute(0, 3, 1, 2).contiguous()
        if rgb_nchw.shape[1] == 1:
            rgb_nchw = rgb_nchw.repeat(1, 3, 1, 1)
        elif rgb_nchw.shape[1] != 3:
            rgb_nchw = rgb_nchw[:, :3, :, :]
        if rgb_nchw.shape[2] != target_h or rgb_nchw.shape[3] != target_w:
            rgb_nchw = F.interpolate(rgb_nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return rgb_nchw

    def _maybe_refresh_inner_sparsh_baselines(self, sensor_rgbs: dict[str, torch.Tensor]) -> None:
        """Refresh per-env inner tactile RGB baselines after reset."""
        pending = self._pending_inner_sparsh_baseline_refresh
        if not torch.any(pending):
            return
        pending_ids = pending.nonzero(as_tuple=False).squeeze(-1)
        if pending_ids.numel() == 0:
            return
        for sensor_name, rgb_tensor in sensor_rgbs.items():
            baseline = self._inner_tactile_baseline_rgb[sensor_name]
            baseline[pending_ids] = rgb_tensor[pending_ids]
            self._inner_tactile_prev_rgb[sensor_name][pending_ids] = rgb_tensor[pending_ids]
        pending[pending_ids] = False

    def _prep_depth_1ch(self, depth_tensor: torch.Tensor | None) -> torch.Tensor:
        """Convert camera-depth NHWC1 into normalized NCHW1 for the down depth encoder."""
        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))
        depth = self._normalize_depth_01(depth_tensor)
        if depth is None:
            return torch.zeros((self.num_envs, 1, target_h, target_w), dtype=torch.float32, device=self.device)
        depth_nchw = depth.permute(0, 3, 1, 2).contiguous()
        if depth_nchw.shape[1] != 1:
            depth_nchw = depth_nchw.mean(dim=1, keepdim=True)
        if depth_nchw.shape[2] != target_h or depth_nchw.shape[3] != target_w:
            depth_nchw = F.interpolate(depth_nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return depth_nchw

    def _encode_inner_sparsh(self, tactile_raw: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor]:
        """Encode inner tactile RGB with a shared Sparsh DINO encoder."""
        tactile_rgb = {key: self._prep_rgb_01(value) for key, value in tactile_raw.items()}
        self._maybe_refresh_inner_sparsh_baselines(tactile_rgb)

        ready_mask = (~self._pending_inner_sparsh_baseline_refresh).clone()
        current_frames = {
            key: torch.zeros_like(tactile_rgb[key], dtype=torch.float32, device=self.device)
            for key in self._inner_sparsh_sensor_keys
        }
        for key in self._inner_sparsh_sensor_keys:
            if torch.any(ready_mask):
                current_frames[key][ready_mask] = tactile_rgb[key][ready_mask]

        features = {
            key: torch.zeros((self.num_envs, self._inner_sparsh_feature_dim), device=self.device, dtype=torch.float32)
            for key in self._inner_sparsh_sensor_keys
        }
        if torch.any(ready_mask):
            inner_batch = torch.cat(
                [
                    torch.cat([current_frames[key][ready_mask], self._inner_tactile_prev_rgb[key][ready_mask]], dim=1)
                    for key in self._inner_sparsh_sensor_keys
                ],
                dim=0,
            )
            dev = self.device
            dev_type = getattr(dev, "type", None)
            if dev_type is None:
                dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
            use_amp = dev_type == "cuda"
            with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
                encoded = self._inner_sparsh_encoder(inner_batch).mean(dim=1).float()
            encoded = project_sparsh_features(encoded, self._inner_sparsh_feature_dim)
            split_encoded = encoded.chunk(len(self._inner_sparsh_sensor_keys), dim=0)
            for key, feat in zip(self._inner_sparsh_sensor_keys, split_encoded):
                features[key][ready_mask] = feat
                self._inner_tactile_prev_rgb[key][ready_mask] = current_frames[key][ready_mask]

        if torch.any(~ready_mask):
            for key in self._inner_sparsh_sensor_keys:
                self._inner_tactile_prev_rgb[key][~ready_mask] = 0.0
        return features

    def _get_base_policy_observations_without_tactile(self) -> dict[str, torch.Tensor]:
        """Build all non-tactile observations, matching the base task."""
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
        hand_quat = hand_quat / torch.linalg.norm(hand_quat, dim=-1, keepdim=True).clamp(min=1e-9)
        offset_rot = self._offset_rot / torch.linalg.norm(self._offset_rot, dim=-1, keepdim=True).clamp(min=1e-9)
        ee_pos, _ = math_utils.combine_frame_transforms(hand_pos, hand_quat, self._offset_pos, offset_rot)
        target_pos_relative = can_pos - ee_pos
        target_distance = torch.norm(target_pos_relative, dim=-1, keepdim=True)

        third_rgb = self.third_person_camera.data.output.get("rgb")
        if third_rgb is None:
            third_rgb = torch.zeros(
                (self.num_envs, self.cfg.third_person_camera.height, self.cfg.third_person_camera.width, 3),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            third_rgb = third_rgb.to(device=self.device, dtype=torch.float32) / 255.0

        dev = self.device
        dev_type = getattr(dev, "type", None)
        if dev_type is None:
            dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
        use_amp = dev_type == "cuda"

        if hasattr(self, "_use_resnet18") and self._use_resnet18:
            xt = third_rgb.permute(0, 3, 1, 2).contiguous().to(self.device)
            if hasattr(self, "_imagenet_mean"):
                xt = (xt - self._imagenet_mean) / self._imagenet_std
            with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
                third_feat = self._resnet18(xt).view(self.num_envs, 256)
        else:
            third_feat = torch.zeros((self.num_envs, 256), device=self.device)

        return {
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

    def _contact_ratio_from_depth(self, sensor_name: str, depth_tensor: torch.Tensor | None) -> torch.Tensor:
        """Compute per-env changed-pixel ratio from baseline-depth differences."""
        depth = self._normalize_depth_01(depth_tensor)
        baseline = self._tactile_baseline_depth.get(sensor_name)
        if depth is None or baseline is None:
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        diff = torch.abs(depth - baseline)
        diff_threshold = float(getattr(self.cfg, "tactile_baseline_diff_threshold", 0.05))
        return (diff >= diff_threshold).to(torch.float32).mean(dim=(1, 2, 3), keepdim=False)

    def _contact_bits_from_ratio(self, contact_ratio: torch.Tensor) -> torch.Tensor:
        """Convert changed-pixel ratio into a binary contact bit."""
        area_threshold = float(getattr(self.cfg, "tactile_contact_area_threshold", 0.01))
        return (contact_ratio > area_threshold).to(torch.float32).unsqueeze(-1)

    def _push_tactile_feature(self, sensor_name: str, feature: torch.Tensor) -> torch.Tensor:
        """Append current tactile feature into per-env history and return flattened frame stack."""
        history = self._tactile_feature_history[sensor_name]
        history[:, :-1, :] = history[:, 1:, :].clone()
        history[:, -1, :] = feature.to(dtype=torch.float32)
        return history.reshape(self.num_envs, -1)

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        obs = self._get_base_policy_observations_without_tactile()

        tact_l_raw = self.gsmini_left.data.output.get("tactile_rgb") if hasattr(self, "gsmini_left") else None
        tact_r_raw = self.gsmini_right.data.output.get("tactile_rgb") if hasattr(self, "gsmini_right") else None
        tact_l_depth_raw = self.gsmini_left.data.output.get("camera_depth") if hasattr(self, "gsmini_left") else None
        tact_r_depth_raw = self.gsmini_right.data.output.get("camera_depth") if hasattr(self, "gsmini_right") else None
        tact_ld_raw = self.gsmini_left_down.data.output.get("camera_depth") if hasattr(self, "gsmini_left_down") else None
        tact_rd_raw = self.gsmini_right_down.data.output.get("camera_depth") if hasattr(self, "gsmini_right_down") else None
        sensor_depths = {
            "left": tact_l_depth_raw,
            "right": tact_r_depth_raw,
            "left_down": tact_ld_raw,
            "right_down": tact_rd_raw,
        }
        self._maybe_refresh_tactile_baselines(sensor_depths)
        inner_features = self._encode_inner_sparsh({"left": tact_l_raw, "right": tact_r_raw})
        tact_ld = self._prep_depth_1ch(tact_ld_raw)
        tact_rd = self._prep_depth_1ch(tact_rd_raw)

        dev = self.device
        dev_type = getattr(dev, "type", None)
        if dev_type is None:
            dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
        use_amp = dev_type == "cuda"

        with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
            fld = self._down_depth_encoder(tact_ld).view(self.num_envs, self._tactile_feature_dim)
            frd = self._down_depth_encoder(tact_rd).view(self.num_envs, self._tactile_feature_dim)

        obs["tactile_left_rgb"] = self._push_tactile_feature("left", inner_features["left"])
        obs["tactile_right_rgb"] = self._push_tactile_feature("right", inner_features["right"])
        obs["tactile_left_down_depth"] = self._push_tactile_feature("left_down", fld)
        obs["tactile_right_down_depth"] = self._push_tactile_feature("right_down", frd)

        contact_ratios = {
            "left": self._contact_ratio_from_depth("left", tact_l_depth_raw),
            "right": self._contact_ratio_from_depth("right", tact_r_depth_raw),
            "left_down": self._contact_ratio_from_depth("left_down", tact_ld_raw),
            "right_down": self._contact_ratio_from_depth("right_down", tact_rd_raw),
        }
        self._latest_tactile_contact_ratios = contact_ratios
        contact_bits = torch.cat([self._contact_bits_from_ratio(contact_ratios[name]) for name in (
            "left", "right", "left_down", "right_down"
        )], dim=-1)
        obs["tactile_contact_bits"] = contact_bits
        self._tactile_contact_history[:, :-1, :] = self._tactile_contact_history[:, 1:, :].clone()
        self._tactile_contact_history[:, -1, :] = contact_bits
        obs["tactile_valid"] = self._tactile_contact_history.max(dim=1).values.max(dim=-1, keepdim=True).values
        self.extras.setdefault("meta", {})
        self.extras["meta"]["mixed_inner_tactile_encoder_name"] = self._inner_sparsh_encoder_name
        self.extras["meta"]["mixed_inner_tactile_encoder_summary"] = self._inner_sparsh_encoder_summary
        self.extras["meta"]["mixed_down_tactile_encoder"] = f"depth-cnn(out_dim={self._tactile_feature_dim})"

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        rewards = super()._get_rewards()

        log = self.extras.setdefault("log", {})
        ratio_means = {
            name: ratio.mean().detach()
            for name, ratio in self._latest_tactile_contact_ratios.items()
        }
        log["info/tactile_ratio_left"] = ratio_means["left"]
        log["info/tactile_ratio_right"] = ratio_means["right"]
        log["info/tactile_ratio_left_down"] = ratio_means["left_down"]
        log["info/tactile_ratio_right_down"] = ratio_means["right_down"]

        if self.cfg.reward_print_interval > 0 and (self.step_count % self.cfg.reward_print_interval == 0):
            try:
                print(
                    f"[Tactile Ratios] step {self.step_count}: "
                    f"left={ratio_means['left'].item():.4f}, "
                    f"right={ratio_means['right'].item():.4f}, "
                    f"left_down={ratio_means['left_down'].item():.4f}, "
                    f"right_down={ratio_means['right_down'].item():.4f}"
                )
            except Exception:
                pass

        return rewards

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self._pending_tactile_baseline_refresh[env_ids] = True
        self._pending_inner_sparsh_baseline_refresh[env_ids] = True
        for baseline in self._tactile_baseline_depth.values():
            if baseline is not None:
                baseline[env_ids] = 0.0
        for baseline in self._inner_tactile_baseline_rgb.values():
            if baseline is not None:
                baseline[env_ids] = 0.0
        for prev_rgb in self._inner_tactile_prev_rgb.values():
            prev_rgb[env_ids] = 0.0
        for history in self._tactile_feature_history.values():
            history[env_ids] = 0.0
        for ratio in self._latest_tactile_contact_ratios.values():
            ratio[env_ids] = 0.0
        self._tactile_contact_history[env_ids] = 0.0
