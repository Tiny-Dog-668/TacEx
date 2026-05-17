"""Occluded grasping with third-person vision and four tactile sensors encoded by Sparsh."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

try:
    import torchvision

    _HAS_TORCHVISION = True
except Exception:
    torchvision = None
    _HAS_TORCHVISION = False

from ..sparch_grasp.sparsh_encoder import SparshFrozenEncoder, project_sparsh_features
from .vt_box import (
    OccludedGraspingVisionFourTactileBoxCfg,
    OccludedGraspingVisionFourTactileBoxEnv,
)

_DEFAULT_ENCODER_ROOT = Path("/home/tinydog/桌面")
_SPARSH_ENCODER_PRESETS = {
    "ijepa_vitsmall": {
        "checkpoint_path": _DEFAULT_ENCODER_ROOT / "ijepa_vitsmall.ckpt",
        "model_size": "small",
        "encoder_type": "ijepa",
    },
    "ijepa_vitbase": {
        "checkpoint_path": _DEFAULT_ENCODER_ROOT / "ijepa_vitbase.ckpt",
        "model_size": "base",
        "encoder_type": "ijepa",
    },
    "dino_vitsmall": {
        "checkpoint_path": _DEFAULT_ENCODER_ROOT / "dino_vitsmall.ckpt",
        "model_size": "small",
        "encoder_type": "dino",
    },
    "dino_vitbase": {
        "checkpoint_path": _DEFAULT_ENCODER_ROOT / "dino_vitbase.ckpt",
        "model_size": "base",
        "encoder_type": "dino",
    },
    "dinov2_vitbase": {
        "checkpoint_path": _DEFAULT_ENCODER_ROOT / "dinov2_vitbase.ckpt",
        "model_size": "base",
        "encoder_type": "dinov2",
    },
}


def _infer_sparsh_model_size(name_or_path: str) -> str:
    text = str(name_or_path).lower()
    if "vitbase" in text or "base" in text:
        return "base"
    if "vitsmall" in text or "small" in text:
        return "small"
    raise ValueError(f"Could not infer Sparsh model size from '{name_or_path}'")


def _infer_sparsh_encoder_type(name_or_path: str) -> str:
    text = str(name_or_path).lower()
    if "ijepa" in text or "jepa" in text:
        return "ijepa"
    if "dinov2" in text:
        return "dinov2"
    if "dino" in text:
        return "dino"
    raise ValueError(f"Could not infer Sparsh encoder type from '{name_or_path}'")


def _resolve_checkpoint_candidate(name_or_path: str) -> Path:
    candidate = Path(name_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if candidate.suffix != ".ckpt":
        candidate_with_suffix = candidate.with_suffix(".ckpt")
        if candidate_with_suffix.exists():
            return candidate_with_suffix.resolve()
        desktop_candidate = (_DEFAULT_ENCODER_ROOT / candidate_with_suffix.name).resolve()
        if desktop_candidate.exists():
            return desktop_candidate
    desktop_candidate = (_DEFAULT_ENCODER_ROOT / candidate.name).resolve()
    if desktop_candidate.exists():
        return desktop_candidate
    raise FileNotFoundError(f"Sparsh checkpoint not found for '{name_or_path}'")


def _feature_dim_from_model_size(model_size: str) -> int:
    model_size = str(model_size).lower()
    if model_size == "small":
        return 384
    if model_size == "base":
        return 768
    raise ValueError(f"Unsupported Sparsh model size: {model_size}")


def _downsample_like_resnet18(size: int) -> int:
    # ResNet-18 downsamples 5 times before the global average pool.
    for _ in range(5):
        size = (int(size) + 1) // 2
    return size


def _resnet18_token_hw(img_size_hw: tuple[int, int]) -> tuple[int, int]:
    height, width = (int(img_size_hw[0]), int(img_size_hw[1]))
    return _downsample_like_resnet18(height), _downsample_like_resnet18(width)


def _sparsh_patch_token_count(img_size_hw: tuple[int, int], patch_size: int = 16) -> int:
    height, width = (int(img_size_hw[0]), int(img_size_hw[1]))
    return (height // int(patch_size)) * (width // int(patch_size))


def _resolve_sparsh_encoder_spec(
    encoder_name: str,
    checkpoint_path: str,
    model_size: str,
    encoder_type: str,
) -> dict[str, str | int]:
    name = str(encoder_name or "").strip()
    ckpt = str(checkpoint_path or "").strip()
    size = str(model_size or "").strip().lower()
    enc_type = str(encoder_type or "").strip().lower()

    if name and name in _SPARSH_ENCODER_PRESETS:
        preset = _SPARSH_ENCODER_PRESETS[name]
        ckpt_path = Path(preset["checkpoint_path"]).resolve()
        resolved_size = str(preset["model_size"])
        resolved_type = str(preset["encoder_type"])
        resolved_name = name
    elif name:
        ckpt_path = _resolve_checkpoint_candidate(name)
        resolved_size = size or _infer_sparsh_model_size(name)
        resolved_type = enc_type or _infer_sparsh_encoder_type(name)
        resolved_name = Path(name).stem
    elif ckpt:
        ckpt_path = _resolve_checkpoint_candidate(ckpt)
        resolved_size = size or _infer_sparsh_model_size(ckpt_path.name)
        resolved_type = enc_type or _infer_sparsh_encoder_type(ckpt_path.name)
        resolved_name = ckpt_path.stem
    else:
        raise ValueError("Set env.sparsh_encoder_name or env.sparsh_checkpoint_path for the Sparsh environment")

    if ckpt:
        ckpt_path = _resolve_checkpoint_candidate(ckpt)
        if not size:
            resolved_size = _infer_sparsh_model_size(ckpt_path.name)
        if not enc_type:
            resolved_type = _infer_sparsh_encoder_type(ckpt_path.name)
        resolved_name = ckpt_path.stem

    if size:
        resolved_size = size
    if enc_type:
        resolved_type = enc_type

    return {
        "name": resolved_name,
        "checkpoint_path": str(ckpt_path),
        "model_size": resolved_size,
        "encoder_type": resolved_type,
        "feature_dim": _feature_dim_from_model_size(resolved_size),
    }


@configclass
class OccludedGraspingVisionFourTactileSparshBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Occluded grasping config with Sparsh tactile features and unchanged scene/reward setup."""

    sparsh_repo_path = "/home/tinydog/Projects/sparsh"
    sparsh_encoder_name = "ijepa_vitsmall"
    sparsh_checkpoint_path = ""
    sparsh_model_size = ""
    sparsh_encoder_type = ""
    sparsh_encoder_summary = ""
    sparsh_num_register_tokens = 0
    sparsh_encoder_embed_dim = 384
    sparsh_feature_dim = 256
    sparsh_frame_stride = 5
    sparsh_baseline_warmup_steps = 4
    sparsh_baseline_retry_steps = 2
    sparsh_baseline_diff_threshold = 0.02

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
        if self.sparsh_feature_dim <= 0:
            raise ValueError(f"sparsh_feature_dim must be positive, got {self.sparsh_feature_dim}")
        self.sparsh_encoder_summary = (
            f"{self.sparsh_encoder_name} | type={self.sparsh_encoder_type} | "
            f"model={self.sparsh_model_size} | encoder_dim={self.sparsh_encoder_embed_dim} | "
            f"out_dim={self.sparsh_feature_dim} | "
            f"ckpt={self.sparsh_checkpoint_path}"
        )

        for attr_name in ("gsmini_left", "gsmini_right", "gsmini_left_down", "gsmini_right_down"):
            sensor_cfg = getattr(self, attr_name)
            sensor_cfg.data_types = ["tactile_rgb"]

        self.observation_space = {
            **self.observation_space,
            "tactile_left_depth_resnet": self.sparsh_feature_dim,
            "tactile_right_depth_resnet": self.sparsh_feature_dim,
            "tactile_left_down_depth_resnet": self.sparsh_feature_dim,
            "tactile_right_down_depth_resnet": self.sparsh_feature_dim,
        }


class OccludedGraspingVisionFourTactileSparshBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Occluded grasping box env with Sparsh replacing the tactile observation encoder."""

    cfg: OccludedGraspingVisionFourTactileSparshBoxCfg

    def __init__(
        self, cfg: OccludedGraspingVisionFourTactileSparshBoxCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        if hasattr(self, "_tactile_encoder"):
            del self._tactile_encoder
        self._sparsh_encoder_name = str(self.cfg.sparsh_encoder_name)
        self._sparsh_encoder_summary = str(self.cfg.sparsh_encoder_summary)

        self._sparsh_encoder = SparshFrozenEncoder(
            repo_path=self.cfg.sparsh_repo_path,
            checkpoint_path=self.cfg.sparsh_checkpoint_path,
            img_size_hw=tuple(getattr(self.cfg, "tactile_img_res_hw", (96, 128))),
            model_size=self.cfg.sparsh_model_size,
            encoder_type=self.cfg.sparsh_encoder_type,
            num_register_tokens=int(getattr(self.cfg, "sparsh_num_register_tokens", 0)),
        ).to(self.device)
        self._sparsh_encoder.eval()
        self._sparsh_encoder_embed_dim = int(self._sparsh_encoder.output_dim)
        self._sparsh_feature_dim = int(self.cfg.sparsh_feature_dim)
        print(f"[INFO] Sparsh tactile encoder: {self._sparsh_encoder_summary}")

        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))
        self._sparsh_sensor_keys = ("left", "right", "left_down", "right_down")
        self._sparsh_refs = {
            key: torch.zeros((self.num_envs, 3, target_h, target_w), device=self.device, dtype=torch.float32)
            for key in self._sparsh_sensor_keys
        }
        self._sparsh_frame_stride = int(getattr(self.cfg, "sparsh_frame_stride", 5))
        self._sparsh_history_len = self._sparsh_frame_stride + 1
        self._sparsh_hists = {
            key: torch.zeros(
                (self._sparsh_history_len, self.num_envs, 3, target_h, target_w),
                device=self.device,
                dtype=torch.float32,
            )
            for key in self._sparsh_sensor_keys
        }
        warmup_steps = int(getattr(self.cfg, "sparsh_baseline_warmup_steps", 4))
        self._sparsh_ref_pending = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self._sparsh_ref_delay = torch.full((self.num_envs,), warmup_steps, device=self.device, dtype=torch.long)
        self._sparsh_hist_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._sparsh_hist_write_idx = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

    def _prep_rgb_01(self, rgb_tensor: torch.Tensor | None) -> torch.Tensor:
        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))
        if rgb_tensor is None:
            return torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
        x = rgb_tensor.to(device=self.device, dtype=torch.float32)
        max_val = x.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            x = x / 255.0
        x = x.clamp(0.0, 1.0)
        x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x

    @staticmethod
    def _sparsh_diff(curr: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return torch.clamp(curr - ref + 0.5, 0.0, 1.0)

    def _get_base_policy_observations_without_tactile(self) -> dict[str, torch.Tensor]:
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

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        obs = self._get_base_policy_observations_without_tactile()

        tactile_raw = {
            "left": self.gsmini_left.data.output.get("tactile_rgb") if hasattr(self, "gsmini_left") else None,
            "right": self.gsmini_right.data.output.get("tactile_rgb") if hasattr(self, "gsmini_right") else None,
            "left_down": (
                self.gsmini_left_down.data.output.get("tactile_rgb") if hasattr(self, "gsmini_left_down") else None
            ),
            "right_down": (
                self.gsmini_right_down.data.output.get("tactile_rgb") if hasattr(self, "gsmini_right_down") else None
            ),
        }
        has_all_rgb = all(value is not None for value in tactile_raw.values())
        tactile_rgb = {key: self._prep_rgb_01(value) for key, value in tactile_raw.items()}

        diff_thresh = float(getattr(self.cfg, "sparsh_baseline_diff_threshold", 0.02))
        score_max = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        scores = {}
        for key in self._sparsh_sensor_keys:
            score = (tactile_rgb[key] - self._sparsh_refs[key]).abs().mean(dim=(1, 2, 3))
            scores[key] = score
            score_max = torch.maximum(score_max, score)

        pending = self._sparsh_ref_pending
        if torch.any(pending):
            dec_mask = pending & (self._sparsh_ref_delay > 0)
            if torch.any(dec_mask):
                self._sparsh_ref_delay[dec_mask] -= 1
            ready = pending & (self._sparsh_ref_delay <= 0)
            if torch.any(ready):
                baseline_empty = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
                for key in self._sparsh_sensor_keys:
                    baseline_empty &= self._sparsh_refs[key].abs().sum(dim=(1, 2, 3)) < 1e-6
                unsafe = score_max > diff_thresh
                safe = ready & (baseline_empty | ~unsafe)
                if not has_all_rgb:
                    safe = torch.zeros_like(safe, dtype=torch.bool, device=self.device)
                if torch.any(safe):
                    for key in self._sparsh_sensor_keys:
                        self._sparsh_refs[key][safe] = tactile_rgb[key][safe]
                    self._sparsh_ref_pending[safe] = False
                    self._sparsh_ref_delay[safe] = 0
                retry = ready & ~safe
                if torch.any(retry):
                    retry_steps = max(int(getattr(self.cfg, "sparsh_baseline_retry_steps", 2)), 1)
                    if retry_steps == 1:
                        self._sparsh_ref_delay[retry] = 1
                    else:
                        self._sparsh_ref_delay[retry] = torch.randint(
                            1, retry_steps + 1, (int(retry.sum().item()),), device=self.device
                        )

        ready_mask = (~self._sparsh_ref_pending).clone()
        if not has_all_rgb:
            ready_mask = torch.zeros_like(ready_mask, dtype=torch.bool, device=self.device)

        if torch.any(~ready_mask):
            for key in self._sparsh_sensor_keys:
                self._sparsh_hists[key][:, ~ready_mask] = 0
            self._sparsh_hist_count[~ready_mask] = 0
            self._sparsh_hist_write_idx[~ready_mask] = 0

        diffs = {
            key: torch.zeros_like(tactile_rgb[key], dtype=torch.float32, device=self.device)
            for key in self._sparsh_sensor_keys
        }
        if torch.any(ready_mask):
            for key in self._sparsh_sensor_keys:
                diffs[key][ready_mask] = self._sparsh_diff(
                    tactile_rgb[key][ready_mask], self._sparsh_refs[key][ready_mask]
                )

        past = {key: diffs[key].clone() for key in self._sparsh_sensor_keys}
        history_ready = ready_mask & (self._sparsh_hist_count >= (self._sparsh_history_len - 1))
        if torch.any(history_ready):
            history_env_ids = torch.nonzero(history_ready, as_tuple=False).squeeze(-1)
            history_read_idx = (self._sparsh_hist_write_idx[history_ready] + 1) % self._sparsh_history_len
            for key in self._sparsh_sensor_keys:
                past[key][history_ready] = self._sparsh_hists[key][history_read_idx, history_env_ids]

        if torch.any(ready_mask):
            ready_env_ids = torch.nonzero(ready_mask, as_tuple=False).squeeze(-1)
            write_idx = self._sparsh_hist_write_idx[ready_mask]
            for key in self._sparsh_sensor_keys:
                self._sparsh_hists[key][write_idx, ready_env_ids] = diffs[key][ready_mask]
            self._sparsh_hist_count[ready_mask] = torch.clamp(
                self._sparsh_hist_count[ready_mask] + 1,
                max=self._sparsh_history_len,
            )
            self._sparsh_hist_write_idx[ready_mask] = (write_idx + 1) % self._sparsh_history_len

        features = {
            key: torch.zeros((self.num_envs, self._sparsh_feature_dim), device=self.device, dtype=torch.float32)
            for key in self._sparsh_sensor_keys
        }
        if torch.any(ready_mask):
            sparsh_batch = torch.cat(
                [
                    torch.cat([diffs[key][ready_mask], past[key][ready_mask]], dim=1)
                    for key in self._sparsh_sensor_keys
                ],
                dim=0,
            )
            dev = self.device
            dev_type = getattr(dev, "type", None)
            if dev_type is None:
                dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
            use_amp = dev_type == "cuda"
            with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
                encoded = self._sparsh_encoder(sparsh_batch).mean(dim=1).float()
            encoded = project_sparsh_features(encoded, self._sparsh_feature_dim)
            split_encoded = encoded.chunk(len(self._sparsh_sensor_keys), dim=0)
            for key, feat in zip(self._sparsh_sensor_keys, split_encoded):
                features[key][ready_mask] = feat

        obs.update(
            {
                "tactile_left_depth_resnet": features["left"],
                "tactile_right_depth_resnet": features["right"],
                "tactile_left_down_depth_resnet": features["left_down"],
                "tactile_right_down_depth_resnet": features["right_down"],
            }
        )
        self.extras.setdefault("log", {})
        self.extras["log"]["sparsh_baseline_ready_ratio"] = float((~self._sparsh_ref_pending).float().mean().item())
        self.extras["log"]["sparsh_score_mean"] = float(score_max.mean().item())
        self.extras.setdefault("meta", {})
        self.extras["meta"]["sparsh_encoder_name"] = self._sparsh_encoder_name
        self.extras["meta"]["sparsh_encoder_summary"] = self._sparsh_encoder_summary
        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        warmup_steps = int(getattr(self.cfg, "sparsh_baseline_warmup_steps", 4))
        for key in self._sparsh_sensor_keys:
            self._sparsh_refs[key][env_ids] = 0
            self._sparsh_hists[key][:, env_ids] = 0
        self._sparsh_ref_pending[env_ids] = True
        self._sparsh_ref_delay[env_ids] = warmup_steps
        self._sparsh_hist_count[env_ids] = 0
        self._sparsh_hist_write_idx[env_ids] = 0


@configclass
class OccludedGraspingVisionFourTactileSparshCrossAttentionBoxCfg(OccludedGraspingVisionFourTactileSparshBoxCfg):
    """Sparsh occluded grasping config exposing token sequences for cross-attention fusion."""

    third_camera_token_hw = (4, 5)
    third_camera_token_dim = 256
    tactile_token_hw = (3, 4)
    sparsh_patch_token_count = 12

    def __post_init__(self):
        super().__post_init__()
        full_visual_hw = _resnet18_token_hw((int(self.third_person_camera.height), int(self.third_person_camera.width)))
        visual_hw = tuple(int(v) for v in getattr(self, "third_camera_token_hw", full_visual_hw))
        if len(visual_hw) != 2 or any(v <= 0 for v in visual_hw):
            raise ValueError(f"third_camera_token_hw must be two positive integers, got {visual_hw}")
        if visual_hw[0] > full_visual_hw[0] or visual_hw[1] > full_visual_hw[1]:
            raise ValueError(f"third_camera_token_hw={visual_hw} exceeds full ResNet token grid {full_visual_hw}")
        self.third_camera_token_hw = visual_hw

        tactile_img_res_hw = tuple(getattr(self, "tactile_img_res_hw", (96, 128)))
        full_tactile_hw = (int(tactile_img_res_hw[0] // 16), int(tactile_img_res_hw[1] // 16))
        tactile_hw = tuple(int(v) for v in getattr(self, "tactile_token_hw", full_tactile_hw))
        if len(tactile_hw) != 2 or any(v <= 0 for v in tactile_hw):
            raise ValueError(f"tactile_token_hw must be two positive integers, got {tactile_hw}")
        if tactile_hw[0] > full_tactile_hw[0] or tactile_hw[1] > full_tactile_hw[1]:
            raise ValueError(f"tactile_token_hw={tactile_hw} exceeds full Sparsh token grid {full_tactile_hw}")
        self.tactile_token_hw = tactile_hw
        self.sparsh_patch_token_count = int(tactile_hw[0] * tactile_hw[1])

        critic_obs = {
            key: value
            for key, value in self.observation_space.items()
            if str(key).startswith("critic_")
        }
        self.observation_space = {
            "proprio_obs": 18,
            "third_resnet_tokens": [
                int(self.third_camera_token_hw[0] * self.third_camera_token_hw[1]),
                int(self.third_camera_token_dim),
            ],
            "tactile_left_tokens": [int(self.sparsh_patch_token_count), int(self.sparsh_feature_dim)],
            "tactile_right_tokens": [int(self.sparsh_patch_token_count), int(self.sparsh_feature_dim)],
            "tactile_left_down_tokens": [int(self.sparsh_patch_token_count), int(self.sparsh_feature_dim)],
            "tactile_right_down_tokens": [int(self.sparsh_patch_token_count), int(self.sparsh_feature_dim)],
            **critic_obs,
        }


class OccludedGraspingVisionFourTactileSparshCrossAttentionBoxEnv(OccludedGraspingVisionFourTactileSparshBoxEnv):
    """Sparsh env variant that exposes unpooled visual/tactile token sequences."""

    cfg: OccludedGraspingVisionFourTactileSparshCrossAttentionBoxCfg

    def __init__(
        self,
        cfg: OccludedGraspingVisionFourTactileSparshCrossAttentionBoxCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._visual_token_hw = tuple(getattr(self.cfg, "third_camera_token_hw", (4, 5)))
        self._visual_token_count = int(self._visual_token_hw[0] * self._visual_token_hw[1])
        self._visual_token_dim = int(getattr(self.cfg, "third_camera_token_dim", 256))
        self._tactile_token_hw = tuple(getattr(self.cfg, "tactile_token_hw", (3, 4)))
        self._sparsh_patch_token_count = int(
            getattr(
                self.cfg,
                "sparsh_patch_token_count",
                _sparsh_patch_token_count(tuple(getattr(self.cfg, "tactile_img_res_hw", (96, 128)))),
            )
        )

        if hasattr(self, "_resnet18"):
            del self._resnet18

        self._resnet18_spatial = None
        if getattr(self, "_use_resnet18", False) and _HAS_TORCHVISION:
            try:
                weights = torchvision.models.ResNet18_Weights.DEFAULT
                backbone = torchvision.models.resnet18(weights=weights)
            except Exception:
                backbone = torchvision.models.resnet18(pretrained=True)
            self._resnet18_spatial = torch.nn.Sequential(*list(backbone.children())[:-2]).to(self.device)
            self._resnet18_spatial.eval()
            for parameter in self._resnet18_spatial.parameters():
                parameter.requires_grad_(False)

    def _get_base_policy_observations_without_tactile(self) -> dict[str, torch.Tensor]:
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

        third_tokens = torch.zeros(
            (self.num_envs, self._visual_token_count, self._visual_token_dim),
            device=self.device,
            dtype=torch.float32,
        )
        if self._resnet18_spatial is not None:
            xt = third_rgb.permute(0, 3, 1, 2).contiguous()
            if hasattr(self, "_imagenet_mean"):
                xt = (xt - self._imagenet_mean) / self._imagenet_std
            dev = self.device
            dev_type = getattr(dev, "type", None)
            if dev_type is None:
                dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
            use_amp = dev_type == "cuda"
            with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
                fmap = self._resnet18_spatial(xt).float()
            if tuple(fmap.shape[-2:]) != self._visual_token_hw:
                fmap = F.adaptive_avg_pool2d(fmap, self._visual_token_hw)
            if fmap.shape[1] == 512 and self._visual_token_dim == 256:
                fmap = fmap.view(fmap.shape[0], 256, 2, fmap.shape[2], fmap.shape[3]).mean(dim=2)
            third_tokens = fmap.flatten(start_dim=2).transpose(1, 2).contiguous()

        return {
            "proprio_obs": proprio_obs,
            "third_resnet_tokens": third_tokens,
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

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        obs = self._get_base_policy_observations_without_tactile()

        tactile_raw = {
            "left": self.gsmini_left.data.output.get("tactile_rgb") if hasattr(self, "gsmini_left") else None,
            "right": self.gsmini_right.data.output.get("tactile_rgb") if hasattr(self, "gsmini_right") else None,
            "left_down": (
                self.gsmini_left_down.data.output.get("tactile_rgb") if hasattr(self, "gsmini_left_down") else None
            ),
            "right_down": (
                self.gsmini_right_down.data.output.get("tactile_rgb") if hasattr(self, "gsmini_right_down") else None
            ),
        }
        has_all_rgb = all(value is not None for value in tactile_raw.values())
        tactile_rgb = {key: self._prep_rgb_01(value) for key, value in tactile_raw.items()}

        diff_thresh = float(getattr(self.cfg, "sparsh_baseline_diff_threshold", 0.02))
        score_max = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        for key in self._sparsh_sensor_keys:
            score = (tactile_rgb[key] - self._sparsh_refs[key]).abs().mean(dim=(1, 2, 3))
            score_max = torch.maximum(score_max, score)

        pending = self._sparsh_ref_pending
        if torch.any(pending):
            dec_mask = pending & (self._sparsh_ref_delay > 0)
            if torch.any(dec_mask):
                self._sparsh_ref_delay[dec_mask] -= 1
            ready = pending & (self._sparsh_ref_delay <= 0)
            if torch.any(ready):
                baseline_empty = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
                for key in self._sparsh_sensor_keys:
                    baseline_empty &= self._sparsh_refs[key].abs().sum(dim=(1, 2, 3)) < 1e-6
                unsafe = score_max > diff_thresh
                safe = ready & (baseline_empty | ~unsafe)
                if not has_all_rgb:
                    safe = torch.zeros_like(safe, dtype=torch.bool, device=self.device)
                if torch.any(safe):
                    for key in self._sparsh_sensor_keys:
                        self._sparsh_refs[key][safe] = tactile_rgb[key][safe]
                    self._sparsh_ref_pending[safe] = False
                    self._sparsh_ref_delay[safe] = 0
                retry = ready & ~safe
                if torch.any(retry):
                    retry_steps = max(int(getattr(self.cfg, "sparsh_baseline_retry_steps", 2)), 1)
                    if retry_steps == 1:
                        self._sparsh_ref_delay[retry] = 1
                    else:
                        self._sparsh_ref_delay[retry] = torch.randint(
                            1, retry_steps + 1, (int(retry.sum().item()),), device=self.device
                        )

        ready_mask = (~self._sparsh_ref_pending).clone()
        if not has_all_rgb:
            ready_mask = torch.zeros_like(ready_mask, dtype=torch.bool, device=self.device)

        if torch.any(~ready_mask):
            for key in self._sparsh_sensor_keys:
                self._sparsh_hists[key][:, ~ready_mask] = 0
            self._sparsh_hist_count[~ready_mask] = 0
            self._sparsh_hist_write_idx[~ready_mask] = 0

        diffs = {
            key: torch.zeros_like(tactile_rgb[key], dtype=torch.float32, device=self.device)
            for key in self._sparsh_sensor_keys
        }
        if torch.any(ready_mask):
            for key in self._sparsh_sensor_keys:
                diffs[key][ready_mask] = self._sparsh_diff(
                    tactile_rgb[key][ready_mask], self._sparsh_refs[key][ready_mask]
                )

        past = {key: diffs[key].clone() for key in self._sparsh_sensor_keys}
        history_ready = ready_mask & (self._sparsh_hist_count >= (self._sparsh_history_len - 1))
        if torch.any(history_ready):
            history_env_ids = torch.nonzero(history_ready, as_tuple=False).squeeze(-1)
            history_read_idx = (self._sparsh_hist_write_idx[history_ready] + 1) % self._sparsh_history_len
            for key in self._sparsh_sensor_keys:
                past[key][history_ready] = self._sparsh_hists[key][history_read_idx, history_env_ids]

        if torch.any(ready_mask):
            ready_env_ids = torch.nonzero(ready_mask, as_tuple=False).squeeze(-1)
            write_idx = self._sparsh_hist_write_idx[ready_mask]
            for key in self._sparsh_sensor_keys:
                self._sparsh_hists[key][write_idx, ready_env_ids] = diffs[key][ready_mask]
            self._sparsh_hist_count[ready_mask] = torch.clamp(
                self._sparsh_hist_count[ready_mask] + 1,
                max=self._sparsh_history_len,
            )
            self._sparsh_hist_write_idx[ready_mask] = (write_idx + 1) % self._sparsh_history_len

        token_sequences = {
            key: torch.zeros(
                (self.num_envs, self._sparsh_patch_token_count, self._sparsh_feature_dim),
                device=self.device,
                dtype=torch.float32,
            )
            for key in self._sparsh_sensor_keys
        }
        if torch.any(ready_mask):
            sparsh_batch = torch.cat(
                [
                    torch.cat([diffs[key][ready_mask], past[key][ready_mask]], dim=1)
                    for key in self._sparsh_sensor_keys
                ],
                dim=0,
            )
            dev = self.device
            dev_type = getattr(dev, "type", None)
            if dev_type is None:
                dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
            use_amp = dev_type == "cuda"
            with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
                encoded = self._sparsh_encoder(sparsh_batch).float()
            if encoded.shape[1] != self._sparsh_patch_token_count:
                tactile_img_res_hw = tuple(getattr(self.cfg, "tactile_img_res_hw", (96, 128)))
                full_tactile_hw = (int(tactile_img_res_hw[0] // 16), int(tactile_img_res_hw[1] // 16))
                encoded = encoded.transpose(1, 2).reshape(
                    encoded.shape[0], encoded.shape[2], full_tactile_hw[0], full_tactile_hw[1]
                )
                encoded = F.adaptive_avg_pool2d(encoded, self._tactile_token_hw)
                encoded = encoded.flatten(start_dim=2).transpose(1, 2).contiguous()
            encoded = project_sparsh_features(encoded, self._sparsh_feature_dim)
            split_encoded = encoded.chunk(len(self._sparsh_sensor_keys), dim=0)
            for key, feat in zip(self._sparsh_sensor_keys, split_encoded):
                token_sequences[key][ready_mask] = feat

        obs.update(
            {
                "tactile_left_tokens": token_sequences["left"],
                "tactile_right_tokens": token_sequences["right"],
                "tactile_left_down_tokens": token_sequences["left_down"],
                "tactile_right_down_tokens": token_sequences["right_down"],
            }
        )
        self.extras.setdefault("log", {})
        self.extras["log"]["sparsh_baseline_ready_ratio"] = float((~self._sparsh_ref_pending).float().mean().item())
        self.extras["log"]["sparsh_score_mean"] = float(score_max.mean().item())
        self.extras.setdefault("meta", {})
        self.extras["meta"]["sparsh_encoder_name"] = self._sparsh_encoder_name
        self.extras["meta"]["sparsh_encoder_summary"] = self._sparsh_encoder_summary
        return {"policy": obs}
