"""Occluded grasping variant exposing reliability/stage supervision targets."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from .vt_box import (
    CAN_RESET_ROOT_Z,
    OccludedGraspingVisionFourTactileBoxCfg,
    OccludedGraspingVisionFourTactileBoxEnv,
)


@configclass
class OccludedGraspingVTReliabilityStageBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Configuration for auxiliary reliability/stage supervision targets."""

    stage_near_probe_distance = 0.085
    stage_lift_height_delta = 0.02
    stage_contact_diff_threshold = 0.05
    stage_contact_area_threshold = 0.01

    reliability_use_bbox3d = True
    reliability_fallback_visible_ratio = 0.0
    reliability_bbox_lazy_init = True
    tactile_valid_hist_len = 5
    tactile_obs_rename_pairs = (
        ("tactile_left_depth_resnet", "tactile_left_rgb_resnet"),
        ("tactile_right_depth_resnet", "tactile_right_rgb_resnet"),
        ("tactile_left_down_depth_resnet", "tactile_left_down_rgb_resnet"),
        ("tactile_right_down_depth_resnet", "tactile_right_down_rgb_resnet"),
    )

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()

        # Use DINO tactile encoder by default for this task.
        self.tactile_encoder_type = "dino"
        if not str(getattr(self, "tactile_dino_encoder_name", "")).strip():
            self.tactile_dino_encoder_name = "dino_vitsmall"
        # Keep one adjacent pair (current, previous) so tactile feature dim stays 256 per sensor.
        # This matches the policy/checkpoint layout used by this task family.
        self.tactile_dino_total_frames = 2
        sync_tactile_obs = getattr(self, "_sync_tactile_observation_space", None)
        if callable(sync_tactile_obs):
            sync_tactile_obs()

        # Keep original policy inputs and append auxiliary supervision tensors.
        self.observation_space = dict(self.observation_space)
        for old_key, new_key in self.tactile_obs_rename_pairs:
            if old_key in self.observation_space:
                self.observation_space[new_key] = self.observation_space[old_key]
        self.observation_space["aux_head_input"] = 256 + 18
        self.observation_space["aux_reliability_gt"] = 1
        self.observation_space["aux_stage_gt"] = 1
        self.observation_space["aux_stage_onehot_gt"] = 4
        self.observation_space["aux_contact_bits"] = 4
        self.observation_space["aux_probe_gt"] = 4
        self.observation_space["aux_grasp_gt"] = 3
        self.observation_space["prev_action"] = int(getattr(self, "action_space", 5))
        hist_len = max(1, int(getattr(self, "tactile_valid_hist_len", 5)))
        self.tactile_valid_hist_len = hist_len
        self.observation_space["bottom_valid_hist"] = hist_len * 2
        self.observation_space["inner_valid_hist"] = hist_len * 2

        # Stage GT uses tactile RGB deltas only (no depth dependency).
        self.gsmini_left.data_types = ["tactile_rgb"]
        self.gsmini_right.data_types = ["tactile_rgb"]
        self.gsmini_left_down.data_types = ["tactile_rgb"]
        self.gsmini_right_down.data_types = ["tactile_rgb"]

        # bbox_3d visibility needs semantic tags on the grasp target.
        try:
            self.can = self.can.replace(
                spawn=self.can.spawn.replace(
                    semantic_tags=[("class", "can")],
                )
            )
        except Exception:
            try:
                self.can.spawn.semantic_tags = [("class", "can")]
            except Exception:
                pass


class OccludedGraspingVTReliabilityStageBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Box task with GT labels for reliability (visibility) and stage classification."""

    cfg: OccludedGraspingVTReliabilityStageBoxCfg

    STAGE_APPROACH = 0
    STAGE_PROBE = 1
    STAGE_GRASP = 2
    STAGE_LIFT = 3

    _STAGE_ID_TO_NAME = {
        STAGE_APPROACH: "approach",
        STAGE_PROBE: "probe",
        STAGE_GRASP: "grasp",
        STAGE_LIFT: "lift",
    }

    _CONTACT_SENSOR_ORDER = ("left", "right", "left_down", "right_down")
    _CONTACT_SENSOR_ATTRS = {
        "left": "gsmini_left",
        "right": "gsmini_right",
        "left_down": "gsmini_left_down",
        "right_down": "gsmini_right_down",
    }

    def __init__(self, cfg: OccludedGraspingVTReliabilityStageBoxCfg, render_mode: str | None = None, **kwargs):
        print("[INFO] [VT-ReliabilityStage] env init: entering base env constructor")
        super().__init__(cfg, render_mode, **kwargs)
        print("[INFO] [VT-ReliabilityStage] env init: base env constructed")

        self._stage_contact_rgb_baseline: dict[str, torch.Tensor | None] = {
            key: None for key in self._CONTACT_SENSOR_ORDER
        }
        self._pending_stage_contact_rgb_baseline_refresh = torch.ones(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._latest_contact_bits = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._latest_stage_gt = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._latest_reliability_gt = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._latest_probe_gt = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self._latest_grasp_gt = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._prev_action_obs = torch.zeros(
            (self.num_envs, int(getattr(self.cfg, "action_space", 5))), dtype=torch.float32, device=self.device
        )
        self._valid_hist_len = max(1, int(getattr(self.cfg, "tactile_valid_hist_len", 5)))
        self._bottom_valid_hist = torch.zeros((self.num_envs, self._valid_hist_len, 2), dtype=torch.float32, device=self.device)
        self._inner_valid_hist = torch.zeros((self.num_envs, self._valid_hist_len, 2), dtype=torch.float32, device=self.device)
        self._latest_tactile_over_thresh_ratio_by_sensor = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._latest_tactile_over_thresh_ratio_mean = torch.zeros(
            (self.num_envs,), dtype=torch.float32, device=self.device
        )

        self._object_init_height = self._can.data.root_pos_w[:, 2].clone()
        default_h = torch.full_like(self._object_init_height, float(CAN_RESET_ROOT_Z))
        self._object_init_height = torch.where(torch.isfinite(self._object_init_height), self._object_init_height, default_h)

        self._bbox_annotator = None
        self._bbox_render_products: list[str] = []
        self._bbox_enabled = False
        self._bbox_init_attempted = False
        if not bool(getattr(self.cfg, "reliability_bbox_lazy_init", True)):
            self._init_bbox_annotator()
        else:
            print("[INFO] [VT-ReliabilityStage] bbox_3d annotator will initialize lazily on first observation.")

    def _resolve_render_product_paths(self) -> list[str]:
        """Resolve camera render products across Isaac Sim/TiledCamera versions."""
        candidates = (
            "render_product_paths",
            "render_product_path",
            "_render_product_paths",
            "_render_product_path",
        )
        for attr in candidates:
            value = getattr(self.third_person_camera, attr, None)
            if value is None:
                continue
            if isinstance(value, str):
                return [value]
            if isinstance(value, (tuple, list)):
                return [str(v) for v in value if v is not None and str(v)]
        return []

    def _init_bbox_annotator(self) -> None:
        """Attach Replicator bbox_3d annotator to third-person render products."""
        self._bbox_init_attempted = True
        if not bool(getattr(self.cfg, "reliability_use_bbox3d", True)):
            return

        try:
            import omni.replicator.core as rep  # type: ignore
        except Exception as e:
            print(f"[WARN] bbox_3d disabled: cannot import omni.replicator.core ({e})")
            return

        try:
            render_products = self._resolve_render_product_paths()
            if not render_products:
                print("[WARN] bbox_3d disabled: no camera render products found.")
                return

            annotator = rep.AnnotatorRegistry.get_annotator(
                "bounding_box_3d",
                init_params={"semanticTypes": ["class"]},
            )
            for render_product in render_products:
                try:
                    annotator.attach(render_product)
                except Exception:
                    annotator.attach([render_product])

            self._bbox_annotator = annotator
            self._bbox_render_products = render_products
            self._bbox_enabled = True
            print(f"[INFO] bbox_3d annotator attached (render_products={len(render_products)}).")
        except Exception as e:
            self._bbox_annotator = None
            self._bbox_render_products = []
            self._bbox_enabled = False
            print(f"[WARN] bbox_3d disabled: failed to initialize annotator ({e})")

    def _detach_bbox_annotator(self) -> None:
        """Detach bbox annotator safely if initialized."""
        if self._bbox_annotator is None:
            return
        try:
            if self._bbox_render_products:
                self._bbox_annotator.detach(self._bbox_render_products)
        except Exception:
            pass
        self._bbox_annotator = None
        self._bbox_render_products = []
        self._bbox_enabled = False

    def close(self):
        self._detach_bbox_annotator()
        super().close()

    @staticmethod
    def _unpack_bbox_output(output):
        if isinstance(output, dict):
            data = output.get("data")
            info = output.get("info", {})
            if not info:
                info = {key: value for key, value in output.items() if key != "data"}
            return data, info
        return output, {}

    @staticmethod
    def _extract_env_id_from_path(path: str) -> int | None:
        prefix = "/World/envs/env_"
        if not path.startswith(prefix):
            return None
        suffix = path[len(prefix):]
        digits: list[str] = []
        for ch in suffix:
            if ch.isdigit():
                digits.append(ch)
                continue
            break
        if not digits:
            return None
        try:
            return int("".join(digits))
        except ValueError:
            return None

    @staticmethod
    def _extract_occlusion_ratio(row) -> float | None:
        value = None
        if isinstance(row, dict):
            value = row.get("occlusionRatio")
        else:
            try:
                value = row["occlusionRatio"]
            except Exception:
                value = getattr(row, "occlusionRatio", None)

        try:
            ratio = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(ratio) or ratio < 0.0:
            return None
        return ratio

    def _compute_bbox_visible_ratio_gt(self) -> torch.Tensor:
        """Compute per-env visible ratio from bbox_3d (1 - occlusionRatio)."""
        fallback = float(getattr(self.cfg, "reliability_fallback_visible_ratio", 0.0))
        visible_ratio = torch.full((self.num_envs,), fallback, dtype=torch.float32, device=self.device)

        if (
            bool(getattr(self.cfg, "reliability_use_bbox3d", True))
            and not self._bbox_enabled
            and not self._bbox_init_attempted
        ):
            self._init_bbox_annotator()

        if not self._bbox_enabled or self._bbox_annotator is None:
            return visible_ratio

        try:
            bbox_output = self._bbox_annotator.get_data()
        except Exception:
            return visible_ratio

        bbox_data, bbox_info = self._unpack_bbox_output(bbox_output)
        prim_paths = bbox_info.get("primPaths", [])
        if bbox_data is None or not isinstance(prim_paths, Iterable):
            return visible_ratio

        max_occlusion_by_env: list[float | None] = [None] * self.num_envs
        for row_idx, prim_path in enumerate(prim_paths):
            path_str = str(prim_path)
            env_id = self._extract_env_id_from_path(path_str)
            if env_id is None or env_id < 0 or env_id >= self.num_envs:
                continue

            can_prefix = f"/World/envs/env_{env_id}/can"
            if path_str != can_prefix and not path_str.startswith(f"{can_prefix}/"):
                continue

            try:
                row = bbox_data[row_idx]
            except Exception:
                continue
            occlusion_ratio = self._extract_occlusion_ratio(row)
            if occlusion_ratio is None:
                continue

            previous = max_occlusion_by_env[env_id]
            max_occlusion_by_env[env_id] = (
                occlusion_ratio if previous is None else max(previous, occlusion_ratio)
            )

        for env_id, occlusion_ratio in enumerate(max_occlusion_by_env):
            if occlusion_ratio is None:
                continue
            visible_ratio[env_id] = float(max(0.0, min(1.0, 1.0 - occlusion_ratio)))

        return visible_ratio

    def _prep_rgb_01(self, rgb_tensor: torch.Tensor | None) -> torch.Tensor | None:
        if rgb_tensor is None:
            return None
        rgb = rgb_tensor.to(device=self.device, dtype=torch.float32)
        max_val = rgb.max()
        if torch.isfinite(max_val) and max_val > 1.5:
            rgb = rgb / 255.0
        rgb = rgb.clamp(0.0, 1.0)
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(-1)
        if rgb.ndim != 4:
            return None
        # NHWC -> NCHW for channel-stable diff reduction
        return rgb.permute(0, 3, 1, 2).contiguous()

    def _maybe_refresh_stage_contact_baselines(self, sensor_rgbs: dict[str, torch.Tensor | None]) -> None:
        pending = self._pending_stage_contact_rgb_baseline_refresh
        if not torch.any(pending):
            return

        pending_ids = pending.nonzero(as_tuple=False).squeeze(-1)
        if pending_ids.numel() == 0:
            return

        captured_any = False
        for sensor_name, rgb_tensor in sensor_rgbs.items():
            rgb = self._prep_rgb_01(rgb_tensor)
            if rgb is None:
                continue

            baseline = self._stage_contact_rgb_baseline.get(sensor_name)
            if baseline is None or baseline.shape != rgb.shape:
                baseline = torch.zeros_like(rgb, dtype=torch.float32, device=self.device)
                self._stage_contact_rgb_baseline[sensor_name] = baseline

            baseline[pending_ids] = rgb[pending_ids]
            captured_any = True

        if captured_any:
            pending[pending_ids] = False

    def _contact_ratio_from_rgb(self, sensor_name: str, rgb_tensor: torch.Tensor | None) -> torch.Tensor:
        rgb = self._prep_rgb_01(rgb_tensor)
        baseline = self._stage_contact_rgb_baseline.get(sensor_name)
        if rgb is None or baseline is None:
            return torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        diff = torch.abs(rgb - baseline)
        diff_threshold = float(getattr(self.cfg, "stage_contact_diff_threshold", 0.05))
        return (diff >= diff_threshold).to(torch.float32).mean(dim=(1, 2, 3), keepdim=False)

    def _compute_stage_gt(self, contact_bits: torch.Tensor) -> torch.Tensor:
        """Compute stage labels with priority: lift > grasp > probe > approach."""
        inner_contact = (contact_bits[:, 0] > 0.5) | (contact_bits[:, 1] > 0.5)
        bottom_contact = (contact_bits[:, 2] > 0.5) | (contact_bits[:, 3] > 0.5)

        hand_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        ee_tip_pos_w, _ = math_utils.combine_frame_transforms(
            hand_pos_w, hand_quat_w, self._offset_pos, self._offset_rot
        )
        can_pos_w = self._can.data.root_pos_w

        near_probe_distance = float(getattr(self.cfg, "stage_near_probe_distance", 0.085))
        near_probe_zone = torch.norm(ee_tip_pos_w - can_pos_w, dim=-1) < near_probe_distance

        lift_height_delta = float(getattr(self.cfg, "stage_lift_height_delta", 0.02))
        object_lifted = can_pos_w[:, 2] > (self._object_init_height + lift_height_delta)

        stage = torch.full(
            (self.num_envs,),
            fill_value=self.STAGE_APPROACH,
            dtype=torch.long,
            device=self.device,
        )
        stage = torch.where(near_probe_zone | bottom_contact, torch.full_like(stage, self.STAGE_PROBE), stage)
        stage = torch.where(inner_contact, torch.full_like(stage, self.STAGE_GRASP), stage)
        stage = torch.where(object_lifted, torch.full_like(stage, self.STAGE_LIFT), stage)
        return stage

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]

        # Expose tactile features with RGB-based key names for the auxiliary task config.
        for old_key, new_key in getattr(self.cfg, "tactile_obs_rename_pairs", ()):
            if old_key in obs:
                obs[new_key] = obs[old_key]

        proprio = obs.get("proprio_obs")
        if proprio is None:
            proprio = torch.zeros((self.num_envs, 18), dtype=torch.float32, device=self.device)
        else:
            proprio = proprio.to(device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)

        vision_feat = obs.get("third_resnet")
        if vision_feat is None:
            vision_feat = torch.zeros((self.num_envs, 256), dtype=torch.float32, device=self.device)
        else:
            vision_feat = vision_feat.to(device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)

        aux_head_input = torch.cat([vision_feat, proprio], dim=-1)

        sensor_rgbs = {}
        for sensor_name, sensor_attr in self._CONTACT_SENSOR_ATTRS.items():
            sensor = getattr(self, sensor_attr, None)
            rgb = sensor.data.output.get("tactile_rgb") if sensor is not None else None
            sensor_rgbs[sensor_name] = rgb
        self._maybe_refresh_stage_contact_baselines(sensor_rgbs)

        contact_ratios = {
            sensor_name: self._contact_ratio_from_rgb(sensor_name, sensor_rgbs[sensor_name])
            for sensor_name in self._CONTACT_SENSOR_ORDER
        }
        self._latest_tactile_over_thresh_ratio_by_sensor = torch.stack(
            [contact_ratios[sensor_name] for sensor_name in self._CONTACT_SENSOR_ORDER], dim=-1
        )
        self._latest_tactile_over_thresh_ratio_mean = self._latest_tactile_over_thresh_ratio_by_sensor.mean(dim=-1)
        probe_left = contact_ratios["left_down"]
        probe_right = contact_ratios["right_down"]
        probe_diff = torch.abs(probe_left - probe_right)
        probe_mean = 0.5 * (probe_left + probe_right)
        probe_gt = torch.stack([probe_left, probe_right, probe_diff, probe_mean], dim=-1)

        grasp_left = contact_ratios["left"]
        grasp_right = contact_ratios["right"]
        grasp_balance = (1.0 - torch.abs(grasp_left - grasp_right)).clamp(0.0, 1.0)
        grasp_gt = torch.stack([grasp_left, grasp_right, grasp_balance], dim=-1)

        area_threshold = float(getattr(self.cfg, "stage_contact_area_threshold", 0.01))
        contact_bits = torch.cat(
            [(contact_ratios[sensor_name] > area_threshold).to(torch.float32).unsqueeze(-1)
             for sensor_name in self._CONTACT_SENSOR_ORDER],
            dim=-1,
        )
        inner_valid_now = contact_bits[:, :2]
        bottom_valid_now = contact_bits[:, 2:4]
        if self._valid_hist_len > 1:
            self._inner_valid_hist[:, :-1] = self._inner_valid_hist[:, 1:].clone()
            self._bottom_valid_hist[:, :-1] = self._bottom_valid_hist[:, 1:].clone()
        self._inner_valid_hist[:, -1] = inner_valid_now
        self._bottom_valid_hist[:, -1] = bottom_valid_now

        stage_gt = self._compute_stage_gt(contact_bits)
        stage_gt_float = stage_gt.to(torch.float32).unsqueeze(-1)
        stage_onehot = F.one_hot(stage_gt, num_classes=4).to(dtype=torch.float32)

        reliability_gt = self._compute_bbox_visible_ratio_gt().unsqueeze(-1)

        self._latest_contact_bits = contact_bits
        self._latest_stage_gt = stage_gt
        self._latest_reliability_gt = reliability_gt.squeeze(-1)
        self._latest_probe_gt = probe_gt
        self._latest_grasp_gt = grasp_gt

        obs["aux_head_input"] = aux_head_input
        obs["aux_reliability_gt"] = reliability_gt
        obs["aux_stage_gt"] = stage_gt_float
        obs["aux_stage_onehot_gt"] = stage_onehot
        obs["aux_contact_bits"] = contact_bits
        obs["aux_probe_gt"] = probe_gt
        obs["aux_grasp_gt"] = grasp_gt
        obs["bottom_valid_hist"] = self._bottom_valid_hist.reshape(self.num_envs, -1)
        obs["inner_valid_hist"] = self._inner_valid_hist.reshape(self.num_envs, -1)
        obs["prev_action"] = self._prev_action_obs

        if hasattr(self, "actions"):
            actions = self.actions.to(device=self.device, dtype=torch.float32).reshape(self.num_envs, -1)
            width = min(actions.shape[1], self._prev_action_obs.shape[1])
            self._prev_action_obs[:, :width] = actions[:, :width]
            if width < self._prev_action_obs.shape[1]:
                self._prev_action_obs[:, width:] = 0.0

        log = self.extras.setdefault("log", {})
        log["aux/tactile_over_thresh_ratio_left_mean"] = self._latest_tactile_over_thresh_ratio_by_sensor[:, 0].mean().detach()
        log["aux/tactile_over_thresh_ratio_right_mean"] = self._latest_tactile_over_thresh_ratio_by_sensor[:, 1].mean().detach()
        log["aux/tactile_over_thresh_ratio_left_down_mean"] = self._latest_tactile_over_thresh_ratio_by_sensor[:, 2].mean().detach()
        log["aux/tactile_over_thresh_ratio_right_down_mean"] = self._latest_tactile_over_thresh_ratio_by_sensor[:, 3].mean().detach()
        log["aux/tactile_over_thresh_ratio_mean"] = self._latest_tactile_over_thresh_ratio_mean.mean().detach()

        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        self._pending_stage_contact_rgb_baseline_refresh[env_ids] = True
        for baseline in self._stage_contact_rgb_baseline.values():
            if baseline is not None:
                baseline[env_ids] = 0.0
        self._latest_contact_bits[env_ids] = 0.0
        self._latest_stage_gt[env_ids] = self.STAGE_APPROACH
        self._latest_reliability_gt[env_ids] = float(getattr(self.cfg, "reliability_fallback_visible_ratio", 0.0))
        self._latest_probe_gt[env_ids] = 0.0
        self._latest_grasp_gt[env_ids] = 0.0
        self._bottom_valid_hist[env_ids] = 0.0
        self._inner_valid_hist[env_ids] = 0.0
        self._latest_tactile_over_thresh_ratio_by_sensor[env_ids] = 0.0
        self._latest_tactile_over_thresh_ratio_mean[env_ids] = 0.0
        self._prev_action_obs[env_ids] = 0.0

        self._object_init_height[env_ids] = (
            torch.full((len(env_ids),), float(CAN_RESET_ROOT_Z), device=self.device)
            + self.scene.env_origins[env_ids, 2]
        )
