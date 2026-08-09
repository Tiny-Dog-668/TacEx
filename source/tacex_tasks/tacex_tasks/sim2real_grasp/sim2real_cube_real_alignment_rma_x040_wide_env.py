"""Independent X040-Wide RMA profile with XYZ-only privileged Teacher input."""

from __future__ import annotations

import colorsys

import gymnasium as gym
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from pxr import UsdShade

from .sim2real_cube_real_alignment_rma_env import (
    _CRITIC_OBSERVATION_SPACE,
    _make_rma_cube_cfg,
    _make_rma_cube_contact_sensor_cfg,
    _make_rma_robot_cfg,
    Sim2RealCubeRealAlignmentRMAStudentDREnv,
    Sim2RealCubeRealAlignmentRMAStudentDREnvCfg,
    Sim2RealCubeRealAlignmentRMATeacherEnv,
    Sim2RealCubeRealAlignmentRMATeacherEnvCfg,
)


class _X040WideMixin:
    """Add the profile-only TCP clearance reward without changing termination."""

    def _compute_additional_reward(self):
        reward, log = super()._compute_additional_reward()
        tcp_z = self._compute_reach_center_world()[:, 2]
        violation = tcp_z < float(self.cfg.tcp_table_clearance_min_z_m)
        penalty = violation.to(tcp_z.dtype) * float(self.cfg.tcp_table_clearance_penalty)
        self._last_tcp_table_clearance_violation = violation.detach().clone()
        self._last_tcp_height_m = tcp_z.detach().clone()
        log.update({
            "reward/tcp_table_clearance": penalty.mean().detach(),
            "info/tcp_table_clearance_violation_fraction": violation.float().mean().detach(),
            "info/tcp_height_min_m": tcp_z.min().detach(),
        })
        return reward + penalty, log


@configclass
class Sim2RealCubeRealAlignmentRMAX040WideTeacherEnvCfg(Sim2RealCubeRealAlignmentRMATeacherEnvCfg):
    """Privileged Teacher with cube XYZ but no contact observation."""

    robot = _make_rma_robot_cfg()
    cube = _make_rma_cube_cfg()
    rma_cube_contact_sensor = _make_rma_cube_contact_sensor_cfg()
    cube.init_state.pos = (0.40, 0.00, 0.026)
    cube_x_pos_range = 0.08
    cube_y_pos_range = 0.10
    cube_position_curriculum_enabled = False
    cube_position_curriculum_force_full_range = True
    tcp_table_clearance_min_z_m = 0.011
    tcp_table_clearance_penalty = -10.0
    rma_actor_feature_dim = 28
    rma_object_pose_components = "position_xyz_only"
    rma_contact_components = "none"
    observation_space = {"proprio_obs": 15, "action_history": 4, "rma_cube_pos": 3, **_CRITIC_OBSERVATION_SPACE}


class Sim2RealCubeRealAlignmentRMAX040WideTeacherEnv(_X040WideMixin, Sim2RealCubeRealAlignmentRMATeacherEnv):
    cfg: Sim2RealCubeRealAlignmentRMAX040WideTeacherEnvCfg

    def _get_observations(self):
        observations = super()._get_observations()
        observations["policy"].pop("rma_contact_state", None)
        return observations


@configclass
class Sim2RealCubeRealAlignmentRMAX040WideStudentDREnvCfg(Sim2RealCubeRealAlignmentRMAStudentDREnvCfg):
    """Visual Student DR environment; cube XYZ is a training-only label."""

    robot = _make_rma_robot_cfg()
    cube = _make_rma_cube_cfg()
    rma_cube_contact_sensor = _make_rma_cube_contact_sensor_cfg()
    cube.init_state.pos = (0.40, 0.00, 0.026)
    cube_x_pos_range = 0.08
    cube_y_pos_range = 0.10
    cube_position_curriculum_enabled = False
    cube_position_curriculum_force_full_range = True
    camera_position_delta_max_m = (0.01, 0.01, 0.01)
    camera_rotation_delta_max_deg = (2.0, 2.0, 2.0)
    # This is the visible vertical status strip on the robot base. It is a
    # GeomSubset below an instanceable visual prim in the referenced Panda USD,
    # so only that visual branch is made editable before rebinding the subset.
    student_base_led_subset_path = "panda_link0/visuals/panda_link0/subset_5"
    student_base_led_emissive_color = (0.0, 1.0, 0.0)
    student_base_led_hue_deg_range = (105.0, 135.0)
    student_base_led_saturation_range = (0.75, 1.0)
    student_base_led_value_range = (0.35, 1.0)
    tcp_table_clearance_min_z_m = 0.011
    tcp_table_clearance_penalty = -10.0
    rma_actor_feature_dim = 28
    rma_object_pose_components = "position_xyz_only"
    rma_contact_components = "none"
    observation_space = {
        "proprio_obs": 15, "action_history": 4,
        "wrist_rgb": gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        "rma_cube_pos": 3, **_CRITIC_OBSERVATION_SPACE,
    }


class Sim2RealCubeRealAlignmentRMAX040WideStudentDREnv(_X040WideMixin, Sim2RealCubeRealAlignmentRMAStudentDREnv):
    cfg: Sim2RealCubeRealAlignmentRMAX040WideStudentDREnvCfg

    def _validate_student_base_led_randomization(self) -> None:
        hue_low, hue_high = self.cfg.student_base_led_hue_deg_range
        saturation_low, saturation_high = self.cfg.student_base_led_saturation_range
        value_low, value_high = self.cfg.student_base_led_value_range
        if not 0.0 <= float(hue_low) <= float(hue_high) <= 360.0:
            raise ValueError("Student base LED hue range must lie within [0, 360] degrees")
        for name, low, high in (
            ("saturation", saturation_low, saturation_high),
            ("value", value_low, value_high),
        ):
            if not 0.0 <= float(low) <= float(high) <= 1.0:
                raise ValueError(f"Student base LED {name} range must lie within [0, 1]")

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._validate_student_base_led_randomization()

        # Do not edit the referenced official Panda USD.  This task-local
        # per-environment material is bound solely to the base LED subset; the
        # wrist LED remains bound to the asset's original blue material.
        stage = sim_utils.stage_utils.get_current_stage()
        self._student_base_led_material_paths: list[str] = []
        for env_id in range(self.cfg.scene.num_envs):
            material_path = f"/World/Looks/X040WideStudentBaseVerticalLed_env_{env_id}"
            if not stage.GetPrimAtPath(material_path).IsValid():
                sim_utils.spawn_preview_surface(
                    material_path,
                    sim_utils.PreviewSurfaceCfg(
                        diffuse_color=self.cfg.student_base_led_emissive_color,
                        emissive_color=self.cfg.student_base_led_emissive_color,
                        roughness=0.298,
                    ),
                )
            self._student_base_led_material_paths.append(material_path)
            target_path = (
                f"/World/envs/env_{env_id}/Robot/"
                f"{self.cfg.student_base_led_subset_path}"
            )
            # Input target hierarchy:
            #   .../panda_link0/visuals/panda_link0/subset_5 (instance proxy)
            # Editable output hierarchy after this call has the same path but
            # is no longer backed by the ``visuals`` instance proxy.
            visual_instance_path = target_path.rsplit("/", maxsplit=2)[0]
            if not stage.GetPrimAtPath(visual_instance_path).IsValid():
                raise RuntimeError(
                    f"Student base visual instance is missing: {visual_instance_path}"
                )
            sim_utils.make_uninstanceable(visual_instance_path, stage=stage)
            if not stage.GetPrimAtPath(target_path).IsValid():
                raise RuntimeError(f"Student base LED subset is missing: {target_path}")
            if stage.GetPrimAtPath(target_path).IsInstanceProxy():
                raise RuntimeError(f"Student base LED subset is still instance-backed: {target_path}")
            sim_utils.bind_visual_material(target_path, material_path, stage=stage)
            bound_material, _ = UsdShade.MaterialBindingAPI(
                stage.GetPrimAtPath(target_path)
            ).ComputeBoundMaterial()
            if not bound_material or bound_material.GetPath().pathString != material_path:
                raise RuntimeError(f"Failed to bind Student base LED material: {target_path}")

    def _randomize_student_base_led(self, env_ids: torch.Tensor) -> None:
        """Sample one independent HSV color per reset environment and episode."""
        env_ids = self._normalize_dr_env_ids(env_ids)
        count = int(env_ids.numel())
        if count == 0:
            return

        scale = self._dr_curriculum_scale()
        hue_low, hue_high = self._curriculum_bounds(
            *self.cfg.student_base_led_hue_deg_range,
            center=120.0,
            scale=scale,
        )
        saturation_low, saturation_high = self._curriculum_bounds(
            *self.cfg.student_base_led_saturation_range,
            center=1.0,
            scale=scale,
        )
        value_low, value_high = self._curriculum_bounds(
            *self.cfg.student_base_led_value_range,
            center=1.0,
            scale=scale,
        )

        # [K,3] uniform samples map to HSV for K reset environments. USD
        # material edits are CPU-side, so conversion happens in the same loop.
        samples = torch.rand((count, 3), device=self.device).detach().cpu().tolist()
        stage = sim_utils.stage_utils.get_current_stage()
        for env_id, sample in zip(env_ids.detach().cpu().tolist(), samples):
            hue_deg = hue_low + sample[0] * (hue_high - hue_low)
            saturation = saturation_low + sample[1] * (saturation_high - saturation_low)
            value = value_low + sample[2] * (value_high - value_low)
            rgb = colorsys.hsv_to_rgb(hue_deg / 360.0, saturation, value)
            shader_path = f"{self._student_base_led_material_paths[env_id]}/Shader"
            shader_prim = stage.GetPrimAtPath(shader_path)
            if not shader_prim.IsValid():
                raise RuntimeError(f"Student base LED shader is missing: {shader_path}")
            for attribute_name in ("inputs:diffuse_color", "inputs:emissive_color"):
                sim_utils.safe_set_attribute_on_usd_prim(
                    shader_prim,
                    attribute_name,
                    rgb,
                    camel_case=True,
                )

    def _randomize_scene_visuals(self, env_ids: torch.Tensor) -> None:
        super()._randomize_scene_visuals(env_ids)
        self._randomize_student_base_led(env_ids)

    def _get_observations(self):
        observations = super()._get_observations()
        observations["policy"].pop("rma_contact_state", None)
        return observations
