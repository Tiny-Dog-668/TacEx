"""Simple per-environment appearance randomization without streamed MDL materials."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from .sim2real_cube_real_alignment_rma_x040_wide_static_size_buckets_env import (
    Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsThreeFrameStudentDREnv,
    Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsThreeFrameStudentDREnvCfg,
)


_PLATE_COLORS = (
    {"id": "near_black", "kind": "preview_surface", "weight": 0.20, "diffuse_color": (0.006, 0.006, 0.006), "metallic": 0.0, "roughness": 0.98},
    {"id": "charcoal", "kind": "preview_surface", "weight": 0.20, "diffuse_color": (0.03, 0.03, 0.035), "metallic": 0.0, "roughness": 0.92},
    {"id": "dark_brown", "kind": "preview_surface", "weight": 0.20, "diffuse_color": (0.18, 0.07, 0.025), "metallic": 0.0, "roughness": 0.82},
    {"id": "oak", "kind": "preview_surface", "weight": 0.20, "diffuse_color": (0.45, 0.24, 0.09), "metallic": 0.0, "roughness": 0.78},
    {"id": "plywood", "kind": "preview_surface", "weight": 0.20, "diffuse_color": (0.62, 0.42, 0.20), "metallic": 0.0, "roughness": 0.84},
)

_BACKDROP_COLORS = (
    {"id": "near_black", "kind": "preview_surface", "weight": 0.25, "diffuse_color": (0.004, 0.004, 0.004), "metallic": 0.0, "roughness": 0.98},
    {"id": "dark_gray", "kind": "preview_surface", "weight": 0.25, "diffuse_color": (0.04, 0.04, 0.045), "metallic": 0.0, "roughness": 0.98},
    {"id": "mid_gray", "kind": "preview_surface", "weight": 0.25, "diffuse_color": (0.16, 0.16, 0.17), "metallic": 0.0, "roughness": 0.95},
    {"id": "green", "kind": "preview_surface", "weight": 0.25, "diffuse_color": (0.03, 0.24, 0.05), "metallic": 0.0, "roughness": 0.98},
)

_CUBE_COLORS = (
    {"id": "white", "kind": "preview_surface", "weight": 0.20, "diffuse_color": (0.95, 0.95, 0.94), "metallic": 0.0, "roughness": 0.75},
    {"id": "red", "kind": "preview_surface", "weight": 0.16, "diffuse_color": (0.75, 0.03, 0.03), "metallic": 0.0, "roughness": 0.65},
    {"id": "blue", "kind": "preview_surface", "weight": 0.16, "diffuse_color": (0.03, 0.12, 0.80), "metallic": 0.0, "roughness": 0.65},
    {"id": "green", "kind": "preview_surface", "weight": 0.16, "diffuse_color": (0.03, 0.55, 0.08), "metallic": 0.0, "roughness": 0.65},
    {"id": "yellow", "kind": "preview_surface", "weight": 0.16, "diffuse_color": (0.85, 0.65, 0.03), "metallic": 0.0, "roughness": 0.65},
    {"id": "orange", "kind": "preview_surface", "weight": 0.16, "diffuse_color": (0.90, 0.25, 0.03), "metallic": 0.0, "roughness": 0.65},
)


@configclass
class Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameIndependentAppearanceStudentDREnvCfg(
    Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsThreeFrameStudentDREnvCfg
):
    """Per-env PreviewSurface colors updated only by that environment's reset."""

    appearance_randomization_enabled = True
    appearance_profile = "x040_three_frame_independent_preview_surface_v5"
    appearance_sampling_frequency = "per_environment_on_actual_done_reset"
    appearance_sampling_scope = "per_environment"
    appearance_material_instance_scope = "per_environment_unique_prim"
    appearance_full_strength_from_step = 0
    appearance_material_source = "UsdPreviewSurface"
    appearance_plate_materials = _PLATE_COLORS
    appearance_backdrop_materials = _BACKDROP_COLORS
    appearance_cube_materials = _CUBE_COLORS
    appearance_timeout_termination_enabled = True

    # Shared stage lighting and inherited plate/backdrop mutations would couple envs.
    light_randomization_enabled = False
    plate_color_randomization_enabled = False
    backdrop_color_randomization_enabled = False


class Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameIndependentAppearanceStudentDREnv(
    Sim2RealCubeRealAlignmentRMAX040WideStaticSizeBucketsThreeFrameStudentDREnv
):
    """Own one plate, backdrop, and cube PreviewSurface material per environment."""

    cfg: Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameIndependentAppearanceStudentDREnvCfg

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._setup_independent_appearance()

    def _setup_independent_appearance(self) -> None:
        categories = {
            "plate": self.cfg.appearance_plate_materials,
            "backdrop": self.cfg.appearance_backdrop_materials,
            "cube": self.cfg.appearance_cube_materials,
        }
        self._independent_material_paths: dict[str, list[str]] = {
            category: [] for category in categories
        }
        self._appearance_palettes: dict[str, torch.Tensor] = {}
        self._appearance_weights: dict[str, torch.Tensor] = {}
        for category, entries in categories.items():
            self._validate_palette(category, entries)
            self._appearance_palettes[category] = torch.tensor(
                [entry["diffuse_color"] for entry in entries],
                device=self.device,
                dtype=torch.float32,
            )
            self._appearance_weights[category] = torch.tensor(
                [entry["weight"] for entry in entries],
                device=self.device,
                dtype=torch.float32,
            )
            for env_id in range(self.num_envs):
                # Keep materials outside the cloned env hierarchy. Authoring a
                # child below env_0 propagates through the Isaac cloner and is
                # therefore not an independent per-env material definition.
                path = (
                    f"/World/Looks/X040IndependentAppearance/env_{env_id}/{category}"
                )
                sim_utils.spawn_preview_surface(
                    path,
                    sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(entries[0]["diffuse_color"]),
                        metallic=float(entries[0]["metallic"]),
                        roughness=float(entries[0]["roughness"]),
                    ),
                )
                self._independent_material_paths[category].append(path)

        self._appearance_plate_material_ids = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._appearance_backdrop_material_ids = torch.full_like(
            self._appearance_plate_material_ids, -1
        )
        self._appearance_cube_material_ids = torch.full_like(
            self._appearance_plate_material_ids, -1
        )
        self._appearance_cube_bucket_ids = torch.full_like(
            self._appearance_plate_material_ids, -1
        )
        self._appearance_material_bound = {
            category: torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            for category in categories
        }
        # Material relationship edits on a live rigid body force USD-to-PhysX
        # synchronization, which is illegal with Direct GPU API enabled. Bind
        # every fixed target once here, before simulation starts; episode
        # resets then update shader colors only and never touch relationships.
        for env_id in range(self.num_envs):
            root_paths = {
                "plate": f"/World/envs/env_{env_id}/floor_panel",
                "backdrop": (
                    f"/World/envs/env_{env_id}/real_alignment_backdrop"
                ),
                "cube": self._active_cube_root_prim_path(env_id),
            }
            for category, root_path in root_paths.items():
                sim_utils.bind_visual_material(
                    self._resolve_visual_target_prim(root_path),
                    self._independent_material_paths[category][env_id],
                )
                self._appearance_material_bound[category][env_id] = True
        # The inherited constructor invokes scene randomization once after
        # setup, before the caller's explicit env.reset(). Avoid an observable
        # bootstrap recolor followed immediately by the real initial reset.
        self._appearance_bootstrap_pending = True

    @staticmethod
    def _validate_palette(
        category: str,
        entries: Sequence[Mapping[str, object]],
    ) -> None:
        if not entries:
            raise ValueError(f"Empty independent appearance palette: {category}")
        total_weight = sum(float(entry["weight"]) for entry in entries)
        if abs(total_weight - 1.0) > 1.0e-6:
            raise ValueError(f"Appearance weights must sum to 1 for {category}")
        if any(entry["kind"] != "preview_surface" for entry in entries):
            raise ValueError("Independent appearance accepts PreviewSurface only")

    def _sample_color_ids(self, category: str, count: int) -> torch.Tensor:
        return torch.multinomial(
            self._appearance_weights[category],
            num_samples=count,
            replacement=True,
        )

    def _set_env_material_color(
        self,
        category: str,
        env_id: int,
        material_id: int,
    ) -> None:
        path = self._independent_material_paths[category][env_id]
        color = tuple(
            float(value)
            for value in self._appearance_palettes[category][material_id]
            .detach()
            .cpu()
            .tolist()
        )
        shader_path = f"{path}/Shader"
        sim_utils.safe_set_attribute_on_usd_prim(
            sim_utils.stage_utils.get_current_stage().GetPrimAtPath(shader_path),
            "inputs:diffuse_color",
            color,
            camel_case=True,
        )

    def _apply_independent_colors(
        self,
        env_ids: torch.Tensor,
        plate_ids: torch.Tensor,
        backdrop_ids: torch.Tensor,
        cube_ids: torch.Tensor,
    ) -> None:
        env_ids = self._normalize_dr_env_ids(env_ids)
        requested = {
            "plate": plate_ids.to(self.device, dtype=torch.long).reshape(-1),
            "backdrop": backdrop_ids.to(self.device, dtype=torch.long).reshape(-1),
            "cube": cube_ids.to(self.device, dtype=torch.long).reshape(-1),
        }
        if any(ids.shape != env_ids.shape for ids in requested.values()):
            raise ValueError("Independent appearance IDs must match env_ids shape")

        plate_roots = self._plate.root_physx_view.prim_paths
        backdrop_roots = self._backdrop.root_physx_view.prim_paths
        id_buffers = {
            "plate": self._appearance_plate_material_ids,
            "backdrop": self._appearance_backdrop_material_ids,
            "cube": self._appearance_cube_material_ids,
        }
        for local_index, env_id in enumerate(env_ids.detach().cpu().tolist()):
            for category, roots in (
                ("plate", plate_roots),
                ("backdrop", backdrop_roots),
            ):
                material_id = int(requested[category][local_index].item())
                self._set_env_material_color(category, env_id, material_id)
                if not bool(self._appearance_material_bound[category][env_id].item()):
                    sim_utils.bind_visual_material(
                        self._resolve_visual_target_prim(roots[env_id]),
                        self._independent_material_paths[category][env_id],
                    )
                    self._appearance_material_bound[category][env_id] = True
                id_buffers[category][env_id] = material_id

            cube_id = int(requested["cube"][local_index].item())
            bucket_id = int(self._active_cube_bucket_ids[env_id].item())
            self._set_env_material_color("cube", env_id, cube_id)
            if not bool(self._appearance_material_bound["cube"][env_id].item()):
                cube_root = self._active_cube_root_prim_path(env_id)
                sim_utils.bind_visual_material(
                    self._resolve_visual_target_prim(cube_root),
                    self._independent_material_paths["cube"][env_id],
                )
                self._appearance_material_bound["cube"][env_id] = True
            self._appearance_cube_material_ids[env_id] = cube_id
            self._appearance_cube_bucket_ids[env_id] = bucket_id

    def _randomize_scene_visuals(self, env_ids: torch.Tensor) -> None:
        # Inherited camera/image DR remains per-env. Shared DomeLight and inherited
        # plate/backdrop mutations are disabled by this config.
        super()._randomize_scene_visuals(env_ids)
        env_ids = self._normalize_dr_env_ids(env_ids)
        if self._appearance_bootstrap_pending:
            self._appearance_bootstrap_pending = False
            return
        count = int(env_ids.numel())
        if count == 0:
            return
        self._apply_independent_colors(
            env_ids,
            self._sample_color_ids("plate", count),
            self._sample_color_ids("backdrop", count),
            self._sample_color_ids("cube", count),
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminate on collision and truncate each env at its episode horizon."""
        time_out = (
            self.episode_length_buf >= self.max_episode_length - 1
        ).bool()
        current_center_height = self._cube.data.root_pos_w[:, 2]
        upright_cos = self._compute_cube_upright_cos(self._cube.data.root_quat_w)
        _, _, above_success_height = self._compute_cube_lift_terms(
            current_center_height, upright_cos
        )
        self._success_hold_counter = torch.where(
            above_success_height,
            self._success_hold_counter + 1,
            torch.zeros_like(self._success_hold_counter),
        )
        success = self._success_hold_counter >= int(self.cfg.success_hold_steps)
        self._rma_episode_success_ever |= success

        joint_z_positions = self._robot.data.body_link_pos_w[:, :, 2]
        collision_with_ground = torch.any(
            joint_z_positions < self.cfg.ground_height,
            dim=1,
        ).bool()
        terminated = collision_with_ground
        completed = terminated | time_out
        self._record_episode_outcomes_for_step(
            completed_count=completed.to(dtype=torch.long).sum(),
            success_count=(completed & self._rma_episode_success_ever)
            .to(dtype=torch.long)
            .sum(),
        )
        self._publish_episode_success_statistics()
        self._last_rma_success_nonterminal = success.detach().clone()
        return terminated, time_out


__all__ = (
    "Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameIndependentAppearanceStudentDREnvCfg",
    "Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameIndependentAppearanceStudentDREnv",
)
