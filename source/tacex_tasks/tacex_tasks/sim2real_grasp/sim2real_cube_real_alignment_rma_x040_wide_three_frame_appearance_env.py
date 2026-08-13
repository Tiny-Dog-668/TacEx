"""Three-frame X040-Wide Student with episode-fixed appearance randomization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR, check_file_path

from .sim2real_cube_real_alignment_rma_x040_wide_three_frame_env import (
    Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameStudentDREnv,
    Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameStudentDREnvCfg,
)


_PLATE_APPEARANCE_MATERIALS = (
    {
        "id": "near_black",
        "kind": "preview_surface",
        "weight": 0.15,
        "diffuse_color": (0.006, 0.006, 0.006),
        "metallic": 0.0,
        "roughness": 0.98,
    },
    {
        "id": "charcoal",
        "kind": "preview_surface",
        "weight": 0.15,
        "diffuse_color": (0.03, 0.03, 0.035),
        "metallic": 0.0,
        "roughness": 0.92,
    },
    {
        "id": "dark_brown",
        "kind": "preview_surface",
        "weight": 0.20,
        "diffuse_color": (0.18, 0.07, 0.025),
        "metallic": 0.0,
        "roughness": 0.82,
    },
    {
        "id": "oak",
        "kind": "mdl",
        "weight": 0.20,
        "mdl_path": "Materials/Base/Wood/Oak.mdl",
        "texture_scale": (0.5, 0.5),
    },
    {
        "id": "walnut",
        "kind": "mdl",
        "weight": 0.15,
        "mdl_path": "Materials/Base/Wood/Walnut.mdl",
        "texture_scale": (0.5, 0.5),
    },
    {
        "id": "plywood",
        "kind": "mdl",
        "weight": 0.15,
        "mdl_path": "Materials/Base/Wood/Plywood.mdl",
        "texture_scale": (0.5, 0.5),
    },
)

_BACKDROP_APPEARANCE_MATERIALS = (
    {
        "id": "near_black",
        "kind": "preview_surface",
        "weight": 0.25,
        "diffuse_color": (0.004, 0.004, 0.004),
        "metallic": 0.0,
        "roughness": 0.95,
    },
    {
        "id": "dark_gray",
        "kind": "preview_surface",
        "weight": 0.15,
        "diffuse_color": (0.04, 0.04, 0.045),
        "metallic": 0.0,
        "roughness": 0.98,
    },
    {
        "id": "green_curtain",
        "kind": "preview_surface",
        "weight": 0.20,
        "diffuse_color": (0.03, 0.24, 0.05),
        "metallic": 0.0,
        "roughness": 0.98,
    },
    {
        "id": "cloth_black",
        "kind": "mdl",
        "weight": 0.20,
        "mdl_path": "Materials/Base/Textiles/Cloth_Black.mdl",
        "texture_scale": (0.5, 0.5),
    },
    {
        "id": "cloth_gray",
        "kind": "mdl",
        "weight": 0.20,
        "mdl_path": "Materials/Base/Textiles/Cloth_Gray.mdl",
        "texture_scale": (0.5, 0.5),
    },
)

_CUBE_APPEARANCE_MATERIALS = (
    {
        "id": "near_white",
        "kind": "preview_surface",
        "weight": 0.20,
        "diffuse_color": (0.95, 0.95, 0.94),
        "metallic": 0.0,
        "roughness": 0.88,
    },
    {
        "id": "red",
        "kind": "preview_surface",
        "weight": 0.10,
        "diffuse_color": (0.75, 0.03, 0.03),
        "metallic": 0.0,
        "roughness": 0.65,
    },
    {
        "id": "blue",
        "kind": "preview_surface",
        "weight": 0.10,
        "diffuse_color": (0.03, 0.12, 0.80),
        "metallic": 0.0,
        "roughness": 0.65,
    },
    {
        "id": "green",
        "kind": "preview_surface",
        "weight": 0.10,
        "diffuse_color": (0.03, 0.55, 0.08),
        "metallic": 0.0,
        "roughness": 0.65,
    },
    {
        "id": "yellow",
        "kind": "preview_surface",
        "weight": 0.10,
        "diffuse_color": (0.85, 0.65, 0.03),
        "metallic": 0.0,
        "roughness": 0.65,
    },
    {
        "id": "orange",
        "kind": "preview_surface",
        "weight": 0.10,
        "diffuse_color": (0.90, 0.25, 0.03),
        "metallic": 0.0,
        "roughness": 0.65,
    },
    {
        "id": "plastic",
        "kind": "mdl",
        "weight": 0.10,
        "mdl_path": "Materials/Base/Plastics/Plastic.mdl",
        "texture_scale": (1.0, 1.0),
    },
    {
        "id": "plastic_abs",
        "kind": "mdl",
        "weight": 0.10,
        "mdl_path": "Materials/Base/Plastics/Plastic_ABS.mdl",
        "texture_scale": (1.0, 1.0),
    },
    {
        "id": "rubber_textured",
        "kind": "mdl",
        "weight": 0.10,
        "mdl_path": "Materials/Base/Plastics/Rubber_Textured.mdl",
        "texture_scale": (1.0, 1.0),
    },
)


@configclass
class Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameAppearanceStudentDREnvCfg(
    Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameStudentDREnvCfg
):
    """Three-frame Student with realistic per-episode material randomization."""

    appearance_randomization_enabled = True
    appearance_profile = "x040_three_frame_realistic_material_v2"
    appearance_sampling_frequency = "per_environment_per_episode"
    appearance_sampling_scope = "per_environment"
    appearance_material_instance_scope = "per_environment"
    appearance_full_strength_from_step = 0
    appearance_material_source = "NVIDIA_NUCLEUS_DIR"
    appearance_plate_materials = _PLATE_APPEARANCE_MATERIALS
    appearance_backdrop_materials = _BACKDROP_APPEARANCE_MATERIALS
    appearance_cube_materials = _CUBE_APPEARANCE_MATERIALS

    # The inherited DR implementation mutates PreviewSurface diffuse colors.
    # This profile instead binds one immutable per-env material-bank entry per asset.
    plate_color_randomization_enabled = False
    backdrop_color_randomization_enabled = False


class Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameAppearanceStudentDREnv(
    Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameStudentDREnv
):
    """Bind independent plate, backdrop and active-cube materials on reset."""

    cfg: Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameAppearanceStudentDREnvCfg

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._setup_appearance_material_bank()

    @staticmethod
    def _validate_material_entries(
        category: str,
        entries: Sequence[Mapping[str, Any]],
    ) -> None:
        if not entries:
            raise ValueError(f"Appearance material category is empty: {category}")
        identifiers = [str(entry["id"]) for entry in entries]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Appearance material IDs must be unique in {category}")
        total_weight = sum(float(entry["weight"]) for entry in entries)
        if any(float(entry["weight"]) <= 0.0 for entry in entries):
            raise ValueError(f"Appearance material weights must be positive in {category}")
        if abs(total_weight - 1.0) > 1.0e-6:
            raise ValueError(
                f"Appearance material weights must sum to 1 in {category}, got {total_weight}"
            )
        for entry in entries:
            kind = str(entry["kind"])
            if kind not in ("preview_surface", "mdl"):
                raise ValueError(f"Unsupported appearance material kind: {kind}")
            if kind == "preview_surface":
                color = tuple(float(value) for value in entry["diffuse_color"])
                if len(color) != 3 or any(value < 0.0 or value > 1.0 for value in color):
                    raise ValueError(f"Invalid PreviewSurface color in {category}: {color}")

    def _spawn_appearance_material(
        self,
        env_id: int,
        category: str,
        entry: Mapping[str, Any],
    ) -> str:
        material_path = (
            f"/World/Looks/X040Appearance/env_{env_id}/{category}/{entry['id']}"
        )
        stage = sim_utils.stage_utils.get_current_stage()
        if stage.GetPrimAtPath(material_path).IsValid():
            return material_path

        if entry["kind"] == "preview_surface":
            material_cfg = sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(float(value) for value in entry["diffuse_color"]),
                metallic=float(entry["metallic"]),
                roughness=float(entry["roughness"]),
            )
        else:
            mdl_path = f"{NVIDIA_NUCLEUS_DIR}/{entry['mdl_path']}"
            if check_file_path(mdl_path) == 0:
                raise FileNotFoundError(
                    f"Required X040 appearance MDL is unavailable: {mdl_path}"
                )
            material_cfg = sim_utils.MdlFileCfg(
                mdl_path=mdl_path,
                project_uvw=True,
                texture_scale=tuple(float(value) for value in entry["texture_scale"]),
            )
        material_cfg.func(material_path, material_cfg)
        return material_path

    def _setup_appearance_material_bank(self) -> None:
        categories = {
            "plate": self.cfg.appearance_plate_materials,
            "backdrop": self.cfg.appearance_backdrop_materials,
            "cube": self.cfg.appearance_cube_materials,
        }
        self._appearance_material_paths: dict[str, list[list[str]]] = {}
        self._appearance_material_weights: dict[str, torch.Tensor] = {}
        for category, entries in categories.items():
            self._validate_material_entries(category, entries)
            self._appearance_material_paths[category] = [
                [
                    self._spawn_appearance_material(env_id, category, entry)
                    for entry in entries
                ]
                for env_id in range(self.num_envs)
            ]
            self._appearance_material_weights[category] = torch.tensor(
                [float(entry["weight"]) for entry in entries],
                device=self.device,
                dtype=torch.float32,
            )

        self._appearance_plate_material_ids = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._appearance_backdrop_material_ids = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._appearance_cube_material_ids = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._appearance_cube_bucket_ids = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )

    def _sample_appearance_material_ids(
        self,
        category: str,
        count: int,
    ) -> torch.Tensor:
        return torch.multinomial(
            self._appearance_material_weights[category],
            num_samples=count,
            replacement=True,
        )

    def _validate_requested_material_ids(
        self,
        category: str,
        env_ids: torch.Tensor,
        material_ids: torch.Tensor,
    ) -> torch.Tensor:
        material_ids = material_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if material_ids.shape != env_ids.shape:
            raise ValueError(
                f"{category} material IDs must have shape {tuple(env_ids.shape)}, "
                f"got {tuple(material_ids.shape)}"
            )
        if material_ids.numel() > 0 and (
            int(material_ids.min().item()) < 0
            or int(material_ids.max().item())
            >= len(self._appearance_material_paths[category][0])
        ):
            raise ValueError(f"{category} material ID is outside the configured bank")
        return material_ids

    def _apply_appearance_material_ids(
        self,
        env_ids: torch.Tensor,
        plate_material_ids: torch.Tensor,
        backdrop_material_ids: torch.Tensor,
        cube_material_ids: torch.Tensor,
    ) -> None:
        """Bind requested material IDs to K reset environments and active cubes."""
        env_ids = self._normalize_dr_env_ids(env_ids)
        plate_material_ids = self._validate_requested_material_ids(
            "plate", env_ids, plate_material_ids
        )
        backdrop_material_ids = self._validate_requested_material_ids(
            "backdrop", env_ids, backdrop_material_ids
        )
        cube_material_ids = self._validate_requested_material_ids(
            "cube", env_ids, cube_material_ids
        )
        plate_roots = self._plate.root_physx_view.prim_paths
        backdrop_roots = self._backdrop.root_physx_view.prim_paths

        for local_index, env_id in enumerate(env_ids.detach().cpu().tolist()):
            plate_id = int(plate_material_ids[local_index].item())
            if int(self._appearance_plate_material_ids[env_id].item()) != plate_id:
                target = self._resolve_visual_target_prim(plate_roots[env_id])
                sim_utils.bind_visual_material(
                    target,
                    self._appearance_material_paths["plate"][env_id][plate_id],
                )
                self._appearance_plate_material_ids[env_id] = plate_id

            backdrop_id = int(backdrop_material_ids[local_index].item())
            if int(self._appearance_backdrop_material_ids[env_id].item()) != backdrop_id:
                target = self._resolve_visual_target_prim(backdrop_roots[env_id])
                sim_utils.bind_visual_material(
                    target,
                    self._appearance_material_paths["backdrop"][env_id][backdrop_id],
                )
                self._appearance_backdrop_material_ids[env_id] = backdrop_id

            cube_id = int(cube_material_ids[local_index].item())
            bucket_id = int(self._active_cube_bucket_ids[env_id].item())
            cube_binding_changed = (
                int(self._appearance_cube_material_ids[env_id].item()) != cube_id
                or int(self._appearance_cube_bucket_ids[env_id].item()) != bucket_id
            )
            if cube_binding_changed:
                cube_root = f"/World/envs/env_{env_id}/cube_size_bucket_{bucket_id}"
                target = self._resolve_visual_target_prim(cube_root)
                sim_utils.bind_visual_material(
                    target,
                    self._appearance_material_paths["cube"][env_id][cube_id],
                )
                self._appearance_cube_material_ids[env_id] = cube_id
                self._appearance_cube_bucket_ids[env_id] = bucket_id

    def _randomize_scene_visuals(self, env_ids: torch.Tensor) -> None:
        super()._randomize_scene_visuals(env_ids)
        if not bool(self.cfg.appearance_randomization_enabled):
            return
        env_ids = self._normalize_dr_env_ids(env_ids)
        count = int(env_ids.numel())
        if count == 0:
            return
        self._apply_appearance_material_ids(
            env_ids,
            self._sample_appearance_material_ids("plate", count),
            self._sample_appearance_material_ids("backdrop", count),
            self._sample_appearance_material_ids("cube", count),
        )


__all__ = (
    "Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameAppearanceStudentDREnvCfg",
    "Sim2RealCubeRealAlignmentRMAX040WideSizeBucketsThreeFrameAppearanceStudentDREnv",
)
