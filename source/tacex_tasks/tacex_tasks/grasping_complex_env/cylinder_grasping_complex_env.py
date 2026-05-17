"""Complex cylinder grasping env with extra objects and per-env visual randomization."""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from tacex_tasks.cylinder_grasping.cylinder_grasping_one_vision_two_tactile import (
    CylinderGraspingVisionOnlyCfg,
    CylinderGraspingVisionOnlyEnv,
)


@configclass
class CylinderGraspingComplexEnvCfg(CylinderGraspingVisionOnlyCfg):
    """Extension of vision+tactile env with extra objects and visual randomization."""

    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)

    # allow per-env material edits
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=1.5,
        replicate_physics=False,
        lazy_sensor_update=True,
    )

    # extra objects
    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.05, 0.02]),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=1.0,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0005,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.8),
                metallic=0.0,
                roughness=0.6,
            ),
        ),
    )

    sphere = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sphere",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.6, -0.05, 0.02]),
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=1.0,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0005,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.2),
                metallic=0.0,
                roughness=0.4,
            ),
        ),
    )


class CylinderGraspingComplexEnv(CylinderGraspingVisionOnlyEnv):
    """Cylinder grasping env with extra objects and visual randomization."""

    cfg: CylinderGraspingComplexEnvCfg

    def __init__(self, cfg: CylinderGraspingComplexEnvCfg, render_mode: str | None = None, **kwargs):
        self._materials_randomized = False
        super().__init__(cfg, render_mode, **kwargs)

    # ---------------- Scene ---------------- #
    def _setup_scene(self):
        super()._setup_scene()

        self._cube = RigidObject(self.cfg.cube)
        self.scene.rigid_objects["cube"] = self._cube

        self._sphere = RigidObject(self.cfg.sphere)
        self.scene.rigid_objects["sphere"] = self._sphere

    # ---------------- Helpers ---------------- #
    def _random_color(self, low=0.1, high=0.9):
        return tuple(float(c) for c in np.random.uniform(low, high, size=3))

    def _random_wood(self):
        base = np.array([0.65, 0.45, 0.30], dtype=np.float32)
        jitter = np.random.uniform(-0.2, 0.2, size=3)
        color = np.clip(base + jitter, 0.05, 0.95)
        return tuple(float(c) for c in color)

    def _apply_material(self, material_path: str, target_prim_path: str, color, roughness, metallic):
        stage = sim_utils.stage_utils.get_current_stage()
        try:
            prim = stage.GetPrimAtPath(material_path)
            if not prim.IsValid():
                sim_utils.spawn_preview_surface(material_path, sim_utils.PreviewSurfaceCfg())
            sim_utils.bind_visual_material(target_prim_path, material_path)
            shader_prim = stage.GetPrimAtPath(f"{material_path}/Shader")
            if shader_prim.IsValid():
                sim_utils.safe_set_attribute_on_usd_prim(shader_prim, "inputs:diffuse_color", color, camel_case=True)
                sim_utils.safe_set_attribute_on_usd_prim(shader_prim, "inputs:roughness", float(roughness), camel_case=True)
                sim_utils.safe_set_attribute_on_usd_prim(shader_prim, "inputs:metallic", float(metallic), camel_case=True)
        except Exception as exc:
            print(f"[WARN] material update failed for {material_path}: {exc}")

    def _randomize_visuals_once(self, env_ids):
        if self._materials_randomized:
            return
        env_id_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        for env_id in env_id_list:
            # cylinder
            cyl_root = self._cylinder.root_physx_view.prim_paths[env_id]
            self._apply_material(
                f"{cyl_root}/geometry/material",
                f"{cyl_root}/geometry/mesh",
                color=self._random_color(0.15, 0.9),
                roughness=np.random.uniform(0.1, 0.8),
                metallic=np.random.uniform(0.0, 0.6),
            )
            # plate
            plate_root = self._plate.root_physx_view.prim_paths[env_id]
            self._apply_material(
                f"{plate_root}/material",
                plate_root,
                color=self._random_wood(),
                roughness=np.random.uniform(0.3, 0.9),
                metallic=np.random.uniform(0.0, 0.2),
            )
            # cube
            cube_root = self._cube.root_physx_view.prim_paths[env_id]
            self._apply_material(
                f"{cube_root}/geometry/material",
                f"{cube_root}/geometry/mesh",
                color=self._random_color(0.1, 0.9),
                roughness=np.random.uniform(0.2, 0.8),
                metallic=np.random.uniform(0.0, 0.6),
            )
            # sphere
            sphere_root = self._sphere.root_physx_view.prim_paths[env_id]
            self._apply_material(
                f"{sphere_root}/geometry/material",
                f"{sphere_root}/geometry/mesh",
                color=self._random_color(0.1, 0.9),
                roughness=np.random.uniform(0.2, 0.7),
                metallic=np.random.uniform(0.0, 0.6),
            )
        self._materials_randomized = True

    # ---------------- Reset ---------------- #
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        # reset extra objects
        self._cube.reset(env_ids)
        self._sphere.reset(env_ids)

        env_origins = self.scene.env_origins[env_ids].to(device=self.device)

        jitter_xy = (torch.rand(len(env_ids), 2, device=self.device) - 0.5) * 0.04  # ±2 cm

        cube_pos_local = torch.tensor([0.4, 0.05, 0.02], device=self.device).expand(len(env_ids), 3).clone()
        cube_pos_local[:, :2] = cube_pos_local[:, :2] + jitter_xy
        cube_pos = env_origins + cube_pos_local
        cube_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        cube_pose = torch.cat([cube_pos, cube_rot], dim=-1)
        self._cube.write_root_pose_to_sim(cube_pose, env_ids)
        self._cube.write_root_velocity_to_sim(torch.zeros_like(cube_pose[:, :6]), env_ids)

        sphere_pos_local = torch.tensor([0.6, -0.05, 0.02], device=self.device).expand(len(env_ids), 3).clone()
        sphere_pos_local[:, :2] = sphere_pos_local[:, :2] + jitter_xy
        sphere_pos = env_origins + sphere_pos_local
        sphere_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        sphere_pose = torch.cat([sphere_pos, sphere_rot], dim=-1)
        self._sphere.write_root_pose_to_sim(sphere_pose, env_ids)
        self._sphere.write_root_velocity_to_sim(torch.zeros_like(sphere_pose[:, :6]), env_ids)

        # apply visuals once
        self._randomize_visuals_once(env_ids)
