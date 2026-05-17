"""Spawn a wooden board with a cube on top in Isaac Sim.

Usage:
    isaaclab -p scripts/cylinder_grasping/cube_shape_test.py
"""

from __future__ import annotations

import argparse
import random
from typing import List, Tuple

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Spawn a wooden board and a cube.")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

# USD / PhysX APIs (Isaac Sim provides these)
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema


def add_random_protrusions_to_cube(
    cube_prim_path: str,
    cube_size: Tuple[float, float, float],
    num_bumps: int = 12,
    radius_range: Tuple[float, float] = (0.003, 0.007),
    gap: float = 0.001,
    seed: int | None = None,
    enable_visual: bool = True,
) -> None:
    """Add random spherical protrusions (collision spheres) on the surface of a cube rigid body.

    Implementation detail:
    - The spheres are created as *child prims* under the cube prim.
    - We apply CollisionAPI (no rigid body API), so they become compound colliders of the cube body.
    """
    if seed is not None:
        random.seed(seed)

    stage = omni.usd.get_context().get_stage()

    # Container prim under the cube
    container_path = f"{cube_prim_path}/Protrusions"

    # If Protrusions already exists, remove it (so repeated runs don't accumulate)
    if stage.GetPrimAtPath(container_path).IsValid():
        stage.RemovePrim(container_path)

    UsdGeom.Xform.Define(stage, container_path)

    hx, hy, hz = cube_size[0] * 0.5, cube_size[1] * 0.5, cube_size[2] * 0.5

    # We avoid bottom face (-Z) because the cube sits on a board
    faces = ["+X", "-X", "+Y", "-Y", "+Z"]

    # Store placed bumps for simple non-overlap rejection
    placed: List[Tuple[Gf.Vec3d, float]] = []

    def sample_on_face(face: str, r: float) -> Gf.Vec3d:
        """Sample a hemisphere center on a given cube face.
        Hemisphere is achieved by placing the sphere center exactly on the face plane.
        """
        # margin so hemispheres don't cross edges
        mx = hx - r - gap
        my = hy - r - gap
        mz = hz - r - gap

        if face == "+X":
            x = hx              # <-- hemisphere: center on face plane
            y = random.uniform(-my, my)
            z = random.uniform(-mz, mz)
        elif face == "-X":
            x = -hx
            y = random.uniform(-my, my)
            z = random.uniform(-mz, mz)
        elif face == "+Y":
            x = random.uniform(-mx, mx)
            y = hy
            z = random.uniform(-mz, mz)
        elif face == "-Y":
            x = random.uniform(-mx, mx)
            y = -hy
            z = random.uniform(-mz, mz)
        elif face == "+Z":
            x = random.uniform(-mx, mx)
            y = random.uniform(-my, my)
            z = hz              # <-- hemisphere: center on face plane
        else:
            raise ValueError(face)

        return Gf.Vec3d(x, y, z)

    def is_non_overlapping(pos: Gf.Vec3d, r: float) -> bool:
        for p, pr in placed:
            if (pos - p).GetLength() < (r + pr + gap):
                return False
        return True

    # Create bumps
    max_tries_per_bump = 200
    created = 0

    for i in range(num_bumps):
        r = random.uniform(radius_range[0], radius_range[1])

        pos = None
        for _ in range(max_tries_per_bump):
            face = random.choice(faces)
            cand = sample_on_face(face, r)
            if is_non_overlapping(cand, r):
                pos = cand
                break

        if pos is None:
            # couldn't place without overlap; skip this bump
            continue

        placed.append((pos, r))

        bump_path = f"{container_path}/bump_{i:02d}"
        sphere = UsdGeom.Sphere.Define(stage, bump_path)
        sphere.GetRadiusAttr().Set(r)

        # Set local pose (relative to cube)
        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(pos)

        # Make it a collision shape (no rigid body => compound collider of cube)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(sphere.GetPrim())

        # Optional: make it visible so you can see protrusions in viewport
        if enable_visual:
            sphere.CreateDisplayColorAttr().Set([(0.0, 0.0, 0.0)])
            sphere.CreateDisplayOpacityAttr().Set([1.0])

        created += 1

    print(f"[INFO] Added {created}/{num_bumps} protrusions under {cube_prim_path}.")


def main() -> None:
    """Build a simple scene with a wooden board and a cube on it."""
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60)
    sim = sim_utils.SimulationContext(sim_cfg)

    # camera view
    sim.set_camera_view([1.2, 0.8, 0.6], [0.5, 0.0, 0.05])

    # ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # light
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg, translation=(0.0, 0.0, 3.0))

    # wooden board (thin cuboid, kinematic)
    board_size = (0.6, 0.6, 0.01)
    board_cfg = sim_utils.CuboidCfg(
        size=board_size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001,
            rest_offset=0.0005,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.65, 0.45, 0.30),
            roughness=0.6,
            metallic=0.0,
        ),
    )
    board_z = board_size[2] * 0.5
    board_cfg.func("/World/WoodBoard", board_cfg, translation=(0.5, 0.0, board_z))

    # cube on the board
    cube_size = (0.05, 0.05, 0.05)
    cube_cfg = sim_utils.CuboidCfg(
        size=cube_size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001,
            rest_offset=0.0005,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.2, 0.8),
            roughness=0.5,
            metallic=0.0,
        ),
    )
    cube_z = board_size[2] + cube_size[2] * 0.5 + 0.002
    cube_cfg.func("/World/Cube", cube_cfg, translation=(0.5, 0.0, cube_z))

    # >>> Add random protrusions to cube (compound colliders)
    add_random_protrusions_to_cube(
        cube_prim_path="/World/Cube",
        cube_size=cube_size,
        num_bumps=14,                 # 凸起数量
        radius_range=(0.0025, 0.006), # 凸起半径范围（米）
        gap=0.001,                    # 凸起间最小间距
        seed=None,                    # None=每次随机；固定数=可复现
        enable_visual=True,           # 可视化凸起
    )

    # reset and run
    sim.reset()
    print("[INFO] Scene ready. Press Play in Isaac Sim to run.")

    while simulation_app.is_running():
        sim.render()
        if sim.is_playing():
            sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
