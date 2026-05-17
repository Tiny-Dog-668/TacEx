"""Standalone occlusion-ratio test for the cylinder in the partial drawer scene.

Usage examples:
    isaaclab -p scripts/cylinder_grasping/cylinder_occlusion_annotator_env.py --headless
    isaaclab -p scripts/cylinder_grasping/cylinder_occlusion_annotator_env.py --num_frames 600 --print_every 5
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Test cylinder occlusion ratio using the Replicator bounding_box_3d annotator.")
parser.add_argument("--num_frames", type=int, default=300, help="Number of simulation frames to run.")
parser.add_argument("--print_every", type=int, default=10, help="Print occlusion ratio every N frames.")
parser.add_argument("--warmup_frames", type=int, default=20, help="Warm-up frames before measurement.")
parser.add_argument("--cylinder_dx", type=float, default=0.0, help="Cylinder X offset in meters.")
parser.add_argument("--cylinder_dy", type=float, default=0.0, help="Cylinder Y offset in meters.")
parser.add_argument("--cylinder_dz", type=float, default=0.0, help="Cylinder Z offset in meters.")
parser.add_argument(
    "--move_every_s",
    type=float,
    default=5.0,
    help="Teleport cylinder every N simulated seconds (<=0 disables automatic movement).",
)
parser.add_argument(
    "--move_mode",
    type=str,
    default="cycle",
    choices=["cycle", "random", "none"],
    help="Automatic movement mode for cylinder position updates.",
)
parser.add_argument(
    "--move_margin",
    type=float,
    default=0.003,
    help="Safety margin from XY bounds when auto-moving the cylinder.",
)
parser.add_argument(
    "--debug_dump_rows",
    action="store_true",
    default=False,
    help="Print bbox row snippets for debugging.",
)
parser.add_argument(
    "--debug_max_rows",
    type=int,
    default=16,
    help="Maximum row count shown when --debug_dump_rows is enabled.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import omni.replicator.core as rep
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg


# Keep dimensions/poses aligned with occluded_grasping/vt_box.py
CAN_RADIUS = 0.025
CAN_HEIGHT = 0.07
CAN_CENTROID_OFFSET_Z_DEFAULT = 0.5 * CAN_HEIGHT

DRAWER_SCALE = 1.8
DRAWER_DEPTH_SCALE = 1.0
DRAWER_WIDTH_SCALE = 1.0
DRAWER_HEIGHT_SCALE = 0.8
DRAWER_OPEN_RATIO = 0.6
DRAWER_FRONT_FACE_X = 0.435
DRAWER_CENTER_Y = 0.00

BASE_DRAWER_INNER_X = 0.22
BASE_DRAWER_INNER_Y = 0.26
BASE_DRAWER_WALL_THICKNESS = 0.015
BASE_DRAWER_FLOOR_THICKNESS = 0.02
BASE_DRAWER_WALL_HEIGHT = 0.10
BASE_DRAWER_FRONT_PANEL_HEIGHT = 0.14

BASE_DRAWER_CABINET_WALL_THICKNESS = 0.015
BASE_DRAWER_CABINET_SIDE_CLEARANCE_Y = 0.004
BASE_DRAWER_CABINET_TOP_CLEARANCE_Z = 0.006
BASE_DRAWER_CABINET_REAR_CLEARANCE_X = 0.020
BASE_DRAWER_FRONT_PANEL_SIDE_GAP = 0.004

BASE_DRAWER_HANDLE_DEPTH = 0.018
BASE_DRAWER_HANDLE_WIDTH_Y = 0.07
BASE_DRAWER_HANDLE_HEIGHT_Z = 0.014

DRAWER_X_SCALE = DRAWER_SCALE * DRAWER_DEPTH_SCALE
DRAWER_Y_SCALE = DRAWER_SCALE * DRAWER_WIDTH_SCALE
DRAWER_Z_SCALE = DRAWER_SCALE * DRAWER_HEIGHT_SCALE

BOX_INNER_X = BASE_DRAWER_INNER_X * DRAWER_X_SCALE
BOX_INNER_Y = BASE_DRAWER_INNER_Y * DRAWER_Y_SCALE
BOX_WALL_THICKNESS = BASE_DRAWER_WALL_THICKNESS * DRAWER_SCALE
BOX_FLOOR_THICKNESS = BASE_DRAWER_FLOOR_THICKNESS * DRAWER_Z_SCALE
BOX_WALL_HEIGHT = BASE_DRAWER_WALL_HEIGHT * DRAWER_Z_SCALE
BOX_FRONT_PANEL_HEIGHT = BASE_DRAWER_FRONT_PANEL_HEIGHT * DRAWER_Z_SCALE

DRAWER_OUTER_DEPTH_X = BOX_INNER_X + 2 * BOX_WALL_THICKNESS
DRAWER_OUTER_WIDTH_Y = BOX_INNER_Y + 2 * BOX_WALL_THICKNESS
DRAWER_BODY_HEIGHT_Z = BOX_FLOOR_THICKNESS + BOX_WALL_HEIGHT

BOX_CENTER_X = DRAWER_FRONT_FACE_X + DRAWER_OUTER_DEPTH_X * 0.5
BOX_CENTER_Y = DRAWER_CENTER_Y
BOX_WALL_CENTER_Z = BOX_FLOOR_THICKNESS + BOX_WALL_HEIGHT * 0.5
BOX_FRONT_PANEL_CENTER_Z = BOX_FRONT_PANEL_HEIGHT * 0.5
BOX_OBJECT_Z = BOX_FLOOR_THICKNESS + CAN_CENTROID_OFFSET_Z_DEFAULT

# Match object spawn region used in vt_box.py
CAN_RANDOM_X_MIN = 0.5158
CAN_RANDOM_X_MAX = 0.6515
CAN_RANDOM_Y_MIN = -0.1838
CAN_RANDOM_Y_MAX = 0.1772
CAN_RANDOM_CENTER_X = 0.5 * (CAN_RANDOM_X_MIN + CAN_RANDOM_X_MAX)
CAN_RANDOM_CENTER_Y = 0.5 * (CAN_RANDOM_Y_MIN + CAN_RANDOM_Y_MAX)
CAN_RESET_ROOT_Z = BOX_FLOOR_THICKNESS + 0.5 * CAN_HEIGHT + 0.005

DRAWER_CABINET_WALL_THICKNESS = BASE_DRAWER_CABINET_WALL_THICKNESS * DRAWER_SCALE
DRAWER_CABINET_SIDE_CLEARANCE_Y = BASE_DRAWER_CABINET_SIDE_CLEARANCE_Y * DRAWER_Y_SCALE
DRAWER_CABINET_TOP_CLEARANCE_Z = BASE_DRAWER_CABINET_TOP_CLEARANCE_Z * DRAWER_Z_SCALE
DRAWER_CABINET_REAR_CLEARANCE_X = BASE_DRAWER_CABINET_REAR_CLEARANCE_X * DRAWER_X_SCALE
DRAWER_FRONT_PANEL_SIDE_GAP = BASE_DRAWER_FRONT_PANEL_SIDE_GAP * DRAWER_Y_SCALE
DRAWER_FRONT_PROTRUSION_X = DRAWER_OUTER_DEPTH_X * DRAWER_OPEN_RATIO

DRAWER_CABINET_CENTER_Y = BOX_CENTER_Y
DRAWER_CABINET_FRONT_X = BOX_CENTER_X - DRAWER_OUTER_DEPTH_X * 0.5 + DRAWER_FRONT_PROTRUSION_X
DRAWER_CABINET_OUTER_DEPTH_X = DRAWER_OUTER_DEPTH_X - DRAWER_FRONT_PROTRUSION_X + DRAWER_CABINET_REAR_CLEARANCE_X
DRAWER_CABINET_CENTER_X = DRAWER_CABINET_FRONT_X + DRAWER_CABINET_OUTER_DEPTH_X * 0.5
DRAWER_CABINET_OUTER_WIDTH_Y = (
    DRAWER_OUTER_WIDTH_Y + 2 * (DRAWER_CABINET_SIDE_CLEARANCE_Y + DRAWER_CABINET_WALL_THICKNESS)
)
DRAWER_CABINET_HEIGHT = DRAWER_BODY_HEIGHT_Z + DRAWER_CABINET_TOP_CLEARANCE_Z + DRAWER_CABINET_WALL_THICKNESS
DRAWER_FRONT_PANEL_WIDTH_Y = DRAWER_CABINET_OUTER_WIDTH_Y - 2 * DRAWER_FRONT_PANEL_SIDE_GAP

DRAWER_HANDLE_DEPTH = BASE_DRAWER_HANDLE_DEPTH * DRAWER_SCALE
DRAWER_HANDLE_WIDTH_Y = BASE_DRAWER_HANDLE_WIDTH_Y * DRAWER_Y_SCALE
DRAWER_HANDLE_HEIGHT_Z = BASE_DRAWER_HANDLE_HEIGHT_Z * DRAWER_Z_SCALE

THIRD_PERSON_CAMERA_POS = (-0.2, -0.35, 1.2)
THIRD_PERSON_CAMERA_ROT = (0.68301, 0.18301, -0.18301, -0.68301)


def make_fixed_panel_cfg(
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    semantic_label: str = "drawer",
) -> sim_utils.CuboidCfg:
    """Create a kinematic panel used for drawer/cabinet pieces."""
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
            max_angular_velocity=100.0,
            max_linear_velocity=10.0,
            max_depenetration_velocity=1.0,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001,
            rest_offset=0.0001,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=0.65,
            metallic=0.0,
        ),
        semantic_tags=[("class", semantic_label)],
    )


def spawn_fixed_panel(
    prim_path: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
    semantic_label: str = "drawer",
) -> None:
    panel_cfg = make_fixed_panel_cfg(size=size, color=color, semantic_label=semantic_label)
    panel_cfg.func(prim_path, panel_cfg, translation=translation)


def spawn_ground_and_light() -> None:
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    light_cfg.func("/World/light", light_cfg)


def spawn_partial_drawer() -> None:
    # Drawer body
    spawn_fixed_panel(
        "/World/Drawer/box_floor",
        size=(BOX_INNER_X + 2 * BOX_WALL_THICKNESS, BOX_INNER_Y + 2 * BOX_WALL_THICKNESS, BOX_FLOOR_THICKNESS),
        translation=(BOX_CENTER_X, BOX_CENTER_Y, BOX_FLOOR_THICKNESS * 0.5),
        color=(0.40, 0.27, 0.18),
    )
    spawn_fixed_panel(
        "/World/Drawer/box_wall_front",
        size=(BOX_WALL_THICKNESS, DRAWER_FRONT_PANEL_WIDTH_Y, BOX_FRONT_PANEL_HEIGHT),
        translation=(BOX_CENTER_X - BOX_INNER_X * 0.5 - BOX_WALL_THICKNESS * 0.5, BOX_CENTER_Y, BOX_FRONT_PANEL_CENTER_Z),
        color=(0.56, 0.39, 0.26),
    )
    spawn_fixed_panel(
        "/World/Drawer/box_wall_back",
        size=(BOX_WALL_THICKNESS, BOX_INNER_Y, BOX_WALL_HEIGHT),
        translation=(BOX_CENTER_X + BOX_INNER_X * 0.5 + BOX_WALL_THICKNESS * 0.5, BOX_CENTER_Y, BOX_WALL_CENTER_Z),
        color=(0.50, 0.35, 0.24),
    )
    spawn_fixed_panel(
        "/World/Drawer/box_wall_left",
        size=(BOX_INNER_X, BOX_WALL_THICKNESS, BOX_WALL_HEIGHT),
        translation=(BOX_CENTER_X, BOX_CENTER_Y - BOX_INNER_Y * 0.5 - BOX_WALL_THICKNESS * 0.5, BOX_WALL_CENTER_Z),
        color=(0.50, 0.35, 0.24),
    )
    spawn_fixed_panel(
        "/World/Drawer/box_wall_right",
        size=(BOX_INNER_X, BOX_WALL_THICKNESS, BOX_WALL_HEIGHT),
        translation=(BOX_CENTER_X, BOX_CENTER_Y + BOX_INNER_Y * 0.5 + BOX_WALL_THICKNESS * 0.5, BOX_WALL_CENTER_Z),
        color=(0.50, 0.35, 0.24),
    )

    # Cabinet shell around drawer
    spawn_fixed_panel(
        "/World/Drawer/drawer_cabinet_left_side",
        size=(DRAWER_CABINET_OUTER_DEPTH_X, DRAWER_CABINET_WALL_THICKNESS, DRAWER_CABINET_HEIGHT),
        translation=(
            DRAWER_CABINET_CENTER_X,
            DRAWER_CABINET_CENTER_Y - DRAWER_CABINET_OUTER_WIDTH_Y * 0.5 + DRAWER_CABINET_WALL_THICKNESS * 0.5,
            DRAWER_CABINET_HEIGHT * 0.5,
        ),
        color=(0.44, 0.31, 0.22),
    )
    spawn_fixed_panel(
        "/World/Drawer/drawer_cabinet_right_side",
        size=(DRAWER_CABINET_OUTER_DEPTH_X, DRAWER_CABINET_WALL_THICKNESS, DRAWER_CABINET_HEIGHT),
        translation=(
            DRAWER_CABINET_CENTER_X,
            DRAWER_CABINET_CENTER_Y + DRAWER_CABINET_OUTER_WIDTH_Y * 0.5 - DRAWER_CABINET_WALL_THICKNESS * 0.5,
            DRAWER_CABINET_HEIGHT * 0.5,
        ),
        color=(0.44, 0.31, 0.22),
    )
    spawn_fixed_panel(
        "/World/Drawer/drawer_cabinet_back",
        size=(
            DRAWER_CABINET_WALL_THICKNESS,
            DRAWER_CABINET_OUTER_WIDTH_Y - 2 * DRAWER_CABINET_WALL_THICKNESS,
            DRAWER_CABINET_HEIGHT,
        ),
        translation=(
            DRAWER_CABINET_CENTER_X + DRAWER_CABINET_OUTER_DEPTH_X * 0.5 - DRAWER_CABINET_WALL_THICKNESS * 0.5,
            DRAWER_CABINET_CENTER_Y,
            DRAWER_CABINET_HEIGHT * 0.5,
        ),
        color=(0.44, 0.31, 0.22),
    )
    spawn_fixed_panel(
        "/World/Drawer/drawer_cabinet_top",
        size=(DRAWER_CABINET_OUTER_DEPTH_X, DRAWER_CABINET_OUTER_WIDTH_Y, DRAWER_CABINET_WALL_THICKNESS),
        translation=(
            DRAWER_CABINET_CENTER_X,
            DRAWER_CABINET_CENTER_Y,
            DRAWER_CABINET_HEIGHT - DRAWER_CABINET_WALL_THICKNESS * 0.5,
        ),
        color=(0.46, 0.32, 0.22),
    )
    spawn_fixed_panel(
        "/World/Drawer/drawer_handle",
        size=(DRAWER_HANDLE_DEPTH, DRAWER_HANDLE_WIDTH_Y, DRAWER_HANDLE_HEIGHT_Z),
        translation=(
            BOX_CENTER_X - BOX_INNER_X * 0.5 - BOX_WALL_THICKNESS - DRAWER_HANDLE_DEPTH * 0.5,
            BOX_CENTER_Y,
            BOX_FRONT_PANEL_CENTER_Z,
        ),
        color=(0.72, 0.72, 0.74),
        semantic_label="handle",
    )


def spawn_cylinder() -> RigidObject:
    cylinder_cfg = RigidObjectCfg(
        prim_path="/World/cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                CAN_RANDOM_CENTER_X + args_cli.cylinder_dx,
                CAN_RANDOM_CENTER_Y + args_cli.cylinder_dy,
                CAN_RESET_ROOT_Z + args_cli.cylinder_dz,
            )
        ),
        spawn=sim_utils.CylinderCfg(
            radius=CAN_RADIUS,
            height=CAN_HEIGHT,
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
                rest_offset=0.0001,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.84, 0.84, 0.86),
                roughness=0.45,
                metallic=0.0,
            ),
            semantic_tags=[("class", "cylinder")],
        ),
    )
    return RigidObject(cylinder_cfg)


def spawn_third_person_camera() -> Camera:
    camera_cfg = CameraCfg(
        prim_path="/World/third_person_camera",
        update_period=0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=30.0,
            focus_distance=200.0,
            horizontal_aperture=40.0,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=THIRD_PERSON_CAMERA_POS,
            rot=THIRD_PERSON_CAMERA_ROT,
            convention="opengl",
        ),
    )
    return Camera(camera_cfg)


def step_scene(sim: sim_utils.SimulationContext, cylinder: RigidObject, camera: Camera, num_steps: int = 1) -> None:
    for _ in range(num_steps):
        cylinder.write_data_to_sim()
        sim.step()
        cylinder.update(sim.cfg.dt)
        camera.update(sim.cfg.dt)


def unpack_annotator_output(output):
    if isinstance(output, dict):
        data = output.get("data")
        info = output.get("info", {})
        if not info:
            info = {k: v for k, v in output.items() if k != "data"}
        return data, info
    return output, {}


def _to_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def find_bbox_rows_for_prim(prim_path: str, bbox_output) -> list[dict]:
    bbox_data, bbox_info = unpack_annotator_output(bbox_output)
    prim_paths = bbox_info.get("primPaths", [])
    if bbox_data is None or not isinstance(prim_paths, Iterable):
        return []

    rows: list[dict] = []
    for idx, path in enumerate(prim_paths):
        if path == prim_path or str(path).startswith(f"{prim_path}/"):
            semantic_id = _to_int(bbox_data[idx]["semanticId"])
            occlusion_ratio = float(bbox_data[idx]["occlusionRatio"])
            rows.append(
                {
                    "idx": idx,
                    "primPath": str(path),
                    "semanticId": semantic_id,
                    "occlusionRatio": occlusion_ratio,
                }
            )
    return rows


def choose_bbox_ratio(bbox_rows: list[dict]) -> float | None:
    if not bbox_rows:
        return None
    valid: list[float] = []
    for row in bbox_rows:
        ratio = float(row["occlusionRatio"])
        if math.isfinite(ratio) and ratio >= 0.0:
            valid.append(ratio)
    if not valid:
        return None
    return float(max(valid))


def format_bbox_rows(bbox_rows: list[dict], max_rows: int) -> str:
    if not bbox_rows:
        return "none"
    lines: list[str] = []
    row_count = len(bbox_rows)
    show_count = min(max_rows, row_count)
    for i in range(show_count):
        row = bbox_rows[i]
        lines.append(
            f"#{i}: idx={int(row['idx'])}, prim={row['primPath']}, sem={int(row['semanticId'])}, occ={float(row['occlusionRatio']):.4f}"
        )
    if row_count > show_count:
        lines.append(f"... ({row_count - show_count} more)")
    return "; ".join(lines)


def _safe_bounds(min_v: float, max_v: float, margin: float) -> tuple[float, float]:
    lo = min_v + margin
    hi = max_v - margin
    if lo <= hi:
        return lo, hi
    center = 0.5 * (min_v + max_v)
    return center, center


def build_cycle_positions() -> list[tuple[float, float, float]]:
    x_min, x_max = _safe_bounds(
        CAN_RANDOM_X_MIN + args_cli.cylinder_dx, CAN_RANDOM_X_MAX + args_cli.cylinder_dx, args_cli.move_margin
    )
    y_min, y_max = _safe_bounds(
        CAN_RANDOM_Y_MIN + args_cli.cylinder_dy, CAN_RANDOM_Y_MAX + args_cli.cylinder_dy, args_cli.move_margin
    )
    x_c = 0.5 * (x_min + x_max)
    y_c = 0.5 * (y_min + y_max)
    z = CAN_RESET_ROOT_Z + args_cli.cylinder_dz

    candidates = [
        (x_c, y_c, z),
        (x_min, y_c, z),
        (x_max, y_c, z),
        (x_c, y_min, z),
        (x_c, y_max, z),
        (x_min, y_min, z),
        (x_min, y_max, z),
        (x_max, y_min, z),
        (x_max, y_max, z),
    ]
    unique: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for pos in candidates:
        key = (round(pos[0], 6), round(pos[1], 6), round(pos[2], 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append(pos)
    return unique


def sample_random_position() -> tuple[float, float, float]:
    x_min, x_max = _safe_bounds(
        CAN_RANDOM_X_MIN + args_cli.cylinder_dx, CAN_RANDOM_X_MAX + args_cli.cylinder_dx, args_cli.move_margin
    )
    y_min, y_max = _safe_bounds(
        CAN_RANDOM_Y_MIN + args_cli.cylinder_dy, CAN_RANDOM_Y_MAX + args_cli.cylinder_dy, args_cli.move_margin
    )
    z = CAN_RESET_ROOT_Z + args_cli.cylinder_dz

    if x_max > x_min:
        x = float(x_min + (x_max - x_min) * torch.rand(1).item())
    else:
        x = float(x_min)
    if y_max > y_min:
        y = float(y_min + (y_max - y_min) * torch.rand(1).item())
    else:
        y = float(y_min)
    return (x, y, z)


def teleport_cylinder(cylinder: RigidObject, position: tuple[float, float, float]) -> None:
    root_state = cylinder.data.root_state_w.clone()
    root_state[:, 0] = position[0]
    root_state[:, 1] = position[1]
    root_state[:, 2] = position[2]
    root_state[:, 3] = 1.0
    root_state[:, 4:7] = 0.0
    root_state[:, 7:13] = 0.0
    env_ids = torch.arange(root_state.shape[0], device=root_state.device, dtype=torch.long)
    cylinder.write_root_state_to_sim(root_state, env_ids=env_ids)
    cylinder.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=root_state.device), env_ids=env_ids)


def main() -> None:
    sim_cfg = SimulationCfg(
        dt=1 / 60,
        physx=PhysxCfg(enable_ccd=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.45, 0.85, 0.52], [BOX_CENTER_X, BOX_CENTER_Y, 0.08])

    spawn_ground_and_light()
    spawn_partial_drawer()
    cylinder = spawn_cylinder()
    camera = spawn_third_person_camera()

    sim.reset()
    step_scene(sim, cylinder, camera, num_steps=max(args_cli.warmup_frames, 1))

    render_product = camera.render_product_paths[0]
    print(f"[INFO] Using render product: {render_product}")

    available_annotators = set(rep.AnnotatorRegistry.get_registered_annotators())
    if "bounding_box_3d" not in available_annotators:
        raise RuntimeError("bounding_box_3d annotator is unavailable in this Isaac Sim runtime.")

    bbox_annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_3d", init_params={"semanticTypes": ["class"]})

    bbox_annotator.attach(render_product)

    cylinder_prim_path = "/World/cylinder"
    move_mode = args_cli.move_mode
    if args_cli.move_every_s <= 0:
        move_mode = "none"
    if move_mode == "none":
        move_every_frames = 0
    else:
        move_every_frames = max(1, int(round(args_cli.move_every_s / sim_cfg.dt)))
    cycle_positions = build_cycle_positions()
    cycle_index = 0
    print(
        f"[INFO] movement mode={move_mode}, move_every_s={args_cli.move_every_s:.2f}, "
        f"move_every_frames={move_every_frames}"
    )

    try:
        for frame_idx in range(max(args_cli.num_frames, 1)):
            if not simulation_app.is_running():
                break

            if move_mode != "none" and move_every_frames > 0 and frame_idx > 0 and frame_idx % move_every_frames == 0:
                if move_mode == "cycle":
                    cycle_index = (cycle_index + 1) % max(len(cycle_positions), 1)
                    target_pos = cycle_positions[cycle_index]
                else:
                    target_pos = sample_random_position()
                teleport_cylinder(cylinder, target_pos)
                print(
                    f"[MOVE][FRAME {frame_idx:04d}] cylinder -> "
                    f"x={target_pos[0]:.4f}, y={target_pos[1]:.4f}, z={target_pos[2]:.4f}"
                )

            step_scene(sim, cylinder, camera, num_steps=1)

            if frame_idx % max(args_cli.print_every, 1) != 0:
                continue

            bbox_output = bbox_annotator.get_data()
            bbox_rows = find_bbox_rows_for_prim(cylinder_prim_path, bbox_output)
            bbox_ratio = choose_bbox_ratio(bbox_rows)

            if bbox_ratio is None:
                reason = "bbox row missing or invalid"
                if not bbox_rows:
                    reason = "cylinder not found in bbox_3d (likely out of camera view)"
                print(
                    f"[FRAME {frame_idx:04d}] cylinder occlusion unavailable "
                    f"(source=bbox_3d, reason={reason})"
                )
                if args_cli.debug_dump_rows:
                    print(
                        f"[DEBUG][FRAME {frame_idx:04d}] bbox rows for cylinder: "
                        f"{format_bbox_rows(bbox_rows, args_cli.debug_max_rows)}"
                    )
            else:
                visible_ratio = 1.0 - bbox_ratio
                print(
                    f"[FRAME {frame_idx:04d}] cylinder occlusionRatio={bbox_ratio:.4f}, "
                    f"visible_ratio={visible_ratio:.4f}, source=bbox_3d"
                )
                if args_cli.debug_dump_rows:
                    print(
                        f"[DEBUG][FRAME {frame_idx:04d}] bbox rows for cylinder: "
                        f"{format_bbox_rows(bbox_rows, args_cli.debug_max_rows)}"
                    )
    finally:
        bbox_annotator.detach([render_product])


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
