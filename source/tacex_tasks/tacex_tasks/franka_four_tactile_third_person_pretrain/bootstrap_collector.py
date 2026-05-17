from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from isaaclab.app import AppLauncher

from dataset_config import (
    SUPPORTED_PRIMITIVES,
    ObjectSpec,
    load_collection_protocol,
    load_supported_object_specs,
)
from video_export import export_episode_videos


parser = argparse.ArgumentParser(
    description="Bootstrap data collector for Franka + four tactile sensors + one third-person camera."
)
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect.")
parser.add_argument("--output_dir", type=str, default="outputs/pretrain_bootstrap", help="Output directory.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--object_id", type=str, default=None, help="Fix the collector to a single supported object id.")
parser.add_argument(
    "--primitive",
    type=str,
    default="random",
    help="Interaction primitive: random or one of touch/press/slide/pinch_grasp.",
)
parser.add_argument(
    "--splits",
    nargs="+",
    default=["train", "val", "test_id"],
    help="Object splits to sample from.",
)
parser.add_argument("--settle_steps", type=int, default=12, help="Control steps for scene settling before recording.")
parser.add_argument("--debug_vis", action="store_true", default=False, help="Enable tactile debug visualization in GUI.")
parser.add_argument(
    "--contact_duration_scale",
    type=float,
    default=1.0,
    help="Scale factor applied to contact-rich primitive stages to collect longer tactile sequences.",
)
parser.add_argument(
    "--contact_plan_mode",
    type=str,
    default="diverse",
    choices=("diverse", "centered"),
    help="Contact planning mode for touch/press. Use 'centered' for a deterministic center-contact baseline.",
)
parser.add_argument(
    "--center_primary_sensor",
    type=str,
    default="left_down",
    choices=("left_down", "right_down", "random"),
    help="Primary down-facing tactile sensor used in centered contact mode.",
)
parser.add_argument(
    "--force_contact_mode",
    type=str,
    default="auto",
    help=(
        "Force touch/press contact mode instead of random sampling. "
        "Examples: auto, face_center, x_edge, x_edge_pos, x_edge_neg, "
        "y_edge, y_edge_pos, y_edge_neg, corner, corner_pp, corner_pn, corner_np, corner_nn, "
        "top_center, top_band, top_rim, center, off_axis_soft, off_axis_mid."
    ),
)
parser.add_argument(
    "--press_depth_mm",
    type=float,
    default=None,
    help="Override press depth in millimeters for press primitives.",
)
parser.add_argument(
    "--export_mp4",
    action="store_true",
    default=False,
    help="Export third-person and tactile grid mp4 videos after each episode is saved.",
)
parser.add_argument("--mp4_fps", type=int, default=20, help="FPS for exported mp4 videos.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from tacex import GelSightSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg


SCRIPT_DIR = Path(__file__).resolve().parent
OBJECT_SPECS = load_supported_object_specs(splits=tuple(args_cli.splits))
COLLECTION_PROTOCOL = load_collection_protocol()

if not OBJECT_SPECS:
    raise RuntimeError(f"No supported object specs found for splits={args_cli.splits}.")
if args_cli.object_id is not None and args_cli.object_id not in OBJECT_SPECS:
    raise ValueError(
        f"Unsupported or unavailable object_id={args_cli.object_id!r}. "
        f"Supported ids: {sorted(OBJECT_SPECS.keys())}"
    )
if args_cli.primitive != "random" and args_cli.primitive not in SUPPORTED_PRIMITIVES:
    raise ValueError(f"Unsupported primitive={args_cli.primitive!r}. Choices: random, {', '.join(SUPPORTED_PRIMITIVES)}")


BOARD_SIZE = (0.60, 0.60, 0.04)
TABLE_TOP_Z = BOARD_SIZE[2]
OPEN_GRIPPER_WIDTH = 0.08
CLOSED_GRIPPER_WIDTH = 0.0

# Estimated from the down-facing gelpad attachment points in
# `physx_rigid_gelpads.down.usda`, using the gelpad face center rather than the
# internal camera as the centered-contact reference.
DOWN_SENSOR_CONTACT_FACE_CENTER_OFFSETS = {
    "gsmini_right_down": (-0.000254067, -0.000812414, 0.028459837),
    "gsmini_left_down": (0.001538189, 0.000733595, 0.028755258),
}

CONTACT_RICH_STAGES = {
    "touch": {"initial_contact", "hold"},
    "press": {"press_ramp", "peak_hold"},
    "slide": {"contact_establish", "tangential_slide"},
    "pinch_grasp": {"close_until_contact", "squeeze_hold", "micro_reposition"},
}


def _rigid_props(kinematic: bool, disable_gravity: bool) -> RigidBodyPropertiesCfg:
    return RigidBodyPropertiesCfg(
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=4,
        max_angular_velocity=100.0,
        max_linear_velocity=10.0,
        max_depenetration_velocity=1.0,
        kinematic_enabled=kinematic,
        disable_gravity=disable_gravity,
    )


def _collision_props() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0005)


def _preview_material(color: tuple[float, float, float], roughness: float = 0.55) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.0, roughness=roughness)


def _cuboid_cfg(size: tuple[float, float, float], color: tuple[float, float, float], kinematic: bool = False) -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=size,
        rigid_props=_rigid_props(kinematic=kinematic, disable_gravity=kinematic),
        collision_props=_collision_props(),
        visual_material=_preview_material(color),
    )


def _sphere_cfg(radius: float, color: tuple[float, float, float]) -> sim_utils.SphereCfg:
    return sim_utils.SphereCfg(
        radius=radius,
        rigid_props=_rigid_props(kinematic=False, disable_gravity=False),
        collision_props=_collision_props(),
        visual_material=_preview_material(color),
    )


def _cylinder_cfg(radius: float, height: float, color: tuple[float, float, float]) -> sim_utils.CylinderCfg:
    return sim_utils.CylinderCfg(
        radius=radius,
        height=height,
        rigid_props=_rigid_props(kinematic=False, disable_gravity=False),
        collision_props=_collision_props(),
        visual_material=_preview_material(color),
    )


@configclass
class BootstrapCollectorEnvCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.5, 1.5, 1.2)
    viewer.lookat = (0.5, 0.0, 0.08)

    debug_vis = False
    decimation = 1
    episode_length_s = 10.0
    action_space = 5
    observation_space = 1
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            solver_type=1,
            max_position_iteration_count=64,
            max_velocity_iteration_count=4,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=2.0,
        replicate_physics=True,
        lazy_sensor_update=True,
    )

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.80, 0.80, 0.80), intensity=3500.0),
    )

    board = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table_board",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.0, BOARD_SIZE[2] * 0.5)),
        spawn=_cuboid_cfg(BOARD_SIZE, (0.55, 0.38, 0.22), kinematic=True),
    )

    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.down.usda"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": -0.6199,
                "panda_joint2": -0.2263,
                "panda_joint3": 0.1508,
                "panda_joint4": -2.8275,
                "panda_joint5": 0.0273,
                "panda_joint6": 2.6023,
                "panda_joint7": 0.3497,
                "panda_finger_joint.*": 0.02,
            },
        ),
    )

    third_person_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
        update_period=0,
        height=512,
        width=512,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=30.0,
            focus_distance=400.0,
            horizontal_aperture=24.0,
            clipping_range=(0.05, 30.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.88, -0.06, 0.20),
            rot=(0.609721, 0.454117, 0.388040, 0.521003),
            convention="opengl",
        ),
    )

    tactile_img_res_hw = (96, 128)

    gsmini_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(64, 48),
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=args_cli.debug_vis,
        marker_motion_sim_cfg=None,
        data_types=["camera_depth", "tactile_rgb"],
    )
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0]),
    )
    gsmini_right = gsmini_left.replace(prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right")
    gsmini_left_down = gsmini_left.replace(prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left_down")
    gsmini_right_down = gsmini_left.replace(prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right_down")

    sphere_xs_25 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sphere_xs_25",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_sphere_cfg(radius=0.0125, color=(0.85, 0.45, 0.35)),
    )
    sphere_s_35 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sphere_s_35",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_sphere_cfg(radius=0.0175, color=(0.80, 0.30, 0.25)),
    )
    sphere_m_50 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sphere_m_50",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_sphere_cfg(radius=0.0250, color=(0.25, 0.60, 0.85)),
    )
    sphere_l_65 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/sphere_l_65",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_sphere_cfg(radius=0.0325, color=(0.30, 0.70, 0.60)),
    )
    box_cube_25 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_cube_25",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cuboid_cfg((0.025, 0.025, 0.025), (0.40, 0.45, 0.85)),
    )
    box_cube_40 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_cube_40",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cuboid_cfg((0.04, 0.04, 0.04), (0.25, 0.30, 0.85)),
    )
    box_cube_55 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_cube_55",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cuboid_cfg((0.055, 0.055, 0.055), (0.35, 0.55, 0.90)),
    )
    box_rect_60_40_30 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_rect_60_40_30",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cuboid_cfg((0.06, 0.04, 0.03), (0.55, 0.25, 0.80)),
    )
    box_rect_80_30_25 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_rect_80_30_25",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cuboid_cfg((0.08, 0.03, 0.025), (0.70, 0.35, 0.78)),
    )
    box_flat_50_50_20 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_flat_50_50_20",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cuboid_cfg((0.05, 0.05, 0.02), (0.88, 0.55, 0.22)),
    )
    box_tall_20_20_60 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_tall_20_20_60",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cuboid_cfg((0.02, 0.02, 0.06), (0.42, 0.82, 0.42)),
    )
    box_ridge_80_12_30 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_ridge_80_12_30",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cuboid_cfg((0.08, 0.012, 0.03), (0.75, 0.35, 0.20)),
    )
    box_slim_16_16_35 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_slim_16_16_35",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cuboid_cfg((0.016, 0.016, 0.035), (0.55, 0.78, 0.30)),
    )
    cylinder_slim_18_60 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder_slim_18_60",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cylinder_cfg(radius=0.018, height=0.06, color=(0.92, 0.62, 0.28)),
    )
    cylinder_short_30_60 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder_short_30_60",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cylinder_cfg(radius=0.03, height=0.06, color=(0.85, 0.55, 0.20)),
    )
    cylinder_mid_25_70 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder_mid_25_70",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cylinder_cfg(radius=0.025, height=0.07, color=(0.78, 0.64, 0.24)),
    )
    cylinder_tall_25_100 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder_tall_25_100",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cylinder_cfg(radius=0.025, height=0.10, color=(0.20, 0.70, 0.35)),
    )
    cylinder_thick_40_50 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder_thick_40_50",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cylinder_cfg(radius=0.04, height=0.05, color=(0.78, 0.46, 0.18)),
    )
    plate_small_60_60_8 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/plate_small_60_60_8",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cuboid_cfg((0.06, 0.06, 0.008), (0.85, 0.80, 0.35)),
    )
    plate_80_80_10 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/plate_80_80_10",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cuboid_cfg((0.08, 0.08, 0.01), (0.80, 0.75, 0.30)),
    )
    plate_rect_100_60_8 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/plate_rect_100_60_8",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -0.5)),
        spawn=_cuboid_cfg((0.10, 0.06, 0.008), (0.72, 0.68, 0.26)),
    )

    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")


@dataclass
class EpisodeSelection:
    object_spec: ObjectSpec
    primitive: str


@dataclass
class ContactPlan:
    primitive: str
    primary_sensor: str
    contact_key: str
    contact_mode: str
    object_local_offset_xy_m: tuple[float, float]
    object_world_offset_xy_m: tuple[float, float]
    target_world_xy_m: tuple[float, float]
    gripper_width_m: float
    sensor_hover_clearance_m: float
    search_step_m: float
    search_contact_threshold: float
    touch_hold_shift_xy_m: tuple[float, float] = (0.0, 0.0)
    press_extra_depth_m: float = 0.0
    press_lateral_shift_xy_m: tuple[float, float] = (0.0, 0.0)


class EpisodeRecorder:
    def __init__(self, output_dir: Path, episode_index: int, selection: EpisodeSelection, seed: int):
        self.output_dir = output_dir
        self.episode_index = episode_index
        self.selection = selection
        self.seed = seed
        self.frames: dict[str, list[np.ndarray]] = {}
        self.frame_meta: list[dict[str, Any]] = []
        self.contact_plan: ContactPlan | None = None

    def set_contact_plan(self, contact_plan: ContactPlan):
        self.contact_plan = contact_plan

    def add_frame(self, env: "BootstrapCollectorEnv", stage_name: str, control_step: int):
        frame = env.capture_frame()
        for key, value in frame.items():
            self.frames.setdefault(key, []).append(value)
        self.frame_meta.append(
            {
                "stage": stage_name,
                "control_step": control_step,
                "sim_time_s": float(control_step) / 20.0,
            }
        )

    def finalize(self) -> dict[str, Any]:
        episode_name = f"episode_{self.episode_index:06d}_{self.selection.object_spec.id}_{self.selection.primitive}"
        episode_npz_path = self.output_dir / "episodes" / f"{episode_name}.npz"
        episode_json_path = self.output_dir / "episodes" / f"{episode_name}.json"
        episode_npz_path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {key: np.stack(value, axis=0) for key, value in self.frames.items()}
        contact_keys = [
            "contact_inner_left",
            "contact_inner_right",
            "contact_down_left",
            "contact_down_right",
        ]
        max_contact = {key: float(arrays[key].max()) for key in contact_keys if key in arrays}
        quality = {
            "num_frames": int(len(self.frame_meta)),
            "any_contact_frames": int(
                np.sum(
                    np.logical_or.reduce(
                        [arrays[key].reshape(-1) > 0.0 for key in contact_keys if key in arrays]
                    )
                )
            ),
            "max_contact_ratio": max_contact,
            "final_object_height_m": float(arrays["object_pose"][-1, 2]),
        }

        np.savez_compressed(episode_npz_path, **arrays)
        meta = {
            "episode_index": self.episode_index,
            "episode_name": episode_name,
            "seed": self.seed,
            "selection": {
                "object_id": self.selection.object_spec.id,
                "family": self.selection.object_spec.family,
                "shape": self.selection.object_spec.shape,
                "primitive": self.selection.primitive,
                "split": self.selection.object_spec.split,
            },
            "contact_plan": asdict(self.contact_plan) if self.contact_plan is not None else None,
            "frame_meta": self.frame_meta,
            "quality": quality,
            "npz_path": str(episode_npz_path),
        }
        with episode_json_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta

class BootstrapCollectorEnv(DirectRLEnv):
    cfg: BootstrapCollectorEnvCfg

    def __init__(self, cfg: BootstrapCollectorEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._finger_joint_ids, _ = self._robot.find_joints(["panda_finger.*"])
        self._jacobi_body_idx = self._body_idx - 1
        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self._ee_down_roll = float(np.pi)
        self._ee_down_pitch = 0.0
        self._max_rot_step = 0.25
        self._rot_gain = 1.0
        self.action_scale = 0.04
        self.control_decimation = 3
        self.current_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.processed_actions = torch.zeros((self.num_envs, 6), device=self.device)
        self._active_object_id: str | None = None
        self._active_object_spec: ObjectSpec | None = None
        self._control_step = 0

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._board = RigidObject(self.cfg.board)
        self.scene.rigid_objects["board"] = self._board

        self.third_person_camera = Camera(self.cfg.third_person_camera)
        self.scene.sensors["third_person_camera"] = self.third_person_camera

        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right
        self.gsmini_left_down = GelSightSensor(self.cfg.gsmini_left_down)
        self.scene.sensors["gsmini_left_down"] = self.gsmini_left_down
        self.gsmini_right_down = GelSightSensor(self.cfg.gsmini_right_down)
        self.scene.sensors["gsmini_right_down"] = self.gsmini_right_down

        self._objects: dict[str, RigidObject] = {}
        for object_id in OBJECT_SPECS:
            obj = RigidObject(getattr(self.cfg, object_id))
            self.scene.rigid_objects[object_id] = obj
            self._objects[object_id] = obj

        self.scene.clone_environments(copy_from_source=False)

        self.cfg.ground.spawn.func(
            self.cfg.ground.prim_path,
            self.cfg.ground.spawn,
            translation=self.cfg.ground.init_state.pos,
            orientation=self.cfg.ground.init_state.rot,
        )
        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros((self.num_envs,), device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        done = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        return done, done

    def _reset_idx(self, env_ids: torch.Tensor):
        self._robot.reset(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        for asset in self._objects.values():
            pose = torch.tensor([[0.25, -0.30, -0.50, 1.0, 0.0, 0.0, 0.0]], device=self.device)
            asset.write_root_pose_to_sim(pose, env_ids)
            asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids)
        self.third_person_camera.reset(env_ids)
        self.gsmini_left.reset(env_ids)
        self.gsmini_right.reset(env_ids)
        self.gsmini_left_down.reset(env_ids)
        self.gsmini_right_down.reset(env_ids)
        self._control_step = 0

    @staticmethod
    def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=-1)

    @staticmethod
    def _quat_wxyz_from_rpy_rad(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return torch.stack((w, x, y, z), dim=-1)

    @staticmethod
    def _quat_wxyz_to_rpy_rad(quat_wxyz: torch.Tensor) -> torch.Tensor:
        q = quat_wxyz / (torch.norm(quat_wxyz, dim=-1, keepdim=True) + 1e-8)
        w, x, y, z = q.unbind(-1)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)
        pitch = torch.asin(t2)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(t3, t4)
        return torch.stack((roll, pitch, yaw), dim=-1)

    @staticmethod
    def _quat_wxyz_to_rotvec(quat_wxyz: torch.Tensor) -> torch.Tensor:
        q = quat_wxyz / (torch.norm(quat_wxyz, dim=-1, keepdim=True) + 1e-8)
        q = torch.where(q[:, :1] < 0.0, -q, q)
        w = torch.clamp(q[:, 0], -1.0, 1.0)
        v = q[:, 1:]
        v_norm = torch.norm(v, dim=-1)
        angle = 2.0 * torch.atan2(v_norm, w)
        axis = v / (v_norm.unsqueeze(-1) + 1e-8)
        rotvec = axis * angle.unsqueeze(-1)
        small = v_norm < 1e-6
        if small.any():
            rotvec = rotvec.clone()
            rotvec[small] = 0.0
        return rotvec

    @staticmethod
    def _clamp_rotvec(rotvec: torch.Tensor, max_norm: float) -> torch.Tensor:
        n = torch.norm(rotvec, dim=-1, keepdim=True)
        scale = torch.clamp(max_norm / (n + 1e-8), max=1.0)
        return rotvec * scale

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b, ee_quat_b, self._offset_pos, self._offset_rot
        )
        return ee_pos_b, ee_quat_b

    def _compute_frame_jacobian(self) -> torch.Tensor:
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        jacobian[:, 0:3, :] += torch.bmm(
            -math_utils.skew_symmetric_matrix(self._offset_pos),
            jacobian[:, 3:, :],
        )
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian

    def _pre_physics_step(self, actions: torch.Tensor):
        self.current_actions = actions.clone()
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        self.processed_actions[:, :3] = self.current_actions[:, :3] * self.action_scale
        rpy_curr = self._quat_wxyz_to_rpy_rad(ee_quat_curr_b)
        yaw_curr = rpy_curr[:, 2]
        dyaw_cmd = self.current_actions[:, 3] * self.action_scale * 10.0
        yaw_des = yaw_curr + dyaw_cmd
        roll_des = torch.full_like(yaw_des, self._ee_down_roll)
        pitch_des = torch.full_like(yaw_des, self._ee_down_pitch)
        ee_quat_des_b = self._quat_wxyz_from_rpy_rad(roll_des, pitch_des, yaw_des)
        quat_err = self._quat_mul_wxyz(ee_quat_des_b, math_utils.quat_inv(ee_quat_curr_b))
        rotvec = self._quat_wxyz_to_rotvec(quat_err)
        rotvec = self._clamp_rotvec(rotvec, self._max_rot_step)
        self.processed_actions[:, 3:6] = rotvec * self._rot_gain
        self._ik_controller.set_command(self.processed_actions, ee_pos_curr_b, ee_quat_curr_b)
        self._apply_joint_control()

    def _apply_joint_control(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]
        jacobian = self._compute_frame_jacobian()
        arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)[:, :7]
        gripper_width = torch.clamp(self.current_actions[:, 4], min=CLOSED_GRIPPER_WIDTH, max=OPEN_GRIPPER_WIDTH)
        gripper_pos_des = 0.5 * gripper_width.unsqueeze(-1).expand(-1, len(self._finger_joint_ids))
        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_pos_des], dim=1)
        self._robot.set_joint_position_target(joint_pos_des)

    def step_control(self, dx: float, dy: float, dz: float, dyaw: float, gripper_width: float):
        actions = torch.zeros((self.num_envs, 5), device=self.device)
        actions[:, 0] = dx
        actions[:, 1] = dy
        actions[:, 2] = dz
        actions[:, 3] = dyaw
        actions[:, 4] = gripper_width
        self._pre_physics_step(actions)
        for _ in range(self.control_decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
        self.sim.render()
        self._control_step += 1

    def settle(self, gripper_width: float, steps: int):
        for _ in range(steps):
            self.step_control(0.0, 0.0, 0.0, 0.0, gripper_width)

    def _set_object_pose(self, asset: RigidObject, position_xyz: tuple[float, float, float], yaw_rad: float):
        quat = torch.tensor(
            [[math.cos(yaw_rad * 0.5), 0.0, 0.0, math.sin(yaw_rad * 0.5)]],
            device=self.device,
            dtype=torch.float32,
        )
        pos = torch.tensor([list(position_xyz)], device=self.device, dtype=torch.float32)
        pose = torch.cat([pos, quat], dim=-1)
        asset.write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=self.device, dtype=torch.long))
        asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device), env_ids=torch.tensor([0], device=self.device, dtype=torch.long))

    def prepare_episode(self, spec: ObjectSpec, rng: random.Random):
        self.reset()
        self._active_object_id = spec.id
        self._active_object_spec = spec

        for object_id, asset in self._objects.items():
            if object_id == spec.id:
                continue
            self._set_object_pose(asset, (0.25, -0.30, -0.50), 0.0)

        x = rng.uniform(0.44, 0.56)
        y = rng.uniform(-0.08, 0.08)
        yaw = rng.uniform(-math.pi, math.pi)
        z = TABLE_TOP_Z + spec.half_height_m + 0.002
        self._set_object_pose(self._objects[spec.id], (x, y, z), yaw)
        self.settle(gripper_width=OPEN_GRIPPER_WIDTH, steps=args_cli.settle_steps)

    def get_active_object_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._active_object_id is None:
            raise RuntimeError("No active object selected.")
        asset = self._objects[self._active_object_id]
        return asset.data.root_pos_w[0], asset.data.root_quat_w[0]

    def get_active_object_top_z(self) -> float:
        if self._active_object_spec is None:
            raise RuntimeError("No active object selected.")
        position, _ = self.get_active_object_pose()
        return float(position[2].item() + self._active_object_spec.half_height_m)

    def get_sensor_world_pose(self, sensor_attr: str) -> tuple[torch.Tensor, torch.Tensor]:
        sensor = getattr(self, sensor_attr)
        positions, orientations = sensor.prim_view.get_world_poses()
        pos = torch.as_tensor(positions, device=self.device, dtype=torch.float32)
        quat = torch.as_tensor(orientations, device=self.device, dtype=torch.float32)
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
        if quat.ndim == 1:
            quat = quat.unsqueeze(0)
        return pos, quat

    def sensor_offset_from_ee(self, sensor_attr: str) -> torch.Tensor:
        sensor_pos, _ = self.get_sensor_world_pose(sensor_attr)
        ee_pos, _ = self._compute_frame_pose()
        return sensor_pos - ee_pos

    def sensor_camera_offset_from_ee(self, sensor_attr: str) -> torch.Tensor:
        sensor = getattr(self, sensor_attr)
        if sensor.camera is None:
            raise RuntimeError(f"Sensor {sensor_attr} does not have an internal camera.")
        camera_pos = sensor.camera.data.pos_w.to(device=self.device, dtype=torch.float32)
        if camera_pos.ndim == 1:
            camera_pos = camera_pos.unsqueeze(0)
        ee_pos, _ = self._compute_frame_pose()
        return camera_pos - ee_pos

    def sensor_contact_center_offset_from_ee(self, sensor_attr: str) -> torch.Tensor:
        if sensor_attr in DOWN_SENSOR_CONTACT_FACE_CENTER_OFFSETS:
            sensor_pos, sensor_quat = self.get_sensor_world_pose(sensor_attr)
            if sensor_pos.ndim == 1:
                sensor_pos = sensor_pos.unsqueeze(0)
            if sensor_quat.ndim == 1:
                sensor_quat = sensor_quat.unsqueeze(0)
            local_offset = torch.tensor(
                [DOWN_SENSOR_CONTACT_FACE_CENTER_OFFSETS[sensor_attr]],
                device=self.device,
                dtype=torch.float32,
            )
            contact_center_pos = sensor_pos + math_utils.quat_apply(sensor_quat, local_offset)
            ee_pos, _ = self._compute_frame_pose()
            return contact_center_pos - ee_pos

        sensor = getattr(self, sensor_attr)
        if sensor.camera is None:
            raise RuntimeError(f"Sensor {sensor_attr} does not have an internal camera.")
        camera_pos = sensor.camera.data.pos_w.to(device=self.device, dtype=torch.float32)
        camera_quat_ros = sensor.camera.data.quat_w_ros.to(device=self.device, dtype=torch.float32)
        if camera_pos.ndim == 1:
            camera_pos = camera_pos.unsqueeze(0)
        if camera_quat_ros.ndim == 1:
            camera_quat_ros = camera_quat_ros.unsqueeze(0)
        contact_distance_m = (
            float(sensor.cfg.optical_sim_cfg.gelpad_to_camera_min_distance) + float(sensor.cfg.gelpad_dimensions.height)
        )
        local_forward = torch.zeros((camera_pos.shape[0], 3), device=self.device, dtype=torch.float32)
        local_forward[:, 2] = contact_distance_m
        contact_center_pos = camera_pos + math_utils.quat_apply(camera_quat_ros, local_forward)
        ee_pos, _ = self._compute_frame_pose()
        return contact_center_pos - ee_pos

    @staticmethod
    def _to_numpy(value: torch.Tensor | None, squeeze_env: bool = True) -> np.ndarray:
        if value is None:
            return np.zeros((0,), dtype=np.float32)
        tensor = value.detach().cpu()
        if squeeze_env and tensor.ndim >= 1 and tensor.shape[0] == 1:
            tensor = tensor[0]
        return tensor.numpy()

    def _contact_ratio(self, tensor: torch.Tensor | None, threshold: float = 0.80) -> float:
        if tensor is None:
            return 0.0
        d = tensor.to(device=self.device, dtype=torch.float32)
        if d.numel() > 0 and d.max() > 2.0:
            d = d / 255.0
        contact = (d < threshold).float()
        return float(contact.mean().item())

    def contact_ratios(self) -> dict[str, float]:
        return {
            "inner_left": self._contact_ratio(self.gsmini_left.data.output.get("camera_depth")),
            "inner_right": self._contact_ratio(self.gsmini_right.data.output.get("camera_depth")),
            "down_left": self._contact_ratio(self.gsmini_left_down.data.output.get("camera_depth")),
            "down_right": self._contact_ratio(self.gsmini_right_down.data.output.get("camera_depth")),
        }

    def capture_frame(self) -> dict[str, np.ndarray]:
        ee_pos, ee_quat = self._compute_frame_pose()
        object_pos, object_quat = self.get_active_object_pose()
        contact = self.contact_ratios()
        third_rgb = self.third_person_camera.data.output.get("rgb")
        if third_rgb is not None and third_rgb.shape[-1] > 3:
            third_rgb = third_rgb[..., :3]
        frame = {
            "third_person_rgb": self._to_numpy(third_rgb),
            "tactile_inner_left_depth": self._to_numpy(self.gsmini_left.data.output.get("camera_depth")),
            "tactile_inner_right_depth": self._to_numpy(self.gsmini_right.data.output.get("camera_depth")),
            "tactile_down_left_depth": self._to_numpy(self.gsmini_left_down.data.output.get("camera_depth")),
            "tactile_down_right_depth": self._to_numpy(self.gsmini_right_down.data.output.get("camera_depth")),
            "tactile_inner_left_rgb": self._to_numpy(self.gsmini_left.data.output.get("tactile_rgb")),
            "tactile_inner_right_rgb": self._to_numpy(self.gsmini_right.data.output.get("tactile_rgb")),
            "tactile_down_left_rgb": self._to_numpy(self.gsmini_left_down.data.output.get("tactile_rgb")),
            "tactile_down_right_rgb": self._to_numpy(self.gsmini_right_down.data.output.get("tactile_rgb")),
            "joint_pos": self._to_numpy(self._robot.data.joint_pos),
            "joint_vel": self._to_numpy(self._robot.data.joint_vel),
            "gripper_width": np.asarray([float(self._robot.data.joint_pos[0, self._finger_joint_ids].sum().item())], dtype=np.float32),
            "ee_pose": np.concatenate([self._to_numpy(ee_pos), self._to_numpy(ee_quat)], axis=-1),
            "object_pose": np.concatenate([self._to_numpy(object_pos, squeeze_env=False), self._to_numpy(object_quat, squeeze_env=False)], axis=-1),
            "contact_inner_left": np.asarray([contact["inner_left"]], dtype=np.float32),
            "contact_inner_right": np.asarray([contact["inner_right"]], dtype=np.float32),
            "contact_down_left": np.asarray([contact["down_left"]], dtype=np.float32),
            "contact_down_right": np.asarray([contact["down_right"]], dtype=np.float32),
        }
        return frame


def _sample_selection(rng: random.Random) -> EpisodeSelection:
    if args_cli.object_id is not None:
        spec = OBJECT_SPECS[args_cli.object_id]
    else:
        spec = OBJECT_SPECS[rng.choice(sorted(OBJECT_SPECS.keys()))]
    compatible_primitives = [p for p in spec.target_primitives if p in SUPPORTED_PRIMITIVES]
    if not compatible_primitives:
        raise RuntimeError(f"Object {spec.id} does not support any implemented primitive.")
    if args_cli.primitive == "random":
        primitive = rng.choice(compatible_primitives)
    else:
        if args_cli.primitive not in compatible_primitives:
            raise ValueError(
                f"Primitive {args_cli.primitive!r} is not compatible with object {spec.id}. "
                f"Compatible: {compatible_primitives}"
            )
        primitive = args_cli.primitive
    return EpisodeSelection(object_spec=spec, primitive=primitive)


def _stage_steps(primitive: str) -> dict[str, int]:
    steps: dict[str, int] = {}
    for stage in COLLECTION_PROTOCOL["primitives"][primitive]["stages"]:
        duration_s = float(stage["duration_s"])
        if stage["name"] in CONTACT_RICH_STAGES.get(primitive, set()):
            duration_s *= max(args_cli.contact_duration_scale, 1.0)
        steps[stage["name"]] = max(1, int(round(duration_s * 20.0)))
    return steps


def _servo_to_target(
    env: BootstrapCollectorEnv,
    recorder: EpisodeRecorder,
    target_xyz: tuple[float, float, float],
    gripper_width: float,
    stage_name: str,
    max_steps: int,
    tol_m: float = 0.004,
    contact_group: str | None = None,
    contact_threshold: float = 0.004,
    move_xy: bool = True,
    move_z: bool = True,
):
    target = torch.tensor(target_xyz, device=env.device, dtype=torch.float32)
    for _ in range(max_steps):
        ee_pos, _ = env._compute_frame_pose()
        delta = target - ee_pos[0]
        if not move_xy:
            delta[0] = 0.0
            delta[1] = 0.0
        if not move_z:
            delta[2] = 0.0
        if torch.norm(delta).item() < tol_m:
            break
        step_world = torch.clamp(delta, min=-env.action_scale, max=env.action_scale)
        action_units = step_world / env.action_scale
        env.step_control(
            dx=float(action_units[0].item()),
            dy=float(action_units[1].item()),
            dz=float(action_units[2].item()),
            dyaw=0.0,
            gripper_width=gripper_width,
        )
        recorder.add_frame(env, stage_name=stage_name, control_step=env._control_step)
        if contact_group is not None:
            ratios = env.contact_ratios()
            if contact_group == "down":
                signal = 0.5 * (ratios["down_left"] + ratios["down_right"])
            elif contact_group == "inner_bilateral":
                signal = min(ratios["inner_left"], ratios["inner_right"])
            else:
                raise ValueError(f"Unknown contact_group={contact_group!r}")
            if signal >= contact_threshold:
                break


def _hold(
    env: BootstrapCollectorEnv,
    recorder: EpisodeRecorder,
    steps: int,
    stage_name: str,
    gripper_width: float,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
):
    for _ in range(steps):
        env.step_control(dx=dx, dy=dy, dz=dz, dyaw=0.0, gripper_width=gripper_width)
        recorder.add_frame(env, stage_name=stage_name, control_step=env._control_step)


def _search_down_until_sensor_contact(
    env: BootstrapCollectorEnv,
    recorder: EpisodeRecorder,
    contact_key: str,
    gripper_width: float,
    stage_name: str,
    max_steps: int,
    step_world_m: float = 0.002,
    contact_threshold: float = 0.002,
):
    dz_cmd = -abs(step_world_m / env.action_scale)
    for _ in range(max_steps):
        env.step_control(dx=0.0, dy=0.0, dz=dz_cmd, dyaw=0.0, gripper_width=gripper_width)
        recorder.add_frame(env, stage_name=stage_name, control_step=env._control_step)
        if env.contact_ratios()[contact_key] >= contact_threshold:
            return True
    return False


def _quat_yaw_from_wxyz(quat_wxyz: torch.Tensor) -> float:
    q = quat_wxyz.detach().cpu().numpy()
    w, x, y, z = (float(v) for v in q)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def _rotate_xy(xy: tuple[float, float], yaw_rad: float) -> tuple[float, float]:
    x, y = xy
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return (c * x - s * y, s * x + c * y)


def _sample_top_contact_offset(spec: ObjectSpec, primitive: str, rng: random.Random) -> tuple[str, tuple[float, float]]:
    forced_mode = args_cli.force_contact_mode
    if forced_mode != "auto":
        forced = _forced_top_contact_offset(spec, forced_mode, rng)
        if forced is not None:
            return forced

    if spec.shape == "sphere":
        radius = 0.5 * spec.size_m[0]
        if primitive == "touch":
            choices = [("center", 0.0), ("off_axis_soft", 0.18 * radius), ("off_axis_mid", 0.30 * radius)]
            weights = [0.45, 0.35, 0.20]
        elif spec.size_m[0] <= 0.04:
            choices = [("center", 0.0), ("off_axis_soft", 0.15 * radius), ("off_axis_mid", 0.24 * radius)]
            weights = [0.55, 0.30, 0.15]
        else:
            choices = [("center", 0.0), ("off_axis_soft", 0.18 * radius), ("off_axis_mid", 0.28 * radius)]
            weights = [0.40, 0.35, 0.25]
        mode, radial = rng.choices(choices, weights=weights, k=1)[0]
        theta = 0.0 if radial <= 0.0 else rng.uniform(-math.pi, math.pi)
        return mode, (radial * math.cos(theta), radial * math.sin(theta))

    if spec.shape == "cylinder":
        radius = spec.size_m[0]
        choices = [("top_center", 0.0), ("top_band", 0.38 * radius), ("top_rim", 0.68 * radius)]
        weights = [0.30, 0.45, 0.25] if primitive == "touch" else [0.20, 0.35, 0.45]
        mode, radial = rng.choices(choices, weights=weights, k=1)[0]
        theta = 0.0 if radial <= 0.0 else rng.uniform(-math.pi, math.pi)
        return mode, (radial * math.cos(theta), radial * math.sin(theta))

    if spec.shape in {"box", "plate"}:
        hx = 0.5 * spec.size_m[0]
        hy = 0.5 * spec.size_m[1]
        safe_x = max(hx - 0.004, 0.002)
        safe_y = max(hy - 0.004, 0.002)
        choices = ["face_center", "x_edge", "y_edge", "corner"]
        weights = [0.35, 0.25, 0.25, 0.15] if primitive == "touch" else [0.20, 0.30, 0.25, 0.25]
        mode = rng.choices(choices, weights=weights, k=1)[0]
        if mode == "face_center":
            return mode, (rng.uniform(-0.002, 0.002), rng.uniform(-0.002, 0.002))
        if mode == "x_edge":
            return mode, (rng.choice([-1.0, 1.0]) * 0.55 * safe_x, rng.uniform(-0.18 * safe_y, 0.18 * safe_y))
        if mode == "y_edge":
            return mode, (rng.uniform(-0.18 * safe_x, 0.18 * safe_x), rng.choice([-1.0, 1.0]) * 0.55 * safe_y)
        return mode, (rng.choice([-1.0, 1.0]) * 0.42 * safe_x, rng.choice([-1.0, 1.0]) * 0.42 * safe_y)

    return "center_fallback", (0.0, 0.0)


def _forced_top_contact_offset(
    spec: ObjectSpec, forced_mode: str, rng: random.Random
) -> tuple[str, tuple[float, float]] | None:
    if spec.shape == "sphere":
        radius = 0.5 * spec.size_m[0]
        if forced_mode == "center":
            return ("center", (0.0, 0.0))
        if forced_mode == "off_axis_soft":
            theta = rng.uniform(-math.pi, math.pi)
            radial = 0.18 * radius
            return (forced_mode, (radial * math.cos(theta), radial * math.sin(theta)))
        if forced_mode == "off_axis_mid":
            theta = rng.uniform(-math.pi, math.pi)
            radial = 0.30 * radius
            return (forced_mode, (radial * math.cos(theta), radial * math.sin(theta)))
        return None

    if spec.shape == "cylinder":
        radius = spec.size_m[0]
        radial_lookup = {"top_center": 0.0, "top_band": 0.38 * radius, "top_rim": 0.68 * radius}
        if forced_mode in radial_lookup:
            radial = radial_lookup[forced_mode]
            theta = 0.0 if radial <= 0.0 else rng.uniform(-math.pi, math.pi)
            return (forced_mode, (radial * math.cos(theta), radial * math.sin(theta)))
        return None

    if spec.shape in {"box", "plate"}:
        hx = 0.5 * spec.size_m[0]
        hy = 0.5 * spec.size_m[1]
        safe_x = max(hx - 0.004, 0.002)
        safe_y = max(hy - 0.004, 0.002)
        edge_x = 0.55 * safe_x
        edge_y = 0.55 * safe_y
        corner_x = 0.42 * safe_x
        corner_y = 0.42 * safe_y
        if forced_mode == "face_center":
            return ("face_center", (0.0, 0.0))
        if forced_mode in {"x_edge", "x_edge_pos", "x_edge_neg"}:
            sign = rng.choice([-1.0, 1.0]) if forced_mode == "x_edge" else (1.0 if forced_mode.endswith("pos") else -1.0)
            return ("x_edge", (sign * edge_x, 0.0))
        if forced_mode in {"y_edge", "y_edge_pos", "y_edge_neg"}:
            sign = rng.choice([-1.0, 1.0]) if forced_mode == "y_edge" else (1.0 if forced_mode.endswith("pos") else -1.0)
            return ("y_edge", (0.0, sign * edge_y))
        if forced_mode in {"corner", "corner_pp", "corner_pn", "corner_np", "corner_nn"}:
            if forced_mode == "corner":
                sx = rng.choice([-1.0, 1.0])
                sy = rng.choice([-1.0, 1.0])
            else:
                suffix = forced_mode.split("_", 1)[1]
                sx = 1.0 if suffix[0] == "p" else -1.0
                sy = 1.0 if suffix[1] == "p" else -1.0
            return ("corner", (sx * corner_x, sy * corner_y))
        return None

    return None


def _sample_single_pad_contact_plan(
    env: BootstrapCollectorEnv,
    spec: ObjectSpec,
    primitive: str,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    rng: random.Random,
) -> ContactPlan:
    press_depth_override_m = None if args_cli.press_depth_mm is None else max(args_cli.press_depth_mm, 0.0) / 1000.0
    deterministic_mode = args_cli.contact_plan_mode == "centered" or args_cli.force_contact_mode != "auto"
    if deterministic_mode:
        if args_cli.center_primary_sensor == "random":
            sensor_attr = rng.choice(["gsmini_left_down", "gsmini_right_down"])
        else:
            sensor_attr = "gsmini_left_down" if args_cli.center_primary_sensor == "left_down" else "gsmini_right_down"
        contact_key = "down_left" if sensor_attr == "gsmini_left_down" else "down_right"
        if args_cli.force_contact_mode != "auto":
            contact_mode, local_offset_xy = _sample_top_contact_offset(spec, primitive, rng)
        else:
            contact_mode, local_offset_xy = ("center_locked", (0.0, 0.0))
        object_yaw = _quat_yaw_from_wxyz(object_quat)
        world_offset_xy = _rotate_xy(local_offset_xy, object_yaw)
        target_world_xy = (
            float(object_pos[0].item() + world_offset_xy[0]),
            float(object_pos[1].item() + world_offset_xy[1]),
        )
        if primitive in {"press", "touch"} and sensor_attr == "gsmini_right_down":
            # Temporary calibration nudge: shift the down-right target 1.2 cm toward
            # the right side in the current world/camera setup.
            target_world_xy = (target_world_xy[0], target_world_xy[1] + 0.012)
        if spec.shape == "sphere":
            sensor_hover_clearance_m = 0.020 if primitive == "press" else 0.024
            search_step_m = 0.0008 if primitive == "press" else 0.0010
            search_contact_threshold = 0.0045 if primitive == "press" else 0.0025
            touch_width_m = min(max(spec.nominal_grasp_width_m * 0.98, 0.045), 0.06)
            press_extra_depth_m = 0.015 if primitive == "press" else 0.0
        elif spec.shape == "cylinder":
            sensor_hover_clearance_m = 0.023 if primitive == "press" else 0.027
            search_step_m = 0.0010 if primitive == "press" else 0.0014
            search_contact_threshold = 0.0035 if primitive == "press" else 0.0020
            touch_width_m = min(max(spec.nominal_grasp_width_m * 0.98, 0.045), 0.06)
            press_extra_depth_m = 0.010 if primitive == "press" else 0.0
        else:
            sensor_hover_clearance_m = 0.024 if primitive == "press" else 0.028
            search_step_m = 0.0012 if primitive == "press" else 0.0016
            search_contact_threshold = 0.0030 if primitive == "press" else 0.0022
            touch_width_m = min(max(spec.nominal_grasp_width_m * 0.98, 0.045), 0.06)
            press_extra_depth_m = 0.008 if primitive == "press" else 0.0
        if primitive == "press" and press_depth_override_m is not None:
            press_extra_depth_m = press_depth_override_m

        return ContactPlan(
            primitive=primitive,
            primary_sensor=sensor_attr,
            contact_key=contact_key,
            contact_mode=contact_mode,
            object_local_offset_xy_m=(float(local_offset_xy[0]), float(local_offset_xy[1])),
            object_world_offset_xy_m=(float(world_offset_xy[0]), float(world_offset_xy[1])),
            target_world_xy_m=target_world_xy,
            gripper_width_m=float(touch_width_m),
            sensor_hover_clearance_m=float(sensor_hover_clearance_m),
            search_step_m=float(search_step_m),
            search_contact_threshold=float(search_contact_threshold),
            touch_hold_shift_xy_m=(0.0, 0.0),
            press_extra_depth_m=float(press_extra_depth_m),
            press_lateral_shift_xy_m=(0.0, 0.0),
        )

    sensor_attr = rng.choice(["gsmini_left_down", "gsmini_right_down"])
    contact_key = "down_left" if sensor_attr == "gsmini_left_down" else "down_right"
    contact_mode, local_offset_xy = _sample_top_contact_offset(spec, primitive, rng)
    object_yaw = _quat_yaw_from_wxyz(object_quat)
    world_offset_xy = _rotate_xy(local_offset_xy, object_yaw)
    target_world_xy = (
        float(object_pos[0].item() + world_offset_xy[0]),
        float(object_pos[1].item() + world_offset_xy[1]),
    )
    touch_width_m = min(max(spec.nominal_grasp_width_m * rng.uniform(0.88, 1.03), 0.043), 0.06)
    sensor_hover_clearance_m = rng.uniform(0.020, 0.032)
    if spec.shape == "sphere":
        search_step_m = rng.uniform(0.0010, 0.0018) if primitive == "press" else rng.uniform(0.0012, 0.0020)
    else:
        search_step_m = rng.uniform(0.0015, 0.0028)
    search_contact_threshold = rng.uniform(0.0015, 0.0030)

    touch_hold_shift_xy_m = (0.0, 0.0)
    press_extra_depth_m = 0.0
    press_lateral_shift_xy_m = (0.0, 0.0)

    if primitive == "touch":
        shift_mag = rng.uniform(0.0006, 0.0020) if spec.shape == "sphere" else rng.uniform(0.001, 0.004)
        shift_dir = rng.uniform(-math.pi, math.pi)
        touch_hold_shift_xy_m = (shift_mag * math.cos(shift_dir), shift_mag * math.sin(shift_dir))
    elif primitive == "press":
        if spec.shape == "sphere":
            shift_mag = rng.uniform(0.0005, 0.0018)
            press_extra_depth_m = rng.uniform(0.006, 0.010)
        elif spec.shape == "cylinder":
            shift_mag = rng.uniform(0.0010, 0.0030)
            press_extra_depth_m = rng.uniform(0.005, 0.010)
        else:
            shift_mag = rng.uniform(0.001, 0.004)
            press_extra_depth_m = rng.uniform(0.004, 0.010)
        shift_dir = rng.uniform(-math.pi, math.pi)
        press_lateral_shift_xy_m = (shift_mag * math.cos(shift_dir), shift_mag * math.sin(shift_dir))
        if press_depth_override_m is not None:
            press_extra_depth_m = press_depth_override_m

    return ContactPlan(
        primitive=primitive,
        primary_sensor=sensor_attr,
        contact_key=contact_key,
        contact_mode=contact_mode,
        object_local_offset_xy_m=(float(local_offset_xy[0]), float(local_offset_xy[1])),
        object_world_offset_xy_m=(float(world_offset_xy[0]), float(world_offset_xy[1])),
        target_world_xy_m=target_world_xy,
        gripper_width_m=float(touch_width_m),
        sensor_hover_clearance_m=float(sensor_hover_clearance_m),
        search_step_m=float(search_step_m),
        search_contact_threshold=float(search_contact_threshold),
        touch_hold_shift_xy_m=(float(touch_hold_shift_xy_m[0]), float(touch_hold_shift_xy_m[1])),
        press_extra_depth_m=float(press_extra_depth_m),
        press_lateral_shift_xy_m=(float(press_lateral_shift_xy_m[0]), float(press_lateral_shift_xy_m[1])),
    )


def _single_pad_hover_pose(env: BootstrapCollectorEnv, top_z: float, contact_plan: ContactPlan) -> tuple[float, float, float]:
    align_offset = _single_pad_alignment_offset(env, contact_plan)
    return (
        float(contact_plan.target_world_xy_m[0] - align_offset[0].item()),
        float(contact_plan.target_world_xy_m[1] - align_offset[1].item()),
        float(top_z + contact_plan.sensor_hover_clearance_m - align_offset[2].item()),
    )


def _single_pad_alignment_offset(env: BootstrapCollectorEnv, contact_plan: ContactPlan) -> torch.Tensor:
    sensor_offset = env.sensor_offset_from_ee(contact_plan.primary_sensor)[0]
    align_offset = sensor_offset
    if args_cli.contact_plan_mode == "centered":
        try:
            align_offset = env.sensor_contact_center_offset_from_ee(contact_plan.primary_sensor)[0]
        except Exception:
            pass
    return align_offset


def _single_pad_target_pose(
    env: BootstrapCollectorEnv, top_z: float, contact_plan: ContactPlan, contact_depth_m: float
) -> tuple[float, float, float]:
    align_offset = _single_pad_alignment_offset(env, contact_plan)
    return (
        float(contact_plan.target_world_xy_m[0] - align_offset[0].item()),
        float(contact_plan.target_world_xy_m[1] - align_offset[1].item()),
        float(top_z - contact_depth_m - align_offset[2].item()),
    )


def _deterministic_contact_execution_enabled() -> bool:
    return args_cli.contact_plan_mode == "centered" or args_cli.force_contact_mode != "auto"


def _action_units_from_world_delta(
    env: BootstrapCollectorEnv, delta_xy_m: tuple[float, float], delta_z_m: float, steps: int
) -> tuple[float, float, float]:
    if steps <= 0:
        return (0.0, 0.0, 0.0)
    return (
        float(delta_xy_m[0] / (steps * env.action_scale)),
        float(delta_xy_m[1] / (steps * env.action_scale)),
        float(delta_z_m / (steps * env.action_scale)),
    )


def _run_touch(env: BootstrapCollectorEnv, recorder: EpisodeRecorder, selection: EpisodeSelection, rng: random.Random):
    spec = selection.object_spec
    steps = _stage_steps("touch")
    object_pos, object_quat = env.get_active_object_pose()
    contact_plan = _sample_single_pad_contact_plan(env, spec, "touch", object_pos, object_quat, rng)
    recorder.set_contact_plan(contact_plan)
    top_z = env.get_active_object_top_z()
    center_hover = (
        float(contact_plan.target_world_xy_m[0]),
        float(contact_plan.target_world_xy_m[1]),
        top_z + spec.top_clearance_hover_m,
    )
    single_pad_hover = _single_pad_hover_pose(env, top_z, contact_plan)
    _servo_to_target(
        env, recorder, center_hover, contact_plan.gripper_width_m, "approach", max_steps=steps["approach"] + 6
    )
    if _deterministic_contact_execution_enabled():
        planar_hover = (single_pad_hover[0], single_pad_hover[1], center_hover[2])
        _servo_to_target(
            env,
            recorder,
            planar_hover,
            contact_plan.gripper_width_m,
            "approach",
            max_steps=max(4, steps["approach"] // 2),
            move_xy=True,
            move_z=False,
        )
        _servo_to_target(
            env,
            recorder,
            single_pad_hover,
            contact_plan.gripper_width_m,
            "approach",
            max_steps=max(4, steps["approach"] // 2),
            move_xy=False,
            move_z=True,
            tol_m=0.0015,
        )
        contact_depth_m = 0.0015 if spec.shape == "sphere" else 0.0010
        contact_pose = _single_pad_target_pose(env, top_z, contact_plan, contact_depth_m=contact_depth_m)
        _servo_to_target(
            env,
            recorder,
            contact_pose,
            contact_plan.gripper_width_m,
            "initial_contact",
            max_steps=max(steps["initial_contact"], 18),
            move_xy=False,
            move_z=True,
            tol_m=0.0015,
        )
        _hold(env, recorder, steps["hold"], "hold", contact_plan.gripper_width_m)
        _servo_to_target(
            env, recorder, single_pad_hover, contact_plan.gripper_width_m, "release", max_steps=steps["release"] + 6
        )
    else:
        _servo_to_target(
            env,
            recorder,
            single_pad_hover,
            contact_plan.gripper_width_m,
            "approach",
            max_steps=max(4, steps["approach"] // 2),
        )
        _search_down_until_sensor_contact(
            env,
            recorder,
            contact_plan.contact_key,
            contact_plan.gripper_width_m,
            "initial_contact",
            max_steps=steps["initial_contact"] + 10,
            step_world_m=contact_plan.search_step_m,
            contact_threshold=contact_plan.search_contact_threshold,
        )
        hold_steps = steps["hold"]
        forward_steps = max(1, hold_steps // 2)
        return_steps = max(0, hold_steps - forward_steps)
        dx, dy, _ = _action_units_from_world_delta(env, contact_plan.touch_hold_shift_xy_m, 0.0, forward_steps)
        _hold(env, recorder, forward_steps, "hold", contact_plan.gripper_width_m, dx=dx, dy=dy)
        if return_steps > 0:
            dx_back, dy_back, _ = _action_units_from_world_delta(
                env,
                (-contact_plan.touch_hold_shift_xy_m[0], -contact_plan.touch_hold_shift_xy_m[1]),
                0.0,
                return_steps,
            )
            _hold(env, recorder, return_steps, "hold", contact_plan.gripper_width_m, dx=dx_back, dy=dy_back)
        _servo_to_target(
            env, recorder, single_pad_hover, contact_plan.gripper_width_m, "release", max_steps=steps["release"] + 6
        )


def _run_press(env: BootstrapCollectorEnv, recorder: EpisodeRecorder, selection: EpisodeSelection, rng: random.Random):
    spec = selection.object_spec
    steps = _stage_steps("press")
    object_pos, object_quat = env.get_active_object_pose()
    contact_plan = _sample_single_pad_contact_plan(env, spec, "press", object_pos, object_quat, rng)
    recorder.set_contact_plan(contact_plan)
    top_z = env.get_active_object_top_z()
    center_hover = (
        float(contact_plan.target_world_xy_m[0]),
        float(contact_plan.target_world_xy_m[1]),
        top_z + spec.top_clearance_hover_m,
    )
    single_pad_hover = _single_pad_hover_pose(env, top_z, contact_plan)
    _servo_to_target(
        env, recorder, center_hover, contact_plan.gripper_width_m, "approach", max_steps=steps["approach"] + 6
    )
    if _deterministic_contact_execution_enabled():
        planar_hover = (single_pad_hover[0], single_pad_hover[1], center_hover[2])
        _servo_to_target(
            env,
            recorder,
            planar_hover,
            contact_plan.gripper_width_m,
            "approach",
            max_steps=max(4, steps["approach"] // 2),
            move_xy=True,
            move_z=False,
        )
        _servo_to_target(
            env,
            recorder,
            single_pad_hover,
            contact_plan.gripper_width_m,
            "approach",
            max_steps=max(4, steps["approach"] // 2),
            move_xy=False,
            move_z=True,
            tol_m=0.0015,
        )
        contact_depth_m = 0.0015 if spec.shape == "sphere" else 0.0010
        contact_pose = _single_pad_target_pose(env, top_z, contact_plan, contact_depth_m=contact_depth_m)
        press_pose = _single_pad_target_pose(env, top_z, contact_plan, contact_depth_m=contact_plan.press_extra_depth_m)
        _servo_to_target(
            env,
            recorder,
            contact_pose,
            contact_plan.gripper_width_m,
            "press_ramp",
            max_steps=max(steps["press_ramp"], 18),
            move_xy=False,
            move_z=True,
            tol_m=0.0015,
        )
        _servo_to_target(
            env,
            recorder,
            press_pose,
            contact_plan.gripper_width_m,
            "press_ramp",
            max_steps=max(steps["press_ramp"], 24),
            move_xy=False,
            move_z=True,
            tol_m=0.0010,
        )
        _hold(env, recorder, steps["peak_hold"], "peak_hold", contact_plan.gripper_width_m)
    else:
        _servo_to_target(
            env,
            recorder,
            single_pad_hover,
            contact_plan.gripper_width_m,
            "approach",
            max_steps=max(4, steps["approach"] // 2),
        )
        search_steps = max(4, steps["press_ramp"] // 3)
        _search_down_until_sensor_contact(
            env,
            recorder,
            contact_plan.contact_key,
            contact_plan.gripper_width_m,
            "press_ramp",
            max_steps=search_steps + 4,
            step_world_m=contact_plan.search_step_m,
            contact_threshold=contact_plan.search_contact_threshold,
        )
        ramp_steps = max(4, steps["press_ramp"] - search_steps)
        dx_press, dy_press, dz_press = _action_units_from_world_delta(
            env, contact_plan.press_lateral_shift_xy_m, -contact_plan.press_extra_depth_m, ramp_steps
        )
        _hold(
            env,
            recorder,
            ramp_steps,
            "press_ramp",
            contact_plan.gripper_width_m,
            dx=dx_press,
            dy=dy_press,
            dz=dz_press,
        )
        peak_steps = steps["peak_hold"]
        peak_dx, peak_dy, _ = _action_units_from_world_delta(
            env,
            (-0.35 * contact_plan.press_lateral_shift_xy_m[0], -0.35 * contact_plan.press_lateral_shift_xy_m[1]),
            0.0,
            max(1, peak_steps),
        )
        _hold(env, recorder, peak_steps, "peak_hold", contact_plan.gripper_width_m, dx=peak_dx, dy=peak_dy)
    _servo_to_target(
        env, recorder, single_pad_hover, contact_plan.gripper_width_m, "release", max_steps=steps["release"] + 6
    )


def _run_slide(env: BootstrapCollectorEnv, recorder: EpisodeRecorder, selection: EpisodeSelection, rng: random.Random):
    spec = selection.object_spec
    steps = _stage_steps("slide")
    object_pos, _ = env.get_active_object_pose()
    hover = (float(object_pos[0]), float(object_pos[1]), env.get_active_object_top_z() + spec.top_clearance_hover_m)
    touch = (float(object_pos[0]), float(object_pos[1]), env.get_active_object_top_z() + 0.006)
    slide_dir = rng.choice([-1.0, 1.0])
    _servo_to_target(env, recorder, hover, OPEN_GRIPPER_WIDTH, "approach", max_steps=steps["approach"] + 6)
    _servo_to_target(
        env,
        recorder,
        touch,
        OPEN_GRIPPER_WIDTH,
        "contact_establish",
        max_steps=steps["contact_establish"] + 6,
        contact_group="down",
        contact_threshold=0.002,
    )
    _hold(
        env,
        recorder,
        steps["tangential_slide"],
        "tangential_slide",
        OPEN_GRIPPER_WIDTH,
        dy=0.30 * slide_dir,
        dz=-0.06,
    )
    _servo_to_target(env, recorder, hover, OPEN_GRIPPER_WIDTH, "release", max_steps=steps["release"] + 6)


def _run_pinch_grasp(env: BootstrapCollectorEnv, recorder: EpisodeRecorder, selection: EpisodeSelection):
    spec = selection.object_spec
    steps = _stage_steps("pinch_grasp")
    object_pos, _ = env.get_active_object_pose()
    center_z = float(object_pos[2].item())
    hover = (float(object_pos[0]), float(object_pos[1]), center_z + spec.top_clearance_hover_m)
    grasp_pose = (float(object_pos[0]), float(object_pos[1]), center_z + spec.side_grasp_height_offset_m)
    _servo_to_target(env, recorder, hover, OPEN_GRIPPER_WIDTH, "pre_grasp", max_steps=steps["pre_grasp"] + 6)
    _servo_to_target(env, recorder, grasp_pose, OPEN_GRIPPER_WIDTH, "close_until_contact", max_steps=steps["close_until_contact"] + 6)

    close_steps = max(steps["close_until_contact"], 8)
    widths = np.linspace(spec.nominal_grasp_width_m, CLOSED_GRIPPER_WIDTH, close_steps)
    for width in widths:
        env.step_control(dx=0.0, dy=0.0, dz=0.0, dyaw=0.0, gripper_width=float(width))
        recorder.add_frame(env, stage_name="close_until_contact", control_step=env._control_step)
        bilateral = min(env.contact_ratios()["inner_left"], env.contact_ratios()["inner_right"])
        if bilateral >= 0.0015:
            break

    _hold(env, recorder, steps["squeeze_hold"], "squeeze_hold", CLOSED_GRIPPER_WIDTH)
    _hold(env, recorder, steps["micro_reposition"], "micro_reposition", CLOSED_GRIPPER_WIDTH, dz=0.12)
    _hold(env, recorder, max(4, steps["release"] // 2), "release", OPEN_GRIPPER_WIDTH)
    _servo_to_target(env, recorder, hover, OPEN_GRIPPER_WIDTH, "release", max_steps=steps["release"] + 6)


def _run_episode(env: BootstrapCollectorEnv, recorder: EpisodeRecorder, selection: EpisodeSelection, rng: random.Random):
    if selection.primitive == "touch":
        _run_touch(env, recorder, selection, rng)
    elif selection.primitive == "press":
        _run_press(env, recorder, selection, rng)
    elif selection.primitive == "slide":
        _run_slide(env, recorder, selection, rng)
    elif selection.primitive == "pinch_grasp":
        _run_pinch_grasp(env, recorder, selection)
    else:
        raise ValueError(f"Unsupported primitive: {selection.primitive}")


def _write_run_metadata(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": args_cli.seed,
        "num_episodes": args_cli.num_episodes,
        "splits": args_cli.splits,
        "object_id": args_cli.object_id,
        "primitive": args_cli.primitive,
        "contact_duration_scale": args_cli.contact_duration_scale,
        "contact_plan_mode": args_cli.contact_plan_mode,
        "center_primary_sensor": args_cli.center_primary_sensor,
        "force_contact_mode": args_cli.force_contact_mode,
        "press_depth_mm": args_cli.press_depth_mm,
        "contact_plan_primitives": ["touch", "press"],
        "supported_objects": sorted(OBJECT_SPECS.keys()),
        "implemented_primitives": list(SUPPORTED_PRIMITIVES),
        "limitations": [
            "bootstrap collector only supports procedurally spawned primitive-shape objects",
            "local-geometry and daily objects still need explicit USD assets or generators",
            "lift_and_disturb is intentionally left out of the first implementation",
        ],
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_collection():
    rng = random.Random(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    output_dir = Path(args_cli.output_dir)
    _write_run_metadata(output_dir)

    env = BootstrapCollectorEnv(BootstrapCollectorEnvCfg())
    env.reset()

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        for episode_idx in range(args_cli.num_episodes):
            selection = _sample_selection(rng)
            env.prepare_episode(selection.object_spec, rng)
            recorder = EpisodeRecorder(output_dir, episode_idx, selection, seed=args_cli.seed + episode_idx)
            recorder.add_frame(env, stage_name="episode_start", control_step=env._control_step)
            _run_episode(env, recorder, selection, rng)
            meta = recorder.finalize()
            if args_cli.export_mp4:
                export_dir = export_episode_videos(Path(meta["npz_path"]), fps=args_cli.mp4_fps, include_per_sensor=False)
                print(f"[collector] mp4_export={export_dir}")
            manifest_file.write(json.dumps(meta) + "\n")
            manifest_file.flush()
            print(
                f"[collector] episode={episode_idx + 1}/{args_cli.num_episodes} "
                f"object={selection.object_spec.id} primitive={selection.primitive} "
                f"frames={meta['quality']['num_frames']} "
                f"contact_mode={meta['contact_plan']['contact_mode'] if meta.get('contact_plan') else 'n/a'} "
                f"sensor={meta['contact_plan']['primary_sensor'] if meta.get('contact_plan') else 'n/a'}"
            )

    env.close()


if __name__ == "__main__":
    try:
        run_collection()
    finally:
        simulation_app.close()
