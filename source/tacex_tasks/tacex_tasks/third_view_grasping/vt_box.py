# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Occluded grasping with third-person vision and four tactile sensors."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn as nn

# for Domain Randomization
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg, TiledCamera, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
# from isaaclab.utils.noise import GaussianNoiseCfg  # not used currently
from tacex import GelSightSensor
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
try:
    import torchvision
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)

from ..sparch_grasp.sparsh_encoder import SparshFrozenEncoder, project_sparsh_features
from ..occluded_grasping.occluded_grasping_cabinet_scene import make_fixed_panel_cfg
from ..occluded_grasping.policy_alpha import get_latest_gate_mean, get_latest_gate_values
from ..occluded_grasping.policy_aux_heads import (
    get_latest_alpha,
    get_latest_m_grasp,
    get_latest_m_probe,
    get_latest_w_grasp,
    get_latest_w_probe,
)

# from tacex_tasks.utils import DirectLiveVisualizer  # unused

CAN_RADIUS = 0.025
CAN_HEIGHT = 0.07
CAN_COLLISION_REST_OFFSET = 0.0001

# Keep the tactile RL drawer scene consistent with the interactive demo scene.
DRAWER_SCALE = 1.8
DRAWER_DEPTH_SCALE = 1.2
DRAWER_WIDTH_SCALE = 1.2
DRAWER_HEIGHT_SCALE = 0.8
DRAWER_OPEN_RATIO = 1.0
DRAWER_FRONT_FACE_X = 0.335
DRAWER_CENTER_Y = 0.00

BASE_DRAWER_INNER_X = 0.2314814815
BASE_DRAWER_INNER_Y = 0.2314814815
BASE_DRAWER_WALL_THICKNESS = 0.0138888889
BASE_DRAWER_FLOOR_THICKNESS = 0.02
BASE_DRAWER_WALL_HEIGHT = 0.08
BASE_DRAWER_FRONT_PANEL_HEIGHT = 0.1
CAN_CENTROID_OFFSET_Z_DEFAULT = 0.5 * CAN_HEIGHT

BASE_DRAWER_CABINET_WALL_THICKNESS = 0.015
BASE_DRAWER_CABINET_SIDE_CLEARANCE_Y = 0.004
BASE_DRAWER_CABINET_TOP_CLEARANCE_Z = 0.006
BASE_DRAWER_CABINET_REAR_CLEARANCE_X = 0.020
BASE_DRAWER_FRONT_PANEL_SIDE_GAP = 0.004

BASE_DRAWER_HANDLE_DEPTH = 0.022
BASE_DRAWER_HANDLE_WIDTH_Y = 0.07
BASE_DRAWER_HANDLE_HEIGHT_Z = 0.01

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
# For centered cylinder root: floor top + half-height.
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

# Reset area: centered in the tray with +/- 0.075 m XY random offset.
CAN_RESET_XY_HALF_RANGE = 0.075
CAN_RANDOM_CENTER_X = BOX_CENTER_X
CAN_RANDOM_CENTER_Y = BOX_CENTER_Y
CAN_RANDOM_X_MIN = CAN_RANDOM_CENTER_X - CAN_RESET_XY_HALF_RANGE
CAN_RANDOM_X_MAX = CAN_RANDOM_CENTER_X + CAN_RESET_XY_HALF_RANGE
CAN_RANDOM_Y_MIN = CAN_RANDOM_CENTER_Y - CAN_RESET_XY_HALF_RANGE
CAN_RANDOM_Y_MAX = CAN_RANDOM_CENTER_Y + CAN_RESET_XY_HALF_RANGE


class CustomEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment."""

    def __init__(self, env: DirectRLEnvCfg, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


class _PairwiseFeatureMean(nn.Module):
    """Reduce a flattened 512-d backbone feature into 256 dims without trainable weights."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            x = x.flatten(start_dim=1)
        if x.shape[1] % 2 != 0:
            raise ValueError(f"Expected an even feature dimension, got {tuple(x.shape)}")
        return x.view(x.shape[0], x.shape[1] // 2, 2).mean(dim=-1)


_DEFAULT_DINO_ENCODER_ROOT = Path("/home/tinydog/桌面")
_DINO_ENCODER_PRESETS = {
    "dino_vitsmall": {
        "checkpoint_path": _DEFAULT_DINO_ENCODER_ROOT / "dino_vitsmall.ckpt",
        "model_size": "small",
    },
    "dino_vitbase": {
        "checkpoint_path": _DEFAULT_DINO_ENCODER_ROOT / "dino_vitbase.ckpt",
        "model_size": "base",
    },
}


def _infer_dino_model_size(name_or_path: str) -> str:
    text = str(name_or_path).lower()
    if "vitbase" in text or "base" in text:
        return "base"
    if "vitsmall" in text or "small" in text:
        return "small"
    raise ValueError(f"Could not infer DINO model size from '{name_or_path}'")


def _resolve_tactile_dino_spec(encoder_name: str, checkpoint_path: str, model_size: str) -> dict[str, str]:
    name = str(encoder_name or "").strip()
    ckpt = str(checkpoint_path or "").strip()
    size = str(model_size or "").strip().lower()

    if ckpt:
        resolved_ckpt = Path(ckpt).expanduser().resolve()
        if not resolved_ckpt.exists():
            raise FileNotFoundError(f"DINO checkpoint not found: {resolved_ckpt}")
        resolved_name = resolved_ckpt.stem
        resolved_size = size or _infer_dino_model_size(resolved_ckpt.name)
        return {
            "name": resolved_name,
            "checkpoint_path": str(resolved_ckpt),
            "model_size": resolved_size,
        }

    if not name:
        name = "dino_vitsmall"

    if name in _DINO_ENCODER_PRESETS:
        preset = _DINO_ENCODER_PRESETS[name]
        resolved_ckpt = Path(preset["checkpoint_path"]).expanduser().resolve()
        if not resolved_ckpt.exists():
            raise FileNotFoundError(
                f"DINO checkpoint preset '{name}' not found: {resolved_ckpt}. "
                "Set env.tactile_dino_checkpoint_path to a valid ckpt."
            )
        resolved_size = size or str(preset["model_size"])
        return {
            "name": name,
            "checkpoint_path": str(resolved_ckpt),
            "model_size": resolved_size,
        }

    candidate = Path(name).expanduser()
    if candidate.exists():
        resolved_ckpt = candidate.resolve()
    else:
        if candidate.suffix != ".ckpt":
            candidate_with_suffix = candidate.with_suffix(".ckpt")
            desktop_candidate = (_DEFAULT_DINO_ENCODER_ROOT / candidate_with_suffix.name).resolve()
            if candidate_with_suffix.exists():
                resolved_ckpt = candidate_with_suffix.resolve()
            elif desktop_candidate.exists():
                resolved_ckpt = desktop_candidate
            else:
                raise FileNotFoundError(
                    f"DINO checkpoint not found for '{name}'. "
                    "Set env.tactile_dino_checkpoint_path to a valid ckpt."
                )
        else:
            desktop_candidate = (_DEFAULT_DINO_ENCODER_ROOT / candidate.name).resolve()
            if desktop_candidate.exists():
                resolved_ckpt = desktop_candidate
            else:
                raise FileNotFoundError(
                    f"DINO checkpoint not found for '{name}'. "
                    "Set env.tactile_dino_checkpoint_path to a valid ckpt."
                )

    resolved_size = size or _infer_dino_model_size(resolved_ckpt.name)
    return {
        "name": resolved_ckpt.stem,
        "checkpoint_path": str(resolved_ckpt),
        "model_size": resolved_size,
    }


@configclass
class OccludedGraspingVisionFourTactileBoxCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.45, 0.85, 0.52)
    viewer.lookat = (BOX_CENTER_X, BOX_CENTER_Y, 0.08)

    debug_vis = False
    ui_window_class_type = CustomEnvWindow

    decimation = 1
    # required by DirectRLEnvCfg validation; converts seconds -> steps internally
    episode_length_s = 3.33
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            solver_type=1,
            max_position_iteration_count=32,
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,  # 使用4个环境进行测试
        env_spacing=2.0,
        replicate_physics=True,
        lazy_sensor_update=True,
    )

    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 0)),
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    box_floor = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_floor",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(BOX_CENTER_X, BOX_CENTER_Y, BOX_FLOOR_THICKNESS * 0.5)
        ),
        spawn=make_fixed_panel_cfg(
            size=(
                BOX_INNER_X + 2 * BOX_WALL_THICKNESS,
                BOX_INNER_Y + 2 * BOX_WALL_THICKNESS,
                BOX_FLOOR_THICKNESS,
            ),
            color=(0.40, 0.27, 0.18),
        ),
    )

    box_wall_front = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_wall_front",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                BOX_CENTER_X - BOX_INNER_X * 0.5 - BOX_WALL_THICKNESS * 0.5,
                BOX_CENTER_Y,
                BOX_FRONT_PANEL_CENTER_Z,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(BOX_WALL_THICKNESS, DRAWER_FRONT_PANEL_WIDTH_Y, BOX_FRONT_PANEL_HEIGHT),
            color=(0.56, 0.39, 0.26),
        ),
    )

    box_wall_back = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_wall_back",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                BOX_CENTER_X + BOX_INNER_X * 0.5 + BOX_WALL_THICKNESS * 0.5,
                BOX_CENTER_Y,
                BOX_WALL_CENTER_Z,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(BOX_WALL_THICKNESS, BOX_INNER_Y, BOX_WALL_HEIGHT),
            color=(0.50, 0.35, 0.24),
        ),
    )

    box_wall_left = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_wall_left",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                BOX_CENTER_X,
                BOX_CENTER_Y - BOX_INNER_Y * 0.5 - BOX_WALL_THICKNESS * 0.5,
                BOX_WALL_CENTER_Z,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(BOX_INNER_X, BOX_WALL_THICKNESS, BOX_WALL_HEIGHT),
            color=(0.50, 0.35, 0.24),
        ),
    )

    box_wall_right = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box_wall_right",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                BOX_CENTER_X,
                BOX_CENTER_Y + BOX_INNER_Y * 0.5 + BOX_WALL_THICKNESS * 0.5,
                BOX_WALL_CENTER_Z,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(BOX_INNER_X, BOX_WALL_THICKNESS, BOX_WALL_HEIGHT),
            color=(0.50, 0.35, 0.24),
        ),
    )

    drawer_cabinet_left_side = RigidObjectCfg(
        prim_path="/World/envs/env_.*/drawer_cabinet_left_side",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                DRAWER_CABINET_CENTER_X,
                DRAWER_CABINET_CENTER_Y - DRAWER_CABINET_OUTER_WIDTH_Y * 0.5 + DRAWER_CABINET_WALL_THICKNESS * 0.5,
                DRAWER_CABINET_HEIGHT * 0.5,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(
                DRAWER_CABINET_OUTER_DEPTH_X,
                DRAWER_CABINET_WALL_THICKNESS,
                DRAWER_CABINET_HEIGHT,
            ),
            color=(0.44, 0.31, 0.22),
        ),
    )

    drawer_cabinet_right_side = drawer_cabinet_left_side.replace(
        prim_path="/World/envs/env_.*/drawer_cabinet_right_side",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                DRAWER_CABINET_CENTER_X,
                DRAWER_CABINET_CENTER_Y + DRAWER_CABINET_OUTER_WIDTH_Y * 0.5 - DRAWER_CABINET_WALL_THICKNESS * 0.5,
                DRAWER_CABINET_HEIGHT * 0.5,
            )
        ),
    )

    drawer_cabinet_back = RigidObjectCfg(
        prim_path="/World/envs/env_.*/drawer_cabinet_back",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                DRAWER_CABINET_CENTER_X + DRAWER_CABINET_OUTER_DEPTH_X * 0.5 - DRAWER_CABINET_WALL_THICKNESS * 0.5,
                DRAWER_CABINET_CENTER_Y,
                DRAWER_CABINET_HEIGHT * 0.5,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(
                DRAWER_CABINET_WALL_THICKNESS,
                DRAWER_CABINET_OUTER_WIDTH_Y - 2 * DRAWER_CABINET_WALL_THICKNESS,
                DRAWER_CABINET_HEIGHT,
            ),
            color=(0.44, 0.31, 0.22),
        ),
    )

    drawer_cabinet_top = RigidObjectCfg(
        prim_path="/World/envs/env_.*/drawer_cabinet_top",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                DRAWER_CABINET_CENTER_X,
                DRAWER_CABINET_CENTER_Y,
                DRAWER_CABINET_HEIGHT - DRAWER_CABINET_WALL_THICKNESS * 0.5,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(
                DRAWER_CABINET_OUTER_DEPTH_X,
                DRAWER_CABINET_OUTER_WIDTH_Y,
                DRAWER_CABINET_WALL_THICKNESS,
            ),
            color=(0.46, 0.32, 0.22),
        ),
    )

    drawer_handle = RigidObjectCfg(
        prim_path="/World/envs/env_.*/drawer_handle",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                BOX_CENTER_X - BOX_INNER_X * 0.5 - BOX_WALL_THICKNESS - DRAWER_HANDLE_DEPTH * 0.5,
                BOX_CENTER_Y,
                BOX_FRONT_PANEL_CENTER_Z,
            )
        ),
        spawn=make_fixed_panel_cfg(
            size=(DRAWER_HANDLE_DEPTH, DRAWER_HANDLE_WIDTH_Y, DRAWER_HANDLE_HEIGHT_Z),
            color=(0.72, 0.72, 0.74),
        ),
    )

    can = RigidObjectCfg(
        prim_path="/World/envs/env_.*/can",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[CAN_RANDOM_CENTER_X, CAN_RANDOM_CENTER_Y, CAN_RESET_ROOT_Z]),
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
                rest_offset=CAN_COLLISION_REST_OFFSET,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.84, 0.84, 0.86),
                roughness=0.45,
                metallic=0.0,
            ),
        ),
    )

    # robot with inner + down tactile pads
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.down.usda"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.4716,
                "panda_joint2": 0.0114,
                "panda_joint3": 0.5090,
                "panda_joint4": -2.3263,
                "panda_joint5": -0.0645,
                "panda_joint6": 2.3565,
                "panda_joint7": 0.8718,
                "panda_finger_joint.*": 0.02,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # third-person camera
    third_person_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
        height=96,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=30.0,
            focus_distance=200.0,
            horizontal_aperture=40.0,
            clipping_range=(0.05, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.2, -0.35, 1.2),
            rot=(0.68301, 0.18301, -0.18301, -0.68301),
            convention="opengl",
        ),
    )

    tactile_img_res_hw = (96, 128)
    tactile_encoder_type = "cnn"  # supported: "resnet", "cnn", "dino"
    tactile_dino_repo_path = "/home/tinydog/Projects/sparsh"
    tactile_dino_encoder_name = "dino_vitsmall"
    tactile_dino_checkpoint_path = ""
    tactile_dino_model_size = ""
    tactile_dino_encoder_type = "dino"
    tactile_dino_num_register_tokens = 0
    tactile_dino_total_frames = 6  # current + recent 5 frames -> 5 adjacent frame-pairs
    
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
    # expose only tactile RGB to the RL environment
    gsmini_left.data_types = ["tactile_rgb"]
    # drop marker motion simulation
    gsmini_left.marker_motion_sim_cfg = None
    # set tactile image resolution explicitly (no try/except needed as default optical cfg exists)
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
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
    gsmini_right.data_types = ["tactile_rgb"]
    gsmini_right.marker_motion_sim_cfg = None
    gsmini_right.optical_sim_cfg = gsmini_right.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    # additional down-facing tactile sensors (suffix _down)
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
    gsmini_left_down.data_types = ["tactile_rgb"]
    gsmini_left_down.marker_motion_sim_cfg = None
    gsmini_left_down.optical_sim_cfg = gsmini_left_down.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

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
    gsmini_right_down.data_types = ["tactile_rgb"]
    gsmini_right_down.marker_motion_sim_cfg = None
    gsmini_right_down.optical_sim_cfg = gsmini_right_down.optical_sim_cfg.replace(
        tactile_img_res=(tactile_img_res_hw[1], tactile_img_res_hw[0])
    )

    # IK controller - 优化配置
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose", #是求解关节角度还是位姿
        use_relative_mode=True,  # 使用相对模式，期望6维动作
        ik_method="dls",  # 使用阻尼最小二乘法，更稳定
    )

    # environment settings
    max_episode_length = 200  # 直接设置最大步数
    action_space = 5  # 5维动作空间
    # 动作含义：
    # [0] dx: X方向位移增量 (m)
    # [1] dy: Y方向位移增量 (m) 
    # [2] dz: Z方向位移增量 (m)
    # [3] dyaw: 绕Z轴旋转增量 (rad) - 只控制yaw，roll和pitch固定为0
    # [4] gripper: 夹爪开合动作 (m)
    # 注意：前4维给IK控制器（xyz + yaw），第5维直接控制夹爪
    observation_space = {
        "proprio_obs": 18,  # joint positions and velocities (9 joints * 2)
        "third_resnet": 512,
        "tactile_left_depth_resnet": 256,
        "tactile_right_depth_resnet": 256,
        "tactile_left_down_depth_resnet": 256,
        "tactile_right_down_depth_resnet": 256,
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
    state_space = 0
    action_scale = 0.05  # [m] for position, [rad] for orientation - arm动作缩放
    gripper_action_scale = 1.0  # 夹爪动作单独缩放
    gripper_step_size = 0.01  # [m] 每步夹爪开合最大增量
    
    # reward configuration
    reach_sigma = 0.15
    reach_weight = 5.0
    lift_reward_start_height = 0.065
    lift_weight = 15.0
    success_height = 0.135
    success_hold_steps = 5
    success_reward_weight = 100.0
    # 盒底顶面相对盒底中心的 z 偏移（m）
    box_floor_top_offset = BOX_FLOOR_THICKNESS * 0.5

    # task specific parameters 
    can_radius = CAN_RADIUS
    lift_height = 0.06
    ground_height = 0.0  # 地面高度，用于判断机械臂关节是否碰撞地面

    # randomization
    can_rot_range = 0.0
    # Legacy field kept for config compatibility. Box envs use CAN_RANDOM_* bounds below.
    can_reset_xy_half_range = 0.075
    # can reset Z jitter around CAN_RESET_ROOT_Z (meters)
    can_reset_z_jitter = 0.0
    
    # 机械臂随机化配置（打破并行环境的同步起步）
    robot_joint_pos_noise = 0.03  # 关节位置随机化范围 (弧度)
    robot_joint_vel_noise = 0.10  # 关节速度随机化范围
    action_noise_scale = 0.01  # 动作噪声缩放
    reset_jitter_max_steps = 15  # 重置后给 episode 计数随机抖动，错峰超时
    # 每 N 步统计一次“窗口内完成的 episode 的平均奖励”（跨完成的 env 求均值）
    reward_print_interval = 200
    # 最近 N 个已结束 episode 的滑动成功率窗口
    recent_success_rate_window = 100
    # 是否打印每个完成环境的单独均值（默认不打印，仅打印整体均值）
    print_per_env_episode_avg = False

    def _compute_tactile_obs_dim(self) -> int:
        """Compute per-sensor tactile observation dimension exposed to policy."""
        base_dim = 256
        tactile_encoder_type = str(getattr(self, "tactile_encoder_type", "cnn")).strip().lower()
        if tactile_encoder_type == "dino":
            total_frames = max(2, int(getattr(self, "tactile_dino_total_frames", 6)))
            return base_dim * (total_frames - 1)
        return base_dim

    def _sync_tactile_observation_space(self) -> None:
        """Keep tactile observation-space dims consistent with tactile encoder settings."""
        tactile_obs_dim = int(self._compute_tactile_obs_dim())
        self.observation_space = dict(self.observation_space)
        for key in (
            "tactile_left_depth_resnet",
            "tactile_right_depth_resnet",
            "tactile_left_down_depth_resnet",
            "tactile_right_down_depth_resnet",
        ):
            if key in self.observation_space:
                self.observation_space[key] = tactile_obs_dim

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()
        self._sync_tactile_observation_space()


class OccludedGraspingVisionFourTactileBoxEnv(DirectRLEnv):
    """Occluded grasping with third-person vision and four tactile sensors."""

    cfg: OccludedGraspingVisionFourTactileBoxCfg

    def __init__(self, cfg: OccludedGraspingVisionFourTactileBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        
        self.step_count = 0
        self.reward_print_interval = int(getattr(self.cfg, "reward_print_interval", 200))

        # --- For IK actions ---
        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        # Obtain the frame index of the end-effector
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.
        self._jacobi_body_idx = self._body_idx - 1

        # ee offset w.r.t panda hand -> based on the asset
        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # create auxiliary variables for computing applied action, observations and rewards
        self.processed_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Optional ResNet18 backbone for third-person RGB -> 512-d feature
        self._use_resnet18 = _HAS_TORCHVISION
        if self._use_resnet18:
            # use ImageNet-pretrained weights with BC for different torchvision versions
            try:
                weights = torchvision.models.ResNet18_Weights.DEFAULT  # torchvision>=0.13
                backbone = torchvision.models.resnet18(weights=weights)
            except Exception:
                # fallback for older versions
                backbone = torchvision.models.resnet18(pretrained=True)
            # remove final fc; keep feature extractor
            self._resnet18 = torch.nn.Sequential(*list(backbone.children())[:-1]).to(self.device)
            self._resnet18.eval()
            for p in self._resnet18.parameters():
                p.requires_grad_(False)
            # ImageNet normalization buffers (NCHW broadcasting)
            self._imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self._imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Tactile RGB encoder: choose between frozen ResNet18, lightweight CNN, and Sparsh-DINO.
        self._tactile_feature_dim = 256
        self._tactile_encoder_type = str(getattr(self.cfg, "tactile_encoder_type", "resnet")).strip().lower()
        self._tactile_encoder = None
        self._tactile_sparsh_encoder = None
        self._tactile_sensor_keys = ("left", "right", "left_down", "right_down")
        if self._tactile_encoder_type not in {"resnet", "cnn", "dino"}:
            raise ValueError(
                f"Unsupported tactile_encoder_type='{self._tactile_encoder_type}'. Use 'resnet', 'cnn', or 'dino'."
            )

        if self._tactile_encoder_type == "dino":
            resolved = _resolve_tactile_dino_spec(
                encoder_name=str(getattr(self.cfg, "tactile_dino_encoder_name", "dino_vitsmall")),
                checkpoint_path=str(getattr(self.cfg, "tactile_dino_checkpoint_path", "")),
                model_size=str(getattr(self.cfg, "tactile_dino_model_size", "")),
            )
            self._tactile_dino_summary = (
                f"{resolved['name']} | type={str(getattr(self.cfg, 'tactile_dino_encoder_type', 'dino'))} | "
                f"model={resolved['model_size']} | out_dim={self._tactile_feature_dim} | "
                f"ckpt={resolved['checkpoint_path']}"
            )
            self._tactile_sparsh_encoder = SparshFrozenEncoder(
                repo_path=str(getattr(self.cfg, "tactile_dino_repo_path", "/home/tinydog/Projects/sparsh")),
                checkpoint_path=resolved["checkpoint_path"],
                img_size_hw=tuple(getattr(self.cfg, "tactile_img_res_hw", (96, 128))),
                model_size=resolved["model_size"],
                encoder_type=str(getattr(self.cfg, "tactile_dino_encoder_type", "dino")),
                num_register_tokens=int(getattr(self.cfg, "tactile_dino_num_register_tokens", 0)),
            ).to(self.device)
            self._tactile_sparsh_encoder.eval()
            self._tactile_sparsh_encoder.requires_grad_(False)
            target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (96, 128))
            self._tactile_dino_total_frames = max(2, int(getattr(self.cfg, "tactile_dino_total_frames", 6)))
            self._tactile_dino_pair_count = self._tactile_dino_total_frames - 1
            self._tactile_rgb_history = {
                key: torch.zeros(
                    (self.num_envs, self._tactile_dino_total_frames, 3, target_h, target_w),
                    dtype=torch.float32,
                    device=self.device,
                )
                for key in self._tactile_sensor_keys
            }
            self._pending_tactile_history_refresh = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
            print(
                f"[INFO] Tactile encoder: dino ({self._tactile_dino_summary}, "
                f"total_frames={self._tactile_dino_total_frames}, pairs={self._tactile_dino_pair_count})"
            )
        elif self._tactile_encoder_type == "resnet" and self._use_resnet18:
            try:
                tactile_weights = torchvision.models.ResNet18_Weights.DEFAULT
                tactile_backbone = torchvision.models.resnet18(weights=tactile_weights)
            except Exception:
                tactile_backbone = torchvision.models.resnet18(pretrained=True)
            self._tactile_encoder = nn.Sequential(
                *list(tactile_backbone.children())[:-1],
                nn.Flatten(),
                _PairwiseFeatureMean(),
            ).to(self.device)
        else:
            if self._tactile_encoder_type == "resnet" and not self._use_resnet18:
                print("[WARN] tactile_encoder_type='resnet' requested but torchvision is unavailable; using cnn instead.")
                self._tactile_encoder_type = "cnn"
            # Lightweight fallback / alternative tactile encoder with the same 256-d output shape.
            self._tactile_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, self._tactile_feature_dim),
                nn.ReLU(inplace=True),
            ).to(self.device)
        if self._tactile_encoder is not None:
            self._tactile_encoder.eval()
            for p in self._tactile_encoder.parameters():
                p.requires_grad_(False)
            print(
                f"[INFO] Tactile encoder: {self._tactile_encoder_type} "
                f"(feature_dim={self._tactile_feature_dim})"
            )

        # Index of fingers -> first id is left, second id is right finger
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])
        self._left_finger_body_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self._right_finger_body_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        # per-episode return trackers (avg reward per episode, per env)
        self._ep_return = torch.zeros((self.num_envs,), device=self.device)
        self._ep_len = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        # window accumulators for episode-avg rewards (sum & count since last print)
        self._ep_avg_accum_sum = torch.tensor(0.0, device=self.device)
        self._ep_avg_accum_count = torch.tensor(0, dtype=torch.long, device=self.device)
        self._episode_start_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._episode_count = 0
        self._success_count = 0
        self._timeout_count = 0
        self._collision_count = 0
        self._recent_success_window = max(1, int(getattr(self.cfg, "recent_success_rate_window", 100)))
        self._recent_success_buf = torch.zeros((self._recent_success_window,), device=self.device, dtype=torch.float32)
        self._recent_success_count = torch.tensor(0, dtype=torch.long, device=self.device)
        self._recent_success_write_idx = torch.tensor(0, dtype=torch.long, device=self.device)
        self._success_hold_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._success_achieved_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._gate_checkpoints = (40, 80, 120, 160, 200)
        self._ep_gate_running_sum = torch.zeros((self.num_envs,), device=self.device)
        self._ep_gate_running_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._ep_gate_checkpoint_values = {
            step: torch.zeros((self.num_envs,), device=self.device) for step in self._gate_checkpoints
        }
        self._ep_gate_checkpoint_recorded = {
            step: torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device) for step in self._gate_checkpoints
        }

        # add handle for debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Initialize can positions randomly inside the box
        self._initialize_can_positions()

        # （门禁逻辑已移除）

    def _setup_scene(self):
        """Setup the scene."""
        # robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # can
        self._can = RigidObject(self.cfg.can)
        self.scene.rigid_objects["can"] = self._can

        # simplified tray: one bottom board + four surrounding boards
        self._box_floor = RigidObject(self.cfg.box_floor)
        self.scene.rigid_objects["box_floor"] = self._box_floor
        self._box_wall_front = RigidObject(self.cfg.box_wall_front)
        self.scene.rigid_objects["box_wall_front"] = self._box_wall_front
        self._box_wall_back = RigidObject(self.cfg.box_wall_back)
        self.scene.rigid_objects["box_wall_back"] = self._box_wall_back
        self._box_wall_left = RigidObject(self.cfg.box_wall_left)
        self.scene.rigid_objects["box_wall_left"] = self._box_wall_left
        self._box_wall_right = RigidObject(self.cfg.box_wall_right)
        self.scene.rigid_objects["box_wall_right"] = self._box_wall_right

        # clone environments first
        self.scene.clone_environments(copy_from_source=False)

        self.third_person_camera = TiledCamera(self.cfg.third_person_camera)
        self.scene.sensors["third_person_camera"] = self.third_person_camera

        # visuotactile sensors
        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left
        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right
        # optional down-facing sensors (create if prims exist)
        def _try_add_sensor(attr: str, cfg_attr: str):
            try:
                sensor = GelSightSensor(getattr(self.cfg, cfg_attr))
                setattr(self, attr, sensor)
                self.scene.sensors[attr] = sensor
            except Exception as e:
                print(f"[WARN] Skipping optional sensor '{attr}': {e}")

        _try_add_sensor("gsmini_left_down", "gsmini_left_down")
        _try_add_sensor("gsmini_right_down", "gsmini_right_down")

        # add frame transformer for end-effector
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        ee_frame_cfg = FrameTransformerCfg(
            prim_path="/World/envs/env_.*/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="/World/envs/env_.*/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.11841),
                    ),
                ),
            ],
        )

        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        # Spawn AssetBase objects manually
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _initialize_can_positions(self):
        """Initialize can positions inside the drawer bounds used by the demo scene."""
        can_state = self._can.data.default_root_state.clone()

        x_min, x_max, y_min, y_max = self._get_can_reset_xy_bounds()
        can_state[:, 0] = sample_uniform(x_min, x_max, (self.num_envs,), self.device)
        can_state[:, 1] = sample_uniform(y_min, y_max, (self.num_envs,), self.device)
        can_state[:, 2] = CAN_RESET_ROOT_Z

        z_jitter = max(float(getattr(self.cfg, "can_reset_z_jitter", 0.0)), 0.0)
        can_state[:, 2] += sample_uniform(-z_jitter, z_jitter, (self.num_envs,), self.device)
        rand_yaw = sample_uniform(-self.cfg.can_rot_range, self.cfg.can_rot_range, (self.num_envs,), self.device)
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        can_state[:, 3:7] = rand_quat

        env_ids = torch.arange(self.num_envs, device=self.device)
        can_state[:, :3] += self.scene.env_origins
        self._can.write_root_state_to_sim(can_state, env_ids)
        self._can.write_root_velocity_to_sim(torch.zeros((self.num_envs, 6), device=self.device), env_ids)

    def _get_can_reset_xy_bounds(self) -> tuple[float, float, float, float]:
        """Use the same can sampling region as the interactive drawer demo."""
        return CAN_RANDOM_X_MIN, CAN_RANDOM_X_MAX, CAN_RANDOM_Y_MIN, CAN_RANDOM_Y_MAX

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions before physics step."""
        # store actions for reward computation
        self.actions = actions.clone()

        # scale arm and gripper actions separately
        self.processed_actions = self.actions.clone()
        self.processed_actions[:, :4] *= self.cfg.action_scale
        self.processed_actions[:, 4] *= self.cfg.gripper_action_scale
        # sanitize processed actions (avoid NaN/Inf propagating into IK)
        finite_mask = torch.isfinite(self.processed_actions)
        self.processed_actions = torch.where(finite_mask, self.processed_actions, torch.zeros_like(self.processed_actions))
        # conservative clamp per action group
        self.processed_actions[:, :4] = torch.clamp(
            self.processed_actions[:, :4], -self.cfg.action_scale, self.cfg.action_scale
        )
        self.processed_actions[:, 4] = torch.clamp(
            self.processed_actions[:, 4], -self.cfg.gripper_action_scale, self.cfg.gripper_action_scale
        )

        # 添加动作噪声以增加多样性（打破并行环境同步动作）
        action_noise = sample_uniform(
            -self.cfg.action_noise_scale,
            self.cfg.action_noise_scale,
            (self.num_envs, self.cfg.action_space),
            self.device,
        )
        # 不对姿态的 roll/pitch 施加噪声：
        # - 目前动作维度为 [dx, dy, dz, dyaw, gripper]，无 roll/pitch 维度
        # - 为避免通过 yaw 噪声间接影响 roll/pitch（IK 耦合/数值效应），此处也移除对 yaw 的噪声
        noise_mask = torch.ones((self.num_envs, self.cfg.action_space), device=self.device)
        noise_mask[:, 3] = 0.0  # 关闭 dyaw 噪声
        self.processed_actions += action_noise * noise_mask

        # set commands for IK controller (only arm joint actions, not gripper)
        # 将5维动作转换为6维：前4维给IK控制器，第5维给夹爪
        # 构造6维动作：xyz + rpy，其中 roll 和 pitch 设为 0，yaw 来自第 4 维动作
        arm_actions = torch.zeros((self.num_envs, 6), device=self.device)
        arm_actions[:, :3] = self.processed_actions[:, :3]  # xyz
        arm_actions[:, 3] = 0.0  # roll = 0
        arm_actions[:, 4] = 0.0  # pitch = 0  
        arm_actions[:, 5] = self.processed_actions[:, 3]  # yaw (第4维动作)
        
        # 获取当前末端执行器位姿（相对模式需要）
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()

        self._ik_controller.set_command(arm_actions, ee_pos_curr_b, ee_quat_curr_b)

    def _apply_action(self):
        """Apply actions to the robot using IK controller."""
        # obtain quantities from simulation
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]
        
        # compute the delta in joint-space for arm joints（始终计算，避免零范数分支造成行为差异）
        jacobian = self._compute_frame_jacobian()
        # IK controller expects all joint positions, but only updates the first 7 (arm joints)
        arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        # Extract only the arm joint positions (first 7)
        arm_joint_pos_des = arm_joint_pos_des[:, :7]
        # sanitize IK output
        arm_joint_pos_des = torch.where(torch.isfinite(arm_joint_pos_des), arm_joint_pos_des, joint_pos[:, :7])
        # clamp to soft joint limits
        joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, :7, 0]
        joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, :7, 1]
        arm_joint_pos_des = torch.max(arm_joint_pos_des, joint_lower_limits.unsqueeze(0))
        arm_joint_pos_des = torch.min(arm_joint_pos_des, joint_upper_limits.unsqueeze(0))
        
        # handle gripper action (last action)
        # 将夹爪动作解释为“增量”而非“绝对位置”，使策略更易从任意开度收拢/放开
        # action ∈ [-1, 1]，负数趋向闭合（减小开度），正数趋向张开（增大开度）
        gripper_action = self.processed_actions[:, -1]

        # 当前夹爪关节位置（两指应相同）
        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]

        # 增量步长（m）：每步最多 10 mm 变化，避免数值不稳定与反复冲击
        gripper_delta = gripper_action.unsqueeze(-1) * self.cfg.gripper_step_size
        gripper_pos_des = gripper_joint_pos + gripper_delta
        gripper_pos_des = torch.where(torch.isfinite(gripper_pos_des), gripper_pos_des, gripper_joint_pos)
        gripper_pos_des = torch.clamp(gripper_pos_des, 0.0, 0.04)
        
        # combine arm and gripper joint targets
        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_pos_des], dim=1)
        # final safety: replace non-finite with current joint positions
        joint_pos_des = torch.where(torch.isfinite(joint_pos_des), joint_pos_des, joint_pos)

        self._robot.set_joint_position_target(joint_pos_des)

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame."""
        # obtain quantities from simulation
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w

        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

        # account for the offset
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame."""
        # Jacobian at hand frame (world), then express in base frame
        J = self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

        base_quat_w = self._robot.data.root_link_quat_w
        R_b_w = math_utils.matrix_from_quat(math_utils.quat_inv(base_quat_w))  # world -> base
        Jv_b = torch.bmm(R_b_w, J[:, :3, :])
        Jw_b = torch.bmm(R_b_w, J[:, 3:, :])

        # translate Jacobian from hand origin to offset point: Jv' = Jv - [p_b]_x Jw
        # rotate local offset into base frame using current hand orientation
        hand_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        R_w_e = math_utils.matrix_from_quat(hand_quat_w)
        R_b_e = torch.bmm(R_b_w, R_w_e)
        p_local = self._offset_pos  # (N,3)
        p_b = torch.bmm(R_b_e, p_local.unsqueeze(-1)).squeeze(-1)
        Jv_b = Jv_b + torch.bmm(-math_utils.skew_symmetric_matrix(p_b), Jw_b)

        J_b = torch.cat([Jv_b, Jw_b], dim=1)
        return J_b

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        """Get observations from the environment."""
        # Proprioceptive observations
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)

        # 特权信息观察 - can 信息
        can_pos = self._can.data.root_pos_w
        can_quat = self._can.data.root_quat_w
        can_lin_vel = self._can.data.root_lin_vel_w
        can_ang_vel = self._can.data.root_ang_vel_w

        # 特权信息观察 - 夹爪信息
        gripper_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        gripper_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        gripper_lin_vel = self._robot.data.body_link_lin_vel_w[:, self._body_idx]
        gripper_ang_vel = self._robot.data.body_link_ang_vel_w[:, self._body_idx]

        # Target information
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        # normalize quaternions used in computation path
        def _norm_quat(q: torch.Tensor) -> torch.Tensor:
            n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp(min=1e-9)
            return q / n
        hand_quat = _norm_quat(hand_quat)
        offset_rot = _norm_quat(self._offset_rot)
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, offset_rot
        )
        target_pos_relative = can_pos - ee_pos  # 相对位置
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

        # Tactile RGB from GelSight sensors (left/right + left_down/right_down)
        tact_l_raw = self.gsmini_left.data.output.get("tactile_rgb")
        tact_r_raw = self.gsmini_right.data.output.get("tactile_rgb")
        tact_ld_raw = self.gsmini_left_down.data.output.get("tactile_rgb") if hasattr(self, "gsmini_left_down") else None
        tact_rd_raw = self.gsmini_right_down.data.output.get("tactile_rgb") if hasattr(self, "gsmini_right_down") else None
        target_h, target_w = getattr(self.cfg, "tactile_img_res_hw", (240, 320))
        def _prep_rgb_3ch(rgb_tensor):
            # expected NHWC with C=3; convert to float [0,1], resize if needed; return NCHW with C=3
            if rgb_tensor is None:
                return torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
            x = rgb_tensor.to(device=self.device, dtype=torch.float32)
            max_val = x.max()
            if torch.isfinite(max_val) and max_val > 1.5:
                x = x / 255.0
            x = x.clamp(0.0, 1.0)
            xn = x.permute(0, 3, 1, 2).contiguous()
            if xn.shape[2] != target_h or xn.shape[3] != target_w:
                xn = F.interpolate(xn, size=(target_h, target_w), mode="bilinear", align_corners=False)
            return xn

        tact_l = _prep_rgb_3ch(tact_l_raw)
        tact_r = _prep_rgb_3ch(tact_r_raw)
        tact_ld = _prep_rgb_3ch(tact_ld_raw)
        tact_rd = _prep_rgb_3ch(tact_rd_raw)
        tactile_curr = {
            "left": tact_l,
            "right": tact_r,
            "left_down": tact_ld,
            "right_down": tact_rd,
        }

        if self._tactile_encoder_type == "resnet" and hasattr(self, "_imagenet_mean"):
            tact_l = (tact_l - self._imagenet_mean) / self._imagenet_std
            tact_r = (tact_r - self._imagenet_mean) / self._imagenet_std
            tact_ld = (tact_ld - self._imagenet_mean) / self._imagenet_std
            tact_rd = (tact_rd - self._imagenet_mean) / self._imagenet_std

        # （门禁逻辑已移除）

        # 计算腕部与触觉的 ResNet18 特征（使用 FP16 自动混合精度，如可用）
        # 判断是否为 CUDA 设备（self.device 可能是 torch.device 或 str）
        dev = self.device
        dev_type = getattr(dev, "type", None)
        if dev_type is None:
            dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
        use_amp = (dev_type == "cuda")

        if hasattr(self, "_use_resnet18") and self._use_resnet18:
            xt = third_rgb.permute(0, 3, 1, 2).contiguous().to(self.device)
            if hasattr(self, "_imagenet_mean"):
                xt = (xt - self._imagenet_mean) / self._imagenet_std
            with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
                third_feat = self._resnet18(xt).view(self.num_envs, 512)
        else:
            third_feat = torch.zeros((self.num_envs, 512), device=self.device)

        # 触觉特征
        if self._tactile_encoder_type == "dino":
            pending = self._pending_tactile_history_refresh
            if torch.any(pending):
                pending_ids = pending.nonzero(as_tuple=False).squeeze(-1)
                for key in self._tactile_sensor_keys:
                    current = tactile_curr[key][pending_ids]
                    self._tactile_rgb_history[key][pending_ids] = current.unsqueeze(1).expand(
                        -1, self._tactile_dino_total_frames, -1, -1, -1
                    )
                pending[pending_ids] = False

            for key in self._tactile_sensor_keys:
                history = self._tactile_rgb_history[key]
                history[:, :-1] = history[:, 1:].clone()
                history[:, -1] = tactile_curr[key]

            pair_batches = []
            for key in self._tactile_sensor_keys:
                history = self._tactile_rgb_history[key]
                curr_frames = history[:, 1:]   # (N, pair_count, 3, H, W)
                prev_frames = history[:, :-1]  # (N, pair_count, 3, H, W)
                pair_input = torch.cat([curr_frames, prev_frames], dim=2).reshape(
                    self.num_envs * self._tactile_dino_pair_count,
                    6,
                    history.shape[-2],
                    history.shape[-1],
                )
                pair_batches.append(pair_input)
            sparsh_batch = torch.cat(
                pair_batches,
                dim=0,
            )
            with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
                encoded = self._tactile_sparsh_encoder(sparsh_batch).mean(dim=1).float()
            encoded = project_sparsh_features(encoded, self._tactile_feature_dim)
            encoded = encoded.reshape(
                len(self._tactile_sensor_keys),
                self.num_envs,
                self._tactile_dino_pair_count,
                self._tactile_feature_dim,
            ).flatten(start_dim=2)
            fl, fr, fld, frd = encoded[0], encoded[1], encoded[2], encoded[3]
        else:
            # 共享 tactile encoder（resnet/cnn）输出 256 维
            with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
                fl = self._tactile_encoder(tact_l).view(self.num_envs, self._tactile_feature_dim)
                fr = self._tactile_encoder(tact_r).view(self.num_envs, self._tactile_feature_dim)
                fld = self._tactile_encoder(tact_ld).view(self.num_envs, self._tactile_feature_dim)
                frd = self._tactile_encoder(tact_rd).view(self.num_envs, self._tactile_feature_dim)

        obs = {
            "proprio_obs": proprio_obs,
            "third_resnet": third_feat,
            "tactile_left_depth_resnet": fl,
            "tactile_right_depth_resnet": fr,
            "tactile_left_down_depth_resnet": fld,
            "tactile_right_down_depth_resnet": frd,
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
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards based on reaching, lifting, and success."""
        can_pos = self._can.data.root_pos_w
        hand_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        hand_quat = self._robot.data.body_link_quat_w[:, self._body_idx]
        ee_pos, _ = math_utils.combine_frame_transforms(
            hand_pos, hand_quat, self._offset_pos, self._offset_rot
        )

        # reaching_object: r_reach = 1 - tanh(||p_obj - p_ee|| / sigma)
        sigma = max(self.cfg.reach_sigma, 1e-6)
        reach_distance = torch.norm(can_pos - ee_pos, dim=-1)
        reach_reward = 1.0 - torch.tanh(reach_distance / sigma)

        current_height = self._get_object_height_for_success()

        # lifting_object: linear lift progress from configured start height to success height
        lift_span = max(self.cfg.success_height - self.cfg.lift_reward_start_height, 1e-6)
        lift_reward = torch.clamp(
            (current_height - self.cfg.lift_reward_start_height) / lift_span, min=0.0, max=1.0
        )

        # Match the vision-only env: reward every step above the success
        # threshold, while hold/latch is tracked separately for episode stats.
        success_now = current_height > self.cfg.success_height
        self._update_success_tracking(success_now)
        success_reward = success_now.float()

        reach_term = self.cfg.reach_weight * reach_reward
        lift_term = self.cfg.lift_weight * lift_reward
        success_term = self.cfg.success_reward_weight * success_reward
        rewards = reach_term + lift_term + success_term

        # accumulate per-episode returns and lengths
        self._ep_return = self._ep_return + rewards
        self._ep_len = self._ep_len + 1

        can_avg_height = can_pos[:, 2].mean()
        log = self.extras.setdefault("log", {})
        for key in (
            "reward/reach_term",
            "reward/lift_term",
            "reward/success_term",
            "info/can_avg_height",
            "info/episode_avg_reward_window",
            "info/policy_gate_mean",
            "info/hard_gate_bottom_mean",
            "info/hard_gate_inner_mean",
        ):
            log.pop(key, None)
        log["reward/reach"] = reach_reward.mean().detach()
        log["reward/lift"] = lift_reward.mean().detach()
        log["reward/success"] = success_reward.mean().detach()
        log["reward/total"] = rewards.mean().detach()
        log["info/reach_distance"] = reach_distance.mean().detach()
        log["info/can_height"] = can_avg_height.detach()
        gate_mean = get_latest_gate_mean()
        if gate_mean is not None:
            log["info/policy_gate_mean"] = torch.tensor(gate_mean, device=self.device)
        hard_gate_bottom = get_latest_m_probe()
        if hard_gate_bottom is not None:
            log["info/hard_gate_bottom_mean"] = hard_gate_bottom.to(device=self.device, dtype=torch.float32).mean()
        hard_gate_inner = get_latest_m_grasp()
        if hard_gate_inner is not None:
            log["info/hard_gate_inner_mean"] = hard_gate_inner.to(device=self.device, dtype=torch.float32).mean()
        gate_values = get_latest_gate_values()
        if gate_values is not None:
            gate_values = gate_values.to(device=self.device, dtype=torch.float32)
            self._ep_gate_running_sum = self._ep_gate_running_sum + gate_values
            self._ep_gate_running_count = self._ep_gate_running_count + 1
            valid_gate = self._ep_gate_running_count > 0
            for checkpoint in self._gate_checkpoints:
                checkpoint_mask = (
                    valid_gate
                    & (self._ep_len >= checkpoint)
                    & ~self._ep_gate_checkpoint_recorded[checkpoint]
                )
                if torch.any(checkpoint_mask):
                    running_mean = (
                        self._ep_gate_running_sum[checkpoint_mask]
                        / self._ep_gate_running_count[checkpoint_mask].to(torch.float32)
                    )
                    self._ep_gate_checkpoint_values[checkpoint][checkpoint_mask] = running_mean
                    self._ep_gate_checkpoint_recorded[checkpoint][checkpoint_mask] = True

        # 每 reward_print_interval 步打印一次当前步的奖励各分量（跨 env 取均值）
        if self.cfg.reward_print_interval > 0 and ((self.step_count + 1) % self.cfg.reward_print_interval == 0):
            # Print sliding-window success rate for more responsive training feedback.
            success_rate = self._get_recent_success_rate()
            w_probe = get_latest_w_probe()
            w_grasp = get_latest_w_grasp()
            alpha_gate = get_latest_alpha()
            hard_gate_bottom = get_latest_m_probe()
            hard_gate_inner = get_latest_m_grasp()
            try:
                gate_text = f", gate_mean={gate_mean:.4f}" if gate_mean is not None else ""
                aux_gate_text = ""
                if w_probe is not None and w_grasp is not None:
                    aux_gate_text = (
                        f", w_probe={float(w_probe.mean().item()):.4f}"
                        f", w_grasp={float(w_grasp.mean().item()):.4f}"
                    )
                    if alpha_gate is not None:
                        aux_gate_text += f", alpha={float(alpha_gate.mean().item()):.4f}"
                hard_gate_text = ""
                if hard_gate_bottom is not None and hard_gate_inner is not None:
                    hard_gate_text = (
                        f", hard_gate_bottom={float(hard_gate_bottom.mean().item()):.4f}"
                        f", hard_gate_inner={float(hard_gate_inner.mean().item()):.4f}"
                    )
                tactile_ratio_text = ""
                tactile_ratio_by_sensor = getattr(self, "_latest_tactile_over_thresh_ratio_by_sensor", None)
                if tactile_ratio_by_sensor is not None:
                    tactile_ratio_by_sensor = torch.as_tensor(
                        tactile_ratio_by_sensor, device=self.device, dtype=torch.float32
                    )
                    if tactile_ratio_by_sensor.ndim == 1 and tactile_ratio_by_sensor.numel() % 4 == 0:
                        tactile_ratio_by_sensor = tactile_ratio_by_sensor.view(-1, 4)
                    if tactile_ratio_by_sensor.ndim >= 2 and tactile_ratio_by_sensor.shape[-1] >= 4:
                        means = tactile_ratio_by_sensor[:, :4].mean(dim=0)
                        tactile_ratio_text = (
                            f", tactile_ratio_left={float(means[0].item()):.4f}"
                            f", tactile_ratio_right={float(means[1].item()):.4f}"
                            f", tactile_ratio_left_down={float(means[2].item()):.4f}"
                            f", tactile_ratio_right_down={float(means[3].item()):.4f}"
                        )
                if not tactile_ratio_text:
                    tactile_over_thresh_ratio = getattr(self, "_latest_tactile_over_thresh_ratio_mean", None)
                    if tactile_over_thresh_ratio is not None:
                        tactile_over_thresh_ratio = torch.as_tensor(
                            tactile_over_thresh_ratio, device=self.device, dtype=torch.float32
                        ).reshape(-1)
                        if tactile_over_thresh_ratio.numel() > 0:
                            tactile_ratio_text = (
                                f", tactile_over_thresh_ratio_mean={float(tactile_over_thresh_ratio.mean().item()):.4f}"
                            )
                print(
                    f"[Step Rewards] step {self.step_count + 1}: "
                    f"reach={reach_reward.mean().item():.3f} (w={self.cfg.reach_weight}), "
                    f"lift={lift_reward.mean().item():.3f} (w={self.cfg.lift_weight}), "
                    f"success_reward={success_reward.mean().item():.3f} (w={self.cfg.success_reward_weight}), "
                    f"success_rate={success_rate:.3f}, "
                    f"can_avg_height={can_avg_height.item():.4f}, "
                    f"total={rewards.mean().item():.3f}"
                    f"{gate_text}"
                    f"{aux_gate_text}"
                    f"{hard_gate_text}"
                    f"{tactile_ratio_text}"
                )
            except Exception:
                pass

        self.step_count += 1

        return rewards

    def _get_object_height_for_success(self) -> torch.Tensor:
        """Return object height used by success condition in _get_dones."""
        return self._can.data.root_pos_w[:, 2]

    def _update_success_tracking(self, success_now: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update success hold buffers and latch success once achieved."""
        hold_steps = max(1, int(getattr(self.cfg, "success_hold_steps", 1)))
        if hold_steps > 1:
            self._success_hold_buf = torch.where(
                success_now,
                torch.clamp(self._success_hold_buf + 1, max=hold_steps),
                torch.zeros_like(self._success_hold_buf),
            )
            success_hold = self._success_hold_buf >= hold_steps
        else:
            success_hold = success_now
        success_first = success_hold & ~self._success_achieved_buf
        self._success_achieved_buf |= success_first
        return success_hold, success_first

    def _get_logged_success_rate(self) -> float:
        """Return the cumulative success rate used by terminal logs."""
        total_episodes = max(self._episode_count, 1)
        return float(self._success_count) / total_episodes

    def _get_recent_success_rate(self) -> float:
        """Return the success rate over the most recent completed episodes."""
        count = int(self._recent_success_count.item())
        if count <= 0:
            return 0.0
        return float(self._recent_success_buf[:count].mean().item())

    def _update_recent_success_rate(self, successes: torch.Tensor) -> None:
        """Append completed-episode success flags into the sliding window."""
        if successes.numel() == 0:
            return
        successes = successes.to(device=self.device, dtype=torch.float32).view(-1)
        window = self._recent_success_window
        write_idx = int(self._recent_success_write_idx.item())
        count = int(self._recent_success_count.item())
        for value in successes:
            self._recent_success_buf[write_idx] = value
            write_idx = (write_idx + 1) % window
            count = min(count + 1, window)
        self._recent_success_write_idx.fill_(write_idx)
        self._recent_success_count.fill_(count)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate done flags."""
        # Episode timeout - 基于步数检查
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out = time_out.bool()

        # Success tracking is latched in the reward path and does not terminate the episode.
        success = self._success_achieved_buf

        # 失败条件 - 机械臂关节碰撞地面
        # 获取机械臂所有关节的世界坐标位置
        joint_positions = self._robot.data.body_link_pos_w  # [num_envs, num_links, 3]
        
        # 检查是否有任何关节的Z坐标低于地面高度
        joint_z_positions = joint_positions[:, :, 2]  # [num_envs, num_links]
        collision_with_ground = torch.any(joint_z_positions < self.cfg.ground_height, dim=1)  # [num_envs]
        collision_with_ground = collision_with_ground.bool()

        # Combine done conditions
        dones = time_out | collision_with_ground

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        log = self.extras.setdefault("log", {})
        for checkpoint in self._gate_checkpoints:
            log.pop(f"info/episode_gate_mean_step_{checkpoint}", None)
        if done_ids.numel() > 0:
            num_done = int(done_ids.numel())
            num_success = int(success[done_ids].sum().item())
            num_timeout = int(time_out[done_ids].sum().item())
            num_collision = int(collision_with_ground[done_ids].sum().item())
            self._episode_count += num_done
            self._success_count += num_success
            self._update_recent_success_rate(success[done_ids])
            self._timeout_count += num_timeout
            self._collision_count += num_collision
            ep_len = torch.clamp(self._ep_len[done_ids].to(torch.float32), min=1.0)
            ep_avg = self._ep_return[done_ids] / ep_len
            self._ep_avg_accum_sum = self._ep_avg_accum_sum + ep_avg.sum()
            self._ep_avg_accum_count = self._ep_avg_accum_count + ep_avg.numel()
            if getattr(self.cfg, "print_per_env_episode_avg", False):
                ids = done_ids.detach().cpu().numpy().tolist()
                vals = ep_avg.detach().cpu().numpy().tolist()
                details = " ".join([f"env{eid}={v:.3f}" for eid, v in zip(ids, vals)])
                print(f"[Episode Avg Reward per-env] {details}")
            for checkpoint in self._gate_checkpoints:
                recorded = self._ep_gate_checkpoint_recorded[checkpoint][done_ids]
                if torch.any(recorded):
                    checkpoint_mean = self._ep_gate_checkpoint_values[checkpoint][done_ids][recorded].mean().detach()
                    log[f"info/episode_gate_mean_step_{checkpoint}"] = checkpoint_mean
            self._ep_return[done_ids] = 0.0
            self._ep_len[done_ids] = 0
            self._ep_gate_running_sum[done_ids] = 0.0
            self._ep_gate_running_count[done_ids] = 0
            for checkpoint in self._gate_checkpoints:
                self._ep_gate_checkpoint_values[checkpoint][done_ids] = 0.0
                self._ep_gate_checkpoint_recorded[checkpoint][done_ids] = False

        log.pop("info/episode_avg_reward_window", None)
        log["success_rate"] = torch.tensor(self._get_logged_success_rate(), device=self.device)
        log["recent_success_rate"] = torch.tensor(self._get_recent_success_rate(), device=self.device)

        if self.cfg.reward_print_interval > 0 and (self.step_count % self.cfg.reward_print_interval == 0):
            cnt = int(self._ep_avg_accum_count.item())
            if cnt > 0:
                avg = float(self._ep_avg_accum_sum.item()) / cnt
                log["info/episode_avg_reward_window"] = torch.tensor(avg, device=self.device)
                print(f"[Episode Avg Reward] avg={avg:.3f}")
                self._ep_avg_accum_sum = torch.tensor(0.0, device=self.device)
                self._ep_avg_accum_count = torch.tensor(0, dtype=torch.long, device=self.device)
        
        # 简化的Episode结束信息
        # if dones.any():
        #     success_envs = success.nonzero(as_tuple=False).squeeze(-1).cpu().numpy() if success.any() else []
        #     collision_envs = collision_with_ground.nonzero(as_tuple=False).squeeze(-1).cpu().numpy() if collision_with_ground.any() else []
        #     timeout_envs = time_out.nonzero(as_tuple=False).squeeze(-1).cpu().numpy() if time_out.any() else []
            
        #     print(f"\n🏁 Episode结束 (步骤 {self.step_count}): 成功{len(success_envs)} | 碰撞{len(collision_envs)} | 超时{len(timeout_envs)}")

        return dones, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments at the given indices - 学习ball_rolling_tactile_rgb.py的重置机制."""
        # 调用父类重置方法
        super()._reset_idx(env_ids)
        if self._tactile_encoder_type == "dino" and hasattr(self, "_pending_tactile_history_refresh"):
            self._pending_tactile_history_refresh[env_ids] = True
            for key in self._tactile_sensor_keys:
                self._tactile_rgb_history[key][env_ids] = 0.0
        # also reset episode trackers on manual resets
        self._ep_return[env_ids] = 0.0
        self._ep_len[env_ids] = 0
        self._success_hold_buf[env_ids] = 0
        self._success_achieved_buf[env_ids] = False
        self._ep_gate_running_sum[env_ids] = 0.0
        self._ep_gate_running_count[env_ids] = 0
        for checkpoint in self._gate_checkpoints:
            self._ep_gate_checkpoint_values[checkpoint][env_ids] = 0.0
            self._ep_gate_checkpoint_recorded[checkpoint][env_ids] = False
        
        # 重置机械臂到随机化初始位置
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        
        # 添加关节位置随机化 - 只对arm关节（前7个）
        joint_noise = sample_uniform(
            -self.cfg.robot_joint_pos_noise, 
            self.cfg.robot_joint_pos_noise, 
            (len(env_ids), 7), 
            self.device
        )
        joint_pos[:, :7] += joint_noise
        
        # 限制在关节范围内
        joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, :7, 0]
        joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, :7, 1]
        joint_pos[:, :7] = torch.clamp(
            joint_pos[:, :7],
            joint_lower_limits.unsqueeze(0),
            joint_upper_limits.unsqueeze(0)
        )
        
        # 添加关节速度随机化
        joint_vel = sample_uniform(
            -self.cfg.robot_joint_vel_noise, 
            self.cfg.robot_joint_vel_noise, 
            (len(env_ids), joint_pos.shape[1]), 
            self.device
        )
        
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 重置物体位置 - 随机放置在开口盒内
        can_state = self._can.data.default_root_state[env_ids].clone()

        x_min, x_max, y_min, y_max = self._get_can_reset_xy_bounds()
        can_state[:, 0] = sample_uniform(x_min, x_max, (len(env_ids),), self.device)
        can_state[:, 1] = sample_uniform(y_min, y_max, (len(env_ids),), self.device)
        can_state[:, 2] = CAN_RESET_ROOT_Z

        z_jitter = max(float(getattr(self.cfg, "can_reset_z_jitter", 0.0)), 0.0)
        can_state[:, 2] += sample_uniform(-z_jitter, z_jitter, (len(env_ids),), self.device)
        rand_yaw = sample_uniform(-self.cfg.can_rot_range, self.cfg.can_rot_range, (len(env_ids),), self.device)
        rand_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(rand_yaw), torch.zeros_like(rand_yaw), rand_yaw
        )
        can_state[:, 3:7] = rand_quat
        
        # 添加环境原点偏移（包含z轴）
        can_state[:, :3] += self.scene.env_origins[env_ids]

        # 写入仿真
        self._can.write_root_state_to_sim(can_state, env_ids=env_ids)
        self._can.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)

        # 重置动作缓冲区，并为超时步数加入抖动，错峰重置
        self.actions[env_ids] = 0
        self.processed_actions[env_ids] = 0
        jitter = torch.randint(0, self.cfg.reset_jitter_max_steps, (len(env_ids),), device=self.device)
        self.episode_length_buf[env_ids] = jitter
        self._episode_start_length_buf[env_ids] = jitter
        self._success_hold_buf[env_ids] = 0
        self._success_achieved_buf[env_ids] = False

    def _compute_collision_penalty(
        self, left_finger_pos: torch.Tensor, right_finger_pos: torch.Tensor, can_pos: torch.Tensor
    ) -> torch.Tensor:
        """计算与桌面/地面的穿透性碰撞（工具函数，当前奖励未使用）。"""
        floor_top_z = self._box_floor.data.root_pos_w[:, 2] + self.cfg.box_floor_top_offset
        ground_height = self.cfg.ground_height

        left_below_floor = left_finger_pos[:, 2] < (floor_top_z - 0.001)
        right_below_floor = right_finger_pos[:, 2] < (floor_top_z - 0.001)
        object_below_ground = can_pos[:, 2] < ground_height

        collision_detected = left_below_floor | right_below_floor | object_below_ground
        return collision_detected.float()


@configclass
class OccludedGraspingVisionFourTactileWristBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Configuration for occluded grasping with wrist-camera vision and four tactile sensors."""

    third_person_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera",
        height=96,
        width=128,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.05, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.12, 0.0, -0.12),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="ros",
        ),
    )


class OccludedGraspingVisionFourTactileWristBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Occluded grasping environment using wrist-camera vision and four tactile sensors."""

    cfg: OccludedGraspingVisionFourTactileWristBoxCfg


class OccludedGraspingVTAlphaBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Alias cfg for the alpha-gated VT task."""


class OccludedGraspingVTAlphaBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Alias env for the alpha-gated VT task."""


class OccludedGraspingVTConvexBoxCfg(OccludedGraspingVisionFourTactileBoxCfg):
    """Alias cfg for the convex-combination VT task."""


class OccludedGraspingVTConvexBoxEnv(OccludedGraspingVisionFourTactileBoxEnv):
    """Alias env for the convex-combination VT task."""
