from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Franka cylinder grasping demo with a partially opened drawer and partial third-person view"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sys", type=bool, default=True, help="Whether to track system utilization.")
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="Whether to render tactile images in the GUI",
)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import traceback
from contextlib import suppress

import carb
import carb.input as carb_input
import omni.appwindow
import omni.ui
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim

with suppress(ImportError):
    # isaacsim.gui is not available when running in headless mode.
    import isaacsim.gui.components.ui_utils as ui_utils

import pynvml

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from tacex import GelSightSensor

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg

# Equivalent cylinder size matched to the previous YCB can mesh (after scale=0.5).
CAN_RADIUS = 0.025
CAN_HEIGHT = 0.07

# Drawer controls:
# - DRAWER_SCALE: overall proportional scale for the whole drawer assembly.
# - DRAWER_DEPTH_SCALE / WIDTH_SCALE / HEIGHT_SCALE: fine-tune one axis on top of DRAWER_SCALE.
# - DRAWER_OPEN_RATIO: how far the drawer is pulled out relative to drawer depth.
# - DRAWER_FRONT_FACE_X / DRAWER_CENTER_Y: world placement of the drawer front.
DRAWER_SCALE = 1.8
DRAWER_DEPTH_SCALE = 1.2
DRAWER_WIDTH_SCALE = 1.2
DRAWER_HEIGHT_SCALE = 0.8
DRAWER_OPEN_RATIO = 1.0
DRAWER_FRONT_FACE_X = 0.435
DRAWER_CENTER_Y = 0.00

# Base dimensions before scaling.
BASE_DRAWER_INNER_X = 0.22
BASE_DRAWER_INNER_Y = 0.26
BASE_DRAWER_WALL_THICKNESS = 0.015
BASE_DRAWER_FLOOR_THICKNESS = 0.02
BASE_DRAWER_WALL_HEIGHT = 0.08
BASE_DRAWER_FRONT_PANEL_HEIGHT = 0.1
# Can centroid height above drawer floor (set z to centroid, not bottom).
CAN_CENTROID_OFFSET_Z_DEFAULT = 0.5 * CAN_HEIGHT
CAN_PRINT_EVERY_STEPS = 1

BASE_DRAWER_CABINET_WALL_THICKNESS = 0.015
BASE_DRAWER_CABINET_SIDE_CLEARANCE_Y = 0.004
BASE_DRAWER_CABINET_TOP_CLEARANCE_Z = 0.006
BASE_DRAWER_CABINET_REAR_CLEARANCE_X = 0.020
BASE_DRAWER_FRONT_PANEL_SIDE_GAP = 0.004

BASE_DRAWER_HANDLE_DEPTH = 0.022
BASE_DRAWER_HANDLE_WIDTH_Y = 0.07
BASE_DRAWER_HANDLE_HEIGHT_Z = 0.01

BASE_DRAWER_OBJECT_RESET_MARGIN_X = 0.045
BASE_DRAWER_OBJECT_RESET_MARGIN_Y = 0.05

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

# Rectangle defined by two opposite corners from the observed can positions.
CAN_RECT_CORNER_A = (0.5158, -0.1838)
CAN_RECT_CORNER_B = (0.6515, 0.1772)
CAN_RANGE_X_MIN = min(CAN_RECT_CORNER_A[0], CAN_RECT_CORNER_B[0])
CAN_RANGE_X_MAX = max(CAN_RECT_CORNER_A[0], CAN_RECT_CORNER_B[0])
CAN_RANGE_Y_MIN = min(CAN_RECT_CORNER_A[1], CAN_RECT_CORNER_B[1])
CAN_RANGE_Y_MAX = max(CAN_RECT_CORNER_A[1], CAN_RECT_CORNER_B[1])

def make_fixed_panel_cfg(
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> sim_utils.CuboidCfg:
    """Create a fixed wooden panel used for the cabinet and drawer geometry."""
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
    )



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


@configclass
class CylinderGraspingDemoCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.45, 0.85, 0.52)
    viewer.lookat = (BOX_CENTER_X, BOX_CENTER_Y, 0.08)

    debug_vis = True
    ui_window_class_type = CustomEnvWindow

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_ccd=True,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=5.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
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

    cylinder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[BOX_CENTER_X, BOX_CENTER_Y, BOX_OBJECT_Z]),
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
        ),
    )

    # robot configuration
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": -0.4510,
                "panda_joint2": 0.1890,
                "panda_joint3": 0.4750,
                "panda_joint4": -2.3660,
                "panda_joint5": -0.2250,
                "panda_joint6": 2.5450,
                "panda_joint7": 0.9910,
                "panda_finger_joint.*": 0.02,
            },
        ),
        spawn=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.replace(
            rigid_props=FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.rigid_props.replace(
                disable_gravity=True,
            ),
        ),
    )

    # wrist camera
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera",
        update_period=0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=40.0,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.1, 0, -0.12),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="ros",
        ),
    )

    # third-person camera (fixed to env root)
    third_person_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
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
        # Side-top view focused on the partially opened drawer and the object inside.
        offset=CameraCfg.OffsetCfg(
            pos=(-0.2, -0.35, 1.2),
            rot=(0.68301, 0.18301, -0.18301, -0.68301),
            convention="opengl",
        ),
    )



    gsmini_left = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(32, 24),
            data_types=["depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=True,  # for rendering sensor output in the gui
        # update Taxim cfg
        marker_motion_sim_cfg=None,
        data_types=["tactile_rgb"],  # marker_motion
    )
    # settings for optical sim
    gsmini_left.optical_sim_cfg = gsmini_left.optical_sim_cfg.replace(
        with_shadow=False,
        device="cuda",
        tactile_img_res=(320, 240),
    )
    gsmini_right = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
    )

    # add two outer tactile sensors (mirror inner settings)
    gsmini_left_outer = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left_outer",
    )
    gsmini_right_outer = gsmini_left.replace(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right_outer",
    )



    # IK controller - 使用相对模式，仿照cylinder_grasping_privileged.py
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,  # 使用相对模式
        ik_method="dls",  # 可选: 'pinv', 'svd', 'trans', 'dls'
        # 添加更多参数以提高精度
    )
    episode_length_s = 0
    action_space = 5  # 5维动作空间：xyz + rz + gripper
    observation_space = 0
    state_space = 0


class CylinderGraspingDemo(DirectRLEnv):
    """Cylinder grasping demo environment."""

    cfg: CylinderGraspingDemoCfg

    def __init__(self, cfg: CylinderGraspingDemoCfg, render_mode: str | None = None, **kwargs):
        self._can_centroid_offset_z = CAN_CENTROID_OFFSET_Z_DEFAULT
        self._can_pose_view = None
        super().__init__(cfg, render_mode, **kwargs)


       # IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )

        # Get end-effector body index
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # Index of fingers -> first id is left, second id is right finger
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])

        # For a fixed base robot, the frame index is one less than the body index.
        # This is because the root body is not included in the returned Jacobians.

        self._jacobi_body_idx = self._body_idx - 1

        # ee offset w.r.t panda hand -> based on the asset
        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # ---

        # create buffer to store actions (= ik_commands)
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        
        # 动作空间配置 - 仿照cylinder_grasping_privileged.py
        self.action_scale = 0.1  # 动作缩放因子
        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        
        # 当前动作状态
        self.current_actions = torch.zeros((self.num_envs, 5), device=self.device)  # [dx, dy, dz, droll, gripper]

        self.step_count = 0
        
        # 关节角度控制参数 - 可在这里修改关节初始角度
        self.joint_angles = {

            'joint1': -0.4716,
            'joint2': 0.0114,
            'joint3': 0.5090,
            'joint4': -2.3263,
            'joint5': -0.0645,
            'joint6': 2.3565,
            'joint7': 0.8718,
            
            'finger_left': 0.02,
            'finger_right': 0.02,
        }

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _get_can_centroid_z(self) -> float:
        """Return can z-position using centroid height above the drawer floor."""
        return BOX_FLOOR_THICKNESS + float(self._can_centroid_offset_z)


    def _setup_scene(self):

        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._cylinder = RigidObject(self.cfg.cylinder)
        self.scene.rigid_objects["cylinder"] = self._cylinder

        # partially opened drawer (floor + walls) with a simple cabinet shell
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

        self._drawer_cabinet_left_side = RigidObject(self.cfg.drawer_cabinet_left_side)
        self.scene.rigid_objects["drawer_cabinet_left_side"] = self._drawer_cabinet_left_side

        self._drawer_cabinet_right_side = RigidObject(self.cfg.drawer_cabinet_right_side)
        self.scene.rigid_objects["drawer_cabinet_right_side"] = self._drawer_cabinet_right_side

        self._drawer_cabinet_back = RigidObject(self.cfg.drawer_cabinet_back)
        self.scene.rigid_objects["drawer_cabinet_back"] = self._drawer_cabinet_back

        self._drawer_cabinet_top = RigidObject(self.cfg.drawer_cabinet_top)
        self.scene.rigid_objects["drawer_cabinet_top"] = self._drawer_cabinet_top

        self._drawer_handle = RigidObject(self.cfg.drawer_handle)
        self.scene.rigid_objects["drawer_handle"] = self._drawer_handle

        self.wrist_camera = Camera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera 
        # third-person camera
        self.third_person_camera = Camera(self.cfg.third_person_camera)
        self.scene.sensors["third_person_camera"] = self.third_person_camera

        self.scene.clone_environments(copy_from_source=False)
        # XForm view used to compare get_world_poses() with rigid-body root_state_w.
        self._can_pose_view = XFormPrim(prim_paths_expr="/World/envs/env_.*/cylinder", name="can_pose_view", usd=True)

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

        # sensors
        self._ee_frame = FrameTransformer(ee_frame_cfg)
        self.scene.sensors["ee_frame"] = self._ee_frame

        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left

        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right

        # outer sensors (only if prims exist); prevent crash if USD lacks these prims
        try:
            self.gsmini_left_outer = GelSightSensor(self.cfg.gsmini_left_outer)
            self.scene.sensors["gsmini_left_outer"] = self.gsmini_left_outer
        except Exception as e:
            print(f"[WARN] Skipping gsmini_left_outer: {e}")
            self.gsmini_left_outer = None
        try:
            self.gsmini_right_outer = GelSightSensor(self.cfg.gsmini_right_outer)
            self.scene.sensors["gsmini_right_outer"] = self.gsmini_right_outer
        except Exception as e:
            print(f"[WARN] Skipping gsmini_right_outer: {e}")
            self.gsmini_right_outer = None

        # Spawn AssetBase objects manually
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )


        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions before physics step."""
        # 存储当前动作
        self.current_actions = actions.clone()
        
        # 仿照cylinder_grasping_privileged.py的动作处理
        # 将5维动作扩展为6维给IK控制器
        # 获取当前末端执行器位姿
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        
        # 构造6维动作：dx, dy, dz, droll, dpitch, dyaw
        # 我们的5维动作：[dx, dy, dz, drz, gripper] (第4维是绕Z轴旋转)
        # 扩展为6维：[dx, dy, dz, droll=0, dpitch=0, dyaw=drz]
        self.processed_actions[:, :3] = self.current_actions[:, :3] * self.action_scale  # dx, dy, dz
        self.processed_actions[:, 3] = 0.0  # droll = 0 (绕X轴旋转)
        self.processed_actions[:, 4] = 0.0  # dpitch = 0 (绕Y轴旋转)
        self.processed_actions[:, 5] = self.current_actions[:, 3] * self.action_scale *10   # dyaw (绕Z轴旋转)
        
        # 设置IK命令（相对模式）
        self._ik_controller.set_command(self.processed_actions, ee_pos_curr_b, ee_quat_curr_b)
        
        # 应用关节控制
        self._apply_joint_control()

    def _apply_joint_control(self):
        """Apply joint control to the robot using IK controller."""
        # 仿照cylinder_grasping_privileged.py设计：使用IK控制器计算关节位置
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]
        
        # compute the delta in joint-space for arm joints only
        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            # IK controller returns all joint positions, but we only need the first 7 (arm joints)
            arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
            # Extract only the arm joint positions (first 7)
            arm_joint_pos_des = arm_joint_pos_des[:, :7]
        else:
            arm_joint_pos_des = joint_pos[:, :7].clone()
        
        # 设置夹爪动作（第5维是夹爪动作）
        gripper_action = self.current_actions[:, -1]  # 第5维是夹爪动作
        
        # 获取当前夹爪关节位置
        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]
        
        # 转换夹爪动作为位置（假设动作在[-1, 1]范围内）
        # -1 = 完全打开, 1 = 完全闭合
        # Franka夹爪有2个手指关节，都应该一起移动
        gripper_pos_des = gripper_action.unsqueeze(-1) * 0.5  # 缩放到夹爪范围
        gripper_pos_des = gripper_pos_des.expand(-1, len(self._finger_joint_ids))  # 扩展到两个手指关节
        
        # 合并机械臂和夹爪关节目标
        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_pos_des], dim=1)
        
        # 应用关节目标位置
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
        # 获取基座雅可比矩阵
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]
        
        # 转换到基座坐标系
        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        
        # 考虑末端执行器偏移量
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        
        return jacobian

    def set_joint_angle(self, joint_name: str, angle: float):
        """设置指定关节的角度
        
        Args:
            joint_name: 关节名称 ('joint1' 到 'joint7', 'finger_left', 'finger_right')
            angle: 角度值 (弧度)
        """
        if joint_name in self.joint_angles:
            self.joint_angles[joint_name] = angle
        else:
            print(f"错误: 未知的关节名称 '{joint_name}'")
            print(f"可用的关节: {list(self.joint_angles.keys())}")

    def set_all_joint_angles(self, angles: dict):
        """批量设置所有关节角度
        
        Args:
            angles: 关节角度字典，例如 {'joint1': 0.5, 'joint2': -1.0, ...}
        """
        for joint_name, angle in angles.items():
            if joint_name in self.joint_angles:
                self.joint_angles[joint_name] = angle
            else:
                print(f"警告: 未知的关节名称 '{joint_name}'")

    def get_current_joint_angles(self):
        """获取当前关节角度"""    # print("平移后关节角度（弧度）：")
    # for name, angle in zip(joint_names, moved_joint_angles.numpy()):
    #     print(f"  {name}: {angle:.4f} rad ({angle * 180.0 / np.pi:.2f}°)")
        return self._robot.data.joint_pos[0].cpu().numpy()
    
    def _set_initial_joint_angles(self):
        """设置机械臂初始关节角度"""
        try:
            # 检查机器人是否已正确初始化
            if not hasattr(self._robot, 'joint_names'):
                print("  警告: 机器人对象尚未完全初始化，跳过关节角度设置")
                return
                
            # 获取所有关节名称
            all_joint_names = self._robot.joint_names
            
            # 关节名称映射：从我们的命名到实际关节名称
            joint_name_mapping = {
                'joint1': 'panda_joint1',
                'joint2': 'panda_joint2', 
                'joint3': 'panda_joint3',
                'joint4': 'panda_joint4',
                'joint5': 'panda_joint5',
                'joint6': 'panda_joint6',
                'joint7': 'panda_joint7',
                'finger_left': 'panda_finger_joint1',
                'finger_right': 'panda_finger_joint2'
            }
            
            # 设置每个关节的角度
            for joint_name, angle in self.joint_angles.items():
                actual_joint_name = joint_name_mapping.get(joint_name, joint_name)
                
                if actual_joint_name in all_joint_names:
                    # 直接设置关节目标位置
                    joint_idx = all_joint_names.index(actual_joint_name)
                    self._robot.data.joint_pos_target[:, joint_idx] = angle
                else:
                    print(f"  警告: 关节 {joint_name} ({actual_joint_name}) 不存在于机器人配置中")
            
            # 应用关节目标位置到仿真
            self._robot.write_joint_state_to_sim(
                self._robot.data.joint_pos_target,
                self._robot.data.joint_vel
            )
        except Exception as e:
            print(f"  错误: 设置关节角度时发生异常: {e}")
            print("  将使用默认关节角度继续运行")
    
    def move_end_effector(self, delta_xyz: tuple[float, float, float], settle_steps: int = 120) -> torch.Tensor:
        """Translate the end-effector by ``delta_xyz`` (meters) using the IK controller and return joint angles."""
        delta = torch.zeros((self.num_envs, 6), device=self.device)
        delta[:, 0] = delta_xyz[0]
        delta[:, 1] = delta_xyz[1]
        delta[:, 2] = delta_xyz[2]

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        self._ik_controller.set_command(delta, ee_pos_curr_b, ee_quat_curr_b)

        joint_pos = self._robot.data.joint_pos[:, :]
        jacobian = self._compute_frame_jacobian()
        arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        arm_joint_pos_des = arm_joint_pos_des[:, :7]

        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]
        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_joint_pos], dim=1)
        self._robot.set_joint_position_target(joint_pos_des)

        for _ in range(settle_steps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        self.sim.render()
        return self._robot.data.joint_pos[0, :7].detach().cpu()

    def rotate_end_effector(self, delta_rpy: tuple[float, float, float], settle_steps: int = 120) -> torch.Tensor:
        """Rotate the end-effector by ``delta_rpy`` = (droll, dpitch, dyaw) in radians using relative IK.

        Returns the updated 7 arm joint angles (CPU tensor).
        """
        delta = torch.zeros((self.num_envs, 6), device=self.device)
        # fill rotational components only (relative mode)
        delta[:, 3] = delta_rpy[0]
        delta[:, 4] = delta_rpy[1]
        delta[:, 5] = delta_rpy[2]

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        self._ik_controller.set_command(delta, ee_pos_curr_b, ee_quat_curr_b)

        joint_pos = self._robot.data.joint_pos[:, :]
        jacobian = self._compute_frame_jacobian()
        arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        arm_joint_pos_des = arm_joint_pos_des[:, :7]

        # keep current gripper joints
        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]
        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_joint_pos], dim=1)
        self._robot.set_joint_position_target(joint_pos_des)

        for _ in range(settle_steps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        self.sim.render()
        return self._robot.data.joint_pos[0, :7].detach().cpu()

    def move_and_rotate_end_effector(
        self,
        delta_xyz: tuple[float, float, float],
        delta_rpy: tuple[float, float, float],
        settle_steps: int = 120,
    ) -> torch.Tensor:
        """Apply a combined (dx, dy, dz, droll, dpitch, dyaw) relative command via IK.

        Returns the updated 7 arm joint angles (CPU tensor).
        """
        delta = torch.zeros((self.num_envs, 6), device=self.device)
        delta[:, 0] = delta_xyz[0]
        delta[:, 1] = delta_xyz[1]
        delta[:, 2] = delta_xyz[2]
        delta[:, 3] = delta_rpy[0]
        delta[:, 4] = delta_rpy[1]
        delta[:, 5] = delta_rpy[2]

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        self._ik_controller.set_command(delta, ee_pos_curr_b, ee_quat_curr_b)

        joint_pos = self._robot.data.joint_pos[:, :]
        jacobian = self._compute_frame_jacobian()
        arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
        arm_joint_pos_des = arm_joint_pos_des[:, :7]

        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]
        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_joint_pos], dim=1)
        self._robot.set_joint_position_target(joint_pos_des)

        for _ in range(settle_steps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        self.sim.render()
        return self._robot.data.joint_pos[0, :7].detach().cpu()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        """Get observations from the environment."""
        # Proprioceptive observations
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)
        
        # Vision observations
        wrist_rgb = self.wrist_camera.data.output["rgb"]
        tactile_left = self.gsmini_left.data.output["tactile_rgb"]
        tactile_right = self.gsmini_right.data.output["tactile_rgb"]
        
        return {
            "proprio_obs": proprio_obs,
            "wrist_rgb": wrist_rgb,
            "third_person_rgb": self.third_person_camera.data.output["rgb"],
            "tactile_left": tactile_left,
            "tactile_right": tactile_right,
        }

    def _get_rewards(self) -> torch.Tensor:
        """Calculate rewards for the current state."""
        # Simple reward based on distance to cylinder
        cylinder_pos = self._cylinder.data.root_pos_w
        gripper_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        
        distance = torch.norm(gripper_pos - cylinder_pos, dim=-1)
        reward = torch.exp(-distance / 0.1)
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate done flags."""
        # No episode termination for demo
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return dones, dones

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments at the given indices."""
        # Reset robot
        self._robot.reset(env_ids)
        
        # Reset the object inside the configured XY rectangle.
        cylinder_pos = torch.zeros(len(env_ids), 3, device=self.device)
        cylinder_pos[:, 0] = (
            torch.rand(len(env_ids), device=self.device) * (CAN_RANGE_X_MAX - CAN_RANGE_X_MIN) + CAN_RANGE_X_MIN
        )
        cylinder_pos[:, 1] = (
            torch.rand(len(env_ids), device=self.device) * (CAN_RANGE_Y_MAX - CAN_RANGE_Y_MIN) + CAN_RANGE_Y_MIN
        )
        cylinder_pos[:, 2] = self._get_can_centroid_z()
        
        cylinder_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        
        # Combine position and rotation into pose tensor
        cylinder_pose = torch.cat([cylinder_pos, cylinder_rot], dim=-1)  # Shape: (len(env_ids), 7)
        self._cylinder.write_root_pose_to_sim(cylinder_pose, env_ids)
        self._cylinder.write_root_velocity_to_sim(
            torch.zeros_like(cylinder_pose[:,:6]), env_ids
        )

    def get_can_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return can position in world frame and env-local frame for env 0."""
        world_pos = self._cylinder.data.root_pos_w[0].detach().clone()
        local_pos = world_pos - self.scene.env_origins[0]
        return world_pos, local_pos

    def _get_can_xform_world_position(self) -> torch.Tensor | None:
        """Get can world position via XFormPrim.get_world_poses (env 0)."""
        if self._can_pose_view is None:
            return None
        try:
            positions, _ = self._can_pose_view.get_world_poses()
            return torch.as_tensor(positions[0], device=self.device).detach().clone()
        except Exception:
            return None

    def print_can_position(self, prefix: str = "[CAN]") -> None:
        """Print the current can position and height for env 0."""
        world_pos, local_pos = self.get_can_positions()
        xform_world_pos = self._get_can_xform_world_position()
        xform_local_pos = None if xform_world_pos is None else (xform_world_pos - self.scene.env_origins[0])
        centroid_above_floor = local_pos[2].item() - BOX_FLOOR_THICKNESS
        if xform_world_pos is None or xform_local_pos is None:
            print(
                f"{prefix} world=({world_pos[0].item():.4f}, {world_pos[1].item():.4f}, {world_pos[2].item():.4f}) "
                f"local=({local_pos[0].item():.4f}, {local_pos[1].item():.4f}, {local_pos[2].item():.4f}) "
                f"world_z={world_pos[2].item():.4f} local_z={local_pos[2].item():.4f} "
                f"centroid_above_floor_z={centroid_above_floor:.4f} xform_world_pose_z=N/A"
            )
            return
        root_vs_xform_dz = world_pos[2].item() - xform_world_pos[2].item()
        print(
            f"{prefix} world=({world_pos[0].item():.4f}, {world_pos[1].item():.4f}, {world_pos[2].item():.4f}) "
            f"local=({local_pos[0].item():.4f}, {local_pos[1].item():.4f}, {local_pos[2].item():.4f}) "
            f"world_z={world_pos[2].item():.4f} local_z={local_pos[2].item():.4f} "
            f"centroid_above_floor_z={centroid_above_floor:.4f} "
            f"xform_world_pose_z={xform_world_pos[2].item():.4f} "
            f"xform_local_z={xform_local_pos[2].item():.4f} root_minus_xform_dz={root_vs_xform_dz:.4f}"
        )

    def move_can_by(self, delta_xyz: tuple[float, float, float]) -> None:
        """Teleport the can by a small delta within the configured XY rectangle."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        root_pose = self._cylinder.data.root_state_w[:, :7].clone()
        root_vel = torch.zeros((self.num_envs, 6), device=self.device)

        delta = torch.tensor(delta_xyz, device=self.device, dtype=root_pose.dtype).view(1, 3)
        root_pose[:, :3] += delta
        local_pos = root_pose[:, :3] - self.scene.env_origins
        local_pos[:, 0] = torch.clamp(local_pos[:, 0], min=CAN_RANGE_X_MIN, max=CAN_RANGE_X_MAX)
        local_pos[:, 1] = torch.clamp(local_pos[:, 1], min=CAN_RANGE_Y_MIN, max=CAN_RANGE_Y_MAX)
        root_pose[:, :3] = local_pos + self.scene.env_origins
        root_pose[:, 2] = self._get_can_centroid_z()
        root_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=root_pose.dtype)

        self._cylinder.write_root_pose_to_sim(root_pose, env_ids)
        self._cylinder.write_root_velocity_to_sim(root_vel, env_ids)


def run_simulator(env: CylinderGraspingDemo):
    """Runs the simulation loop."""

    # 用于显示tactile图像
    if env.cfg.gsmini_left.debug_vis:
        for data_type in env.cfg.gsmini_left.data_types:
            env.gsmini_left._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
            
    if env.cfg.gsmini_right.debug_vis:
        for data_type in env.cfg.gsmini_right.data_types:
            env.gsmini_right._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
    # outer sensors (if available): require both cfg enabled and sensor instance present
    if (
        hasattr(env, 'gsmini_left_outer')
        and env.gsmini_left_outer is not None
        and hasattr(env.cfg, 'gsmini_left_outer')
        and env.cfg.gsmini_left_outer
        and env.cfg.gsmini_left_outer.debug_vis
    ):
        for data_type in env.cfg.gsmini_left_outer.data_types:
            try:
                env.gsmini_left_outer._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
            except Exception as e:
                print(f"[WARN] Cannot enable debug for gsmini_left_outer {data_type}: {e}")
    if (
        hasattr(env, 'gsmini_right_outer')
        and env.gsmini_right_outer is not None
        and hasattr(env.cfg, 'gsmini_right_outer')
        and env.cfg.gsmini_right_outer
        and env.cfg.gsmini_right_outer.debug_vis
    ):
        for data_type in env.cfg.gsmini_right_outer.data_types:
            try:
                env.gsmini_right_outer._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)
            except Exception as e:
                print(f"[WARN] Cannot enable debug for gsmini_right_outer {data_type}: {e}")
    

    print(f"Starting cylinder grasping demo with {env.num_envs} env(s)")

    env.reset()
    
    # 设置机械臂初始关节角度
    env._set_initial_joint_angles()
    # 让仿真前进几步，确保初始角度应用
    for _ in range(60):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    env.sim.render()

    # keyboard control for Franka end-effector
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard() if app_window is not None else None
    input_iface = carb_input.acquire_input_interface() if keyboard is not None else None
    kb_sub = None

    kb_step_lin_m = 0.01
    kb_step_yaw_rad = float(np.deg2rad(3.0))
    kb_step_cylinder_m = 0.01
    action_scale = max(float(getattr(env, "action_scale", 0.1)), 1e-6)
    kb_step_lin_units = kb_step_lin_m / action_scale
    kb_step_yaw_units = kb_step_yaw_rad / max(action_scale * 10.0, 1e-6)

    pressed = {
        "up": False,
        "down": False,
        "left": False,
        "right": False,
        "forward": False,
        "backward": False,
        "yaw_plus": False,
        "yaw_minus": False,
        "open": False,
        "close": False,
        "can_up": False,
        "can_down": False,
        "can_left": False,
        "can_right": False,
    }
    # this maps to finger target = action * 0.5 in _apply_joint_control
    gripper_action_units = 0.04

    def _on_kb_event(event, *args, **kwargs):
        if event.type in (
            carb_input.KeyboardEventType.KEY_PRESS,
            carb_input.KeyboardEventType.KEY_REPEAT,
            carb_input.KeyboardEventType.KEY_RELEASE,
        ):
            is_down = event.type in (
                carb_input.KeyboardEventType.KEY_PRESS,
                carb_input.KeyboardEventType.KEY_REPEAT,
            )
            if event.input == carb_input.KeyboardInput.W:
                pressed["up"] = is_down
            elif event.input == carb_input.KeyboardInput.S:
                pressed["down"] = is_down
            elif event.input == carb_input.KeyboardInput.A:
                pressed["left"] = is_down
            elif event.input == carb_input.KeyboardInput.D:
                pressed["right"] = is_down
            elif event.input == carb_input.KeyboardInput.E:
                pressed["forward"] = is_down
            elif event.input == carb_input.KeyboardInput.Q:
                pressed["backward"] = is_down
            elif event.input == carb_input.KeyboardInput.Z:
                pressed["yaw_plus"] = is_down
            elif event.input == carb_input.KeyboardInput.X:
                pressed["yaw_minus"] = is_down
            elif event.input == carb_input.KeyboardInput.J and is_down:
                pressed["open"] = True
            elif event.input == carb_input.KeyboardInput.K and is_down:
                pressed["close"] = True
            elif event.input == carb_input.KeyboardInput.UP:
                pressed["can_up"] = is_down
            elif event.input == carb_input.KeyboardInput.DOWN:
                pressed["can_down"] = is_down
            elif event.input == carb_input.KeyboardInput.LEFT:
                pressed["can_left"] = is_down
            elif event.input == carb_input.KeyboardInput.RIGHT:
                pressed["can_right"] = is_down
        return True

    if keyboard is not None and input_iface is not None:
        kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_kb_event)
        print("[INFO] EE control keys:")
        print("       Q/E: -X/+X, A/D: -Y/+Y, W/S: +Z/-Z")
        print("       Z/X: +Yaw/-Yaw, J/K: open/close gripper")
        print("[INFO] Can control keys:")
        print("       ↑/↓: +X/-X, ←/→: +Y/-Y")
        print("[INFO] Click viewport first, then use keyboard.")
    else:
        print("[WARN] Keyboard unavailable. Franka keyboard control disabled.")

    env._reset_idx(torch.arange(env.num_envs, device=env.device))
    env.print_can_position(prefix="[CAN][INIT]")

    step_counter = 0
    try:
        while simulation_app.is_running():
            if pressed["open"]:
                gripper_action_units = 0.08
                pressed["open"] = False
            if pressed["close"]:
                gripper_action_units = 0.0
                pressed["close"] = False

            dx_dir = (1.0 if pressed["forward"] else 0.0) + (-1.0 if pressed["backward"] else 0.0)
            dy_dir = (1.0 if pressed["right"] else 0.0) + (-1.0 if pressed["left"] else 0.0)
            dz_dir = (1.0 if pressed["up"] else 0.0) + (-1.0 if pressed["down"] else 0.0)
            dyaw_dir = (1.0 if pressed["yaw_plus"] else 0.0) + (-1.0 if pressed["yaw_minus"] else 0.0)
            cyl_dx_dir = (1.0 if pressed["can_up"] else 0.0) + (-1.0 if pressed["can_down"] else 0.0)
            cyl_dy_dir = (1.0 if pressed["can_left"] else 0.0) + (-1.0 if pressed["can_right"] else 0.0)

            if cyl_dx_dir != 0.0 or cyl_dy_dir != 0.0:
                env.move_can_by(
                    (
                        cyl_dx_dir * kb_step_cylinder_m,
                        cyl_dy_dir * kb_step_cylinder_m,
                        0.0,
                    )
                )

            actions = torch.zeros((env.num_envs, 5), device=env.device)
            actions[:, 0] = dx_dir * kb_step_lin_units
            actions[:, 1] = dy_dir * kb_step_lin_units
            actions[:, 2] = dz_dir * kb_step_lin_units
            actions[:, 3] = dyaw_dir * kb_step_yaw_units
            actions[:, 4] = gripper_action_units

            env._pre_physics_step(actions)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

            if CAN_PRINT_EVERY_STEPS > 0 and step_counter % CAN_PRINT_EVERY_STEPS == 0:
                env.print_can_position(prefix=f"[CAN][STEP={step_counter}]")

            env.sim.render()

            if step_counter % 60 == 0:
                joint_names = env._robot.joint_names
                joint_pos = env._robot.data.joint_pos[0].detach().cpu().numpy()
                print(f"[JOINT] step={step_counter}, gripper_cmd={gripper_action_units:.3f}")
                for name, angle in zip(joint_names, joint_pos):
                    print(f"  {name}: {angle:.4f} rad ({np.degrees(angle):.2f} deg)")
            step_counter += 1
    finally:
        if kb_sub is not None and keyboard is not None and input_iface is not None:
            try:
                input_iface.unsubscribe_from_keyboard_events(keyboard, kb_sub)
            except Exception:
                pass
        env.close()
        pynvml.nvmlShutdown()


def main():
    """Main function."""
    # Define simulation env
    env_cfg = CylinderGraspingDemoCfg()
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini_left.debug_vis = args_cli.debug_vis
    env_cfg.gsmini_right.debug_vis = args_cli.debug_vis
    if hasattr(env_cfg, 'gsmini_left_outer'):
        env_cfg.gsmini_left_outer.debug_vis = args_cli.debug_vis
    if hasattr(env_cfg, 'gsmini_right_outer'):
        env_cfg.gsmini_right_outer.debug_vis = args_cli.debug_vis

    experiment = CylinderGraspingDemo(env_cfg)

    
    # Run the simulator
    run_simulator(env=experiment)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim apply
        simulation_app.close()
