from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Cylinder Grasping Demo with Franka robot equipped with GelSight sensors and wrist camera"
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sys", type=bool, default=True, help="Whether to track system utilization.")
parser.add_argument(
    "--debug_vis",
    default=True,
    action="store_true",
    help="Whether to render tactile images in the GUI",
)
parser.add_argument(
    "--terminal_input",
    action="store_true",
    default=False,
    help="Enable terminal keyboard fallback (not used now, kept for compatibility).",
)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



import carb

def enable_translucency():
    s = carb.settings.get_settings()
    s.set_bool("/rtx/translucency/enabled", True)
    s.set_bool("/rtx/pathtracing/fractionalCutoutOpacity", True)

enable_translucency()

import numpy as np
import torch
import traceback
from contextlib import suppress

import omni.ui


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
from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG,
)
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg


class CustomEnvWindow(BaseEnvWindow):
    """Window manager for the RL environment."""

    def __init__(self, env: DirectRLEnvCfg, window_name: str = "IsaacLab"):
        """Initialize the window."""
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)
                    # add simple manual control buttons (UI fallback)
                    import omni.ui as ui

                    with ui.CollapsableFrame("Manual Control (XY)", collapsed=False):
                        with ui.HStack(height=0):
                            ui.Label("Step: 0.004m", style={"color": 0xFFAAAAAA})

                        def _mk_cb(dx, dy):
                            def _cb():
                                try:
                                    action_scale = getattr(self.env, "action_scale", 0.1)
                                    step_units = 0.004 / max(action_scale, 1e-6)
                                    if hasattr(self.env, "enqueue_manual_move"):
                                        self.env.enqueue_manual_move(dx * step_units, dy * step_units)
                                except Exception:
                                    pass

                            return _cb

                        with ui.HStack():
                            ui.Spacer(width=8)
                            ui.Button("⬅", clicked_fn=_mk_cb(-1.0, 0.0))
                            ui.Button("⬆", clicked_fn=_mk_cb(0.0, 1.0))
                            ui.Button("⬇", clicked_fn=_mk_cb(0.0, -1.0))
                            ui.Button("➡", clicked_fn=_mk_cb(1.0, 0.0))
                            ui.Spacer(width=8)


@configclass
class CylinderGraspingDemoCfg(DirectRLEnvCfg):
    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.9, 1.4, 0.3)
    viewer.lookat = (-1.5, -1.9, -1.1)

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
        replicate_physics=False,
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

    # plate under cylinder
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block.usd",
            scale=(10, 10, 0.01),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=10.0,
                max_depenetration_velocity=1.0,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0005,
            ),
        ),
    )

    cylinder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.02]),
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.06,
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
                diffuse_color=(0.8, 0.2, 0.2),
                metallic=0.0,
                roughness=0.5,
            ),
        ),
    )

    bottle = AssetBaseCfg(
        prim_path="/World/envs/env_.*/bottle",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, -0.08, 0.02)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/tinydog/download/drive-download-20251214T055420Z-1-001/bottle.usd",
            scale=(1.0, 1.0, 1.0),
        ),
    )

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

    transparent_cylinder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/transparent_cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.08, 0.03]),
        spawn=sim_utils.CylinderCfg(
            radius=0.02,
            height=0.06,
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
                diffuse_color=(0.6, 0.8, 1.0),
                metallic=0.0,
                roughness=0.1,
                opacity=0.3,  # 半透明
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
                "panda_joint1": -0.3136,
                "panda_joint2": -0.4448,
                "panda_joint3": 0.3858,
                "panda_joint4": -3.0236,
                "panda_joint5": 0.2319,
                "panda_joint6": 2.5776,
                "panda_joint7": 0.6249,
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
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.15, 0, -0.12),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="ros",
        ),
    )

    # third-person camera
    third_person_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/third_person_camera",
        update_period=0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=40.0,
            clipping_range=(0.05, 30.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.25, 0.0, 0.5),
            rot=(0.5, -0.5, -0.5, 0.5),
            convention="ros",
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
        debug_vis=True,
        marker_motion_sim_cfg=None,
        # 仅输出深度（相机深度），不渲染触觉 RGB
        data_types=["camera_depth"],
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

    gsmini_left_outer = None
    gsmini_right_outer = None
    gsmini_left_out = None
    gsmini_right_out = None

    # IK controller
    ik_controller_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=True,
        ik_method="dls",
    )
    episode_length_s = 0
    action_space = 5  # xyz + rz + gripper
    observation_space = 0
    state_space = 0


class CylinderGraspingDemo(DirectRLEnv):
    """Cylinder grasping demo environment."""

    cfg: CylinderGraspingDemoCfg

    def __init__(self, cfg: CylinderGraspingDemoCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )

        # end-effector body index
        body_ids, body_names = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # finger joints
        self._finger_joint_ids, self._finger_joint_names = self._robot.find_joints(["panda_finger.*"])

        # For a fixed base robot, the frame index is one less than the body index.
        self._jacobi_body_idx = self._body_idx - 1

        # ee offset w.r.t panda hand
        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        # IK command buffers
        self.ik_commands = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.action_scale = 0.1
        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)

        # 当前动作 [dx, dy, dz, d_yaw, gripper]
        self.current_actions = torch.zeros((self.num_envs, 5), device=self.device)

        self.step_count = 0
        self.manual_move_units = torch.zeros((self.num_envs, 2), device=self.device)

        self.joint_angles = {
            "joint1": -0.3136,
            "joint2": -0.4448,
            "joint3": 0.3858,
            "joint4": -3.0236,
            "joint5": 0.2319,
            "joint6": 2.5776,
            "joint7": 0.6249,
            "finger_left": 0.02,
            "finger_right": 0.02,
        }

        # Only randomize visuals once per app launch.
        self._materials_randomized = False

        self.set_debug_vis(self.cfg.debug_vis)

    def _env_ids_to_list(self, env_ids: torch.Tensor | None):
        """Normalize env_ids to a Python list for indexing prim paths."""
        if env_ids is None:
            return list(range(self.num_envs))
        if isinstance(env_ids, torch.Tensor):
            return env_ids.cpu().tolist()
        if isinstance(env_ids, (list, tuple)):
            return list(env_ids)
        return [int(env_ids)]

    def _random_color_uniform(self, low: float = 0.1, high: float = 0.9) -> tuple[float, float, float]:
        """Sample an RGB color in [low, high]."""
        return tuple(float(c) for c in np.random.uniform(low, high, size=3))

    def _random_wood_color(self) -> tuple[float, float, float]:
        """Sample a warm wood-like color with jitter."""
        base = np.array([0.65, 0.45, 0.30], dtype=np.float32)
        jitter = np.random.uniform(-0.2, 0.2, size=3)
        color = np.clip(base + jitter, 0.05, 0.95)
        return tuple(float(c) for c in color)

    def _randomize_env_lights(self, env_ids: torch.Tensor | None):
        """(Disabled) Per-env light randomization not used currently."""
        return

    def _apply_preview_material(
        self,
        material_path: str,
        target_prim_path: str,
        color: tuple[float, float, float],
        roughness: float,
        metallic: float,
        opacity: float | None = None,
        opacity_threshold: float | None = None,
    ):
        """Ensure a PreviewSurface material exists, bind it, and update its parameters."""
        stage = sim_utils.stage_utils.get_current_stage()
        try:
            material_prim = stage.GetPrimAtPath(material_path)
            if not material_prim.IsValid():
                sim_utils.spawn_preview_surface(material_path, sim_utils.PreviewSurfaceCfg())
            sim_utils.bind_visual_material(target_prim_path, material_path)
            shader_prim = stage.GetPrimAtPath(f"{material_path}/Shader")
            if not shader_prim.IsValid():
                print(f"[WARN] 无法找到材质节点: {material_path}/Shader")
                return
            sim_utils.safe_set_attribute_on_usd_prim(shader_prim, "inputs:diffuse_color", color, camel_case=True)
            sim_utils.safe_set_attribute_on_usd_prim(shader_prim, "inputs:roughness", float(roughness), camel_case=True)
            sim_utils.safe_set_attribute_on_usd_prim(shader_prim, "inputs:metallic", float(metallic), camel_case=True)
            if opacity is not None:
                sim_utils.safe_set_attribute_on_usd_prim(shader_prim, "inputs:opacity", float(opacity), camel_case=True)
            if opacity_threshold is not None:
                sim_utils.safe_set_attribute_on_usd_prim(
                    shader_prim, "inputs:opacity_threshold", float(opacity_threshold), camel_case=True
                )
        except Exception as exc:
            print(f"[WARN] 随机化材质失败 ({material_path}): {exc}")

    def _randomize_object_materials(self, env_ids: torch.Tensor | None):
        """Randomize visual material and color for cylinder and plate per environment."""
        env_id_list = self._env_ids_to_list(env_ids)
        for env_id in env_id_list:
            # cylinder material on its mesh
            cylinder_root = self._cylinder.root_physx_view.prim_paths[env_id]
            cylinder_mesh_path = f"{cylinder_root}/geometry/mesh"
            cylinder_material_path = f"{cylinder_root}/geometry/material"
            self._apply_preview_material(
                material_path=cylinder_material_path,
                target_prim_path=cylinder_mesh_path,
                color=self._random_color_uniform(0.15, 0.9),
                roughness=float(np.random.uniform(0.1, 0.8)),
                metallic=float(np.random.uniform(0.0, 0.6)),
            )

            # plate material bound to the plate root (applies to its meshes)
            plate_root = self._plate.root_physx_view.prim_paths[env_id]
            plate_material_path = f"{plate_root}/material"
            self._apply_preview_material(
                material_path=plate_material_path,
                target_prim_path=plate_root,
                color=self._random_wood_color(),
                roughness=float(np.random.uniform(0.3, 0.9)),
                metallic=float(np.random.uniform(0.0, 0.2)),
            )

            # cube material
            cube_root = self._cube.root_physx_view.prim_paths[env_id]
            cube_mesh_path = f"{cube_root}/geometry/mesh"
            cube_material_path = f"{cube_root}/geometry/material"
            self._apply_preview_material(
                material_path=cube_material_path,
                target_prim_path=cube_mesh_path,
                color=self._random_color_uniform(0.1, 0.9),
                roughness=float(np.random.uniform(0.2, 0.8)),
                metallic=float(np.random.uniform(0.0, 0.6)),
            )

            # sphere material
            sphere_root = self._sphere.root_physx_view.prim_paths[env_id]
            sphere_mesh_path = f"{sphere_root}/geometry/mesh"
            sphere_material_path = f"{sphere_root}/geometry/material"
            self._apply_preview_material(
                material_path=sphere_material_path,
                target_prim_path=sphere_mesh_path,
                color=self._random_color_uniform(0.1, 0.9),
                roughness=float(np.random.uniform(0.2, 0.7)),
                metallic=float(np.random.uniform(0.0, 0.6)),
            )

            # transparent cylinder material (keep low opacity)
            tcy_root = self._transparent_cylinder.root_physx_view.prim_paths[env_id]
            tcy_mesh_path = f"{tcy_root}/geometry/mesh"
            tcy_material_path = f"{tcy_root}/geometry/material"
            self._apply_preview_material(
                material_path=tcy_material_path,
                target_prim_path=tcy_mesh_path,
                color=(0.6, 0.8, 1.0),
                roughness=0.1,
                metallic=0.0,
                opacity=0.3,
                opacity_threshold=0.00,
            )

    def enqueue_manual_move(self, dx_units: float, dy_units: float):
        try:
            self.manual_move_units[:, 0] += float(dx_units)
            self.manual_move_units[:, 1] += float(dy_units)
        except Exception:
            pass

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._cylinder = RigidObject(self.cfg.cylinder)
        self.scene.rigid_objects["cylinder"] = self._cylinder

        self._cube = RigidObject(self.cfg.cube)
        self.scene.rigid_objects["cube"] = self._cube

        self._sphere = RigidObject(self.cfg.sphere)
        self.scene.rigid_objects["sphere"] = self._sphere

        self._transparent_cylinder = RigidObject(self.cfg.transparent_cylinder)
        self.scene.rigid_objects["transparent_cylinder"] = self._transparent_cylinder

        self._plate = RigidObject(self.cfg.plate)
        self.scene.rigid_objects["plate"] = self._plate

        self.wrist_camera = Camera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera

        self.third_person_camera = Camera(self.cfg.third_person_camera)
        self.scene.sensors["third_person_camera"] = self.third_person_camera

        self.scene.clone_environments(copy_from_source=False)

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

        self.gsmini_left = GelSightSensor(self.cfg.gsmini_left)
        self.scene.sensors["gsmini_left"] = self.gsmini_left

        self.gsmini_right = GelSightSensor(self.cfg.gsmini_right)
        self.scene.sensors["gsmini_right"] = self.gsmini_right

        # outer sensors
        # outer / out sensors disabled for this demo (prim paths not present)
        self.gsmini_left_outer = None
        self.gsmini_right_outer = None
        self.gsmini_left_out = None
        self.gsmini_right_out = None

        # ground
        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        # spawn bottle as a static asset (USD lacks rigid body API)
        self.cfg.bottle.spawn.func(
            self.cfg.bottle.prim_path,
            self.cfg.bottle.spawn,
            translation=self.cfg.bottle.init_state.pos,
            orientation=self.cfg.bottle.init_state.rot,
        )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions before physics step."""
        self.current_actions = actions.clone()

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()

        # 5维动作 [dx, dy, dz, d_yaw, gripper] -> 6维 [dx, dy, dz, droll, dpitch, dyaw]
        self.processed_actions[:, :3] = self.current_actions[:, :3] * self.action_scale
        self.processed_actions[:, 3] = 0.0
        self.processed_actions[:, 4] = 0.0
        self.processed_actions[:, 5] = self.current_actions[:, 3] * self.action_scale * 10.0

        self._ik_controller.set_command(self.processed_actions, ee_pos_curr_b, ee_quat_curr_b)
        self._apply_joint_control()

    def _apply_joint_control(self):
        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()
        joint_pos = self._robot.data.joint_pos[:, :]

        if ee_pos_curr_b.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            arm_joint_pos_des = self._ik_controller.compute(ee_pos_curr_b, ee_quat_curr_b, jacobian, joint_pos)
            arm_joint_pos_des = arm_joint_pos_des[:, :7]
        else:
            arm_joint_pos_des = joint_pos[:, :7].clone()

        gripper_action = self.current_actions[:, -1]
        gripper_joint_pos = joint_pos[:, self._finger_joint_ids]

        gripper_pos_des = gripper_action.unsqueeze(-1) * 0.5
        gripper_pos_des = gripper_pos_des.expand(-1, len(self._finger_joint_ids))

        joint_pos_des = torch.cat([arm_joint_pos_des, gripper_pos_des], dim=1)
        self._robot.set_joint_position_target(joint_pos_des)

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w

        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )
        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

        base_rot = self._robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        jacobian[:, 0:3, :] += torch.bmm(
            -math_utils.skew_symmetric_matrix(self._offset_pos),
            jacobian[:, 3:, :],
        )
        jacobian[:, 3:, :] = torch.bmm(
            math_utils.matrix_from_quat(self._offset_rot),
            jacobian[:, 3:, :],
        )

        return jacobian

    def set_joint_angle(self, joint_name: str, angle: float):
        if joint_name in self.joint_angles:
            self.joint_angles[joint_name] = angle
            print(f"设置 {joint_name} 角度为 {angle} 弧度 ({angle * 180 / 3.14159:.1f} 度)")
        else:
            print(f"错误: 未知的关节名称 '{joint_name}'")
            print(f"可用的关节: {list(self.joint_angles.keys())}")

    def set_all_joint_angles(self, angles: dict):
        for joint_name, angle in angles.items():
            if joint_name in self.joint_angles:
                self.joint_angles[joint_name] = angle
            else:
                print(f"警告: 未知的关节名称 '{joint_name}'")
        print("所有关节角度已更新")

    def get_current_joint_angles(self):
        return self._robot.data.joint_pos[0].cpu().numpy()

    def _set_initial_joint_angles(self):
        print("设置机械臂初始关节角度...")
        try:
            if not hasattr(self._robot, "joint_names"):
                print("  警告: 机器人对象尚未完全初始化，跳过关节角度设置")
                return

            all_joint_names = self._robot.joint_names
            print(f"  可用关节: {all_joint_names}")

            joint_name_mapping = {
                "joint1": "panda_joint1",
                "joint2": "panda_joint2",
                "joint3": "panda_joint3",
                "joint4": "panda_joint4",
                "joint5": "panda_joint5",
                "joint6": "panda_joint6",
                "joint7": "panda_joint7",
                "finger_left": "panda_finger_joint1",
                "finger_right": "panda_finger_joint2",
            }

            for joint_name, angle in self.joint_angles.items():
                actual_joint_name = joint_name_mapping.get(joint_name, joint_name)
                if actual_joint_name in all_joint_names:
                    joint_idx = all_joint_names.index(actual_joint_name)
                    self._robot.data.joint_pos_target[:, joint_idx] = angle
                    print(
                        f"  设置 {joint_name} ({actual_joint_name}) 角度为 "
                        f"{angle:.4f} 弧度 ({angle * 180 / 3.14159:.1f} 度)"
                    )
                else:
                    print(f"  警告: 关节 {joint_name} ({actual_joint_name}) 不存在于机器人配置中")

            self._robot.write_joint_state_to_sim(
                self._robot.data.joint_pos_target,
                self._robot.data.joint_vel,
            )
            print("机械臂初始关节角度设置完成")
        except Exception as e:
            print(f"  错误: 设置关节角度时发生异常: {e}")
            print("  将使用默认关节角度继续运行")

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        proprio_obs = torch.cat([joint_pos, joint_vel], dim=-1)

        wrist_rgb = self.wrist_camera.data.output["rgb"]
        # 使用相机深度（已在 GelSight 实现中做归一化到 0~255，单通道，形状 N×H×W×1）
        tactile_left_depth = self.gsmini_left.data.output.get("camera_depth")
        tactile_right_depth = self.gsmini_right.data.output.get("camera_depth")

        return {
            "proprio_obs": proprio_obs,
            "wrist_rgb": wrist_rgb,
            "tactile_left_depth": tactile_left_depth,
            "tactile_right_depth": tactile_right_depth,
        }

    def _get_rewards(self) -> torch.Tensor:
        cylinder_pos = self._cylinder.data.root_pos_w
        gripper_pos = self._robot.data.body_link_pos_w[:, self._body_idx]
        distance = torch.norm(gripper_pos - cylinder_pos, dim=-1)
        reward = torch.exp(-distance / 0.1)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return dones, dones

    def _reset_idx(self, env_ids: torch.Tensor):
        self._robot.reset(env_ids)
        # reset cylinder state to clear any residual velocities/transforms
        self._cylinder.reset(env_ids)
        self._cube.reset(env_ids)
        self._sphere.reset(env_ids)
        self._transparent_cylinder.reset(env_ids)

        # 随机化圆柱和木板的视觉材质与颜色（仅首次运行）
        if not self._materials_randomized:
            self._randomize_object_materials(env_ids)
            self._materials_randomized = True

        env_origins = self.scene.env_origins[env_ids].to(device=self.device)
        cylinder_pos_local = torch.tensor([0.5, 0.0, 0.03], device=self.device).expand(len(env_ids), 3).clone()
        cylinder_pos = env_origins + cylinder_pos_local

        # place cube and sphere on the plate with small xy jitter
        jitter_xy = (torch.rand(len(env_ids), 2, device=self.device) - 0.5) * 0.04  # ±2 cm

        cube_pos_local = torch.tensor([0.4, 0.05, 0.02], device=self.device).expand(len(env_ids), 3).clone()
        cube_pos_local[:, :2] = cube_pos_local[:, :2] + jitter_xy
        cube_pos = env_origins + cube_pos_local

        sphere_pos_local = torch.tensor([0.6, -0.05, 0.02], device=self.device).expand(len(env_ids), 3).clone()
        sphere_pos_local[:, :2] = sphere_pos_local[:, :2] + jitter_xy
        sphere_pos = env_origins + sphere_pos_local

        cylinder_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        cylinder_pose = torch.cat([cylinder_pos, cylinder_rot], dim=-1)
        self._cylinder.write_root_pose_to_sim(cylinder_pose, env_ids)
        self._cylinder.write_root_velocity_to_sim(
            torch.zeros_like(cylinder_pose[:, :6]), env_ids
        )

        cube_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        cube_pose = torch.cat([cube_pos, cube_rot], dim=-1)
        self._cube.write_root_pose_to_sim(cube_pose, env_ids)
        self._cube.write_root_velocity_to_sim(torch.zeros_like(cube_pose[:, :6]), env_ids)

        sphere_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        sphere_pose = torch.cat([sphere_pos, sphere_rot], dim=-1)
        self._sphere.write_root_pose_to_sim(sphere_pose, env_ids)
        self._sphere.write_root_velocity_to_sim(torch.zeros_like(sphere_pose[:, :6]), env_ids)

        trans_cyl_pos_local = torch.tensor([0.55, 0.08, 0.03], device=self.device).expand(len(env_ids), 3).clone()
        trans_cyl_pos = env_origins + trans_cyl_pos_local
        trans_cyl_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        trans_cyl_pose = torch.cat([trans_cyl_pos, trans_cyl_rot], dim=-1)
        self._transparent_cylinder.write_root_pose_to_sim(trans_cyl_pose, env_ids)
        self._transparent_cylinder.write_root_velocity_to_sim(torch.zeros_like(trans_cyl_pose[:, :6]), env_ids)


def run_simulator(env: CylinderGraspingDemo):
    """Runs the simulation loop."""

    print(f"Starting cylinder grasping demo with {env.num_envs} env(s)")

    env.reset()
    # enforce desired joint angles
    env._set_initial_joint_angles()
    try:
        env.scene.filter_collisions([])  # filter collisions across envs when replicate_physics=False
    except Exception:
        pass
    # allow state to settle a bit
    for _ in range(60):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    env.sim.render()

    try:
        while simulation_app.is_running():
            actions = torch.zeros((env.num_envs, 5), device=env.device)
            env._pre_physics_step(actions)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            env.sim.render()
    finally:
        env.close()
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError_Uninitialized:
            pass


def main():
    env_cfg = CylinderGraspingDemoCfg()
    env_cfg.seed = 42
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.gsmini_left.debug_vis = args_cli.debug_vis
    env_cfg.gsmini_right.debug_vis = args_cli.debug_vis
    if getattr(env_cfg, "gsmini_left_outer", None) is not None:
        env_cfg.gsmini_left_outer.debug_vis = args_cli.debug_vis
    if getattr(env_cfg, "gsmini_right_outer", None) is not None:
        env_cfg.gsmini_right_outer.debug_vis = args_cli.debug_vis
    if getattr(env_cfg, "gsmini_left_out", None) is not None:
        env_cfg.gsmini_left_out.debug_vis = args_cli.debug_vis
    if getattr(env_cfg, "gsmini_right_out", None) is not None:
        env_cfg.gsmini_right_out.debug_vis = args_cli.debug_vis

    experiment = CylinderGraspingDemo(env_cfg)
    run_simulator(env=experiment)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        simulation_app.close()
