# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the cylinder grasping environment."""

import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab.utils.noise import NoiseModelCfg

from tacex import GelSightSensor, GelSightSensorCfg
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.robots.franka import FRANKA_PANDA_ARM_GSMINI_GRIPPER_UIPC_CFG
from tacex_assets.sensors.gelsight_mini import GelSightMiniCfg


@configclass
class CylinderCfg(AssetBaseCfg):
    """Configuration for the cylinder object."""

    def __post_init__(self):
        super().__post_init__()
        self.spawn = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block_1.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,
                rest_offset=0.0,
            ),
        )
        self.init_state = AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
        )


@configclass
class CylinderGraspingEnvCfg(DirectRLEnvCfg):
    """Configuration for the cylinder grasping environment."""

    # viewer settings
    viewer: sim_utils.ViewerCfg = sim_utils.ViewerCfg()
    viewer.eye = (1.5, 1.5, 1.0)
    viewer.lookat = (0.0, 0.0, 0.5)

    # simulation
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 120,
        substeps=1,
        physx=sim_utils.PhysxCfg(
            enable_ccd=True,
            enable_gyroscopic_forces=True,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_max_num_partitions=1,
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=2.0,
        replicate_physics=True,
    )

    # robot configuration with dual GelSight sensors
    robot: ArticulationCfg = FRANKA_PANDA_ARM_GSMINI_GRIPPER_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # wrist camera configuration
    wrist_camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )

    # GelSight sensors on gripper fingers
    gelsight_left: GelSightSensorCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(64, 64),
            data_types=["rgb", "depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=False,
        data_types=["tactile_rgb", "camera_rgb", "camera_depth"],
    )

    gelsight_right: GelSightSensorCfg = GelSightMiniCfg(
        prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=0,
            resolution=(64, 64),
            data_types=["rgb", "depth"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=False,
        data_types=["tactile_rgb", "camera_rgb", "camera_depth"],
    )

    # cylinder object
    cylinder: CylinderCfg = CylinderCfg(
        prim_path="/World/envs/env_.*/Cylinder",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # noise models
    action_noise_model = NoiseModelCfg(
        noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
    )

    # reward configuration
    reward_terms = {
        "grasp_reward": {"weight": 1.0, "threshold": 0.02},
        "lift_reward": {"weight": 0.5, "target_height": 0.2},
        "approach_reward": {"weight": 0.3, "threshold": 0.1},
        "gripper_penalty": {"weight": -0.1},
        "action_penalty": {"weight": -0.01},
    }

    # environment settings
    episode_length_s = 10.0
    action_space = 7  # 6 for arm + 1 for gripper
    observation_space = {
        "proprio_obs": 14,  # joint positions and velocities
        "wrist_rgb": [480, 640, 3],  # wrist camera RGB
        "tactile_left": [64, 64, 3],  # left GelSight RGB
        "tactile_right": [64, 64, 3],  # right GelSight RGB
    }
    state_space = 0

    # task specific parameters
    cylinder_height = 0.1
    cylinder_radius = 0.02
    grasp_threshold = 0.02
    lift_height = 0.2
    approach_threshold = 0.1

    # randomization
    cylinder_pos_range = 0.1
    cylinder_rot_range = 0.2
