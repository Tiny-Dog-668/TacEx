# Copyright (c) 2022-2023, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Modified version of the original FRANKA_PANDA_CFG of Isaac Lab
#
"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG`: Franka Emika Panda robot with Panda hand

Reference: https://github.com/frankaemika/franka_ros
"""

import hashlib
import os
import tempfile
import uuid
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from pxr import Gf, Sdf, Usd, UsdShade

from tacex_assets import TACEX_ASSETS_DATA_DIR


GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_PROFILE = (
    "gelsight_physics_isaaclab_panda_arm_link0_7_visuals_green_base_led_usd_v5"
)
GELSIGHT_STANDARD_FRANKA_BASE_LED_SUBSET_PATH = (
    "panda_link0/standard_visuals/panda_link0/subset_5"
)
GELSIGHT_STANDARD_FRANKA_BASE_LED_COLOR_RGB = (0.0, 1.0, 0.0)
_STANDARD_FRANKA_ARM_LINKS = tuple(f"panda_link{index}" for index in range(8))


def create_gelsight_standard_franka_arm_visual_usd() -> str:
    """Create a cached USD that combines GelSight physics with standard Panda arm visuals."""
    gelsight_usd = str(
        Path(TACEX_ASSETS_DATA_DIR)
        / "Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.usd"
    )
    standard_franka_usd = (
        f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    )
    cache_key = hashlib.sha256(
        "\0".join(
            (
                GELSIGHT_STANDARD_FRANKA_ARM_VISUAL_PROFILE,
                gelsight_usd,
                standard_franka_usd,
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = Path(tempfile.gettempdir()) / "tacex_usd_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"franka_gsmini_standard_arm_visuals_{cache_key}.usda"
    if output.is_file():
        return str(output)

    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.usda")
    try:
        stage = Usd.Stage.CreateNew(str(temporary))
        robot = stage.DefinePrim("/panda", "Xform")
        if not robot.GetReferences().SetReferences([Sdf.Reference(gelsight_usd)]):
            raise RuntimeError("Failed to reference the GelSight Franka USD")
        stage.SetDefaultPrim(robot)
        for link_name in _STANDARD_FRANKA_ARM_LINKS:
            original_visual = stage.OverridePrim(f"/panda/{link_name}/visuals")
            original_visual.SetActive(False)
            standard_visual = stage.DefinePrim(
                f"/panda/{link_name}/standard_visuals", "Xform"
            )
            # link0 stays editable so its status-light subset can receive the
            # task-local green binding below.  link1-7 remain instanceable.
            standard_visual.SetInstanceable(link_name != "panda_link0")
            if not standard_visual.GetReferences().SetReferences(
                [
                    Sdf.Reference(
                        standard_franka_usd,
                        Sdf.Path(f"/panda/{link_name}/visuals"),
                    )
                ]
            ):
                raise RuntimeError(f"Failed to author standard visual for {link_name}")

        green_material = UsdShade.Material.Define(
            stage, "/panda/Looks/TacExBaseStatusGreen"
        )
        green_shader = UsdShade.Shader.Define(
            stage, "/panda/Looks/TacExBaseStatusGreen/Shader"
        )
        green_shader.CreateIdAttr("UsdPreviewSurface")
        green_rgb = Gf.Vec3f(*GELSIGHT_STANDARD_FRANKA_BASE_LED_COLOR_RGB)
        green_shader.CreateInput(
            "diffuseColor", Sdf.ValueTypeNames.Color3f
        ).Set(green_rgb)
        green_shader.CreateInput(
            "emissiveColor", Sdf.ValueTypeNames.Color3f
        ).Set(green_rgb)
        green_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.298)
        green_material.CreateSurfaceOutput().ConnectToSource(
            green_shader.ConnectableAPI(), "surface"
        )
        base_led = stage.OverridePrim(
            f"/panda/{GELSIGHT_STANDARD_FRANKA_BASE_LED_SUBSET_PATH}"
        )
        if not base_led.IsValid():
            raise RuntimeError("Standard Panda base status-light subset is missing")
        UsdShade.MaterialBindingAPI.Apply(base_led).Bind(green_material)
        stage.GetRootLayer().Save()
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(output)

# todo find a good way to save the prim path of the sensor for the user?
# -> currently, we need to look into the asset to figure out the prim name (in this case its /gelsight_mini_case)
FRANKA_PANDA_ARM_GSMINI_GRIPPER_RIGID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.down.usda",
        usd_path=f"{TACEX_ASSETS_DATA_DIR}/Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
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
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=40.0,
            velocity_limit_sim=0.2,
            stiffness=400,
            damping=40,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot with a Gripper and two GelSight Mini sensors.

The gelpads are simulated via PhysX and rigid.

Sensor case prim names:
- `gelsight_mini_case_left`
- `gelsight_mini_case_right`

Gelpad prim names:
- `gelpad_left`
- `gelpad_right`
"""


FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG = FRANKA_PANDA_ARM_GSMINI_GRIPPER_RIGID_CFG.copy()
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.

Sensor case prim names:
- `gelsight_mini_case_left`
- `gelsight_mini_case_right`

Gelpad prim names:
- `gelpad_left`
- `gelpad_right`
"""
FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG.actuators["panda_forearm"].damping = 80.0
