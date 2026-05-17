"""Cube grasping demo with protrusions and keyboard-controlled Franka.

Usage:
    isaaclab -p scripts/cylinder_grasping/cube_grasping_demo.py
"""

from __future__ import annotations

import argparse
import random
from typing import List, Tuple

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Cube grasping demo with a Franka arm.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--sys", type=bool, default=True, help="Whether to track system utilization.")
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import carb.input as carb_input
import omni.appwindow
import omni.usd
import torch
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from tacex_assets.robots.franka.franka_gsmini_gripper_rigid import FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG


def add_random_protrusions_to_cube(
    cube_prim_path: str,
    cube_size: Tuple[float, float, float],
    fixed_radius: float = 0.0125,
    faces: Tuple[str, ...] = ("+X", "-X", "+Y", "-Y", "+Z"),
    gap: float = 0.001,
    seed: int | None = None,
    enable_visual: bool = True,
    bump_friction: Tuple[float, float] = (0.5, 0.5),
    bump_restitution: float = 0.0,
    max_tries_per_face: int = 200,
) -> None:
    """Add spherical protrusions (collision spheres) on cube surface."""
    if seed is not None:
        random.seed(seed)

    stage = omni.usd.get_context().get_stage()
    container_path = f"{cube_prim_path}/Protrusions"
    material_suffix = cube_prim_path.replace("/", "_").strip("_")
    material_path = f"/World/Materials/bump_mat_{material_suffix}"

    if stage.GetPrimAtPath(material_path).IsValid():
        bump_material_path = stage.GetPrimAtPath(material_path).GetPath()
    else:
        bump_mat = sim_utils.spawn_rigid_body_material(
            prim_path=material_path,
            cfg=sim_utils.RigidBodyMaterialCfg(
                static_friction=bump_friction[0],
                dynamic_friction=bump_friction[1],
                restitution=bump_restitution,
            ),
        )
        bump_material_path = bump_mat.GetPath()

    if stage.GetPrimAtPath(container_path).IsValid():
        stage.RemovePrim(container_path)

    UsdGeom.Xform.Define(stage, container_path)

    hx, hy, hz = cube_size[0] * 0.5, cube_size[1] * 0.5, cube_size[2] * 0.5
    placed: List[Tuple[Gf.Vec3d, float]] = []

    def sample_on_face(face: str, r: float) -> Gf.Vec3d:
        mx = hx - r - gap
        my = hy - r - gap
        mz = hz - r - gap

        if face == "+X":
            x = hx
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
            z = hz
        else:
            raise ValueError(face)

        return Gf.Vec3d(x, y, z)

    def is_non_overlapping(pos: Gf.Vec3d, r: float) -> bool:
        for p, pr in placed:
            if (pos - p).GetLength() < (r + pr + gap):
                return False
        return True

    created = 0

    for i, face in enumerate(faces):
        r = float(fixed_radius)
        pos = None
        for _ in range(max_tries_per_face):
            cand = sample_on_face(face, r)
            if is_non_overlapping(cand, r):
                pos = cand
                break

        if pos is None:
            if face == "+X":
                pos = Gf.Vec3d(hx, 0.0, 0.0)
            elif face == "-X":
                pos = Gf.Vec3d(-hx, 0.0, 0.0)
            elif face == "+Y":
                pos = Gf.Vec3d(0.0, hy, 0.0)
            elif face == "-Y":
                pos = Gf.Vec3d(0.0, -hy, 0.0)
            elif face == "+Z":
                pos = Gf.Vec3d(0.0, 0.0, hz)
            else:
                continue

        placed.append((pos, r))

        bump_path = f"{container_path}/bump_{i:02d}"
        sphere = UsdGeom.Sphere.Define(stage, bump_path)
        sphere.GetRadiusAttr().Set(r)

        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(pos)

        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(sphere.GetPrim())
        sim_utils.bind_physics_material(bump_path, bump_material_path)

        if enable_visual:
            sphere.CreateDisplayColorAttr().Set([(0.0, 0.0, 0.0)])
            sphere.CreateDisplayOpacityAttr().Set([1.0])

        created += 1

    print(f"[INFO] Added {created}/{len(faces)} protrusions under {cube_prim_path}.")


@configclass
class CubeGraspingDemoCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (1.6, 1.2, 0.7)
    viewer.lookat = (0.5, 0.0, 0.05)

    debug_vis = True
    decimation = 1

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(enable_ccd=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,
    )

    board_size = (0.6, 0.6, 0.01)
    cube_size = (0.06, 0.06, 0.06)

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

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    board = RigidObjectCfg(
        prim_path="/World/envs/env_.*/wood_board",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, board_size[2] * 0.5)),
        spawn=sim_utils.CuboidCfg(
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
        ),
    )

    cube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, board_size[2] + cube_size[2] * 0.5 + 0.002)),
        spawn=sim_utils.CuboidCfg(
            size=cube_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=5.0,
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
                roughness=0.5,
                metallic=0.0,
            ),
        ),
    )

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

    wrist_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/panda_hand/wrist_camera",
        update_period=0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.05, 10.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.12, 0, -0.12),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="ros",
        ),
    )


    ik_controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls")

    episode_length_s = 0
    action_space = 5
    observation_space = 0
    state_space = 0


class CubeGraspingDemo(DirectRLEnv):
    cfg: CubeGraspingDemoCfg

    def __init__(self, cfg: CubeGraspingDemoCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )

        body_ids, _ = self._robot.find_bodies("panda_hand")
        self._body_idx = body_ids[0]
        self._jacobi_body_idx = self._body_idx - 1

        self._finger_joint_ids, _ = self._robot.find_joints(["panda_finger.*"])

        self._offset_pos = torch.tensor([0.0, 0.0, 0.11841], device=self.device).repeat(self.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        self.action_scale = 0.1
        self.processed_actions = torch.zeros((self.num_envs, self._ik_controller.action_dim), device=self.device)
        self.current_actions = torch.zeros((self.num_envs, 5), device=self.device)
        self.joint_angles = {
            "joint1": -0.4510,
            "joint2": 0.1890,
            "joint3": 0.4750,
            "joint4": -2.3660,
            "joint5": -0.2250,
            "joint6": 2.5450,
            "joint7": 0.9910,
            "finger_left": 0.02,
            "finger_right": 0.02,
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._board = RigidObject(self.cfg.board)
        self.scene.rigid_objects["board"] = self._board

        self._cube = RigidObject(self.cfg.cube)
        self.scene.rigid_objects["cube"] = self._cube

        self.wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self.scene.sensors["wrist_camera"] = self.wrist_camera

        self.scene.clone_environments(copy_from_source=False)

        ground = self.cfg.ground
        ground.spawn.func(
            ground.prim_path, ground.spawn, translation=ground.init_state.pos, orientation=ground.init_state.rot
        )

        light = self.cfg.light
        light.spawn.func(light.prim_path, light.spawn)

        for env_id in range(self.num_envs):
            try:
                cube_root = self._cube.root_physx_view.prim_paths[env_id]
            except Exception:
                cube_root = f"/World/envs/env_{env_id}/cube"
            add_random_protrusions_to_cube(
                cube_prim_path=cube_root,
                cube_size=self.cfg.cube_size,
                fixed_radius=0.0125,
                faces=("+X", "-X", "+Y", "-Y", "+Z"),
                gap=0.001,
                seed=None,
                enable_visual=True,
            )

    def _pre_physics_step(self, actions: torch.Tensor):
        self.current_actions = actions.clone()

        ee_pos_curr_b, ee_quat_curr_b = self._compute_frame_pose()

        self.processed_actions[:, :3] = self.current_actions[:, :3] * self.action_scale
        self.processed_actions[:, 3] = 0.0
        self.processed_actions[:, 4] = 0.0
        self.processed_actions[:, 5] = self.current_actions[:, 3] * self.action_scale

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

        gripper_pos = torch.clamp(self.current_actions[:, 4], 0.0, 0.04)
        gripper_pos_des = gripper_pos.unsqueeze(-1).expand(-1, len(self._finger_joint_ids))

        joint_pos_des = joint_pos.clone()
        joint_pos_des[:, :7] = arm_joint_pos_des
        joint_pos_des[:, self._finger_joint_ids] = gripper_pos_des

        self._robot.set_joint_position_target(joint_pos_des)

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._robot.data.root_link_pos_w
        root_quat_w = self._robot.data.root_link_quat_w

        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
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

        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian

    def _set_initial_joint_angles(self) -> None:
        """Apply desired initial joint angles to the Franka arm."""
        try:
            all_joint_names = self._robot.joint_names
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
                else:
                    print(f"[WARN] Joint {joint_name} ({actual_joint_name}) not found on robot.")

            self._robot.write_joint_state_to_sim(
                self._robot.data.joint_pos_target,
                self._robot.data.joint_vel,
            )
        except Exception as exc:
            print(f"[WARN] Failed to set initial joint angles: {exc}")

    def _get_observations(self) -> dict[str, torch.Tensor]:
        return {"policy": torch.zeros((self.num_envs, 1), device=self.device)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return dones, dones

    def _reset_idx(self, env_ids: torch.Tensor):
        self._robot.reset(env_ids)
        self._board.reset(env_ids)
        self._cube.reset(env_ids)


def main() -> None:
    cfg = CubeGraspingDemoCfg()
    if args_cli.num_envs is not None:
        cfg.scene.num_envs = args_cli.num_envs
    if getattr(args_cli, "device", None) is not None:
        cfg.sim.device = args_cli.device

    env = CubeGraspingDemo(cfg, render_mode=None)
    env.reset()
    env._set_initial_joint_angles()

    for _ in range(60):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    env.sim.render()

    kb_step_lin_m = 0.025
    kb_step_units = kb_step_lin_m / max(env.action_scale, 1e-6)

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_iface = carb_input.acquire_input_interface()

    pressed = {
        "up": False,
        "down": False,
        "left": False,
        "right": False,
        "forward": False,
        "backward": False,
        "open": False,
        "close": False,
    }
    gripper_pos = 0.04

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
            if event.input in (carb_input.KeyboardInput.UP, carb_input.KeyboardInput.W):
                pressed["up"] = is_down
            elif event.input in (carb_input.KeyboardInput.DOWN, carb_input.KeyboardInput.S):
                pressed["down"] = is_down
            elif event.input in (carb_input.KeyboardInput.LEFT, carb_input.KeyboardInput.A):
                pressed["left"] = is_down
            elif event.input in (carb_input.KeyboardInput.RIGHT, carb_input.KeyboardInput.D):
                pressed["right"] = is_down
            elif event.input == carb_input.KeyboardInput.E:
                pressed["forward"] = is_down
            elif event.input == carb_input.KeyboardInput.Q:
                pressed["backward"] = is_down
            elif event.input == carb_input.KeyboardInput.J and is_down:
                pressed["open"] = True
            elif event.input == carb_input.KeyboardInput.K and is_down:
                pressed["close"] = True
        return True

    kb_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_kb_event)
    print("[INFO] Keyboard ready. Click the viewport, then use:")
    print("       WASD/Arrow keys: move in Y/Z, Q/E: move in X, J/K: open/close gripper.")

    try:
        while simulation_app.is_running():
            if pressed["open"]:
                gripper_pos = 0.04
                pressed["open"] = False
            if pressed["close"]:
                gripper_pos = 0.0
                pressed["close"] = False

            dx_dir = (1.0 if pressed["forward"] else 0.0) + (-1.0 if pressed["backward"] else 0.0)
            dy_dir = (-1.0 if pressed["left"] else 0.0) + (1.0 if pressed["right"] else 0.0)
            dz_dir = (1.0 if pressed["up"] else 0.0) + (-1.0 if pressed["down"] else 0.0)

            actions = torch.zeros((env.num_envs, 5), device=env.device)
            actions[:, 0] = dx_dir * kb_step_units
            actions[:, 1] = dy_dir * kb_step_units
            actions[:, 2] = dz_dir * kb_step_units
            actions[:, 3] = 0.0
            actions[:, 4] = gripper_pos

            env._pre_physics_step(actions)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            env.sim.render()
    finally:
        try:
            input_iface.unsubscribe_from_keyboard_events(keyboard, kb_sub)
        except Exception:
            pass

        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
