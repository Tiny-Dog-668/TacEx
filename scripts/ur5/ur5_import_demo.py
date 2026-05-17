from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Import a UR5/UR5e robot asset into Isaac Lab for TacEx.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="Spacing between cloned environments.")
parser.add_argument(
    "--robot_asset",
    type=str,
    default="/home/tinydog/Projects/TacEx/ur5e_gripper/ur5e_gripper.usd",
    help="Absolute path to the UR5 asset (.urdf, .usd, or .usda).",
)
parser.add_argument(
    "--usd_cache_dir",
    type=str,
    default="/home/tinydog/Projects/TacEx/.cache/urdf/ur5_robotiq",
    help="Directory used to cache the USD converted from URDF assets.",
)
parser.add_argument(
    "--fix_base",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether to fix the UR5 base to the world when importing a URDF.",
)
parser.add_argument(
    "--force_usd_conversion",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to force URDF-to-USD conversion even if a cached USD already exists.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Maximum number of simulation steps to run. Use 0 to keep the app running until closed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaacsim.core.utils.extensions import enable_extension

from tacex_assets.robots.ur5.ur5_robotiq import UR5_ROBOTIQ_CFG
from tacex_assets.robots.ur5.ur5e_gripper_usd import UR5E_GRIPPER_USD_CFG


def _get_base_robot_cfg(asset_path: Path) -> ArticulationCfg:
    if asset_path.suffix.lower() in {".usd", ".usda", ".usdc"}:
        return UR5E_GRIPPER_USD_CFG
    return UR5_ROBOTIQ_CFG


def _build_robot_spawn_cfg(asset_path: Path):
    base_cfg = _get_base_robot_cfg(asset_path)
    asset_suffix = asset_path.suffix.lower()
    if asset_suffix == ".urdf":
        return sim_utils.UrdfFileCfg(
            asset_path=str(asset_path),
            usd_dir=args_cli.usd_cache_dir,
            fix_base=args_cli.fix_base,
            merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=False,
            force_usd_conversion=args_cli.force_usd_conversion,
            make_instanceable=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
            ),
            rigid_props=base_cfg.spawn.rigid_props,
            articulation_props=base_cfg.spawn.articulation_props,
        )
    if asset_suffix in {".usd", ".usda", ".usdc"}:
        return sim_utils.UsdFileCfg(
            usd_path=str(asset_path),
            rigid_props=base_cfg.spawn.rigid_props,
            articulation_props=base_cfg.spawn.articulation_props,
        )
    raise ValueError(f"Unsupported asset suffix '{asset_suffix}'. Expected .urdf, .usd, .usda, or .usdc.")


def _build_robot_cfg(asset_path: Path) -> ArticulationCfg:
    base_cfg = _get_base_robot_cfg(asset_path)
    return base_cfg.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=_build_robot_spawn_cfg(asset_path),
    )


def _print_robot_summary(robot):
    print(f"[INFO]: Joint names ({len(robot.joint_names)}): {robot.joint_names}")
    print(f"[INFO]: Body names ({len(robot.body_names)}): {robot.body_names}")
    for candidate in ("ee_link", "tool0", "wrist_3_link", "left_inner_finger"):
        if candidate in robot.body_names:
            print(f"[INFO]: Suggested end-effector body: {candidate}")
            break


@configclass
class UR5ImportSceneCfg(InteractiveSceneCfg):
    """Minimal scene used to verify UR5 import."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot: ArticulationCfg = UR5E_GRIPPER_USD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    asset_path = Path(args_cli.robot_asset).expanduser().resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"UR5 asset not found: {asset_path}")
    if asset_path.suffix.lower() == ".urdf":
        Path(args_cli.usd_cache_dir).mkdir(parents=True, exist_ok=True)

    enable_extension("isaacsim.asset.importer.urdf")

    sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 1.8], [0.0, 0.0, 0.5])

    scene_cfg = UR5ImportSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene_cfg.robot = _build_robot_cfg(asset_path)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    robot = scene["robot"]
    controlled_joint_ids = list(range(len(robot.joint_names)))
    controlled_joint_names = list(robot.joint_names)
    print(f"[INFO]: Loaded asset: {asset_path}")
    print(f"[INFO]: Robot prim path: {robot.cfg.prim_path}")
    _print_robot_summary(robot)
    print(f"[INFO]: Controlled joints: {controlled_joint_names}")

    sim_dt = sim.get_physics_dt()
    step_count = 0

    while simulation_app.is_running():
        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break
        if step_count % 240 == 0:
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

        robot.set_joint_position_target(robot.data.default_joint_pos[:, controlled_joint_ids], joint_ids=controlled_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
