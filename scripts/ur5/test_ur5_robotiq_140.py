from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


def _ensure_isaaclab_on_path():
    """Allow running the script from Isaac Sim's python without a preconfigured ISAACLAB_PATH."""
    repo_root = Path(__file__).resolve().parents[2]
    candidate_roots: list[Path] = []

    if "ISAACLAB_PATH" in os.environ:
        candidate_roots.append(Path(os.environ["ISAACLAB_PATH"]))
    candidate_roots.extend(
        [
            repo_root.parent / "IsaacLab",
            Path.home() / "IsaacLab",
            Path.home() / "IsaacLab-2.1.0",
            Path.home() / "isaaclab-2.1.0" / "IsaacLab",
            Path.home() / "isaaclab-2.1.1" / "IsaacLab",
        ]
    )

    for root in candidate_roots:
        source_dir = root / "source" / "isaaclab"
        if source_dir.is_dir() and str(source_dir) not in sys.path:
            sys.path.append(str(source_dir))
            return


_ensure_isaaclab_on_path()

import torch
from isaaclab.app import AppLauncher


ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_MASTER_PREFERRED_NAMES = [
    "finger_joint",
    "right_outer_knuckle_joint",
]
KNOWN_GRIPPER_RELATED_NAMES = [
    "finger_joint",
    "left_outer_knuckle_joint",
    "right_outer_knuckle_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_outer_finger_joint",
    "right_outer_finger_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
    "left_inner_finger_pad_joint",
    "right_inner_finger_pad_joint",
]
CPU_FALLBACK_ENV = "UR5_ROBOTIQ_VALIDATE_FORCED_CPU"


@dataclass(frozen=True)
class JointLayout:
    arm_joint_names: list[str]
    gripper_master_joint_name: str | None
    gripper_mimic_joint_names: list[str]


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    target: torch.Tensor
    command_joint_ids: list[int]
    tracking_joint_ids: list[int]


parser = argparse.ArgumentParser(description="Test the assembled UR5 + Robotiq 2F-140 USDA asset.")
parser.add_argument(
    "--robot_asset",
    type=str,
    default=str(Path(__file__).resolve().parents[2] / "ur5_robotiq_140.usda"),
    help="Path to the assembled UR5 + Robotiq 2F-140 USD/USDA asset.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="Spacing between cloned environments.")
parser.add_argument(
    "--settle_steps",
    type=int,
    default=120,
    help="Number of physics steps used for each motion phase.",
)
parser.add_argument(
    "--max_tracking_error",
    type=float,
    default=0.20,
    help="Tracking error threshold in rad used for the printed PASS/FAIL summary.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False


def _replace_or_append_device_arg(argv: list[str], device: str) -> list[str]:
    updated: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--device":
            skip_next = True
            continue
        if arg.startswith("--device="):
            continue
        updated.append(arg)
    updated.extend(["--device", device])
    return updated


def _relaunch_with_cpu(reason: Exception | str):
    print(f"[WARN] Falling back to CPU because GPU startup failed: {reason}", flush=True)
    os.environ[CPU_FALLBACK_ENV] = "1"
    new_argv = _replace_or_append_device_arg(list(sys.argv), "cpu")
    os.execv(sys.executable, [sys.executable, *new_argv])


def _resolve_requested_device():
    if getattr(args_cli, "device_explicit", False):
        return

    if torch.cuda.is_available():
        args_cli.device = "cuda:0"
        print(
            "[INFO] No --device specified. Defaulting the UR5 validation script to GPU: cuda:0.",
            flush=True,
        )
    else:
        args_cli.device = "cpu"
        print(
            "[WARN] No --device specified and CUDA is unavailable. Falling back to CPU.",
            flush=True,
        )


_resolve_requested_device()


def _resolve_safe_runtime_mode():
    if not str(args_cli.device).startswith("cuda") or args_cli.headless:
        return

    message = (
        "Isaac Sim 4.5 GUI + GPU can enable the Direct GPU API path, which rejects implicit "
        "joint drive targets for this UR5 + Robotiq validation asset."
    )
    if getattr(args_cli, "device_explicit", False):
        print(
            f"[WARN] {message} You explicitly requested GPU, so the script will continue. "
            "If PhysX reports setDriveTarget() errors, rerun with --headless or --device cpu.",
            flush=True,
        )
    else:
        args_cli.device = "cpu"
        print(
            f"[WARN] {message} Falling back to CPU for GUI validation. "
            "Use --headless to keep the default GPU path.",
            flush=True,
        )


_resolve_safe_runtime_mode()


def _launch_app():
    try:
        app_launcher = AppLauncher(args_cli)
    except Exception as exc:
        if (
            not getattr(args_cli, "device_explicit", False)
            and str(args_cli.device).startswith("cuda")
            and os.environ.get(CPU_FALLBACK_ENV) != "1"
        ):
            _relaunch_with_cpu(exc)
        raise
    return app_launcher, app_launcher.app


app_launcher, simulation_app = _launch_app()

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


def _guess_gripper_master_joint_name_from_asset(asset_path: Path) -> str | None:
    if asset_path.suffix.lower() != ".usda":
        return None

    try:
        asset_text = asset_path.read_text(encoding="utf-8")
    except OSError:
        return None

    for joint_name in GRIPPER_MASTER_PREFERRED_NAMES:
        if f'"{joint_name}"' in asset_text:
            return joint_name

    # Best-effort fallback for textual USDA assets with non-standard master-joint names.
    for match in re.finditer(r'over "([^"]+)"', asset_text):
        joint_name = match.group(1)
        name_lower = joint_name.lower()
        if "finger" in name_lower and "joint" in name_lower:
            return joint_name
    for match in re.finditer(r'over "([^"]+)"', asset_text):
        joint_name = match.group(1)
        name_lower = joint_name.lower()
        if "knuckle" in name_lower and "joint" in name_lower:
            return joint_name
    return None


PRECONFIGURED_GRIPPER_MASTER_JOINT_NAME = _guess_gripper_master_joint_name_from_asset(
    Path(args_cli.robot_asset).expanduser().resolve()
)


def _make_robot_cfg(asset_path: Path) -> ArticulationCfg:
    actuators: dict[str, ImplicitActuatorCfg] = {
        "arm": ImplicitActuatorCfg(
            joint_names_expr=ARM_JOINT_NAMES,
            effort_limit_sim=150.0,
            velocity_limit_sim=3.2,
            stiffness=400.0,
            damping=40.0,
        ),
    }

    if PRECONFIGURED_GRIPPER_MASTER_JOINT_NAME is not None:
        # Robotiq mimic/passive joints follow the master drive joint through the asset's linkage.
        # Driving only the master joint keeps the command semantics aligned with the imported gripper.
        actuators["gripper_master"] = ImplicitActuatorCfg(
            joint_names_expr=[PRECONFIGURED_GRIPPER_MASTER_JOINT_NAME],
            effort_limit_sim=1000.0,
            velocity_limit_sim=2.0,
            stiffness=40.0,
            damping=4.0,
        )

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators=actuators,
        soft_joint_pos_limit_factor=0.95,
    )


@configclass
class UR5RobotiqTestSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=3000.0),
    )

    robot: ArticulationCfg = _make_robot_cfg(Path(args_cli.robot_asset).expanduser().resolve())


def _joint_limits(robot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    limits = robot.data.soft_joint_pos_limits[0]
    lower = limits[:, 0].clone()
    upper = limits[:, 1].clone()
    valid = torch.isfinite(lower) & torch.isfinite(upper) & (upper > lower)
    return lower, upper, valid


def _clamp_pose(robot, pose: torch.Tensor) -> torch.Tensor:
    lower, upper, valid = _joint_limits(robot)
    clamped_pose = pose.clone()
    if torch.any(valid):
        valid_lower = lower[valid].unsqueeze(0)
        valid_upper = upper[valid].unsqueeze(0)
        clamped_pose[:, valid] = torch.clamp(clamped_pose[:, valid], min=valid_lower, max=valid_upper)
    return clamped_pose


def _print_summary(robot):
    print(f"[INFO] Joint names ({len(robot.joint_names)}): {robot.joint_names}", flush=True)
    print(f"[INFO] Body names ({len(robot.body_names)}): {robot.body_names}", flush=True)
    print(f"[INFO] Fixed base: {robot.is_fixed_base}", flush=True)


def _collect_arm_joint_names(robot) -> list[str]:
    names = [name for name in ARM_JOINT_NAMES if name in robot.joint_names]
    if names:
        return names

    fallback_patterns = [
        ".*shoulder.*",
        ".*elbow.*",
        ".*wrist.*",
    ]
    for pattern in fallback_patterns:
        _, matched_names = robot.find_joints(pattern)
        for name in matched_names:
            if name not in names:
                names.append(name)
    return names


def _discover_joint_layout(robot) -> JointLayout:
    arm_joint_names = _collect_arm_joint_names(robot)

    gripper_related_joint_names: list[str] = []
    for joint_name in KNOWN_GRIPPER_RELATED_NAMES:
        if joint_name in robot.joint_names and joint_name not in arm_joint_names:
            gripper_related_joint_names.append(joint_name)

    for joint_name in robot.joint_names:
        if joint_name in arm_joint_names or joint_name in gripper_related_joint_names:
            continue
        joint_name_lower = joint_name.lower()
        if any(token in joint_name_lower for token in ("finger", "knuckle", "gripper")):
            gripper_related_joint_names.append(joint_name)

    gripper_master_joint_name = next(
        (joint_name for joint_name in GRIPPER_MASTER_PREFERRED_NAMES if joint_name in gripper_related_joint_names),
        None,
    )

    if gripper_master_joint_name is None:
        careful_fallback_patterns = [
            re.compile(r"finger.*joint", re.IGNORECASE),
            re.compile(r"gripper.*joint", re.IGNORECASE),
            re.compile(r"outer.*knuckle.*joint", re.IGNORECASE),
            re.compile(r"knuckle.*joint", re.IGNORECASE),
        ]
        for pattern in careful_fallback_patterns:
            for joint_name in gripper_related_joint_names:
                if pattern.search(joint_name):
                    gripper_master_joint_name = joint_name
                    break
            if gripper_master_joint_name is not None:
                break

    gripper_mimic_joint_names = [
        joint_name for joint_name in gripper_related_joint_names if joint_name != gripper_master_joint_name
    ]
    return JointLayout(
        arm_joint_names=arm_joint_names,
        gripper_master_joint_name=gripper_master_joint_name,
        gripper_mimic_joint_names=gripper_mimic_joint_names,
    )


def _make_pose(robot, base_pose: torch.Tensor, updates: dict[str, float]) -> torch.Tensor:
    pose = base_pose.clone()
    name_to_id = {name: idx for idx, name in enumerate(robot.joint_names)}
    for joint_name, value in updates.items():
        if joint_name in name_to_id:
            pose[:, name_to_id[joint_name]] = value
    return _clamp_pose(robot, pose)


def _make_gripper_pose(robot, base_pose: torch.Tensor, master_joint_name: str | None, close: bool) -> torch.Tensor:
    pose = base_pose.clone()
    if master_joint_name is None:
        return pose

    name_to_id = {name: idx for idx, name in enumerate(robot.joint_names)}
    joint_id = name_to_id[master_joint_name]
    lower, upper, valid = _joint_limits(robot)

    if valid[joint_id]:
        joint_range = upper[joint_id] - lower[joint_id]
        margin = 0.15 * joint_range
        pose[:, joint_id] = upper[joint_id] - margin if close else lower[joint_id] + margin
        return _clamp_pose(robot, pose)

    default_value = base_pose[:, joint_id]
    offset = torch.full_like(default_value, 0.05)
    if close:
        pose[:, joint_id] = default_value + offset
    else:
        pose[:, joint_id] = default_value - offset
    print(
        "[WARN] Gripper master joint has invalid soft limits. "
        "Using conservative relative offsets around the default position.",
        flush=True,
    )
    return pose


def _phase_error(robot, target: torch.Tensor, joint_ids: list[int]) -> float:
    if not joint_ids:
        return 0.0
    current = robot.data.joint_pos[:, joint_ids]
    desired = target[:, joint_ids]
    return torch.max(torch.abs(current - desired)).item()


def _run_phase(
    sim,
    scene,
    robot,
    phase: PhaseSpec,
    num_steps: int,
) -> float:
    sim_dt = sim.get_physics_dt()
    for _ in range(num_steps):
        if phase.command_joint_ids:
            robot.set_joint_position_target(phase.target[:, phase.command_joint_ids], joint_ids=phase.command_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
    error = _phase_error(robot, phase.target, phase.tracking_joint_ids)
    tracked_names = [robot.joint_names[joint_id] for joint_id in phase.tracking_joint_ids]
    commanded_names = [robot.joint_names[joint_id] for joint_id in phase.command_joint_ids]
    print(
        f"[PHASE] {phase.name}: commanded_joints={commanded_names}, "
        f"tracked_joints={tracked_names}, max_error={error:.4f} rad",
        flush=True,
    )
    return error


def _log_joint_layout(layout: JointLayout, active_actuated_joint_names: list[str]):
    print(f"[INFO] Detected arm joints: {layout.arm_joint_names}", flush=True)
    print(f"[INFO] Detected gripper master joint: {layout.gripper_master_joint_name}", flush=True)
    print(f"[INFO] Detected mimic/passive gripper joints: {layout.gripper_mimic_joint_names}", flush=True)
    print(f"[INFO] Active actuator joints: {active_actuated_joint_names}", flush=True)
    print(f"[INFO] Arm phase tracking joints: {layout.arm_joint_names}", flush=True)
    if layout.gripper_master_joint_name is not None:
        print(f"[INFO] Gripper phase tracking joint: {[layout.gripper_master_joint_name]}", flush=True)


def main():
    asset_path = Path(args_cli.robot_asset).expanduser().resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Robot asset not found: {asset_path}")

    print(f"[INFO] Preparing simulation for asset: {asset_path}", flush=True)
    sim_cfg = sim_utils.SimulationCfg(
        dt=1 / 120,
        device=args_cli.device,
        use_fabric=not str(args_cli.device).startswith("cpu"),
    )
    try:
        sim = sim_utils.SimulationContext(sim_cfg)
    except Exception as exc:
        if (
            not getattr(args_cli, "device_explicit", False)
            and str(args_cli.device).startswith("cuda")
            and os.environ.get(CPU_FALLBACK_ENV) != "1"
        ):
            _relaunch_with_cpu(exc)
        raise

    sim.set_camera_view([2.8, 2.2, 1.8], [0.0, 0.0, 0.45])

    scene_cfg = UR5RobotiqTestSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    scene_cfg.robot = _make_robot_cfg(asset_path)
    print("[INFO] Creating interactive scene...", flush=True)
    scene = InteractiveScene(scene_cfg)

    print("[INFO] Resetting simulation...", flush=True)
    sim.reset()
    robot = scene["robot"]

    print(f"[INFO] Loaded asset: {asset_path}", flush=True)
    print(f"[INFO] Robot prim path expression: {robot.cfg.prim_path}", flush=True)
    print(f"[INFO] Resolved runtime device: {args_cli.device}", flush=True)
    print(f"[INFO] use_fabric={sim_cfg.use_fabric}", flush=True)
    _print_summary(robot)

    layout = _discover_joint_layout(robot)
    name_to_id = {name: idx for idx, name in enumerate(robot.joint_names)}
    arm_joint_ids = [name_to_id[name] for name in layout.arm_joint_names]
    gripper_master_joint_id = (
        name_to_id[layout.gripper_master_joint_name] if layout.gripper_master_joint_name is not None else None
    )

    if not layout.arm_joint_names:
        print("[WARN] No UR5 arm joints were matched by name. The robot asset may use different joint names.", flush=True)

    if (
        layout.gripper_master_joint_name is not None
        and PRECONFIGURED_GRIPPER_MASTER_JOINT_NAME is not None
        and layout.gripper_master_joint_name != PRECONFIGURED_GRIPPER_MASTER_JOINT_NAME
    ):
        print(
            "[WARN] Runtime gripper master joint does not match the actuator chosen before asset load. "
            f"Configured={PRECONFIGURED_GRIPPER_MASTER_JOINT_NAME}, detected={layout.gripper_master_joint_name}",
            flush=True,
        )
    elif layout.gripper_master_joint_name is not None and PRECONFIGURED_GRIPPER_MASTER_JOINT_NAME is None:
        print(
            "[WARN] Detected a gripper master joint at runtime but could not preconfigure a dedicated "
            "gripper actuator before load. Gripper motion may be limited.",
            flush=True,
        )

    active_actuated_joint_names = list(layout.arm_joint_names)
    if (
        layout.gripper_master_joint_name is not None
        and layout.gripper_master_joint_name == PRECONFIGURED_GRIPPER_MASTER_JOINT_NAME
    ):
        active_actuated_joint_names.append(layout.gripper_master_joint_name)
    _log_joint_layout(layout, active_actuated_joint_names)
    print(
        "[INFO] Passive gripper joints are intentionally left without actuators. "
        "Isaac Lab may warn that only 7 of 14 joints are actuated; this is expected here.",
        flush=True,
    )

    home_updates = {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.712,
        "elbow_joint": 1.712,
        "wrist_1_joint": 0.0,
        "wrist_2_joint": 0.0,
        "wrist_3_joint": 0.0,
    }
    if layout.gripper_master_joint_name is not None:
        home_updates[layout.gripper_master_joint_name] = float(
            robot.data.default_joint_pos[0, name_to_id[layout.gripper_master_joint_name]].item()
        )

    home_pose = _make_pose(robot, robot.data.default_joint_pos.clone(), home_updates)
    robot.write_joint_state_to_sim(home_pose, torch.zeros_like(home_pose))
    robot.reset()

    gripper_open_pose = _make_gripper_pose(robot, home_pose, layout.gripper_master_joint_name, close=False)
    gripper_closed_pose = _make_gripper_pose(robot, home_pose, layout.gripper_master_joint_name, close=True)

    arm_pose_a = _make_pose(
        robot,
        home_pose,
        {
            "shoulder_pan_joint": 0.60,
            "shoulder_lift_joint": -1.45,
            "elbow_joint": 1.40,
            "wrist_1_joint": -0.35,
        },
    )
    arm_pose_b = _make_pose(
        robot,
        home_pose,
        {
            "shoulder_pan_joint": -0.60,
            "shoulder_lift_joint": -1.95,
            "elbow_joint": 1.95,
            "wrist_1_joint": 0.35,
        },
    )
    wrist_pose = _make_pose(
        robot,
        home_pose,
        {
            "wrist_1_joint": 0.70,
            "wrist_2_joint": -0.55,
            "wrist_3_joint": 0.55,
        },
    )

    gripper_tracking_joint_ids = [gripper_master_joint_id] if gripper_master_joint_id is not None else []
    return_home_command_joint_ids = list(arm_joint_ids)
    if (
        gripper_master_joint_id is not None
        and layout.gripper_master_joint_name == PRECONFIGURED_GRIPPER_MASTER_JOINT_NAME
    ):
        return_home_command_joint_ids.append(gripper_master_joint_id)

    phases: list[PhaseSpec] = [
        PhaseSpec("Home", home_pose, arm_joint_ids, arm_joint_ids),
        PhaseSpec("Arm Pose A", arm_pose_a, arm_joint_ids, arm_joint_ids),
        PhaseSpec("Arm Pose B", arm_pose_b, arm_joint_ids, arm_joint_ids),
        PhaseSpec("Wrist Pose", wrist_pose, arm_joint_ids, arm_joint_ids),
    ]
    if gripper_tracking_joint_ids:
        phases.extend(
            [
                PhaseSpec("Gripper Open", gripper_open_pose, gripper_tracking_joint_ids, gripper_tracking_joint_ids),
                PhaseSpec("Gripper Close", gripper_closed_pose, gripper_tracking_joint_ids, gripper_tracking_joint_ids),
                PhaseSpec("Gripper Reopen", gripper_open_pose, gripper_tracking_joint_ids, gripper_tracking_joint_ids),
            ]
        )
    else:
        print("[WARN] No gripper master joint detected. The script will only test UR5 arm motion.", flush=True)
    phases.append(PhaseSpec("Return Home", home_pose, return_home_command_joint_ids, arm_joint_ids))

    errors: list[tuple[str, float]] = []
    for phase in phases:
        errors.append((phase.name, _run_phase(sim, scene, robot, phase, args_cli.settle_steps)))

    max_error = max((error for _, error in errors), default=0.0)
    threshold = args_cli.max_tracking_error
    passed = max_error <= threshold
    print(f"[RESULT] Max phase tracking error: {max_error:.4f} rad (threshold={threshold:.4f})", flush=True)
    print(
        f"[RESULT] {'PASS' if passed else 'WARN'}: "
        f"{'robot and gripper responded as expected' if passed else 'tracking error is larger than expected'}",
        flush=True,
    )

    if args_cli.headless:
        return

    print("[INFO] Test sequence finished. Keeping the last pose until the app is closed.", flush=True)
    hold_joint_ids = return_home_command_joint_ids
    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        if hold_joint_ids:
            robot.set_joint_position_target(home_pose[:, hold_joint_ids], joint_ids=hold_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    if args_cli.headless:
        # Isaac Sim 4.5 can hang or abort on close() in short-lived headless utility scripts.
        # This test script is a standalone process, so exiting directly is the most reliable path.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    simulation_app.close()
