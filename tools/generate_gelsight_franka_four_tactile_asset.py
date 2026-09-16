"""Generate V8 by grafting and rotating the complete downward GelSight pair."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


_ROOT = Path(__file__).resolve().parents[1]
_ASSETS = _ROOT / "source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper"
_BASE = _ASSETS / "franka_gsmini_standard_arm_visuals_v7.usd"
_DONOR = _ASSETS / "physx_rigid_gelpads.down.usda"
_OUTPUT = _ASSETS / "franka_gsmini_four_tactile_standard_arm_visuals_v8.usd"
_COPIES = (
    ("/panda/gelpad_left_01", "/panda/gelpad_left"),
    ("/panda/gelpad_right_01", "/panda/gelpad_right"),
    ("/panda/gelsight_mini_case_left_down", "/panda/gelsight_mini_case_left"),
    ("/panda/gelsight_mini_case_right_down", "/panda/gelsight_mini_case_right"),
)
_DOWN_ASSEMBLIES = (
    (
        "/panda/gelsight_mini_case_left_down",
        "/panda/gelpad_left_01",
        "/panda/gelsight_mini_case_left_down/FixedJointLeft2",
    ),
    (
        "/panda/gelsight_mini_case_right_down",
        "/panda/gelpad_right_01",
        "/panda/gelsight_mini_case_right_down/FixedJointRight2",
    ),
)
# The donor case has an existing 180-degree X rotation. In the Omniverse UI,
# this parent-frame rotation changes the displayed Euler Z from -135 to -45.
_DOWN_ASSEMBLY_PARENT_YAW_DEG = -90.0


def _translation(stage: Usd.Stage, path: str) -> tuple[float, float, float]:
    value = stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Get()
    if value is None:
        raise RuntimeError(f"Missing xformOp:translate at {path}")
    return tuple(float(v) for v in value)


def _set_translation(stage: Usd.Stage, path: str, value: tuple[float, float, float]) -> None:
    attr = stage.GetPrimAtPath(path).GetAttribute("xformOp:translate")
    if not attr.Set(Gf.Vec3d(*value)):
        raise RuntimeError(f"Failed to update {path}.xformOp:translate")


def _orientation(stage: Usd.Stage, path: str) -> Gf.Quatd:
    value = stage.GetPrimAtPath(path).GetAttribute("xformOp:orient").Get()
    if value is None:
        raise RuntimeError(f"Missing xformOp:orient at {path}")
    return Gf.Quatd(float(value.GetReal()), Gf.Vec3d(*[float(v) for v in value.GetImaginary()]))


def _set_orientation(stage: Usd.Stage, path: str, value: Gf.Quatd) -> None:
    value.Normalize()
    imaginary = value.GetImaginary()
    attr = stage.GetPrimAtPath(path).GetAttribute("xformOp:orient")
    if not attr.Set(
        Gf.Quatf(
            float(value.GetReal()),
            Gf.Vec3f(float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
        )
    ):
        raise RuntimeError(f"Failed to update {path}.xformOp:orient")


def _quatd(value: Gf.Quatf | Gf.Quatd) -> Gf.Quatd:
    return Gf.Quatd(float(value.GetReal()), Gf.Vec3d(*[float(v) for v in value.GetImaginary()]))


def _rotate_down_assembly(
    stage: Usd.Stage,
    case_path: str,
    gelpad_path: str,
    mounting_joint_path: str,
) -> None:
    """Rotate one complete sensor around its case origin in the panda parent frame."""
    half_angle = 0.5 * _DOWN_ASSEMBLY_PARENT_YAW_DEG
    yaw = Gf.Quatd(
        math.cos(math.radians(half_angle)),
        Gf.Vec3d(0.0, 0.0, math.sin(math.radians(half_angle))),
    )
    rotation = Gf.Rotation(yaw)
    pivot = Gf.Vec3d(*_translation(stage, case_path))

    for body_path in (case_path, gelpad_path):
        old_translation = Gf.Vec3d(*_translation(stage, body_path))
        new_translation = pivot + rotation.TransformDir(old_translation - pivot)
        _set_translation(stage, body_path, tuple(new_translation))
        _set_orientation(stage, body_path, yaw * _orientation(stage, body_path))

    # FixedJoint1 joins the case and downward GelPad; both bodies receive the
    # same rigid transform, so its two local frames remain valid. FixedJoint2
    # joins the rotated case to the unchanged inner GelPad, so only the case-
    # side local rotation must be recomputed to preserve the world anchor.
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(mounting_joint_path))
    body0_targets = joint.GetBody0Rel().GetTargets()
    body1_targets = joint.GetBody1Rel().GetTargets()
    if len(body0_targets) != 1 or len(body1_targets) != 1:
        raise RuntimeError(f"Expected two resolved bodies at {mounting_joint_path}")
    xforms = UsdGeom.XformCache()
    body0_world = xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(body0_targets[0]))
    body1_world = xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(body1_targets[0]))
    anchor_world = body0_world.ExtractRotationQuat() * _quatd(joint.GetLocalRot0Attr().Get())
    local_rot1 = body1_world.ExtractRotationQuat().GetInverse() * anchor_world
    local_rot1.Normalize()
    imaginary = local_rot1.GetImaginary()
    if not joint.GetLocalRot1Attr().Set(
        Gf.Quatf(
            float(local_rot1.GetReal()),
            Gf.Vec3f(float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
        )
    ):
        raise RuntimeError(f"Failed to reframe {mounting_joint_path}.physics:localRot1")


def generate(output: Path) -> None:
    base = Usd.Stage.Open(str(_BASE))
    donor = Usd.Stage.Open(str(_DONOR))
    if base is None or donor is None:
        raise RuntimeError("Unable to open V7 base or downward-sensor donor USD")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not base.GetRootLayer().Export(str(output)):
        raise RuntimeError(f"Unable to export V8 base to {output}")
    generated = Usd.Stage.Open(str(output))
    assert generated is not None

    for source_path, inner_path in _COPIES:
        if not Sdf.CopySpec(
            donor.GetRootLayer(), source_path, generated.GetRootLayer(), source_path
        ):
            raise RuntimeError(f"Unable to copy donor prim {source_path}")
        donor_down = _translation(donor, source_path)
        donor_inner = _translation(donor, inner_path)
        target_inner = _translation(generated, inner_path)
        _set_translation(
            generated,
            source_path,
            tuple(donor_down[i] + target_inner[i] - donor_inner[i] for i in range(3)),
        )

    for case_path, gelpad_path, mounting_joint_path in _DOWN_ASSEMBLIES:
        _rotate_down_assembly(generated, case_path, gelpad_path, mounting_joint_path)

    generated.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(output))
    assert reopened is not None
    required = tuple(path for path, _ in _COPIES)
    required += (
        "/panda/gelsight_mini_case_left_down/FixedJointLeft1",
        "/panda/gelsight_mini_case_left_down/FixedJointLeft2",
        "/panda/gelsight_mini_case_right_down/FixedJointRight1",
        "/panda/gelsight_mini_case_right_down/FixedJointRight2",
    )
    missing = [path for path in required if not reopened.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"Generated V8 is missing prims: {missing}")
    for prim in base.Traverse():
        path = str(prim.GetPath())
        for name in ("xformOp:translate", "xformOp:orient", "xformOp:scale"):
            expected = prim.GetAttribute(name).Get()
            if expected is not None and reopened.GetPrimAtPath(path).GetAttribute(name).Get() != expected:
                raise RuntimeError(f"V7 transform changed in V8: {path}.{name}")
    print(f"Generated four-tactile V8 asset: {output}")


try:
    generate((args.output or _OUTPUT).resolve())
finally:
    app_launcher.app.close()
