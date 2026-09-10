"""Generate the persistent Franka/GelSight asset with a shifted finger assembly.

Run through ``./tacex.sh -p`` so that the Isaac Sim USD Python bindings are
available.  The v5 asset is the immutable zero-extension reference; this
script moves the complete finger/GelSight rigid-body assembly and repairs the
three hand-level joint anchors by the same amount.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension_m",
        type=float,
        default=0.021,
        help="Finger/GelSight extension from the v5 reference along hand local +Z.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output USD path (default: the repository v7 asset path).",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args = _parse_args()
app_launcher = AppLauncher(args)

from pxr import Gf, Sdf, Usd  # noqa: E402


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ASSET_DIRECTORY = (
    _REPOSITORY_ROOT
    / "source/tacex_assets/tacex_assets/data/Robots/Franka/GelSight_Mini/Gripper"
)
_REFERENCE_USD = _ASSET_DIRECTORY / "franka_gsmini_standard_arm_visuals_v5.usd"
_DEFAULT_OUTPUT_USD = _ASSET_DIRECTORY / "franka_gsmini_standard_arm_visuals_v7.usd"

_SHIFTED_BODY_PATHS = (
    "/panda/panda_leftfinger",
    "/panda/panda_rightfinger",
    "/panda/gelsight_mini_case_left",
    "/panda/gelsight_mini_case_right",
    "/panda/gelpad_left",
    "/panda/gelpad_right",
    "/panda/panda_fingertip_centered",
)
_FINGER_JOINT_PATHS = (
    "/panda/panda_hand/panda_finger_joint1",
    "/panda/panda_hand/panda_finger_joint2",
)
_CENTERED_JOINT_PATH = "/panda/panda_hand/panda_fingertip_centered"


def _set_z(attribute: Usd.Attribute, z: float) -> None:
    value = attribute.Get()
    if value is None:
        raise RuntimeError(f"Missing value for USD attribute: {attribute.GetPath()}")
    type_name = attribute.GetTypeName()
    if type_name in (Sdf.ValueTypeNames.Float3, Sdf.ValueTypeNames.Point3f):
        updated = Gf.Vec3f(float(value[0]), float(value[1]), float(z))
    elif type_name in (Sdf.ValueTypeNames.Double3, Sdf.ValueTypeNames.Point3d):
        updated = Gf.Vec3d(float(value[0]), float(value[1]), float(z))
    else:
        raise TypeError(f"Unsupported vector type {type_name} at {attribute.GetPath()}")
    if not attribute.Set(updated):
        raise RuntimeError(f"Failed to write USD attribute: {attribute.GetPath()}")


def _attribute(stage: Usd.Stage, prim_path: str, name: str) -> Usd.Attribute:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing USD prim: {prim_path}")
    attribute = prim.GetAttribute(name)
    if not attribute.IsValid():
        raise RuntimeError(f"Missing USD attribute: {prim_path}.{name}")
    return attribute


def _vector(stage: Usd.Stage, prim_path: str, name: str) -> tuple[float, float, float]:
    value = _attribute(stage, prim_path, name).Get()
    return tuple(float(component) for component in value)


def _verify(output: Path, extension_m: float) -> None:
    reference_stage = Usd.Stage.Open(str(_REFERENCE_USD))
    generated_stage = Usd.Stage.Open(str(output))
    if reference_stage is None or generated_stage is None:
        raise RuntimeError("Failed to open USD layers for generated-asset verification")

    tolerance = 1.0e-7
    for prim_path in ("/panda/panda_link8", "/panda/panda_hand"):
        reference = _vector(reference_stage, prim_path, "xformOp:translate")
        generated = _vector(generated_stage, prim_path, "xformOp:translate")
        if any(abs(actual - expected) > tolerance for actual, expected in zip(generated, reference)):
            raise RuntimeError(f"Stationary body moved during generation: {prim_path}")
    for prim_path in _SHIFTED_BODY_PATHS:
        reference = _vector(reference_stage, prim_path, "xformOp:translate")
        generated = _vector(generated_stage, prim_path, "xformOp:translate")
        expected = (reference[0], reference[1], reference[2] - extension_m)
        if any(abs(actual - target) > tolerance for actual, target in zip(generated, expected)):
            raise RuntimeError(f"Incorrect generated body transform: {prim_path}")
    for prim_path in _FINGER_JOINT_PATHS:
        reference = _vector(reference_stage, prim_path, "physics:localPos0")
        generated = _vector(generated_stage, prim_path, "physics:localPos0")
        expected = (reference[0], reference[1], reference[2] + extension_m)
        if any(abs(actual - target) > tolerance for actual, target in zip(generated, expected)):
            raise RuntimeError(f"Incorrect generated finger joint anchor: {prim_path}")
    reference = _vector(reference_stage, _CENTERED_JOINT_PATH, "physics:localPos0")
    generated = _vector(generated_stage, _CENTERED_JOINT_PATH, "physics:localPos0")
    expected = (reference[0], reference[1], reference[2] - extension_m)
    if any(abs(actual - target) > tolerance for actual, target in zip(generated, expected)):
        raise RuntimeError("Incorrect generated centered-fingertip joint anchor")


def generate(output: Path, extension_m: float) -> None:
    if not 0.0 <= extension_m <= 0.10:
        raise ValueError(f"Unreasonable finger extension: {extension_m} m")
    reference_stage = Usd.Stage.Open(str(_REFERENCE_USD))
    if reference_stage is None:
        raise RuntimeError(f"Failed to open reference USD: {_REFERENCE_USD}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not reference_stage.GetRootLayer().Export(str(output)):
        raise RuntimeError(f"Failed to export reference layer to: {output}")

    stage = Usd.Stage.Open(str(output))
    if stage is None:
        raise RuntimeError(f"Failed to reopen generated USD: {output}")

    # Body roots move toward hand local +Z in the semantic gripper frame.  In
    # this USD's root authoring convention that is a negative translate-Z.
    for prim_path in _SHIFTED_BODY_PATHS:
        attribute = _attribute(stage, prim_path, "xformOp:translate")
        reference_z = float(attribute.Get()[2])
        _set_z(attribute, reference_z - extension_m)

    # The two finger joints are parented by panda_hand, so localPos0 follows
    # the child bodies. The centered fixed joint stores the inverse anchor.
    for prim_path in _FINGER_JOINT_PATHS:
        attribute = _attribute(stage, prim_path, "physics:localPos0")
        reference_z = float(attribute.Get()[2])
        _set_z(attribute, reference_z + extension_m)
    centered_attribute = _attribute(stage, _CENTERED_JOINT_PATH, "physics:localPos0")
    centered_reference_z = float(centered_attribute.Get()[2])
    _set_z(centered_attribute, centered_reference_z - extension_m)

    stage.GetRootLayer().Save()
    _verify(output, extension_m)
    print(f"Generated {output} with finger/GelSight extension {extension_m:.6f} m")


try:
    generate((args.output or _DEFAULT_OUTPUT_USD).resolve(), args.extension_m)
finally:
    app_launcher.app.close()
