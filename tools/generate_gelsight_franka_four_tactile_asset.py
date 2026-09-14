"""Generate the V8 Franka asset by grafting the legacy downward GelSight pair onto V7."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)

from pxr import Gf, Sdf, Usd  # noqa: E402


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


def _translation(stage: Usd.Stage, path: str) -> tuple[float, float, float]:
    value = stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Get()
    if value is None:
        raise RuntimeError(f"Missing xformOp:translate at {path}")
    return tuple(float(v) for v in value)


def _set_translation(stage: Usd.Stage, path: str, value: tuple[float, float, float]) -> None:
    attr = stage.GetPrimAtPath(path).GetAttribute("xformOp:translate")
    if not attr.Set(Gf.Vec3d(*value)):
        raise RuntimeError(f"Failed to update {path}.xformOp:translate")


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
