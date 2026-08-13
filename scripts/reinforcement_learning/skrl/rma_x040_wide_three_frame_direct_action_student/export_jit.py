"""Export an X040-Wide three-frame Student with its strict runtime signature."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / "source" / "tacex_tasks"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--student_checkpoint", required=True)
parser.add_argument("--output", default=None)
parser.add_argument(
    "--allow_legacy_appearance_evaluation_export",
    action="store_true",
    help=(
        "Export a strictly validated legacy Appearance v1 checkpoint for "
        "evaluation/rollout only; the metadata retains its original visual contract"
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch

from tacex_tasks.sim2real_grasp.rma_x040_wide_three_frame_artifacts import (
    load_three_frame_student_checkpoint,
    load_three_frame_student_model_state,
    sha256_file,
    state_dict_sha256,
)
from tacex_tasks.sim2real_grasp.rma_x040_wide_three_frame_models import (
    RMAX040WideThreeFrameDirectActionVisualStudent,
)


def _atomic_json(value: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    payload = load_three_frame_student_checkpoint(
        checkpoint,
        device="cpu",
        allow_appearance_v1_evaluation=(
            args.allow_legacy_appearance_evaluation_export
        ),
    )
    model = RMAX040WideThreeFrameDirectActionVisualStudent().eval()
    load_three_frame_student_model_state(model, payload["model"])
    if (
        state_dict_sha256(model.vision_encoder.state_dict())
        != payload["vision_encoder_state_dict_sha256"]
    ):
        raise RuntimeError("Vision encoder hash mismatch")

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint.parent
        / "exported"
        / f"rma_x040_wide_three_frame_student_{checkpoint.stem}.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    probe = (
        torch.randint(0, 256, (2, 3, 224, 224, 3), dtype=torch.uint8),
        torch.randn(2, 15),
        torch.randn(2, 4) * 0.01,
    )
    with torch.inference_mode():
        scripted = torch.jit.script(model)
        eager_actions = model(*probe)
        scripted_actions = scripted(*probe)
    script_error = float((eager_actions - scripted_actions).abs().max())
    if script_error > 1.0e-5:
        raise RuntimeError(f"TorchScript mismatch: {script_error}")

    scripted.save(str(output))
    reloaded = torch.jit.load(str(output)).eval()
    with torch.inference_mode():
        reloaded_actions = reloaded(*probe)
        dynamic_actions = reloaded(*(value[:1] for value in probe))
    reload_error = float((eager_actions - reloaded_actions).abs().max())
    if (
        reload_error > 1.0e-5
        or dynamic_actions.shape != (1, 4)
        or not torch.all(reloaded_actions.abs() <= 1.0)
    ):
        raise RuntimeError("Reloaded three-frame TorchScript validation failed")

    appearance_profile = (
        payload.get("student_environment_contract", {})
        .get("appearance_randomization", {})
        .get("profile")
    )
    legacy_appearance_evaluation_only = (
        appearance_profile == "x040_three_frame_realistic_material_v1"
    )
    metadata = {
        "kind": "tacex_rma_x040_wide_three_frame_direct_action_torchscript",
        "version": 1,
        "task": payload["task"],
        "student_checkpoint": str(checkpoint),
        "student_checkpoint_sha256": sha256_file(checkpoint),
        "torchscript_sha256": sha256_file(output),
        "teacher_checkpoint": payload["teacher_checkpoint"],
        "teacher_checkpoint_sha256": payload["teacher_checkpoint_sha256"],
        "input_signature": payload["student_input_contract"],
        "input_order": ["wrist_rgb_history", "proprio_obs", "action_history"],
        "output_signature": {"mean_actions": [4]},
        "runtime_privileged_inputs": [],
        "training_privileged_inputs": payload["training_privileged_inputs"],
        "student_environment_contract": payload["student_environment_contract"],
        "legacy_appearance_evaluation_only": legacy_appearance_evaluation_only,
        "position_head": "training-only auxiliary output; not in TorchScript signature",
        "script_max_abs_error": script_error,
        "reload_max_abs_error": reload_error,
    }
    _atomic_json(metadata, output.with_suffix(".json"))
    print(f"[INFO] Exported X040-Wide three-frame Student: {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
