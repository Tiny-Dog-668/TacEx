"""Export a three-input direct-action RMA Student as TorchScript."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    root = Path(__file__).resolve().parents[4]
    path = root / "source" / "tacex_tasks"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


_extend_repo_pythonpath()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--student_checkpoint", required=True)
parser.add_argument("--output", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from tacex_tasks.sim2real_grasp.rma_direct_action_student.artifacts import (
    load_student_checkpoint, load_student_model_state, sha256_file, state_dict_sha256,
)
from tacex_tasks.sim2real_grasp.rma_direct_action_student.models import RMADirectActionVisualStudent


def _atomic_json_dump(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    payload = load_student_checkpoint(checkpoint, device="cpu")
    model = RMADirectActionVisualStudent().cpu().eval()
    load_student_model_state(model, payload["model"])
    if state_dict_sha256(model.vision_encoder.state_dict()) != payload.get("vision_encoder_state_dict_sha256"):
        raise RuntimeError("Direct-action Student vision encoder hash mismatch")
    output = Path(args.output).expanduser().resolve() if args.output else (
        checkpoint.parent / "exported" / f"rma_direct_action_student_{checkpoint.stem}.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    probe = (
        torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8),
        torch.randn((2, 15), dtype=torch.float32),
        torch.randn((2, 4), dtype=torch.float32) * 0.01,
    )
    with torch.inference_mode():
        scripted = torch.jit.script(model)
        eager = model(*probe)
        scripted_actions = scripted(*probe)
    trace_error = float(torch.max(torch.abs(eager - scripted_actions)).item())
    if trace_error > 1e-5:
        raise RuntimeError(f"TorchScript mismatch: max_abs_error={trace_error}")
    scripted.save(str(output))
    reloaded = torch.jit.load(str(output), map_location="cpu").eval()
    with torch.inference_mode():
        reloaded_actions = reloaded(*probe)
        dynamic_actions = reloaded(*(value[:1] for value in probe))
    reload_error = float(torch.max(torch.abs(eager - reloaded_actions)).item())
    if reload_error > 1e-5 or not torch.isfinite(reloaded_actions).all() or dynamic_actions.shape != (1, 4):
        raise RuntimeError("Reloaded direct-action TorchScript validation failed")
    if not torch.all((reloaded_actions >= -1.0) & (reloaded_actions <= 1.0)):
        raise RuntimeError("Exported direct-action Student actions violate tanh bounds")
    _atomic_json_dump(
        {
            "kind": "tacex_rma_direct_action_student_torchscript",
            "version": 1,
            "student_checkpoint": str(checkpoint),
            "student_checkpoint_sha256": sha256_file(checkpoint),
            "torchscript_sha256": sha256_file(output),
            "teacher_checkpoint": payload["teacher_checkpoint"],
            "teacher_checkpoint_sha256": payload["teacher_checkpoint_sha256"],
            "input_signature": payload["student_input_contract"],
            "input_order": ["wrist_rgb", "proprio_obs", "action_history"],
            "output_signature": {"mean_actions": [4]},
            "normalization": payload["normalization"],
            "model_contract": payload["student_model_contract"],
            "runtime_privileged_inputs": [],
            "training_privileged_inputs": payload["training_privileged_inputs"],
            "trace_max_abs_error": trace_error,
            "reload_max_abs_error": reload_error,
            "notes": [
                "wrist_rgb is uint8 RGB in NHWC layout.",
                "No cube position or contact force is accepted at runtime.",
                "Real RGB must use the calibrated crop/resize contract.",
            ],
        },
        output.with_suffix(".json"),
    )
    print(f"[INFO] Exported direct-action Student: {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
