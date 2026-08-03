"""Export a distilled Real-Alignment RMA student as end-to-end TorchScript."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "source" / "tacex_tasks"
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

from tacex_tasks.sim2real_grasp.rma_artifacts import (
    load_student_model_state,
    load_student_checkpoint,
    sha256_file,
    state_dict_sha256,
)
from tacex_tasks.sim2real_grasp.rma_models import RMAActorCore, RMAVisualStudent


# CUDA convolution kernels need not be bitwise identical to CPU kernels.
CUDA_VALIDATION_ATOL = 1e-3


def _atomic_json_dump(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    payload = load_student_checkpoint(checkpoint, device="cpu")
    model = RMAVisualStudent(RMAActorCore(), pretrained_backbone=False).cpu().eval()
    load_student_model_state(model, payload["model"])
    if state_dict_sha256(model.vision_encoder.state_dict()) != payload.get(
        "vision_encoder_state_dict_sha256"
    ):
        raise RuntimeError("Student checkpoint vision encoder hash mismatch")
    if state_dict_sha256(model.actor_core.state_dict()) != payload.get(
        "teacher_actor_state_dict_sha256"
    ):
        raise RuntimeError("Student checkpoint teacher Actor hash mismatch")

    output = Path(args.output).expanduser().resolve() if args.output else (
        checkpoint.parent / "exported" / f"rma_student_e2e_{checkpoint.stem}.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    example = (
        torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
        torch.zeros((1, 15), dtype=torch.float32),
        torch.zeros((1, 4), dtype=torch.float32),
    )
    with torch.inference_mode():
        traced = torch.jit.script(model)
        probe = (
            torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8),
            torch.randn((2, 15), dtype=torch.float32),
            torch.randn((2, 4), dtype=torch.float32) * 0.01,
        )
        eager_actions = model(*probe)
        traced_actions = traced(*probe)
        trace_error = float(torch.max(torch.abs(eager_actions - traced_actions)).item())
    if trace_error > 1e-5:
        raise RuntimeError(f"TorchScript trace mismatch: max_abs_error={trace_error}")
    traced.save(str(output))
    reloaded = torch.jit.load(str(output), map_location="cpu").eval()
    with torch.inference_mode():
        reloaded_actions = reloaded(*probe)
    reload_error = float(torch.max(torch.abs(eager_actions - reloaded_actions)).item())
    if reload_error > 1e-5 or not torch.isfinite(reloaded_actions).all():
        raise RuntimeError(f"Reloaded TorchScript validation failed: error={reload_error}")
    if not torch.all((reloaded_actions >= -1.0) & (reloaded_actions <= 1.0)):
        raise RuntimeError("Exported student actions violate tanh bounds")

    cuda_validation: dict[str, bool | float | None] = {
        "available": bool(torch.cuda.is_available()),
        "max_abs_error": None,
    }
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda:0")
        reloaded_cuda = torch.jit.load(str(output), map_location=cuda_device).eval()
        probe_cuda = tuple(value.to(cuda_device) for value in probe)
        with torch.inference_mode():
            cuda_actions = reloaded_cuda(*probe_cuda)
        cuda_error = float(torch.max(torch.abs(eager_actions - cuda_actions.cpu())).item())
        if cuda_error > CUDA_VALIDATION_ATOL or not torch.isfinite(cuda_actions).all():
            raise RuntimeError(f"CUDA TorchScript validation failed: error={cuda_error}")
        cuda_validation["max_abs_error"] = cuda_error

    metadata = {
        "kind": "tacex_rma_student_torchscript",
        "version": 6,
        "student_checkpoint": str(checkpoint),
        "student_checkpoint_sha256": sha256_file(checkpoint),
        "teacher_checkpoint": payload.get("teacher_checkpoint"),
        "teacher_checkpoint_sha256": payload.get("teacher_checkpoint_sha256"),
        "torchscript_sha256": sha256_file(output),
        "input_signature": {
            "wrist_rgb": [224, 224, 3],
            "proprio_obs": [15],
            "action_history": [4],
        },
        "input_order": ["wrist_rgb", "proprio_obs", "action_history"],
        "output_signature": {"mean_actions": [4]},
        "actor_mean_bounds": [-1.0, 1.0],
        "normalization": payload["normalization"],
        "vision_encoder_state_dict_sha256": payload["vision_encoder_state_dict_sha256"],
        "teacher_actor_state_dict_sha256": payload["teacher_actor_state_dict_sha256"],
        "actor_contract": model.actor_core.contract(),
        "trace_max_abs_error": trace_error,
        "reload_max_abs_error": reload_error,
        "cuda_validation": cuda_validation,
        "cuda_validation_atol": CUDA_VALIDATION_ATOL,
        "privileged_inputs": [],
        "notes": [
            "wrist_rgb is uint8 RGB in NHWC layout.",
            "Real input must already use the calibrated crop/resize contract.",
            "The simulator-only nominal intrinsic compensation is not embedded.",
            "Fingertip-midpoint XYZ is computed inside the Actor with Panda FK "
            "from proprio_obs joint positions.",
            "Cube and end-effector orientation are not Actor features.",
            "Left/right cube-finger contact probabilities are predicted from wrist_rgb.",
            "Load with map_location='cpu' or map_location='cuda:0'; inputs must use the same device.",
        ],
    }
    metadata_path = output.with_suffix(".json")
    _atomic_json_dump(metadata, metadata_path)
    print(f"[INFO] Exported RMA student: {output}")
    print(f"[INFO] Metadata: {metadata_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
