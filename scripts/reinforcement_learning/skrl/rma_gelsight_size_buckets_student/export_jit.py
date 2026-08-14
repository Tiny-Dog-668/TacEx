"""Export the fixed-size GelSight reference-delta Student to TorchScript."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    root = Path(__file__).resolve().parents[4]
    value = str(root / "source/tacex_tasks")
    if value not in sys.path:
        sys.path.insert(0, value)


_extend_repo_pythonpath()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--student_checkpoint", required=True)
parser.add_argument("--output", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_size_buckets_artifacts import (
    load_student_checkpoint,
    load_student_model_state,
    sha256_file,
)
from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_size_buckets_models import (
    RMAGelSightReferenceStudent,
)


def _atomic_json_dump(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _probe(batch: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.randint(0, 256, (batch, 224, 224, 3), dtype=torch.uint8),
        torch.randn(batch, 15),
        torch.randn(batch, 4) * 0.01,
        torch.randint(0, 256, (batch, 96, 128, 3), dtype=torch.uint8),
        torch.randint(0, 256, (batch, 96, 128, 3), dtype=torch.uint8),
        torch.randint(0, 256, (batch, 96, 128, 3), dtype=torch.uint8),
        torch.randint(0, 256, (batch, 96, 128, 3), dtype=torch.uint8),
    )


def main() -> None:
    checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    payload = load_student_checkpoint(checkpoint)
    model = RMAGelSightReferenceStudent(pretrained_backbone=False).cpu().eval()
    load_student_model_state(model, payload["model"])
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint.parent / "exported" / f"gelsight_reference_student_{checkpoint.stem}.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    validation = {}
    with torch.inference_mode():
        for batch in (1, 8):
            probe = _probe(batch)
            eager = model(*probe)
            actual = scripted(*probe)
            error = torch.max(torch.abs(eager - actual)).item()
            if error > 1.0e-5 or not torch.isfinite(actual).all():
                raise RuntimeError(f"TorchScript validation failed for batch={batch}: {error}")
            validation[f"cpu_batch_{batch}_max_abs_error"] = error
    scripted.save(str(output))
    reloaded = torch.jit.load(str(output), map_location="cpu").eval()
    with torch.inference_mode():
        reload_output = reloaded(*_probe(1))
    if not torch.isfinite(reload_output).all():
        raise RuntimeError("Reloaded TorchScript returned non-finite actions")
    cuda_validation = None
    if torch.cuda.is_available():
        cuda_model = torch.jit.load(str(output), map_location="cuda:0").eval()
        with torch.inference_mode():
            cuda_output = cuda_model(*tuple(value.cuda() for value in _probe(8)))
        if not torch.isfinite(cuda_output).all():
            raise RuntimeError("CUDA TorchScript returned non-finite actions")
        cuda_validation = True
    teacher_environment = payload["teacher_manifest"]["environment_contract"]
    policy_frequency_hz = float(teacher_environment.get("policy_frequency_hz", 30.0))
    episode_length_s = float(teacher_environment["episode_length_s"])
    max_episode_length_steps = int(
        teacher_environment.get(
            "max_episode_length_steps",
            round(policy_frequency_hz * episode_length_s),
        )
    )
    if (
        policy_frequency_hz != 30.0
        or episode_length_s != 5.0
        or max_episode_length_steps != 150
    ):
        raise RuntimeError("GelSight Size-Buckets deployment requires 30 Hz and 150 steps")
    metadata = {
        "kind": "tacex_rma_gelsight_size_buckets_student_torchscript",
        "version": 2,
        "student_checkpoint": str(checkpoint),
        "student_checkpoint_sha256": sha256_file(checkpoint),
        "torchscript_sha256": sha256_file(output),
        "input_order": payload["student_input_contract"]["input_order"],
        "input_signature": {
            "wrist_rgb": [224, 224, 3],
            "proprio_obs": [15],
            "action_history": [4],
            "gsmini_left_rgb": [96, 128, 3],
            "gsmini_right_rgb": [96, 128, 3],
            "gsmini_left_reference_rgb": [96, 128, 3],
            "gsmini_right_reference_rgb": [96, 128, 3],
        },
        "output_signature": {"mean_actions": [4]},
        "tactile_delta": "signed_float32_current_minus_reference_div_255",
        "student_model_contract": payload["student_model_contract"],
        "actor_fusion": "visual_512+left_tactile_256+right_tactile_256+proprio_15+history_4",
        "contact_to_actor": "continuous_tactile_features; logits_are_auxiliary_only",
        "training_only_auxiliary_labels": [
            "cube_position_xyz",
            "projected_cube_center_heatmap_14x14",
            "left_right_physics_contact",
        ],
        "deployment_contract": {
            "policy_frequency_hz": policy_frequency_hz,
            "episode_length_s": episode_length_s,
            "max_episode_length_steps": max_episode_length_steps,
        },
        "validation": validation,
        "cuda_validation": cuda_validation,
    }
    _atomic_json_dump(metadata, output.with_suffix(".json"))
    print(f"[INFO] Exported: {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
