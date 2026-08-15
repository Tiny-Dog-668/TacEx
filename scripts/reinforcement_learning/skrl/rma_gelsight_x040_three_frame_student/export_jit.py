"""Export a GelSight X040-DR three-frame Student to TorchScript."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    root = Path(__file__).resolve().parents[4]
    for relative in ("source/tacex_tasks", "source/tacex", "source/tacex_assets"):
        value = str(root / relative)
        if value not in sys.path:
            sys.path.insert(0, value)


_extend_repo_pythonpath()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--student_checkpoint", required=True)
parser.add_argument("--output", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch

from tacex_tasks.sim2real_gelsight_rma.rma_gelsight_x040_three_frame_artifacts import (
    load_student_checkpoint,
    load_student_model_state,
    make_student_model_for_checkpoint,
    sha256_file,
)


def _probe(batch: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.randint(0, 256, (batch, 3, 224, 224, 3), dtype=torch.uint8),
        torch.randn(batch, 15),
        torch.randn(batch, 4) * 0.01,
        torch.randint(0, 256, (batch, 96, 128, 3), dtype=torch.uint8),
        torch.randint(0, 256, (batch, 96, 128, 3), dtype=torch.uint8),
        torch.randint(0, 256, (batch, 96, 128, 3), dtype=torch.uint8),
        torch.randint(0, 256, (batch, 96, 128, 3), dtype=torch.uint8),
    )


def _atomic_json_dump(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    checkpoint = Path(args.student_checkpoint).expanduser().resolve()
    payload = load_student_checkpoint(checkpoint)
    model = make_student_model_for_checkpoint(
        payload, pretrained_backbone=False
    ).cpu().eval()
    load_student_model_state(model, payload["model"])
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else checkpoint.parent / "exported" / f"gelsight_x040_three_frame_{checkpoint.stem}.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    validation = {}
    with torch.inference_mode():
        for batch in (1, 8):
            probe = _probe(batch)
            eager = model(*probe)
            actual = scripted(*probe)
            errors = [torch.max(torch.abs(a - b)).item() for a, b in zip(eager, actual)]
            if max(errors) > 1.0e-5 or not all(torch.isfinite(value).all() for value in actual):
                raise RuntimeError(f"TorchScript validation failed for batch={batch}: {errors}")
            validation[f"cpu_batch_{batch}_max_abs_error"] = errors
    scripted.save(str(output))
    reloaded = torch.jit.load(str(output), map_location="cpu").eval()
    with torch.inference_mode():
        reloaded_output = reloaded(*_probe(1))
    if not all(torch.isfinite(value).all() for value in reloaded_output):
        raise RuntimeError("Reloaded TorchScript returned non-finite output")
    cuda_validation = False
    if torch.cuda.is_available():
        cuda_model = torch.jit.load(str(output), map_location="cuda:0").eval()
        with torch.inference_mode():
            cuda_output = cuda_model(*tuple(value.cuda() for value in _probe(8)))
        if not all(torch.isfinite(value).all() for value in cuda_output):
            raise RuntimeError("CUDA TorchScript returned non-finite output")
        cuda_validation = True
    input_signature = dict(payload["student_input_contract"])
    tactile_shape = input_signature.pop("tactile_rgb")
    input_signature.update(
        {
            "gsmini_left_rgb": tactile_shape,
            "gsmini_right_rgb": tactile_shape,
            "gsmini_left_reference_rgb": tactile_shape,
            "gsmini_right_reference_rgb": tactile_shape,
        }
    )
    _atomic_json_dump(
        {
            "kind": "tacex_rma_gelsight_x040_dr_three_frame_student_torchscript",
            "version": 1,
            "student_checkpoint": str(checkpoint),
            "student_checkpoint_sha256": sha256_file(checkpoint),
            "torchscript_sha256": sha256_file(output),
            "input_order": payload["student_input_contract"]["input_order"],
            "input_signature": input_signature,
            "output_signature": payload["student_input_contract"]["runtime_output"],
            "student_model_contract": payload["student_model_contract"],
            "validation": validation,
            "cuda_validation": cuda_validation,
        },
        output.with_suffix(".json"),
    )
    print(f"[INFO] Exported: {output}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
