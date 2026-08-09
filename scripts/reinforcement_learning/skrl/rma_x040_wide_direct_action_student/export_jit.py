"""Export an X040-Wide Student with the strict three-input TorchScript signature."""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / "source" / "tacex_tasks"))
parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--student_checkpoint", required=True); parser.add_argument("--output", default=None)
AppLauncher.add_app_launcher_args(parser); args = parser.parse_args(); simulation_app = AppLauncher(args).app
import torch
from tacex_tasks.sim2real_grasp.rma_x040_wide_artifacts import load_student_checkpoint, load_student_model_state, sha256_file, state_dict_sha256
from tacex_tasks.sim2real_grasp.rma_x040_wide_models import RMAX040WideDirectActionVisualStudent

def main() -> None:
    checkpoint = Path(args.student_checkpoint).expanduser().resolve(); payload = load_student_checkpoint(checkpoint, device="cpu")
    model = RMAX040WideDirectActionVisualStudent().eval(); load_student_model_state(model, payload["model"])
    if state_dict_sha256(model.vision_encoder.state_dict()) != payload["vision_encoder_state_dict_sha256"]: raise RuntimeError("Vision encoder hash mismatch")
    output = Path(args.output).expanduser().resolve() if args.output else checkpoint.parent / "exported" / f"rma_x040_wide_student_{checkpoint.stem}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    probe = (torch.randint(0, 256, (2,224,224,3), dtype=torch.uint8), torch.randn(2,15), torch.randn(2,4)*.01)
    with torch.inference_mode():
        scripted = torch.jit.script(model); eager = model(*probe); scripted_actions = scripted(*probe)
    error = float((eager-scripted_actions).abs().max());
    if error > 1e-5: raise RuntimeError(f"TorchScript mismatch: {error}")
    scripted.save(str(output)); reloaded = torch.jit.load(str(output)).eval()
    with torch.inference_mode(): reloaded_actions = reloaded(*probe); dynamic_actions = reloaded(*(item[:1] for item in probe))
    reload_error = float((eager-reloaded_actions).abs().max())
    if reload_error > 1e-5 or dynamic_actions.shape != (1,4) or not torch.all(reloaded_actions.abs() <= 1): raise RuntimeError("Reloaded TorchScript validation failed")
    metadata = {"kind":"tacex_rma_x040_wide_direct_action_torchscript", "version":1, "student_checkpoint":str(checkpoint), "student_checkpoint_sha256":sha256_file(checkpoint), "torchscript_sha256":sha256_file(output), "teacher_checkpoint":payload["teacher_checkpoint"], "teacher_checkpoint_sha256":payload["teacher_checkpoint_sha256"], "input_signature":payload["student_input_contract"], "input_order":["wrist_rgb","proprio_obs","action_history"], "output_signature":{"mean_actions":[4]}, "runtime_privileged_inputs":[], "training_privileged_inputs":payload["training_privileged_inputs"], "position_head":"training-only auxiliary output; not in TorchScript signature", "trace_max_abs_error":error, "reload_max_abs_error":reload_error}
    path = output.with_suffix(".json"); temporary = path.with_name(f".{path.name}.tmp"); temporary.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,path)
    print(f"[INFO] Exported X040-Wide Student: {output}")
if __name__ == "__main__":
    try: main()
    finally: simulation_app.close()
