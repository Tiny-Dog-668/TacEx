"""Evaluate X040-Wide Student actions and RGB-only cube XYZ prediction on a fixed grid."""

from __future__ import annotations
import argparse, csv, json, sys, types
from pathlib import Path
from isaaclab.app import AppLauncher

root = Path(__file__).resolve().parents[4]
for name in ("tacex_tasks", "tacex", "tacex_assets", "tacex_uipc"):
    sys.path.insert(0, str(root / "source" / name))
parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--student_checkpoint", required=True); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--output", default=None)
AppLauncher.add_app_launcher_args(parser); args = parser.parse_args(); args.enable_cameras = True; simulation_app = AppLauncher(args).app
import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import tacex_tasks  # noqa: F401
from tacex_tasks.sim2real_grasp.rma_x040_wide_artifacts import load_student_checkpoint, load_student_model_state, load_teacher_policy_state, sha256_file, validate_live_teacher_contract
from tacex_tasks.sim2real_grasp.rma_x040_wide_models import RMAX040WideActorCore, RMAX040WideDirectActionVisualStudent, extract_x040_wide_actor_core_state_dict

def _install_grid(base_env):
    positions = [(float(x), float(y)) for x in torch.linspace(.32,.48,10).tolist() for y in torch.linspace(-.10,.10,10).tolist()]
    nominal = base_env.cfg.cube.init_state.pos; cursor = {"value": 0}
    def sample(self, count):
        result = torch.empty((count,2),device=self.device)
        for index in range(count):
            x,y = positions[cursor["value"] % len(positions)]; cursor["value"] += 1
            result[index] = torch.tensor((x-nominal[0],y-nominal[1]),device=self.device)
        return result
    base_env._sample_cube_xy_offsets = types.MethodType(sample, base_env)
    return positions

def main():
    checkpoint = Path(args.student_checkpoint).expanduser().resolve(); payload = load_student_checkpoint(checkpoint, device="cpu")
    env_cfg = parse_env_cfg(payload["task"],device=args.device,num_envs=1); env_cfg.seed=args.seed; validate_live_teacher_contract(env_cfg,payload["teacher_manifest"])
    env = gym.make(payload["task"],cfg=env_cfg); base=env.unwrapped; positions=_install_grid(base)
    student=RMAX040WideDirectActionVisualStudent().to(base.device).eval(); load_student_model_state(student,payload["model"])
    teacher=RMAX040WideActorCore().to(base.device).eval(); teacher.load_state_dict(extract_x040_wide_actor_core_state_dict(load_teacher_policy_state(payload["teacher_checkpoint"],base.device)),strict=True)
    observations,_=env.reset(); rows=[]; action_sq=position_sq=0.; action_count=position_count=0; previous=None; delta_sq=0.; delta_count=0; total_return=0.; steps=0
    try:
        with torch.inference_mode():
            while len(rows)<len(positions):
                obs=observations["policy"]; action,pred_norm=student.forward_with_position(obs["wrist_rgb"],obs["proprio_obs"].float(),obs["action_history"].float()); truth=obs["rma_cube_pos"].float(); teacher_action=teacher(obs["proprio_obs"].float(),obs["action_history"].float(),truth)
                predicted=student.normalizer.denormalize_position(pred_norm); action_sq+=float((action-teacher_action).square().sum()); action_count+=action.numel(); position_sq+=float((predicted-truth).square().sum()); position_count+=predicted.numel()
                if previous is not None: delta_sq+=float((action-previous).square().sum()); delta_count+=action.numel()
                previous=action; observations,rewards,terminated,truncated,_=env.step(action); total_return+=float(rewards[0]); steps+=1
                if not bool((terminated|truncated)[0]): continue
                x,y=positions[len(rows)]; success=bool(base._last_rma_success_nonterminal[0]); rows.append({"episode":len(rows),"cube_x":x,"cube_y":y,"success":int(success),"steps":steps,"return":total_return}); total_return=0.;steps=0;previous=None
    finally: env.close()
    output=Path(args.output).expanduser().resolve() if args.output else checkpoint.parent.parent/"metrics"/"x040_wide_grid.json"; output.parent.mkdir(parents=True,exist_ok=True)
    csv_path=output.with_suffix(".csv");
    with csv_path.open("w",newline="",encoding="utf-8") as stream: writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report={"kind":"tacex_rma_x040_wide_grid_evaluation","protocol":"fixed_10x10_xy_grid_v1","student_checkpoint":str(checkpoint),"student_checkpoint_sha256":sha256_file(checkpoint),"episodes":len(rows),"success_rate":sum(row["success"] for row in rows)/len(rows),"mean_return":sum(row["return"] for row in rows)/len(rows),"teacher_action_rmse":(action_sq/action_count)**.5,"cube_position_rmse_m":(position_sq/position_count)**.5,"student_action_delta_rmse":(delta_sq/delta_count)**.5 if delta_count else 0.,"episodes_csv":str(csv_path)}
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=="__main__":
    try: main()
    finally: simulation_app.close()
