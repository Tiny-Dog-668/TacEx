"""Collect X040-Wide Student rollouts without writing privileged cube labels."""

from __future__ import annotations
import argparse, csv, json, os, sys
from datetime import datetime
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

root=Path(__file__).resolve().parents[4]
for name in ("tacex_tasks","tacex","tacex_assets","tacex_uipc"): sys.path.insert(0,str(root/"source"/name))
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--student_checkpoint",required=True);parser.add_argument("--episodes",type=int,default=10);parser.add_argument("--seed",type=int,default=42);parser.add_argument("--frame_stride",type=int,default=1);parser.add_argument("--output_dir",default=None)
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();args.enable_cameras=True;simulation_app=AppLauncher(args).app
import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import tacex_tasks  # noqa:F401
from tacex_tasks.sim2real_grasp.rma_x040_wide_artifacts import load_student_checkpoint,load_student_model_state,sha256_file,validate_live_teacher_contract
from tacex_tasks.sim2real_grasp.rma_x040_wide_models import RMAX040WideDirectActionVisualStudent

def _np(value): return value.detach().cpu().numpy().copy()
def _save_npz(path,arrays):
    temporary=path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream: np.savez_compressed(stream,**arrays)
    os.replace(temporary,path)
def main():
    if args.episodes<=0 or args.frame_stride<=0: raise ValueError("episodes and frame_stride must be positive")
    checkpoint=Path(args.student_checkpoint).expanduser().resolve();payload=load_student_checkpoint(checkpoint,device="cpu")
    env_cfg=parse_env_cfg(payload["task"],device=args.device,num_envs=1);env_cfg.seed=args.seed;validate_live_teacher_contract(env_cfg,payload["teacher_manifest"])
    directory=Path(args.output_dir).expanduser().resolve() if args.output_dir else checkpoint.parent.parent/"rollouts"/datetime.now().strftime("%Y%m%d_%H%M%S");directory.mkdir(parents=True,exist_ok=False)
    env=gym.make(payload["task"],cfg=env_cfg);base=env.unwrapped;student=RMAX040WideDirectActionVisualStudent().to(base.device).eval();load_student_model_state(student,payload["model"]);observations,_=env.reset();summaries=[]
    try:
        with torch.inference_mode():
            for episode in range(args.episodes):
                records={name:[] for name in ("proprio_obs","action_history","student_action","predicted_cube_position_root_m","reward","terminated","truncated","success","table_collision","tcp_clearance_violation")};frames=[];frame_steps=[]
                while True:
                    obs=observations["policy"];step=len(records["reward"])
                    if step%args.frame_stride==0: frames.append(_np(obs["wrist_rgb"][0]));frame_steps.append(np.asarray(step,dtype=np.int32))
                    action,pred_norm=student.forward_with_position(obs["wrist_rgb"],obs["proprio_obs"].float(),obs["action_history"].float());predicted=student.normalizer.denormalize_position(pred_norm);observations,rewards,terminated,truncated,_=env.step(action)
                    for name,value in (("proprio_obs",obs["proprio_obs"][0]),("action_history",obs["action_history"][0]),("student_action",action[0]),("predicted_cube_position_root_m",predicted[0]),("reward",rewards[0]),("terminated",terminated[0]),("truncated",truncated[0]),("success",base._last_rma_success_nonterminal[0]),("table_collision",base._last_table_collision[0]),("tcp_clearance_violation",base._last_tcp_table_clearance_violation[0])): records[name].append(_np(value))
                    if bool((terminated|truncated)[0]): break
                arrays={key:np.stack(value) for key,value in records.items()};arrays["wrist_rgb"]=np.stack(frames);arrays["frame_step_indices"]=np.stack(frame_steps);out=directory/f"episode_{episode:04d}.npz";_save_npz(out,arrays)
                summaries.append({"episode":episode,"steps":int(arrays["reward"].shape[0]),"return":float(arrays["reward"].sum()),"success_ever":int(arrays["success"].any()),"table_collision_ever":int(arrays["table_collision"].any()),"tcp_clearance_violation_ever":int(arrays["tcp_clearance_violation"].any()),"frame_count":int(arrays["wrist_rgb"].shape[0]),"file":out.name})
    finally: env.close()
    with (directory/"episodes.csv").open("w",newline="",encoding="utf-8") as stream: writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    manifest={"kind":"tacex_rma_x040_wide_rollout_dataset","version":1,"task":payload["task"],"student_checkpoint":str(checkpoint),"student_checkpoint_sha256":sha256_file(checkpoint),"episodes":args.episodes,"frame_stride":args.frame_stride,"student_runtime_inputs":payload["student_input_contract"],"stored_transition_inputs":["wrist_rgb","proprio_obs","action_history"],"stored_model_outputs":["student_action","predicted_cube_position_root_m"],"excluded_privileged_inputs":["rma_cube_pos","rma_contact_force","rma_contact_state"],"episode_files":[row["file"] for row in summaries]}
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");print(f"[INFO] Saved X040-Wide rollout dataset: {directory}")
if __name__=="__main__":
    try: main()
    finally: simulation_app.close()
