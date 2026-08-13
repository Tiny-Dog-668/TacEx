"""Replay an X040-Wide Student and record action/XYZ prediction metrics."""

from __future__ import annotations
import argparse, csv, json, sys
from datetime import datetime
from pathlib import Path
from isaaclab.app import AppLauncher

root=Path(__file__).resolve().parents[4]
for name in ("tacex_tasks","tacex","tacex_assets","tacex_uipc"):sys.path.insert(0,str(root/"source"/name))
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--student_checkpoint",required=True);parser.add_argument("--num_envs",type=int,default=1);parser.add_argument("--steps",type=int,default=1000);parser.add_argument("--seed",type=int,default=42);parser.add_argument("--metrics_interval",type=int,default=100);parser.add_argument("--output_dir",default=None)
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();args.enable_cameras=True;simulation_app=AppLauncher(args).app
import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import tacex_tasks # noqa:F401
from tacex_tasks.sim2real_grasp.rma_x040_wide_artifacts import load_student_checkpoint,load_student_model_state,load_teacher_policy_state,sha256_file,validate_live_teacher_contract
from tacex_tasks.sim2real_grasp.rma_x040_wide_models import RMAX040WideActorCore,RMAX040WideDirectActionVisualStudent,extract_x040_wide_actor_core_state_dict

def main():
    checkpoint=Path(args.student_checkpoint).expanduser().resolve();payload=load_student_checkpoint(checkpoint,device="cpu")
    env_cfg=parse_env_cfg(payload["task"],device=args.device,num_envs=args.num_envs);env_cfg.seed=args.seed;validate_live_teacher_contract(env_cfg,payload["teacher_manifest"])
    directory=Path(args.output_dir).expanduser().resolve() if args.output_dir else checkpoint.parent.parent/"metrics"/"x040_wide_play";directory.mkdir(parents=True,exist_ok=True);tag=datetime.now().strftime("%Y%m%d_%H%M%S");csv_path=directory/f"play_metrics_{tag}.csv"
    env=gym.make(payload["task"],cfg=env_cfg);base=env.unwrapped;student=RMAX040WideDirectActionVisualStudent().to(base.device).eval();load_student_model_state(student,payload["model"]);teacher=RMAX040WideActorCore().to(base.device).eval();teacher.load_state_dict(extract_x040_wide_actor_core_state_dict(load_teacher_policy_state(payload["teacher_checkpoint"],base.device)),strict=True);observations,_=env.reset();reward_sum=action_sq=position_sq=0.;action_count=position_count=0;previous=None;delta_sq=0.;delta_count=0
    try:
        with csv_path.open("w",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream);writer.writerow(("step","mean_step_reward","teacher_action_rmse","student_action_delta_rmse","cube_position_rmse_m","cumulative_success_rate","recent_success_rate","tcp_clearance_violation_fraction"))
            with torch.inference_mode():
                for step in range(1,args.steps+1):
                    obs=observations["policy"];action,pred_norm=student.forward_with_position(obs["wrist_rgb"],obs["proprio_obs"].float(),obs["action_history"].float());truth=obs["rma_cube_pos"].float();teacher_action=teacher(obs["proprio_obs"].float(),obs["action_history"].float(),truth);predicted=student.normalizer.denormalize_position(pred_norm)
                    action_sq+=float((action-teacher_action).square().sum());action_count+=action.numel();position_sq+=float((predicted-truth).square().sum());position_count+=predicted.numel()
                    if previous is not None:delta_sq+=float((action-previous).square().sum());delta_count+=action.numel()
                    previous=action;observations,rewards,_,_,_=env.step(action);reward_sum+=float(rewards.mean())
                    if step%args.metrics_interval==0 or step==args.steps:
                        stats=base._episode_success_statistics();writer.writerow((step,reward_sum/step,(action_sq/action_count)**.5,(delta_sq/delta_count)**.5 if delta_count else 0.,(position_sq/position_count)**.5,float(stats["cumulative_rate"]),float(stats["window_rate"]),float(base._last_tcp_table_clearance_violation.float().mean())));stream.flush()
    finally:env.close()
    summary={"kind":"tacex_rma_x040_wide_direct_action_play","student_checkpoint":str(checkpoint),"student_checkpoint_sha256":sha256_file(checkpoint),"task":payload["task"],"seed":args.seed,"num_envs":args.num_envs,"steps":args.steps,"metrics_csv":str(csv_path),"student_runtime_inputs":["wrist_rgb","proprio_obs","action_history"],"teacher_action_rmse":(action_sq/action_count)**.5 if action_count else None,"cube_position_rmse_m":(position_sq/position_count)**.5 if position_count else None}
    (directory/f"play_summary_{tag}.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
if __name__=="__main__":
    try:main()
    finally:simulation_app.close()
