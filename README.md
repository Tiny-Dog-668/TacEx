# Occluded Visuotactile Grasping with TacEx

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

This repository builds on TacEx to study robot grasping under visual occlusion with vision-tactile fusion policies.
The current project focus is an Isaac Sim / Isaac Lab reinforcement-learning task where a Franka Panda arm grasps and lifts objects in drawer-style or self-occlusion scenes using a third-person RGB camera, four GelSight Mini tactile sensors, proprioception, and PPO policies.

## Project Focus

The active task family is implemented under `source/tacex_tasks/tacex_tasks/occluded_grasping`.
It provides composable task names:

```text
TacEx-{Fusion}-{Scene}-{Object}
```

Supported scene names in the registered task matrix are:

```text
Drawer-Occlusion
Self-Occlusion
```

Supported object names in the registered task matrix are:

```text
Cylinder
Cube
Cuboid
SoftCylinder
SoftCube
SoftCuboid
```

The fusion axis includes vision-only, tactile-proprioception, vision-tactile, GelFusion-style, frozen ViT GelFusion-style, alpha-gated, GRU, cross-attention, auxiliary-head, Sparsh, and token-transformer variants. See `source/tacex_tasks/tacex_tasks/occluded_grasping/README.md` for the complete list.

## Main Components

- Environment and task registration:
  `source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py`
- Main vision-tactile environment:
  `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py`
- Vision-only baseline:
  `source/tacex_tasks/tacex_tasks/occluded_grasping/vision_box.py`
- Policy and PPO configs:
  `source/tacex_tasks/tacex_tasks/occluded_grasping/*policy.py`
  and `source/tacex_tasks/tacex_tasks/occluded_grasping/agents/*.yaml`
- skrl training entry point:
  `scripts/reinforcement_learning/skrl/train.py`
- skrl checkpoint playback:
  `scripts/reinforcement_learning/skrl/play.py`
- Fixed-position bucket evaluation:
  `scripts/reinforcement_learning/skrl/play_bucket.py`
- GelSight sensor implementation:
  `source/tacex/tacex/gelsight_sensor.py`
- GelSight Mini asset config:
  `source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py`

## Observation and Action Summary

The main environment `OccludedGraspingVisionFourTactileBoxEnv` exposes policy observations from:

- `proprio_obs`: robot joint positions and velocities.
- `third_resnet`: third-person RGB camera features.
- `third_vit_cls`: optional frozen ViT-B/16 CLS visual feature used by GelFusion-ViT tasks.
- `tactile_left_depth_resnet`, `tactile_right_depth_resnet`, `tactile_left_down_depth_resnet`, `tactile_right_down_depth_resnet`: four GelSight tactile feature streams.
- `tactile_dynamic_stats`: optional GelFusion-style 8D tactile frame-difference statistics exposed by GelFusion tasks.
- Privileged critic keys for object pose, gripper pose, velocities, and target distance.

The `TacEx-T-{Scene}-{Object}` baseline exposes only `proprio_obs` and the four tactile feature streams to the actor while keeping the same action, reward, termination, and privileged critic definitions as the full VT task.

The action space is 5-dimensional:

```text
[dx, dy, dz, dyaw, gripper]
```

The first four values are converted to a differential IK command for the Franka arm. The last value controls the gripper as an incremental opening/closing command.

## Installation

TacEx currently targets Isaac Sim 4.5 and Isaac Lab 2.1.0 on Linux with Python 3.10.

Clone the repository with submodules and install the extensions following the existing TacEx installation docs:

```bash
git lfs install
git clone --recurse-submodules https://github.com/DH-Ng/TacEx
cd TacEx
```

Then install locally using:

```bash
./tacex.sh --install
```

For detailed setup notes, see:

- `docs/source/installation/Local-Installation.md`
- `docs/source/installation/Docker-Container-Setup.md`

## Train

Example training command from the registered occluded-grasping tasks:

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Alpha-GRU-Drawer-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras
```

GelFusion-style PPO with a frozen ViT-B/16 visual encoder:

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-GelFusion-ViT-Downsample-Drawer-Occlusion-Cuboid \
  --num_envs 4 \
  --enable_cameras
```

The training script loads the task config from the Gym/Isaac Lab registry, creates the environment, wraps it with `SkrlVecEnvWrapper`, and trains the PPO agent configured by the task's `skrl_cfg_entry_point`.

### UR10 + Robotiq Direct Tasks

This repository also includes migrated Isaac Lab direct RL tasks for UR10 + Robotiq:

```text
Isaac-UR10-Robotiq-Pick-Place-Direct-v0
Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0
Isaac-UR10-Robotiq-2F85-Third-Person-Pick-Place-Direct-v0
Isaac-UR10-Robotiq-2F85-Grasp-Direct-v0
Isaac-UR10-Robotiq-Gripper-Close-Direct-v0
```

The UR10 + Robotiq 2F85 pick-place task uses a 27-dimensional state observation and 4-dimensional action `[dx, dy, dz, gripper]`. The Robotiq 2F85 policy controls `finger_joint` as the single active gripper DOF, while the other finger joints are driven by code-level coupling targets.
The third-person variant is a standalone DirectRLEnv implementation for easier parameter edits. It adds a 224x224 RGB `third_person_camera` sensor using the camera intrinsics and pose from `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_grasp_env.py`; its PPO policy observation remains the same 27-dimensional state vector unless a separate vision policy is added.

Example training command:

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0 \
  --num_envs 64 \
  --headless
```

Third-person camera variant:

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-UR10-Robotiq-2F85-Third-Person-Pick-Place-Direct-v0 \
  --num_envs 64 \
  --enable_cameras \
  --headless
```

Keyboard reward debugger for the same UR/object/table layout:

```bash
python scripts/ur10_robotiq/teleop_2f85_rewards.py
```

Migrated checkpoint:

```text
logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/checkpoints/best_agent.pt
```

## Evaluate

### Train the real-reference-aligned cube task

`TacEx-Sim2Real-Cube-Real-Alignment-v0` is the clean vision-only Franka cube task aligned to the current real setup. It uses a white 5 cm cube on a 1 mm near-black board and a near-black backdrop. Cube positions are absolute in the robot-root frame: the full range is `x=[0.45,0.55] m`, `y=[-0.05,0.05] m`. Position curriculum holds `x=[0.48,0.52] m`, `y=[-0.02,0.02] m` through policy step 20,000, expands linearly to the full range at step 100,000, and is forced to the full range during evaluation. `TacEx-Sim2Real-Cube-Real-Alignment-DR-v0` is defined as this exact Clean scene plus curriculum-scaled visual perturbations: its DR scale is zero through step 100,000, grows linearly to full range at step 220,000, and remains full through the 300,000-step DR run. At zero scale, camera pose/intrinsics, post-processing, near-black board/backdrop materials, and DomeLight match Clean exactly. Both tasks keep a 224x224 simulator camera buffer and use a fixed GPU warp to reproduce the calibrated model-input intrinsics. Real deployment captures 640x480, crops `x=[100,500), y=[34,432)`, then bilinearly resizes the 400x398 crop to 224x224.

The sim-to-real Cube policy contract is intentionally strict:

- ResNet18 is frozen and kept in `eval()` so BatchNorm always uses the ImageNet running statistics.
- `action_history [N,4]` in observation `t+1` is the requested command associated with transition `t -> t+1`. The current Clean/DR Real-Alignment v9 scale is `[0.025,0.025,0.025,0.005]` m; retained v8 and v7 contracts remain readable only with their saved 25/2 mm and 10/2 mm configurations. The fourth component is the requested total-gripper-width delta. Sim2real Cube tasks disable the ground-truth object-XY `dz` gate, so the XYZ history also matches the command sent to IK before ordinary IK/joint/workspace limits.
- The Gaussian Actor mean applies `tanh` inside the policy model and is therefore bounded to `[-1,1]` in training, deterministic playback, and export.
- Arm IK uses a fixed Panda hand-to-TCP offset of `0.1034 m`. Reach reward and privileged critic gripper position/target distance use the world-space midpoint of the left/right fingertip centers, each transformed `0.045 m` from its finger-link origin.
- Both Real-Alignment tasks measure lift from the cube center of mass relative to its expected settled table height. A `0--35 mm` center lift maps linearly to lift reward `[0,1]`; success requires at least 35 mm for 5 consecutive policy steps and does not depend on cube tilt. Tilt remains diagnostic-only. A filtered table ContactSensor applies `-10` on a policy step when any active Panda arm/hand/finger link exceeds 1 N against the board; it does not terminate the episode.
- Training logs report the episode success rate over the latest 200 policy steps: successful episodes completed in that time window divided by all episodes completed in the same window. The same reward line also prints the window numerator/denominator and cumulative episode success rate; one successful episode is counted once after its 5-step hold.
- Each training run stores `params/vision_encoder_resnet18.pt` and `params/vision_encoder_resnet18.json`. The manifest binds the task, policy/history contract, deployment-relevant env values, agent/env config hashes, and encoder hashes. Keep the entire `params/` directory with the checkpoints.

These changes preserve the 4-D action and observation dimensions but change visual, temporal, and Actor semantics. Checkpoints trained before this contract must be retrained rather than resumed or re-exported with the new configuration; train, play, bucket play, and export reject legacy Cube runs that lack the manifest contract.

```bash
conda run -n isaaclab_2.1.1 --no-capture-output python \
  scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-v0 \
  --num_envs 4 \
  --enable_cameras \
  --headless
```

Train the broad domain-randomized profile separately:

```bash
conda run -n isaaclab_2.1.1 --no-capture-output python \
  scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-DR-v0 \
  --num_envs 4 \
  --enable_cameras \
  --headless
```

Train the no-camera privileged-position diagnostic task:

```bash
conda run -n isaaclab_2.1.1 --no-capture-output python \
  scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-Privileged-v0 \
  --num_envs 256 \
  --headless
```

This diagnostic Actor uses `proprio_obs [N,15]`, `action_history [N,4]`,
the environment-local cube position `[N,3]`, fingertip-midpoint position
`[N,3]`, and their relative target vector `[N,3]`. It creates no RGB camera
sensor or ResNet18 encoder. It shares the Clean scene/reward implementation
but explicitly retains its existing 2 mm gripper increment and `x/y ±10 cm`
reset range; reward, success, and the 150-step horizon remain shared.
It is an upper-bound reward/control experiment and cannot be deployed to the
real robot because its Actor inputs contain simulator ground truth.

### RMA teacher-student distillation

The Real-Alignment Clean task also provides an isolated RMA-style route. The
teacher Actor receives `proprio_obs[15]`, one-step `action_history[4]`, and
robot-root cube XYZ `[3]` plus privileged left/right cube-finger contact `[2]`.
A deployable Panda FK inside the shared Actor derives
the fingertip-midpoint XYZ `[3]` from the seven joint positions and then derives
`cube_xyz - fingertip_xyz [3]`, producing a 30-D Actor feature vector. The
student predicts cube XYZ and two contact probabilities from one calibrated
224x224 RGB frame and is optimized with normalized-position SmoothL1, contact
BCE, and deterministic teacher/student action MSE.
Both action branches share the same frozen Teacher Actor, and every simulator
transition executes the Student action. Object and end-effector orientation are
not Actor features because this task fixes the commanded tool orientation and
uses position-only reach/lift semantics.

Only the RMA tasks add cube-finger contact shaping. A filtered cube ContactSensor
uses a `0.2 N` threshold for each finger. Its current weight is `3.0`, with
the single-contact term enabled (`single_contact_reward_fraction=1.0`). The
Teacher receives the binary contact state, while the Student receives it only as
a training label and supplies its own differentiable visual contact probability
to the frozen Actor.
For the RMA Teacher/Student tasks, success is a reward/statistics condition only:
an episode terminates on timeout or severe robot ground penetration, not when the
cube first reaches the success height. The rolling success rate counts episodes
that reached success at least once before their timeout/collision terminal.
RMA also keeps the Clean `10 mm/step` total-width gripper action scale but lowers
only the Panda finger position actuator to `effort=40 N`, `stiffness=400`, and
`damping=40` to reduce cube penetration during hard contact.

Train the 200k privileged Teacher:

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-RMA-Teacher-v0 \
  --num_envs 256 --headless
```

The 100k Student run directly distills from a compatible Teacher checkpoint and
always uses the full Clean XY range:

```bash
python scripts/reinforcement_learning/skrl/train_rma_student.py \
  --teacher_checkpoint <teacher-run>/checkpoints/best_agent.pt \
  --num_envs 4 --timesteps 100000 --enable_cameras --headless
```

For a Student trained with full-strength visual domain randomization from its
first update (no curriculum), select the separate DR task. It keeps the same
Teacher physics/contact/action contract, but resamples camera pose/intrinsics,
image appearance/noise, board/backdrop appearance, and batch-global light:

```bash
conda run -n isaaclab_2.1.1 --no-capture-output python \
  scripts/reinforcement_learning/skrl/train_rma_student.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-DR-v0 \
  --teacher_checkpoint <teacher-run>/checkpoints/best_agent.pt \
  --num_envs 64 --timesteps 100000 --headless
```

This is a distinct Student profile and must start from scratch; do not resume a
Clean Student checkpoint into it (or the reverse). The resulting Student
checkpoint records its task profile, while the exported TorchScript input and
four-action output signatures remain unchanged.

For Student distillation, one `timesteps` unit is one optimizer/environment
update across all parallel environments. Thus `--num_envs 4 --timesteps 100000`
performs 100,000 updates and 400,000 simulator transitions. The terminal status
line reports update and sample throughput, elapsed time, and an ETA from the
observed average update rate.

The fixed-grid evaluator remains optional for comparing Teacher and Student:

```bash
python scripts/reinforcement_learning/skrl/evaluate_rma.py \
  --teacher_checkpoint <teacher.pt> --student_checkpoint <student.pt> \
  --episodes 100 --enable_cameras --headless

python scripts/reinforcement_learning/skrl/export_rma_student_jit.py \
  --student_checkpoint <student.pt> --headless
```

Replay a distilled Student directly in Isaac Sim with its checkpoint-recorded
CLEAN or DR task. This writes periodic success metrics to CSV and a final JSON
summary; unlike the fixed-grid evaluator, it samples the full random XY range:

```bash
conda run -n isaaclab_2.1.1 --no-capture-output python \
  scripts/reinforcement_learning/skrl/play_rma_student.py \
  --student_checkpoint <student-run>/checkpoints/student_0010000.pt \
  --num_envs 64 --steps 3000 --headless
```

The TorchScript signature is `wrist_rgb uint8[N,224,224,3]`, `proprio_obs
float32[N,15]`, and `action_history float32[N,4]` to a bounded `float32[N,4]`
action mean. No simulator position is part of the deployment input. New RMA
exports support both CPU and CUDA loading via `torch.jit.load(...,
map_location="cpu" | "cuda:0")`; all three input tensors must be on the
selected device.
The exported Actor computes fingertip-midpoint XYZ internally from the first
seven joint positions in `proprio_obs`; the real robot must provide joint angles
in the documented Panda joint order and robot-root frame convention.

The clean and DR tasks use separate log roots: `logs/skrl/sim2real_cube_real_alignment/` and `logs/skrl/sim2real_cube_real_alignment_dr/`. Do not resume a checkpoint across profiles. When resuming a DR run inside the same profile, set `dr_curriculum_step_offset` to the restored outer policy timestep; this offset is not inferred automatically.

The current camera pose and 640x480 intrinsics were supplied separately from the reference capture; their calibration provenance remains pending confirmation. Perform a calibrated image-alignment check before treating them as physically verified.

Run a checkpoint with the regular playback script:

```bash
python scripts/reinforcement_learning/skrl/play.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-v0 \
  --checkpoint logs/skrl/sim2real_cube_real_alignment/<new-run>/checkpoints/best_agent.pt \
  --num_envs 1 \
  --enable_cameras \
  --headless
```

### Export a sim-to-real cube policy

Export the trained Cube vision-only actor as an end-to-end CPU TorchScript model:

```bash
conda run -n isaaclab_2.1.1 --no-capture-output python \
  scripts/reinforcement_learning/skrl/export_sim2real_grasp_jit.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-v0 \
  --checkpoint logs/skrl/sim2real_cube_real_alignment/<new-run>/checkpoints/best_agent.pt \
  --num_envs 1 \
  --headless
```

The default output is `checkpoints/exported/policy_actor_e2e_best_agent.pt`, accompanied by a JSON metadata file. The model inputs are batched `action_history` `[N, 4]`, `proprio_obs` `[N, 15]`, and `wrist_rgb` `[N, 224, 224, 3]` as uint8 RGB. Although the observation key remains `wrist_rgb`, this Cube task uses a fixed third-person camera. Its output is the tanh-bounded four-dimensional Actor mean `[dx, dy, dz, gripper_total_width_delta]`; real-robot code must reproduce the exact per-dimension scales recorded by that run's contract, one-step history delay, robot-root XYZ frame, cached total-width gripper target, limits, watchdog, and emergency-stop behavior.

The exporter restores the checkpoint run's saved agent and environment configs instead of applying the current registry values. It verifies task identity, config hashes, encoder identity, normalization, and finite outputs, then reloads the saved model and compares eager/traced/reloaded outputs and the embedded encoder hash. These hashes detect accidental config/encoder drift or corruption; checkpoint-to-run association still relies on the run-directory layout, and the hashes are not a cryptographic publisher signature.

The Real-Alignment gripper controller updates one cached total-width target per 30 Hz policy step, clamps it to `[0,0.08]` m, and reapplies the same symmetric half-width finger targets during both 60 Hz physics substeps without incrementing the target again. Camera, rendering, policy, actions, observations, rewards, and history are aligned at 30 Hz; physics remains at 60 Hz for contact stability. Each episode lasts 5 seconds, exactly 150 policy steps.

Real-Alignment contract v9 binds the current Clean/DR 25/5 mm action scales and `x/y ±5 cm` reset bounds. Contract v8 remains the immutable 25/2 mm, `x/y ±10 cm` checkpoint contract, and v7 remains the 10/2 mm legacy contract. The existing real configuration `franka/configs/e2e_bundle_real_alignment_v8_25mm.json` is therefore only for its v8 model; a v9 real bundle must be generated from a newly trained v9 checkpoint.

Run the Isaac-independent smoke test after exporting:

```bash
python scripts/reinforcement_learning/skrl/test_sim2real_grasp_policy_jit.py \
  --model logs/skrl/sim2real_cube_real_alignment/<new-run>/checkpoints/exported/policy_actor_e2e_best_agent.pt
```

Collect batched simulation rollouts from the exact exported CPU Actor:

```bash
conda run -n isaaclab_2.1.1 --no-capture-output python \
  scripts/reinforcement_learning/skrl/collect_sim2real_cube_rollouts.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-v0 \
  --model logs/skrl/sim2real_cube_real_alignment/<run>/checkpoints/exported/policy_actor_e2e_best_agent.pt \
  --num_envs 32 \
  --steps 450 \
  --rgb_env_id 0 \
  --rgb_stride 10 \
  --headless
```

The collector restores the run's saved `env.pkl`, forces the full object-position range, verifies the exported model hash, runs the TorchScript Actor on CPU as in deployment, and writes a `T x N` compressed NPZ plus summary JSON under `metrics/sim2real_rollouts`. It records normalized/processed action, action history, requested/IK command, joints, cube pose, fingertip center, gripper target/measurement, RGB statistics, reward, lift, diagnostic tilt, table-collision force/flag/penalty, and done state. Legacy contracts require the explicit diagnostic override and are never made deployable by that option.

Convert the NPZ into a flattened CSV and episode-step action/state plots:

```bash
python scripts/reinforcement_learning/skrl/analyze_sim2real_cube_rollouts.py \
  --npz logs/skrl/sim2real_cube_real_alignment/<run>/metrics/sim2real_rollouts/rollout_v4_32env_450steps.npz
```

Record one environment's rollout actions and state traces during playback:

```bash
python scripts/reinforcement_learning/skrl/play.py \
  --task TacEx-T-Drawer-Occlusion-Cuboid \
  --num_envs 4 \
  --enable_cameras \
  --checkpoint logs/skrl/occluded_grasping/downsample/cuboid/2026-07-04_20-30-46_ppo_torch_tactile_box/checkpoints/best_agent.pt \
  --record_rollout \
  --record_rollout_env_id 0 \
  --record_rollout_steps 200
```

Rollout recording writes a compressed NPZ under `metrics/play_rollout` with action, processed action, joint state, object pose, gripper position, reward, and done arrays for the selected environment.

Run fixed-grid bucket evaluation over object start positions:

```bash
python scripts/reinforcement_learning/skrl/play_bucket.py \
  --task TacEx-VT-Downsample-Drawer-Occlusion-Cube \
  --num_envs 128 \
  --enable_cameras \
  --checkpoint logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt \
  --bucket_rounds 1 \
  --headless
```

Bucket evaluation writes detail and summary CSV files under the checkpoint run's `metrics/play_bucket` directory.

## Useful Checks

Lightweight syntax check:

```bash
python -m compileall source scripts tools
```

Isaac Lab test discovery:

```bash
./tacex.sh -p tools/run_all_tests.py --discover_only
```

Environment tests require Isaac Sim / Isaac Lab, GPU, and a compatible rendering setup:

```bash
./tacex.sh -p tools/run_all_tests.py --extension tacex_tasks
```

## Documentation

Project-specific architecture notes are in:

- `docs/PROJECT_OVERVIEW.md`
- `docs/ARCHITECTURE.md`
- `docs/DATA_FLOW.md`
- `docs/KNOWN_ISSUES.md`
- `docs/DECISIONS.md`

The original TacEx framework documentation remains under `docs/source`.

## Upstream TacEx

TacEx brings vision-based tactile sensors into Isaac Sim / Isaac Lab. The upstream framework includes:

- GPU-accelerated tactile RGB simulation via Taxim.
- Marker-motion simulation via FOTS.
- UIPC integration for GPU-accelerated incremental potential contact.
- FEM-based marker-motion simulation inspired by the ManiSkill-ViTac challenge.

## Citation

```bibtex
@article{nguyen2024tacexgelsighttactilesimulation,
      title={TacEx: GelSight Tactile Simulation in Isaac Sim -- Combining Soft-Body and Visuotactile Simulators},
      author={Duc Huy Nguyen and Tim Schneider and Guillaume Duret and Alap Kshirsagar and Boris Belousov and Jan Peters},
      year={2024},
      eprint={2411.04776},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.04776},
}
```

## Acknowledgements

This project builds on:

- [TacEx](https://github.com/DH-Ng/TacEx)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Taxim](https://github.com/Robo-Touch/Taxim)
- [FOTS](https://github.com/Rancho-zhao/FOTS)
- [UIPC](https://github.com/spiriMirror/libuipc)
- [ManiSkill-ViTac challenge](https://github.com/chuanyune/ManiSkill-ViTac2025)
