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

`TacEx-Sim2Real-Cube-Real-Alignment-v0` is the vision-only Franka cube task aligned to `20260711_214450_real_alignment_reference`. It uses a measured 5x5x5 cm target cube, the captured Franka joint state and gripper width, the cropped D435 model-input intrinsics, a 30 Hz camera/policy cadence, an image-aligned table and cube start region, and reference-centred appearance randomization.

```bash
conda run -n isaaclab_2.1.1 --no-capture-output python \
  scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-v0 \
  --num_envs 4 \
  --enable_cameras \
  --headless
```

The D435-to-Franka extrinsics were not measured in the reference capture. The task therefore retains the existing approximate third-person pose; perform hand-eye calibration before treating the camera pose as physically calibrated.

Run a checkpoint with the regular playback script:

```bash
python scripts/reinforcement_learning/skrl/play.py \
  --task TacEx-VT-Downsample-Drawer-Occlusion-Cube \
  --num_envs 128 \
  --enable_cameras \
  --checkpoint logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt
```

### Export a sim-to-real cube policy

Export the trained Cube vision-only actor as an end-to-end CPU TorchScript model:

```bash
conda run -n isaaclab_2.1.1 --no-capture-output python \
  scripts/reinforcement_learning/skrl/export_sim2real_grasp_jit.py \
  --task TacEx-Sim2Real-Cube-Grasp-v0 \
  --checkpoint logs/skrl/sim2real_cube_grasp/2026-06-26_23-14-35_ppo_torch_vision_only_resnet18/checkpoints/best_agent.pt \
  --num_envs 1 \
  --headless
```

The default output is `checkpoints/exported/policy_actor_e2e_best_agent.pt`, accompanied by a JSON metadata file. The model inputs are batched `action_history` `[N, 4]`, `proprio_obs` `[N, 15]`, and `wrist_rgb` `[N, H, W, 3]` as uint8 RGB. Although the observation key remains `wrist_rgb`, this Cube task uses a fixed third-person camera. Its output is the raw four-dimensional actor mean `[dx, dy, dz, gripper]`; real-robot code must still reproduce the environment-side scaling, IK, limits, watchdog, and emergency-stop behavior.

Run the Isaac-independent smoke test after exporting:

```bash
python scripts/reinforcement_learning/skrl/test_sim2real_grasp_policy_jit.py \
  --model logs/skrl/sim2real_cube_grasp/2026-06-26_23-14-35_ppo_torch_vision_only_resnet18/checkpoints/exported/policy_actor_e2e_best_agent.pt
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
