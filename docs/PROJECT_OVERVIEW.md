# Project Overview

## 1. 项目目标

TacEx 当前仓库的主线已由用户确认为 `occluded_grasping`：在 Isaac Sim / Isaac Lab 中构建可训练、可评估的遮挡视觉触觉机器人抓取实验平台。`source/tacex` 和 `source/tacex_assets` 提供 GelSight Mini 等 vision-based tactile sensor 的仿真输出与资产配置，作为当前主线的基础设施。

依据：`source/tacex/tacex/gelsight_sensor.py:GelSightSensor`；`source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py:GelSightMiniCfg`；`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxEnv`。

## 2. 当前任务

当前代码中的主要任务是 `occluded_grasping` 遮挡抓取：在 `Drawer-Occlusion` 或 `Self-Occlusion` 场景中，用 Franka Panda 末端夹爪抓取目标物体，并把物体稳定抬升到成功高度。

已确认任务注册矩阵支持：

- Scene：`Drawer-Occlusion`、`Self-Occlusion`
- Object：`Cylinder`、`Cube`、`Cuboid`、`SoftCylinder`、`SoftCube`、`SoftCuboid`
- Fusion/Policy 变体：V、T、VT、GelFusion、Alpha、GRU、Cross、Aux、Sparsh、Policy-Token-Transformer 等

依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_scene_object_fusion_matrix`。

## 3. 系统组成

| 组成 | 作用 | 依据 |
| --- | --- | --- |
| Isaac Lab extension | 组织 sensor、assets、tasks、uipc 扩展 | `source/*/config/extension.toml` |
| TacEx sensor core | 维护 GelSight 输出和触觉仿真更新 | `source/tacex/tacex/gelsight_sensor.py:GelSightSensor._update_buffers_impl` |
| TacEx assets | 提供 GelSight Mini、Franka、UR5 等资产配置 | `source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py:GelSightMiniCfg` |
| TacEx tasks | 注册和实现 RL tasks | `source/tacex_tasks/tacex_tasks/__init__.py` |
| Occluded grasping env | 核心遮挡抓取环境 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxEnv` |
| UR10 + Robotiq direct tasks | 从本机 IsaacLab 迁移的 UR10 + Robotiq 2F85/2F140 direct RL 环境 | `source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace/__init__.py`、`source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_gripper_close/__init__.py` |
| skrl scripts | 训练、回放、bucket 评估 | `scripts/reinforcement_learning/skrl/train.py:main`、`play.py:main`、`play_bucket.py:main` |
| logs | 保存 skrl 参数、checkpoint、评估 CSV | `logs/skrl/occluded_grasping/` |

## 4. 当前已实现功能

- Franka Panda + GS Mini gripper 任务配置：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.robot` 使用 `source/tacex_assets/tacex_assets/robots/franka/franka_gsmini_gripper_rigid.py:FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG`。
- 第三视角 RGB 相机：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.third_person_camera`。
- 四路触觉输入：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.gsmini_left`、`gsmini_right`、`gsmini_left_down`、`gsmini_right_down`。
- 5 维动作空间 `[dx, dy, dz, dyaw, gripper]`：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.action_space`、`OccludedGraspingVisionFourTactileBoxEnv._pre_physics_step`。
- 观测字典：`proprio_obs`、`third_resnet`、四路 tactile feature、critic privileged keys：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.observation_space`、`_get_observations`。
- 触觉 + 本体 baseline：`vt_tactile_box.py` 暴露 `proprio_obs` 和四路 tactile feature 给 actor，注册 task 包括 `TacEx-T-Drawer-Occlusion-Cube` 等矩阵组合。
- GelFusion 风格 RL 分支：`vt_gelfusion_box.py` 额外提供 8 维 `tactile_dynamic_stats`，`vt_gelfusion_policy.py` 实现 vision-led cross attention actor；注册 task 包括 `TacEx-GelFusion-Drawer-Occlusion-Cube` 和 `TacEx-GelFusion-Downsample-Drawer-Occlusion-Cube` 等矩阵组合。
- 奖励：reach、lift、success、drop-after-success、inner/down tactile contact：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_rewards`、`_compute_tactile_contact_rewards`。
- done：timeout 或 robot body 低于 `ground_height`；success 只统计不终止：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_dones`。
- skrl PPO 训练入口：`scripts/reinforcement_learning/skrl/train.py:main`。
- 普通 checkpoint 回放入口：`scripts/reinforcement_learning/skrl/play.py:main`。
- 固定物体位置 bucket 评估入口：`scripts/reinforcement_learning/skrl/play_bucket.py:main`。
- UR10 + Robotiq direct RL 任务：
  - `Isaac-UR10-Robotiq-Pick-Place-Direct-v0`
  - `Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0`
  - `Isaac-UR10-Robotiq-2F85-Third-Person-Pick-Place-Direct-v0`
  - `Isaac-UR10-Robotiq-2F85-Grasp-Direct-v0`
  - `Isaac-UR10-Robotiq-Gripper-Close-Direct-v0`
  - 2F85 pick-place 使用 27 维状态观测和 4 维动作 `[dx, dy, dz, gripper]`，`finger_joint` 是策略直接控制的夹爪主关节，其余 2F85 指关节由环境代码按 coupling rules 设置目标位置。
  - 2F85 third-person pick-place 变体是独立 `DirectRLEnv` 实现，便于单独改参数；新增 224x224 RGB `third_person_camera` 传感器，相机内参和位姿参考 `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_grasp_env.py`，默认 PPO policy 输入仍保持 27 维状态观测。
  - 资产已迁移到 `source/tacex_assets/tacex_assets/data/Robots/URRobotiq/`。

## 5. 部分实现功能

- 多策略实验矩阵已经存在，但配置和实现数量较多，推荐主线仍需确认。依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/*.yaml`、`vt_alpha_gru_policy.py`、`vt_cross_policy.py`、`vt_tactile_cross_alpha_aux_policy.py`。
- Auxiliary PPO 已实现可读取 `aux_loss` 等 policy outputs，但具体实验稳定性待确认。依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_aux_ppo_agent.py:PPOWithAuxHeadsLoss._update`。
- Soft object 注册和 deformable 分支存在，但 UIPC/FEM 运行链路在当前遮挡抓取主线中的依赖关系待确认。依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_scene_object_fusion_matrix`、`vt_box.py:_is_can_deformable`。
- DINO/Sparsh 触觉编码路径存在，但包含本机绝对路径，跨机器可复现性待确认。依据：`vt_box.py:_DEFAULT_DINO_ENCODER_ROOT`、`OccludedGraspingVisionFourTactileBoxCfg.tactile_dino_repo_path`。

## 6. 尚未实现或待确认功能

- 是否将通用 TacEx sensor framework 从 `occluded_grasping` 主线中拆分为独立仓库或继续作为内部基础设施维护：待确认。
- 是否去除 `occluded_grasping/__init__.py:_patch_skrl_runner_for_custom_models` 的 import-time monkey patch：待确认。
- 是否统一 `train.py`、`play.py`、`play_bucket.py` 的重复 cfg/checkpoint/custom component 加载逻辑：待确认。
- 是否清理 legacy task id、历史入口脚本和未使用 policy 文件：待确认。
- CPU-only GelSight Mini 路径是否可运行：待确认。依据：`source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py:GelSightMiniCfg.device` 默认 `cuda`。

## 7. 当前主要技术路线

当前主线是 Isaac Lab `DirectRLEnv` + skrl PPO：

1. `scripts/reinforcement_learning/skrl/train.py:main` 启动 Isaac Sim。
2. `import tacex_tasks` 触发 task 注册。
3. `gym.make(args_cli.task, cfg=env_cfg)` 创建环境。
4. 环境生成视觉、触觉、本体和 privileged critic 观测。
5. policy 读取视觉/触觉/proprio，critic 读取 privileged state。
6. PPO 更新 policy 和 value network。

依据：`scripts/reinforcement_learning/skrl/train.py:main`；`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py`；`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml`。

## 8. 当前基线方法

- Vision-only baseline：`source/tacex_tasks/tacex_tasks/occluded_grasping/vision_box.py:OccludedGraspingVisionOnlyBoxEnv`。
- Tactile-proprioception baseline：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_tactile_box.py:OccludedGraspingTactileProprioBoxEnv`，配置 `source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_tactile.yaml`。
- Vision + tactile feature concat/downsample baseline：已确认 checkpoint 目录 `logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/`。
- Alpha-GRU 策略：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_policy.py:OccludedGraspingVTAlphaGRUPolicy`，配置 `source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml`。
- Cross attention 策略：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_cross_policy.py:OccludedGraspingVisionTactileCrossAttentionPolicy`。
- Aux heads 策略：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_tactile_cross_alpha_aux_policy.py:OccludedGraspingVTTactileCrossAlphaAuxPolicy`。
- GelFusion-style 策略：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_policy.py:OccludedGraspingVTGelFusionPolicy`。

## 9. 当前主要实验变量

| 变量 | 已确认来源 |
| --- | --- |
| Scene | `source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_scene_object_fusion_matrix` |
| Object | `source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_scene_object_fusion_matrix` |
| Fusion/policy variant | `source/tacex_tasks/tacex_tasks/occluded_grasping/agents/*.yaml` |
| 视觉退化 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:_degrade_third_person_rgb` |
| 触觉 encoder | `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.tactile_encoder_type`、`_get_observations` |
| 触觉时间窗 | `source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml:tactile_time_window` |
| 奖励权重 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxCfg` |
| bucket 物体位置 | `scripts/reinforcement_learning/skrl/play_bucket.py:_build_bucket_positions` |

## 10. 当前项目状态

- 当前分支：`dev/0517`。
- 当前 HEAD：`48e8f0a add some pipeline`。
- 该提交相对前序提交大规模调整 `occluded_grasping` pipeline，统计为 53 files changed, 6169 insertions, 1447 deletions。依据：`git show --stat --oneline HEAD`。
- 当前工作区存在未提交改动，集中在 `source/tacex_tasks/tacex_tasks/occluded_grasping`、`scripts/reinforcement_learning/skrl/play.py` 和 `scripts/occluded_grasping/*`。依据：`git status --short`。
- 本文档只描述已从仓库和用户说明确认的信息；未提交改动的作者、意图和最终去留均为待确认。

## 11. 下一步建议

1. 先固定一个推荐训练入口、一个推荐评估入口和一个推荐 task id，减少 README/AGENTS/docs 之间的漂移。
2. 把 `train.py`、`play.py`、`play_bucket.py` 中重复的 cfg/checkpoint/custom component 逻辑抽成共享 helper。
3. 把 `vt_box.py` 中 scene/sensor encoding/reward/reset/logging 职责拆分前，先补充 smoke test 和文档化张量契约。
4. 将本机绝对路径改成配置项或环境变量，尤其是 DINO/Sparsh checkpoint 和 repo 路径。
5. 对已存在 checkpoint 和 metrics 建立 `docs/EXPERIMENTS.md` 中的实验索引。

## 12. 快速开始

已确认训练命令示例：

```bash
python scripts/reinforcement_learning/skrl/train.py --task TacEx-Alpha-GRU-Drawer-Occlusion-Cube --num_envs 4 --enable_cameras
```

已确认普通评估命令示例：

```bash
python scripts/reinforcement_learning/skrl/play.py --task TacEx-VT-Downsample-Drawer-Occlusion-Cube --checkpoint logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt --num_envs 16 --enable_cameras
```

已确认 bucket 评估命令示例：

```bash
python scripts/reinforcement_learning/skrl/play_bucket.py --task TacEx-VT-Downsample-Drawer-Occlusion-Cube --checkpoint logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt --num_envs 16 --enable_cameras
```

已确认轻量验证命令：

```bash
python -m compileall source scripts tools
./tacex.sh -p tools/run_all_tests.py --discover_only
```

UR10 + Robotiq 2F85 pick-place 训练示例：

```bash
python scripts/reinforcement_learning/skrl/train.py --task Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0 --num_envs 64 --headless
```

UR10 + Robotiq 2F85 third-person camera 训练示例：

```bash
python scripts/reinforcement_learning/skrl/train.py --task Isaac-UR10-Robotiq-2F85-Third-Person-Pick-Place-Direct-v0 --num_envs 64 --enable_cameras --headless
```

UR10 + Robotiq 2F85 pick-place checkpoint 回放示例：

```bash
python scripts/reinforcement_learning/skrl/play.py --task Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0 --checkpoint logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/checkpoints/best_agent.pt --num_envs 16
```
