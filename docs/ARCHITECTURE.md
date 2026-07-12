# Architecture

## 1. 系统总体结构

TacEx 采用 Isaac Lab extension 结构组织代码：

1. `source/tacex` 提供 GelSight sensor core 和触觉仿真方法。
2. `source/tacex_assets` 提供传感器、机器人和物体资产配置。
3. `source/tacex_tasks` 提供 RL task 注册和环境实现。
4. `source/tacex_uipc` 提供 UIPC/pyuipc 集成，当前遮挡抓取主线依赖关系待确认。
5. `scripts/reinforcement_learning/skrl` 提供 skrl 训练、回放和评估脚本。
6. `logs/skrl/occluded_grasping` 保存训练参数、checkpoint 和评估 CSV。

## 2. 目录结构

| 目录 | 职责 | 关键文件 |
| --- | --- | --- |
| `source/tacex/tacex` | 核心触觉传感器和仿真接口 | `gelsight_sensor.py`、`gelsight_sensor_cfg.py`、`simulation_approaches/*` |
| `source/tacex_assets/tacex_assets` | 资产配置 | `sensors/gelsight_mini/gsmini_cfg.py`、`robots/franka/franka_gsmini_gripper_rigid.py` |
| `source/tacex_tasks/tacex_tasks` | RL task 包 | `__init__.py`、`occluded_grasping/*` |
| `source/tacex_tasks/tacex_tasks/occluded_grasping` | 遮挡抓取环境、策略、agent YAML | `vt_box.py`、`vision_box.py`、`agents/ppo_vt_alpha_gru.yaml` |
| `source/tacex_tasks/tacex_tasks/sim2real_grasp` | Franka vision-only sim2real Cube/Bottle 和现实参考对齐变体 | `sim2real_cube_grasp_env.py`、`sim2real_cube_real_alignment_env.py`、`agents/skrl_ppo_cube_real_alignment_cfg_resnet18.yaml` |
| `source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace` | UR10 + Robotiq pick-place/grasp direct RL 环境 | `ur10_robotiq_pick_place_env.py`、`ur10_robotiq_2f85_pick_place_env_cfg.py`、`ur10_robotiq_2f85_third_person_pick_place_env.py`、`agents/skrl_ppo_cfg.yaml` |
| `source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_gripper_close` | UR10 + Robotiq 夹爪关闭 direct RL 环境 | `ur10_robotiq_gripper_close_env.py`、`agents/rsl_rl_ppo_cfg.py` |
| `source/tacex_assets/tacex_assets/data/Robots/URRobotiq` | 迁移的 UR10 + Robotiq USD 资产 | `ur10_robotiq_2f85.usda`、`ur10_robotiq_140.usda`、`ur10_robotiq_f140.usda` |
| `scripts/reinforcement_learning/skrl` | skrl CLI 入口 | `train.py`、`play.py`、`play_bucket.py` |
| `scripts/occluded_grasping` | 分析/辅助脚本 | bbox、bucket、occlusion predictor 相关脚本 |
| `tools` | 测试工具 | `run_all_tests.py` |
| `docs` | 项目长期维护文档 | 本文件及同目录文档 |

## 3. 训练入口

训练入口是 `scripts/reinforcement_learning/skrl/train.py:main`。

主要调用关系：

1. `AppLauncher` 启动 Isaac Sim。
2. `import tacex_tasks` 触发 task 注册。
3. `hydra_task_config(args_cli.task, agent_cfg_entry_point)` 读取 env 和 agent cfg。
4. `gym.make(args_cli.task, cfg=env_cfg, render_mode=...)` 创建 Isaac Lab 环境。
5. `SkrlVecEnvWrapper` 包装环境。
6. 根据 agent/model class 是否为 `module:Class`，选择 skrl Runner 或手动 custom agent/trainer 路径。

关键 helper：`train.py:_process_cfg`、`train.py:_load_component`。

## 4. 评估入口

普通评估入口是 `scripts/reinforcement_learning/skrl/play.py:main`。

职责：

- 读取 `--task`、`--checkpoint`、`--num_envs` 等 CLI 参数。
- 在 checkpoint 未指定时查找 skrl 日志目录中的 checkpoint。
- 从 checkpoint 附近的 `params/env.yaml`、`params/agent.yaml` 恢复配置。
- 创建环境并加载 agent。
- 循环执行 `agent.act` 和 `env.step`，写 play metrics。

固定位置评估入口是 `scripts/reinforcement_learning/skrl/play_bucket.py:main`。

新增职责：

- `play_bucket.py:_build_bucket_positions` 生成固定物体位置网格。
- `play_bucket.py:_write_bucket_object_positions` 将目标物体写入环境。
- 输出 bucket trial CSV 和 summary CSV。

## 5. 环境模块

核心配置类：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxCfg`。

核心环境类：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxEnv`。

继承关系：

- `OccludedGraspingVisionFourTactileBoxCfg(DirectRLEnvCfg)`
- `OccludedGraspingVisionFourTactileBoxEnv(DirectRLEnv)`

生命周期方法：

- `_setup_scene`
- `_pre_physics_step`
- `_apply_action`
- `_get_observations`
- `_get_rewards`
- `_get_dones`
- `_reset_idx`

视觉-only 基线：`source/tacex_tasks/tacex_tasks/occluded_grasping/vision_box.py:OccludedGraspingVisionOnlyBoxEnv`。

触觉 + 本体 baseline：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_tactile_box.py:OccludedGraspingTactileProprioBoxEnv`，继承 full VT 环境但从 actor policy observation 中移除 `third_resnet`。

## 6. 机器人与控制模块

机器人资产来源：`source/tacex_assets/tacex_assets/robots/franka/franka_gsmini_gripper_rigid.py:FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG`。

任务内配置位置：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.robot`。

控制方式：

- policy 输出 5 维 action。
- `_pre_physics_step` 将 `[dx, dy, dz, dyaw, gripper]` 变成 IK command `[dx, dy, dz, 0, 0, dyaw]`。
- `_apply_action` 调用 `DifferentialIKController.compute` 得到 arm joint target，并用第 5 维 action 更新夹爪 joint target。

UR10 + Robotiq 2F85 direct task 的控制方式：

- `source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace/ur10_robotiq_pick_place_env.py` 使用 `DifferentialIKController` 控制 UR10 末端位姿。
- action space 为 4，语义是 `[dx, dy, dz, gripper]`。
- `finger_joint` 是 Robotiq 2F85 的策略直接控制关节。
- 原 2F85 pick-place 任务的其他 2F85 指关节通过 `UR10Robotiq2F85PickPlaceEnvCfg.gripper_coupling_rules` 由环境代码设置跟随目标，而不是 policy 独立输出；third-person 独立变体在自己的 cfg 文件中单独定义同类 coupling rules。
- 2F85 pick-place 默认 USD 路径为 `source/tacex_assets/tacex_assets/data/Robots/URRobotiq/ur10_robotiq_2f85.usda`，可用 `UR10_ROBOTIQ_2F85_USD_PATH` 覆盖。
- `UR10Robotiq2F85ThirdPersonPickPlaceEnv` 是独立 `DirectRLEnv` 实现，不继承 `UR10RobotiqPickPlaceEnv` 或 `UR10Robotiq2F85PickPlaceEnvCfg`，便于后续单独改环境参数。

## 7. 视觉模块

第三视角相机配置：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.third_person_camera`。

现实参考对齐相机配置：`sim2real_cube_real_alignment_env.py:Sim2RealCubeRealAlignmentEnvCfg.wrist_camera`。该配置把 D435 原始 640x480 图像的中央 480x480 crop 等效为直接 224x224 渲染，使用变换后的内参 `fx=282.4603, fy=282.2679, cx=113.1917, cy=116.3019`，并将 camera/policy 周期设为 30 Hz。当前图像拟合位姿为 world `pos=(1.90,0.0,0.468)`、光轴向下约 15°；Omniverse 4.5 不支持非方形像素和 aperture offset，会使用平均焦距和图像中心主点。该位姿仍是图像推算值，待 hand-eye calibration。

视觉数据生成：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_observations` 从 `self.third_person_camera.data.output["rgb"]` 读取 RGB。

UR10 + Robotiq 2F85 third-person 变体的相机配置在 `source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace/ur10_robotiq_2f85_third_person_pick_place_env.py:UR10Robotiq2F85ThirdPersonPickPlaceEnvCfg.third_person_camera`。该相机使用 224x224 RGB 输出，内参和位姿参考 `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_grasp_env.py:Sim2RealCubeGraspEnvCfg.wrist_camera`。当前变体默认不把 RGB 加入 policy observation，因此现有 `agents/skrl_ppo_cfg.yaml` 的 MLP 输入仍是 `STATES`。该文件内同时独立定义 2F85 的 USD 路径、coupling rules、初始关节、噪声、夹爪目标角度、桌面、物体、奖励和 camera 参数。

视觉退化：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._degrade_third_person_rgb`。

视觉编码：`OccludedGraspingVisionFourTactileBoxEnv.__init__` 初始化 frozen ResNet18；`_get_observations` 输出 `third_resnet`。torchvision 不可用时，代码会输出零向量，风险记录见 `docs/KNOWN_ISSUES.md`。

## 8. 触觉模块

传感器实现：`source/tacex/tacex/gelsight_sensor.py:GelSightSensor`。

GelSight Mini 配置：`source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py:GelSightMiniCfg`。

任务内四路触觉：

- `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.gsmini_left`
- `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.gsmini_right`
- `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.gsmini_left_down`
- `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.gsmini_right_down`

触觉观测输出 key：

- `tactile_left_depth_resnet`
- `tactile_right_depth_resnet`
- `tactile_left_down_depth_resnet`
- `tactile_right_down_depth_resnet`
- GelFusion 分支额外输出 `tactile_dynamic_stats`，维度为 8，对应四路触觉相邻帧二值差分的 mean/variance。
- `T` baseline 的 actor 只读取上述四路触觉 feature 和 `proprio_obs`；critic 仍读取 privileged state。

依据：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_observations`。

## 9. 本体状态模块

本体观测 key：`proprio_obs`。

默认维度：18。依据：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.observation_space`。

生成方式：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_observations` 拼接 robot joint position 和 joint velocity。

## 10. 视触融合模块

已确认策略实现包括：

- Alpha-GRU：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_policy.py:OccludedGraspingVTAlphaGRUPolicy`。
- Cross attention：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_cross_policy.py:OccludedGraspingVisionTactileCrossAttentionPolicy`。
- Aux heads：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_tactile_cross_alpha_aux_policy.py:OccludedGraspingVTTactileCrossAlphaAuxPolicy`。
- GelFusion-style：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_policy.py:OccludedGraspingVTGelFusionPolicy`，视觉作为 query，视觉和四路触觉静态特征作为 key/value，动态触觉统计单独拼接。

代表配置：`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml`。其中 policy 输入包括 `third_resnet`、四路 tactile keys 和 `proprio_obs`，并包含 `tactile_time_window`、`tactile_gru_hidden_dim`、`fused_dim` 等字段。

## 11. Actor

Actor 由 skrl policy model 实现。以 `ppo_vt_alpha_gru.yaml` 为例：

- class：`tacex_tasks.occluded_grasping.vt_alpha_gru_policy:OccludedGraspingVTAlphaGRUPolicy`
- vision key：`third_resnet`
- tactile keys：四路 `*_depth_resnet`
- vector/proprio key：`proprio_obs`
- action 输出维度：由环境 `action_space = 5` 决定

关键函数：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_policy.py:OccludedGraspingVTAlphaGRUPolicy.compute`。

## 12. Critic

Critic 多数使用 privileged state，而不是原始视觉/触觉 feature。

代表配置：`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml:models.value.network.input`。

环境提供的 critic keys 包括：

- `critic_can_position`
- `critic_can_quat`
- `critic_can_lin_vel`
- `critic_can_ang_vel`
- `critic_gripper_position`
- `critic_gripper_quat`
- `critic_gripper_lin_vel`
- `critic_gripper_ang_vel`
- `critic_target_relative_position`
- `critic_target_distance`

定义和生成位置：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.observation_space`、`OccludedGraspingVisionFourTactileBoxEnv._get_observations`。

## 13. 奖励与终止条件

奖励函数：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_rewards`。

奖励项：

- reach reward
- lift reward
- success reward
- drop-after-success penalty
- inner tactile contact reward
- down tactile contact reward

触觉 contact reward：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._compute_tactile_contact_rewards`。

终止函数：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_dones`。

当前 done 条件：

- episode timeout
- robot body 低于 `ground_height`

当前 success 语义：success 会用于 reward 和统计，但不直接终止 episode。

## 14. 配置系统

任务注册中通过 `gym.register(..., kwargs={"env_cfg_entry_point": ..., "skrl_cfg_entry_point": ...})` 绑定环境配置和 agent YAML。依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_task`。

训练脚本通过 `hydra_task_config(args_cli.task, agent_cfg_entry_point)` 加载配置。依据：`scripts/reinforcement_learning/skrl/train.py:main`。

代表 agent YAML：`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml`。

## 15. 模型保存与加载

训练日志和模型默认位于 `logs/skrl/occluded_grasping/`。

已确认存在 checkpoint：

```text
logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt
```

同一实验目录下存在：

- `params/agent.yaml`
- `params/env.yaml`
- `metrics/play/play_metrics_20260531_131229.csv`
- `metrics/play_bucket/play_bucket_20260531_152044.csv`
- `metrics/play_bucket/play_bucket_20260531_152044_summary.csv`

普通评估加载路径：`scripts/reinforcement_learning/skrl/play.py:main`。

Bucket 评估加载路径：`scripts/reinforcement_learning/skrl/play_bucket.py:main`。

## 16. 关键调用关系

```text
train.py / play.py / play_bucket.py
  -> import tacex_tasks
    -> tacex_tasks.__init__.py:import_packages
      -> occluded_grasping.__init__.py:_register_scene_object_fusion_matrix
        -> gym.register(task_id, entry_point, env_cfg_entry_point, skrl_cfg_entry_point)
  -> hydra_task_config(task_id, agent_cfg_entry_point)
  -> gym.make(task_id, cfg=env_cfg)
    -> OccludedGraspingVisionFourTactileBoxEnv.__init__
      -> _setup_scene
      -> DirectRLEnv loop
        -> _pre_physics_step
        -> _apply_action
        -> _get_observations
        -> _get_rewards
        -> _get_dones
        -> _reset_idx
  -> SkrlVecEnvWrapper
  -> skrl Runner / custom PPO agent
```
