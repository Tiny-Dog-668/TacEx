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
| `source/tacex_tasks/tacex_tasks/sim2real_grasp` | Franka vision-only sim2real Cube/Bottle、现实参考对齐和 privileged diagnostic 变体 | `sim2real_cube_grasp_env.py`、`sim2real_cube_real_alignment_env.py`、`sim2real_cube_real_alignment_privileged_env.py`、`agents/skrl_ppo_cube_real_alignment_cfg_resnet18.yaml` |
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

Clean/DR Real-Alignment Cube 的第四维使用总夹爪宽度增量控制。每个 30 Hz policy step 将 clipped action `g` 映射为 `delta_width=0.005*g m`，在上一缓存目标上累加并 clamp 到 `[0,0.08] m`；两个 60 Hz physics substep 都发送相同的 `left_target=right_target=desired_width/2`，不会在 `_apply_action()` 中重复累加。`action_history[3]` 保存请求的总宽度增量；当前 v9 的前三维为最大 `0.025 m` 的 XYZ processed command。Privileged 诊断变体显式保留旧 `0.002 m` 夹爪尺度，基础 Cube/Cylinder 保留 legacy per-finger 行为。

Real-Alignment 的控制 TCP 与奖励中心使用两个明确不同但几何一致的来源。Differential IK/Jacobian 使用 `panda_hand` 沿本地 z 轴固定偏移 `0.1034 m`；reach reward、critic 的 gripper position/target distance 以及近台面 `dz` gate，使用左右 `panda_*finger` link 各沿本地 z 轴偏移 `0.045 m` 后的世界坐标中点。这样 reward 会跟随两侧夹指真实位姿，而不再把固定 `panda_hand+0.107 m` 当作抓取中心。基础 Cube/Cylinder 通过默认 hook 保留原中心语义。

`sim2real_cube_real_alignment_privileged_env.py` 提供独立的无相机上界任务。其 `_setup_scene()` 不创建 `TiledCamera`，配置关闭继承链中的 ResNet18 构造；Actor 使用 28-D 向量：`proprio 15 + history 4 + cube position 3 + fingertip-midpoint position 3 + relative target 3`。位置从 world position 减去 `scene.env_origins`，因此多环境共享 robot-root-aligned 数值范围。Critic 保持 49-D privileged contract。这个任务不进入 strict vision encoder/export allowlist，checkpoint 只用于仿真诊断。

## 7. 视觉模块

第三视角相机配置：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.third_person_camera`。

现实参考对齐相机配置：`sim2real_cube_real_alignment_env.py:Sim2RealCubeRealAlignmentEnvCfg.wrist_camera`。现实 D435 原始内参为 `fx=604.897400, fy=605.085815, cx=320.980103, cy=247.913223 @ 640x480`；部署侧裁剪 `x=[100,500), y=[34,432)` 后将 400x398 双线性缩放到 224x224，得到策略有效内参 `fx=338.742544, fy=340.550811, cx=123.748857, cy=120.393372`。Omniverse 4.5 不支持非方形像素和 aperture offset，因此仿真先用 centered `K_native=[300,0,112;0,300,112;0,0,1]` 直接渲染 224x224 coverage view，再以固定 batched GPU affine warp 映射到目标 K；300 px native focal 能覆盖全部目标光线而不依赖 padding。Panda base 的 world 平移为 `(0,0,0.02) m`，以匹配真机 20 mm 垫高。当前 `base_T_camera_color_optical` 为 `pos=(1.166091088407,0.035901608197,0.514200335898)`、ROS optical `rot(wxyz)=(0.378248136306,-0.604227000834,-0.586824979121,0.384024117374)`；它来自输入的 ROS `xyzw=(-0.604227000834,-0.586824979121,0.384024117374,0.378248136306)`。由于 TiledCamera 是独立 world prim，实际仿真 world camera position 为 `(1.166091088407,0.035901608197,0.534200335898)`，从而仍严格保持上述 base 外参。物理保持 60 Hz；`decimation=2`、camera `update_period=1/30 s` 和 render interval 共同使 camera/render/policy/action/observation/reward/history 对齐为 30 Hz。episode 为 5 秒，即 150 个 policy steps。

随机化 profile 在同一模块内通过配置继承分离：`Sim2RealCubeRealAlignmentEnvCfg` 是 clean 基线，关闭所有视觉、光照和场景颜色随机化，只保留 cube XY reset；`Sim2RealCubeRealAlignmentDREnvCfg` 继承 clean 配置并开启 curriculum DR。DR 的 plate/backdrop/DomeLight 中心值直接复用 Clean 常量，scale=0 时 camera `delta XYZ/RPY=0`、GPU focal/principal-point warp 为 identity、后处理为 identity、材质和光照与 Clean 完全一致。DR 每环境每 episode 缓存各扰动参数；Gaussian pixel noise 每帧重采样。`TiledCamera.set_world_poses(env_ids=...)` 支持多环境外参，图像后处理在 `[N,3,224,224]` 上批处理。stage-global DomeLight 只在 full-batch reset 更新，global ground 固定。对应 task id 分别为 `TacEx-Sim2Real-Cube-Real-Alignment-v0` 和 `TacEx-Sim2Real-Cube-Real-Alignment-DR-v0`，DR 使用独立 Agent YAML 和 `sim2real_cube_real_alignment_dr` 日志目录。

DR curriculum 使用 `DirectRLEnv.common_step_counter`（每个 30 Hz outer/policy step 加一，与 `num_envs` 无关）：0–100k step 的范围系数为 0，使物体位置课程先扩展完成；100k–220k 从 0 线性增到 1.0，220k–300k 保持 full range。当前 scale 同时写入 `info/dr_curriculum_scale`。`dr_curriculum_step_offset` 是显式 resume hook；当前训练入口不会自动从 checkpoint 推导该值。

Sim2real Cube 的 ResNet18 使用 `ResNet18_Weights.IMAGENET1K_V1`，参数冻结且模块始终处于 `eval()`。训练入口在每个 run 的 `params/` 下保存一次完整 encoder state artifact 和 manifest；encoder 不重复写入每个 PPO checkpoint。当前 v9 manifest 额外绑定 Real-Alignment 的逐维 action/history scale、ROS optical 相机位姿、原始内参/裁剪、D435 serial、物体完整范围与位置课程、1 mm 木板、桌面碰撞过滤和无倾角 success。训练 resume、普通 play、bucket play、rollout 和 exporter 都验证该 contract；v7、v8、v9 分别严格接受 10/2 mm、25/2 mm、25/5 mm 及各自保存的 reset bounds，防止同 shape checkpoint 被错误部署。

Real-Alignment 木板以一个 kinematic 1 mm Cuboid 表示，顶面为 `z=0.001 m`。每环境木板附加一个 ContactSensor，filter 对象为 `panda_link1–7`、`panda_hand`、`panda_leftfinger` 和 `panda_rightfinger`；其历史长度为 2，对应一个 policy step 内的两个 physics substeps。奖励读取 `[N,2,1,10,3]` filtered force matrix 的最大法向力，超过 1 N 时返回固定 `-10`，但不修改 observation 或 done。

Real-Alignment 在 `_get_dones()` 后按 policy step 记录该步并行环境中完成和成功的 episode 数。两个长度为 200 的环形缓冲保存最近 200 个 policy steps 的计数，窗口成功率为 `sum(successes)/sum(completed)`；无完成 episode 时为 0。日志同时发布 `recent_success_rate`、`episode_success_rate_window`、累计成功率和窗口/累计计数，并在每 200 步的奖励行中一起打印。统计不进入 observation、reward 或 policy contract。

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

Sim2real RGB run 额外保存：

- `params/vision_encoder_resnet18.pt`：训练实际使用的 frozen encoder state dict。
- `params/vision_encoder_resnet18.json`：架构、权重 ID、eval/frozen 状态、torch/torchvision 版本、ImageNet normalization、artifact/state SHA-256，以及 task/agent/env/action/history/timing policy contract。

Cube train resume/play/play_bucket/export 均要求 checkpoint run manifest。Exporter 恢复 run 自己的 `params/agent.yaml` 和 `params/env.pkl`，并校验 task 与四份 agent/env artifact hash，避免当前 registry 或错 task 改变模型语义。新 Cube checkpoint 必须声明 `models.policy.output: "tanh(ACTIONS)"`；旧 checkpoint 即使参数 shape 可加载，也会被拒绝 resume/play/export。

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

Sim2real Cube 批量诊断入口：

- `scripts/reinforcement_learning/skrl/collect_sim2real_cube_rollouts.py`：从 export metadata 找到并恢复 run 自己的 `env.pkl`，校验 TorchScript SHA-256，在 CPU 执行部署 Actor，并保存全部并行环境的 `T x N` action/state transition。正常路径继续拒绝旧 contract；仅显式 `--allow_legacy_contract` 可复现旧 run 用于离线诊断。
- `scripts/reinforcement_learning/skrl/analyze_sim2real_cube_rollouts.py`：不启动 Isaac Sim，读取 collector NPZ，生成逐 transition CSV、分 episode step 的 action/state PNG 和 analysis JSON。
- `cylinder_grasping_vision_only_resnet18.py` 中 `_last_requested_arm_command`、`_last_ik_arm_command`、`_last_privileged_dz_gate_active` 是只读诊断缓存，不改变控制输出。

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

## 17. Real-Alignment RMA 路线

RMA Teacher/Student 使用新增 task，不修改 Clean、DR 或旧 Privileged：

- Teacher：无相机，外部输入为15维本体、4维 history、3维 cube XYZ 和2维真实左右指接触；`RMAActorCore` 从关节角执行 Panda FK，追加3维指尖中点位置与3维 cube-to-gripper 相对向量，形成30维特征并输出4维 tanh action；现有 `train.py` 运行 PPO 200k。
- Student：环境输出 `wrist_rgb[224,224,3] uint8`、15维本体、4维 history，以及仅用于 loss 的 cube XYZ 和左右指接触标签。
- `train_rma_student.py` 冻结 ResNet18 layer4 与 Teacher Actor，只训练共享 spatial-softmax adaptation head 的位置与接触分支，共100k步；学生向 Actor 输入 sigmoid 接触概率以保留动作 loss 梯度，并额外用上一拍环境 action 约束 Student 输出的 action-rate smoothness。
- `TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-DR-v0` 是独立的 Student-only profile：复用通用 DR 的相机外参、GPU 内参 warp、颜色/模糊/噪声、板/背景和 DomeLight 扰动，但 `dr_curriculum_enabled=false`，因此从第一个 update 起始终为 full scale。它不改变 Teacher、接触标签、物理、奖励、动作或部署 TorchScript 输入；Student artifact 会记录 Clean 或 DR task，resume 只能在同一 profile 内进行。
- 当前 Real-Alignment visual nominal 基于用户提供的裁剪 D435 图作首轮人工对齐：保留测得的相机外参/内参，将近黑桌面、近黑背景和较暗中性 DomeLight 作为中心；DR 仅覆盖小残差（camera XYZ ±1.5 mm、RPY ±0.5°、focal ±1%、principal point ±1 px，以及受限的曝光、色彩、模糊和噪声）。画面中的真实导轨尚未建模，不能由 camera/photometric DR 代替；这是待确认的 scene-geometry gap。
- `evaluate_rma.py` 使用固定10x10 XY 网格；`export_rma_student_jit.py` 导出不含 privileged 输入的端到端 Actor，并对可用 CUDA 执行 CPU/CUDA reload 前向一致性验证；导出文件可由 `map_location="cpu"` 或 `"cuda:0"` 加载。
- `play_rma_student.py` 专门读取蒸馏 Student artifact，而不实例化 skrl Agent；它根据 checkpoint 的 task 选择 Clean 或全强度 DR 环境，校验 Teacher 环境 contract 与 encoder/Actor 哈希，在完整随机 XY 范围回放确定性 Student 动作，并写 CSV/JSON 成功率摘要。
- `train_rma_student.py --save_start_frame --start_frame_count N` 会在首次 reset 后将最多 `N` 个不同环境的 `wrist_rgb` 写到本次 run 的 `camera_frames/`；这是 Student 真正接收的 224×224 uint8 输入，DR task 的每张图保留各自 episode 的相机和外观随机化，不额外推进环境。
- 两个 RMA task 单独注册 cube ContactSensor，以0.2 N阈值生成左右二值接触；Teacher 与 Student 共用 `contact_reward_weight=3.0` 和 `single_contact_reward_fraction=1.0`，并在 reward 中加入 `rma_action_rate_penalty_weight=0.05` 的动作变化惩罚。惩罚按当前 `action_history` 与上一拍 `action_history` 的差计算，再除以环境动作尺度 `[0.05,0.05,0.05,0.01]`，不修改 Clean/DR/旧 Privileged reward。
- RMA 的 success 只作为奖励和统计条件，不触发 done；episode 只因 timeout 或严重机器人穿地碰撞结束，窗口成功率按 episode 内是否曾达到 success 统计。
- RMA 保持 Clean 的 `10 mm/step` 夹爪总宽度动作尺度，但单独降低 Panda finger actuator 到 `effort_limit_sim=40`、`stiffness=400`、`damping=40`。
- Teacher `params/rma_manifest.json` 绑定动作、位置/接触坐标语义、30维 feature order、Panda FK、归一化、RMA finger actuator、动作平滑 reward 和配置哈希；Student checkpoint 绑定 Teacher/encoder/Actor 哈希。物体与末端姿态不进入 Actor；RMA v6 及更早 Teacher manifest 与当前 v7 训练目标不兼容。
