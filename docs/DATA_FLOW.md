# Data Flow

## 1. 总体数据流

```text
policy action
  -> OccludedGraspingVisionFourTactileBoxEnv._pre_physics_step
  -> OccludedGraspingVisionFourTactileBoxEnv._apply_action
  -> Isaac Sim physics/render/sensor update
  -> OccludedGraspingVisionFourTactileBoxEnv._get_observations
  -> skrl policy/value model
  -> OccludedGraspingVisionFourTactileBoxEnv._get_rewards
  -> OccludedGraspingVisionFourTactileBoxEnv._get_dones
```

训练入口：`scripts/reinforcement_learning/skrl/train.py:main`。

评估入口：`scripts/reinforcement_learning/skrl/play.py:main`、`scripts/reinforcement_learning/skrl/play_bucket.py:main`。

核心环境：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxEnv`。

## 2. 原始环境观测

环境观测由 `vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_observations` 构建，并以 `{"policy": obs}` 形式返回给 Isaac Lab / skrl wrapper。

配置定义位于 `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.observation_space`。

主要观测组：

- policy 可用：`proprio_obs`、`third_resnet`、四路 tactile feature。
- critic privileged：object pose/vel、gripper pose/vel、target relative position/distance。
- 额外统计/aux 输出：随派生环境和策略变体变化，需逐个 YAML 与 env class 确认。

`TacEx-T-{Scene}-{Object}` 触觉 + 本体 baseline 由 `vt_tactile_box.py` 提供，actor policy observation 会移除 `third_resnet`，只保留 `proprio_obs` 和四路 tactile feature；critic privileged keys 不变。

## 3. 视觉数据流

现实参考对齐 Cube 分支使用相同的 frozen ResNet18 编码路径，但传感器和时间契约为：

```text
D435 real: 640x480 RGB @ 30 Hz
  -> crop x=[80, 560), y=[0, 480)
  -> resize 480x480 to 224x224

aligned simulation: direct 224x224 render with transformed crop intrinsics @ 30 Hz
  -> reference-centred brightness/contrast/gamma/blur/noise
  -> ImageNet normalization
  -> frozen ResNet18 in eval mode (BatchNorm running statistics never update)
  -> wrist_resnet [N, 512]
```

对应 task 为 `TacEx-Sim2Real-Cube-Real-Alignment-v0`；`proprio_obs [N,15]` 和 `action_history [N,4]` 保持不变，policy 输出仍是 `[dx,dy,dz,gripper]`。Cube Actor 在 policy model 内执行 `tanh`，因此 deterministic mean 在训练、play 和 export 中统一位于 `[-1,1]`。相机外参没有出现在现实参考文件中，当前仅为近似对齐。

动作历史时序为：

```text
actor action a_t
  -> finite check / action_scale / clamp / optional action noise
  -> processed_action p_t
  -> observation history for transition: action_history_{t+1} = detach(p_t)
  -> privileged near-table/object-XY dz gate modifies the IK-only arm command
  -> IK and gripper control during transition t -> t+1
  -> observation o_{t+1}
```

`action_history` 记录的是与该 transition 对应、privileged near-table `dz` gate 之前的 processed command；它不一定等于最终送入 IK 的 `dz`。sim2real Cube 配置的 action noise 为 0，因此范围为 `[-0.05,0.05]`。旧实现从 `prev_actions` 取值，会额外延迟一个决策周期。

1. `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.third_person_camera` 创建第三视角 `TiledCameraCfg`。
2. `_get_observations` 读取 `self.third_person_camera.data.output["rgb"]`。
3. RGB 被转为 float 并归一化到 `[0, 1]`。
4. `_degrade_third_person_rgb` 可进行 downsample 或 gaussian blur。
5. `self._resnet18` 将视觉输入编码为 `third_resnet`。
6. 若 torchvision 或 encoder 不可用，代码路径会使用零向量，具体运行风险见 `docs/KNOWN_ISSUES.md`。

关键路径：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_observations`。

## 4. 四路触觉数据流

四路触觉传感器配置：

- `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.gsmini_left`
- `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.gsmini_right`
- `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.gsmini_left_down`
- `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.gsmini_right_down`

触觉仿真实现：`source/tacex/tacex/gelsight_sensor.py:GelSightSensor._update_buffers_impl`。

数据流：

1. GelSight sensor 输出 `data.output["tactile_rgb"]`。
2. `_get_observations` 将 tactile RGB 转为 NCHW float。
3. 根据 `tactile_encoder_type` 选择 encoder：
   - `resnet`：ResNet18 feature 到 256 维。
   - `cnn`：轻量 ConvNet 到 256 维。
   - `dino`：`SparshFrozenEncoder` 对相邻帧 pair 编码，再经 `project_sparsh_features` 投影；若拼接多个时间 pair，维度会随配置变化。
4. 输出四路 tactile keys：
   - `tactile_left_depth_resnet`
   - `tactile_right_depth_resnet`
   - `tactile_left_down_depth_resnet`
   - `tactile_right_down_depth_resnet`
5. GelFusion 分支 `vt_gelfusion_box.py:OccludedGraspingVTGelFusionBoxEnv` 额外读取当前四路 tactile RGB，与上一帧做绝对差分、阈值二值化，并输出 `tactile_dynamic_stats`。维度为 8，顺序为 left/right/left_down/right_down 每路 `[mean, variance]`。
6. 触觉 + 本体 baseline `vt_tactile_box.py:OccludedGraspingTactileProprioBoxEnv` 沿用同样四路 tactile feature，但 actor 不读取 `third_resnet`。

## 5. 本体状态数据流

本体观测 key：`proprio_obs`。

默认维度：18。依据：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.observation_space`。

生成位置：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_observations`。

组成：robot joint position 与 joint velocity 的拼接。具体 joint 顺序由 `self._robot` 的 articulation data 决定，当前文档不额外推断。

## 6. 跨模态融合

代表 Alpha-GRU 配置：`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml`。

已确认输入：

- `vision_key: third_resnet`
- 四路 tactile keys
- `proprio_key: proprio_obs`
- `tactile_time_window: 10`
- `tactile_gru_hidden_dim: 64`
- `tactile_latent_dim: 128`
- `fused_dim: 256`

代表实现：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_policy.py:OccludedGraspingVTAlphaGRUPolicy.compute`。

其他融合策略：

- Cross attention：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_cross_policy.py:OccludedGraspingVisionTactileCrossAttentionPolicy.compute`。
- Aux heads：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_tactile_cross_alpha_aux_policy.py:OccludedGraspingVTTactileCrossAlphaAuxPolicy.compute`。
- GelFusion-style：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_policy.py:OccludedGraspingVTGelFusionPolicy.compute`，使用视觉 query 对视觉+触觉静态特征做 cross attention，再拼接原始视觉特征、`tactile_dynamic_stats` 和 `proprio_obs`。

## 7. Actor 数据流

Actor 输入来自环境 observation dict 中的 policy keys。以 `ppo_vt_alpha_gru.yaml` 为例：

```text
third_resnet
四路 tactile feature
proprio_obs
  -> OccludedGraspingVTAlphaGRUPolicy.compute
  -> skrl Gaussian policy distribution
  -> 5D action
```

Actor 输出维度由环境 `action_space = 5` 决定。依据：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.action_space`。

## 8. Critic 数据流

Critic 使用 privileged keys。以 `ppo_vt_alpha_gru.yaml` 为例，value network input 包含：

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

生成位置：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_observations`。

## 9. 动作到环境

动作定义：`[dx, dy, dz, dyaw, gripper]`。

路径：

1. skrl policy 产生 5D action。
2. `vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._pre_physics_step` 清理 NaN/Inf、缩放动作、加入 action noise，并构造 IK command `[dx, dy, dz, 0, 0, dyaw]`。
3. `vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._apply_action` 调用 differential IK 得到 arm joint target。
4. `_apply_action` 使用第 5 维 action 按 `gripper_step_size` 增量更新夹爪。
5. `_robot.set_joint_position_target(joint_pos_des)` 写入关节目标。

## 10. 奖励计算

奖励函数：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_rewards`。

已确认奖励项：

- `reach_reward = 1 - tanh(distance / sigma)`
- `lift_reward` 根据物体高度从 `lift_reward_start_height` 到 `success_height` 的进度计算
- `success_reward` 在物体高度达到 `success_height` 后触发
- `drop_after_success` 在已成功后掉落时惩罚
- inner/down tactile contact reward 由触觉 RGB 差分 contact bits 给出

触觉 contact 计算：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._compute_tactile_contact_rewards` 使用 reset-time tactile RGB baseline 与当前 tactile RGB 的像素差分。

`Sim2RealCubeGraspEnv` 及其现实对齐子类使用 reset-relative Cube lift：

```text
spawn_lowest = reset_center_z - cube_height / 2
lift_delta = current_lowest - spawn_lowest
lift_progress = clamp((lift_delta - lift_start_delta) / (success_delta - lift_start_delta), 0, 1)
lift_reward = lift_progress * upright_mask
success = (lift_delta >= success_delta) and upright
```

方块静止在台面时 lift reward 为 0。基础 6 cm Cube task 使用 `lift_start_delta=5 mm`、`success_delta=40 mm`；现实对齐 5 cm Cube task 覆盖为 `lift_start_delta=5 mm`、`success_delta=35 mm`。旧的绝对高度字段为 saved cfg 兼容保留，但不再用于 Cube reward/done。

## 11. 终止与成功判定

终止函数：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_dones`。

Done 条件：

- episode timeout
- robot body 低于 `ground_height`

成功判定：

- 成功由物体高度超过 `success_height` 触发。
- success 会 latch 并用于 reward/统计。
- success 当前不直接终止 episode。

## 12. 训练和评估差异

训练：`scripts/reinforcement_learning/skrl/train.py:main`。

- 创建 trainer 并更新 PPO。
- 使用 agent YAML 中的 `timesteps`、`rollouts`、`learning_epochs`、`mini_batches` 等训练参数。

普通评估：`scripts/reinforcement_learning/skrl/play.py:main`。

- 加载 checkpoint。
- 可从 run 目录恢复 `params/env.yaml` 和 `params/agent.yaml`。
- 执行策略并写 play metrics。

Bucket 评估：`scripts/reinforcement_learning/skrl/play_bucket.py:main`。

- 固定物体位置网格。
- 写 trial CSV 和 summary CSV。
- 当前默认 bbox occlusion 关闭时，已确认既有 summary 中 `occlusion_ratio_mean` 可为 `nan`。

## 13. 关键张量表

| 名称 | 默认维度/形状 | 生产者 | 消费者 | 备注 |
| --- | --- | --- | --- | --- |
| action | 5 | policy | `_pre_physics_step` | `[dx, dy, dz, dyaw, gripper]` |
| IK command | 6 | `_pre_physics_step` | `_apply_action` | `[dx, dy, dz, 0, 0, dyaw]` |
| `proprio_obs` | 18 | `_get_observations` | policy | joint position + joint velocity |
| `third_resnet` | 256 | `_get_observations` | policy | torchvision 不可用时可能为零向量 |
| `tactile_left_depth_resnet` | 256 默认；DINO 时间拼接时可变 | `_get_observations` | policy | 左内侧触觉 |
| `tactile_right_depth_resnet` | 256 默认；DINO 时间拼接时可变 | `_get_observations` | policy | 右内侧触觉 |
| `tactile_left_down_depth_resnet` | 256 默认；DINO 时间拼接时可变 | `_get_observations` | policy | 左下侧触觉 |
| `tactile_right_down_depth_resnet` | 256 默认；DINO 时间拼接时可变 | `_get_observations` | policy | 右下侧触觉 |
| `critic_can_position` | 3 | `_get_observations` | value network | privileged |
| `critic_can_quat` | 4 | `_get_observations` | value network | privileged |
| `critic_can_lin_vel` | 3 | `_get_observations` | value network | privileged |
| `critic_can_ang_vel` | 3 | `_get_observations` | value network | privileged |
| `critic_gripper_position` | 3 | `_get_observations` | value network | privileged |
| `critic_gripper_quat` | 4 | `_get_observations` | value network | privileged |
| `critic_gripper_lin_vel` | 3 | `_get_observations` | value network | privileged |
| `critic_gripper_ang_vel` | 3 | `_get_observations` | value network | privileged |
| `critic_target_relative_position` | 3 | `_get_observations` | value network | privileged |
| `critic_target_distance` | 1 | `_get_observations` | value network | privileged |
