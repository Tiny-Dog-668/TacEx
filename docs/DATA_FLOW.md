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
aligned simulation camera: direct 224x224 RGB @ 30 Hz
  -> centered 300 px-focal coverage render
  -> fixed GPU warp to calibrated K=[338.742544,0,123.748857;
                                     0,340.550811,120.393372;
                                     0,0,1]
  -> clean task: no image or scene appearance randomization
  -> DR task, episode-fixed per env:
       calibrated pose * delta XYZ/RPY
       -> GPU focal/principal-point affine warp
       -> brightness/contrast/saturation/gamma/hue/white balance/blur
       -> frame-wise Gaussian pixel noise
       -> per-env near-black plate/backdrop colors centered on Clean
       -> batch-global DomeLight only on full-batch reset
  -> DR range scale: 0.00 through step 100k (exact Clean visual path)
                     linear 0.00 -> 1.00 from 100k to 220k
                     full range from 220k to the 300k training end

Real-Alignment deployment camera: 640x480 RGB @ 30 Hz
  -> crop x=[100,500), y=[34,432) to 400x398
  -> bilinear resize to 224x224

both paths:
  -> ImageNet normalization
  -> frozen ResNet18 in eval mode (BatchNorm running statistics never update)
  -> wrist_resnet [N, 512]
```

Clean task 为 `TacEx-Sim2Real-Cube-Real-Alignment-v0`，DR task 为 `TacEx-Sim2Real-Cube-Real-Alignment-DR-v0`。两者的 `proprio_obs [N,15]`、`action_history [N,4]`、policy 输出 `[dx,dy,dz,gripper_total_width_delta]` 及全部控制/奖励语义一致，只有随机化 profile 和日志目录不同。DR 参数不进入 Actor/Critic observation，课程阶段是全局训练进度而不是特权状态。Cube Actor 在 policy model 内执行 `tanh`，因此 deterministic mean 在训练、play 和 export 中统一位于 `[-1,1]`。

Privileged reward/control diagnostic 使用另一条 Actor 数据流：

```text
cube root position world - env_origin       -> privileged_cube_pos [N,3]
left/right fingertip world midpoint
  - env_origin                              -> privileged_gripper_pos [N,3]
cube position - gripper midpoint            -> privileged_target_pos [N,3]

concat(proprio [N,15], history [N,4],
       cube/gripper/target positions [N,9])  -> MLP Actor [N,28]
                                             -> tanh action [N,4]
```

该路径不创建 RGB camera、不读取 pixel buffer、也不构造 ResNet18。后续 IK、奖励和 done 复用 Clean 实现，但显式保留旧的夹爪 `2 mm/step` 和物体 `x/y ±10 cm` reset 配置；位置真值不可在现实部署侧获得。

动作历史时序为：

```text
actor action a_t
  -> finite check / per-dimension scale / clamp
  -> xyz processed command [0.025*x, 0.025*y, 0.025*z]
  -> requested total-width delta 0.005*g
  -> cached desired_width = clamp(previous_desired_width + 0.005*g, 0, 0.08)
  -> observation history for transition: action_history_{t+1} = detach(p_t)
  -> IK plus symmetric finger targets desired_width/2 during transition t -> t+1
  -> observation o_{t+1}
```

当前 Clean/DR Real-Alignment v9 的 `action_history` 逐维 scale 为 `[0.025,0.025,0.025,0.005] m`；保留的 v8/v7 contract 分别为 `[0.025,0.025,0.025,0.002]` 与 `[0.010,0.010,0.010,0.002] m`。第四维记录请求的总宽度增量，即使缓存目标已在 0/80 mm 边界而被 clamp，也保留请求值。缓存宽度每个 policy step 只更新一次，后续 `_apply_action()` 只重发同一目标。Sim2real Cube 禁用 privileged `dz` gate，因此前三维 history 与送入 IK 的请求 XYZ 一致；随后仍可能受 IK、关节限位和工作空间约束。Cylinder 保留 legacy gate 和 uniform `0.05` processed history。

当前 `sim.dt=1/60 s`、`decimation=2`：physics 为 60 Hz，camera/render/policy/action/observation/reward/history 为 30 Hz。每个 policy step 内，`_apply_action()` 在两个 physics substep 重发同一缓存夹爪目标，但总宽度只在 `_pre_physics_step()` 累加一次。`episode_length_s=5.0`，所以 timeout horizon 为 150 个 policy steps。

批量 rollout 诊断的数据流为：

```text
saved env.pkl + exported Actor metadata/hash
  -> exact saved environment config
  -> uint8 RGB + action_history + proprio_obs
  -> CPU TorchScript Actor
  -> normalized action [N,4]
  -> environment processed action / IK / gripper target
  -> pre/post transition snapshots
  -> NPZ [T,N,...]
  -> offline CSV + episode-step action/state plots
```

DirectRLEnv 在 `env.step()` 内自动 reset 已完成环境，因此 collector 的 `next_*` 对 done transition 表示 reset 后的新 episode 状态；离线 state plot 会排除这些 done transition，CSV 保留原始值并同时提供 `done` 字段。

Real-Alignment 抓取中心数据流为：

```text
panda_leftfinger pose  + local [0,0,0.045] -> left fingertip world position
panda_rightfinger pose + local [0,0,0.045] -> right fingertip world position
                                                |
                                                v
                                    midpoint of both fingertips
                                      -> reach reward distance
                                      -> critic_gripper_pos
                                      -> critic_target_pos/distance

panda_hand pose + local [0,0,0.1034] -> fixed IK/Jacobian TCP
```

Actor 仍只读取 `wrist_resnet [N,512]`、`proprio_obs [N,15]` 和 `action_history [N,4]`，没有增加 privileged center 输入；critic 继续使用 privileged state，但 Actor 输出后的控制链不再读取方块真值。

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

Real-Alignment 使用 settled-center-relative Cube lift，并额外计算桌面碰撞：

```text
settled_center = reset_center_z + cube_rest_offset + plate_rest_offset
lift_delta = current_center_z - settled_center
lift_progress = clamp((lift_delta - lift_start_delta) / (success_delta - lift_start_delta), 0, 1)
lift_reward = lift_progress
success = lift_delta >= success_delta
table_collision = max(filtered table/robot normal force over 2 physics substeps) > 1 N
reward = 5*reach + 15*lift + 100*success - 10*table_collision
```

RMA-style Student 的每步数据流为：

```text
calibrated wrist_rgb uint8 [N,224,224,3]
  -> frozen ImageNet ResNet18 layer4 [N,512,7,7]
  -> trainable 1x1 conv + 32-channel spatial softmax [N,64]
  -> trainable shared MLP
     -> normalized predicted cube XYZ [N,3]
     -> predicted left/right contact logits [N,2] -> sigmoid probabilities
  -> denormalized predicted cube XYZ in robot-root frame
proprio joint positions [N,7] -> embedded Panda FK -> fingertip midpoint XYZ [N,3]
predicted cube XYZ - fingertip XYZ -> relative target XYZ [N,3]
  -> frozen shared RMAActorCore 30-D features including predicted contact[2]
  -> student tanh action [N,4] -> env.step

sim cube XYZ -> normalized target -> SmoothL1 position loss
sim cube-finger filtered force >= 0.2 N -> contact target[2] -> BCE contact loss
sim cube XYZ + true contact -> same frozen RMAActorCore -> teacher action -> action MSE
```

Heatmap-supervised Student task 在不改变部署 `forward()` 的前提下，从同一冻结
ResNet18 额外读取 layer3 特征：

```text
calibrated wrist_rgb uint8 [N,224,224,3]
  -> frozen ResNet18 layer3 [N,256,14,14]
  -> trainable 3x3 Conv + ReLU + 1x1 Conv
  -> predicted cube-center heatmap [N,1,14,14]

cube_position_root [N,3]
  -> root_T_camera optical pose + effective Student K
  -> projected uv [N,2] in 224x224 pixels + valid mask [N]
  -> Gaussian GT heatmap [N,1,14,14], sigma default 1.5 heatmap px
```

投影坐标约定为 robot-root/base 坐标中的 cube center 和 camera optical origin；
相机四元数为 `wxyz` 且表示 `R_root_camera`，投影时使用
`R_root_camera.T @ (p_root - t_root)` 得到 ROS optical camera 坐标
`x-right, y-down, z-forward`。位于相机后方或 224x224 图像范围外的样本通过
valid mask 排除 heatmap MSE。DR Student Heatmap task 会读取每环境随机化后的相机
pose、焦距 scale 和主点 shift 来生成对应 GT heatmap。predicted heatmap 的归一化
期望坐标只用于日志和 debug overlay，不进入 Actor 或部署接口。

默认 Student 训练仍冻结整个 ResNet18。若训练脚本启用
`--train_backbone_after_layer2`，则只允许 ResNet18 layer3/layer4 更新，conv1、bn1、
layer1、layer2 及 Actor Core 保持冻结；ResNet BatchNorm 仍保持 eval 模式，避免
训练过程改写 running statistics。

RMA Student DR task 在上述 `calibrated wrist_rgb` 与 Student adaptation head
之间插入全强度视觉扰动；每个 episode 为每环境采样相机 `delta XYZ/RPY`、焦距/主点
warp、brightness/contrast/saturation/gamma/hue/white-balance 和 blur 参数，pixel
Gaussian noise 每帧重采样。板/背景颜色也按 episode 重采样，DomeLight 仅在全环境
batch reset 时更新，因为它属于共享 USD stage。该 task 的 `dr_curriculum_enabled=false`，
所以全部范围从第 1 个 policy update 起生效；RGB shape/dtype 仍为
`uint8 [N,224,224,3]`，本体/history、特权 loss label、动作、奖励与 done 均不变。

Actor Core 在两个动作分支中是同一份冻结参数；动作 loss 仍通过 Actor 对 cube
位置、相对位置与连续接触概率的 Jacobian 回传到 adaptation head。末端 XYZ 由
`proprio_obs[:7]` 在模型内计算，不读取仿真 link 真值；特权 cube XYZ 不进入
Student `forward()`，真实接触标签也不进入部署前向。当前 Actor 不使用 cube
quaternion 或 end-effector quaternion。Teacher/Student 使用相同的 RMA 接触
reward contract：权重3.0、single-contact fraction 1.0；Clean/DR仍使用原公式。

RMA 的策略动作尺度仍与 Clean 相同：第四维为 `0.010*g m` 的总夹爪宽度增量。
只有 RMA copied robot cfg 的 finger actuator 被降低为 `effort_limit_sim=40`、
`stiffness=400`、`damping=40`，用于减少强位置伺服导致的 cube/finger 嵌入。

方块静止在台面时 lift reward 为 0；现实对齐 5 cm Cube 使用 `lift_start_delta=0 mm`、`success_delta=35 mm`，因此 17.5 mm 对应 lift 0.5。倾角仍计算并写日志，但不作用于 lift、success 或 done。RMA Teacher/Student 中 success 只用于奖励和 episode 统计，不触发 done；episode 仅因 timeout 或严重机器人穿地碰撞结束。木板 ContactSensor 是每环境单一 sensor body，过滤 `panda_link1–7`、`panda_hand` 和两侧 finger，因此方块落桌或 finger 接触方块不会单独触发撞桌惩罚。

基础任务的奖励控制台每 `reward_print_interval=200` 个 policy steps 打印一次。RMA Teacher/Student 单独使用精简行：`reach/lift/success/contact/table` 均为打印时刻跨环境平均后的加权奖励贡献，`total` 为当前平均总奖励，`avg_reward_200` 为最近200步的平均总奖励，`success_window_200` 为该窗口内已终止 episode 中曾达到 success 的比例，`avg_lift` 为当前跨环境平均的质心抬升高度（mm）。TensorBoard `extras["log"]` 仍保留原有完整诊断字段。

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
