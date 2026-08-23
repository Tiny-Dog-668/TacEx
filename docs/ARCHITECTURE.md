# Architecture

本文件是 TacEx 的系统说明，分三部分：第一部分项目概览（做什么、当前状态、快速开始），第二部分模块架构（代码结构与各模块职责），第三部分数据流（张量在环境、策略、奖励之间如何流动）。变更日志与实验记录见 `DEVLOG.md`，设计取舍与已知问题见 `DECISIONS.md`。

## 第一部分 项目概览

### 1. 项目目标

TacEx 当前仓库的主线已由用户确认为 `occluded_grasping`：在 Isaac Sim / Isaac Lab 中构建可训练、可评估的遮挡视觉触觉机器人抓取实验平台。`source/tacex` 和 `source/tacex_assets` 提供 GelSight Mini 等 vision-based tactile sensor 的仿真输出与资产配置，作为当前主线的基础设施。

依据：`source/tacex/tacex/gelsight_sensor.py:GelSightSensor`；`source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py:GelSightMiniCfg`；`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxEnv`。

### 2. 当前任务

当前代码中的主要任务是 `occluded_grasping` 遮挡抓取：在 `Drawer-Occlusion` 或 `Self-Occlusion` 场景中，用 Franka Panda 末端夹爪抓取目标物体，并把物体稳定抬升到成功高度。

已确认任务注册矩阵支持：

- Scene：`Drawer-Occlusion`、`Self-Occlusion`
- Object：`Cylinder`、`Cube`、`Cuboid`、`SoftCylinder`、`SoftCube`、`SoftCuboid`
- Fusion/Policy 变体：V、T、VT、GelFusion、Alpha、GRU、Cross、Aux、Sparsh、Policy-Token-Transformer 等

依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_scene_object_fusion_matrix`。

### 3. 系统组成

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

### 4. 当前已实现功能

- 现实参考对齐的 vision-only Cube clean task：`TacEx-Sim2Real-Cube-Real-Alignment-v0`。它使用 5x5x5 cm 白色方块、1 mm 近黑色木板和近黑色背景；方块在 robot-root frame 的完整范围为 `x=[0.45,0.55] m, y=[-0.05,0.05] m`，20k 前使用 `x=[0.48,0.52] m, y=[-0.02,0.02] m`，并在 100k 扩至完整范围。物理保持 60 Hz，camera/render/policy/action/observation/reward/history 对齐为 30 Hz；5 秒 episode 对应 150 个 policy steps。
- 对应的 curriculum DR task：`TacEx-Sim2Real-Cube-Real-Alignment-DR-v0`。它严格以 Clean 为零扰动基线，继承完全相同的机器人、224x224 有效 K 相机、方块、动作、观测、奖励和 done；plate/backdrop/DomeLight 中心值也与 Clean 相同。固定 GPU warp 先将 centered coverage render 映射到 400x398 crop 对应的目标 K，之后每环境每 episode 再随机相机外参、GPU 等效内参、图像后处理和近黑材质。DomeLight 只在整批 reset 时更新，global ground 固定。DR scale 在 100k policy step 前为 0，100k–220k 线性增至 100%，220k–300k 保持 full range，并记录 `info/dr_curriculum_scale`。两个 profile 使用独立日志目录，旧 DR checkpoint 不作为新视觉分布的续训起点。
- 奖励/控制上界诊断 task：`TacEx-Sim2Real-Cube-Real-Alignment-Privileged-v0`。它继承 clean task 的场景、奖励、success 和 150-step horizon，但显式保留旧的夹爪 `2 mm/step` 与 `x/y ±10 cm` reset 配置；它不创建 RGB camera 或 ResNet18。Actor 输入为 `proprio [N,15] + history [N,4] + cube/gripper/relative-target position [N,9]`。三个位置均去除并行环境 origin，使用 robot-root-aligned frame。该策略含仿真真值，只用于判断视觉之外的抓取可学习性，不能部署到真机。
- Sim2real Cube strict policy contract：ResNet18 参数冻结且始终 `eval()`，Actor deterministic mean 在模型内通过 `tanh` 约束到 `[-1,1]`。当前 Clean/DR 的逐维 action/history scale 为 `[0.025,0.025,0.025,0.005] m`；第四维是每个 policy step 请求的总夹爪宽度增量，缓存总宽度目标后对称映射到两个 finger joints。IK TCP 为 `panda_hand+0.1034 m`，reward/critic 中心为两侧 `finger link+0.045 m` 指尖的世界坐标中点。Real-Alignment contract v9 绑定 25/5 mm 尺度与 `x/y ±5 cm` 范围；v8 和 v7 继续严格读取各自的旧 25/2 mm 与 10/2 mm contract，不能混用。
- Real-Alignment 以考虑 cube/plate rest offset 的稳定落桌质心高度为零点，`0–35 mm` 线性映射到 lift `[0,1]`；success 只要求达到 35 mm 并连续保持 5 个 policy steps，倾角仅记录日志。任一活动 Panda arm/hand/finger link 对木板的过滤接触力超过 1 N 时，该 policy step 加 `-10`，但不直接终止。控制台同时打印碰撞统计、当前 env mean total 和最近打印窗口平均奖励。
- Franka Panda + GS Mini gripper 任务配置：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.robot` 使用 `source/tacex_assets/tacex_assets/robots/franka/franka_gsmini_gripper_rigid.py:FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG`。
- 第三视角 RGB 相机：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.third_person_camera`。
- 四路触觉输入：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.gsmini_left`、`gsmini_right`、`gsmini_left_down`、`gsmini_right_down`。
- 5 维动作空间 `[dx, dy, dz, dyaw, gripper]`：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.action_space`、`OccludedGraspingVisionFourTactileBoxEnv._pre_physics_step`。
- 观测字典：`proprio_obs`、`third_resnet`、四路 tactile feature、critic privileged keys：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.observation_space`、`_get_observations`。
- 触觉 + 本体 baseline：`vt_tactile_box.py` 暴露 `proprio_obs` 和四路 tactile feature 给 actor，注册 task 包括 `TacEx-T-Drawer-Occlusion-Cube` 等矩阵组合。
- GelFusion 风格 RL 分支：`vt_gelfusion_box.py` 额外提供 8 维 `tactile_dynamic_stats`，`vt_gelfusion_policy.py` 实现 vision-led cross attention actor；注册 task 包括 `TacEx-GelFusion-Drawer-Occlusion-Cube` 和 `TacEx-GelFusion-Downsample-Drawer-Occlusion-Cube` 等矩阵组合。
- Hansen-style hard tactile gate 对比：`vt_hard_gate_box.py` 在环境层用 tactile depth baseline contact ratio 对四路 tactile feature 做硬门控；注册 task 包括 `TacEx-Hard-Gate-Drawer-Occlusion-Cube` 和 `TacEx-Hard-Gate-Downsample-Drawer-Occlusion-Cube` 等矩阵组合。
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

### 5. 部分实现功能

- 多策略实验矩阵已经存在，但配置和实现数量较多，推荐主线仍需确认。依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/*.yaml`、`vt_alpha_gru_policy.py`、`vt_cross_policy.py`、`vt_tactile_cross_alpha_aux_policy.py`。
- Auxiliary PPO 已实现可读取 `aux_loss` 等 policy outputs，但具体实验稳定性待确认。依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_aux_ppo_agent.py:PPOWithAuxHeadsLoss._update`。
- Soft object 注册和 deformable 分支存在，但 UIPC/FEM 运行链路在当前遮挡抓取主线中的依赖关系待确认。依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_scene_object_fusion_matrix`、`vt_box.py:_is_can_deformable`。
- DINO/Sparsh 触觉编码路径存在，但包含本机绝对路径，跨机器可复现性待确认。依据：`vt_box.py:_DEFAULT_DINO_ENCODER_ROOT`、`OccludedGraspingVisionFourTactileBoxCfg.tactile_dino_repo_path`。

### 6. 尚未实现或待确认功能

- 是否将通用 TacEx sensor framework 从 `occluded_grasping` 主线中拆分为独立仓库或继续作为内部基础设施维护：待确认。
- 是否去除 `occluded_grasping/__init__.py:_patch_skrl_runner_for_custom_models` 的 import-time monkey patch：待确认。
- 是否统一 `train.py`、`play.py`、`play_bucket.py` 的重复 cfg/checkpoint/custom component 加载逻辑：待确认。
- 是否清理 legacy task id、历史入口脚本和未使用 policy 文件：待确认。
- CPU-only GelSight Mini 路径是否可运行：待确认。依据：`source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py:GelSightMiniCfg.device` 默认 `cuda`。

### 7. 当前主要技术路线

当前主线是 Isaac Lab `DirectRLEnv` + skrl PPO：

1. `scripts/reinforcement_learning/skrl/train.py:main` 启动 Isaac Sim。
2. `import tacex_tasks` 触发 task 注册。
3. `gym.make(args_cli.task, cfg=env_cfg)` 创建环境。
4. 环境生成视觉、触觉、本体和 privileged critic 观测。
5. policy 读取视觉/触觉/proprio，critic 读取 privileged state。
6. PPO 更新 policy 和 value network。

依据：`scripts/reinforcement_learning/skrl/train.py:main`；`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py`；`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml`。

### 8. 当前基线方法

- Vision-only baseline：`source/tacex_tasks/tacex_tasks/occluded_grasping/vision_box.py:OccludedGraspingVisionOnlyBoxEnv`。
- Tactile-proprioception baseline：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_tactile_box.py:OccludedGraspingTactileProprioBoxEnv`，配置 `source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_tactile.yaml`。
- Vision + tactile feature concat/downsample baseline：已确认 checkpoint 目录 `logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/`。
- Alpha-GRU 策略：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_policy.py:OccludedGraspingVTAlphaGRUPolicy`，配置 `source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml`。
- Cross attention 策略：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_cross_policy.py:OccludedGraspingVisionTactileCrossAttentionPolicy`。
- Aux heads 策略：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_tactile_cross_alpha_aux_policy.py:OccludedGraspingVTTactileCrossAlphaAuxPolicy`。
- GelFusion-style 策略：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_policy.py:OccludedGraspingVTGelFusionPolicy`。
- Real-Alignment RMA-style：PandaHand 使用独立 XY/force Teacher–Student 契约：Teacher 接收 cube XY 和左右接触力，Student 仅从第三视角 RGB 预测 XY；Actor 从本体 FK 生成夹爪 XY，组成26维特征。每策略步的两次物理子步接触力取最大值，左右均 `>=1 N` 时仅生成一个 `grasped` 特征；Student TorchScript 输入为 `RGB + proprio[15] + history[4] + contact_force_n[2]`。接触 reward 仅在 grasped 时给出，旧 PandaHand RMA checkpoint 不兼容并需重训。GelSight RMA profile 仍保持原有配置、触觉接触预测与 artifact 契约，未随 PandaHand XY/force 路线变更。
- X040-Wide RMA（独立 profile）：Teacher Actor 只接收 `proprio[15] + history[4] + cube_position_root[3]`，并由本体 FK 派生 TCP 与相对 cube 的 XYZ，形成 28 维特征；不接收接触力或接触状态。Student 部署仍严格为 `RGB[224,224,3] + proprio[15] + history[4] -> action[4]`，训练时以 simulator cube XYZ 作 ResNet-512-GAP 位置头的辅助 MSE 标签。该 profile 的 cube reset 是 x=0.40、x±8 cm、y±10 cm，无 curriculum；Student 相机 pose DR 是局部 xyz±1 cm、XYZ±2°；仅 Student 把底座 `panda_link0/visuals/panda_link0/subset_5` 绑定为逐 env、逐 episode 的 HSV 自发光材质（H 105°–135°、S 0.75–1.0、V 0.35–1.0），腕部 LED 仍是蓝色。TCP world z<0.011 m 每步额外 -10，但不改变动作或 done。
- X040-Wide Size-Buckets（独立实验 task）：每个 env 预生成 8 个等间隔 `4–6 cm` 物理方块，reset 均匀选择一个作为当前 `_cube`；其余方块的停车区根据 env-origin X 跨度、腕部相机 X 向 DR 上界和 `0.5 m` 裕量动态移到所有相机后方，避免规则 env 网格中相邻环境的停车方块出现在本环境 Franka 底座旁，同时不修改 USD visibility 或 PhysX 属性。reset 同时为 Franka 7 个臂关节采样截断高斯初始角偏移（`σ=0.01 rad`、`|Δq|≤0.03 rad`），夹爪角固定、关节速度清零。位置、速度、critic、奖励、success 与 Student `rma_cube_pos` 标签都通过 selected-cube adapter 只读取当前桶；每个桶各有一个 cube→双指 one-to-many contact sensor，环境按当前桶汇总回原 `[left,right]` 内部奖励语义。Actor/Student 输入维度与导出签名不变，cube center Z 随尺寸覆盖 `0.021–0.031 m`。
- X040-Wide Size-Buckets Three-Frame Student（独立实验 task）：环境把连续 3 个 30 Hz policy frame 缓存为 `wrist_rgb_history[N,3,224,224,3] uint8`，顺序为 oldest→newest；reset 将首张新 episode 图复制到三个槽，禁止跨 episode 图像泄漏。三帧分别进入同一个共享、BatchNorm-eval 的 ResNet18，各得 512 维 GAP，拼接的 1536 维经 Linear+ELU 压回 512，再与 `proprio[15] + history[4]` 组成原 531 维动作头输入；当前位置辅助头也读取融合后的 512 维。该 task 复用 Size-Buckets Teacher、物理、DR、奖励与 done，但使用独立 Student artifact，单帧 checkpoint 不兼容并需重新蒸馏；训练与评测入口位于 `rma_x040_wide_three_frame_direct_action_student/`。
- X040-Wide Three-Frame Independent-Appearance（独立实验 task）：原 Appearance task id 现注册到 `sim2real_cube_real_alignment_rma_x040_wide_three_frame_independent_appearance_env.py`，物理基类位于新增的 `sim2real_cube_real_alignment_rma_x040_wide_static_size_buckets_env.py`。每个 env 仅创建一个 `/cube`，在 PhysX 启动前按 `env_id % 8` 固定为 4–6 cm 八档；环境数必须为 8 的倍数，故每档数量严格相同且 reset 不换尺寸。该异构场景保持 `replicate_physics=False`、显式过滤跨 env 碰撞，同时启用 PhysX GPU dynamics。场景使用 3.5 m env spacing，3×3 m 木板之间至少留 0.5 m；每个 env 精确拥有一个 `floor_panel` 和一个 `real_alignment_backdrop`，共享 GroundPlane 保留碰撞但不可见，避免相邻场景重叠或共享网格进入图像。每个 env 在 `/World/Looks/X040IndependentAppearance/env_<id>` 下拥有木板、背景、方块三个独立 PreviewSurface；材质关系启动前绑定，局部 reset 只修改对应 env 的 shader 颜色，未 done env 的物理和外观均不更新。episode horizon 沿用 5 s/150 policy steps：机器人碰地返回 `terminated`，达到时限返回 `truncated`，二者只 reset 对应 env；success 仍非终止。不使用独立相位，同期启动且没有提前 done 的 env 在时限处同时 reset 属于预期。Appearance profile v5 记录静态分桶、隔离布局和 timeout，旧 Appearance v1 Student 仅允许显式 rollout，指标不可直接横比。三帧输入、动作、奖励和坐标系不变。
- Size-Buckets 自动训练入口为 `scripts/reinforcement_learning/skrl/train_rma_x040_wide_size_buckets_pipeline.py`。它在独立 Isaac 子进程中先运行 Teacher，只把“进程正常退出 + 新 run 唯一 + manifest task 匹配 + 精确最终 checkpoint 存在”视为完成；随后显式选择 best/final Teacher 文件并启动 Student，Student artifact 继续记录 Teacher 与 encoder 的路径和 hash。

### 9. 当前主要实验变量

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

### 10. 当前项目状态

- 当前分支：`dev/0517`。
- 当前 HEAD：`c388bdd add the heatmap`，统计为 14 files changed, 814 insertions, 27 deletions。依据：`git show --stat --oneline HEAD`。
- 当前工作区存在未提交改动，集中在 `source/tacex_tasks/tacex_tasks/sim2real_grasp`、新增的 `source/tacex_tasks/tacex_tasks/sim2real_gelsight_rma`、`scripts/reinforcement_learning/skrl/*` 和 `docs/*`。依据：`git status --short`。
- 本文档只描述已从仓库和用户说明确认的信息；未提交改动的作者、意图和最终去留均为待确认。

### 11. 下一步建议

1. 先固定一个推荐训练入口、一个推荐评估入口和一个推荐 task id，减少 README/AGENTS/docs 之间的漂移。
2. 把 `train.py`、`play.py`、`play_bucket.py` 中重复的 cfg/checkpoint/custom component 逻辑抽成共享 helper。
3. 把 `vt_box.py` 中 scene/sensor encoding/reward/reset/logging 职责拆分前，先补充 smoke test 和文档化张量契约。
4. 将本机绝对路径改成配置项或环境变量，尤其是 DINO/Sparsh checkpoint 和 repo 路径。
5. 对已存在 checkpoint 和 metrics 建立 `docs/DEVLOG.md` 第二部分中的实验索引。

### 12. 快速开始

现实参考对齐 Cube task：

```bash
python scripts/reinforcement_learning/skrl/train.py --task TacEx-Sim2Real-Cube-Real-Alignment-v0 --num_envs 4 --enable_cameras --headless
```

RMA Teacher 与 Student：

```bash
python scripts/reinforcement_learning/skrl/train.py --task TacEx-Sim2Real-Cube-Real-Alignment-RMA-Teacher-v0 --num_envs 256 --headless
python scripts/reinforcement_learning/skrl/train_rma_student.py --teacher_checkpoint <teacher.pt> --num_envs 4 --timesteps 100000 --enable_cameras --headless
python scripts/reinforcement_learning/skrl/train_rma_student.py --task TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-DR-v0 --teacher_checkpoint <teacher.pt> --num_envs 128 --timesteps 100000 --train_backbone_after_layer2 --backbone_learning_rate 3e-5 --headless
python scripts/reinforcement_learning/skrl/train.py --task TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Teacher-v0 --num_envs 256 --headless
python scripts/reinforcement_learning/skrl/train_rma_student.py --task TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-Heatmap-DR-v0 --teacher_checkpoint <gelsight-teacher.pt> --num_envs 128 --timesteps 100000 --train_backbone_after_layer2 --backbone_learning_rate 3e-5 --headless
python scripts/reinforcement_learning/skrl/train.py --task TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Size-Buckets-Teacher-v0 --num_envs 256 --headless
python scripts/reinforcement_learning/skrl/rma_gelsight_size_buckets_student/train.py --teacher_checkpoint <new-gelsight-size-teacher.pt> --num_envs 128 --timesteps 100000 --train_backbone_after_layer2 --headless
```

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

## 第二部分 模块架构

### 1. 系统总体结构

TacEx 采用 Isaac Lab extension 结构组织代码：

1. `source/tacex` 提供 GelSight sensor core 和触觉仿真方法。
2. `source/tacex_assets` 提供传感器、机器人和物体资产配置。
3. `source/tacex_tasks` 提供 RL task 注册和环境实现。
4. `source/tacex_uipc` 提供 UIPC/pyuipc 集成，当前遮挡抓取主线依赖关系待确认。
5. `scripts/reinforcement_learning/skrl` 提供 skrl 训练、回放和评估脚本。
6. `logs/skrl/occluded_grasping` 保存训练参数、checkpoint 和评估 CSV。

### 2. 目录结构

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
| `docs` | 项目长期维护文档 | `ARCHITECTURE.md`、`DEVLOG.md`、`DECISIONS.md`、`archive/` |

### 3. 训练入口

训练入口是 `scripts/reinforcement_learning/skrl/train.py:main`。

主要调用关系：

1. `AppLauncher` 启动 Isaac Sim。
2. `import tacex_tasks` 触发 task 注册。
3. `hydra_task_config(args_cli.task, agent_cfg_entry_point)` 读取 env 和 agent cfg。
4. `gym.make(args_cli.task, cfg=env_cfg, render_mode=...)` 创建 Isaac Lab 环境。
5. `SkrlVecEnvWrapper` 包装环境。
6. 根据 agent/model class 是否为 `module:Class`，选择 skrl Runner 或手动 custom agent/trainer 路径。

关键 helper：`train.py:_process_cfg`、`train.py:_load_component`。

### 4. 评估入口

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

### 5. 环境模块

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

### 6. 机器人与控制模块

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

### 7. 视觉模块

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

视觉编码：`OccludedGraspingVisionFourTactileBoxEnv.__init__` 初始化 frozen ResNet18；`_get_observations` 输出 `third_resnet`。torchvision 不可用时，代码会输出零向量，风险记录见 `docs/DECISIONS.md` 的 ISSUE-007。

### 8. 触觉模块

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

### 9. 本体状态模块

本体观测 key `proprio_obs`，默认维度 18，由 `vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_observations` 拼接 robot joint position 与 joint velocity 生成。维度定义见 `OccludedGraspingVisionFourTactileBoxCfg.observation_space`，数据流见第三部分第 5 节。

### 10. 视触融合模块

已确认策略实现包括：

- Alpha-GRU：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_policy.py:OccludedGraspingVTAlphaGRUPolicy`。
- Cross attention：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_cross_policy.py:OccludedGraspingVisionTactileCrossAttentionPolicy`。
- Aux heads：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_tactile_cross_alpha_aux_policy.py:OccludedGraspingVTTactileCrossAlphaAuxPolicy`。
- GelFusion-style：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_policy.py:OccludedGraspingVTGelFusionPolicy`，视觉作为 query，视觉和四路触觉静态特征作为 key/value，动态触觉统计单独拼接。

代表配置：`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml`。其中 policy 输入包括 `third_resnet`、四路 tactile keys 和 `proprio_obs`，并包含 `tactile_time_window`、`tactile_gru_hidden_dim`、`fused_dim` 等字段。

### 11. Actor

Actor 由 skrl policy model 实现。以 `ppo_vt_alpha_gru.yaml` 为例：

- class：`tacex_tasks.occluded_grasping.vt_alpha_gru_policy:OccludedGraspingVTAlphaGRUPolicy`
- vision key：`third_resnet`
- tactile keys：四路 `*_depth_resnet`
- vector/proprio key：`proprio_obs`
- action 输出维度：由环境 `action_space = 5` 决定

关键函数：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_policy.py:OccludedGraspingVTAlphaGRUPolicy.compute`。

### 12. Critic

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

### 13. 奖励与终止条件

奖励函数：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_rewards`。

奖励项：

- reach reward
- lift reward
- success reward
- drop-after-success penalty
- inner tactile contact reward
- down tactile contact reward

触觉 contact reward：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._compute_tactile_contact_rewards`。

终止函数：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_dones`；done 条件与 success 语义见第三部分第 11 节。

### 14. 配置系统

任务注册中通过 `gym.register(..., kwargs={"env_cfg_entry_point": ..., "skrl_cfg_entry_point": ...})` 绑定环境配置和 agent YAML。依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_task`。

训练脚本通过 `hydra_task_config(args_cli.task, agent_cfg_entry_point)` 加载配置。依据：`scripts/reinforcement_learning/skrl/train.py:main`。

代表 agent YAML：`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml`。

### 15. 模型保存与加载

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

### 16. 关键调用关系

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

### 17. Real-Alignment RMA 路线

RMA Teacher/Student 使用新增 task，不修改 Clean、DR 或旧 Privileged：

- PandaHand Teacher：无相机，外部输入为15维本体、4维 history、cube XY 和左右真实接触力；`RMAXYActorCore` 从关节角执行 Panda FK 并仅使用夹爪 XY 与 cube-to-gripper XY，再加单一 bilateral `grasped` 状态，形成26维特征后输出4维 tanh action；Z 不是物体输入特征。
- PandaHand Student：环境输出 `wrist_rgb[224,224,3] uint8`、15维本体、4维 history、仅用于位置损失的 cube XY，以及部署时必须提供的 `contact_force_n[2]`。Student 仅从 RGB 预测 XY，不预测接触。
- PandaHand Direct-Action Student：`TacEx-Sim2Real-Cube-Real-Alignment-RMA-Direct-Action-Student-DR-v0` 复用同一 full-strength DR 环境和训练标签，但模型独立位于 `rma_direct_action_student/`。ResNet18 layer4 做 global average pooling 得512维视觉特征，与归一化 `proprio[15]`、`history[4]` 拼成531维，经 `531→512→256→128→64→4` ELU MLP 与 `tanh` 直接给出动作；运行时/导出/rollout 不接收或存储 cube XY、左右接触力，这些张量仅由训练端冻结 Teacher 使用以构造动作标签。新 artifact kind/version fail-closed，不能加载原四输入 XY Student。
- Archived PandaHand RMA v5 replay：`TacEx-Sim2Real-Cube-Real-Alignment-RMA-Legacy-Student-Heatmap-DR-v0` 仅供采集旧 `tacex_rma_student` v5 / model v3 artifact；它保留 RGB 预测 XYZ 和左右接触概率的 30 维 Actor，不能用于训练或与当前 XY/force v2 artifact 混用。
- `train_rma_student.py` 冻结 Teacher Actor，并训练 spatial-softmax XY 分支；Student 与 Teacher 均在 Actor 内将左右接触力以每侧 `>=1 N` 二值化，双侧为真时才认为 grasped。接触 BCE 已移除，保留动作 MSE 与动作平滑约束。
- `TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-DR-v0` 是独立的 Student-only profile：复用通用 DR 的相机外参、GPU 内参 warp、颜色/模糊/噪声、板/背景和 DomeLight 扰动，但 `dr_curriculum_enabled=false`，因此从第一个 update 起始终为 full scale。它不改变 Teacher、接触标签、物理、奖励、动作或部署 TorchScript 输入；Student artifact 会记录 Clean 或 DR task，resume 只能在同一 profile 内进行。
- 当前 Real-Alignment visual nominal 基于用户提供的裁剪 D435 图作首轮人工对齐：保留测得的相机外参/内参，将近黑桌面、近黑背景和较暗中性 DomeLight 作为中心；DR 仅覆盖小残差（camera XYZ ±1.5 mm、RPY ±0.5°、focal ±1%、principal point ±1 px，以及受限的曝光、色彩、模糊和噪声）。画面中的真实导轨尚未建模，不能由 camera/photometric DR 代替；这是待确认的 scene-geometry gap。
- `evaluate_rma.py` 使用固定10x10 XY 网格；`export_rma_student_jit.py` 导出不含 privileged 输入的端到端 Actor，并对可用 CUDA 执行 CPU/CUDA reload 前向一致性验证；导出文件可由 `map_location="cpu"` 或 `"cuda:0"` 加载。
- `play_rma_student.py` 专门读取蒸馏 Student artifact，而不实例化 skrl Agent；它根据 checkpoint 的 task 选择 Clean 或全强度 DR 环境，校验 Teacher 环境 contract 与 encoder/Actor 哈希，在完整随机 XY 范围回放确定性 Student 动作，并写 CSV/JSON 成功率摘要。
- `train_rma_student.py --save_start_frame --start_frame_count N` 会在首次 reset 后将最多 `N` 个不同环境的 `wrist_rgb` 写到本次 run 的 `camera_frames/`；这是 Student 真正接收的 224×224 uint8 输入，DR task 的每张图保留各自 episode 的相机和外观随机化，不额外推进环境。
- PandaHand RMA 单独注册 cube ContactSensor，以每个30 Hz策略步内两次物理子步的左右最大力作为输入；每侧 `>=1 N` 且双侧同时满足才给 `contact_reward_weight=3.0` 的抓取奖励。动作变化惩罚保持 `rma_action_rate_penalty_weight=0.05`；当前 PandaHand RMA reward 不含单侧接触或切向力项。
- RMA 的 success 只作为奖励和统计条件，不触发 done；episode 只因 timeout 或严重机器人穿地碰撞结束，窗口成功率按 episode 内是否曾达到 success 统计。
- RMA 保持 Clean 的 `10 mm/step` 夹爪总宽度动作尺度，但单独降低 Panda finger actuator 到 `effort_limit_sim=40`、`stiffness=400`、`damping=40`。
- `source/tacex_tasks/tacex_tasks/sim2real_gelsight_rma` 是 RMA 的 GelSight robot profile 薄封装：其中全部 11 个已注册 task 统一加载持久组合资产 `franka_gsmini_standard_arm_visuals_v6.usd`，Franka 根位姿统一为 `(0,0,0.015) m`。v6 保持 `panda_hand` 与 `panda_link8` 原点重合，只将左右 finger、GelSight case/GelPad 和中心指尖参考沿 hand 局部 `+Z` 延长 `26 mm`；hand→GelPad 中心/最低点统一为 `0.1442/0.1613 m`，IK、reach/critic/safety 和 Teacher FK 使用同一几何。外部相机继续保持原 base_T_camera 标定，因此 world Z 同步降低 5 mm。GelSight RMA 的 Cube ContactSensor 过滤目标为 `gelpad_left/right`；GelSight `tactile_rgb` 不进入 Teacher Actor observation，Teacher 仍是 `proprio[15]+history[4]+cube_xyz[3]+contact[2]`。GelSight Student 使用第三视角 RGB 预测 Cube XYZ、左右 tactile RGB 预测 contact；Teacher 默认不创建触觉渲染 sensor，Student 按各自契约创建。
- GelSight Size-Buckets 是独立契约：8 个 4–6 cm 等间距尺寸按 `env_id % 8` 固定，一 env 一 Cube、reset 不缩放，环境数必须为 8 的倍数；3.5 m spacing、不可见共享 GroundPlane 和显式 collision filtering 隔离环境。gelpad `>0.2 N` 生成左右二值接触并给予单侧 +1.5/双侧 +3；Cube—非 gelpad 机器人或机器人—桌面统一为 `illegal_collision`，惩罚阈值在0–100k policy step从20 N线性降至5 N、权重固定 -10，`>10 N` 独立 terminated，Cube—桌面中性，5 s timeout 只 truncated。当前 Teacher manifest/Student checkpoint 为 v4/v5，环境 profile 为 v3。
- GelSight X040-DR Size-Buckets Teacher/三帧 Student 同样使用一 env 一 Cube：启动前按 `env_id % 8` 固定 4–6 cm 尺寸，reset 保持尺寸并继续采样 `x=0.40±0.08 m/y=±0.10 m` 与7臂关节截断高斯噪声。Student 的位置辅助头使用 X040-Wide robot-root XYZ 归一化，中心为 `[0.40,0,0.026] m`、尺度为 `[0.08,0.10,0.10] m`；ResNet18 必须从经 artifact 校验的 RMA XY Heatmap-DR Student encoder 初始化。RTX 渲染以 `render_interval=decimation=2` 对齐30 Hz策略步；左右内部 depth camera 不回读 latest pose，Taxim `[0,1]` 输出固定乘255转 `uint8`。异构碰撞几何采用 `replicate_physics=false`、3.5 m spacing、可见黑色共享 GroundPlane、显式跨 env 过滤和 GPU dynamics；`floor_panel` 为 `3.5×3.5 m`，visual-only backdrop 同宽。该路线与其他 GelSight task 一样加载统一 v6 USD，标准 Panda arm visual、绿色底座灯、GelSight 物理/碰撞和 26 mm finger offset 均包含在持久资产中。Teacher manifest/Student checkpoint 为 v5/v8；illegal-collision 惩罚阈值仍在0–100k从20 N收紧至5 N、权重为 -10，但接触力不再触发 terminated；机器人刚体原点穿过 ground 仍 terminated，timeout 仍只 truncated。观测、4维动作、奖励权重和 success 语义不变。
- 双 GelSight Pulled-Drawer 是与 X040 隔离的独立 profile，并按 `occluded_grasping` 的抽屉拓扑建模：Franka 根位姿为 `(0,0,0.015) m`；白色外柜为开口朝 `-X` 的四板壳体，拉出的抽屉为 8 mm 底板及四壁组成的顶部开口刚体布局。每个 env 在 PhysX 启动前按 seed 采样一个共享 isotropic scale `0.9–1.1×` 并同时应用到外柜/内抽屉 XYZ，另采样柜体 XY `±0.02 m` 与抽屉 Y `±0.02 m`；抽屉 X 由零接缝约束计算，标称中心为 `(0.5275,0) m`，此后 reset 不改几何。Cube 的 4–6 cm 八档尺寸在每组8个 env 内 seeded permutation 且固定，episode reset 相对抽屉中心采样 XY `±0.03 m`，Z 为底板顶面加 Cube 半边长。Student 逐 env reset 更新外柜/抽屉颜色和左右 GelSight reference，但 opacity 固定 `1.0`。Cube—抽屉与 Cube—GelPad 合法，机器人—外柜/抽屉及 Cube—外柜进入渐进碰撞惩罚与强碰撞终止；板件和机器人 solver 保持 `32/4`。Pulled-Drawer 现在复用全局 v6 USD 和 `0.1442/0.1613 m` 末端偏移，不再拥有额外叠加位移；Teacher 输入、Student 七路输入及三路输出不变。geometry/Teacher manifest/Student artifact 为 v9，Teacher model 为 v4、Student model/normalizer 为 v3；旧完整 checkpoint fail-closed，仅允许显式复用经校验的旧视觉 encoder 初始化。
- Size-Buckets Student 每个 episode 在首个 post-reset 触觉更新时分别锁存左右 reference，局部 reset 不修改其他 env。腕部 ResNet18 layer4 GAP 保留 512 维视觉特征；共享单侧 CNN 从左右 `(current-reference)/255` 各提取 256 维连续触觉特征，与归一化 `proprio[15]+history[4]` 拼成 1043 维，经独立 MLP 直接输出 `action[4]`。训练端以 Teacher mean action、cube XYZ、投影的 `14×14` cube-center heatmap 和左右 physics contact 分别监督动作、位置、heatmap 与接触头；这些 privileged label 不进入部署模型。部署七输入 key/shape 不变，v2/model-v1 的 XYZ/硬接触瓶颈 Student 不兼容。
- Teacher `params/rma_manifest.json` 绑定动作、位置/接触坐标语义、30维 feature order、Panda FK、归一化、RMA finger actuator、动作平滑 reward、robot/GelSight profile、contact filter 和配置哈希；Student checkpoint 绑定 Teacher/encoder/Actor 哈希。物体与末端姿态不进入 Actor；RMA v9 及更早 Teacher manifest 与当前 v10 训练目标不兼容。GelSight Teacher manifest 只能给 GelSight Student 使用，旧 PandaHand Teacher 与 GelSight Student 也会被 contract 拒绝。

## 第三部分 数据流

### 1. 总体数据流

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

### 2. 原始环境观测

环境观测由 `vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_observations` 构建，并以 `{"policy": obs}` 形式返回给 Isaac Lab / skrl wrapper。

配置定义位于 `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.observation_space`。

主要观测组：

- policy 可用：`proprio_obs`、`third_resnet`、四路 tactile feature。
- critic privileged：object pose/vel、gripper pose/vel、target relative position/distance。
- 额外统计/aux 输出：随派生环境和策略变体变化，需逐个 YAML 与 env class 确认。

`TacEx-T-{Scene}-{Object}` 触觉 + 本体 baseline 由 `vt_tactile_box.py` 提供，actor policy observation 会移除 `third_resnet`，只保留 `proprio_obs` 和四路 tactile feature；critic privileged keys 不变。

### 3. 视觉数据流

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
6. 若 torchvision 或 encoder 不可用，代码路径会使用零向量，具体运行风险见 `docs/DECISIONS.md` 的 ISSUE-007。

关键路径：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_observations`。

### 4. 四路触觉数据流

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
4. 输出第二部分第 8 节列出的四路 tactile keys。
5. GelFusion 分支 `vt_gelfusion_box.py:OccludedGraspingVTGelFusionBoxEnv` 额外读取当前四路 tactile RGB，与上一帧做绝对差分、阈值二值化，并输出 `tactile_dynamic_stats`。维度为 8，顺序为 left/right/left_down/right_down 每路 `[mean, variance]`。
6. 触觉 + 本体 baseline `vt_tactile_box.py:OccludedGraspingTactileProprioBoxEnv` 沿用同样四路 tactile feature，但 actor 不读取 `third_resnet`。

### 5. 本体状态数据流

本体观测 key：`proprio_obs`。

默认维度：18。依据：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.observation_space`。

生成位置：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_observations`。

组成：robot joint position 与 joint velocity 的拼接。具体 joint 顺序由 `self._robot` 的 articulation data 决定，当前文档不额外推断。

### 6. 跨模态融合

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

### 7. Actor 数据流

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

### 8. Critic 数据流

Critic 使用 privileged keys。以 `ppo_vt_alpha_gru.yaml` 为例，value network input 包含第二部分第 12 节列出的全部 `critic_*` keys，各自维度见下方第 13 节张量表。

生成位置：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_observations`。

### 9. 动作到环境

动作定义：`[dx, dy, dz, dyaw, gripper]`。

路径：

1. skrl policy 产生 5D action。
2. `vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._pre_physics_step` 清理 NaN/Inf、缩放动作、加入 action noise，并构造 IK command `[dx, dy, dz, 0, 0, dyaw]`。
3. `vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._apply_action` 调用 differential IK 得到 arm joint target。
4. `_apply_action` 使用第 5 维 action 按 `gripper_step_size` 增量更新夹爪。
5. `_robot.set_joint_position_target(joint_pos_des)` 写入关节目标。

### 10. 奖励计算

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

RMA reward 中的接触项使用同一个 ContactSensor。`rma_contact` 基于左右
finger/gelpad 总力模长是否超过 0.2 N 给奖励。当前 RMA reward 不再包含
cube-finger 世界 XY 切向力惩罚。

该项只作用于 RMA Teacher/Student reward，不进入 Student 部署输入。

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

GelSight RMA task family 复用上述 Actor 数据流，但 Student adaptation 改为视觉/触觉分工：
第三视角 `wrist_rgb` 预测 cube XYZ，`gsmini_left/right` 的 tactile RGB 预测左右接触概率。
RMA contact label/reward 来自 cube ContactSensor 对 `gelpad_left/right` 的两路 filter，
而不是旧 Panda finger body。Teacher 默认不创建 tactile-rendering sensors；GelSight Student
默认创建 tactile sensors 作为部署输入：

```text
gsmini_left.data.output["tactile_rgb"]  -> left tactile preview, 128x96 RGB
gsmini_right.data.output["tactile_rgb"] -> right tactile preview, 128x96 RGB
```

这些 buffer 只用于只读调试脚本 `save_rma_gelsight_preview.py` 或后续扩展；训练默认
不保存 PNG。Teacher/Student Actor observation key、Student
TorchScript 输入输出和 4维 action 均保持与普通 RMA 相同。

方块静止在台面时 lift reward 为 0；现实对齐 5 cm Cube 使用 `lift_start_delta=0 mm`、`success_delta=35 mm`，因此 17.5 mm 对应 lift 0.5。倾角仍计算并写日志，但不作用于 lift、success 或 done。RMA Teacher/Student 中 success 只用于奖励和 episode 统计，不触发 done；episode 仅因 timeout 或严重机器人穿地碰撞结束。木板 ContactSensor 是每环境单一 sensor body，过滤 `panda_link1–7`、`panda_hand` 和两侧 finger，因此方块落桌或 finger 接触方块不会单独触发撞桌惩罚。

基础任务的奖励控制台每 `reward_print_interval=200` 个 policy steps 打印一次。RMA Teacher/Student 单独使用精简行：`reach/lift/success/contact/table` 均为打印时刻跨环境平均后的加权奖励贡献，`total` 为当前平均总奖励，`avg_reward_200` 为最近200步的平均总奖励，`success_window_200` 为该窗口内已终止 episode 中曾达到 success 的比例，`avg_lift` 为当前跨环境平均的质心抬升高度（mm）。TensorBoard `extras["log"]` 仍保留原有完整诊断字段。

### 11. 终止与成功判定

终止函数：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_dones`。

Done 条件：

- episode timeout
- robot body 低于 `ground_height`

成功判定：

- 成功由物体高度超过 `success_height` 触发。
- success 会 latch 并用于 reward/统计。
- success 当前不直接终止 episode。

### 12. 训练和评估差异

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

### 13. 关键张量表

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
