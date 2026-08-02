# Design Decisions

## DEC-001 — 使用 Isaac Lab extension 拆分 core、assets、tasks、uipc

- 状态：已采用
- 背景：仓库需要同时维护触觉传感器实现、资产配置和 RL task。
- 决策：按 `source/tacex`、`source/tacex_assets`、`source/tacex_tasks`、`source/tacex_uipc` 拆分。
- 依据：`source/*/config/extension.toml`。
- 影响：模块边界清晰，但当前遮挡抓取主线是否依赖 `source/tacex_uipc` 待确认。

## DEC-002 — RL task 使用 Gymnasium registry 和 Isaac Lab task import 机制

- 状态：已采用
- 背景：训练/评估脚本需要通过 `--task` 选择环境和配置。
- 决策：`source/tacex_tasks/tacex_tasks/__init__.py` 通过 `import_packages` 导入任务包，`occluded_grasping/__init__.py` 调用 `gym.register` 注册 task。
- 依据：`source/tacex_tasks/tacex_tasks/__init__.py`；`source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_task`。
- 影响：训练入口只需传 task id；注册逻辑集中，但 legacy 和动态矩阵并存带来维护成本。

## DEC-003 — 遮挡抓取环境采用 DirectRLEnv

- 状态：已采用
- 背景：环境需要自定义 scene、传感器、动作控制、观测、奖励和 reset 流程。
- 决策：`OccludedGraspingVisionFourTactileBoxCfg(DirectRLEnvCfg)` 和 `OccludedGraspingVisionFourTactileBoxEnv(DirectRLEnv)`。
- 依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py`。
- 影响：生命周期清晰，但 `vt_box.py` 单文件职责过重。

## DEC-004 — 动作空间固定为 5 维 xyz + yaw + gripper

- 状态：已采用
- 背景：抓取任务主要需要平移、绕 z/yaw 调整和夹爪开合。
- 决策：policy 输出 `[dx, dy, dz, dyaw, gripper]`；roll/pitch 固定为 0；通过 differential IK 转为关节目标。
- 依据：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.action_space`；`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._pre_physics_step`；`_apply_action`。
- 影响：简化动作学习；若未来需要完整 6D 末端姿态，旧 checkpoint 不能直接兼容。

## DEC-005 — Actor 使用视觉/触觉/本体，Critic 使用 privileged state

- 状态：已采用
- 背景：训练时可访问 object/gripper privileged state，但部署 policy 不应依赖这些信息。
- 决策：policy 输入 `third_resnet`、四路 tactile feature 和 `proprio_obs`；value network 使用 `critic_*` keys。
- 依据：`vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.observation_space`；`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml`。
- 影响：符合 asymmetric actor-critic；评估时需确保 observation dict 与训练一致。

## DEC-006 — Success 不直接终止 episode

- 状态：已采用
- 背景：任务需要统计抬升后是否保持稳定，并惩罚成功后掉落。
- 决策：success 用于 reward 和统计 latch；done 由 timeout 或 robot body 低于 `ground_height` 控制。
- 依据：`vt_box.py:OccludedGraspingVisionFourTactileBoxEnv._get_rewards`；`_get_dones`。
- 影响：episode 长度不因成功提前结束；成功率解释时必须区分 success 与 done。

## DEC-007 — 使用可组合 task id，同时保留 legacy task id

- 状态：已采用
- 背景：已有实验可能依赖旧 `TacEx-VT-...-v0` 命名，新实验需要 scene/object/fusion 组合矩阵。
- 决策：注册 `TacEx-{Fusion}-{Scene}-{Object}`，同时保留 `TacEx-VT-{Fusion}-{Scene}-{Object}-v0`。
- 依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_scene_object_fusion_matrix`。
- 影响：兼容旧实验；但推荐入口和重复注册风险需长期维护。

## DEC-008 — 固定网格 bucket evaluation 用于 checkpoint 对比

- 状态：已采用
- 背景：随机 reset 难以比较不同策略在不同遮挡/位置条件下的表现。
- 决策：`play_bucket.py` 构造固定物体位置网格，逐点评估并写 CSV。
- 依据：`scripts/reinforcement_learning/skrl/play_bucket.py:_build_bucket_positions`；`play_bucket.py:main`。
- 影响：便于离线分析；bbox occlusion 默认关闭时，occlusion ratio 可为 `nan`，结果需标注评估条件。

## DEC-009 — 通过 import-time monkey patch 支持 skrl custom model class

- 状态：已采用但有风险
- 背景：skrl Runner 默认 component 加载不完全匹配当前 `module:Class` 配置需求。
- 决策：在 `occluded_grasping/__init__.py:_patch_skrl_runner_for_custom_models` 中 patch `Runner._component`。
- 依据：`source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_patch_skrl_runner_for_custom_models`。
- 影响：custom policy 加载更方便；但 import-time 全局副作用可能影响其他 skrl Runner 行为。是否改为局部 loader 待确认。

## DEC-010 — 已确认 checkpoint 作为当前文档示例基线

- 状态：已采用为文档示例，不代表最佳模型
- 背景：文档需要使用真实存在的 checkpoint 和命令。
- 决策：使用 `logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt` 作为普通评估和 bucket 评估示例。
- 依据：该 checkpoint、`params/agent.yaml`、`params/env.yaml` 和 metrics CSV 在仓库中存在。
- 影响：命令可追溯；该 checkpoint 是否为当前推荐 baseline 待确认。

## DEC-011 — Sim2real Cube 使用统一的 frozen-vision、history 和 bounded-mean contract

- 状态：已采用；需要按新 contract 重新训练
- 背景：训练时 BatchNorm statistics、策略看到的动作历史和导出 Actor mean 与真机推理存在漂移，会直接破坏 sim2real 输入/输出语义。
- 决策：ResNet18 参数冻结且每次编码前强制 `eval()`；`action_history(t+1)` 保存与 transition 对应、privileged `dz` gate 之前的 `processed_actions(t)`；Cube Gaussian Actor 在模型输出处使用 `tanh(ACTIONS)`。
- 依据：`cylinder_grasping_vision_only_resnet18.py`、`sim2real_grasp_env.py`、两个 `skrl_ppo_cube_*_cfg_resnet18.yaml`。
- 影响：观测 key、`512+15+4` Actor 输入维度和 4 维动作 shape 不变，但视觉、时间和 Actor mean 语义改变；修复前 checkpoint 不能继续训练或直接迁移。

## DEC-012 — Encoder provenance 作为 run-level artifact 保存并在导出时 fail closed

- 状态：已采用
- 背景：视觉 encoder 位于环境而不在 PPO Actor checkpoint 中；exporter 重新创建 ImageNet encoder 无法证明它与训练实际使用的 state 完全一致。
- 决策：训练 run 的 `params/` 保存 frozen ResNet18 state dict 和 manifest；manifest 记录架构、权重 ID、normalization、task、policy/history contract、agent/env config hash 和 encoder SHA-256。Resume/play/export 校验 run contract；exporter 恢复 saved env cfg，严格加载 encoder，并比较 eager/traced/reloaded output 与 embedded encoder hash。
- 依据：`scripts/reinforcement_learning/skrl/vision_encoder_artifact.py`、`train.py`、`export_sim2real_grasp_jit.py`。
- 影响：encoder 不重复写入每个 checkpoint，但 deployment artifact 必须与整个 run 的 `params/` 一起保存；缺失或 hash 不符时拒绝 RGB 导出。

## DEC-013 — Real-Alignment Cube 使用缓存总宽度增量控制夹爪

- 状态：已采用；需要重新训练
- 背景：Isaac Sim 的两个 finger joints 是资产实现细节，旧环境每个 physics substep 基于实测 finger position 重新累加 target，无法与真机总宽度接口形成稳定的一步一命令语义。
- 决策：第四维 clipped action `g` 每个 30 Hz policy step 请求 `0.01*g m` 总宽度增量；在缓存目标上累加并限制到 `[0,0.08] m`，两侧 target 均为总宽度一半；两个 60 Hz physics substep 只重发缓存 target。`action_history[3]` 保存请求的总宽度增量。
- 影响：动作和观测 shape 不变，XYZ scale 仍为 `0.05 m`；第四维 history 和夹爪 transition 语义改变，旧 Real-Alignment checkpoint 不兼容。当前真机仍为 `2 mm/step`，本次不修改 `franka`，只对齐总宽度接口语义。

## DEC-014 — Real-Alignment 仿真直接渲染现实中央 crop 的 224x224 等效视野

- 状态：已采用；需要重新训练
- 背景：仿真先渲染 640x480 再缩放到 224x224 会保留高分辨率相机缓冲，无法达到降低并行环境显存占用的目标；完整 4:3 帧直接压成正方形也不匹配当前真机的中央方形 crop。
- 决策：真机保持 `640x480 -> crop x=[80,560), y=[0,480) -> bilinear resize 224x224`；仿真将原始 K 按同一 crop 和 `224/480` scale 转换，直接渲染 224x224。
- 影响：Actor observation key、`wrist_resnet:512`、动作、奖励和 done 不变；相机 contract 和视觉分布改变，旧 Real-Alignment checkpoint 不得续训或部署。Omniverse 4.5 会近似非中心主点和非方形像素，仍需真机/仿真关键点对齐验证。

## DEC-015 — Real-Alignment 分离 Clean 与 broad DR task

- 状态：已采用；两个 profile 均需独立训练
- 背景：固定对齐误差与域随机化收益需要独立评估；在同一 task 中直接开关随机化会让 run provenance 和 checkpoint 语义不清晰。
- 决策：现有 `TacEx-Sim2Real-Cube-Real-Alignment-v0` 作为只随机 cube XY 的 Clean 基线；新增 `TacEx-Sim2Real-Cube-Real-Alignment-DR-v0`，仅通过配置继承增加视觉/场景随机化，并使用独立 Agent YAML/日志目录。具体 curriculum 和随机化范围由 DEC-018 约束。
- 影响：两个 task 的张量 shape、动作、控制、奖励和 done 完全相同，但 task id、env config hash 和视觉分布不同；checkpoint 不允许跨 profile resume。质量、摩擦和机器人动力学仍不随机化。

## DEC-016 — Real-Alignment 分离固定 IK TCP 与动态抓取中心

- 状态：已采用；需要重新训练
- 背景：旧实现把 `panda_hand` 固定偏移 `0.107 m` 同时用于 IK 和 reach reward，不能直接表达左右夹指实际位姿；现实 TCP 对齐中标准 Panda 几何为 hand 到 finger origin `0.0584 m` 加 finger origin 到 tip `0.045 m`，合计 `0.1034 m`。
- 决策：Differential IK/Jacobian 使用固定 `panda_hand+[0,0,0.1034] m`；reach reward、critic gripper position/target distance 和当时的 privileged `dz` gate 使用左右 finger link 各经本地 `[0,0,0.045] m` 变换后的世界坐标中点。Clean/DR 使用同一语义，基础 Cube/Cylinder 通过默认 hook 保留旧行为；Cube gate 后续由 DEC-020 删除。
- 影响：Actor keys、Actor 输入维度、action shape 和 done 不变；reward 距离、critic privileged state、安全 gate 和 IK TCP 数值改变。strict contract 增加 TCP/中心字段，缺少这些字段的旧 Real-Alignment checkpoint fail closed，必须重新训练。

## DEC-017 — Real-Alignment 外部控制链路统一为 30 Hz

- 状态：已采用；需要按新时间语义重新训练
- 背景：`decimation=1` 时 policy 为 60 Hz、camera 为 30 Hz，相邻 policy step 可能复用同一图像，且真机 30 Hz 动作/history 时间尺度不一致；直接把 physics 降到 30 Hz 又会降低抓取接触稳定性。
- 决策：physics 保持 `dt=1/60 s`，使用 `decimation=2`；camera `update_period=1/30 s`，render interval 跟随 decimation。camera/render/policy/action/observation/reward/history 均为 30 Hz，每个动作由两个 physics substep 执行。episode 设为 5 秒，即 150 个 policy steps。
- 影响：张量 shape、动作尺度、奖励阈值、success/done 类型不变；每秒动作累计幅度、history 时间尺度和 timeout horizon 改变。旧 60 Hz policy checkpoint 不得在当前 30 Hz 环境中续训或部署。

## DEC-018 — Real-Alignment DR 使用按训练步增长的 per-episode curriculum

- 状态：课程机制保留；10% 初值和旧颜色中心已由 DEC-027 取代
- 背景：一开始使用完整相机、光照和后处理范围会增加探索难度；但按帧重采样相机或曝光会形成现实中不存在的高速跳变，多环境中的共享 DomeLight 也不能被单个环境独立控制。
- 决策：DR range scale 在 0–20k policy step 为 `0.10`，20k–120k 线性增到 `1.00`，之后保持 full range。相机 `XYZ ±3 mm/RPY ±1°`、focal `0.985–1.015`、principal point `±2 px`、图像颜色/模糊强度、plate/backdrop 颜色按该 scale 收缩并在每环境 reset 时采样一次；Gaussian pixel noise 按 episode 固定 std 但逐帧采样。DomeLight intensity `1000–3000`、temperature `3800–7200 K` 只在 full-batch reset 更新，global ground 固定。
- 多环境语义：外参通过 `TiledCamera.set_world_poses(..., env_ids=...)` 独立设置；内参和图像后处理在 GPU 上批处理；plate/backdrop 使用每环境独立 material。部分 reset 不改变未 reset 环境缓存，也不改变共享 DomeLight。
- 影响：Actor/Critic key 和 shape、动作、奖励及 done 不变；视觉分布和 env config hash 改变，已有 DR checkpoint 不可直接作为新 profile 的等价训练结果。课程计数基于 `common_step_counter`，resume 时需显式设置 `dr_curriculum_step_offset`。

## DEC-019 — 两个 sim2real Cube task 使用质心 lift 与收紧 success 倾角课程

- 状态：已采用；需要重新训练
- 背景：旧奖励使用理论 reset 最低角点，而 cube/plate 两侧各 `0.5 mm` rest offset 使静止方块天然出现约 `1 mm` 的假 lift delta；最低角点还会把倾斜导致的包围盒变化混入质心抬升。控制台只打印某一时刻的 env mean，不能反映最近训练窗口的平均每步回报。
- 决策：`TacEx-Sim2Real-Cube-Grasp-v0` 和 Real-Alignment Clean/DR 都以 `reset center + cube rest offset + plate rest offset` 为稳定落桌质心基准，质心 `0–35 mm` 线性映射到 lift `[0,1]`。倾角不再清零 lift，只约束 success；最大允许倾角从 step 0 的 `40°` 线性收紧到 step 120k 的 `10°`，使策略先学习抓取/抬升，再提高姿态质量。控制台新增最近 `reward_print_interval` 个 policy steps 的平均每环境/每步总奖励。
- 影响：reward、success/done 和 env config hash 改变；Actor/Critic observation key/shape、4-D action、控制和部署输入输出不变。Policy contract 升级到 v3，strict Cube v2 contract 使用旧 lift 语义并 fail closed。

## DEC-020 — Sim2real Cube 删除 ground-truth object-XY `dz` gate

- 状态：已采用；需要重新训练
- 背景：旧 `_pre_physics_step()` 在 Actor 输出后读取仿真方块真实 XY 和指尖/台面高度，阻止或限制负向 `dz`。Actor 本身没有该输入，TorchScript export 也不包含 gate，真机因此无法复现训练时的实际控制链。
- 决策：父 Cylinder 环境增加 `privileged_dz_gate_enabled` 开关并默认保留旧行为；`TacEx-Sim2Real-Cube-Grasp-v0`、Real-Alignment Clean/DR 全部设为 false。Cube 的 XYZ processed action 直接送入 IK，`action_history` 与请求 XYZ 一致；普通 IK、关节和工作空间限制仍保留。
- 影响：Actor/Critic observation key/shape、4-D action、奖励和 done 不变，实际下降控制语义改变。Policy contract 升级到 v4并记录 `privileged_dz_gate=disabled`；Cube v1/v2/v3 checkpoint（包括 2026-07-25 22:01 的 200k run）fail closed，必须重新训练。

## DEC-021 — Real-Alignment v5 统一几何、视觉、奖励与真机动作 contract

- 状态：已采用；其中物体 x 范围后由 DEC-022 更新
- 背景：旧 Real-Alignment 的物体范围、台面高度、相机外参、XYZ/夹爪动作幅度和倾角 success gate 与当前真机设置不一致，也没有显式的机械臂撞桌学习信号。
- 决策：Clean/DR/Privileged Real-Alignment 共用 1 mm 台面、方块绝对范围 `x=[0.50,0.70], y=[-0.10,0.10] m` 及 20k–100k 位置课程；相机改用指定的 ROS optical 外参和 `640x480 -> x=[80,560) crop -> 224x224` contract。控制统一为 30 Hz XYZ `5 mm/step`、夹爪总宽度 `2 mm/step`、一拍逐维 history。Lift 仍使用质心 0–35 mm，success 只要求 35 mm 连续 5 步；倾角仅诊断。目标 Franka link 对台面接触力 `>1 N` 时奖励 `-10`，不直接终止。
- Clean/DR：Clean 固定近黑台面/背景和白方块；DR 保留视觉课程和 per-env 随机化，但继承同一几何、动作、奖励与 success。评估、bucket 和 rollout 强制完整物体范围。
- 部署：Policy contract 升级到 v5并记录相机/crop/序列号、动作/history、物体课程、台面和碰撞语义；旧 v4 Real-Alignment fail closed。新增独立真机 v5 配置并保留旧配置。
- 影响：Actor/Critic observation shape 和 4-D action shape 不变；动作物理幅度、视觉分布、reward/success、reset distribution 和 env config hash 改变。所有旧 Real-Alignment checkpoint 不得续训或部署。

## DEC-022 — Real-Alignment 方块 x 范围前移到相机完整可见区域

- 状态：已采用；需要从头训练
- 背景：在当前真实标定外参和 224x224 crop 等效视野下，5 cm 方块中心位于 `x=0.70 m` 时，其最下沿投影约为 `v=246`，超出图像；内参变换正确，问题来自工作区与相机外参覆盖范围不匹配。
- 决策：保持相机内外参、y 范围和动作尺度不变，将 Clean/DR/Privileged 共用的方块 nominal x 从 `0.60 m` 改为 `0.50 m`，完整范围由 `x=[0.50,0.70] m` 改为 `x=[0.40,0.60] m`。位置课程前 20k steps 相应使用 `x=[0.48,0.52] m`，100k 后覆盖完整范围。
- 影响：Actor/Critic observation shape、4-D action、奖励、success 和 done 不变；reset distribution 与 contract 中的 nominal/bounds 改变。按旧 x 范围生成的 Real-Alignment v5 checkpoint 会被 live contract 校验拒绝。

## DEC-023 — Real-Alignment XYZ 最大增量提高到 10 mm

- 状态：已采用；需要从头训练
- 背景：初始指尖中心到方块台面高度的垂直距离约 274 mm，5 mm 最大增量理论上至少需要约 55 个饱和 policy steps；用户确认该范围过小。
- 决策：Clean/DR/Privileged Real-Alignment 的 XYZ scale 从 `0.005` 提高为 `0.010 m/policy step`，夹爪总宽度 scale 保持 `0.002 m/policy step`。History 同步记录 `[0.010*u_x,0.010*u_y,0.010*u_z,0.002*g]`；真机部署配置使用完全相同的逐维 scale。
- 影响：Actor/Critic observation shape 和 4-D action shape 不变；XYZ transition 和 history 数值范围改变。Policy contract 升级到 v6，Real-Alignment v5 checkpoint fail closed，必须重新训练。

## DEC-024 — Real-Alignment 采用实测 400x398 crop 与有效 K

- 状态：已采用；需要从头训练
- 背景：2026-07-26 19:48 的 D435 样本确认真机实际使用 `x=[100,500), y=[34,432)` crop，再将 400x398 双线性缩放到 224x224；此前 v6 仍声明中央 480x480 crop。Omniverse 4.5 原生相机会强制居中主点与方形像素，直接把新 K 传给 USD 不会真实保留 `cx/cy` 和 `fx/fy` 差异。
- 决策：真机使用上述 crop；策略有效 K 固定为 `[338.742544,0,123.748857; 0,340.550811,120.393372; 0,0,1]`。仿真保持 224x224 camera buffer，原生渲染 centered `K_native=[300,0,112; 0,300,112; 0,0,1]` coverage view，再以固定 batched GPU affine warp 得到目标 K；native 300 px focal 覆盖全部目标像素光线。DR 在该固定映射后继续施加其每环境随机 intrinsic warp。
- 部署与兼容：Policy contract 升级到 v7并记录 raw K、crop、目标 K、native render K 和补偿模式；新增 `franka/configs/e2e_bundle_real_alignment_v7.json`，保留 v6/0712 配置。Real-Alignment v6 及更旧 checkpoint fail closed。
- 影响：Actor/Critic observation key/shape、4-D action、奖励、success、done 和 camera buffer resolution 不变；策略实际视觉分布与部署 crop 改变，必须从头训练。当前 `x=0.60 m` 方块在新 crop 下可能接近或越过底边，完整 reset 范围的可见性仍需单独 smoke。

## DEC-025 — Real-Alignment 以独立 v8 bundle 部署 25/2 mm checkpoint

- 状态：已采用；真机闭环待验证
- 背景：`2026-07-26_20-51-57_ppo_torch_vision_only_resnet18` 的保存环境明确使用 XYZ `0.025 m/step`、夹爪总宽度 `0.002 m/step`；强行套用固定 10/2 mm 的 v7 真机配置会同时破坏动作幅度和 action history 语义。
- 决策：保留 v7 的 10/2 mm 严格读取能力，新增 v8 严格绑定 `[0.025,0.025,0.025,0.002]`，并为该 run 新建 `franka/configs/e2e_bundle_real_alignment_v8_25mm.json`。不通过修改模型输出或仅修改 exporter 来补偿尺度。
- 影响：Actor/Critic observation shape、4-D action shape、相机和奖励不变；v7/v8 的张量 shape 相同但控制 transition 不同，部署时必须由 metadata 与配置双向精确校验。25 mm 单步平移在真机上风险较高，首次测试必须预览后单步确认。

## DEC-026 — Clean/DR 采用 5 mm 夹爪增量与 XY ±5 cm 范围

- 状态：已采用；需要从头训练
- 背景：用户要求缩小 Clean 的物体 XY reset 范围，并提高每个 30 Hz policy step 的夹爪总宽度控制幅度；DR 必须继承相同的物理、控制和 reset 语义。
- 决策：Clean/DR 保持 nominal cube `(0.50,0.00) m`，完整范围改为 `x=[0.45,0.55], y=[-0.05,0.05] m`；前 20k step 仍为 `x/y ±0.02 m`，20k–100k 线性扩展到 `±0.05 m`。第四维 clipped action 映射为总宽度增量 `0.005*g m/policy step`，XYZ 保持 `0.025 m/step`。Privileged 诊断变体显式保留旧 `0.002 m` 夹爪尺度和 `x/y ±0.10 m` reset 范围。
- 影响：动作与观测 shape、奖励、success、done、相机和频率不变；Clean/DR 的 action history 第四维、控制 transition、reset distribution 与 env hash 改变。Policy contract 升级为 v9；v7/v8 仍可按各自保存的语义读取，但不能在 v9 live 环境续训或播放。新真机 bundle 待 v9 策略训练完成后生成。

## DEC-027 — DR 严格以 Clean 视觉场景为零扰动基线

- 状态：已采用；取代 DEC-018 的 10% 初始范围和绿色背景中心
- 背景：旧 DR 虽继承 Clean 的物理、动作和奖励，但 plate 中心为灰绿色、backdrop 中心为绿色，DomeLight 也切换到另一套 renderer color-temperature 模型；因此 DR 即使在课程起点也不是 Clean 场景，不利于 Clean checkpoint 到 DR 的连续课程学习。
- 决策：DR scale 在 0–20k policy step 固定为 0，20k–120k 从 0 线性增至 1。plate/backdrop center 直接复用 Clean 的 `(0.02,0.02,0.02)` 和 `(0.01,0.01,0.01)`；full ranges 分别为每通道 `[0.01,0.05]` 与 `[0.005,0.03]`。DomeLight intensity/color 以 Clean `2000/(0.75,0.75,0.75)` 为中心，色温随机化改为相对 5500 K 的 RGB tint，避免 scale=0 切换 renderer 光照模型。相机、等效内参、GPU 后处理在 scale=0 全部为 identity/标定值。
- 影响：物理、动作、观测 key/shape、奖励、success、done，以及相机/内参/图像后处理/光照的 full ranges 不变；plate/backdrop full ranges、DR 视觉训练分布及 env config hash 改变。Clean 与 DR 的网络 shape/控制语义一致，但当前 strict resume 不跨 profile；如需用 Clean 权重初始化 DR，应另行实现并验证 weight-only warm start。旧 DR checkpoint 不作为新课程的等价续训起点。

## DEC-028 — DR 在物体位置课程完成后再启动

- 状态：已采用；取代 DEC-027 中的 20k–120k DR 时间表，不改变其 Clean-centered 随机化定义
- 背景：旧 200k DR run 在物体位置范围和视觉扰动从 20k 起同时扩大；训练曲线在课程后段出现 reach/lift 回退，且累计成功接近 0。用户确认不希望将零扰动阶段延长到 200k，选择 100k 作为 DR 起点。
- 决策：DR 在 0–100k 保持 `scale=0`，使继承的物体位置课程先扩展到完整 `x/y ±5 cm`；100k–220k 将所有 DR range 线性增至 1，220k–300k 保持 full range。DR Agent trainer 独立延长到 300k，并记录 `info/dr_curriculum_scale`；Clean Agent 仍为 200k，Clean 环境配置不变。
- 影响：只改变 DR 训练分布、训练时长、日志和 env config hash；Actor/Critic observation key/shape、4-D action、奖励、success、done、物理和 Clean task 均不变。旧 DR checkpoint 不得按当前 profile resume，必须从头训练。

## DEC-029 — Real-Alignment 使用单帧 RMA-style 教师学生蒸馏

- 状态：已实现；训练结果待确认
- 背景：视觉策略需要区分控制学习与物体定位误差，同时部署端不能使用仿真特权状态。
- 决策：新增独立 Teacher/Student task。Teacher 使用当前 Clean v9 物理与控制，并以机器人根坐标系 cube XYZ 作为唯一物体特权输入；共享 Actor 从 `proprio_obs[:7]` 通过固定 Panda FK 计算指尖中点 XYZ，并追加 cube-to-gripper XYZ，使网络输入为28维。Student 用冻结 ResNet18 layer4 和可训练 spatial-softmax head 预测 XYZ。Teacher Actor Core 冻结共享，逐步联合优化位置 SmoothL1 与动作 MSE，仿真只执行 Student action。Teacher 200k、Student 100k，不做 PPO fine-tune。当前固定末端朝向、位置动作与 position-only 成功语义不需要 cube/末端 quaternion，因此不加入姿态。
- 兼容性：不复用旧 Privileged checkpoint；RMA manifest/checkpoint 绑定 normalization、Panda FK、28维 feature order、25/5 mm action contract 和 Teacher/encoder 哈希。RMA v1 的22维 checkpoint fail closed，必须重新训练；现有 Clean/DR/Privileged checkpoint 与行为不变。
- 验收：固定100位置 Teacher success >=80%，Student success >=Teacher的90%，Student 3D RMSE <=15 mm；结果必须来自评估 JSON/CSV。

## DEC-030 — RMA Teacher 使用特权接触且 Student 从 RGB 预测接触

- 状态：已实现；训练结果待确认
- 背景：原 RMA Teacher 只有几何位置，训练日志中几乎没有 lift/success，缺少明确的夹持接触中间目标。部署端不能读取仿真接触真值。
- 决策：只在 RMA Teacher/Student task 注册 cube-to-left/right-finger 过滤 ContactSensor。每个30 Hz policy step取两个物理子步的最大力，0.5 N以上记为接触；单侧接触奖励0.1，双侧接触奖励2.0。Teacher Actor 接收真实二值接触；Student adaptation head 从单帧 RGB 预测两路 logits，以 BCE 监督，并向冻结 Actor 输入 sigmoid 概率。总蒸馏损失为位置 SmoothL1、接触 BCE 和动作 MSE 的加权和。
- 影响：RMA Actor feature 从28维增至30维，后续又将 success 改为非终止统计条件并降低 RMA-only finger actuator；model 仍为v3，manifest/student/export artifact 升至v5，v4及更早 RMA checkpoint fail closed并需重新训练。Student TorchScript 外部三输入和4维动作不变；Clean、DR、旧 Privileged 的观测、奖励和 checkpoint 不变。
- 边界：单帧 RGB 下接触可能因遮挡、形变不可见或时序而部分不可观，实际 precision/recall/F1 必须由固定网格评估确认。

## DEC-031 — RMA success 不再触发 episode done

- 状态：已实现；训练结果待确认
- 背景：RMA 教师训练中 success 终止会让策略刚抬起即重置，减少继续保持、调整接触和稳定夹持的样本。
- 决策：仅在 RMA Teacher/Student 中移除 success terminal；success 仍给奖励、更新 hold counter，并以 episode 内曾经达到 success 的方式进入滚动成功率。Done 只保留 timeout 和严重机器人穿地碰撞。
- 影响：Clean、DR 和旧 Privileged 的 done 语义不变。RMA manifest/student/export artifact 升至v4；后续 RMA-only finger actuator 变更又升至v5，旧 v4 RMA checkpoint 不再兼容并需重新训练。

## DEC-032 — RMA 单独降低 finger actuator 但保持5mm夹爪动作尺度

- 状态：已实现；训练结果待确认
- 背景：RMA 中 success 不终止后，策略会在成功后继续执行，强 finger 位置 PD 容易把 cube 压入夹指或桌面。用户明确要求不要把夹爪动作尺度改为2mm/step。
- 决策：只在 RMA Teacher/Student 的 copied robot cfg 中覆盖 `panda_hand` actuator 为 `effort_limit_sim=40`、`stiffness=400`、`damping=40`；`gripper_width_delta_scale` 仍继承 Clean 的 `0.005 m/policy step`。
- 影响：Clean、DR、旧 Privileged 和 GelSight asset 不变。RMA 环境物理/控制 contract 改变，manifest/student/export artifact 升至v5，v4及更早 RMA checkpoint 需重新训练。
