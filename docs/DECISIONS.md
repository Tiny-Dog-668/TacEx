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
