# Development Log

本文件记录工程、代码、配置和文档维护变更。实验结果记录到 `docs/EXPERIMENTS.md`，设计取舍记录到 `docs/DECISIONS.md`，问题和风险记录到 `docs/KNOWN_ISSUES.md`。

## 记录规则

- 每条记录必须写明日期、修改范围、依据和验证情况。
- 不要把未运行的训练或评估写成已完成结果。
- 不确定的作者、意图、commit、seed、checkpoint 或结果统一写“待确认”。
- 涉及观测、动作、奖励、done、网络输入维度或 checkpoint 兼容性的改动必须明确说明。

## 2026-07-12 CST — 统一 sim2real Cube 训练与导出 contract

- 类型：环境修复 / Agent 配置 / export provenance / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/cylinder_grasping/cylinder_grasping_vision_only_resnet18.py`
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_grasp_env.py`
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/agents/skrl_ppo_cube_vision_only_cfg_resnet18.yaml`
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/agents/skrl_ppo_cube_real_alignment_cfg_resnet18.yaml`
  - `scripts/reinforcement_learning/skrl/vision_encoder_artifact.py`
  - `scripts/reinforcement_learning/skrl/train.py`
  - `scripts/reinforcement_learning/skrl/play.py`
  - `scripts/reinforcement_learning/skrl/play_bucket.py`
  - `scripts/reinforcement_learning/skrl/export_sim2real_grasp_jit.py`
  - `scripts/reinforcement_learning/skrl/test_sim2real_grasp_policy_jit.py`
  - `scripts/reinforcement_learning/skrl/summary_utils.py`
  - `README.md`、`docs/PROJECT_OVERVIEW.md`、`docs/ARCHITECTURE.md`、`docs/DATA_FLOW.md`、`docs/EXPERIMENTS.md`、`docs/DECISIONS.md`、`docs/KNOWN_ISSUES.md`、`docs/DEVLOG.md`
- 修改内容：
  - ResNet18 初始化后立即 `eval()` 并冻结全部参数；两个实际 feature extraction 路径在每次 forward 前再次强制 `eval()`，使用 `torch.no_grad()`，避免 BatchNorm running statistics 在训练 rollout 中变化，同时让输出保持为可供下游 Actor 反向传播使用的普通 detached tensor。
  - `action_history` 从 `prev_actions` 改为当前 `processed_actions.detach().clone()`，使 observation `t+1` 记录与 transition 对应的上一拍 scaled/clamped command；sim2real action noise 为 0，所以其范围约为 `[-0.05,0.05]`。该值位于 privileged near-table `dz` gate 之前，不一定等于最终 IK `dz`。
  - 两个 Cube PPO 配置的 Gaussian Actor 输出改为 `tanh(ACTIONS)`；训练、按 checkpoint saved cfg 的 deterministic play 和 export 返回相同的 bounded mean。训练时 Gaussian sample 仍可能越界，环境继续负责 processed-action clamp。
  - 每个 sim2real vision training run 在 `params/` 保存 encoder state dict 与 manifest；manifest 绑定 task、agent/env config hash、history 版本、相机/频率、action scale、夹爪 substep target 重发语义和 encoder hash。旧 sim2real resume、错 task、source config 篡改、Actor/preprocessor、normalization 或已声明 live-env 字段漂移会 fail closed；play/play_bucket/exporter 直接恢复 saved env cfg，加载同一 encoder，并在 export 中检查 finite output 及 eager/traced/reloaded 一致性。
- 影响的观测：key 和 shape 不变，仍为 `wrist_resnet:512`、`proprio_obs:15`、`action_history:4` 及原 `critic_*`；`action_history` 的时间语义改变。
- 影响的动作：shape 不变，仍为 `[N,4] = [dx,dy,dz,gripper]`；deterministic Actor mean 从 unbounded 改为 `[-1,1]`，乘 `action_scale=0.05` 后 processed command 为约 `[-0.05,0.05]`。
- 影响的奖励和 done：公式与阈值未修改；action-rate reward 仍使用 `processed_actions` 和 `prev_actions`。
- checkpoint 兼容性：修复前 checkpoint 虽然参数 shape 兼容，但已适应旧 BatchNorm/history/Actor 语义，不能 resume 或通过 exporter 临时补 `tanh`；必须从头训练。共享基类的 BN/history 修复也影响 `TacEx-Cylinder-Grasping-Vision-Only-ResNet18-v0` 和 `CylinderGraspingComplexEnv` 子类，对应旧 checkpoint 同样存在语义变化。
- 验证情况：
  - ResNet18 动态检查：train + `no_grad` 会改变 60 个 BatchNorm buffers；eval + `no_grad` 改变 0 个。
  - 两个 Cube YAML 通过 skrl `gaussian_model` 实例化，随机 batch 的 deterministic mean shape 为 `[32,4]` 且全部位于 `[-1,1]`。
  - encoder artifact 临时目录 round-trip 通过：artifact/state hash 一致，strict load 后 encoder 为 eval 且全部参数 frozen。
  - Isaac headless 环境 smoke：非零 raw action `[0.4,-0.2,0.1,-0.5]` 得到 history `[0.02,-0.01,0.005,-0.025]`，一次 step 后 60 个 BN buffers 变化数为 0；观测 shape 为 `512+15+4`。
  - PPO/export/play smoke：`2026-07-12_14-29-22_ppo_torch_vision_only_resnet18` 以 1 env 完成 128 steps 和一次 update，退出码 0；生成的 skrl policy source 明确为 `nn.functional.tanh(net)`，并保存 encoder artifact、v1 contract 和测试 checkpoint `agent_128.pt`。RGB/feature 两种 export 的 eager/trace/reload error 均为 0；独立 tester 得到 `(2,4)` finite/bounded output、repeat diff 0；普通 play 用 exact saved env cfg 和 verified encoder 执行 2 steps，退出码 0。该 smoke checkpoint 不能作为可用策略。
  - `python -m compileall source scripts tools`、最终相关文件 `py_compile` 和 `git diff --check` 通过。
  - `TERM=xterm conda run -n isaaclab_2.1.1 --no-capture-output ./tacex.sh -p tools/run_all_tests.py --discover_only` 完成 Isaac warm start 后失败：测试工具配置要求跳过不存在的 `test_argparser_launch.py`。这是 discovery 配置问题，不是本次环境 smoke 失败。
- 未运行：修复后的完整 200000-step PPO、bucket 成功率和真机部署。
- 待确认：修复后策略的训练收敛、仿真抓取成功率和真机成功率。

## 2026-07-12 CST — 导出现实对齐 Cube best policy

- 类型：checkpoint artifact / TorchScript export / 文档
- 修改文件：
  - `logs/skrl/sim2real_cube_real_alignment/2026-07-11_23-15-23_ppo_torch_vision_only_resnet18/checkpoints/exported/policy_actor_e2e_best_agent.pt`
  - `logs/skrl/sim2real_cube_real_alignment/2026-07-11_23-15-23_ppo_torch_vision_only_resnet18/checkpoints/exported/policy_actor_e2e_best_agent.json`
  - `docs/EXPERIMENTS.md`
  - `docs/DEVLOG.md`
- 修改内容：
  - 从该 run 的 `best_agent.pt` 导出包含 frozen ResNet18、state preprocessor 和 PPO Actor 的端到端 CPU TorchScript。
  - 输入 contract 为 `action_history [N,4]`、`proprio_obs [N,15]`、`wrist_rgb [N,224,224,3] uint8 RGB`；输出为 raw mean `[dx,dy,dz,gripper]`。
  - metadata 记录 nominal 30 Hz、environment action scale 0.05 和 trace error。
- 影响的观测、动作、奖励、done、网络维度：无代码修改；仅固化已训练 checkpoint。
- checkpoint 兼容性：源 checkpoint 已成功加载；export 专用于非 recurrent TorchScript inference。
- 验证情况：
  - export 退出码 0，`trace_max_abs_err=0`。
  - synthetic RGB、run 内仿真起始帧、真实 `rgb_model_median.png` 三种输入均得到 finite `(1,4)` output，repeat diff 为 0。
  - 本机 CPU 平均推理约 8.4-9.3 ms；TorchScript 大小 44.53 MiB，参数量 11,621,640。
  - SHA-256：PT `9606058905f7204c54a5da24d56bfe139e2902ce889b980c72e5311bafd4fa67`；JSON `6095c2a91ddac73d895b963e9b301f1e18a2bbefb4952cb7a6dcd253e4db8dc3`。
- 待确认：
  - `play_metrics_20260712_120929.csv` 中 success rate 为 `nan`，仿真有效成功率待重新评估。
  - 尚未执行真实 Franka 闭环部署、安全限幅验证或真实抓取成功率评估。

## 2026-07-04 CST — 新增四路触觉 + 本体 baseline 环境

- 类型：代码 / 配置 / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_tactile_box.py`
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_tactile.yaml`
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py`
  - `README.md`
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/README.md`
  - `docs/PROJECT_OVERVIEW.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DATA_FLOW.md`
  - `docs/DEVLOG.md`
- 修改内容：
  - 新增 `T` fusion 分支，可生成 `TacEx-T-Drawer-Occlusion-Cube`、`TacEx-T-Self-Occlusion-Cuboid` 等 Scene/Object 矩阵 task。
  - 新增 `OccludedGraspingTactileProprioBoxEnv`，继承 full VT 环境并从 actor policy observation 中移除 `third_resnet`。
  - `OccludedGraspingTactileProprioBoxCfg` 显式声明不含 `third_resnet` 的 `observation_space`，避免在 `configclass` 处理后的父类上进行 import-time 类属性读取。
  - 新增 `ppo_tactile.yaml`，actor 输入为四路 tactile feature 和 `proprio_obs`，critic 仍使用 privileged state。
- 影响的观测：actor 不再读取 `third_resnet`；保留 `proprio_obs: 18` 和四路 tactile feature，默认每路 256 维。
- 影响的动作、奖励、done：无。仍为 5 维 `[dx, dy, dz, dyaw, gripper]`，奖励和终止逻辑沿用 `vt_box.py`。
- 影响的网络输入维度：actor 输入由 VT 的视觉+触觉+本体改为触觉压缩 latent + 本体；critic 输入不变。
- checkpoint 兼容性：现有 task/checkpoint 不受影响；`T` 新 task 的 actor 结构不同，需要重新训练，不能直接加载旧 VT actor checkpoint。
- 验证情况：
  - 已执行 `python -m py_compile source/tacex_tasks/tacex_tasks/occluded_grasping/vt_tactile_box.py source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py`。
  - 未运行 Isaac Sim 环境创建、训练或 `play_bucket.py` 评估。
- 待确认：
  - Isaac Sim 中 `TacEx-T-Drawer-Occlusion-Cube` 是否可完整创建并训练。
  - 触觉-only baseline 在无视觉输入下的收敛速度和成功率。

## 2026-07-04 CST — 新增 GelFusion 风格视触融合 RL 对比环境

- 类型：代码 / 配置 / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_box.py`
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_policy.py`
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_gelfusion.yaml`
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py`
  - `README.md`
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/README.md`
  - `docs/PROJECT_OVERVIEW.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DATA_FLOW.md`
  - `docs/EXPERIMENTS.md`
  - `docs/DEVLOG.md`
- 修改内容：
  - 新增 `GelFusion` 和 `GelFusion-Downsample` task 注册矩阵分支，可生成 `TacEx-GelFusion-Downsample-Drawer-Occlusion-Cube` 等 task id。
  - 新增 `OccludedGraspingVTGelFusionBoxEnv`，继承现有 VT 环境，保持动作、奖励、done/success 和 reset 语义不变。
  - actor 观测新增 `tactile_dynamic_stats`，维度为 8，表示四路触觉相邻帧二值差分的 `[mean, variance]`。
  - 新增 `OccludedGraspingVTGelFusionPolicy`，实现 vision-led cross attention：视觉特征作为 query，视觉和四路触觉静态特征作为 key/value，再拼接原始视觉、动态触觉统计和 `proprio_obs` 输出 5 维动作分布。
  - 新增 PPO 配置 `ppo_vt_gelfusion.yaml`，critic 仍使用原 privileged state。
- 影响的观测：新增分支的 policy observation 增加 `tactile_dynamic_stats: 8`；四路 tactile feature 默认使用 `tactile_encoder_type="resnet"`，每路仍为 256 维。
- 影响的动作、奖励、done：无。仍为 5 维 `[dx, dy, dz, dyaw, gripper]`，奖励和终止逻辑沿用 `vt_box.py`。
- 影响的网络输入维度：仅 GelFusion actor 变化；critic 输入不变。
- checkpoint 兼容性：现有 task/checkpoint 不受影响；GelFusion 新 task 需要重新训练，不能直接加载旧 VT actor checkpoint。
- 验证情况：
  - 已执行 `python -m py_compile source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_box.py source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_policy.py source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py`。
  - 已执行基于文件路径导入的 `quick_gelfusion_policy_smoke_test()`，输出 `mean shape=(4, 5)`。
  - 曾尝试通过包路径导入 smoke test，但当前非 Isaac Sim Python 环境缺少 `omni`，触发 `ModuleNotFoundError: No module named 'omni'`；随后改用文件路径导入完成 policy 测试。
  - 未运行 Isaac Sim 环境创建、训练或 `play_bucket.py` 评估。
- 待确认：
  - Isaac Sim 中新 task 是否可完整创建并训练。
  - ResNet tactile encoder 相比默认 CNN 的速度和显存开销是否可接受。
  - GelFusion 分支成功率需通过训练和 `play_bucket.py` 固定网格评估确认。

## 2026-06-28 CST — 将软物体材料调为更适合抓取的橡胶块参数

- 类型：配置 / 物理参数 / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_box.py`
  - `docs/DEVLOG.md`
- 修改内容：
  - 新增显式软物体材料密度 `SOFT_OBJECT_DENSITY = 300.0`，避免继续依赖 PhysX 默认密度。
  - 将 `SOFT_OBJECT_YOUNGS_MODULUS` 从 `5.0e5` 提高到 `1.0e7`，减少类似果冻的大形变。
  - 将 `SOFT_OBJECT_POISSONS_RATIO` 调整为 `0.35`，降低接近不可压材料时的侧向鼓胀趋势。
  - 将 `SOFT_OBJECT_DYNAMIC_FRICTION` 设为 `2.0`，避免继续使用过高摩擦值掩盖材料刚度问题。
  - 新增 `SOFT_OBJECT_ELASTICITY_DAMPING = 0.03` 并保留合法的 `SOFT_OBJECT_DAMPING_SCALE = 1.0`，抑制软体回弹和晃动。
  - 将 `SOFT_OBJECT_SOLVER_POSITION_ITERATIONS` 从 `16` 提高到 `32`，增强软体接触/形变求解稳定性。
- 影响范围：
  - SoftCylinder、SoftCube、SoftCuboid 三类软物体任务共用 `_make_soft_material()` 和 `_make_soft_props()`，都会使用该参数组。
- 影响的观测、动作、奖励、done：无代码语义变化；只改变软物体物理材料和 deformable solver 参数。
- 影响的网络输入维度：无。
- checkpoint 兼容性：模型结构和 checkpoint 加载不受影响；物理参数改变会影响 rollout 分布和评估/训练结果。
- 验证情况：
  - 已执行 `python -m py_compile source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_box.py`。
  - 未运行 Isaac Sim GUI 实测。
- 待确认：
  - Isaac Sim 中该参数组是否足以从“果冻感”转为可夹持的橡胶块行为。
  - `density=300.0` 与当前物体尺寸下的质量是否符合任务期望；如仍难以抬起，可进一步下调密度或提高夹爪侧摩擦。

## 2026-07-11 CST — 将现实对齐目标方块改为实测 5 cm

- 类型：环境配置 / 奖励阈值 / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_real_alignment_env.py`
  - `README.md`
  - `docs/PROJECT_OVERVIEW.md`
  - `docs/DATA_FLOW.md`
  - `docs/EXPERIMENTS.md`
  - `docs/DEVLOG.md`
- 修改内容：
  - 现实对齐 task 使用独立 `0.05x0.05x0.05 m` Cuboid、`cube_half_xy_extent=0.025 m` 和相匹配的 inherited action gate radius。
  - 保持 `lift_reward_start_delta=0.005 m` 过滤接触抖动；将现实对齐 task 的 `success_lift_delta` 从 0.040 m 调整为 0.035 m。
  - 原 `TacEx-Sim2Real-Cube-Grasp-v0` 继续使用 6 cm 方块和 40 mm success delta，不受影响。
- 影响的观测、动作、网络输入维度：无。
- 影响的奖励/done：仅现实对齐 task 在相对抬升 35 mm 时达到完整 lift 和 success height 条件，仍需 upright 且 hold 5 steps。
- checkpoint 兼容性：网络 shape 兼容，但物体几何和 success 阈值改变；当前训练必须重启并建议新开 run。
- 验证情况：
  - Isaac Sim headless 确认 cfg 和实际 spawn size 均为 `(0.05,0.05,0.05) m`。
  - 静止 smoke：`center_z≈0.0360 m`、`lowest_z≈0.0110 m`、`lift_delta≈0.0010 m`、`lift=0`、`success=0`。
  - 边界 smoke：delta `[0,0.020,0.035] m` 对应 lift `[0,0.5,1]`、success `[False,False,True]`。
  - 5 cm 仿真方块 bbox 为 `(82,123)-(92,136)`，现实参考为 `(80,121)-(92,136)`。
- 待确认：5 mm/35 mm 阈值的最终 PPO 训练和真机成功率。

## 2026-07-11 CST — 修复 sim2real Cube 静止状态误获 lift reward

- 类型：奖励修复 / 环境 / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_grasp_env.py`
  - `docs/DATA_FLOW.md`
  - `docs/EXPERIMENTS.md`
  - `docs/DEVLOG.md`
- 修改内容：
  - 将 Cube lift reward 和 success 从绝对 world lowest-z 阈值改为相对每个 env reset lowest height 的抬升增量。
  - 新增 `lift_reward_start_delta=0.005 m` 和 `success_lift_delta=0.040 m`。
  - 删除 lift 计算中的 `base_reward=1` 跳变；lift progress 现在严格限制在 `[0,1]`。
  - reward 与 done/success hold 共用 `_compute_cube_lift_terms()`，避免阈值语义漂移。
  - 增加 `info/cube_lift_delta` 日志和 reward print 字段。
- 影响的观测、动作：无；仍为原 observation dict 和 4 维 `[dx,dy,dz,gripper]`。
- 影响的奖励：方块静止在台面时 lift 从原约 `1.093` 修正为 0；抬升 5 mm 后开始线性奖励，40 mm 时 lift=1 并满足 success height 条件。
- 影响的 done/success：success hold 改用相对 reset 抬升 40 mm；timeout 和 ground collision 不变。
- 影响的网络输入维度：无。
- checkpoint 兼容性：模型参数和 shape 兼容；reward/done 语义改变，已有 checkpoint 的重新评估指标会变化，继续训练建议新开 run。
- 验证情况：
  - 已执行 `python -m py_compile` 和 `git diff --check`。
  - 已用 Isaac Sim 4.5 headless 创建 `TacEx-Sim2Real-Cube-Real-Alignment-v0`，reset 后连续执行 4 个零动作。
  - smoke 结果：`lift_delta=0.00099995`、`lift=0`、`success=0`、`total=0.022164`、`terminated=False`、`truncated=False`。
  - lift terms 边界 smoke：delta `[0,0.0225,0.040] m` 分别得到 lift `[0,0.5,1]` 和 success `[False,False,True]`。
- 待确认：
  - 5 mm lift start 和 40 mm success delta 的最终训练效果需通过新 PPO run 确认。
  - reach shaping 是否需要在 lift 修复后单独调大 `reach_sigma`，待观察训练曲线后决定。

## 2026-07-11 CST — 新增现实参考对齐 Cube 强化学习环境

- 类型：环境 / 配置 / sim2real / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_real_alignment_env.py`
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/agents/skrl_ppo_cube_real_alignment_cfg_resnet18.yaml`
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/__init__.py`
  - `scripts/reinforcement_learning/skrl/export_sim2real_grasp_jit.py`
  - `README.md`
  - `docs/PROJECT_OVERVIEW.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DATA_FLOW.md`
  - `docs/EXPERIMENTS.md`
  - `docs/DEVLOG.md`
- 修改内容：
  - 注册 `TacEx-Sim2Real-Cube-Real-Alignment-v0`，使用独立 PPO 配置和 `sim2real_cube_real_alignment` 日志目录。
  - 从 `20260711_214450_real_alignment_reference/alignment_reference.json` 对齐真实 Franka 七关节初态、夹爪开口、D435 crop 后内参、RGB 输入格式和 30 Hz 图像周期。
  - 将 policy decimation 设为 2，使 60 Hz physics 下 policy/control dt 和 camera update period 都为 1/30 s。
  - 新增深色台面和绿色背景近似，台面后沿与方块 nominal XY offset 根据 `rgb_model_median.png` 投影估计；保留小范围视觉和物体位置随机化。
  - 根据现实画面更平视的构图，将相机从原约 60° 俯拍改为 world `pos=(1.90,0.0,0.468)`、光轴向下约 15°；同步恢复台面后沿到 robot base 附近并扩大台面前向范围。
  - 将新 task 纳入 sim2real TorchScript exporter 支持列表。
- 影响的观测：key 和维度保持 `wrist_resnet:512`、`proprio_obs:15`、`action_history:4` 以及原 `critic_*` keys；视觉内容、FOV 和更新周期改变。
- 影响的动作：仍为 4 维 `[dx,dy,dz,gripper]`，`action_scale=0.05` 和 IK 逻辑不变；policy/control frequency 从原 Cube task 的 60 Hz 改为新 task 的 30 Hz。
- 影响的奖励、done：无，继承 `Sim2RealCubeGraspEnv` 的 reach/lift/success 和 timeout/ground collision/sustained success 语义。
- 影响的网络输入维度：无。
- checkpoint 兼容性：网络结构兼容旧 Cube checkpoint，但视觉分布和时间尺度不同，不视为性能兼容；建议新 task 重新训练。
- 验证情况：
  - 已执行 Python `py_compile` 和 YAML parse。
  - 已用 Isaac Sim 4.5 headless 创建 task、reset 并执行 4 次零动作；观测 shape、4 维动作、30 Hz control/camera period、真实关节初态均符合配置。
  - 已初始化 skrl `Runner`，新 PPO Actor 对环境 observation 成功输出 finite `(1,4)` action。
  - 已执行 `python -m compileall source scripts tools` 和 `git diff --check`，通过。
  - seed 42 最终仿真帧 mean RGB 为 `[70.6,113.0,70.1]`；现实 `rgb_model_median.png` mean RGB 为 `[80.1,113.0,81.9]`。方块亮区中心分别约 `(86.6,128.6)` 与 `(85.6,128.5)` pixel。
- 待确认：
  - reference 明确未测量 D435-to-Franka 外参，当前相机位姿只是既有人工近似值。
  - 真实方块世界坐标、尺寸、质量、摩擦和台面精确几何没有记录；当前物体/台面几何部分来自既有 task 和参考图估计。
  - 尚未运行 PPO 训练、checkpoint 评估或真机闭环验证。

## 2026-07-11 CST — 导出 Cube sim-to-real TorchScript policy

- 类型：代码 / 部署工具 / checkpoint artifact / 文档
- 修改文件：
  - `scripts/reinforcement_learning/skrl/export_sim2real_grasp_jit.py`
  - `scripts/reinforcement_learning/skrl/test_sim2real_grasp_policy_jit.py`
  - `README.md`
  - `docs/DEVLOG.md`
  - `logs/skrl/sim2real_cube_grasp/2026-06-26_23-14-35_ppo_torch_vision_only_resnet18/checkpoints/exported/policy_actor_e2e_best_agent.pt`
  - `logs/skrl/sim2real_cube_grasp/2026-06-26_23-14-35_ppo_torch_vision_only_resnet18/checkpoints/exported/policy_actor_e2e_best_agent.json`
- 修改内容：
  - 将 `TacEx-Sim2Real-Cube-Grasp-v0` 加入现有 sim-to-real TorchScript 导出器的支持范围，保留原 cylinder task 支持。
  - 为 sidecar JSON 增加 4 维动作语义、训练环境 nominal policy frequency 和 action scale。
  - 从指定 `best_agent.pt` 导出包含冻结 ResNet18 和 actor 的端到端 CPU TorchScript 模型。
  - 修复 smoke test 加载 Pillow 图像时 NumPy 数组只读导致的 PyTorch warning。
- 影响的观测：不改变环境观测；导出模型接受 `action_history [N, 4]`、`proprio_obs [N, 15]` 和 `wrist_rgb [N, 224, 224, 3]` uint8。Cube task 中该 key 实际来自固定第三视角相机。
- 影响的动作：不改变环境动作空间；模型输出 raw actor mean `[dx, dy, dz, gripper]`，真机控制器仍需实现 `action_scale=0.05`、IK 和安全约束。
- 影响的奖励、done：无。
- 影响的网络输入维度：无；只将已有 checkpoint 固化为 TorchScript。
- checkpoint 兼容性：指定 checkpoint 已成功加载和导出，无需重新训练。
- 验证情况：
  - 已执行 `python -m py_compile scripts/reinforcement_learning/skrl/export_sim2real_grasp_jit.py`，通过。
  - 已执行 `conda run -n isaaclab_2.1.1 --no-capture-output python scripts/reinforcement_learning/skrl/export_sim2real_grasp_jit.py --task TacEx-Sim2Real-Cube-Grasp-v0 --checkpoint logs/skrl/sim2real_cube_grasp/2026-06-26_23-14-35_ppo_torch_vision_only_resnet18/checkpoints/best_agent.pt --num_envs 1 --headless`，退出码 0；trace max absolute error 为 0。
  - 已用 synthetic RGB 和训练 run 中保存的 224x224 相机帧分别执行 `test_sim2real_grasp_policy_jit.py`；输出 shape 均为 `(1, 4)`、数值有限、同输入重复误差为 0，本机 CPU 平均推理耗时在重复测试中约 6.7–12.8 ms。
- 待确认：
  - 尚未连接真实 Franka、真实相机或真机安全控制器；TorchScript 成功不等同于真实抓取成功。
  - 真机相机的内参、外参、曝光、颜色分布与仿真相机的一致程度待标定。
  - 当前 checkpoint 的真实成功率和闭环稳定性待低速、受限工作空间条件下验证。

## 2026-06-28 CST — 修正软体 damping scale 合法范围

- 类型：修复 / 配置 / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_box.py`
  - `docs/DEVLOG.md`
- 修改内容：
  - 将 `SOFT_OBJECT_DAMPING_SCALE` 从 `2.0` 改为 `1.0`。
  - 依据 Isaac Sim/PhysX 运行时报错：`PxMaterial::setDampingScale` 要求值在 `[0.0, 1.0]` 范围内。
- 影响的观测、动作、奖励、done：无。
- 影响的网络输入维度：无。
- checkpoint 兼容性：模型结构和 checkpoint 加载不受影响；软体材料阻尼参数合法化会影响 rollout 物理表现。
- 验证情况：
  - 已执行 `python -m py_compile source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_box.py`。
  - 未运行 Isaac Sim GUI 复测。
- 待确认：
  - Isaac Sim 重新启动后是否不再出现 `setDampingScale` invalid float 错误。

## 2026-06-28 CST — 增加软物体动态摩擦配置

- 类型：配置 / 物理参数 / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_box.py`
  - `docs/DEVLOG.md`
- 修改内容：
  - 新增 `SOFT_OBJECT_DYNAMIC_FRICTION = 1.5`。
  - `_make_soft_material()` 创建 `DeformableBodyMaterialCfg` 时显式传入 `dynamic_friction`，避免继续使用 Isaac Lab deformable material 默认值 `0.25`。
  - 该材料函数被 SoftCylinder、SoftCube、SoftCuboid 共用，因此三类软物体任务都会使用新的动态摩擦系数。
- 影响的观测、动作、奖励、done：无代码语义变化；只改变软物体接触物理材料。
- 影响的网络输入维度：无。
- checkpoint 兼容性：模型结构和 checkpoint 加载不受影响；物理参数改变会影响 rollout 分布和评估/训练结果。
- 验证情况：
  - 已执行 `python -m py_compile source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_box.py`。
  - 未运行 Isaac Sim GUI 实测。
- 待确认：
  - Isaac Sim 中新的 `dynamic_friction=1.5` 是否足以减少夹爪与软体之间的滑移。
  - 较高摩擦在大形变接触下是否会引入粘滞、抖动或求解不稳定。

## 2026-06-27 CST — 新增 UR10 + Robotiq 2F85 键盘奖励调试脚本

- 类型：代码 / 调试工具 / 文档
- 修改文件：
  - `scripts/ur10_robotiq/teleop_2f85_rewards.py`
  - `README.md`
  - `docs/DEVLOG.md`
- 修改内容：
  - 新增键盘 teleop 脚本，直接实例化 `UR10Robotiq2F85ThirdPersonPickPlaceEnvCfg` 和 `UR10Robotiq2F85ThirdPersonPickPlaceEnv`，保持当前环境的 UR、物体和木板位置配置。
  - 键盘动作映射到环境原始 4 维 action `[dx, dy, dz, gripper]`。
  - 周期打印 `reward/total`、`reward/reach`、`reward/lift`、`reward/success`、夹爪到物体距离、抬升量、夹爪内侧 link 最低高度、桌面高度、clearance、夹爪中心位置和物体位置。
- 影响的观测、奖励、动作、done：不修改环境本身；脚本复用当前环境奖励公式做诊断打印。
- checkpoint 兼容性：无影响。
- 验证情况：
  - 已执行 `python -m py_compile scripts/ur10_robotiq/teleop_2f85_rewards.py`。
  - 未运行 Isaac Sim GUI teleop 实测。
- 待确认：
  - 当前 Isaac Sim 运行时 `KeyboardInput.SPACE` 在本机是否可用于 stop 按键。

## 2026-06-27 CST — 新增 UR10 + Robotiq 2F85 third-person camera pick-place 变体

- 类型：代码 / 环境注册 / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace/ur10_robotiq_2f85_third_person_pick_place_env.py`
  - `source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace/__init__.py`
  - `README.md`
  - `docs/PROJECT_OVERVIEW.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DEVLOG.md`
- 修改内容：
  - 新增独立 `UR10Robotiq2F85ThirdPersonPickPlaceEnvCfg` 和 `UR10Robotiq2F85ThirdPersonPickPlaceEnv`，两者分别直接继承 `DirectRLEnvCfg` 和 `DirectRLEnv`，不继承其他 UR10 + Robotiq 环境或 cfg 类。
  - 在 UR10 + Robotiq 2F85 pick-place scene 中额外注册 224x224 RGB `third_person_camera`。
  - 相机内参、位姿和 OpenGL convention 参考 `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_grasp_env.py:Sim2RealCubeGraspEnvCfg.wrist_camera`。
  - 注册 task id：`Isaac-UR10-Robotiq-2F85-Third-Person-Pick-Place-Direct-v0`。
- 影响的观测：新增相机 sensor buffer；默认 policy observation 仍是原 27 维 state vector，未把 RGB 加入 `policy` 观测。
- 影响的动作、奖励、done：保持与原 UR10 + Robotiq 2F85 pick-place 当前逻辑一致，但代码已复制到独立文件中，后续可单独改参数。
- checkpoint 兼容性：状态观测和动作维度不变，现有 state-based PPO 模型结构层面兼容；运行 third-person 变体需要 `--enable_cameras` 才能渲染相机。
- 验证情况：
  - 已执行 `python -m py_compile source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace/ur10_robotiq_2f85_third_person_pick_place_env.py source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace/__init__.py`。
  - 已执行 `rg -n "Third-Person|third_person_camera|UR10Robotiq2F85ThirdPerson" source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace README.md docs/PROJECT_OVERVIEW.md docs/ARCHITECTURE.md docs/DEVLOG.md`。
  - 已执行 `rg -n "UR10RobotiqPickPlaceEnv|UR10Robotiq2F85PickPlaceEnvCfg|继承同一|继承原" source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace/ur10_robotiq_2f85_third_person_pick_place_env.py README.md docs/PROJECT_OVERVIEW.md docs/ARCHITECTURE.md docs/DEVLOG.md` 检查继承残留描述。
  - 已执行 `find source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace -type d -name __pycache__ -print` 并清理本次编译生成的 `__pycache__` 目录。
  - 未运行 Isaac Sim 环境创建、训练或相机画面检查。
- 待确认：
  - Isaac Sim 中 `third_person_camera` 实际画面是否完整覆盖 UR10、Robotiq 2F85、桌面和方块。
  - 是否需要进一步新增真正读取 RGB 的视觉策略 YAML 和 image encoder。

## 2026-06-27 CST — 兼容 UR10 direct task 的 Box observation space

- 类型：代码 / 训练入口 / 兼容性
- 修改文件：
  - `scripts/reinforcement_learning/skrl/train.py`
  - `docs/DEVLOG.md`
- 修改内容：
  - 将训练入口中 recurrent tactile policy 自动检测的 observation key 提取逻辑改为同时支持 Gymnasium `Dict`/dict 和普通 `Box` observation space。
  - 对 UR10 + Robotiq direct task 这类 `Box` 状态观测环境，`obs_keys` 为空集合，不再对 `Box` 执行 `set(Box)`。
- 影响的观测：不改变任何环境观测内容或维度；只改变训练脚本对 observation space 类型的分支判断。
- 影响的动作、奖励、done：无。
- checkpoint 兼容性：无模型结构变化。
- 验证情况：
  - 已执行 `python -m py_compile scripts/reinforcement_learning/skrl/train.py`。
  - 未重新运行 Isaac Sim 训练。
- 待确认：
  - `Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0` 完整训练启动是否继续通过后续阶段。

## 2026-06-27 CST — 迁移 UR10 + Robotiq direct RL 任务

- 类型：代码 / 配置 / 资产 / checkpoint artifact / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/direct/__init__.py`
  - `source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace/*`
  - `source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_gripper_close/*`
  - `source/tacex_assets/tacex_assets/data/Robots/URRobotiq/*`
  - `logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/*`
  - `README.md`
  - `docs/PROJECT_OVERVIEW.md`
  - `docs/ARCHITECTURE.md`
  - `docs/EXPERIMENTS.md`
  - `docs/DEVLOG.md`
- 修改内容：
  - 从 `/home/tinydog/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/` 迁移 UR10 + Robotiq pick-place/grasp 和 gripper-close direct RL task 包。
  - 注册 task id：`Isaac-UR10-Robotiq-Pick-Place-Direct-v0`、`Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0`、`Isaac-UR10-Robotiq-2F85-Grasp-Direct-v0`、`Isaac-UR10-Robotiq-Gripper-Close-Direct-v0`。
  - 迁移 UR10 + Robotiq USD 到 `source/tacex_assets/tacex_assets/data/Robots/URRobotiq/`，并将默认 asset path 改为仓库内路径，同时保留环境变量覆盖。
  - 补齐 `ur10_robotiq_2f85.usda` 的相对 payload 依赖目录：`UniversalRobots/ur10/` 和 `Robotiq/2F-85/`。
  - 迁移一个已确认存在的 skrl checkpoint run：`logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/`。
  - 删除迁移代码和 checkpoint `params/env.yaml` 中 Isaac Lab 2.1.1 不支持的 `InteractiveSceneCfg.clone_in_fabric` 字段。
- 影响的观测：新增 UR10 + Robotiq task 使用 27 维状态观测；不影响 `occluded_grasping` 的观测 key 或维度。
- 影响的动作：新增 UR10 + Robotiq 2F85 pick-place 使用 4 维动作 `[dx, dy, dz, gripper]`；Robotiq 2F85 策略只直接控制 `finger_joint`，其他指关节由 coupling rules 生成目标。
- 影响的奖励和 done：仅新增 UR10 + Robotiq task 内部 reward/done；不影响现有 TacEx 遮挡抓取任务。
- checkpoint 兼容性：迁移的 `best_agent.pt` 对应 27 维 observation 和 4 维 action 的 skrl PPO 模型；是否能在当前 TacEx wrapper 下直接 play 待 Isaac Sim 验证。
- 验证情况：
  - 已静态确认迁移源文件、USD、checkpoint 和 params 文件存在。
  - 已执行 `test -f source/tacex_assets/tacex_assets/data/Robots/URRobotiq/UniversalRobots/ur10/ur10.usd && test -f source/tacex_assets/tacex_assets/data/Robots/URRobotiq/Robotiq/2F-85/Robotiq_2F_85_attachable.usda`，确认组合 USD 的两个缺失 payload 文件已存在。
  - 已执行 `python -m py_compile source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace/ur10_robotiq_pick_place_env.py source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_gripper_close/ur10_robotiq_gripper_close_env.py source/tacex_tasks/tacex_tasks/direct/ur10_robotiq_pickplace/ur10_robotiq_2f85_pick_place_env_cfg.py`。
  - 已执行 `rg -n "clone_in_fabric" source/tacex_tasks/tacex_tasks/direct logs/skrl/ur10_robotiq_pick_place_direct README.md docs`，确认无残留。
  - 未运行 Isaac Sim 训练、回放或环境创建。
- 待确认：
  - 当前 TacEx 环境中 `gym.make("Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0")` 是否可成功创建。
  - 迁移 checkpoint 在当前 `scripts/reinforcement_learning/skrl/play.py` 下是否可直接加载。

## 2026-06-26 22:24 CST — 建立项目文档维护机制

- 类型：文档 / 维护机制
- 修改文件：`AGENTS.md`、`docs/PROJECT_OVERVIEW.md`、`docs/ARCHITECTURE.md`、`docs/DATA_FLOW.md`、`docs/DEVLOG.md`、`docs/EXPERIMENTS.md`、`docs/DECISIONS.md`、`docs/KNOWN_ISSUES.md`
- 修改内容：
  - 建立统一项目地图、工作前检查步骤、验证规则和结束输出格式。
  - 根据实际代码补充训练入口、评估入口、环境、传感器、策略、奖励、done/success 和日志路径。
  - 新增实验索引模板和已确认 checkpoint/metrics artifact 记录。
  - 新增设计决策和已知问题条目。
- 依据：
  - `scripts/reinforcement_learning/skrl/train.py:main`
  - `scripts/reinforcement_learning/skrl/play.py:main`
  - `scripts/reinforcement_learning/skrl/play_bucket.py:main`
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxEnv`
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_scene_object_fusion_matrix`
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml`
  - `logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/`
- 接口与维度变化：无。仅文档变更。
- 业务代码变化：无。
- 验证情况：
  - 已读取并整理仓库代码、配置、日志和 Git 状态。
  - 已执行 `date '+%Y-%m-%d %H:%M %Z'` 记录文档时间。
  - 已执行 `git status --short` 确认当前工作区存在既有未提交业务改动。
  - 已执行 `git ls-files --others --exclude-standard AGENTS.md docs/PROJECT_OVERVIEW.md docs/ARCHITECTURE.md docs/DATA_FLOW.md docs/DEVLOG.md docs/EXPERIMENTS.md docs/DECISIONS.md docs/KNOWN_ISSUES.md` 确认本次允许文档处于未跟踪/待提交状态。
  - 已执行 `rg` 检查关键章节和“待填写”等占位符。
  - 已执行 `test -f` 确认训练入口、评估入口、核心环境、Alpha-GRU YAML 和示例 checkpoint 路径存在。
  - 未运行 Isaac Sim 训练、评估或测试。

## 历史状态恢复说明

### 2026-06-26 23:10 CST — 保存 sim2real cube 训练起始相机帧

- 类型：代码 / 训练入口 / 诊断
- 修改文件：
  - `scripts/reinforcement_learning/skrl/train.py`
  - `docs/DEVLOG.md`
- 修改内容：
  - 将已有 `--save_start_frame` 逻辑从单张图扩展为多帧保存。
  - 新增 `--start_frame_count`，默认保存 5 帧。
  - `TacEx-Sim2Real-Cube-Grasp-v0` 训练时自动保存训练开始后的相机帧。
  - 输出目录为本次 skrl run 日志目录下的 `camera_frames/`。
  - 文件名包含 sensor、env index、frame index 和实际图片尺寸，例如 `start_wrist_camera_env0_frame_000_224x224.png`。
- 依据：
  - 用户希望运行 `TacEx-Sim2Real-Cube-Grasp-v0` 时保存几帧 `224x224` 实际相机画面，用于确认方块、夹爪和有效抓取区域是否完整覆盖。
  - skrl 训练日志目录由 `scripts/reinforcement_learning/skrl/train.py:main` 中 `log_root_path` 和 `log_dir` 生成。
- 影响的观测：无。只从 camera sensor buffer 读取 RGB 并保存 env0 图像。
- 影响的动作：无。
- 影响的奖励：无。
- 影响的 done/success：无。
- 影响的网络输入维度：无。
- checkpoint 兼容性：无结构影响。
- 验证情况：
  - 已执行 `python -m py_compile scripts/reinforcement_learning/skrl/train.py source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_grasp_env.py source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_grasp_env.py`。
  - 未运行 Isaac Sim 环境创建、训练或评估。
- 待确认：
  - Isaac Sim 运行时实际保存图像是否满足画面覆盖检查。

### 2026-06-26 23:06 CST — 调整 sim2real cube 相机分辨率

- 类型：代码 / 配置
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_grasp_env.py`
  - `docs/DEVLOG.md`
- 修改内容：
  - 在 `Sim2RealCubeGraspEnvCfg` 中显式覆盖 `wrist_camera`，将 `TiledCameraCfg.height` 和 `TiledCameraCfg.width` 从继承的 `480x640` 改为 `224x224`。
  - 将 `PinholeCameraCfg.from_intrinsic_matrix` 的 `width` 和 `height` 同步改为 `224x224`，并按原 `640x480` 配置比例缩放内参。
- 依据：
  - 用户要求 `TacEx-Sim2Real-Cube-Grasp-v0` 环境相机分辨率从 `640x480` 改为 `224x224`。
  - 原始相机配置来自 `Sim2RealGraspEnvCfg.wrist_camera`。
- 影响的观测：
  - 相机原始 RGB 渲染尺寸变为 `224x224`。
  - `wrist_resnet` 仍为 512 维。
- 影响的动作：无。
- 影响的奖励：无。
- 影响的 done/success：无。
- 影响的网络输入维度：policy/value 输入 key 和维度不变；视觉编码前的图像尺寸与 ResNet 输入尺寸一致。
- checkpoint 兼容性：观测 key 和特征维度不变，模型结构层面不受影响；但视觉输入分布改变，旧 checkpoint 表现需要重新评估。
- 验证情况：
  - 已执行 `python -m py_compile source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_grasp_env.py source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_grasp_env.py`。
  - 未运行 Isaac Sim 环境创建、训练或评估。
- 待确认：
  - Isaac Sim 中 224x224 相机画面是否仍覆盖完整抓取区域。

### 2026-06-26 22:50 CST — 新增 sim2real cube grasp 变体

- 类型：代码 / 环境注册
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_grasp_env.py`
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_grasp_env.py`
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/__init__.py`
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/agents/skrl_ppo_cube_vision_only_cfg_resnet18.yaml`
- 修改内容：
  - 新增独立 `sim2real_cube_grasp_env.py`，不再把 cube 变体放在 bottle/cylinder 环境文件尾部。
  - 新增 `Sim2RealCubeGraspEnvCfg`，复用 `Sim2RealGraspEnvCfg` 的 Franka、固定第三视角相机和视觉随机化，将目标物体定义为 `sim_utils.CuboidCfg`。
  - 新增 `Sim2RealCubeGraspEnv`，移除 bottle cap visual，对外观测使用 `critic_cube_*` key，奖励日志使用 `cube_*` key。
  - 新增 `skrl_ppo_cube_vision_only_cfg_resnet18.yaml`，value network 使用 `critic_cube_*`。
  - 注册 task id：`TacEx-Sim2Real-Cube-Grasp-v0`。
- 影响的观测：保持 `wrist_resnet:512`、`proprio_obs:15`、`action_history:4`，critic keys 改为 `critic_cube_pos`、`critic_cube_quat`、`critic_cube_lin_vel`、`critic_cube_ang_vel`。
- 影响的动作：保持 4 维动作空间不变。
- 影响的奖励：cube 变体重写最低点高度计算，使用 cube 8 个角点的最低世界 z 参与 lift/success；reach/lift/success 权重沿用 sim2real grasp。
- 影响的 done/success：沿用 sim2real grasp 的 timeout、ground collision 和 sustained upright lift success。
- 影响的网络输入维度：policy 输入维度不变；value network key 名从 `critic_cylinder_*` 改为 `critic_cube_*`，总维度不变。
- checkpoint 兼容性：新干净 task 使用独立 YAML 和 `critic_cube_*` key，旧 `TacEx-Sim2Real-Grasp-v0` bottle checkpoint 不作为直接兼容目标；需要为 cube 单独训练或另做 legacy 兼容评估入口。
- 验证情况：
  - 已执行 `python -m py_compile source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_grasp_env.py source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_grasp_env.py source/tacex_tasks/tacex_tasks/sim2real_grasp/__init__.py`。
  - 已执行 `rg` 确认 `Sim2RealCubeGraspEnvCfg`、`Sim2RealCubeGraspEnv` 和 `TacEx-Sim2Real-Cube-Grasp-v0` 存在。
  - 未运行 Isaac Sim 环境创建、训练或评估。
- 待确认：
  - cube 变体实际物理稳定性。
  - cube reward 阈值是否需要进一步调参。
  - 是否需要为 cube 单独训练新 checkpoint。

### 2026-06-26 — 用户确认项目主线

- 状态：已确认
- 确认内容：当前项目主线是 `occluded_grasping` 遮挡抓取实验。
- 影响文档：`AGENTS.md`、`docs/PROJECT_OVERVIEW.md`
- 仍待确认：`source/tacex_uipc` 是否属于当前主线必需依赖；通用 TacEx sensor framework 是否继续作为同仓库基础设施维护。

### HEAD：`48e8f0a add some pipeline`

- 状态：已提交历史
- 分支：`dev/0517`
- 依据：`git log --oneline --decorate -n 30`、`git show --stat --oneline HEAD`
- 已确认内容：该提交大规模调整 `occluded_grasping` pipeline，统计为 53 files changed, 6169 insertions, 1447 deletions。
- 影响范围：新增/调整多种 policy/env/YAML 变体，并删除部分 residual/mixed sensor gate 相关文件。
- 具体设计意图：待确认。

### 当前未提交工作区

- 状态：待确认
- 依据：`git status --short`
- 已确认内容：工作区存在多个已修改、已删除和新增文件，主要集中在 `source/tacex_tasks/tacex_tasks/occluded_grasping`、`scripts/reinforcement_learning/skrl/play.py` 和 `scripts/occluded_grasping/*`。
- 处理规则：后续任务不得回滚这些改动；若修改同一文件，必须先阅读并确认与当前任务相关。

## 2026-06-28 — 将软物体触觉检查脚本改为键盘遥操作

- 类型：实验脚本
- 修改文件：
  - `scripts/occluded_grasping/check_softcube_franka_tactile.py`
- 修改内容：
  - 将原固定 approach/press/close/lift 轨迹改为 GUI 键盘控制。
  - 默认 task 改为 `TacEx-Alpha-GRU-Self-Occlusion-SoftCube`，用于去掉 drawer/cabinet 几何遮挡，仅保留自遮挡场景。
  - 键盘动作映射到遮挡抓取环境 5 维 action：`[dx, dy, dz, dyaw, gripper]`。
  - 保留软体节点形变指标 CSV 记录、周期打印和按键保存内侧 GelSight 触觉 PNG。
  - `--steps 0` 表示持续运行直到 ESC 或窗口关闭；`--save_every 0` 表示只手动保存。
  - 移除运行中的 `R` reset 快捷键；默认覆盖当前测试实例的 `_get_dones()`，阻止 timeout/ground-collision done 触发 `env.step()` 自动 reset。
  - 保留启动后的第一次 `env.reset()`，这是 Gym/Isaac Lab 环境初始化所需，不是交互过程中的重置。
- 影响的观测：无；不改环境观测 key 或张量维度。
- 影响的动作：脚本侧 action 来源从固定控制器改为键盘输入；环境动作空间仍为 5 维。
- 影响的奖励：无。
- 影响的 done/success：仅此脚本运行时默认覆盖 `_get_dones()` 并关闭 done reset；环境源码 done/success 语义不变，可用 `--allow_done_resets` 恢复默认行为。
- 影响的网络输入维度：无。
- checkpoint 兼容性：不涉及 checkpoint 加载或模型结构。
- 验证情况：
  - 已执行 `python -m py_compile scripts/occluded_grasping/check_softcube_franka_tactile.py`。
  - 未运行 Isaac Sim GUI 实测。
- 待确认：
  - Isaac Sim GUI 中键盘事件订阅和 SoftCylinder/SoftCube/SoftCuboid 三类任务的实际交互稳定性。
  - 禁用 done reset 后，极端碰撞/穿模状态下是否仍需人工重启仿真。

## 条目模板

```text
## YYYY-MM-DD HH:MM TZ — 标题

- 类型：代码 / 配置 / 文档 / 实验脚本 / 修复 / 重构
- 修改文件：
- 修改内容：
- 依据：
- 影响的观测：
- 影响的动作：
- 影响的奖励：
- 影响的 done/success：
- 影响的网络输入维度：
- checkpoint 兼容性：
- 验证情况：
- 待确认：
```
