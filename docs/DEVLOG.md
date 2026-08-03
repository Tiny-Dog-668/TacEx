# Development Log

本文件记录工程、代码、配置和文档维护变更。实验结果记录到 `docs/EXPERIMENTS.md`，设计取舍记录到 `docs/DECISIONS.md`，问题和风险记录到 `docs/KNOWN_ISSUES.md`。

## 记录规则

- 每条记录必须写明日期、修改范围、依据和验证情况。
- 不要把未运行的训练或评估写成已完成结果。
- 不确定的作者、意图、commit、seed、checkpoint 或结果统一写“待确认”。
- 涉及观测、动作、奖励、done、网络输入维度或 checkpoint 兼容性的改动必须明确说明。

## 2026-08-03 CST — 新增 RMA Student cube-center heatmap 辅助监督

- 类型：RMA Student 模型 / 训练脚本 / task 注册 / checkpoint 兼容 / 测试 / 文档
- 修改：保留现有 `SpatialSoftmaxAdaptationHead` 和 Student `forward(RGB, proprio, history)->action` 接口；从 ResNet18 layer3 `[N,256,14,14]` 新增 `Conv3x3+ReLU+Conv1x1` heatmap head，输出 `[N,1,14,14]`。新增 Clean/DR heatmap Student task：`TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-v0`、`TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-DR-v0`。
- 监督：用 `cube_position_root`、当前 Student 有效内参和相机 optical pose 投影 cube center，生成14x14 Gaussian GT heatmap；默认 `sigma=1.5` heatmap px。相机后方或224x224图像外样本通过 valid mask 排除 heatmap MSE。DR heatmap task 使用每环境随机化后的相机 pose、focal scale 和 principal shift。
- Backbone 微调：默认仍冻结全部 ResNet18；新增 `--train_backbone_after_layer2` 后只解冻 ResNet18 layer3/layer4，conv1、bn1、layer1、layer2 及 Actor Core 仍冻结。ResNet BatchNorm 保持 eval；backbone 使用独立 `--backbone_learning_rate`，默认 `3e-5`。
- 日志：训练脚本新增 `Loss/heatmap`、`Heatmap/center_error_px`、`Heatmap/valid_fraction`、`Position/mae_x/y/z_m`，终端进度行同步打印 heatmap loss、uv误差和xyz MAE。`--heatmap_debug_interval > 0` 时在 `heatmap_debug/` 保存少量 RGB overlay；默认关闭，不影响正常训练速度。
- 兼容性：旧 Student task 默认 heatmap supervision 关闭；新增 heatmap task 默认权重1.0。旧 Student checkpoint 可通过允许 missing `heatmap_head.*` 加载，play/evaluate/export 路径同步使用兼容加载；若 resume 时 optimizer/backbone 训练配置不同，则只加载模型权重并重新初始化 optimizer。TorchScript 仍只导出4维 action，不输出 heatmap。
- 验证情况：已执行针对修改文件的 `python -m compileall ...`，通过；已用直接文件导入的纯 PyTorch smoke 检查 Student forward、heatmap head 输出、投影/GT heatmap/soft-argmax 和 `torch.jit.script`，通过。系统 Python 直接导入 `tacex_tasks` 因未启动 Isaac/缺少 `omni` 失败，未执行完整 Isaac pytest、训练或评估。

## 2026-08-01 CST — DR 延后到 100k 并训练 300k

- 类型：DR 环境配置 / Agent 配置 / 训练监控 / 测试 / 文档
- 修改：仅将 Real-Alignment DR 的零扰动阶段延长到 100k，100k–220k 线性扩大到 full range，220k–300k 保持 full range；DR Agent trainer 从 200k 延长到 300k。Clean 环境、Clean Agent 及其 200k trainer 不变。
- 日志：DR 的 `_compute_additional_reward()` 保留继承的撞桌 penalty，并新增 `info/dr_curriculum_scale`，不改变 reward 数值。
- 依据：旧 200k DR run 的 TensorBoard 曲线在课程后段出现 reach/lift 回退，累计成功接近 0；新时间表先让继承的物体位置课程在 100k 完成，再启动视觉随机化。
- 影响：不修改物理、Actor/Critic observation key/shape、4-D action、reward、success、done 或 Clean task；DR env config hash 和训练时长改变，旧 DR checkpoint 不能作为当前 profile 的续训起点，需要从头训练。
- 验证情况：`git diff --check`、完整 `python -m compileall -q source scripts tools`、Clean/DR Agent YAML 解析和 4-env Real-Alignment Isaac 定向 pytest 12 项通过；其中 Clean trainer 明确保持 200k，DR trainer 为 300k，DR 课程边界和 `info/dr_curriculum_scale` 断言通过。未执行 300k 训练、正式评估或真机闭环。

## 2026-08-01 CST — DR 改为严格基于 Clean 的视觉课程

- 类型：环境配置 / GPU 视觉随机化 / 光照 / 测试 / 文档
- 修改：Real-Alignment DR 的课程初值从 `0.10` 改为 `0`，前 20k policy steps 与 Clean 视觉路径完全一致，20k–120k 再线性扩大到 full range。Plate/backdrop 的中心改为 Clean 的近黑材质，分别在 full scale 的每通道 `[0.01,0.05]` 和 `[0.005,0.03]` 内变化。
- 光照：DomeLight 以 Clean 的 intensity `2000`、color `(0.75,0.75,0.75)` 为中心；随机色温使用相对 5500 K 的 RGB tint 且关闭 renderer 二次色温处理，使 scale=0 精确恢复 Clean。
- 影响：不修改物理、4-D action、`action_history/proprio/wrist_resnet` shape、奖励、success 或 done；DR 视觉分布和 env config hash 改变，旧 DR checkpoint 不作为新课程的续训起点。Clean checkpoint 的网络 shape 和控制语义保持兼容。
- 验证情况：`git diff --check`、完整 `python -m compileall -q source scripts tools` 和 4-env Real-Alignment Isaac 定向 pytest 12 项通过；其中显式验证 scale=0 的相机/内参/后处理缓存、材质、光照及像素输出与 Clean 一致。标准 discovery 的 base Python 缺少 `isaacsim`；改用 `isaaclab_2.1.1` 后 warm start 成功，但仍被既有 ISSUE-015（缺失 `test_argparser_launch.py` skip target）阻断。未重新训练或执行真机闭环。

## 2026-07-31 CST — 增加最近 200 policy steps 的 episode 成功率

- 类型：训练监控 / play metrics / 测试 / 文档
- 修改：Real-Alignment Clean/DR/Privileged 在每次 `_get_dones()` 后记录该 policy step 完成和成功的 episode 数；长度 200 的 step 环形窗口统计 `sum(success)/sum(completed)`，并保留累计成功率。无完成 episode 的 step 也推进窗口，窗口内无完成 episode 时 rate 为 0。
- 日志：发布 `recent_success_rate`、`episode_success_rate_window`、`info/episode_success_rate_cumulative`、窗口及累计计数；每 200 步的现有 `[奖励]` 行追加窗口成功率、成功/完成分子分母和累计成功率。
- 影响：不修改 reward、done、观测/action shape、控制、相机或 policy contract；一个成功 episode 只在连续保持 5 步触发 success done 时统计一次。
- 验证情况：`git diff --check`、完整 `compileall`、4-env Clean/DR Isaac 定向 pytest 12 项和 2-env 无相机 Privileged pytest 1 项通过；窗口第 201 步淘汰第 1 步、累计统计不受窗口淘汰影响的边界断言通过。未执行重新训练或真机闭环。

## 2026-07-30 CST — 验证 Real-Alignment reach 奖励几何峰值

- 类型：诊断测试 / 实验记录
- 修改：新增 4-env Isaac 回归测试，将方块质心分别写入左右指尖中点、25 mm 偏移、50 mm 偏移和 `panda_hand` origin，并通过生产 `_get_rewards()` 隔离测量 reach 项；没有修改奖励实现、权重或中心定义。
- 结果：指尖中点与 `panda_hand+0.1034 m` TCP 误差最大约 `1.37e-7 m`；四个候选位置的 raw reach 分别为 `1.0000/0.7551/0.5379/0.2245`。reset 时夹爪中心到方块质心约 `0.274 m`，对应 raw reach 约 `0.0083`。
- 结论：当前 reach 峰值确实位于夹持几何中心，未发现 object COM 或 fingertip midpoint 坐标错误；训练初期 reach 很小由 `reach=1-tanh(distance/0.1)` 在长距离区域接近饱和导致。是否调整 `reach_sigma` 或引入分阶段 shaping 尚未决定。
- 验证情况：`git diff --check`、完整 `compileall` 和 4-env Real-Alignment Isaac 定向 pytest 11 项通过。

## 2026-07-30 CST — Clean/DR 夹爪增量与 XY reset 范围更新

- 类型：环境配置 / policy contract / 测试 / 文档
- 修改：Clean 和 DR 的夹爪总宽度增量从 `0.002` 改为 `0.005 m/30 Hz policy step`；方块 nominal `(0.50,0.00) m` 不变，完整 x/y half-range 从 `0.10` 改为 `0.05 m`。位置课程前 20k step 仍为 `±0.02 m`，随后在 100k 达到 `±0.05 m`。Privileged 诊断任务显式保留旧 2 mm 与 `±0.10 m` 配置。
- 契约：当前 Clean/DR action/history scale 为 `[0.025,0.025,0.025,0.005] m`，policy contract 升级到 v9；v7/v8 严格读取能力保留，旧 v8 live config 会因 action/reset 不匹配而 fail closed。
- 影响：4-D action、Actor/Critic observation shape、奖励、success、done、相机和频率不变；第四维 history、夹爪 transition、物体 reset distribution 与 env hash 改变，需要从头训练。当前没有 v9 真机 bundle。
- 验证情况：`git diff --check` 和完整 `compileall` 通过；4-env Clean/DR Isaac 定向 pytest 10 项与 2-env 无相机 Privileged pytest 1 项通过。标准 wrapper 仍误选 base Python；使用正确 Isaac 环境 warm start 成功后，被既有 ISSUE-015（缺失 `test_argparser_launch.py` skip target）阻断。未执行重新训练、正式评估或真机闭环。

## 2026-07-26 CST — 导出 25/2 mm Real-Alignment v8 真机 bundle

- 类型：policy contract / TorchScript export / 真机配置 / 测试 / 文档
- 依据：`2026-07-26_20-51-57_ppo_torch_vision_only_resnet18` 保存的环境配置使用 `[0.025,0.025,0.025,0.002] m/policy step`，不能套用固定 10/2 mm 的 v7 配置。
- 修改：contract validator 新增 v8 25/2 mm 严格语义，同时保留 v7 10/2 mm 读取能力；新增 `franka/configs/e2e_bundle_real_alignment_v8_25mm.json`，并将导出的模型/metadata 放入 `franka/checkpoint/real_alignment_v8_25mm/exported/`。
- Artifact：TorchScript SHA-256 为 `b888e7321c0481f80623ffa84837f32bc8f426eefbd1af33937764d787744c24`，metadata 声明 RGB `224x224x3`、proprio 15、history 4、action 4、contract v8。
- 验证情况：GPU 物理导出首次因并行训练占用显存失败；未终止训练进程，改用 CPU 物理成功导出，trace max abs error 为 0。独立 JIT 测试输出 finite/bounded，17 项真机 runtime 单测在注入 `franky` import stub 后通过，完整 bundle/config/metadata 严格校验通过。未连接或驱动真机。

## 2026-07-26 CST — Real-Alignment 切换到 400x398 实测 crop 和有效 K

- 类型：相机内参 / GPU 预处理 / policy contract / 真机配置 / 测试 / 文档
- 修改：Clean/DR 的真实预处理改为 `640x480 -> crop x=[100,500), y=[34,432) -> bilinear 224x224`，策略有效 K 为 `[338.742544,0,123.748857; 0,340.550811,120.393372; 0,0,1]`。
- Omniverse 补偿：原生 224x224 相机使用 centered 300 px-focal coverage view；ResNet 前用固定 batched GPU affine grid 映射到有效 K，以保留非中心主点和 `fx/fy` 差异。DR 在固定映射之后继续施加其随机 intrinsic/颜色后处理。
- Contract/部署：升级到 v7，记录 raw K、crop、目标 K、native render K 和补偿模式；新增独立 `franka/configs/e2e_bundle_real_alignment_v7.json`，v6 和 0712 配置不覆盖。旧 Real-Alignment v6 checkpoint 因视觉 contract 改变而 fail closed。
- 影响：Actor/Critic observation shape、4-D action、reward/success/done、224x224 camera buffer 和控制语义不变；视觉输入改变，必须重新训练。
- 验证情况：`git diff --check`、完整 `compileall`、JSON 解析、4-env Real-Alignment Isaac 定向 pytest 10 项、无相机 Privileged pytest 1 项和真机 action/history/crop/config 单测 16 项通过。标准 wrapper 仍误选 base Python；用正确 Isaac 环境执行 discovery 可完成 warm start，但被既有 ISSUE-015（缺失 `test_argparser_launch.py` skip target）阻断。未执行重新训练、完整 x 范围图像 smoke 或真实硬件闭环。

## 2026-07-26 CST — Real-Alignment XYZ 动作提高到 10 mm/step

- 类型：动作控制 / history / policy contract / 真机配置 / 测试 / 文档
- 修改：Clean/DR/Privileged Real-Alignment 的 XYZ scale 从 `0.005` 改为 `0.010 m/30 Hz policy step`；夹爪总宽度仍为 `0.002 m/step`。History 和真机 action adapter 同步为 `[0.010,0.010,0.010,0.002]`。
- Contract：升级到 v6；旧 v5 Real-Alignment 明确 fail closed。独立真机配置更新为 `franka/configs/e2e_bundle_real_alignment_v6.json`，旧 `0712` 配置不变。
- 影响：Actor/Critic observation shape、4-D action shape、相机、reward/success/done 不变；XYZ transition 和 history 数值范围改变，必须从头训练。
- 验证情况：`git diff --check`、完整 `compileall`、Real-Alignment Isaac 定向 pytest 9 项、Privileged pytest 1 项和真机 action/history/crop/config 测试 16 项通过；满幅 XYZ action 与 history 均断言为 `±0.010 m`，v5 contract 拒绝逻辑通过。未重新训练。

## 2026-07-26 CST — Real-Alignment 方块 x 范围改为 0.40–0.60 m

- 类型：环境 reset / 位置课程 / policy contract / 测试 / 文档
- 修改：Clean/DR/Privileged 共用的方块 nominal x 从 `0.60 m` 改为 `0.50 m`，完整 robot-root 范围改为 `x=[0.40,0.60] m`；y 仍为 `[-0.10,0.10] m`。前 20k steps 的 x 课程范围相应改为 `[0.48,0.52] m`，20k–100k 线性扩展。
- 原因：当前相机标定下，旧 `x=0.70 m` 方块最下沿落在 224x224 图像之外；新上界 `x=0.60 m` 对完整 5 cm 方块保留约 10 像素底部余量。
- 影响：相机、动作尺度、观测维度、reward/success/done 不变；reset distribution 和 v5 contract nominal/bounds 改变，旧范围 checkpoint 不兼容。
- 验证情况：`git diff --check`、完整 `compileall` 和 Real-Alignment/Privileged 定向 pytest 通过；3-env Isaac 图像 smoke 将方块分别固定在 `x=0.40/0.50/0.60 m`，三者均完整出现在实际 `[224,224,3]` 相机输出中，最远端仍保留底部边缘。预览保存于 `logs/validation/real_alignment_x_0p40_0p60_preview/`；尚未重新训练。

## 2026-07-26 CST — Real-Alignment v5 对齐新真机几何、视觉、动作和碰撞奖励

- 类型：环境 / 奖励与 success / ContactSensor / 课程 / 部署 contract / rollout / 测试 / 文档
- 适用 task：`TacEx-Sim2Real-Cube-Real-Alignment-v0`、DR 和 Privileged 变体；通用 `TacEx-Sim2Real-Cube-Grasp-v0` 保留原动作与倾角语义。
- 几何/reset：台面厚度设为 `1 mm`、顶面 `z=0.001 m`，方块质心 `z=0.026 m`；XY 以 robot-root `(0.60,0.00)` 为中心，前 20k steps 为 `±2 cm`，20k–100k 线性扩展到 `±10 cm`。play、bucket 和 rollout 强制完整范围。
- 相机：改用指定 `base_T_camera_color_optical` ROS 位姿；仿真等效渲染 224x224，真机锁定序列号 `215322076207` 并执行 `640x480 -> crop x=[80,560) -> bilinear 224x224`。Clean 固定近黑台面/背景和白方块，DR 保留既有视觉课程。
- 动作/history：30 Hz normalized action 逐维缩放为 `[0.005,0.005,0.005,0.002] m/step`；第四维为缓存总宽度增量，history 记录上一拍请求增量。真机 runner 新增 vector history scales 和 strict v5 contract 校验，旧 scalar 配置仍可读取，新增独立 v5 配置且未覆盖 `0712`；新配置 TCP workspace 的 x 上限同步到物体中心上界 `0.70 m`，未额外增加安全余量。
- 奖励/success：保留 `5*reach + 15*lift + 100*success`，质心 lift 为 0–35 mm；35 mm 连续 5 步即 success。倾角继续写诊断日志，但不再参与 reward、success 或 done。台面 ContactSensor 过滤 `panda_link1–7`、hand 和双 finger，最大力 `>1 N` 时每 step 加 `-10`，不终止。
- Contract/rollout：contract 升至 v5，旧 v4 Real-Alignment fail closed；rollout 增加碰撞标志/接触力/惩罚，分析不再用可能被碰撞项抵消的 raw reward 判断 success terminal。
- 影响：Actor/Critic observation shape 和 4-D action shape 不变；物理动作、视觉、reset、reward/success 和 config hash 改变，旧 checkpoint 必须废弃并从头训练。
- 验证情况：`git diff --check` 和完整 `compileall` 通过；定向 Real-Alignment pytest 9 项、Privileged pytest 1 项及真机 crop/history/config 测试 16 项通过。1-env Clean Isaac smoke 确认 RGB `[1,224,224,3]`、ContactSensor history `[1,2,1,10,3]`、正常台面位置下 100 步满幅下降无误碰撞；仅在临时把运动学台面移到指尖高度的强制接触 smoke 中，实际 filtered force 触发并施加 `-10`。已保存 5 张仿真帧和 1 张仓库旧现实参考 crop。标准 discovery 在 warm start 后被既有 `ISSUE-015` 阻断；未执行 200k 训练、100-episode 正式评估或真实硬件闭环。

## 2026-07-26 CST — 新增无相机 privileged-position Cube 上界任务

- 类型：环境 / Agent 配置 / task 注册 / 定向测试 / 文档
- 新任务：`TacEx-Sim2Real-Cube-Real-Alignment-Privileged-v0`，继承 Clean Real-Alignment 的物理、reset、4-D action、reward、done 和 5 秒/150-step horizon。
- 场景：不创建 `wrist_camera` 或 visual backdrop；继承链新增默认开启的 `vision_encoder_enabled` 开关，新任务设为 false，因此不构造 ResNet18，已有视觉 task 行为不变。
- Actor：28-D 输入，包含 `proprio [N,15]`、`history [N,4]`、仿真真值 `cube/gripper/target position [N,9]`；位置去除 `scene.env_origins`，支持多环境。Actor mean 仍使用 `tanh`。
- Critic：保持 49-D privileged input；新任务使用独立 PPO YAML 和 `logs/skrl/sim2real_cube_real_alignment_privileged/`。
- 目的：若该任务能稳定抓取而 vision-only 不能，主要问题指向视觉定位/表征；若仍不能，则优先检查奖励、动作时序、物理接触和课程。
- 兼容性：Actor input keys/28-D shape 与视觉 checkpoint 不兼容，必须从头训练；该 checkpoint 含仿真真值，不能导出到真机。
- 验证情况：`git diff --check`、完整 `python -m compileall -q source scripts tools`
  和 Agent YAML 解析通过；2-env `enable_cameras=False` 定向 pytest 通过，实际
  创建/reset/step 并确认无 `wrist_camera`/ResNet、position identity 和有限奖励。
  另以 4 env 完成 128 steps/一次 PPO update，Actor/critic 自动生成和训练链路通过。
  标准 `./tacex.sh -p tools/run_all_tests.py --discover_only` 因 wrapper 误用 base
  Python、缺少 `isaacsim` 而在 warm start 前失败；在正确 conda 环境中直接执行
  discover 时 warm start 成功，但被既有 `ISSUE-015`（skip 列表中的
  `test_argparser_launch.py` 不存在）阻断，均不属于新任务测试失败。

## 2026-07-26 CST — Sim2real Cube 删除 privileged `dz` gate

- 类型：动作控制 / sim2real contract / checkpoint 兼容性 / 测试 / 文档
- 适用 task：`TacEx-Sim2Real-Cube-Grasp-v0`、`TacEx-Sim2Real-Cube-Real-Alignment-v0`、`TacEx-Sim2Real-Cube-Real-Alignment-DR-v0`。
- 修改：父 Cylinder 配置增加 `privileged_dz_gate_enabled=true` 以保留 legacy 行为；Cube 配置覆盖为 false，因此 `_pre_physics_step()` 不再读取 ground-truth object XY，也不再修改 Actor 请求的 `dz`。
- Contract：升级到 v4，Cube 使用 `per_dimension_processed_action_no_privileged_gate_v3` history 语义并记录 `privileged_dz_gate=disabled`；Cube v1/v2/v3 fail closed。
- 影响：观测 key/shape、4-D action、奖励、done、夹爪总宽度和 IK 本身不变；Actor 输出后的 XYZ 控制语义改变，2026-07-25 22:01 的 v3 200k run 必须废弃并从头训练。
- 验证情况：`git diff --check`、`python -m compileall -q source scripts tools`
  通过；4-env Isaac 定向 pytest 共 7 个测试通过，其中运行时断言 Cube
  `_pre_physics_step()` 不访问 gate geometry。另用真实 2026-07-25 22:01
  manifest 验证 v3 checkpoint 被 v4 validator 明确拒绝。

## 2026-07-25 CST — 两个 sim2real Cube task 改为质心 lift curriculum

- 类型：奖励/success 语义 / curriculum / 日志 / policy contract / 测试 / 文档
- 适用 task：`TacEx-Sim2Real-Cube-Grasp-v0`、`TacEx-Sim2Real-Cube-Real-Alignment-v0`、`TacEx-Sim2Real-Cube-Real-Alignment-DR-v0`。
- Lift：不再使用最低角点；稳定落桌质心基准为 reset center 加 cube/plate rest offset。质心抬升 `0–35 mm` 线性映射到 lift `[0,1]`，且倾角超限不再将这个学习信号清零；35 mm 满足 success height。
- Tilt：仅约束 success，最大允许倾角从 policy step 0 的 `40°` 线性收紧到 step 120k 的 `10°`，形成先易后难的课程；`lift_tilt_curriculum_step_offset` 用于显式 resume。
- 日志：保留当前时刻的 env-mean reach/lift/success/total，新增最近 `reward_print_interval` 个 policy steps 的 `avg_step_total`。
- Contract：policy contract 从 v2 升至 v3并记录质心 lift/倾角课程字段；strict Cube v2 checkpoint fail closed。观测 key/shape、4-D action、控制、网络输入输出和真机程序不变。
- 历史物理证据：只读检查 2026-07-25 20:43 的 512-env 旧奖励 event log，确认 step 4000 的 instantaneous max 为 `119.78437` 且 success 标量曾非零；这证明场景曾产生完整抓取轨迹，但不是当前 v3 checkpoint 的成功率结果。
- 验证情况：`git diff --check` 与 `python -m compileall -q source scripts tools`
  通过；定向 Isaac pytest 共 6 个测试通过，并实际创建/重置 4 个 DR 并行环境。
  另以 1 env 实际创建 Clean task，reset 后执行一个零动作 policy step，确认
  30 Hz、150-step horizon、有限奖励和 `(512,15,4)` policy 输入维度。
  标准 `--discover_only` 完成 Isaac warm start 后仍被既有 `ISSUE-015`
  （缺失 `test_argparser_launch.py` skip target）阻断。

## 2026-07-25 CST — Real-Alignment DR 增加相机/光照/GPU 后处理课程学习

- 类型：环境视觉随机化 / curriculum / 多环境语义 / 定向测试 / 文档
- 课程：`common_step_counter` 每个 30 Hz policy step 加一，与 `num_envs` 无关；0–20k step 使用 full range 的 10%，20k–120k 线性增到 100%，之后保持 full range。新增 `dr_curriculum_step_offset` 作为显式 resume hook，训练入口尚未自动恢复该值。
- 相机：以用户提供标定位姿为中心，full range 为 XYZ 各 `±3 mm`、RPY 各 `±1°`，使用 `T_calib*deltaT` 和 `TiledCamera.set_world_poses(env_ids=...)`；focal scale `[0.985,1.015]` 与 principal point `±2 px` 通过 batched GPU affine warp 实现。
- 后处理：每环境每 episode 缓存 brightness/contrast/saturation/gamma/hue/white balance/blur/noise std；Gaussian noise 像素值逐帧采样。处理输入/输出均为 `[N,3,224,224]`，不改变 frozen ResNet18 的 `[N,512]` 输出。
- 场景：plate 在暗灰/黑色附近、backdrop 在绿色附近使用每环境独立 material；global ground 固定。共享 DomeLight 只在 full-batch reset 更新，部分 reset 不影响其他环境缓存或全局光照。
- 影响：Clean task、Actor/Critic observation key/shape、4-D action、控制、奖励和 done 均不变；DR env config hash 和视觉分布改变，旧 DR checkpoint 不应被当作相同 profile 续训结果。
- 验证情况：目标文件 `py_compile` 和局部 `git diff --check` 通过；定向 Isaac pytest 的配置用例进入并通过，但当前另一个 Python 进程占用约 19.1 GiB GPU memory，PhysX 申请约 640 MiB contact buffer 时 OOM，4 个环境级用例未实际完成。

## 2026-07-25 CST — Real-Alignment 对齐 30 Hz 控制链路并延长为 5 秒

- 类型：环境时间配置 / 定向测试 / 文档
- 修改内容：physics 保持 `dt=1/60 s`，Real-Alignment Clean/DR 使用 `decimation=2`；camera `update_period=1/30 s`，render interval 跟随 decimation。camera/render/policy/action/observation/reward/history 统一为 30 Hz。
- Episode：`episode_length_s` 从 2.5 秒改为 5 秒，实际 timeout horizon 为 150 个 policy steps；每个 policy step 包含两个 physics substeps。
- Contract：新增 camera update period/frequency、episode 秒数和 max episode steps；validator 固定要求 60 Hz physics、30 Hz camera/policy、5 秒/150 步，旧时间语义 fail closed。
- 影响：Actor/critic keys 和张量维度、动作尺度、奖励阈值、success/done 类型不变；控制/history 时间尺度和 episode horizon 改变，旧 60 Hz checkpoint 不得在当前配置中续训或部署。
- 验证情况：`python -m compileall -q source scripts tools` 和 `git diff --check` 通过；定向 Isaac pytest 4 个测试通过，实际报告 physics `1/60 s`、render/env step `1/30 s`，并断言 `max_episode_length=150`。标准 discover warm start 成功后仍被既有 `ISSUE-015`（缺失 `test_argparser_launch.py` skip target）阻断。

## 2026-07-25 CST — Real-Alignment 使用左右指尖位姿计算抓取中心

- 类型：环境几何 / 奖励 / privileged critic / action gate / strict contract / 测试 / 文档
- 修改内容：Clean/DR 的 IK/Jacobian TCP 从固定 `panda_hand+0.107 m` 改为 `panda_hand+0.1034 m`；reach reward、critic gripper position/target distance 和近台面 `dz` gate 改用左右 finger link 各经本地 `[0,0,0.045] m` 变换后的世界坐标中点。
- 几何依据：标准 Panda hand 到 finger origin 为 `0.0584 m`，finger origin 到 nominal tip center 为 `0.045 m`，合计固定 TCP `0.1034 m`；运行时左右指尖中点会显式使用两侧 link 位姿。
- 兼容范围：基础 Cube/Cylinder 通过默认 hook 保留旧 reward/gate 中心；Real-Alignment Clean/DR 共用新语义。
- 影响：Actor observation keys/dim、action shape、夹爪总宽度控制和 done 不变；reward、critic privileged state、`dz` gate 与 IK TCP 数值改变。strict v2 contract 新增 TCP/center 字段，旧 Real-Alignment checkpoint 必须重新训练。
- 额外确认：当时发现 `sim.dt=1/60, decimation=1` 使 policy 为 60 Hz、camera 为 30 Hz；后续已按用户确认改为 `decimation=2`，见 `ISSUE-017` 和本日后续记录。
- 验证情况：定向 Isaac pytest 4 个测试通过，包含指尖中点、固定 TCP、critic/reward、夹爪控制与 contract；完整静态验证见本次任务报告。

## 2026-07-25 CST — Real-Alignment 分离 Clean 与 broad DR 环境

- 类型：环境配置继承 / task 注册 / Agent 配置 / strict contract / 测试 / 文档
- Clean：`TacEx-Sim2Real-Cube-Real-Alignment-v0` 只保留 cube XY reset 随机化，关闭图像、光照、ground 和 plate color 随机化。
- DR：新增 `TacEx-Sim2Real-Cube-Real-Alignment-DR-v0`，继承完全相同的机器人、224x224 相机、方块、动作、观测、奖励和 done，启用 broad image/lighting/scene appearance DR。
- 日志：DR 使用独立 `skrl_ppo_cube_real_alignment_dr_cfg_resnet18.yaml` 和 `logs/skrl/sim2real_cube_real_alignment_dr/`。
- Contract：新增 task 已加入 train/export/strict contract allowlist；旧 v1 Real-Alignment DR contract fail closed。
- Checkpoint：Clean 和 DR 的 task id、env hash 和视觉分布不同，不得跨 profile resume；两者均需从头训练。
- 未加入：质量、摩擦、相机位姿/内参和机器人动力学随机化，因为现实范围待确认。
- 验证情况：`git diff --check`、受影响 Python `py_compile`、两个 Agent YAML 解析通过；定向 Isaac pytest 为 `3 passed`。另以 1-env 实际创建 DR task、reset 并执行一步零动作，确认 RGB `[1,224,224,4]`、ResNet `[1,512]`、proprio `[1,15]`、history `[1,4]` 和 reward 均为有限值。

## 2026-07-25 CST — Real-Alignment 相机改为 crop 等效 224x224 直接渲染

- 类型：环境相机配置 / 定向测试 / 文档
- 修改内容：仅将 `TacEx-Sim2Real-Cube-Real-Alignment-v0` 相机从原生 640x480 改为直接 224x224 渲染。等效内参为 `fx=282.285453, fy=282.373380, cx=112.457381, cy=115.692837`，对应真机 `640x480 -> crop x=[80,560), y=[0,480) -> bilinear resize 224x224`。
- 显存语义：仿真不再创建 640x480 RGB sensor buffer；像素数从 307200 降至 50176。总显存还取决于 RTX、物理和 PPO buffer，不能据此推断整体下降 6.1 倍。
- 影响：`wrist_resnet:512`、`proprio_obs:15`、`action_history:4`、动作、奖励和 done 均不变；camera contract 和视觉分布改变，旧 Real-Alignment checkpoint 必须重新训练。
- 真机边界：本次不修改 `franka`；现有 0712 配置已经声明中央 480x480 crop。RGB/BGR、插值、畸变和当前真机运行时是否严格执行该配置仍需实机验证。
- 验证情况：`git diff --check`、`python -m compileall -q source scripts tools` 通过；定向 Isaac 测试
  `conda run -n isaaclab_2.1.1 env TERM=xterm python -m pytest -q -s source/tacex_tasks/test/test_sim2real_cube_real_alignment_gripper.py`
  为 `2 passed`，实际 sensor RGB 和 policy contract 均断言为 224x224。

## 2026-07-25 CST — Real-Alignment Cube 改为总夹爪宽度增量控制

- 类型：环境动作控制 / policy contract / export metadata / 测试 / 文档
- 修改内容：Real-Alignment 第四维 clipped action `g` 每个 30 Hz policy step 请求 `0.01*g m` 总宽度增量；缓存目标限制到 `[0,0.08] m`，两侧 finger target 均为一半，两个 physics substep 只重发同一目标。Reset 恢复选中环境约 40 mm 初始宽度。
- History：`action_history[3]=0.01*g`，即请求的总宽度增量；前三维保持 `0.05*[x,y,z]`。观测 key/shape、动作维度、奖励和 done 均不变。
- Contract/export：新增 v2 逐维 scale `[0.05,0.05,0.05,0.01]`、总宽度缓存模式和对称映射字段；旧 v1 基础 Cube 可继续验证，旧 v1 Real-Alignment fail closed。
- Checkpoint：动作 transition 语义改变，旧 Real-Alignment checkpoint 不得 resume 或部署，需要从头训练。
- 真机边界：本次未修改 `franka`；当前真机总宽度 scale 仍为 `2 mm/step`，与仿真 `10 mm/step` 只在接口语义上对齐。
- 验证情况：`git diff --check`、`python -m compileall -q source scripts tools` 通过；定向 Isaac 测试
  `conda run -n isaaclab_2.1.1 env TERM=xterm python -m pytest -q -s source/tacex_tasks/test/test_sim2real_cube_real_alignment_gripper.py`
  为 `2 passed`。仓库 discover runner 在既有 skip 配置处报
  `ValueError: Test to skip 'test_argparser_launch.py' not found in tests.`，未进入测试发现。

## 2026-07-25 CST — 现实对齐 Cube 相机改为原生 640x480 内参

- 类型：环境相机配置 / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_real_alignment_env.py`
  - `docs/PROJECT_OVERVIEW.md`
  - `docs/ARCHITECTURE.md`
  - `docs/EXPERIMENTS.md`
  - `docs/DEVLOG.md`
- 修改内容：仅将 `TacEx-Sim2Real-Cube-Real-Alignment-v0` 的相机输出从 `224x224` 改为 `640x480`，并使用用户提供的 `fx=604.897400, fy=605.085815, cx=320.980103, cy=247.913223`。
- 视觉数据流：相机产生 `[N,480,640,3]` RGB，现有环境前处理随后双线性缩放为 `[N,3,224,224]`，冻结 ResNet18 继续输出 `wrist_resnet:[N,512]`。
- 影响的观测：Actor observation key 和特征维度不变，但取消了旧的中央方形 crop 等效投影，构图、宽高比和视觉特征分布改变；相机渲染显存和带宽增加。
- 影响的动作、奖励、done、网络维度：无。
- checkpoint/export 兼容性：旧 checkpoint 参数 shape 兼容但视觉分布不再一致，必须重新评估，正式使用建议重新训练；旧 224x224 TorchScript export 不代表当前相机 contract。
- 验证情况：本次任务结束报告记录实际执行命令。
- 待确认：新内参的标定来源、640x480 仿真画面与现实 RGB 的像素级对齐，以及可承受的并行环境数量。

## 2026-07-25 CST — 更新现实对齐 Cube 第三视角相机位姿

- 类型：环境相机配置 / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_real_alignment_env.py`
  - `docs/ARCHITECTURE.md`
  - `docs/EXPERIMENTS.md`
  - `docs/DEVLOG.md`
- 修改内容：仅将 `TacEx-Sim2Real-Cube-Real-Alignment-v0` 的固定第三视角相机更新为用户提供的 world `pos=(1.26349,-0.01190,0.51067)` 和 OpenGL `rot(wxyz)=(0.62066,0.34013,0.37847,0.59653)`；相机仍不挂载到 Franka link。
- 位姿解释：四元数模长约 `0.999997`；OpenGL 光轴约为 `(-0.876,-0.029,-0.482)`，向下约 28.8°。
- 影响的观测：`wrist_resnet` 的图像内容和 ResNet18 特征分布改变；key 和 shape 仍为 `wrist_resnet:512`。
- 影响的动作、奖励、done、网络维度：无。
- checkpoint 兼容性：旧 checkpoint 的张量 shape 兼容，但已适应旧相机构图，必须重新评估；用于正式训练/部署时建议从头训练。
- 验证情况：本次任务结束报告记录实际执行命令。
- 待确认：该外参是否来自实测 hand-eye calibration，以及新构图与现实 RGB 的像素级对齐效果。

## 2026-07-12 CST — 现实对齐 Cube lift reward 起点改为 1 mm

- 类型：奖励配置 / 文档
- 修改文件：
  - `source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_real_alignment_env.py`
  - `docs/EXPERIMENTS.md`
  - `docs/DATA_FLOW.md`
  - `docs/DEVLOG.md`
- 修改内容：仅将 `TacEx-Sim2Real-Cube-Real-Alignment-v0` 的 `lift_reward_start_delta` 从 `0.005 m` 改为 `0.001 m`；`success_lift_delta=0.035 m`、upright 20° 和 hold 5 steps 保持不变。
- 奖励语义：相对 reset 最低点抬升不超过 1 mm 时 lift reward 为 0；超过 1 mm 后在 1–35 mm 区间线性增长，35 mm 时为 1。
- 影响的观测、动作、done、网络维度：无。
- checkpoint 兼容性：奖励 shaping 改变，现有 checkpoint 可以按相同 shape 加载，但不应 resume 继续训练；推荐从头训练。
- 验证情况：本次任务结束报告记录实际命令和边界结果。
- 待确认：1 mm dead band 对接触抖动、PPO 收敛和真机成功率的影响。

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

## 2026-07-26 — 新增 Sim2real Cube 批量 rollout action 诊断

- 类型：实验脚本 / 诊断
- 修改文件：
  - `scripts/reinforcement_learning/skrl/collect_sim2real_cube_rollouts.py`
  - `scripts/reinforcement_learning/skrl/analyze_sim2real_cube_rollouts.py`
  - `source/tacex_tasks/tacex_tasks/cylinder_grasping/cylinder_grasping_vision_only_resnet18.py`
  - `README.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DATA_FLOW.md`
  - `docs/EXPERIMENTS.md`
  - `docs/KNOWN_ISSUES.md`
- 修改内容：新增 exported Actor 多环境 rollout 采集，记录 action/history/IK/joint/cube/fingertip/gripper/RGB/reward/done；新增离线 CSV 与 action/state plot 生成器；环境侧增加只读诊断缓存。
- 影响的观测：无；只读取现有 Actor observation。
- 影响的动作：无；诊断缓存不改变 `_pre_physics_step()` 和 `_apply_action()` 输出。
- 影响的奖励和 done：无。
- 网络输入维度：保持 `action_history[4] + proprio_obs[15] + RGB[224,224,3]`。
- checkpoint 兼容性：正常 train/play/export 继续 fail closed；collector 仅在显式 `--allow_legacy_contract` 时允许旧 v1-v3 离线诊断。
- 实际采集：v3 2026-07-25 run，32 env、450 policy steps、14,400 transition；98 个完成 episode 中 7 个 success terminal。
- 待确认：用户真机实际运行使用的 model/config 路径及 rollout log 尚未提供，未做逐帧 sim-real 同输入对照。

## 2026-08-01 — 新增 Real-Alignment RMA 教师学生蒸馏路线

- 类型：代码 / 配置 / 训练脚本 / 评估脚本 / 导出 / 文档
- 修改内容：新增 Clean v9 RMA Teacher/Student task、共享归一化 Actor Core、单帧 spatial-softmax 定位头、在线双损失蒸馏、固定网格评估和端到端 TorchScript 导出。
- 影响的观测：仅新增 task；Teacher Actor 为22维，Student 训练环境增加 raw RGB 与 cube XYZ label，部署接口不含 label。
- 影响的动作：新增路线继续使用4维 tanh action 和 `[0.025,0.025,0.025,0.005]` 尺度；现有 task 不变。
- 影响的奖励和 done：继承当前 Clean；新增只读 per-env success terminal 缓存供评估。
- checkpoint 兼容性：RMA 使用独立 manifest/checkpoint，旧 Privileged 及 Clean/DR checkpoint 不兼容也不被修改。
- 验证情况：`git diff --check` 和完整 `compileall` 通过；RMA Teacher 定向测试1项、Student/模型/TorchScript定向测试4项通过。Teacher 以4 env完成128步和一次PPO更新并写 manifest；Student 以2个相机环境完成2步双损失更新并保存 checkpoint；导出模型 eager/trace/reload 最大误差均为0。仓库 wrapper 使用 base Python，缺少 `isaacsim`；正确 conda 环境的 discover warm start 成功，但被既有缺失 `test_argparser_launch.py` skip target 阻断。
- 待确认：实际200k Teacher、100k Student 的成功率和位置 RMSE。

## 2026-08-01 — RMA Actor 加入可部署末端位置与相对目标位置

- 类型：代码 / 模型契约 / 测试 / 文档
- 修改内容：共享 Actor 从 `proprio_obs[:7]` 内嵌 Panda FK，计算机器人根坐标系指尖中点 XYZ，并与 cube XYZ 形成相对目标 XYZ；网络特征由22维增至28维。物体与末端 quaternion 不加入。
- 依据：Real-Alignment 使用 `panda_hand + [0,0,0.1034] m` 作为 IK TCP，且该点与左右 finger 各自 `[0,0,0.045] m` 指尖中心的中点一致；Panda 固定关节变换来自任务使用的 URDF。
- 影响的观测：Teacher/Student 环境 key 与 shape 不变；末端 XYZ 在模型内部从7维关节角派生，不引入仿真 privileged link input。
- 影响的动作、奖励和 done：无；保持4维 tanh action、Clean 控制/奖励/success/done。
- 影响的网络输入维度：`RMAActorCore` 第一层由22改为28；Student TorchScript 外部三输入签名不变。
- checkpoint 兼容性：RMA model/manifest/student artifact 升级至 v2，旧22维 RMA checkpoint 拒绝加载并需要重新训练；Clean、DR 和旧 Privileged checkpoint 不受影响。
- 验证情况：纯 Torch 检查确认初始 FK 输出约 `[0.499844,0.000043,0.299519] m`、仅定位头有梯度且 TorchScript 动态 batch 误差为0；Teacher 定向测试1项和 Student/模型/契约/TorchScript定向测试5项通过，其中 FK 与 Isaac 指尖中点按0.2 mm容差比较；4 env Teacher 完成128步和一次 PPO 更新并写 v2 manifest；2 env Student 完成2步双损失更新；导出 eager/trace/reload 最大误差为0。以上均为链路 smoke，不是策略效果。
- 待确认：当前28维方案实际200k Teacher、100k Student 的成功率和位置 RMSE。

## 2026-08-01 — RMA 增加特权接触奖励与视觉接触蒸馏

- 类型：RMA 环境 / 模型 / 训练 / 评估 / 导出 / 测试 / 文档
- 修改内容：仅为 RMA Teacher/Student 新增 cube-to-left/right-finger ContactSensor；0.5 N阈值生成两路二值接触，单侧/双侧接触每步奖励0.1/2.0。Teacher Actor 使用真实接触，Student adaptation head 从RGB联合预测XYZ与接触 logits，并加入接触 BCE。
- 影响的观测：两个 RMA task 新增 `rma_contact_state[N,2]`；Student 中该键只用于训练标签，不进入部署接口。Clean、DR、旧 Privileged 不变。
- 影响的网络输入维度：共享 Actor feature 从28维增至30维；视觉头输出由3维位置扩为3维位置加2维接触 logits。
- checkpoint 兼容性：RMA model/manifest/student artifact 升至v3，v2文件拒绝加载，Teacher和Student均需重新训练；Student TorchScript仍为RGB/本体/history三输入与4维动作输出。
- 验证情况：完整 `compileall`、Teacher YAML解析和 `git diff --check` 通过；Teacher定向测试1项（含实际双指接触力与2.0奖励）及Student/模型/契约/TorchScript定向测试5项通过。4 env Teacher完成128步与一次PPO更新并写v3 manifest；2 env Student完成2步三损失更新并保存v3 checkpoint；TorchScript导出 eager/trace/reload最大误差均为0。标准 discovery 用base Python时缺少`isaacsim`；正确conda环境warm start成功后被既有`ISSUE-015`缺失skip target阻断。
- 待确认：200k Teacher/100k Student正式效果、视觉接触 precision/recall/F1 和真实硬件接触泛化。

## 2026-08-01 — 精简 RMA 奖励控制台输出

- 类型：RMA 日志 / 文档
- 修改内容：只覆盖 RMA Teacher/Student 的控制台奖励摘要，保留加权 reach、lift、success、contact、table、当前 total、最近200步平均奖励、窗口成功率和平均抬升高度；完整诊断量继续写入 TensorBoard log。
- 影响：不改变奖励数值、观测、动作、done/success、网络输入或 checkpoint；Clean、DR、旧 Privileged 的控制台输出不变。
- 验证情况：`py_compile`、完整`compileall`和`git diff --check`通过；4 env Teacher完成256策略步与两次PPO更新，step 200仅打印精简摘要且窗口数值正确；RMA Teacher接触/奖励定向测试通过。

## 2026-08-01 — RMA success 改为非终止统计条件

- 类型：RMA 环境 / checkpoint contract / 测试 / 文档
- 修改内容：只覆盖 RMA Teacher/Student 的 `_get_dones()`；success 继续给奖励并更新 hold counter，但不再触发 done。Episode 只因 timeout 或严重机器人穿地碰撞终止；窗口成功率按 episode 内是否曾达到 success 统计。
- 影响：Clean、DR 和旧 Privileged 的 done/success 语义不变。RMA manifest、Student checkpoint 和导出 metadata 升至v4，v3及更早 RMA artifact fail closed，需要重新训练 Teacher/Student。
- 验证情况：`python -m compileall -q source scripts tools`、`git diff --check` 通过；RMA 配置隔离/旧v3 artifact拒绝/Teacher success非终止定向测试通过。

## 2026-08-01 — RMA 单独降低 finger actuator

- 类型：RMA 环境 / checkpoint contract / 测试 / 文档
- 修改内容：只在 RMA Teacher/Student 的 copied robot cfg 中覆盖 `panda_hand` actuator：`effort_limit_sim=40`、`stiffness=400`、`damping=40`。按用户要求，`gripper_width_delta_scale` 保持 Clean v9 的 `0.005 m/policy step`。
- 影响：Clean、DR、旧 Privileged 和 TacEx GelSight asset 不变。RMA 物理/控制 contract 改变，manifest、Student checkpoint 和导出 metadata 升至v5，v4及更早 RMA artifact fail closed，需要重新训练 Teacher/Student。
- 验证情况：`py_compile`、`git diff --check` 通过；RMA 配置隔离/旧v4 artifact拒绝/Teacher success非终止定向测试通过。2-env RMA Teacher 1-iteration smoke 成功写出 v5 manifest，记录 `velocity_limit_sim=null`、5mm/step 动作尺度和 40/400/40 hand actuator。

## 2026-08-01 — RMA Student 允许直接蒸馏

- 类型：训练脚本 / 文档
- 修改内容：移除 `train_rma_student.py` 的 `--teacher_eval` 与 `--allow_unqualified_teacher` 参数及其固定网格验收门槛。Student 现在只要求兼容的 Teacher checkpoint 及其同一 run 中的 manifest，随后直接开始在线视觉位置、接触和动作蒸馏；`evaluate_rma.py` 保留为可选的后验比较工具。
- 影响：不改变 Student 环境、观测、损失、网络输入、动作或 checkpoint 格式；无需重新训练已存在的 Teacher。未验证 Teacher 成功率的 checkpoint 也可作为蒸馏源，质量由用户自行负责。
- 验证情况：`python -m py_compile scripts/reinforcement_learning/skrl/train_rma_student.py source/tacex_tasks/tacex_tasks/sim2real_grasp/rma_artifacts.py` 与 `git diff --check` 通过。

## 2026-08-01 — 对齐 RMA Student 与 Teacher 的接触 contract

- 类型：RMA 环境配置 / 蒸馏启动修复 / 文档
- 修改内容：Student 原本从 Clean 继承 `0.5 N / 2.0 / 0.1` 的接触阈值、奖励权重和单侧系数，但指定 Teacher manifest 记录的是 `0.2 N / 3.0 / 1.0`。Student 现在显式使用 Teacher 数值，使环境 contract 校验通过，并保证 Student rollout 的接触标签和奖励分布与教师一致。
- 影响：修改 RMA Student 的接触标签阈值与 reward；不改变 Teacher checkpoint、Actor/Student 网络输入维度、loss 形式或 checkpoint 格式。此前尚未完成的 Student run 不应继续使用。
- 验证情况：待本次 Student 一步启动验证。

## 2026-08-02 — 统一 RMA Student 蒸馏进度日志

- 类型：训练脚本 / 训练监控 / 文档
- 修改内容：Student 启动时明确打印总 update、并行环境数、剩余 simulator transition、日志和 checkpoint 间隔；每个日志间隔输出进度、累计 transition、实时 update/sample 吞吐、已耗时、ETA、分项 loss、位置 RMSE、接触 accuracy 和环境发布的累计/最近成功率。checkpoint 输出也改为统一前缀并强制 flush。
- 影响：只改变控制台和 TensorBoard 监控，不改变环境、观测、loss、优化器、动作、奖励、done 或 checkpoint 格式。`timesteps` 仍表示外层 update，单轮采样 transition 数为 `timesteps * num_envs`。
- 验证情况：待本次语法与最小 Student 蒸馏验证。

## 2026-08-02 — RMA Student 增加无课程全强度视觉域随机化

- 类型：RMA 环境 / 训练脚本 / checkpoint 路由 / 测试 / 文档
- 修改内容：新增 `TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-DR-v0`。该 Student 复用 Real-Alignment DR 实现，并显式关闭 `dr_curriculum_enabled`，因此首个 update 即按 full scale 采样相机外参、焦距/主点 GPU warp、图像颜色/模糊/噪声、板/背景和共享 DomeLight 扰动。
- 影响的观测：部署和模型输入仍为 `wrist_rgb uint8[N,224,224,3]`、`proprio_obs[N,15]`、`action_history[N,4]`；仅 RGB 分布改变。`rma_cube_pos[N,3]` 与 `rma_contact_state[N,2]` 仍只作训练标签。
- 影响的物理/奖励/done：不改变 Teacher 或 Student 的机器人、cube、接触阈值/奖励、动作尺度、success 与 done 语义；现有 Teacher manifest 继续校验通过。
- checkpoint 兼容性：Student payload 记录 Clean 或 DR task。旧 Clean Student v5 artifact 仍可导出；resume 会拒绝跨 Clean/DR profile。DR Student 应从头蒸馏，不应把既有 Clean Student checkpoint 作为 resume 输入。
- 验证情况：待本次语法、RMA 定向测试和最小 DR Student 蒸馏 smoke。

## 2026-08-02 — 新增蒸馏 RMA Student 专用回放脚本

- 类型：评估脚本 / 文档
- 修改内容：新增 `play_rma_student.py`，直接加载 RMA Student artifact 的模型 state_dict，不走 skrl PPO `agent.load()`。脚本从 checkpoint 读取 Clean 或 DR task，校验 Teacher manifest 的环境 contract 及 Student encoder/Actor hash，以全随机 XY 范围运行固定步数，并写周期 CSV 与最终 JSON。
- 影响：不修改训练、Teacher、Student 网络、观测、动作、奖励、done 或 checkpoint 格式；通用 `play.py` 保持只服务 skrl checkpoint。
- 验证情况：待本次 Student checkpoint 回放 smoke。

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
## 2026-08-02 — RMA Student TorchScript 支持 CUDA 加载

- 类型：部署导出 / 模型兼容性 / 测试 / 文档
- 修改内容：RMA Student 导出由 `torch.jit.trace` 改为 `torch.jit.script`，避免 trace 将 Panda FK 中静态索引的 buffer 内联为 CPU 常量；RMA Student 导出脚本在 CUDA 可用时自动加载新文件到 `cuda:0` 并与 CPU eager 输出比较，CUDA 的 `1e-3` 动作容差单独记录以允许 CPU/GPU 卷积浮点差异。
- 影响：Teacher/Student 数值、state_dict 键名、观测、动作、训练和现有 checkpoint 加载语义不变；旧 TorchScript 文件仍可能含 CPU 内联常量，应重新导出后再部署 GPU。
- 验证情况：已以 Clean、DR 两个现有 Student checkpoint 直接 script 并在 `cuda:0` reload/forward；两者均成功。导出的 CPU/CUDA 动作最大绝对误差分别为 `1.328e-4`、`3.963e-4`，低于 `1e-3` 容差。
## 2026-08-02 — 更新 Real-Alignment 标定相机外参

- 类型：环境配置 / sim-to-real 视觉 contract / 测试 / 文档
- 修改内容：`Sim2RealCubeRealAlignmentEnvCfg.wrist_camera` 的 nominal `base_T_camera_color_optical` 更新为平移 `(1.166091088407, 0.035901608197, 0.514200335898) m`，以及 ROS optical Isaac `wxyz=(0.378248136306, -0.604227000834, -0.586824979121, 0.384024117374)`；该四元数由提供的 ROS `xyzw` 重排而来，重建旋转矩阵与提供的 `T_base_color` 一致。
- 影响：Clean、DR 和两个 RMA Student 的相机图像分布改变，现有 Student checkpoint 和 TorchScript 不应继续用于新外参；Teacher 是无相机特权策略，外参本身不改变其观测或动作语义。内参、动作、物理、奖励、done 和网络维度不变。
- 验证情况：`python -m py_compile` 通过；`git diff --check` 通过；在 Isaac Lab `isaaclab_2.1.1` 中运行 `pytest -q -k clean_and_dr_randomization_profiles_are_separated` 通过，确认 Clean/DR 均解析为该平移、四元数和 `convention=ros`。

## 2026-08-02 — 对齐真机 Franka 20 mm 基座垫高

- 类型：环境几何 / sim-to-real frame contract / 测试 / 文档
- 修改内容：Real-Alignment Clean 配置将 Panda root world position 设为 `(0,0,0.02) m`；桌面与方块不移动。相机标定仍以提供的 `base_T_camera_color_optical` 保存，独立 TiledCamera 的 world position 同步变为 `(1.166091088407,0.035901608197,0.534200335898)`，故 Panda base 到 camera 的相对变换不变。RMA Teacher manifest 升级至 v6，并把 robot base、camera base/world pose 写入 Teacher–Student environment contract。
- 影响：机器人相对于桌面/物体的初始几何、IK 可达性、碰撞和图像视角均改变；Clean、DR、RMA Teacher 和两个 RMA Student 都必须从头训练，现有 checkpoint/TorchScript 不可继续使用或 resume。观测/动作张量维度、内参、奖励公式与 done 语义不变。
- 验证情况：`python -m py_compile` 和 `git diff --check` 通过；在 Isaac Lab `isaaclab_2.1.1` 中运行 Clean/DR 配置定向测试与 4-env reset/table-collision smoke 均通过；RMA Clean/DR/Teacher/Student 配置契约测试通过。

## 2026-08-02 — 保存 RMA Student 的训练输入帧

- 类型：训练可观测性 / 视觉 DR 诊断 / 文档
- 修改内容：`train_rma_student.py` 新增 `--save_start_frame` 与 `--start_frame_count`。启用后，首次 reset 的不同环境 `wrist_rgb` 以 PNG 写入本次 run 的 `camera_frames/`；保存的是经过当前 Student observation path（含 DR 与 nominal intrinsic warp）的 224×224 uint8 图像，不执行额外环境动作。
- 影响：默认关闭，训练、随机化、损失、checkpoint 和 Student 部署输入不变；启用后仅在训练开始增加有限的图像 I/O。
- 验证情况：`python -m py_compile scripts/reinforcement_learning/skrl/train_rma_student.py` 和 `git diff --check` 通过。以新的 Teacher checkpoint、`RMA-Student-DR-v0`、2 env、1 update 运行保存 smoke，成功写出 `start_wrist_rgb_env000_224x224.png` 与 `start_wrist_rgb_env001_224x224.png`，并完成 checkpoint 保存。

## 2026-08-02 — 以真机裁剪帧重设 Real-Alignment visual nominal 与 DR

- 类型：视觉 sim-to-real / 相机 DR / 场景外观 / 实验配置
- 修改内容：检查了提供的 `224x224` 真机裁剪帧与新外参、20 mm base elevation 下的 Clean/DR Student 输入帧。保留给定标定相机的 nominal 外参和内参；将 Clean 桌面/背景/DomeLight 调为更暗中性的值，并将 DR 收紧为 camera XYZ ±1.5 mm、RPY ±0.5°、focal ±1%、principal ±1 px，brightness/contrast ±10%、gamma ±8%、saturation -10%/+5%、hue ±2°、white balance ±4%、blur 8%、noise std 0–0.006、光照 1300–1900 与 4800–6200 K，以及近黑板/背景材质范围。
- 影响：Clean 与 DR Student 的 RGB 分布改变，所有现有 Student checkpoint/TorchScript 均不可继续训练或部署，须按新 nominal 从头训练。Teacher 没有视觉输入，动作/物理/奖励/张量维度不变；但新 Teacher run 仍应作为新实验的配套 artifact。真实图中的导轨未建模，是当前未覆盖的 visual gap。
- 验证情况：`python -m py_compile` 和 `git diff --check` 通过；Isaac Lab 定向 Clean/DR 配置测试通过。分别以 Clean 1 env 和 DR 4 env、各 1 update 运行保存帧 smoke；Clean/DR 均成功产生 224×224 Student 输入 PNG，新的 DR 样本未再出现旧范围下的明显绿色桌面。

## 2026-08-03 — RMA Teacher/Student 增加动作平滑项

- 类型：RMA reward / Student distillation loss / checkpoint contract / 测试 / 文档
- 修改内容：RMA Teacher、Clean Student 和 DR Student 环境共用 `rma_action_rate_penalty_weight=0.05`，按当前实际下发物理增量 `action_history` 与上一拍的差计算动作变化惩罚，差值除以环境动作尺度 `[0.05,0.05,0.05,0.01]` 后取均方。`train_rma_student.py` 新增 `--action_smoothness_loss_weight`，默认0.05，约束 Student action 接近上一拍环境 action，并将 smooth loss 写入 TensorBoard、控制台和 Student checkpoint loss contract。
- 影响的观测：不改变 `proprio_obs[15]`、`action_history[4]`、`wrist_rgb[224,224,3]`、`rma_cube_pos[3]` 或 `rma_contact_state[2]` 的 key/shape。
- 影响的动作：不改变4维 tanh action 或环境动作尺度；只改变训练目标，使策略倾向减少相邻 policy step 的命令跳变。
- 影响的奖励：RMA reward 新增负项 `reward/rma_action_rate`，Clean/DR/旧 Privileged reward 不变。
- 影响的 done/success：不改变；RMA success 仍只统计不终止。
- checkpoint 兼容性：Teacher manifest 升至 v7，并把 action-rate reward 写入 environment contract；v6 及更早 Teacher 不应继续用于新 Student 蒸馏，需从头训练 Teacher/Student。Student 模型和 TorchScript 输入输出维度不变。
- 验证情况：`python -m py_compile` 通过；`git diff --check` 通过；RMA 配置与 legacy artifact 拒绝定向测试通过；Student 梯度/TorchScript 定向测试通过；独立 Isaac Teacher smoke 确认零动作 `reward/rma_action_rate=0.0`，从0跳到满幅 x action 时 `reward/rma_action_rate=-0.0125`、`info/rma_action_rate_norm_sq_mean=0.25`。完整 Teacher pytest 在旧 FK 对比段输出异常，仅得到 `F` 且无 traceback，本次未作为通过项。
