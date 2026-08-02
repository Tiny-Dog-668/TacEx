# Known Issues

## ISSUE-001 — train/play/play_bucket 重复实现配置和 checkpoint 逻辑

- 状态：未处理
- 严重性：中
- 位置：`scripts/reinforcement_learning/skrl/train.py`、`scripts/reinforcement_learning/skrl/play.py`、`scripts/reinforcement_learning/skrl/play_bucket.py`
- 关键函数：`_process_cfg`、`_load_component`、`_restore_env_cfg_from_data`、`_filter_cfg_overlay`、`main`
- 现象：训练、普通评估、bucket 评估都维护 custom component、cfg 恢复、checkpoint 兼容逻辑。
- 风险：不同入口行为漂移，旧 checkpoint 兼容修复可能只覆盖某个入口。
- 建议：在不改变行为的前提下抽共享 helper，并先补 smoke test。

## ISSUE-002 — import-time monkey patch skrl Runner

- 状态：未处理
- 严重性：中
- 位置：`source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py`
- 关键函数：`_patch_skrl_runner_for_custom_models`
- 现象：导入 `occluded_grasping` 时 patch `skrl.utils.runner.torch.Runner._component`。
- 风险：全局影响后续 skrl Runner 行为，排查加载问题时不直观。
- 建议：长期改为训练/评估入口内的显式 custom loader；改动前确认所有 YAML 中 `module:Class` 用法。

## ISSUE-003 — 本机绝对路径影响可移植性

- 状态：未处理
- 严重性：高
- 位置：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py`、`source/tacex_tasks/tacex_tasks/sparch_grasp/vision_two_tactile_ijepa_env.py`
- 关键字段：`_DEFAULT_DINO_ENCODER_ROOT`、`OccludedGraspingVisionFourTactileBoxCfg.tactile_dino_repo_path`、`sparsh_repo_path`、`sparsh_checkpoint_path`
- 现象：路径包含 `/home/tinydog/桌面`、`/home/tinydog/Projects/sparsh` 等本机绝对路径。
- 风险：其他机器或容器中 DINO/Sparsh 路线无法复现。
- 建议：改为配置项、环境变量或仓库内相对路径；未确认 checkpoint 分发方式前不要删除旧路径记录。

## ISSUE-004 — `vt_box.py` 单文件职责过重

- 状态：未处理
- 严重性：中
- 位置：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py`
- 关键类：`OccludedGraspingVisionFourTactileBoxCfg`、`OccludedGraspingVisionFourTactileBoxEnv`
- 现象：同一文件包含 scene/object/robot/sensor/action/observation/reward/reset/logging/encoder 逻辑。
- 风险：修改局部功能时容易影响观测维度、奖励或旧 checkpoint。
- 建议：先文档化数据流和测试入口，再拆分 reward、sensor encoding、reset/randomization、logging。

## ISSUE-005 — `vision_box.py` 与 `vt_box.py` 存在 drawer 配置重复

- 状态：未处理
- 严重性：低到中
- 位置：`source/tacex_tasks/tacex_tasks/occluded_grasping/vision_box.py`、`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py`
- 关键类：`OccludedGraspingVisionOnlyBoxCfg`、`OccludedGraspingVisionFourTactileBoxCfg`
- 现象：drawer geometry、panel cfg 等常量存在重叠。
- 风险：scene 位置或遮挡几何调整时，视觉-only baseline 与 VT 环境可能不一致。
- 建议：抽共享 scene config；改动前用现有 task id 创建环境做 smoke test。

## ISSUE-006 — `GelSightSensor.reset` 假定 `height_map` 存在

- 状态：待确认
- 严重性：中
- 位置：`source/tacex/tacex/gelsight_sensor.py`
- 关键函数：`GelSightSensor.reset`
- 现象：代码直接访问 `self._data.output["height_map"]`。
- 风险：若某个 `GelSightSensorCfg.output_type` 不包含 `height_map`，reset 可能报错。
- 建议：确认所有使用场景是否固定包含 `height_map`；若不是，改为存在性检查。

## ISSUE-007 — torchvision 不可用时视觉特征可能静默退化为零向量

- 状态：未处理
- 严重性：中
- 位置：`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py`
- 关键函数：`OccludedGraspingVisionFourTactileBoxEnv.__init__`、`_get_observations`
- 现象：视觉 encoder 不可用时，`third_resnet` 可变成零向量。
- 风险：视觉策略可继续运行但实际失去视觉输入，实验结果容易被误读。
- 建议：训练和评估启动时显式记录 encoder 状态；必要时在关键视觉任务中 fail fast。

## ISSUE-008 — Bucket 评估默认 occlusion ratio 可为 `nan`

- 状态：已知行为
- 严重性：低到中
- 位置：`scripts/reinforcement_learning/skrl/play_bucket.py`
- 关键函数：`main`
- 现象：当前已确认 `play_bucket_20260531_152044_summary.csv` 中 `occlusion_ratio_mean` 为 `nan` 的行存在；代码默认可关闭 bbox occlusion。
- 风险：将 bucket 成功率按遮挡程度分析时，可能没有有效遮挡分桶。
- 建议：实验记录必须写明 bbox/occlusion 设置；需要遮挡分桶时打开并验证 bbox 输出。

## ISSUE-009 — 当前工作区存在未提交业务改动

- 状态：待确认
- 严重性：中
- 位置：`source/tacex_tasks/tacex_tasks/occluded_grasping/*`、`scripts/reinforcement_learning/skrl/play.py`、`scripts/occluded_grasping/*`
- 依据：`git status --short`
- 现象：存在修改、删除和新增文件，包含 visible-contact 删除、aux/visual-predict-visible 新增等。
- 风险：文档可能同时描述已提交代码和未提交实验状态；后续修改容易覆盖用户工作。
- 建议：每次工作前先检查 `git status --short`，只修改用户允许的文件。

## ISSUE-010 — 历史/实验入口脚本是否仍使用待确认

- 状态：待确认
- 严重性：低
- 位置：`scripts/reinforcement_learning/skrl/play_111.py`、`play_lstm.py`、`train_ori.py`、`play_ori.py`
- 现象：存在多个类似训练/评估入口。
- 风险：新协作者难以判断推荐入口；修复可能漏掉历史脚本。
- 建议：先在文档中标出推荐入口；清理前查询 Git 历史和使用记录。

## ISSUE-011 — 旧 sim2real Cube checkpoint 不兼容 strict deployment contract

- 状态：已知行为，需要重新训练
- 严重性：高
- 位置：`logs/skrl/sim2real_cube_grasp/`、`logs/skrl/sim2real_cube_real_alignment/`
- 现象：2026-07-12 修复前的 run 使用可能更新 BatchNorm running statistics 的 encoder、额外延迟一拍的 `action_history` 和 unbounded Actor mean；run 中也没有训练 encoder artifact/manifest。
- 风险：旧 checkpoint 即使参数 shape 可以加载，其视觉、时间和动作语义也与新训练/真机 contract 不一致；为旧模型仅在 exporter 中增加 `tanh` 会进一步改变策略。
- 当前处理：Cube train resume、play、play_bucket 和 exporter 都要求 v1 run contract；缺少 training-time `tanh`、task/config hash 或 verified encoder artifact 的旧 run fail closed。
- 建议：使用当前配置从头训练，不要 resume 旧 checkpoint；checkpoint、`params/agent.*` 和 `params/vision_encoder_resnet18.*` 必须作为同一个 run 一起保留。

## ISSUE-012 — `tanh` 约束 Actor mean，不是 squashed Gaussian sample

- 状态：已知范围边界
- 严重性：低到中
- 位置：`source/tacex_tasks/tacex_tasks/sim2real_grasp/agents/skrl_ppo_cube_*_cfg_resnet18.yaml`
- 现象：`output: "tanh(ACTIONS)"` 将 Gaussian mean 约束在 `[-1,1]`，确定性 play/export 因此有界；训练采样仍会叠加 Gaussian noise，单次 sampled action 理论上可越界。
- 当前处理：环境将 normalized action clamp 到 `[-1,1]` 后逐维缩放；通用 Cube 为 `0.05`，当前 Clean/DR Real-Alignment v9 为 XYZ `0.025 m/step`、夹爪总宽度 `0.005 m/step`；保留的 v8/v7 contract 分别为 25/2 mm 与 10/2 mm。Sim2real 环境额外 action noise 为 0。
- 建议：若未来要求概率分布样本本身严格有界，应单独实现 squashed Gaussian 及其 log-prob/Jacobian 修正，并重新训练；不能只在采样后静默加 `tanh`。

## ISSUE-013 — Real-Alignment v8 真机 contract 已静态对齐，硬件时序待验证

- 状态：部分解决；代码和独立配置已对齐，硬件闭环待验证
- 严重性：高
- 位置：`cylinder_grasping_vision_only_resnet18.py:_pre_physics_step`、`franka/configs/e2e_bundle_real_exported_0712.json`
- 当前处理：Cube 配置设 `privileged_dz_gate_enabled=false`；Clean/DR v9 使用 400x398 crop、XYZ `25 mm/step`、夹爪总宽度 `5 mm/step`、一拍 history 和 `x/y ±5 cm` reset。历史 `franka/configs/e2e_bundle_real_alignment_v8_25mm.json` 仍严格对应旧 v8 的 25/2 mm 与 `x/y ±10 cm`，不能用于 v9。Runner 保留 v7/v8 严格读取能力。
- 剩余现象：robot-root XYZ 符号、30 Hz 周期抖动、新 crop 边界和 GPU 有效 K 补偿仍需在真实硬件闭环验证；当前短训练策略没有稳定 lift/success。
- 风险：同 shape 的 v7/v8/v9 checkpoint 不能交换配置；25 mm 最大单步平移和 v9 的 5 mm 夹爪总宽度增量均需在第一轮硬件测试前生成匹配 bundle，并预览、逐步确认。
- 实测：该 v3 run 的 32-env、450-step exported-Actor rollout 共得到 98 个完成 episode，仅 7 个 success terminal（7.14%）；episode 第一步 Z/夹爪 action 近饱和比例分别为 93.08%/89.23%，夹爪目标有 76.51% transition 饱和在 80 mm。该批 rollout 中 legacy `dz` gate 实际触发率为 0%，所以本批失败不能归因于 gate 在执行时替 policy 兜底。
- 建议：历史 v8 run 只使用独立 v8 配置；当前 v9 必须从头训练并另行生成 v9 bundle。首次连接真机先执行 preview，再用 `--confirm-step --steps 1` 单步观察。

## ISSUE-014 — Run hashes 是一致性校验，不是签名认证

- 状态：已知边界
- 严重性：低
- 位置：`vision_encoder_artifact.py`、`export_sim2real_grasp_jit.py`
- 现象：manifest 绑定 config/encoder hash，export metadata 记录 checkpoint/TorchScript hash，可发现普通 config/encoder 误配和损坏；但训练时 manifest 尚未绑定之后产生的每个 checkpoint，当前依赖 `<run>/checkpoints/` 目录归属约定，sidecar 也没有数字签名。
- 风险：把另一个同 shape checkpoint 放进该 run，或同时替换 artifact/manifest，当前机制不能证明 checkpoint 同源或发布者身份。
- 建议：若未来需要供应链认证，对整个 deployment bundle 使用受信公钥签名；当前不要把 SHA-256 一致性描述为防篡改认证。

## ISSUE-015 — 测试 discovery 引用了不存在的 skip target

- 状态：未处理
- 严重性：低
- 位置：`tools/run_all_tests.py` 使用的 test skip 配置
- 复现：`TERM=xterm conda run -n isaaclab_2.1.1 --no-capture-output ./tacex.sh -p tools/run_all_tests.py --discover_only`
- 现象：Isaac warm start 成功后抛出 `ValueError: Test to skip 'test_argparser_launch.py' not found in tests.`。
- 风险：标准 discovery 命令退出码为 1，不能作为 CI 发现阶段的成功信号。
- 建议：从 skip 列表移除已不存在的测试，或把缺失 skip target 降级为 warning；修改前确认该测试是否应恢复。

## ISSUE-016 — DR task 的 DomeLight 仍是跨环境全局状态

- 状态：已缓解，保留批次级限制
- 严重性：中
- 位置：`sim2real_cube_real_alignment_env.py:Sim2RealCubeRealAlignmentDREnv._randomize_scene_visuals`
- 当前处理：Real-Alignment DR 和 RMA Student DR 都固定 global ground；plate/backdrop、相机和 GPU 后处理均为 per-env episode state；部分 reset 不再更新 DomeLight，只有 full-batch reset 才采样共享光照。
- 剩余限制：同一批并行环境不能同时拥有不同的真实 RTX DomeLight intensity/temperature；per-env brightness/white balance 只能在图像空间补充覆盖。
- 建议：若必须研究独立三维光照方向/阴影，应建立每环境局部 light prim 并验证 tiled rendering 隔离与性能，不能把图像曝光等同于物理光照。

## ISSUE-017 — Real-Alignment policy 与相机更新频率当前不一致

- 状态：已解决
- 严重性：高
- 位置：`sim2real_cube_real_alignment_env.py:Sim2RealCubeRealAlignmentEnvCfg`
- 原现象：`sim.dt=1/60`、`decimation=1` 曾使 physics/policy 为 60 Hz，但 camera 为 30 Hz；相邻 policy step 可能读取同一 RGB 帧。
- 当前处理：用户确认保留 60 Hz physics，并将 `decimation=2`、camera update period 和 render interval 对齐，使 camera/render/policy/action/observation/reward/history 均为 30 Hz；episode 改为 5 秒/150 policy steps。
- 剩余要求：真机部署必须同样按 30 Hz 更新 policy action 和 history；旧 60 Hz checkpoint 只能按其原配置复现，不得混用当前时间语义。

## ISSUE-018 — DR 与物体位置 curriculum resume offset 尚未自动恢复

- 状态：待处理
- 严重性：中
- 位置：`sim2real_cube_real_alignment_env.py` 的 `dr_curriculum_step_offset` 与 `cube_position_curriculum_step_offset`
- 现象：两类课程进度都使用环境进程内的 `common_step_counter`；重新启动进程 resume checkpoint 时该计数从 0 开始。配置提供显式 offset，但训练入口尚未从 checkpoint trainer timestep 自动写入。
- 风险：中断续训时未设置 offset 会把 DR 拉回零扰动 Clean 视觉场景，并把物体位置拉回中央 `±2 cm`；视觉与几何分布都不再对应预期训练阶段。
- 建议：当前 resume 前同步设置两个 offset；后续在明确 skrl checkpoint timestep 来源后，由训练入口自动恢复并写入 run metadata。play、bucket 与 rollout 已强制完整物体范围，不受该问题影响。

## ISSUE-019 — RMA Student 的单帧视觉接触标签可能部分不可观

- 状态：待实验确认
- 严重性：中
- 位置：`rma_models.py:SpatialSoftmaxAdaptationHead`、`train_rma_student.py`
- 现象：Teacher 的左右指接触来自0.2 N物理力阈值，但 Student 只读取单帧腕部 RGB。遮挡、刚体无可见形变以及接触刚发生或刚消失的时序状态可能产生外观近似而标签不同的样本。
- 风险：接触 BCE 可能主要学到夹爪与方块几何接近，而不能可靠判断真实受力；高总体 accuracy 也可能来自无接触负样本占比高。
- 当前处理：训练使用正样本权重5.0；评估单独报告 precision、recall、F1、真实和预测双侧接触占比，不以总体 accuracy 单独判断效果。
- 建议：先用固定网格评估确认；若 recall 仍低，应增加短时图像/动作历史或触觉输入，而不是继续放大接触奖励。
