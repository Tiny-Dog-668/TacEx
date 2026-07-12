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
- 当前处理：环境在乘以 `action_scale` 后将 processed action clamp 到 `[-0.05,0.05]`；sim2real 环境额外 action noise 为 0。
- 建议：若未来要求概率分布样本本身严格有界，应单独实现 squashed Gaussian 及其 log-prob/Jacobian 修正，并重新训练；不能只在采样后静默加 `tanh`。

## ISSUE-013 — Sim2real Cube 仍有 privileged `dz` gate 与真机 history 配置差异

- 状态：未处理；本次只修改仿真/训练侧指定 contract
- 严重性：高
- 位置：`cylinder_grasping_vision_only_resnet18.py:_pre_physics_step`、`franka/configs/e2e_bundle_real_exported_0712.json`
- 现象：仿真在接近台面时使用 ground-truth object XY 决定是否允许负向 `dz`，真机没有该 privileged state；history 记录 gate 前 processed command，因此也不包含实际被阻止的 `dz`。当前真机 0712 配置仍是旧策略的 `history_scale=0.05, history_delay_steps=2`，arm adapter scale 约为新训练 contract 的十分之一，并使用阻塞 move，不能稳定等同 30 Hz 仿真控制。仿真夹爪则在每个 physics substep 重算增量 target，单个 policy step 的实际位移取决于 PD 跟踪，不能简化为固定宽度 delta。
- 风险：即使视觉、history 单位和 Actor mean 已对齐，仿真控制 transition 仍可能与真机不同；直接复用旧真机配置会再次引入一拍 history 延迟、arm 约 10 倍尺度差异和不同的夹爪动态。
- 建议：新策略部署配置至少使用 `history_source=clipped_action`、`history_scale=0.05`、`history_delay_steps=1`，并按 metadata 校验 XYZ 尺度、robot-root 坐标系、夹爪控制语义和实测周期；另行决定移除仿真 privileged gate，或在真机增加可验证的 object-pose safety gate 后重新训练。

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
