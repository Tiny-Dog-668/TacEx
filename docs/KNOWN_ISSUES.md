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
