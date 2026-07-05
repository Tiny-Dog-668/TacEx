# AGENTS.md

本文件是当前仓库的长期维护约束。任何自动化助手或协作者在修改代码、配置或文档前，必须先阅读本文件，并优先遵守这里的项目事实、工作步骤和验证规则。

## 1. 项目目标

本项目在 Isaac Sim / Isaac Lab 中研究遮挡条件下的视觉触觉融合机器人抓取。当前主线已由用户确认为 `occluded_grasping`；`source/tacex` 和 `source/tacex_assets` 作为该主线依赖的触觉仿真与资产基础设施保留。

| 状态 | 内容 | 依据 |
| --- | --- | --- |
| 已实现 | Franka Panda + GelSight Mini + 第三视角相机的遮挡抓取环境 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxEnv` |
| 已实现 | Gymnasium task 注册矩阵，支持 `TacEx-{Fusion}-{Scene}-{Object}` 和 legacy `TacEx-VT-...-v0` | `source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_scene_object_fusion_matrix`、`_register_task` |
| 已实现 | skrl PPO 训练、普通评估、固定网格 bucket 评估入口 | `scripts/reinforcement_learning/skrl/train.py:main`、`play.py:main`、`play_bucket.py:main` |
| 已实现 | 视觉、四路触觉、本体状态和 privileged critic state 的观测字典 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.observation_space`、`OccludedGraspingVisionFourTactileBoxEnv._get_observations` |
| 部分实现 | Alpha gate、GRU、cross attention、aux heads、DINO/Sparsh 等策略实验矩阵 | `source/tacex_tasks/tacex_tasks/occluded_grasping/agents/*.yaml`、`vt_alpha_gru_policy.py`、`vt_cross_policy.py`、`vt_tactile_cross_alpha_aux_policy.py` |
| 部分实现 | 软物体 Cylinder/Cube/Cuboid 变体和 deformable reset 分支 | `source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py:_register_scene_object_fusion_matrix`、`vt_box.py:_is_can_deformable` |
| 计划设计 | 统一 train/play/play_bucket 中重复的 custom component、checkpoint、cfg 恢复逻辑 | 目前重复位于 `scripts/reinforcement_learning/skrl/train.py`、`play.py`、`play_bucket.py` |
| 已确认 | 当前项目主线是 `occluded_grasping` 遮挡抓取实验 | 用户于 2026-06-26 确认；`README.md`；`source/tacex_tasks/tacex_tasks/occluded_grasping/*` |
| 待确认 | UIPC/FEM 软体路线是否为遮挡抓取主线依赖 | `source/tacex_uipc/*`、`source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py` |

## 2. 每次工作前必须执行的步骤

1. 阅读 `AGENTS.md`。
2. 阅读 `README.md`。
3. 查看 `git status --short`，识别用户已有未提交改动。
4. 查看与任务相关的最近 Git 历史，例如 `git log --oneline --decorate -n 10`。
5. 用 `rg --files` 或 `rg` 定位相关文件，不要只根据文件名判断功能。
6. 阅读相关入口文件、配置文件和被调用模块。
7. 确认训练入口、评估入口、task id、agent YAML 是否来自实际代码。
8. 确认观测 key、动作维度、奖励项、done/success 语义是否来自实际代码。
9. 确认模型加载、checkpoint 路径和日志目录是否真实存在。
10. 输出对当前实现的简短理解。
11. 列出计划创建或修改的文件。
12. 说明可能影响的模块、接口、张量维度和旧 checkpoint。
13. 对无法确认的内容明确写“待确认”。
14. 在用户限制修改范围时，只修改允许的文件。
15. 修改完成后更新相关文档，并报告实际执行的验证命令。

## 3. 项目地图

| 模块 | 文件路径 | 关键类或函数 | 作用 |
| --- | --- | --- | --- |
| 项目说明 | `README.md` | 文档主体 | 当前项目描述、安装提示、训练/评估示例 |
| Codex 维护约束 | `AGENTS.md` | 本文件 | 工作流程、模块地图、验证和文档维护规则 |
| 训练入口 | `scripts/reinforcement_learning/skrl/train.py` | `main`、`_process_cfg`、`_load_component` | 启动 Isaac Sim，加载 task/env/agent cfg，创建 skrl trainer |
| 普通评估入口 | `scripts/reinforcement_learning/skrl/play.py` | `main`、`_restore_env_cfg_from_data`、`_filter_cfg_overlay` | 加载 checkpoint，回放策略并写 play metrics |
| Bucket 评估入口 | `scripts/reinforcement_learning/skrl/play_bucket.py` | `main`、`_build_bucket_positions`、`_write_bucket_object_positions` | 固定物体位置网格评估 checkpoint 并输出 CSV |
| 任务包导入 | `source/tacex_tasks/tacex_tasks/__init__.py` | `import_packages`、`_configure_local_isaac_asset_root` | 导入 task 子包并 patch 本机 Isaac asset root |
| 遮挡抓取注册 | `source/tacex_tasks/tacex_tasks/occluded_grasping/__init__.py` | `_register_scene_object_fusion_matrix`、`_register_task`、`_patch_skrl_runner_for_custom_models` | 注册 task id，支持 custom model class 路径 |
| 核心环境 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py` | `OccludedGraspingVisionFourTactileBoxCfg`、`OccludedGraspingVisionFourTactileBoxEnv` | 定义 scene、robot、sensor、动作、观测、奖励、reset |
| 视觉-only 基线 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vision_box.py` | `OccludedGraspingVisionOnlyBoxCfg`、`OccludedGraspingVisionOnlyBoxEnv` | 继承/复用 VT 环境，移除触觉策略输入 |
| 动作控制 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py` | `_pre_physics_step`、`_apply_action` | 将 5 维 policy action 转为 differential IK 和夹爪 joint target |
| 观测构建 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py` | `_get_observations` | 生成 `proprio_obs`、`third_resnet`、四路 tactile feature 和 critic privileged keys |
| 奖励计算 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py` | `_get_rewards`、`_compute_tactile_contact_rewards` | 计算 reach、lift、success、drop-after-success、触觉 contact reward |
| 终止与重置 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py` | `_get_dones`、`_reset_idx` | timeout/ground collision done，success 只统计；重置 robot/object/sensor state |
| 机器人配置 | `source/tacex_assets/tacex_assets/robots/franka/franka_gsmini_gripper_rigid.py` | `FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG` | Franka Panda + GS Mini gripper 资产配置 |
| 任务内 robot cfg | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py` | `OccludedGraspingVisionFourTactileBoxCfg.robot` | 将 Franka 资产挂载到环境路径 |
| 第三视角相机 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py` | `third_person_camera`、`_degrade_third_person_rgb` | 生成 224x224 RGB 和视觉退化输入 |
| GelSight 传感器实现 | `source/tacex/tacex/gelsight_sensor.py` | `GelSightSensor`、`_initialize_impl`、`_update_buffers_impl` | 维护 camera/depth/height_map/tactile_rgb/marker_motion 缓冲 |
| GelSight Mini 配置 | `source/tacex_assets/tacex_assets/sensors/gelsight_mini/gsmini_cfg.py` | `GelSightMiniCfg` | 配置 Taxim/FOTS、输出类型、默认 device |
| 任务内四路触觉 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_box.py` | `gsmini_left`、`gsmini_right`、`gsmini_left_down`、`gsmini_right_down` | 左/右内侧和左/右下侧触觉输入 |
| Alpha-GRU 策略 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_alpha_gru_policy.py` | `OccludedGraspingVTAlphaGRUPolicy`、`compute` | 读取视觉、四路触觉、本体状态，输出 policy action 分布 |
| Cross attention 策略 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_cross_policy.py` | `OccludedGraspingVisionTactileCrossAttentionPolicy`、`compute` | 视觉触觉 cross attention 融合策略 |
| Aux 策略 | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_tactile_cross_alpha_aux_policy.py` | `OccludedGraspingVTTactileCrossAlphaAuxPolicy`、`compute` | 带 auxiliary heads 的融合策略 |
| Aux PPO agent | `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_aux_ppo_agent.py` | `PPOWithAuxHeadsLoss`、`_update` | 将 policy outputs 中的 aux loss 加入 PPO 更新 |
| Agent 配置 | `source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_alpha_gru.yaml` | `models.policy.class`、`models.value.network.input`、`agent` | Alpha-GRU PPO 的 key、网络层、超参数和日志配置 |
| 测试入口 | `tools/run_all_tests.py` | `main` | 发现/运行扩展测试 |
| 环境测试 | `source/tacex_tasks/test/test_environments.py` | `TestEnvironments`、`setup_environment` | 遍历 Isaac Lab 环境创建和 smoke test |
| 日志与 checkpoint | `logs/skrl/occluded_grasping/` | `params/agent.yaml`、`params/env.yaml`、`checkpoints/best_agent.pt`、`metrics/*.csv` | skrl 训练和评估产物 |

## 4. 代码修改规则

- 只修改完成当前任务所必需的文件。
- 不得未经用户确认重构无关模块。
- 不得删除已有功能、实验配置、checkpoint 路径记录或历史文档信息，除非用户明确要求。
- 不得随意更改动作空间、观测 key、张量维度、奖励权重、成功判定、done 语义或坐标系约定。
- 新实验优先新增配置文件，不直接覆盖仍在使用的 YAML。
- 任何可能改变实验结果的修改，必须说明影响的观测、奖励、网络输入维度、旧 checkpoint 加载和是否需要重新训练。
- 遇到已有未提交改动时，默认认为是用户改动；不要回滚，不要用 destructive git 命令。
- 复杂张量操作要注明输入/输出维度；坐标系转换要注明约定；奖励项要注明物理含义。
- 不要留下临时调试代码、硬编码本机路径、设备号或环境数量。

## 5. 验证规则

优先使用从实际代码确认过的命令。除非用户明确要求，不要声称未运行的验证已经通过。

| 验证目标 | 命令 | 说明 |
| --- | --- | --- |
| Python 语法检查 | `python -m compileall source scripts tools` | 不启动 Isaac Sim，只检查可编译性 |
| 发现测试 | `./tacex.sh -p tools/run_all_tests.py --discover_only` | 通过仓库 wrapper 调用测试发现 |
| 运行任务测试 | `./tacex.sh -p tools/run_all_tests.py --extension tacex_tasks` | 需要 Isaac/GUI/GPU 环境，耗时和依赖较重 |
| 训练 smoke 示例 | `python scripts/reinforcement_learning/skrl/train.py --task TacEx-Alpha-GRU-Drawer-Occlusion-Cube --num_envs 4 --enable_cameras` | task id 来自 `occluded_grasping/__init__.py` 注册矩阵 |
| 普通评估示例 | `python scripts/reinforcement_learning/skrl/play.py --task TacEx-VT-Downsample-Drawer-Occlusion-Cube --checkpoint logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt --num_envs 16 --enable_cameras` | checkpoint 路径已在仓库日志中确认存在 |
| Bucket 评估示例 | `python scripts/reinforcement_learning/skrl/play_bucket.py --task TacEx-VT-Downsample-Drawer-Occlusion-Cube --checkpoint logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt --num_envs 16 --enable_cameras` | 使用固定物体位置评估并写 CSV |

## 6. 文档维护规则

- `docs/PROJECT_OVERVIEW.md` 维护项目目标、当前任务、实现状态、快速开始和下一步。
- `docs/ARCHITECTURE.md` 维护模块职责、入口、配置、模型保存加载和调用关系。
- `docs/DATA_FLOW.md` 维护观测、视觉、触觉、本体、融合、Actor/Critic、动作、奖励、终止之间的数据流。
- `docs/DEVLOG.md` 记录代码、配置、文档和工程机制变更；不得把未执行实验写成结果。
- `docs/EXPERIMENTS.md` 记录已完成、进行中和计划实验；结果必须来自日志、CSV、checkpoint 或明确的人工记录。
- `docs/DECISIONS.md` 记录设计决策、背景、理由、影响和后续条件。
- `docs/KNOWN_ISSUES.md` 记录重复代码、废弃代码、临时补丁、风险、复现路径和处理状态。
- 每次修改代码或实验配置后，同步更新受影响文档。
- 不确定信息写“待确认”，不要用文件名或目录名推断结论。

## 7. 每次任务结束时的输出格式

任务结束时，输出必须包含：

### 修改文件

列出实际创建或修改的文件路径。

### 已确认信息

列出从代码、配置、日志或 Git 历史确认的关键事实。

### 待确认

列出仍无法从仓库确认的信息。

### 验证情况

列出实际运行的命令和结果；未运行的验证必须说明未运行。

### 业务代码修改

明确说明是否修改了业务代码、配置、训练脚本或评估脚本。

### 建议提交信息

给出一条建议 commit message。当前文档机制更新建议使用：

```text
docs: add project knowledge and experiment tracking system
```
