# AGENTS.md

本仓库的长期维护约束。改代码、配置或文档前先读本文件。流程按改动风险分级（第 2 节），投入与风险匹配；**第 4 节硬规则对所有分级一律适用，不因精简豁免**。

## 1. 项目

Isaac Sim / Isaac Lab 上的机器人抓取研究，涵盖遮挡条件下的视触融合与 Cube 的 sim-to-real 迁移。

- **当前活跃主线**：`sim2real_grasp` 的 Real-Alignment Cube（Clean / DR / Privileged / RMA Teacher-Student），以及 `sim2real_gelsight_rma` 的 GelSight robot profile 薄封装。
- **保留主线**：`occluded_grasping` 遮挡抓取（用户 2026-06-26 确认，近期无改动，配置与 checkpoint 仍保留）。
- **基础设施**：`source/tacex`（GelSight 传感器仿真）、`source/tacex_assets`（机器人与传感器资产）。
- 现状清单与模块细节见 `docs/ARCHITECTURE.md`，其中第二部分第 17 节是 RMA 路线的完整契约描述。
- 待确认：UIPC/FEM 软体路线（`source/tacex_uipc`）是否仍为主线依赖。

## 2. 工作流程（按分级）

分级由**改动触及的面**决定，不由行数决定；触及即算，拿不准按高一级处理。

- **T2 高风险**：观测 key / 动作维度 / 张量维度、奖励项或权重、success/done/终止语义、坐标系约定、contract/manifest/版本号/hash、checkpoint 加载或导出、训练/评估/蒸馏的语义或超参、资产物理参数（质量、actuator、碰撞、接触过滤）、task 注册。
- **T1 常规**：改功能代码、脚本、配置，但不触及任何 T2 面。
- **T0 琐碎**：注释、文档文案、类型注解、纯格式、局部重命名、不改行为的日志文案。

所有分级的最小前置：用第 3 节索引定位文件，不按文件名猜功能；跑 `git status --short` 识别用户已有未提交改动（默认是用户改动，**不回滚、不用 destructive git**）；只读改与本任务直接相关的文件。

- **T1 追加**：读被改函数的直接调用方；确认涉及的 task id、入口脚本、配置路径在代码中真实存在。
- **T2 追加**：查相关 `git log`；逐项确认观测 key、动作维度、奖励项、done/success、checkpoint 与日志路径**来自实际代码**；动手前先输出「当前实现理解 + 计划修改文件 + 影响面（含旧 checkpoint 是否需重训）」；确认不了的写「待确认」。

## 3. 目录索引

| 位置 | 内容 |
| --- | --- |
| `scripts/reinforcement_learning/skrl/train.py`、`play.py`、`play_bucket.py` | skrl PPO 训练、普通评估、固定网格 bucket 评估 |
| `scripts/reinforcement_learning/skrl/train_rma_student.py`、`play_rma_student.py`、`evaluate_rma.py`、`export_rma_student_jit.py` | RMA 蒸馏训练、Student 回放、10x10 网格评估、端到端 TorchScript 导出 |
| `scripts/reinforcement_learning/skrl/custom_models.py`、`custom_agents.py` | 供 task 配置按类路径引用的自定义 model / agent |
| `source/tacex_tasks/tacex_tasks/sim2real_grasp/` | Real-Alignment 主线 env（`*_env.py`）、`rma_artifacts.py`（task 名常量、manifest 与 hash 校验）、`rma_models.py`、`agents/*.yaml` |
| `source/tacex_tasks/tacex_tasks/sim2real_gelsight_rma/` | GelSight robot profile 薄封装，注册 5 个 GelSight RMA task |
| `source/tacex_tasks/tacex_tasks/occluded_grasping/` | 遮挡抓取，`vt_box.py` 为核心 env，同目录为策略实现与 `agents/*.yaml` |
| `source/tacex/tacex/gelsight_sensor.py` | `GelSightSensor`：camera/depth/height_map/tactile_rgb/marker_motion 缓冲 |
| `source/tacex_assets/tacex_assets/` | `robots/franka/*`（含 GelSight gripper 资产）、`sensors/gelsight_mini/` |
| `source/tacex_tasks/test/` | 定向 pytest，`test_sim2real_cube_real_alignment_*` 覆盖 RMA 契约 |
| `tools/run_all_tests.py` | 扩展测试发现入口，当前被 `docs/DECISIONS.md` 的 ISSUE-015 阻断 |
| `logs/skrl/<task>/<run>/` | `params/`、`checkpoints/`、`metrics/` 等训练与评估产物 |

Task id 注册于各子包的 `__init__.py`；RMA task 名常量集中在 `sim2real_grasp/rma_artifacts.py`。

## 4. 代码修改硬规则

- 只修改完成当前任务所必需的文件；不得未经确认重构无关模块。
- 不得删除已有功能、实验配置、checkpoint 路径记录或历史文档信息，除非用户明确要求。
- 不得随意更改动作空间、观测 key、张量维度、奖励权重、成功判定、done 语义或坐标系约定。
- 新实验优先新增配置文件，不覆盖仍在使用的 YAML。
- 任何可能改变实验结果的修改，必须说明受影响的观测、奖励、网络输入维度、旧 checkpoint 加载和是否需要重新训练。
- 遇到已有未提交改动时默认是用户改动；不回滚，不用 destructive git 命令。
- 复杂张量操作注明输入/输出维度，坐标系转换注明约定，奖励项注明物理含义。
- 不留临时调试代码、硬编码本机路径、设备号或环境数量。

## 5. 验证规则

验证遵循“先低成本、后高成本，耗时运行默认交给用户”的原则。**不得声称未运行的验证已经通过**；未运行的写明「未运行 + 原因 + 建议命令」，这不阻止在风险已说明的情况下交付修改。

### 5.1 默认执行边界

- **只读问答、参数定位、代码解释和状态查询**：默认不运行测试、不启动 Isaac Sim；只检查必要代码、配置或已有日志。只有现有证据不足且结果必须依赖运行时行为时，才建议测试。
- **代理可直接运行的快速检查**：预计不超过 30 秒且不启动 Isaac/GPU 的检查，例如改动文件 `compileall`、`bash -n`、`git diff --check` 或一个很小的离线单测。只运行与本次改动直接相关的最小集合，不默认跑全仓测试。
- **默认由用户运行的耗时检查**：任何 Isaac Sim 启动、GPU/相机 smoke、训练/回放、预计超过 30 秒的 pytest，以及全仓测试。代理应提供可直接复制的命令、预期现象和判定标准，不自行启动。
- 用户明确要求代理测试，或用户说明本机无法完成某项必要测试时，由代理运行最小定向测试；开始前先说明测试内容和预计耗时。若实际运行明显超出预期，及时停止并报告，不反复重跑。
- 只有当测试依赖代理当前持有而用户无法复现的状态，或不运行就很可能交付破坏性/不可恢复结果时，代理才可主动执行耗时测试；仍须先告知用户，并限制到最小范围。
- 代码与配置未变化、且已有同一版本同一路径的有效验证证据时，不重复运行测试；直接引用已有证据及其时间/适用范围。

### 5.2 按风险分级

- **T0**：纯文档无需测试；仅做内容复核。若改了 `.py`，只对改动文件做快速 `compileall`。
- **T1**：代理默认执行改动文件的快速语法/静态检查；离线单测只有预计不超过 30 秒时才直接运行，其余交给用户。
- **T2**：代理默认执行改动文件的 `compileall` 和静态契约检查；定向 pytest、Isaac smoke、GPU/相机验证列为建议验证并默认交给用户。用户授权或无法执行时，代理再运行最小必要集合。

### 5.3 沟通与收尾

- 修改前若预计需要耗时测试，先说明哪些由代理做、哪些建议用户做，避免完成代码后才开始长时间运行。
- 收尾必须区分“已运行并通过”“未运行”“由用户待验证”；对用户待验证项给出一条最小命令和明确通过标准。
- 不为回答一个简单追问重复验证整个环境；只有代码发生相关变化或用户要求复验时才重新测试。

| 目标 | 命令 |
| --- | --- |
| 改动文件语法检查 | `python -m compileall -q <本次改动的.py文件或目录>` |
| 定向测试 | `conda run -n isaaclab_2.1.1 env TERM=xterm python -m pytest -q -s source/tacex_tasks/test/<file>.py` |
| 训练 smoke | `python scripts/reinforcement_learning/skrl/train.py --task TacEx-Sim2Real-Cube-Real-Alignment-RMA-Teacher-v0 --num_envs 4 --headless` |

`./tacex.sh -p tools/run_all_tests.py --discover_only` 当前会因 ISSUE-015 非零退出，不能作为成功信号。带相机的 task 需要加 `--enable_cameras`。

## 6. 文档维护

只更新被本次改动**直接影响**的条目。

`docs/` 只保留三个正文文档：`ARCHITECTURE.md`（项目概览 / 模块架构 / 数据流）、`DEVLOG.md`（变更日志 / 实验记录）、`DECISIONS.md`（设计决策 `DEC-xxx` / 已知问题 `ISSUE-xxx`）；历史日志在 `docs/archive/`。不新增正文文档。

- **T0**：一般不更新；仅当改动本身是约定或机制变更时，在 `DEVLOG.md` 第一部分记一行。
- **T1**：`DEVLOG.md` 第一部分追加一条。
- **T2**：`DEVLOG.md` 第一部分必记；再按需更新 `DEVLOG.md` 第二部分（新增或完成实验）、`DECISIONS.md` 第一部分（设计取舍）或第二部分（新增风险与临时补丁）、`ARCHITECTURE.md`（结构、观测/动作/奖励/终止数据流变化）。

每条记录 3–5 行封顶，只写「改了什么 / 契约与兼容性影响 / 验证情况」，细节留在代码和 manifest 里。实验结果必须来自日志、CSV、checkpoint 或明确人工记录，不得把未执行的实验写成结果；确认不了的写「待确认」，不用文件名或目录名推断结论。

## 7. 收尾输出

- **T0 / T1**：修改文件、验证情况（实际跑了什么、未跑的写明）、建议 commit message。
- **T2**：再追加已确认信息、待确认项、影响面（受影响的观测/动作/奖励/张量维度、旧 checkpoint 能否加载、是否需重训）。

commit message 用常规前缀（`feat:` / `fix:` / `refactor:` / `docs:` / `test:` 等），如实描述本次改动。
