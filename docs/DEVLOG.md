# Development Log and Experiments

本文件第一部分记录工程、代码、配置和文档维护变更，第二部分记录已确认与计划中的实验。设计取舍与问题风险见 `DECISIONS.md`，系统说明见 `ARCHITECTURE.md`。

## 第一部分 变更日志

### 2026-09-11 CST — 新增 X040 三帧 Progress Teacher–Student 对照任务

- 新增独立 X040 Progress Teacher/三帧 Student：复用 x=0.40、XY ±0.08/±0.10 m、关节 reset 噪声、全场景和 full-strength 0823 DR，同时固定使用 v7 21 mm GelSight USD。
- 契约：reach 每步使用权重2.5的绝对 proximity，lift 每步使用权重2.5的 reset-relative 归一化绝对高度，contact 保持 signed progress，terminal success 为一次性 `+1000`；其余采用 0911 的20→5 N碰撞和动作平滑。观测保持七路三帧输入、动作保持4维，旧 X040/Size-Buckets task 行为不变。
- 兼容性：artifact 强制新 Teacher–Student 配对，旧 checkpoint 不能用于新任务，需重新训练 Teacher 并蒸馏 Student。验证情况见本次交付，Isaac 8-env 相机 smoke 待用户运行。

### 2026-09-10 CST — 新增 GelSight Size-Buckets 进展奖励与终止成功环境

- 新增独立 Progress Teacher/Student task：reach、lift 与 contact 改为有符号进展奖励，保持状态不再重复得分；成功高度稳定 5 步后仅奖励一次并立即 terminated/reset。
- 奖励新增归一化动作幅值 `-0.05 mean(a²)`，并以 15 N 为阈值使用连续二次过力惩罚；观测 key/shape、4维动作和 v7 USD 不变，旧 Size-Buckets task 完全保留。
- 契约：新旧 Teacher/Student task 强制配对，旧 checkpoint 张量可加载但 MDP 语义不兼容；新 Teacher 需从头训练，Student 随后重新蒸馏。验证：快速静态检查见本次交付，Isaac/GPU 训练与 reset/step 待用户运行。

### 2026-08-23 CST — 全部 GelSight task 增加柔顺抓取与掉落惩罚

- 奖励：左右 GelPad 严格 `>2 N` 生成接触标签并保留单侧/双侧 `+1.5/+3`；任一侧严格 `>15 N` 每步 `-5`。Cube 曾抬升 `>=20 mm` 后回落到 `<5 mm` 时，每 episode 首次事件 `-10`，不终止。
- 契约：全部 11 个 GelSight Teacher/Student 共用，观测/动作 shape 与 success/done 不变；Size-Buckets、X040、Pulled artifact/profile 升版，旧 Teacher 需重训，Student 需重新蒸馏。非 GelSight RMA 不变。
- 验证：新增边界 pytest 通过；全部 11 个 GelSight task 配置解析为同一组 `2/15 N、-5/-10` 参数，2-env Teacher reset/step smoke 通过。未运行长时训练或人为制造真实掉落的轨迹测试。

### 2026-08-23 CST — 全局 GelSight 夹爪组件向上回移 5 mm

- 资产：从零偏移 v5 基准重新生成持久 v7 USD；`panda_hand/link8` 不动，左右 finger、case、GelPad 和中心参考的净延长由 26 mm 改为 21 mm，并同步三个 hand-level joint 锚点，全部 11 个 GelSight task 共用。
- 契约：hand→GelPad 中心/最低点改为 `0.1392/0.1563 m`，IK/reward/safety/Teacher FK 同步；观测/动作 shape、奖励权重和 done 不变。Size-Buckets、X040、Pulled artifact/profile 均升版，旧 Teacher 需重训、Student 需重新蒸馏。
- 验证：生成器已核对 hand/link8 零位移、7 个下游刚体统一 21 mm 位移及 3 个 hand-level joint 锚点；改动文件 `compileall`、`diff --check` 和全部 11 个 GelSight task 配置解析通过。未运行 reset/step 或相机渲染 smoke，交由用户做画面确认。

### 2026-08-23 CST — 取消 X040 强碰撞终止

- 修改：X040 Size-Buckets Teacher/三帧 Student 保留 `20→5 N` 非法碰撞阈值、每步 `-10` 惩罚与 force 日志，但删除 `200→20 N` 强碰撞终止；机器人刚体原点穿过 ground 的终止与 timeout-only truncated 保持不变。
- 契约：观测/动作 shape、奖励权重和 success 不变，terminated 语义改变；环境 profile/Teacher manifest/Student checkpoint 升至 v5/v5/v8，旧 X040 Teacher 需重训且 Student 需重新蒸馏。
- 验证：改动 Python `compileall`、残留引用检查与 `git diff --check` 通过；X040 定向 pytest 因当前执行环境无可用 CUDA，在完成用例前 35 s 超时，Isaac reset/step 与重新训练曲线交由用户运行。

### 2026-08-23 CST — 统一全部 GelSight RMA 的基座高度与偏移资产

- 修改：`sim2real_gelsight_rma` 的 11 个注册 task 统一使用 `franka_gsmini_standard_arm_visuals_v6.usd`，Franka 根 Z 从 `20 mm` 改为 `15 mm`；外部相机随基座下降 5 mm，以保持 base_T_camera 不变，并修复相机基座位姿从 `configclass` 类对象读取导致的 task 注册失败。
- 契约：全局 hand→GelPad 中心/最低点改为 `0.1442/0.1613 m`，IK/reward/safety/Teacher FK 同步；观测/动作维度、奖励权重和 done 不变。Size-Buckets artifact 为 v4/v5、X040 为 v4/v7、Pulled 为 v9，所有旧 GelSight Teacher 需重训且 Student 需重新蒸馏。
- 验证：改动 Python `compileall`、共享资产离线关节检查及 Isaac task 导入/8-env 配置解析通过；当前执行环境无可用 CUDA，reset/step、相机和碰撞 smoke 交由用户运行。非 GelSight 与 `occluded_grasping` 未修改。

### 2026-08-23 CST — Pulled-Drawer GelSight 指组件下移 26 mm

- 资产：新增 Pulled-Drawer 专用 v8 Franka USD；保持 `panda_link8`、`panda_hand` 与两者固定关节原点不变，只将左右 finger、GelSight case/GelPad 和中心指尖参考沿 hand 局部 `+Z` 延长 26 mm，并同步关节锚点。
- 契约：IK/reward/safety 的 hand→GelPad 中心与最低点偏移改为 `0.1442/0.1613 m`，Teacher FK 同步；观测/动作维度、奖励和 done 不变。geometry/manifest/Student artifact 升至 v8、Teacher model 升至 v4，旧 Pulled Teacher 需重训且 Student 需重新蒸馏，X040 不受影响。
- 验证：Pulled USD 离线确认 hand/link8 位姿零变化、7个下游刚体 Z 均为 `-0.026 m`、finger joint 锚点为 `0.0844 m`，16个双刚体关节最大世界锚点误差 `3.12e-8 m`；Python 静态检查见本次交付，Isaac reset/render smoke 交由用户运行。

### 2026-08-23 CST — 固化双 GelSight 标准 Franka 组合资产

- 资产：将现有运行时组合层展开为仓库内持久 USD `franka_gsmini_standard_arm_visuals_v5.usd`，机器人配置直接加载该文件，不再写入 `/tmp/tacex_usd_cache`；原路径函数保留为兼容接口。
- 契约：GelSight 物理/碰撞、标准 arm visual、绿色底座灯及视觉 profile v5 内容不变；观测、动作、奖励、done 和 checkpoint 契约不变，现有 Teacher/Student 无需因此重训。
- 验证：持久 USD 可离线打开，默认 `/panda`、11个关键 prim、17个关节、17组 rigid/collision API 与绿色灯材质绑定均存在；改动 Python `compileall`、无运行时缓存引用检查及 `git diff --check` 通过。Isaac reset/render smoke 交由用户运行。

### 2026-08-22 CST — Pulled-Drawer 接触阈值改为 1 N

- 修改：仅在 Pulled-Drawer Teacher/Student 配置中将 `rma_contact_force_threshold_n` 从继承的 `0.2 N` 覆盖为 `1.0 N`；通用 RMA、X040、loss、网络与其余参数不变。
- 契约：Teacher 接触输入/接触奖励及 Student 接触标签同步改变；现有 Teacher checkpoint 语义不再等价，需重新训练后蒸馏 Student。
- 验证：改动环境文件快速 `compileall`、静态值检查及 `git diff --check` 通过；Isaac 接触行为交由用户运行。

### 2026-08-22 CST — 统一 Pulled-Drawer 内外缩放比例

- 修改：每个 env 从 `[0.9,1.1]` 只采样一个 isotropic scale，同时应用到外柜和内抽屉 XYZ；保留独立位置扰动、侧向间隙拒绝和 X 向零接缝约束。
- 契约：观测、动作、奖励与碰撞规则不变；几何采样和实例 hash 改变，Pulled-Drawer geometry/manifest/Student artifact 升至 v7，旧 v6 Teacher 不可迁移且需重训，Student 需重新蒸馏。
- 验证：改动文件快速 `compileall`、64-env 共享缩放/零接缝离线检查及 `git diff --check` 通过；Isaac/相机画面检查交由用户运行。

### 2026-08-22 CST — 支持保存 Student 初始相机图像

- 新增：GelSight 三帧 Student 训练入口增加可选 `--save_initial_images`，首次 reset 后为每个 env 保存一张策略相机 PNG 和一份 shape/source metadata，默认输出到本次 run 的 `initial_images/`。
- 契约：默认关闭且只进行一次观测副本写盘；不修改训练输入、模型、loss、动作、环境 reset 或 checkpoint 兼容性。
- 验证：改动脚本快速 `compileall`、参数/调用顺序静态检查及 `git diff --check` 通过；带相机实际保存交由用户运行。

### 2026-08-22 CST — 收窄 Pulled-Drawer 内抽屉颜色 DR

- 修改：内抽屉标称 RGB 改为真机近似的冷灰蓝 `(0.34,0.40,0.43)`、roughness `0.18`；Student HSV 收窄为 `H 0.52–0.62/S 0.08–0.20/V 0.30–0.55`，opacity 继续固定 `1.0`。
- 契约：外柜、Cube、物理、观测/动作、奖励和 done 不变；当前无 v6 Student 训练或 checkpoint，故保留 v6，并同步已迁移 10k Teacher manifest 的无相机外观元数据，Teacher 权重无需更改。
- 验证：改动文件快速 `compileall`、源码—manifest 静态契约检查及 `git diff --check` 通过；Isaac/相机画面对齐交由用户运行。

### 2026-08-22 CST — 迁移最新 Pulled-Drawer Teacher manifest 到 v6

- 产物：将 `20-53-02` run 的 `agent_10000.pt` 配套 manifest 从 v5 显式迁移为固定 opacity 的 v6，并在原目录保留 `.v5.backup.json`；policy 权重未修改。
- 兼容性：仅因 v5→v6 是 Teacher 不可见的材质变化而允许该次人工迁移；checkpoint 只有 10k steps，可用于启动蒸馏但不等价于完成 200k Teacher 训练。
- 验证：迁移前后 diff 仅含版本、profile、抽屉外观契约及审计字段；目标/临时文件 SHA-256 一致，原 run config hash 全部仍匹配。未启动 Isaac/Student。

### 2026-08-22 CST — 固定 Pulled-Drawer 抽屉透明度

- 修改：Teacher/Student 抽屉材质统一固定为 `opacity=1.0`；Student reset 继续随机柜体与抽屉颜色，但不再采样或写入 opacity。
- 契约：观测、动作、奖励、done 与物理碰撞不变；Pulled-Drawer geometry/manifest/Student artifact 升至 v6，旧完整 checkpoint 拒绝 resume，Teacher 需重训且 Student 需重新蒸馏。
- 验证：改动文件快速 `compileall` 与 `git diff --check` 通过；Isaac/相机画面检查交由用户运行。

### 2026-08-22 CST — 缩短默认验证流程

- 维护：只读问答不再启动测试；代理默认只执行预计 30 秒内的最小静态检查，Isaac/GPU/相机、长 pytest 与全仓测试默认提供命令交由用户运行。
- 边界：用户明确授权或无法执行必要测试时，代理再代跑最小定向集合；代码、实验契约与 checkpoint 兼容性不变。
- 验证：纯文档规则调整，仅完成内容复核与 diff 检查，未运行代码测试。

### 2026-08-22 CST — 收紧 Pulled-Drawer Cube reset 范围

- 修改：Teacher/Student 的 Cube episode reset 从相对各自抽屉中心 XY `±0.06 m` 收紧为 `±0.03 m`；八档固定尺寸与底板顶面 Z 生成规则不变。
- 契约：观测/动作、normalizer、奖励与 done 不变；初始状态分布改变，geometry/artifact 升至 v5，旧 Pulled-Drawer 完整 checkpoint 需重训/重新蒸馏。
- 验证：全仓 `compileall`、`git diff --check` 与 Pulled-Drawer 6项定向 pytest 通过；8-env GPU 连续64次全量 reset 实测最大绝对 XY 偏移为 `(0.029972,0.029838) m`，Z 生成误差为0。

### 2026-08-22 CST — 隔离 Pulled-Drawer 物理与碰撞参数

- 修改：Pulled-Drawer Teacher/Student 显式声明独立的碰撞课程、板件 contact/rest offset 与机器人 articulation solver；机器人由继承的 `8/0` 提升为本 profile 的 `32/4`，X040 与其他 GelSight task 保持原值。
- 契约：观测、4维动作、奖励权重与 done 阈值数值不变；geometry/artifact 升至 v4 并记录 solver/offset，旧 Pulled-Drawer 完整 checkpoint 拒绝加载且需重训/重新蒸馏，model/normalizer 仍为 v3。
- 验证：全仓 `compileall`、`git diff --check`、Pulled-Drawer 6项定向 pytest 与8-env GPU runtime/contact smoke 通过；运行时 solver 为 `32/4`，下压在约29.6 N首次报告接触并于220.1 N触发200 N终止。完整及两项 X040 pytest 均在无输出下超时，未计为通过；Pulled 定向测试已断言 X040 配置仍为 `8/0`。

### 2026-08-22 CST — 建立双 GelSight 开口 Pulled-Drawer Teacher–Student 契约

- 新增：按 `occluded_grasping` 拓扑建立白色四板外柜与透明五板开口抽屉、启动前固定几何 DR、八档均衡 Cube，以及独立 Teacher/三帧 Student 训练链路；Cube 位于抽屉内部底板，抽屉后沿与柜体前沿零缝连接。
- 契约：九块板均为带摩擦、接触报告和启动校验的 kinematic rigid collider；七路 Student 输入、三路输出、4维动作、奖励及 success/done 不变。位置归一化改为 `[0.5275,0,0.033]/[0.11,0.08,0.10]`，geometry/model/artifact 升至 v3，旧完整 checkpoint 拒绝加载并需重训/重新蒸馏。
- 验证：全仓 `compileall`、Pulled-Drawer 6项定向 pytest、4096-env采样检查及 GPU Teacher 8-env reset/step 通过；九块板各创建8个 PhysX刚体、接缝误差约`0.00006 mm`，机械臂下探测得最大抽屉接触力`171.6 N`。带相机 Student smoke 未运行。

### 2026-08-22 CST — 修复抽屉 Student 回放模块导入

- 修复：抽屉回放脚本与定向测试改为导入实际的完整环境模块名，避免启动时报 `ModuleNotFoundError`。
- 契约：仅修复 Python 导入路径；task、观测、动作、奖励、done 和 checkpoint 格式均不变，旧 checkpoint 无需重训。
- 验证：改动文件 `compileall`、`git diff --check` 与静态引用检查均已通过；Isaac/GPU 回放未运行。

### 2026-08-18 CST — 新增 GelSight X040 抽屉零样本 demo task

- 新增：在 X040 三帧 Student 场景内加入固定抽屉底板、侧壁、后壁和前板，并注册独立 task 与跨场景回放脚本。
- 契约：Student 七路 runtime 输入、4 维动作、原奖励/成功保持不变；抽屉尺寸为待真机标定的近似，旧 checkpoint 仅作显式零样本评估，不能视为已训练抽屉策略。
- 验证：新增文件 `compileall` 通过；IsaacLab 定向 pytest 因当前容器缺少 `omni.log` 未完成，需在 Isaac Lab 运行时复验 reset/step 与成功率。

### 2026-08-18 CST — 新增 GelSight X040 抽屉 Teacher 训练 task

- 新增：注册抽屉 privileged Teacher、独立 PPO YAML，并让 X040 manifest 记录抽屉 task 与几何契约。
- 契约：Teacher 仍为 30 维 privileged actor、4 维动作和原成功语义；抽屉 Teacher checkpoint 与平面 Teacher checkpoint 互不兼容，Student 蒸馏需使用配套抽屉 task。
- 验证：改动文件 `compileall` 与 `git diff --check` 通过；Isaac reset/step smoke 因当前容器无 CUDA 未完成。

### 2026-08-19 CST — 新增抽屉 Teacher→Student 串行训练脚本

- 新增：自动以 Teacher `1024` env 完成训练，定位成功生成的 checkpoint 后再以 Student `128` env 启动蒸馏；Teacher 失败时 Student 不启动。
- 契约：训练 task、Teacher manifest、Student encoder 初始化 checkpoint 均显式传递，默认不覆盖已有日志；可选 `--start-at HH:MM` 延迟启动。
- 验证：脚本 `bash -n`、帮助输出、非法 env 数量拒绝、全仓 `compileall` 和 `git diff --check` 已运行；实际 GPU 训练未运行。

### 日志记录规则

每条 3–5 行，写清「改了什么 / 契约与兼容性影响 / 验证情况」。不得把未运行的训练或评估写成结果；
不确定的作者、commit、seed、checkpoint 或数值写「待确认」。涉及观测、动作、奖励、done 或 checkpoint
兼容性的改动必须显式说明。更早的记录见 `archive/DEVLOG-2026H1.md`。

### 2026-08-15 CST — 新增 GelSight 接触/未接触 RGB rollout 诊断

- 新增：X040 三帧 Student 可直接加载 checkpoint 跑 rollout，按物理左右接触标签及 Taxim `shifted_height_map < 0` 像素覆盖率分组，输出逐侧 NPZ、汇总 JSON 和 reference/current/delta 代表帧。
- 契约：仅新增只读诊断入口；观测、动作、奖励、done、训练与 checkpoint 格式均不变，旧 checkpoint 可直接测试且无需重训。
- 验证：改动文件 `compileall`、diff check 与离线汇总定向 pytest（2项）通过；当前 Codex 容器未挂载 NVIDIA 设备，完整带相机 rollout 待在 GPU 终端运行。

### 2026-08-15 CST — 导出 GelSight X040 Student 30000步 TorchScript

- 产物：将 `2026-08-15_14-38-57_distillation/checkpoints/student_0030000.pt`（v6/model-v3，`global_step=30000`）导出为 TorchScript 与契约 JSON；因沙箱将日志实际根 `/data2/skrl` 挂载为只读，暂存于 `exports/0815_gelsight_30000/`，待移入原 checkpoint 目录。
- 契约：七路输入及 `action[4]/contact[2]/cube_position_root_m[3]` 输出不变；仅生成部署产物，不修改训练、环境、奖励或 done，也不需要重训。
- 验证：CPU batch 1/8 eager—TorchScript 三输出最大绝对误差均为0，落盘重载 batch-8 shape/finite 检查通过；当前沙箱未暴露 CUDA，JSON 中 `cuda_validation=false`。

### 2026-08-15 CST — 导出最新 GelSight X040 Student 10000步 TorchScript

- 产物：将 `2026-08-15_14-38-57_distillation/checkpoints/student_0010000.pt`（v6/model-v3，`global_step=10000`）导出到仓库 `exports/`，并生成含输入输出契约与 SHA-256 的同名 JSON。
- 契约：七路输入及 `action[4]/contact[2]/cube_position_root_m[3]` 输出不变；仅生成部署产物，不修改训练、环境、奖励或 done，也不需要重训。
- 验证：相关脚本 `compileall`/diff check、CPU batch 1/8 eager—TorchScript（误差均0）及落盘重载通过；沙箱未暴露 CUDA，定向 Isaac pytest 静默超2分钟后中止，JSON 记 `cuda_validation=false`，待目标部署机复验。

### 2026-08-15 CST — GelSight X040 底座状态灯改为绿色

- 修改：组合 USD 仅将 `panda_link0` 标准 visual 设为可覆盖，并把底座 `subset_5` 绑定为固定纯绿色 UsdPreviewSurface 自发光材质；link1–7 继续实例化，腕部蓝灯及 GelSight 外观不变。
- 契约：只改变相机 RGB 像素；观测 key/shape、4维动作、奖励、done、坐标系和物理参数不变。Student checkpoint 升至v6；旧v5可play/export但不可resume，正式对齐实验需重新蒸馏；Teacher 无视觉输入，无需重训。
- 验证：改动文件 `compileall` 与 `git diff --check`、12项X040定向pytest及8-env带相机 reset/step smoke 通过；确认底座材质的 diffuse/emissive 均为 `(0,1,0)`、腕部仍绑定 `EmissiveBlue`。

### 2026-08-15 CST — 对齐 GelSight X040 与 0812 视觉场景

- 修改：X040 Teacher/Student 恢复可见黑色共享 GroundPlane，并把 floor/backdrop 扩到完整3.5 m env cell；新增缓存组合 USD，停用旧 arm link0–7 visual 并独立引用标准 Panda visual，hand/finger/GelSight 保持原资产。
- 契约：RGB 像素分布改变，Student artifact 新增视觉 profile；观测 key/shape、4维动作、奖励、done、坐标系、GelSight 额外 prim 和机器人自碰撞均不变。旧 v5 Student 可回放但正式实验应重新蒸馏，Teacher 无视觉输入且物理未变。
- 验证：相关模块 `compileall`、`git diff --check`、12项X040定向pytest与8-env Student reset/step及RGB抓帧通过；组合USD确认每个arm link仅一套标准mesh、材质有效，原hand/finger和GelSight prim存在且自碰撞仍开启。

### 2026-08-14 CST — 修复 GelSight Student 重复渲染与相机位姿回读

- 修改：三个 GelSight config mixin 显式转发 `__post_init__`，恢复 `render_interval=decimation=2`；RMA 的无 marker depth camera 关闭 latest-pose 回读，并以固定 `[0,1]→uint8` scale 移除每步两次 GPU 同步。
- 契约：观测 key/shape、4维动作、奖励、done、触觉像素换算与机器人自碰撞均不变；X040 Student artifact 新增渲染节奏、pose 开关和转换尺度，旧权重可评估但不建议跨修复 resume。
- 验证：全仓 `compileall`/diff check、通用与X040配置契约 pytest、三个配置族实例化检查、X040最小合法8-env相机 reset/step 和触觉 uint8 范围 smoke 通过；Size-Buckets用例仍被既存10/10000 N断言阻断，未修改 Taxim 早退。

### 2026-08-14 CST — GelSight X040 Student 对齐位置归一化与视觉初始化

- 修改：三帧 Student 的 Cube XYZ 改为 X040-Wide 中心/尺度 `[0.40,0,0.026]/[0.08,0.10,0.10]`，训练必须显式传入经校验的 RMA XY Heatmap-DR checkpoint 初始化 ResNet18。
- 契约：七路输入、三路输出、4维动作、Teacher、奖励与 success/done 不变；Student model/checkpoint 升至 v3/v5（v4 已撤回，不复用），新实验必须重新蒸馏，旧 v3 checkpoint 仅保留回放/导出兼容且禁止 resume。
- 验证：待完成全仓 `compileall`、GelSight X040 定向 pytest、旧 checkpoint 回放兼容检查及带相机最小训练 smoke。

### 2026-08-14 CST — 撤回 X040 GelSight Student 底座 LED 外观 DR

- 修改：移除会在场景创建时对 GelSight robot instance 执行 `make_uninstanceable` 和 stage-local 材质绑定的 LED DR；该路径使 Student 在训练日志创建前直接退出。
- 契约：Student 恢复静态底座视觉，runtime 输入 key/shape、4维动作、奖励、done 和 Teacher manifest 均不变；Student checkpoint 回到 v3，未完成的 v4 产物不得恢复或部署。
- 验证：使用 Teacher `agent_60000.pt` 的带相机 Student 8-env/1-step smoke 与配置契约 pytest 已通过；完整带相机场景 pytest 待 Teacher 训练结束后运行，避免争用 GPU。

### 2026-08-14 CST — X040 GelSight 改为逐 env 固定单 Cube

- 修改：Teacher/三帧 Student 每个 env 仅创建一个 Cube，启动前按 `env_id % 8` 等量绑定 4–6 cm 尺寸，reset 不换尺寸；恢复 GPU dynamics，并以 3.5 m spacing、隐藏共享 GroundPlane 和显式碰撞过滤隔离 3 m 木板。
- 契约：物体数从8降为1、尺寸时序从逐回合随机改为逐 env 固定；观测/动作、奖励、碰撞阈值课程、success/done 和关节 reset 噪声不变。Teacher manifest/Student checkpoint 升至v3，旧错误分布 checkpoint 拒绝加载，必须重训/重新蒸馏。
- 验证：全仓 `compileall`/diff check、配置/artifact 定向 pytest、Teacher+Student 8-env reset/step 与局部 reset 固定尺寸 smoke、Teacher 8-env/256-step PPO 通过；1024-env 吞吐 benchmark 未运行。

### 2026-08-14 CST — 修复 X040 GelSight RMA 第 200 步奖励汇总崩溃

- 修改：X040 unified collision profile 增加专用 RMA 汇总格式，读取实际存在的 illegal-collision reward、惩罚/终止阈值和最大力，不再读取已被该 profile 替换的 `reward/table_collision`。
- 契约：仅修复控制台日志；观测、4维动作、奖励值/权重、success/done、阈值课程和 manifest/checkpoint 均不变，旧 checkpoint 可加载且无需重训。
- 验证：全仓 `compileall`、`git diff --check`、定向 pytest，以及 X040 GelSight Teacher 8-env/256-step PPO smoke 均通过；step 200 正常打印并继续至结束。

### 2026-08-14 CST — 修复 privileged/RMA root-frame 观测设备不一致

- 修改：privileged 基类在 world→robot-root 平移前，将 `scene.env_origins[N,3]` 对齐到 Cube state 的 device/dtype，修复 CPU PhysX origins 与 CUDA asset state 相减导致的首次 reset 失败。
- 契约：观测 key、维度、数值与坐标系不变；动作、奖励、success/done、manifest/checkpoint 均不受影响，旧 checkpoint 可继续加载且无需因此重训。
- 验证：全仓 `compileall`、`git diff --check`、privileged 定向 pytest，以及 GelSight Teacher 1-env/128-step PPO smoke 均通过。

### 2026-08-14 CST — GelSight 夹爪中心、最低端与桌面安全契约

- 修改：所有已注册 GelSight RMA task 统一以 `panda_hand +Z 0.1182 m` 作为左右 GelPad 中心，以 `+Z 0.1353 m` 作为 `panda_fingertip_centered` 近似最低端；IK、reach、critic 和 Teacher FK 同步切换。
- 安全：最低端距板面严格小于 `0.010 m` 时每个 policy step 额外 `-10`；仅增加奖励，不截断动作，既有非法接触惩罚和 done 保持独立。
- 兼容：GelSight Teacher manifest、Size-Buckets/X040 Student artifact 与三条 Teacher YAML 升为新几何契约；旧 GelSight Teacher/Student checkpoint 拒绝加载，必须重训和重新蒸馏，非 GelSight task 不受影响。
- 验证：全仓 `compileall`、diff 检查、GelSight 配置/FK/阈值定向 pytest、单环境 Teacher 接触与几何 smoke 已运行；共享的 CPU origins/CUDA state reset 问题已修复，普通 GelSight Teacher 与 X040 Teacher 短 PPO 均通过。

### 2026-08-14 CST — 新增 GelSight X040-DR 三帧 Teacher/Student 实验族

- 修改：新增配套 Teacher 与三帧 Student task；复用 X040 的 4–6 cm 八 Cube reset 选桶、`x=[0.32,0.48] m/y=[-0.10,0.10] m`、相机 DR 与停车策略，并叠加 GelSight reference-delta 输入和左右接触头。
- 契约：Student runtime 输入为三帧 RGB、proprio/history、左右 current/reference tactile；输出 `action[4]`、左右接触概率 `[2]` 和 root 米制 Cube XYZ `[3]`。位置头以无界归一化 XYZ 表示 X040 的 `x=0.32 m` 下界；Student model v2，旧 checkpoint 不兼容且需重训。
- 安全：非法碰撞惩罚阈值在 0–100k policy step 从20 N线性收紧至5 N（每步-10）；终止阈值从200 N收紧至20 N，ground collision 仍终止、timeout 仅 truncated。
- 验证：相关模块 `compileall` 与 `git diff --check` 通过；Isaac 下 task 注册/配置契约与 TorchScript 三输出隔离 pytest、Teacher 8-env/256-step PPO smoke 通过。完整定向 pytest 曾在8分钟无输出后中止，Student 8-env smoke、训练、回放和导出仍待目标 GPU 主机复验。

### 2026-08-14 CST — 新增 GelSight Teacher 键盘几何检查脚本

- 修改：新增单环境 GUI 键盘遥操作脚本，沿用 Teacher 的 4 维 Cartesian action，并输出双指尖中点与由 PhysX body pose 变换的 GelSight case/gelpad 局部包围盒最低角点。
- 契约：不修改 task 注册、观测、动作维度、奖励、done、资产或 checkpoint；Teacher 默认不渲染 GelSight tactile RGB。
- 验证：脚本 `compileall` 与 `git diff --check` 通过；GUI Isaac smoke 待在带图形界面的 Isaac Lab 会话中人工运行。

### 2026-08-13 CST — GelSight Size-Buckets Student 改为视觉触觉连续融合

- 修改：腕部 ResNet18 GAP 保留512维，左右 reference-delta GelSight 共享 CNN 各输出256维，与15维 proprio/4维 history 拼成1043维直接动作输入；XYZ、14×14 heatmap、左右接触及 Teacher action 提供辅助/蒸馏监督。
- 契约：环境七输入 key/shape、4维动作、奖励/done 和 Teacher manifest 不变；Student checkpoint/model 升至v3/v2，旧 XYZ/硬接触瓶颈 Student 不兼容并需重新训练、重新导出。
- 验证：全仓 compileall/diff check、4项新架构 pytest、batch-2 eager/TorchScript、8-env 1-update训练、resume至step 2、1-step play及CPU/CUDA batch 1/8导出通过；完整测试在Teacher smoke后连续创建Student场景超过10分钟未返回而中止，另有既存10 N断言与当前10000 N配置不一致。

### 2026-08-13 CST — 新增 GelSight 固定尺寸与参考触觉 RMA 路线

- 修改：新增固定按 `env_id % 8` 分配 4–6 cm 单 Cube 的 GelSight Teacher/Student-DR task；正接触仅认左右 gelpad，`illegal_collision` 惩罚阈值在0–100k policy step从20 N线性降至5 N，触发仍单步 -10，`>10 N` terminated，5 s timeout 仅 truncated。
- 契约：Student 增加左右 current/reference 触觉输入，逐 env reset 后首帧独立锁存，模型使用 `(current-reference)/255` 和硬二值接触；动作 `[4]`、Teacher 30维 Actor 和惩罚权重不变。新 manifest/checkpoint 升至v2并与旧 GelSight checkpoint fail closed，Teacher/Student 均需重训。
- 验证：全仓 `compileall`/diff check、8项离线契约测试、Teacher/Student 各8-env reset/step 与局部 reference reset、旧 GelSight Student 回归测试通过；另以 v2 契约完成128步 Teacher、断点课程 offset 续训、2-update Student、2-step play 和 CPU/CUDA batch 1/8 TorchScript smoke。正式训练与策略性能评估未运行。

### 2026-08-13 CST — 导出三份 X040-Wide Student step 100000 TorchScript

- 产物：分别从三个 run 的不可变 `student_0100000.pt` 导出单帧、三帧 Appearance 和普通三帧 TorchScript/JSON 到各自 `checkpoints/exported/`；第三个实验族唯一 run 为 `2026-08-12_02-09-40_distillation`。
- 契约：单帧输入为 `RGB[224,224,3]+proprio[15]+history[4]`，两份三帧输入为 `RGB history[3,224,224,3]+proprio[15]+history[4]`，输出均为 `action[4]`；旧 Appearance v1 产物在 JSON 标记为仅评测兼容，无需重训但不可视为当前 v5 视觉分布的正式指标。
- 验证：三份源 checkpoint 与 `latest.pt` 模型哈希一致；script/reload 最大误差均为 0，JSON/source/artifact SHA-256 复核及 CPU batch 1/3 推理通过；全仓 `compileall` 与三帧 artifact/Appearance 兼容定向 pytest 2 项通过。

### 2026-08-13 CST — Appearance 改为逐环境固定单 Cube 分桶

- 修改：Appearance task 每个 env 只创建一个 Cube，按 `env_id % 8` 等量绑定 4–6 cm 尺寸且 reset 不换桶；环境数须为 8 的倍数，三帧 train/play 默认改为 8 env，跨 env 碰撞显式隔离，并恢复 PhysX GPU dynamics。
- 契约：动作/观测 key 与维度、奖励、success/done、坐标系不变；物体数量、尺寸时序分布和物理解算后端改变。旧 Appearance v1 Student 可仅用于 rollout，正式新分布评测/训练应生成新 checkpoint，resume 仍 fail closed。
- 验证：全仓 `compileall`、8-env 定向 pytest（USD/物理尺寸、GPU dynamics、局部 reset/材质隔离）及旧 checkpoint 的 16-env/170-step headless 审计通过（总墙钟 23.75 s，含 Kit/场景启动）；3000-step GUI 与真机评估未运行。

### 2026-08-13 CST — Appearance 木板与背景板空间隔离

- 修改：Appearance 专用 env spacing 从 1.5 m 增至 3.5 m，使 3×3 m 木板间保留至少 0.5 m 间隙；全局 GroundPlane 保留碰撞但强制不可见，消除相邻板重叠和共享网格背景。
- 契约：每 env 精确拥有一个 `floor_panel` 和一个 `real_alignment_backdrop`，启动时对尺寸/间距/全局地面可见性 fail closed；动作、观测维度、奖励、done/success、Cube 分桶和坐标系不变，Appearance profile 升至 v4。
- 验证：8-env 实际 USD 路径、origin 非重叠、GroundPlane visibility、材质/reset 隔离定向 pytest 通过；GUI 俯视人工检查和 3000-step 评测未运行。

### 2026-08-13 CST — 恢复 Appearance episode 最长时间

- 修改：恢复既有 `episode_length_s=5.0`（30 Hz 下 150 policy steps）上限；真实机器人碰地返回 `terminated`，达到上限独立返回 `truncated`，两者都会只 reset 对应 env，success 仍非终止。
- 契约：Appearance profile 升至 v5；观测、动作、奖励、Cube 分桶和坐标系不变，done/外观刷新分布改变。旧 v1 Student 仍仅允许 rollout，正式指标需重新测量。
- 验证：8-env 定向 pytest 强制仅 env 0 timeout，确认只重置其计时器/颜色；旧 checkpoint 的 16-env/170-step 审计在 step 149 达到现有 horizon 并重置全部同期 env。3000-step 与真机评估未运行。

### 2026-08-13 CST — 重建逐环境独立 Appearance task

- 修改：新增独立 PreviewSurface 实现，每个 env 只保留木板/背景/方块三个独立材质；移除 MDL、纹理流送、预热和共享光照随机化，局部 reset 不触碰其他 env。
- 契约：profile 升至 v3并禁用固定 150-step timeout，只保留逐 env 碰地 done；动作 `[4]`、三帧观测维度、奖励、success 和物理不变。旧 v1 Student 可显式 rollout，视觉/终止指标不可与旧实验直接横比，训练 resume 不兼容。
- 验证：全仓 `compileall`、Appearance 契约/实际 USD 绑定/局部 reset 隔离/RGB 变化定向 pytest 通过；16-env/170-step GUI 审计跨过原 step 149 边界，未发生 reset 或材质变化。3000-step 与真机评估未运行。

### 2026-08-13 CST — Size-Buckets 禁用 PhysX GPU dynamics 以兼容 reset

- 修改：在生成八个动态 Cube 前关闭本 task 的 PhysX GPU dynamics，消除 Direct GPU API 场景禁止的 `PxRigidDynamic::set*Velocity` reset 调用；Appearance-DR GUI 回放默认不再无限等待纹理流送，CUDA 策略推理和相机渲染保持启用。
- 契约：active/parked Cube、尺寸采样、观测、动作、奖励、success/done 与 checkpoint/manifest 格式不变；回放首帧可能尚未完成纹理流送，刚体求解后端改变，旧 checkpoint 可加载但正式评测指标需重新测量。
- 验证：全仓 `compileall`、静态建场景顺序检查、4-env Size-Buckets 定向 pytest 已运行；16-env/3000-step Student checkpoint 回放待在目标 GPU 主机复验无 Direct GPU 报错且出现 rollout 进度日志。

### 2026-08-12 CST — 新增 X040 三帧外观随机化环境

- 修改：新增独立 Appearance-DR task；木板、背景和当前活动方块在每个 env reset 时从受控 PreviewSurface/MDL 池独立采样并保持整回合，MDL 缺失时 fail closed；训练、回放与三帧 TorchScript 导出链路同步支持新 task。
- 契约：三帧 RGB、proprio 15 维、history/action 4 维、Teacher、奖励、success/done 与物理不变；旧三帧 checkpoint 继续只加载旧 task，禁止跨 task resume，新外观 Student 复用 Teacher/encoder 初始化但需重新训练。
- 验证：全仓 `compileall`、X040/Size-Buckets/三帧/DR 定向 pytest、全部材质实际 USD 绑定与 RGB 变化、4-env/2-step 训练、checkpoint reload、1-env/2-step 回放和 batch 1/2 TorchScript script/reload smoke 通过；正式训练与真机评估未运行。

### 2026-08-11 CST — 新增训练进程完成后串行启动器

- 修改：新增通用 `run_after_process.py`，绑定外部 PID/start-time 并轮询；仅在进程退出且指定最终文件存在时，以无 shell 的参数列表启动下一条命令。
- 契约：不停止或修改被等待进程，不改变训练 task/超参/checkpoint；缺少最终文件、PID 命令不匹配或调度器被中断时 fail closed，不启动后续训练。
- 验证：脚本 `compileall`、相关 diff 检查和离线成功/缺失完成文件路径 pytest 2 项通过；未运行实际长训练串联。

### 2026-08-11 CST — 新增 Size-Buckets 三帧视觉 Student

- 修改：新增独立三帧 Student task、reset-safe 30 Hz RGB history、共享 ResNet `3×512→512` 融合模型、fail-closed artifact 及训练/评测入口；单帧 task 保持不变。
- 契约：运行时视觉输入变为 `[3,224,224,3]` oldest→newest，动作头仍为 531 维且动作 `[4]`、Teacher、奖励与 done 不变；单帧 checkpoint 不兼容，新 Student 需重新蒸馏。
- 验证：相关 `compileall`/diff 检查、定向 pytest 3 项、2-env/2-update 蒸馏与保存后 2-step 回放、RTX 3090 的 32-env/1-update smoke 通过；正式训练、真机时序与 TorchScript 导出未运行。

### 2026-08-11 CST — X040-Wide Student 记录最近窗口成功率

- 修改：蒸馏训练每个日志周期将最近 200 个 policy step 内已完成 episode 的成功率写入 TensorBoard `Performance/recent_success_rate`。
- 契约：仅增加训练指标，不改变成功判定、观测、动作、奖励、done、张量维度或 checkpoint 格式；旧 checkpoint 兼容且无需重训。
- 验证：改动脚本 `compileall` 与相关文件 `git diff --check` 通过；仅增加标量日志，未运行 Isaac 训练 smoke。

### 2026-08-11 CST — 新增 Size-Buckets Teacher→Student 自动训练流水线

- 修改：新增独立串联脚本；Teacher 入口按需显式保存 `agent_<trainer_timesteps>.pt`，流水线确认进程成功、唯一新 run、manifest 和最终文件后，才选择 `best/final` checkpoint 启动 Student；视觉 encoder checkpoint 必须显式传入。
- 契约：复用现有 Size-Buckets task、训练器和 artifact 校验，不改观测、动作、奖励、done、张量维度或 checkpoint 格式；旧 checkpoint 兼容且不要求重训。
- 验证：全仓 `compileall`、`git diff --check`、离线流水线 pytest 4 项通过；2-env Teacher 1 次 PPO（128 步）→2-env Student 2 update 端到端 smoke 自动切换并生成两个最终 checkpoint。正式训练未运行。

### 2026-08-11 CST — 将 Size-Buckets 停车位移到所有腕部相机后方

- 修改：停车区局部 X 根据 env-origin 的 X 跨度、腕部相机 X 上界和 `0.5 m` 安全裕量动态计算，使所有非选中 bucket 位于全体相机后方；不修改 USD visibility 或 PhysX 属性。
- 契约：仅修正 Student `wrist_rgb` 分布，动作、观测 shape、选中方块物理、奖励与 done 不变；Teacher checkpoint 兼容，旧 Size-Buckets Student smoke 缺少安全停车契约，需重新蒸馏。
- 验证：全仓 `compileall`、`git diff --check`、artifact/config 定向 pytest、4-env Student parking/reset/step pytest 与 RGB 前后对比、2-env Teacher parking/reset/step smoke 已通过；正式 Student 训练和真机评估未运行。

### 2026-08-11 CST — Size-Buckets reset 加入 Franka 初始关节角噪声

- 修改：仅 Size-Buckets Teacher/Student 在每次 reset 为 7 个臂关节采样 `N(0, 0.01²) rad` 并截断至 `±0.03 rad`；夹爪角与全部关节速度仍按标定默认值/零值重置。
- 契约：动作、观测 key/shape、奖励、success/done 不变，但初始物理状态与 `proprio_obs` 分布改变；Size-Buckets contract 升为 v2，旧 smoke Teacher/Student checkpoint 不兼容并需重训/重新蒸馏，原 X040-Wide task 不受影响。
- 验证：全仓 `compileall`、`git diff --check`、Size-Buckets contract 定向 pytest、4-env Student reset/step pytest 与 2-env Teacher reset/step smoke 已通过；正式训练和真机评估未运行。

### 2026-08-10 CST — 新增 X040-Wide 4–6 cm 八尺寸桶 Teacher/Student

- 修改：新增独立 Teacher 与 Student-DR task；每个环境预生成 8 个等间隔 `4–6 cm` 物理方块，reset 均匀选择一个目标，其余方块停放在相机后方；8 个 one-cube/two-finger sensor 只汇总当前桶接触力。
- 契约：动作 `[4]`、Teacher 28 维输入、Student 三运行时输入、奖励权重和 done/success 不变；cube XYZ 的 Z 随尺寸为 `0.021–0.031 m`。新 task/manifest 与旧 checkpoint fail-closed 隔离，Teacher 需重训，Student 需从新 Teacher 重新蒸馏。
- 验证：全仓 Python `compileall`、`git diff --check`、4-env Student 定向 pytest、2-env Teacher reset/step、64-env/1-iteration Teacher PPO+manifest，以及 4-env/2-update Student 蒸馏与 TorchScript 导出 smoke 已通过；正式 100k Teacher/Student 训练和真机评估未运行。

### 2026-08-09 CST — 重新导出 X040-Wide Student step 30000 TorchScript

- 产物：从不可变 `student_0030000.pt` 导出 `checkpoints/exported/rma_x040_wide_student_0030000.pt`（47,301,257 bytes）及同名 JSON；训练仍在继续，未以可变 `latest.pt` 作为最终 provenance。
- 契约：部署输入仍为 `wrist_rgb[224,224,3] + proprio[15] + history[4]`，输出 `action[4]`；观测、动作、奖励、done 及 Student v3 契约不变。
- 验证：script/reload 与 eager 最大误差均为 0；CPU batch 1/3 推理通过，TorchScript SHA-256=`79aea3ec...b23538`，源 checkpoint 与产物 hash 均匹配 JSON。

### 2026-08-09 CST — 导出 X040-Wide Student step 10000 TorchScript

- 产物：从 `2026-08-09_20-37-53_distillation/checkpoints/latest.pt` 导出 `checkpoints/exported/rma_x040_wide_student_latest.pt`（47,301,020 bytes）及同名 JSON provenance。
- 契约：部署输入严格为 `wrist_rgb[224,224,3] + proprio[15] + history[4]`，输出 `action[4]`；训练期 cube XYZ/位置头不进入 runtime，源 checkpoint 与 artifact 契约不变。
- 验证：script/reload 与 eager 最大误差均为 0；CPU batch 1/3 推理输出有限且位于 `[-1,1]`，TorchScript 与源 checkpoint SHA-256 均匹配元数据。

### 2026-08-09 CST — X040-Wide Student 底座 LED HSV 随机化

- 修改：底座 LED 改为每 env 独立材质，并在每个 episode 按 H=`105°–135°`、S=`0.75–1.0`、V=`0.35–1.0` 采样；episode 内稳定，腕部蓝灯不变。
- 契约：仅改变 `wrist_rgb` 像素分布，不改 key/shape、动作、奖励或 done；Student artifact 升至 v3，v2 及更早 Student 拒绝加载并需重新蒸馏，Teacher/encoder checkpoint 不受影响。
- 验证：全仓 `compileall` 与 X040-Wide 定向 pytest 已通过；真实 Teacher/encoder checkpoint 的 4-env、2-update smoke 成功，并确认产物为 v3 且记录完整 HSV 契约。

### 2026-08-09 CST — 新增 X040-Wide XYZ/no-contact RMA profile

- 类型：RMA task / Teacher-Student 蒸馏 / 安全奖励 / artifact 契约。
- 修改：新增独立 X040-Wide Teacher 与 direct-action Student-DR：cube 标称 `(0.40,0,0.026)`、reset `x±8 cm/y±10 cm`、无位置 curriculum；视觉 DR 相机平移 `±1 cm`、旋转 `±2°`。Teacher 输入改为 proprio、历史动作、cube XYZ，不再输入接触力；Student 从 RGB ResNet GAP 特征预测 cube XYZ 并加入归一化 MSE。
- 奖励/兼容性：TCP 低于 world z=`0.011 m` 每步额外 `-10`，不改 done 或动作尺度；接触仅保留为仿真奖励内部信号。profile 使用独立 task、manifest、checkpoint kind/version，旧 XY/force Teacher 与 Student 均拒绝加载，必须重训。
- 验证：已运行全仓 `compileall`、diff 检查及 X040-Wide 定向 pytest；正式 100k Teacher/Student 训练与真机部署未运行。

### 2026-08-09 CST — X040-Wide Student 底座状态条改为绿色

- 修改：修正 Panda instanceable USD 层级，Student-DR 仅将底座 visual 分支去实例化，再把 `panda_link0/visuals/panda_link0/subset_5` 绑定为绿色自发光材质；腕部 LED、Teacher 与官方 USD 不变。
- 契约：Student artifact 升至 v2，并记录该视觉覆写；旧 v1 Student 因 RGB 域变化被 fail-closed 拒绝，必须用绿色图像重新蒸馏。Teacher checkpoint 不受影响。
- 验证：已运行全仓 `compileall`、X040-Wide 定向 pytest；带相机的 1-env reset/step 确认底座绿色且腕部仍蓝色，并以真实 Teacher/encoder checkpoint 完成 4-env、2-update 蒸馏 smoke。

### 2026-08-09 CST — Direct-Action Student 回放支持 MP4

- 类型：回放工具
- 修改：`rma_direct_action_student/play.py` 新增单环境 `--video --video_length`，通过 Gym `RecordVideo` 写入第三视角 MP4；`--input_video` 则逐帧写出 Student 真正消费的 `wrist_rgb[224,224,3]`。
- 验证情况：见本次任务汇报；不影响 Student 输入、动作、奖励、done 或 checkpoint 契约。

### 2026-08-09 CST — 新增 PandaHand RMA Direct-Action Visual Student

- 类型：RMA Student 模型 / 蒸馏 / 评估 / 导出 / rollout / task 注册
- 修改：新增独立 direct-action Student，以现有 Heatmap-DR Student 的 ResNet18 encoder warm start；部署只接收 RGB、本体15维和历史动作4维，经 512+15+4 特征直接输出4维动作，不预测位置或接触。
- 兼容性：Teacher 仍在仿真中用 cube XY 与双侧接触力生成动作标签，但这两个输入不进入 Student/runtime rollout；新增专用 artifact kind/version，旧 XY/force Student 不能 resume/export，Direct-Action Student 必须重新训练。奖励、done、物理和动作尺度不变。
- 验证情况：见本次任务汇报；未把 smoke、正式训练、评估或真机部署结果写为已完成实验。

### 2026-08-08 CST — PandaHand RMA 改为 XY 视觉定位与 1 N 双侧力抓取

- 类型：RMA Teacher/Student 观测、奖励、蒸馏与部署契约
- 修改：PandaHand task 改用独立 XY/force 实现：Actor 为26维，Student 仅从 RGB 预测 cube XY；每策略步读取左右方块接触力最大值，双侧均 `>=1 N` 才生成单一 `grasped` 特征并获得接触奖励。
- 兼容性：Teacher manifest/Student checkpoint、训练/回放/评估/导出接口仍接收 `contact_force_n[2]`，但模型版本升级为 bilateral grasped[1]；此前 PandaHand XY RMA artifact 同样必须重训。GelSight task 配置与旧 artifact 契约未改。
- 修改：新增仅回放的 Legacy Heatmap-DR task 与采集器 v5 分支，用于归档的 RGB/XYZ/视觉接触 30 维 PandaHand Student；新旧 artifact 在 kind、版本、输入与 task ID 上 fail-closed 隔离。
- 验证：待本次定向 RMA pytest 与 legacy checkpoint smoke；当前 XY/force v2、训练入口及 GelSight 配置不改变。
- 验证情况：见本次任务汇报；未将未执行的训练或真机结果记为实验结论。

### 2026-08-08 CST — 新增 D435 标定四 GelSight 抽屉 task

- 类型：抽屉视触环境 / 相机标定 / task 注册
- 修改：新增 `TacEx-Drawer-GelSight-Four-Tactile-D435-v0`，继承原四 GelSight 抽屉场景并保留其几何、5维动作、奖励和 done；第三视角改为 Real-Alignment D435 的 224×224 内外参与 GPU 内参补偿。
- 兼容性：观测 key 和特征维度保持 `third_resnet[512]` 加四路 tactile 各 `[256]`，但 RGB 几何/分布改变；旧抽屉 checkpoint 不作为新 task 的兼容模型，需重新训练。验证情况见本次任务汇报。

### 2026-08-08 CST — 新增 RMA Student rollout 数据采集器

- 类型：RMA Student 分析工具 / 数据产物 / 测试
- 修改：新增 `collect_rma_student_rollouts.py`，以 `--run TASK_ID=CHECKPOINT` 批量回放匹配的 Student artifact；每条 episode 写入压缩 NPZ、`episodes.csv` 和 manifest，包含第三视角/可用 GelSight 图像、真值标签、Student 预测与动作、oracle Actor action、奖励、接触/碰撞诊断及 DR episode 参数；输出 checkpoint 校验、episode 开始/保存和任务完成 INFO 进度。
- 兼容性：脚本严格校验 task、Student checkpoint 与 Teacher manifest live environment contract；不改 observation、4维动作、奖励、done 或 task 注册，跨 PandaHand/GelSight、Clean/DR artifact 仍 fail closed，旧兼容 checkpoint 无需重训。
- 验证情况：已执行脚本/测试 `compileall`、collector 参数与 NPZ 对齐定向 pytest，以及 1-env GelSight Heatmap-DR Student 真实采集 smoke；未执行正式训练或真机部署。

### 2026-08-08 CST — 新增 RMA rollout 图像/视频导出器

- 类型：RMA Student 分析工具 / 数据产物
- 修改：新增 `export_rma_student_rollout_media.py`，从单条 rollout NPZ 只导出一个同步诊断 MP4：第三视角与可用左右 GelSight 拼接在上半部，接触力、合并的 XYZ 真值/Student 预测位置及预测/真实左右接触状态曲线在下半部；不写静态图片或三路独立视频。
- 兼容性：只读取既有 `uint8` RGB frame streams 和 manifest 频率，不改变 rollout、observation、动作、奖励或 checkpoint 契约。
- 验证情况：已执行改造后脚本 `compileall` 与已有 GelSight Heatmap-DR rollout 离线导出 smoke，确认只写出 149 帧的 `rollout_diagnostics.mp4`，其中含相机拼接与接触力/位置诊断曲线。

### 2026-08-07 CST — docs 正文由 7 个文档合并为 3 个

- 类型：文档组织
- 修改：`PROJECT_OVERVIEW.md` + `ARCHITECTURE.md` + `DATA_FLOW.md` 合并为 `ARCHITECTURE.md`（三部分：项目概览 / 模块架构 / 数据流）；`DEVLOG.md` + `EXPERIMENTS.md` 合并为 `DEVLOG.md`（变更日志 / 实验记录）；`DECISIONS.md` + `KNOWN_ISSUES.md` 合并为 `DECISIONS.md`（`DEC-xxx` / `ISSUE-xxx`）。原文逐字保留，仅原标题降一级并加部分标题；正文合计 2018 行降为 2011 行（含新增的部分标题与本条记录）。
- 精简：只合并 5 处重复表述（`proprio_obs` 模块说明、`critic_*` key 列表、四路 tactile key 列表、done/success 条件、两处记录规则前言）改为交叉引用，并按 `git show HEAD` 刷新过期的「当前项目状态」；全部 EXP/DEC/ISSUE 条目、命令和数值一条未删。
- 影响：不涉及代码、观测、动作、奖励、done、checkpoint 或 task 注册；无需重新训练。同步更新 `AGENTS.md` 第 1/3/6 节与 `README.md` 的文档索引；`docs/archive/DEVLOG-2026H1.md` 中指向旧文件名的历史记录按原样保留。
- 遗留：实验编号 `EXP-003` 在合并前就被 GelFusion 对比计划和 Real-Alignment v3 rollout 诊断两个条目重复使用，本次未改动编号。
- 验证情况：已核对三份合并文档的标题层级与重复标题（仅上述 `EXP-003` 一处），并确认仓库内除归档历史记录外无残留旧文件名引用。本次未改 `.py`，未运行训练、评估或 Isaac 测试。

### 2026-08-07 CST — 精简 AGENTS.md 与文档维护机制

- 类型：协作约束 / 文档组织
- 修改：`AGENTS.md` 由 139 行重写为 88 行，主线更正为 `sim2real_grasp` / `sim2real_gelsight_rma` 的 Real-Alignment RMA（原第 1、3 节仍停留在 `occluded_grasping`），26 行项目地图改为 11 行目录级索引并指向 `ARCHITECTURE.md` 第 17 节；验证命令改为已确认可用的定向 pytest，并标注 `run_all_tests.py --discover_only` 受 ISSUE-015 阻断。T0/T1/T2 分级与全部代码修改硬规则保留。
- 修改：`DEVLOG.md` 由 1031 行降为约 200 行，只保留 2026-08-01 起的 23 条；2026-07 及更早的 38 条与「历史状态恢复说明」原文归档至 `archive/DEVLOG-2026H1.md`。记录规则收紧为每条 3–5 行，`EXPERIMENTS.md` 的 18 字段实验模板压缩为 8 字段。
- 影响：不涉及代码、观测、动作、奖励、done、checkpoint 或 task 注册，仅改变协作流程与文档组织；无需重新训练。
- 验证情况：已按 heading 与正文逐条对账，归档前后 61 条记录一一对应且正文字节一致，两文件无重叠；已确认新索引中 20 个路径全部存在。本次未改 `.py`，未运行训练、评估或 Isaac 测试。

### 2026-08-07 CST — 新增 Real-Alignment RMA GelSight 子包

- 类型：RMA task profile / robot asset / manifest contract / 调试脚本 / 测试 / 文档
- 修改：新增 `source/tacex_tasks/tacex_tasks/sim2real_gelsight_rma`，以薄封装继承现有 Real-Alignment RMA Teacher/Student/Heatmap/DR 环境；robot cfg 切换为 `FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_RIGID_CFG`。新增五个 task id：GelSight Teacher、Student、Student-Heatmap、Student-DR、Student-Heatmap-DR。
- 修正：GelSight RMA 的 cube ContactSensor filter 改为 `gelpad_left/right`，不再沿用旧 PandaHand 的 `panda_leftfinger/rightfinger`，避免独立 gelpad rigid body 接触被漏掉。GelSight Teacher 默认不创建 `gsmini_left/right` tactile-rendering sensors；GelSight Student 默认创建左右 tactile sensors 作为接触预测输入。预览脚本仍可保存 PNG。
- 策略接口：GelSight `tactile_rgb` 不进入 Teacher Actor observation。Teacher Actor 输入仍是 `proprio_obs[15] + action_history[4] + rma_cube_pos[3] + rma_contact_state[2]`。GelSight Student TorchScript 调用顺序变为 `RGB[224,224,3] + proprio[15] + history[4] + gsmini_left_rgb[96,128,3] + gsmini_right_rgb[96,128,3]`，action 仍是 `[dx,dy,dz,dgripper]` 4维；PandaHand Student 仍是 RGB-only。
- 兼容性：`RMA_TEACHER_TASKS` 与 `RMA_STUDENT_TASKS` 扩展到 GelSight task；Teacher manifest environment contract 新增 robot/GelSight profile 和 contact filter 字段。RMA Teacher manifest 升至 v10；v9 及更早版本会以 unsupported manifest version 清晰拒绝。GelSight Teacher 不能蒸馏到旧 PandaHand Student，旧 PandaHand Teacher 也不能蒸馏到 GelSight Student。
- 调试：新增 `scripts/reinforcement_learning/skrl/save_rma_gelsight_preview.py`，只读保存左右 `tactile_rgb` PNG；默认训练不保存触觉图，避免 I/O 开销。
- 验证情况：已执行针对新增/修改 RMA 文件的 `python -m compileall ...`、GelSight config contract 定向 pytest、1-env GelSight Teacher Isaac smoke、GelSight tactile preview 保存 smoke；最终完整 `compileall` 与 `git diff --check` 见本次任务汇报。未执行正式 Teacher/Student 训练、评估或真机部署。

### 2026-08-07 CST — RMA reward 移除 cube-finger 切向力惩罚

- 类型：RMA reward / Teacher manifest contract / 测试 / 文档
- 修改：从 RMA Teacher/Student 共用 reward 中移除 `reward/rma_tangential_contact`，不再计算 cube ContactSensor 的世界 XY 切向力代理，也不再输出对应 `info/rma_tangential_contact_*` 日志。
- 奖励：当前 RMA 额外项只保留 `reward/rma_contact` 和 `reward/rma_action_rate`。
- 兼容性：Teacher environment contract 删除切向力惩罚字段；RMA Teacher manifest 升至 v10，v9 及更早 Teacher checkpoint 不应继续用于当前 Student 蒸馏。观测 key/shape、Actor feature 维度、Student TorchScript 部署输入输出不变。
- 验证情况：见本次任务汇报；未执行重新训练、正式评估或真机闭环。

### 2026-08-03 CST — RMA reward 曾新增 cube-finger 切向力惩罚

- 类型：RMA reward / Teacher manifest contract / 测试 / 文档
- 修改：在 RMA Teacher/Student 共用 `_RMATerminalMixin._compute_additional_reward()` 中新增 `reward/rma_tangential_contact`。该项从 cube ContactSensor 的 `force_matrix_w_history [N,2,1,2,3]` 取世界 XY 分量 `sqrt(Fx^2+Fy^2)`，在最近两个 physics substeps 和 cube body 维度取最大，得到左右 finger 切向力 `[N,2]`。
- 奖励：默认 `rma_tangential_contact_penalty_weight=0.2`、`rma_tangential_contact_deadband_n=5.0`，每环境惩罚为 `-0.2 * mean(clamp(tangent_force_lr - 5.0N, min=0))`。无接触或小切向力时 penalty 为 0；不额外判断抓住状态。
- 日志：新增 `reward/rma_tangential_contact`、`info/rma_tangential_contact_force_n`、`info/rma_tangential_contact_excess_n`、`info/rma_tangential_contact_force_max_n`，RMA 控制台摘要新增 `tangent=...`。
- 兼容性：Teacher environment contract 新增切向力惩罚参数；旧 Teacher manifest 与当前 RMA 环境不再匹配，使用新奖励需要重新训练 Teacher 后再蒸馏 Student。观测 key/shape、Actor feature 维度、Student TorchScript 部署输入输出不变。Clean/DR/旧 Privileged reward 不变。
- 验证情况：已执行针对修改文件的 `python -m compileall ...` 和 `git diff --check`，通过；已执行 RMA config/environment contract 定向 pytest，返回码 0；另以 2-env Student 直接 smoke 确认新增四个 tangential log key 存在且 penalty 有限、非正。未执行重新训练、正式评估或真机闭环。

### 2026-08-03 CST — 新增 RMA Student cube-center heatmap 辅助监督

- 类型：RMA Student 模型 / 训练脚本 / task 注册 / checkpoint 兼容 / 测试 / 文档
- 修改：保留现有 `SpatialSoftmaxAdaptationHead` 和 Student `forward(RGB, proprio, history)->action` 接口；从 ResNet18 layer3 `[N,256,14,14]` 新增 `Conv3x3+ReLU+Conv1x1` heatmap head，输出 `[N,1,14,14]`。新增 Clean/DR heatmap Student task：`TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-v0`、`TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-DR-v0`。
- 监督：用 `cube_position_root`、当前 Student 有效内参和相机 optical pose 投影 cube center，生成14x14 Gaussian GT heatmap；默认 `sigma=1.5` heatmap px。相机后方或224x224图像外样本通过 valid mask 排除 heatmap MSE。DR heatmap task 使用每环境随机化后的相机 pose、focal scale 和 principal shift。
- Backbone 微调：默认仍冻结全部 ResNet18；新增 `--train_backbone_after_layer2` 后只解冻 ResNet18 layer3/layer4，conv1、bn1、layer1、layer2 及 Actor Core 仍冻结。ResNet BatchNorm 保持 eval；backbone 使用独立 `--backbone_learning_rate`，默认 `3e-5`。
- 日志：训练脚本新增 `Loss/heatmap`、`Heatmap/center_error_px`、`Heatmap/valid_fraction`、`Position/mae_x/y/z_m`，终端进度行同步打印 heatmap loss、uv误差和xyz MAE。`--heatmap_debug_interval > 0` 时在 `heatmap_debug/` 保存少量 RGB overlay；默认关闭，不影响正常训练速度。
- 兼容性：旧 Student task 默认 heatmap supervision 关闭；新增 heatmap task 默认权重1.0。同版本内 Student checkpoint 可通过允许 missing optional heads 加载；RMA Student artifact 后续升至 v6 后，v5 及更早 checkpoint 不应继续 resume/export。若 resume 时 optimizer/backbone 训练配置不同，则只加载模型权重并重新初始化 optimizer。TorchScript 仍只导出4维 action，不输出 heatmap。
- 验证情况：已执行针对修改文件的 `python -m compileall ...`，通过；已用直接文件导入的纯 PyTorch smoke 检查 Student forward、heatmap head 输出、投影/GT heatmap/soft-argmax 和 `torch.jit.script`，通过。系统 Python 直接导入 `tacex_tasks` 因未启动 Isaac/缺少 `omni` 失败，未执行完整 Isaac pytest、训练或评估。

### 2026-08-03 — RMA Teacher/Student 增加动作平滑项

- 类型：RMA reward / Student distillation loss / checkpoint contract / 测试 / 文档
- 修改内容：RMA Teacher、Clean Student 和 DR Student 环境共用 `rma_action_rate_penalty_weight=0.05`，按当前实际下发物理增量 `action_history` 与上一拍的差计算动作变化惩罚，差值除以环境动作尺度 `[0.05,0.05,0.05,0.01]` 后取均方。`train_rma_student.py` 新增 `--action_smoothness_loss_weight`，默认0.05，约束 Student action 接近上一拍环境 action，并将 smooth loss 写入 TensorBoard、控制台和 Student checkpoint loss contract。
- 影响的观测：不改变 `proprio_obs[15]`、`action_history[4]`、`wrist_rgb[224,224,3]`、`rma_cube_pos[3]` 或 `rma_contact_state[2]` 的 key/shape。
- 影响的动作：不改变4维 tanh action 或环境动作尺度；只改变训练目标，使策略倾向减少相邻 policy step 的命令跳变。
- 影响的奖励：RMA reward 新增负项 `reward/rma_action_rate`，Clean/DR/旧 Privileged reward 不变。
- 影响的 done/success：不改变；RMA success 仍只统计不终止。
- checkpoint 兼容性：Teacher manifest 升至 v7，并把 action-rate reward 写入 environment contract；v6 及更早 Teacher 不应继续用于新 Student 蒸馏，需从头训练 Teacher/Student。Student 模型和 TorchScript 输入输出维度不变。
- 验证情况：`python -m py_compile` 通过；`git diff --check` 通过；RMA 配置与 legacy artifact 拒绝定向测试通过；Student 梯度/TorchScript 定向测试通过；独立 Isaac Teacher smoke 确认零动作 `reward/rma_action_rate=0.0`，从0跳到满幅 x action 时 `reward/rma_action_rate=-0.0125`、`info/rma_action_rate_norm_sq_mean=0.25`。完整 Teacher pytest 在旧 FK 对比段输出异常，仅得到 `F` 且无 traceback，本次未作为通过项。
### 2026-08-02 — 统一 RMA Student 蒸馏进度日志

- 类型：训练脚本 / 训练监控 / 文档
- 修改内容：Student 启动时明确打印总 update、并行环境数、剩余 simulator transition、日志和 checkpoint 间隔；每个日志间隔输出进度、累计 transition、实时 update/sample 吞吐、已耗时、ETA、分项 loss、位置 RMSE、接触 accuracy 和环境发布的累计/最近成功率。checkpoint 输出也改为统一前缀并强制 flush。
- 影响：只改变控制台和 TensorBoard 监控，不改变环境、观测、loss、优化器、动作、奖励、done 或 checkpoint 格式。`timesteps` 仍表示外层 update，单轮采样 transition 数为 `timesteps * num_envs`。
- 验证情况：待本次语法与最小 Student 蒸馏验证。

### 2026-08-02 — RMA Student 增加无课程全强度视觉域随机化

- 类型：RMA 环境 / 训练脚本 / checkpoint 路由 / 测试 / 文档
- 修改内容：新增 `TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-DR-v0`。该 Student 复用 Real-Alignment DR 实现，并显式关闭 `dr_curriculum_enabled`，因此首个 update 即按 full scale 采样相机外参、焦距/主点 GPU warp、图像颜色/模糊/噪声、板/背景和共享 DomeLight 扰动。
- 影响的观测：部署和模型输入仍为 `wrist_rgb uint8[N,224,224,3]`、`proprio_obs[N,15]`、`action_history[N,4]`；仅 RGB 分布改变。`rma_cube_pos[N,3]` 与 `rma_contact_state[N,2]` 仍只作训练标签。
- 影响的物理/奖励/done：不改变 Teacher 或 Student 的机器人、cube、接触阈值/奖励、动作尺度、success 与 done 语义；现有 Teacher manifest 继续校验通过。
- checkpoint 兼容性：Student payload 记录 Clean 或 DR task。旧 Clean Student v5 artifact 仍可导出；resume 会拒绝跨 Clean/DR profile。DR Student 应从头蒸馏，不应把既有 Clean Student checkpoint 作为 resume 输入。
- 验证情况：待本次语法、RMA 定向测试和最小 DR Student 蒸馏 smoke。

### 2026-08-02 — 新增蒸馏 RMA Student 专用回放脚本

- 类型：评估脚本 / 文档
- 修改内容：新增 `play_rma_student.py`，直接加载 RMA Student artifact 的模型 state_dict，不走 skrl PPO `agent.load()`。脚本从 checkpoint 读取 Clean 或 DR task，校验 Teacher manifest 的环境 contract 及 Student encoder/Actor hash，以全随机 XY 范围运行固定步数，并写周期 CSV 与最终 JSON。
- 影响：不修改训练、Teacher、Student 网络、观测、动作、奖励、done 或 checkpoint 格式；通用 `play.py` 保持只服务 skrl checkpoint。
- 验证情况：待本次 Student checkpoint 回放 smoke。

### 2026-08-02 — RMA Student TorchScript 支持 CUDA 加载

- 类型：部署导出 / 模型兼容性 / 测试 / 文档
- 修改内容：RMA Student 导出由 `torch.jit.trace` 改为 `torch.jit.script`，避免 trace 将 Panda FK 中静态索引的 buffer 内联为 CPU 常量；RMA Student 导出脚本在 CUDA 可用时自动加载新文件到 `cuda:0` 并与 CPU eager 输出比较，CUDA 的 `1e-3` 动作容差单独记录以允许 CPU/GPU 卷积浮点差异。
- 影响：Teacher/Student 数值、state_dict 键名、观测、动作、训练和现有 checkpoint 加载语义不变；旧 TorchScript 文件仍可能含 CPU 内联常量，应重新导出后再部署 GPU。
- 验证情况：已以 Clean、DR 两个现有 Student checkpoint 直接 script 并在 `cuda:0` reload/forward；两者均成功。导出的 CPU/CUDA 动作最大绝对误差分别为 `1.328e-4`、`3.963e-4`，低于 `1e-3` 容差。
### 2026-08-02 — 更新 Real-Alignment 标定相机外参

- 类型：环境配置 / sim-to-real 视觉 contract / 测试 / 文档
- 修改内容：`Sim2RealCubeRealAlignmentEnvCfg.wrist_camera` 的 nominal `base_T_camera_color_optical` 更新为平移 `(1.166091088407, 0.035901608197, 0.514200335898) m`，以及 ROS optical Isaac `wxyz=(0.378248136306, -0.604227000834, -0.586824979121, 0.384024117374)`；该四元数由提供的 ROS `xyzw` 重排而来，重建旋转矩阵与提供的 `T_base_color` 一致。
- 影响：Clean、DR 和两个 RMA Student 的相机图像分布改变，现有 Student checkpoint 和 TorchScript 不应继续用于新外参；Teacher 是无相机特权策略，外参本身不改变其观测或动作语义。内参、动作、物理、奖励、done 和网络维度不变。
- 验证情况：`python -m py_compile` 通过；`git diff --check` 通过；在 Isaac Lab `isaaclab_2.1.1` 中运行 `pytest -q -k clean_and_dr_randomization_profiles_are_separated` 通过，确认 Clean/DR 均解析为该平移、四元数和 `convention=ros`。

### 2026-08-02 — 对齐真机 Franka 20 mm 基座垫高

- 类型：环境几何 / sim-to-real frame contract / 测试 / 文档
- 修改内容：Real-Alignment Clean 配置将 Panda root world position 设为 `(0,0,0.02) m`；桌面与方块不移动。相机标定仍以提供的 `base_T_camera_color_optical` 保存，独立 TiledCamera 的 world position 同步变为 `(1.166091088407,0.035901608197,0.534200335898)`，故 Panda base 到 camera 的相对变换不变。RMA Teacher manifest 升级至 v6，并把 robot base、camera base/world pose 写入 Teacher–Student environment contract。
- 影响：机器人相对于桌面/物体的初始几何、IK 可达性、碰撞和图像视角均改变；Clean、DR、RMA Teacher 和两个 RMA Student 都必须从头训练，现有 checkpoint/TorchScript 不可继续使用或 resume。观测/动作张量维度、内参、奖励公式与 done 语义不变。
- 验证情况：`python -m py_compile` 和 `git diff --check` 通过；在 Isaac Lab `isaaclab_2.1.1` 中运行 Clean/DR 配置定向测试与 4-env reset/table-collision smoke 均通过；RMA Clean/DR/Teacher/Student 配置契约测试通过。

### 2026-08-02 — 保存 RMA Student 的训练输入帧

- 类型：训练可观测性 / 视觉 DR 诊断 / 文档
- 修改内容：`train_rma_student.py` 新增 `--save_start_frame` 与 `--start_frame_count`。启用后，首次 reset 的不同环境 `wrist_rgb` 以 PNG 写入本次 run 的 `camera_frames/`；保存的是经过当前 Student observation path（含 DR 与 nominal intrinsic warp）的 224×224 uint8 图像，不执行额外环境动作。
- 影响：默认关闭，训练、随机化、损失、checkpoint 和 Student 部署输入不变；启用后仅在训练开始增加有限的图像 I/O。
- 验证情况：`python -m py_compile scripts/reinforcement_learning/skrl/train_rma_student.py` 和 `git diff --check` 通过。以新的 Teacher checkpoint、`RMA-Student-DR-v0`、2 env、1 update 运行保存 smoke，成功写出 `start_wrist_rgb_env000_224x224.png` 与 `start_wrist_rgb_env001_224x224.png`，并完成 checkpoint 保存。

### 2026-08-02 — 以真机裁剪帧重设 Real-Alignment visual nominal 与 DR

- 类型：视觉 sim-to-real / 相机 DR / 场景外观 / 实验配置
- 修改内容：检查了提供的 `224x224` 真机裁剪帧与新外参、20 mm base elevation 下的 Clean/DR Student 输入帧。保留给定标定相机的 nominal 外参和内参；将 Clean 桌面/背景/DomeLight 调为更暗中性的值，并将 DR 收紧为 camera XYZ ±1.5 mm、RPY ±0.5°、focal ±1%、principal ±1 px，brightness/contrast ±10%、gamma ±8%、saturation -10%/+5%、hue ±2°、white balance ±4%、blur 8%、noise std 0–0.006、光照 1300–1900 与 4800–6200 K，以及近黑板/背景材质范围。
- 影响：Clean 与 DR Student 的 RGB 分布改变，所有现有 Student checkpoint/TorchScript 均不可继续训练或部署，须按新 nominal 从头训练。Teacher 没有视觉输入，动作/物理/奖励/张量维度不变；但新 Teacher run 仍应作为新实验的配套 artifact。真实图中的导轨未建模，是当前未覆盖的 visual gap。
- 验证情况：`python -m py_compile` 和 `git diff --check` 通过；Isaac Lab 定向 Clean/DR 配置测试通过。分别以 Clean 1 env 和 DR 4 env、各 1 update 运行保存帧 smoke；Clean/DR 均成功产生 224×224 Student 输入 PNG，新的 DR 样本未再出现旧范围下的明显绿色桌面。

### 2026-08-01 CST — DR 延后到 100k 并训练 300k

- 类型：DR 环境配置 / Agent 配置 / 训练监控 / 测试 / 文档
- 修改：仅将 Real-Alignment DR 的零扰动阶段延长到 100k，100k–220k 线性扩大到 full range，220k–300k 保持 full range；DR Agent trainer 从 200k 延长到 300k。Clean 环境、Clean Agent 及其 200k trainer 不变。
- 日志：DR 的 `_compute_additional_reward()` 保留继承的撞桌 penalty，并新增 `info/dr_curriculum_scale`，不改变 reward 数值。
- 依据：旧 200k DR run 的 TensorBoard 曲线在课程后段出现 reach/lift 回退，累计成功接近 0；新时间表先让继承的物体位置课程在 100k 完成，再启动视觉随机化。
- 影响：不修改物理、Actor/Critic observation key/shape、4-D action、reward、success、done 或 Clean task；DR env config hash 和训练时长改变，旧 DR checkpoint 不能作为当前 profile 的续训起点，需要从头训练。
- 验证情况：`git diff --check`、完整 `python -m compileall -q source scripts tools`、Clean/DR Agent YAML 解析和 4-env Real-Alignment Isaac 定向 pytest 12 项通过；其中 Clean trainer 明确保持 200k，DR trainer 为 300k，DR 课程边界和 `info/dr_curriculum_scale` 断言通过。未执行 300k 训练、正式评估或真机闭环。

### 2026-08-01 CST — DR 改为严格基于 Clean 的视觉课程

- 类型：环境配置 / GPU 视觉随机化 / 光照 / 测试 / 文档
- 修改：Real-Alignment DR 的课程初值从 `0.10` 改为 `0`，前 20k policy steps 与 Clean 视觉路径完全一致，20k–120k 再线性扩大到 full range。Plate/backdrop 的中心改为 Clean 的近黑材质，分别在 full scale 的每通道 `[0.01,0.05]` 和 `[0.005,0.03]` 内变化。
- 光照：DomeLight 以 Clean 的 intensity `2000`、color `(0.75,0.75,0.75)` 为中心；随机色温使用相对 5500 K 的 RGB tint 且关闭 renderer 二次色温处理，使 scale=0 精确恢复 Clean。
- 影响：不修改物理、4-D action、`action_history/proprio/wrist_resnet` shape、奖励、success 或 done；DR 视觉分布和 env config hash 改变，旧 DR checkpoint 不作为新课程的续训起点。Clean checkpoint 的网络 shape 和控制语义保持兼容。
- 验证情况：`git diff --check`、完整 `python -m compileall -q source scripts tools` 和 4-env Real-Alignment Isaac 定向 pytest 12 项通过；其中显式验证 scale=0 的相机/内参/后处理缓存、材质、光照及像素输出与 Clean 一致。标准 discovery 的 base Python 缺少 `isaacsim`；改用 `isaaclab_2.1.1` 后 warm start 成功，但仍被既有 ISSUE-015（缺失 `test_argparser_launch.py` skip target）阻断。未重新训练或执行真机闭环。

### 2026-08-01 — 新增 Real-Alignment RMA 教师学生蒸馏路线

- 类型：代码 / 配置 / 训练脚本 / 评估脚本 / 导出 / 文档
- 修改内容：新增 Clean v9 RMA Teacher/Student task、共享归一化 Actor Core、单帧 spatial-softmax 定位头、在线双损失蒸馏、固定网格评估和端到端 TorchScript 导出。
- 影响的观测：仅新增 task；Teacher Actor 为22维，Student 训练环境增加 raw RGB 与 cube XYZ label，部署接口不含 label。
- 影响的动作：新增路线继续使用4维 tanh action 和 `[0.025,0.025,0.025,0.005]` 尺度；现有 task 不变。
- 影响的奖励和 done：继承当前 Clean；新增只读 per-env success terminal 缓存供评估。
- checkpoint 兼容性：RMA 使用独立 manifest/checkpoint，旧 Privileged 及 Clean/DR checkpoint 不兼容也不被修改。
- 验证情况：`git diff --check` 和完整 `compileall` 通过；RMA Teacher 定向测试1项、Student/模型/TorchScript定向测试4项通过。Teacher 以4 env完成128步和一次PPO更新并写 manifest；Student 以2个相机环境完成2步双损失更新并保存 checkpoint；导出模型 eager/trace/reload 最大误差均为0。仓库 wrapper 使用 base Python，缺少 `isaacsim`；正确 conda 环境的 discover warm start 成功，但被既有缺失 `test_argparser_launch.py` skip target 阻断。
- 待确认：实际200k Teacher、100k Student 的成功率和位置 RMSE。

### 2026-08-01 — RMA Actor 加入可部署末端位置与相对目标位置

- 类型：代码 / 模型契约 / 测试 / 文档
- 修改内容：共享 Actor 从 `proprio_obs[:7]` 内嵌 Panda FK，计算机器人根坐标系指尖中点 XYZ，并与 cube XYZ 形成相对目标 XYZ；网络特征由22维增至28维。物体与末端 quaternion 不加入。
- 依据：Real-Alignment 使用 `panda_hand + [0,0,0.1034] m` 作为 IK TCP，且该点与左右 finger 各自 `[0,0,0.045] m` 指尖中心的中点一致；Panda 固定关节变换来自任务使用的 URDF。
- 影响的观测：Teacher/Student 环境 key 与 shape 不变；末端 XYZ 在模型内部从7维关节角派生，不引入仿真 privileged link input。
- 影响的动作、奖励和 done：无；保持4维 tanh action、Clean 控制/奖励/success/done。
- 影响的网络输入维度：`RMAActorCore` 第一层由22改为28；Student TorchScript 外部三输入签名不变。
- checkpoint 兼容性：RMA model/manifest/student artifact 升级至 v2，旧22维 RMA checkpoint 拒绝加载并需要重新训练；Clean、DR 和旧 Privileged checkpoint 不受影响。
- 验证情况：纯 Torch 检查确认初始 FK 输出约 `[0.499844,0.000043,0.299519] m`、仅定位头有梯度且 TorchScript 动态 batch 误差为0；Teacher 定向测试1项和 Student/模型/契约/TorchScript定向测试5项通过，其中 FK 与 Isaac 指尖中点按0.2 mm容差比较；4 env Teacher 完成128步和一次 PPO 更新并写 v2 manifest；2 env Student 完成2步双损失更新；导出 eager/trace/reload 最大误差为0。以上均为链路 smoke，不是策略效果。
- 待确认：当前28维方案实际200k Teacher、100k Student 的成功率和位置 RMSE。

### 2026-08-01 — RMA 增加特权接触奖励与视觉接触蒸馏

- 类型：RMA 环境 / 模型 / 训练 / 评估 / 导出 / 测试 / 文档
- 修改内容：仅为 RMA Teacher/Student 新增 cube-to-left/right-finger ContactSensor；0.5 N阈值生成两路二值接触，单侧/双侧接触每步奖励0.1/2.0。Teacher Actor 使用真实接触，Student adaptation head 从RGB联合预测XYZ与接触 logits，并加入接触 BCE。
- 影响的观测：两个 RMA task 新增 `rma_contact_state[N,2]`；Student 中该键只用于训练标签，不进入部署接口。Clean、DR、旧 Privileged 不变。
- 影响的网络输入维度：共享 Actor feature 从28维增至30维；视觉头输出由3维位置扩为3维位置加2维接触 logits。
- checkpoint 兼容性：RMA model/manifest/student artifact 升至v3，v2文件拒绝加载，Teacher和Student均需重新训练；Student TorchScript仍为RGB/本体/history三输入与4维动作输出。
- 验证情况：完整 `compileall`、Teacher YAML解析和 `git diff --check` 通过；Teacher定向测试1项（含实际双指接触力与2.0奖励）及Student/模型/契约/TorchScript定向测试5项通过。4 env Teacher完成128步与一次PPO更新并写v3 manifest；2 env Student完成2步三损失更新并保存v3 checkpoint；TorchScript导出 eager/trace/reload最大误差均为0。标准 discovery 用base Python时缺少`isaacsim`；正确conda环境warm start成功后被既有`ISSUE-015`缺失skip target阻断。
- 待确认：200k Teacher/100k Student正式效果、视觉接触 precision/recall/F1 和真实硬件接触泛化。

### 2026-08-01 — 精简 RMA 奖励控制台输出

- 类型：RMA 日志 / 文档
- 修改内容：只覆盖 RMA Teacher/Student 的控制台奖励摘要，保留加权 reach、lift、success、contact、table、当前 total、最近200步平均奖励、窗口成功率和平均抬升高度；完整诊断量继续写入 TensorBoard log。
- 影响：不改变奖励数值、观测、动作、done/success、网络输入或 checkpoint；Clean、DR、旧 Privileged 的控制台输出不变。
- 验证情况：`py_compile`、完整`compileall`和`git diff --check`通过；4 env Teacher完成256策略步与两次PPO更新，step 200仅打印精简摘要且窗口数值正确；RMA Teacher接触/奖励定向测试通过。

### 2026-08-01 — RMA success 改为非终止统计条件

- 类型：RMA 环境 / checkpoint contract / 测试 / 文档
- 修改内容：只覆盖 RMA Teacher/Student 的 `_get_dones()`；success 继续给奖励并更新 hold counter，但不再触发 done。Episode 只因 timeout 或严重机器人穿地碰撞终止；窗口成功率按 episode 内是否曾达到 success 统计。
- 影响：Clean、DR 和旧 Privileged 的 done/success 语义不变。RMA manifest、Student checkpoint 和导出 metadata 升至v4，v3及更早 RMA artifact fail closed，需要重新训练 Teacher/Student。
- 验证情况：`python -m compileall -q source scripts tools`、`git diff --check` 通过；RMA 配置隔离/旧v3 artifact拒绝/Teacher success非终止定向测试通过。

### 2026-08-01 — RMA 单独降低 finger actuator

- 类型：RMA 环境 / checkpoint contract / 测试 / 文档
- 修改内容：只在 RMA Teacher/Student 的 copied robot cfg 中覆盖 `panda_hand` actuator：`effort_limit_sim=40`、`stiffness=400`、`damping=40`。按用户要求，`gripper_width_delta_scale` 保持 Clean v9 的 `0.005 m/policy step`。
- 影响：Clean、DR、旧 Privileged 和 TacEx GelSight asset 不变。RMA 物理/控制 contract 改变，manifest、Student checkpoint 和导出 metadata 升至v5，v4及更早 RMA artifact fail closed，需要重新训练 Teacher/Student。
- 验证情况：`py_compile`、`git diff --check` 通过；RMA 配置隔离/旧v4 artifact拒绝/Teacher success非终止定向测试通过。2-env RMA Teacher 1-iteration smoke 成功写出 v5 manifest，记录 `velocity_limit_sim=null`、5mm/step 动作尺度和 40/400/40 hand actuator。

### 2026-08-01 — RMA Student 允许直接蒸馏

- 类型：训练脚本 / 文档
- 修改内容：移除 `train_rma_student.py` 的 `--teacher_eval` 与 `--allow_unqualified_teacher` 参数及其固定网格验收门槛。Student 现在只要求兼容的 Teacher checkpoint 及其同一 run 中的 manifest，随后直接开始在线视觉位置、接触和动作蒸馏；`evaluate_rma.py` 保留为可选的后验比较工具。
- 影响：不改变 Student 环境、观测、损失、网络输入、动作或 checkpoint 格式；无需重新训练已存在的 Teacher。未验证 Teacher 成功率的 checkpoint 也可作为蒸馏源，质量由用户自行负责。
- 验证情况：`python -m py_compile scripts/reinforcement_learning/skrl/train_rma_student.py source/tacex_tasks/tacex_tasks/sim2real_grasp/rma_artifacts.py` 与 `git diff --check` 通过。

### 2026-08-01 — 对齐 RMA Student 与 Teacher 的接触 contract

- 类型：RMA 环境配置 / 蒸馏启动修复 / 文档
- 修改内容：Student 原本从 Clean 继承 `0.5 N / 2.0 / 0.1` 的接触阈值、奖励权重和单侧系数，但指定 Teacher manifest 记录的是 `0.2 N / 3.0 / 1.0`。Student 现在显式使用 Teacher 数值，使环境 contract 校验通过，并保证 Student rollout 的接触标签和奖励分布与教师一致。
- 影响：修改 RMA Student 的接触标签阈值与 reward；不改变 Teacher checkpoint、Actor/Student 网络输入维度、loss 形式或 checkpoint 格式。此前尚未完成的 Student run 不应继续使用。
- 验证情况：待本次 Student 一步启动验证。

## 第二部分 实验记录

### 实验记录规则

本部分记录已确认的训练、评估和计划实验。任何结果必须来自实际日志、CSV、checkpoint、配置文件或明确人工记录；无法确认的信息写「待确认」。每个实验有唯一 ID，并按本部分末尾模板记录可复现所需的最小信息。不得根据目录名臆造成功率、训练轮数、最佳 checkpoint 来源或结论；只确认 artifact 存在而未算汇总指标时，写「汇总结果待确认」。

### EXP-004 — Real-reference-aligned vision-only Cube PPO

- 状态：历史 200000-step checkpoint artifact 和 TorchScript export 已存在，但它们来自 2026-07-12 strict sim2real contract 修复之前，仅保留作历史记录；修复后策略待重新训练
- 实验类型：Franka + D435 vision-only Cube sim2real
- Task id：`TacEx-Sim2Real-Cube-Real-Alignment-v0`
- 现实依据：`20260711_214450_real_alignment_reference/alignment_reference.json` 和同目录 RGB reference images
- 环境：`source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_real_alignment_env.py`
- 当前 Agent：`source/tacex_tasks/tacex_tasks/sim2real_grasp/agents/skrl_ppo_cube_real_alignment_cfg_resnet18.yaml`，Actor mean 已改为 `tanh(ACTIONS)`
- 随机化 profile：Clean；只随机 cube XY reset，关闭图像、光照、地面和台面颜色随机化
- Seed：42
- Actor observation：`wrist_resnet:512`、`proprio_obs:15`、`action_history:4`
- Action：4 维 `[dx, dy, dz, gripper_total_width_delta]`；当前环境逐维物理 command scale 为 `[0.05,0.05,0.05,0.01] m/policy step`。RMA checkpoint 内 action-history normalizer 仍为 `[0.025,0.025,0.025,0.005]`，必须存储实际下发的物理增量后再由模型按其训练尺度归一化；历史 v8/v7 分别为 25/2 mm 与 10/2 mm。
- Object：实测尺寸 `0.05x0.05x0.05 m`
- Object reset：Franka base/world 对齐坐标以 `(0.50,0.00,0.026) m` 为中心；前 20k policy steps 使用 `x=[0.48,0.52], y=[-0.02,0.02] m`，20k–100k 线性扩展，之后为完整 `x=[0.45,0.55], y=[-0.05,0.05] m`。play、bucket 和 rollout 强制完整范围
- Geometry：台面厚度 `1 mm`、顶面 `z=0.001 m`，5 cm 方块初始质心 `z=0.026 m`
- Reward：`5*reach + 15*lift + 100*success + table_collision_penalty`；质心抬升 0–35 mm 线性映射到 lift `[0,1]`，35 mm 连续保持 5 个 policy steps 触发 success；倾角只用于诊断，不参与 reward、success 或 done。目标机械臂 link 对台面最大接触力 `>1 N` 时该 step 加 `-10`
- Frequency：physics 60 Hz；camera/render/policy/action/observation/reward/history 30 Hz（`sim.dt=1/60, decimation=2`）
- Episode：5 秒，150 个 policy steps
- 当前策略 camera resolution/intrinsics：`224x224`，有效 `K=[338.742544,0,123.748857; 0,340.550811,120.393372; 0,0,1]`。它对应现实 `640x480` 裁剪 `x=[100,500), y=[34,432)` 后将 400x398 双线性缩放到 224x224；仿真用 centered 300 px-focal coverage render 加固定 GPU warp 实现该有效 K
- 当前 Panda base world pose：`pos=(0,0,0.02) m`，以匹配真机 20 mm 垫高；桌面/方块仍在 world `z=0` 参考下。`base_T_camera_color_optical` 的 `pos=(1.166091088407,0.035901608197,0.514200335898)`、Isaac `rot(wxyz)=(0.378248136306,-0.604227000834,-0.586824979121,0.384024117374)`、`convention=ros`；相机作为独立 world prim 的实际 position 为 `(1.166091088407,0.035901608197,0.534200335898)`，以保持该 base 外参。部署相机序列号锁定为 `215322076207`。
- Num envs：256，依据 `training_summary.txt`
- 日志目录：`logs/skrl/sim2real_cube_real_alignment/2026-07-11_23-15-23_ppo_torch_vision_only_resnet18/`
- Checkpoint：`checkpoints/best_agent.pt`；同目录存在 `agent_200000.pt`
- 历史部署 export：`checkpoints/exported/policy_actor_e2e_best_agent.pt` 和同名 JSON metadata；不能作为修复后 contract 的部署文件
- 历史 Export contract：raw uint8 RGB `[N,224,224,3]`、`proprio_obs [N,15]`、`action_history [N,4]` -> unbounded raw mean action `[N,4]`，nominal frequency 30 Hz
- Export validation：trace max abs error 0；synthetic/sim start/real median RGB 均输出 finite `(1,4)`，同输入重复误差 0；本机 CPU 平均推理约 8.4-9.3 ms
- 普通 play metrics：`metrics/play/play_metrics_20260712_120929.csv` 当前记录的 success rate 为 `nan`，不能据此报告成功率
- Gate-removal 前最后一轮训练：`logs/skrl/sim2real_cube_real_alignment/2026-07-25_22-01-32_ppo_torch_vision_only_resnet18/` 完成 200k steps，manifest 为 contract v3、40°→10° success curriculum，但仍声明 `privileged_dz_gate=applied_after_action_history_and_not_available_on_the_real_robot`；其 `best_agent.pt`、`agent_200000.pt` 不兼容当前 v9，无论训练标量如何都不得续训、导出或部署。对应 play CSV 的 success rate仍为 `nan`。
- 已确认 smoke：环境可创建、reset、执行零动作；观测 shape 和动作空间正确。
- 旧奖励 smoke（已被当前质心语义替代）：reset 后最低角点日志曾为 `lift_delta≈0.001 m`。当前边界测试要求质心 delta `[0,0.0175,0.035] m` 得到 lift `[0,0.5,1]` 和 success `[False,False,True]`。
- 物理可抓取证据（不是当前 v9 策略效果）：`logs/skrl/sim2real_cube_real_alignment/2026-07-25_20-43-05_ppo_torch_vision_only_resnet18/` 的 512-env 旧奖励训练日志在 step 4000 记录 `Instantaneous reward (max)=119.78437`，并在 step 4500 记录非零 `reward/success=0.000148437495`。这确认旧场景至少出现过完整 lift+success 的物理轨迹，但不能据此给出当前策略成功率。
- Reach 几何隔离测试（2026-07-30，4-env Isaac）：reset 时左右指尖中心分别约为 `(0.4999,-0.0200,0.2995)` 与 `(0.4998,0.0200,0.2995) m`，中点约为 `(0.49984,0.000043,0.29952) m`；该中点与 `panda_hand+0.1034 m` IK TCP 的误差最大约 `1.37e-7 m`。使用生产 `_get_rewards()` 且隔离其他奖励后，方块质心位于中点、偏移 25 mm、偏移 50 mm、位于 `panda_hand` origin 时，reach 分别为 `1.0000/0.7551/0.5379/0.2245`，确认峰值位于实际指尖中点而非 hand origin。当前 reset 距离约 `0.274 m`，在 `reach_sigma=0.1 m` 下 raw reach 仅约 `0.0083`，因此早期 reach 数值小主要来自长距离 tanh 饱和，不是中心坐标错误。
- 当前训练监控：`episode_success_rate_window` 是最近 200 个 policy steps 内成功结束 episode 数 / 所有结束 episode 数；一个并行 step 可贡献多个完成 episode，无完成 episode 的 step 仍推进窗口。日志同时记录窗口成功/完成数、累计成功/完成数和累计成功率；它与单步 `reward/success` 分开，后者仍是当前达到 35 mm 的环境比例。
- 历史图像对齐 smoke：旧相机 world `pos=(1.90,0.0,0.468)`、向下约 15° 时，seed 42 仿真帧 RGB mean `[70.6,113.0,70.1]`，现实 model median 为 `[80.1,113.0,81.9]`；方块中心分别为 `(86.6,128.6)` 和 `(85.6,128.5)` pixel。这是旧位姿下的单帧外观检查，不适用于当前相机位姿，也不是策略结果。
- 当前训练 contract v9：ResNet18 始终 `eval()` 且参数冻结；Clean/DR `action_history` 为上一拍逐维 scaled/clipped requested physical command `[0.05*u_x,0.05*u_y,0.05*u_z,0.01*g]`。RMA Student 训练/部署模型对该 history 使用 checkpoint 固有 `[0.025,0.025,0.025,0.005]` normalizer，二者不能只改其一。夹爪目标宽度每个 policy step 只累加一次，physics application 只重发缓存目标；IK TCP 为固定 `panda_hand+0.1034 m`，reach/critic 使用左右指尖世界坐标中点；Actor 后控制链不读取 ground-truth object XY；Actor deterministic mean 为 `tanh(raw_mean)`。Contract 同时记录新 crop/有效 K/GPU 补偿、相机序列号、`x/y ±5 cm` 物体课程、台面、无倾角 success 和碰撞奖励语义。
- Checkpoint 兼容性：上述历史 checkpoint 虽然张量 shape 不变，但已经适应旧 BatchNorm、两拍延迟 history、unbounded mean、per-finger/substep 夹爪语义、旧抓取中心或不同 policy frequency，不能在当前环境中 resume 后继续训练；旧 artifact 只能按其保存的环境配置复现，需要按当前 30 Hz/5 秒配置从头重新训练。
- 当前短训练 run：`logs/skrl/sim2real_cube_real_alignment/2026-07-26_20-51-57_ppo_torch_vision_only_resnet18/`；`best_agent.pt` SHA-256 为 `92c930d2605b8abf58466cbb4717f2004bd92a96629937f62c05efae7fc341d3`。截至约 11.5k steps，TensorBoard 的 reach reward 从约 `0.023` 上升、最好约 `0.331`，reach distance 从约 `0.234 m` 降至约 `0.120 m`；lift 仅短暂约 `0.014`，success 近似为 0。因此它只能用于受控真机链路诊断，不能视为已学会抓取。
- 当前 v8 export：`checkpoints/exported/policy_actor_e2e_best_agent.pt`，SHA-256 `b888e7321c0481f80623ffa84837f32bc8f426eefbd1af33937764d787744c24`；复制到 `franka/checkpoint/real_alignment_v8_25mm/exported/`，独立 JIT 测试确认输出 finite、bounded、repeat diff 为 0。真机成功率：待确认。
- 修复后链路 smoke：`logs/skrl/sim2real_cube_real_alignment/2026-07-12_14-29-22_ppo_torch_vision_only_resnet18/` 以 1 env 完成 128 steps/一次 PPO update，并用测试 `agent_128.pt` 跑通 RGB/feature export、独立 JIT tester 和 2-step deterministic play。该 checkpoint 只验证代码链路，不计作策略效果实验。
- 待确认：新 D435 外参的标定误差、新 crop 下当前 `x=[0.45,0.55] m` 全范围方块的完整可见性、真实方块质量/摩擦、真机实际 30 Hz 抖动和成功率。当前 v9 尚无新训练 checkpoint 或真机 bundle；历史 v8 仿真与独立真机配置仍声明 25/2 mm，不得用于 v9 环境。

#### EXP-004 推荐训练命令

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-v0 \
  --num_envs 256 \
  --enable_cameras \
  --start_frame_count 5 \
  --headless
```

训练完成后在完整位置范围评估至少 100 个完整 episode；当前接受标准为 success rate
`>=80%` 且机械臂撞桌 transition 比例 `<1%`。达到标准前不导出真机 bundle。

### EXP-004-DR — Real-reference-aligned broad domain-randomized Cube PPO

- 状态：延后 DR 的 300k curriculum 环境和训练配置已实现，待从头训练；旧 200k DR run 不等价于当前 profile
- Task id：`TacEx-Sim2Real-Cube-Real-Alignment-DR-v0`
- 继承：与 EXP-004 Clean 使用相同机器人、相机、方块、动作、观测、奖励和 done
- 课程：0–100k outer policy step 使用 `scale=0` 的精确 Clean 视觉场景，并先完成 20k–100k 物体位置课程；100k–220k 从 0 线性增至 100%；220k–300k 保持 full range。`common_step_counter` 与 `num_envs` 无关，当前 scale 记录为 `info/dr_curriculum_scale`，resume offset 当前需手动设置。
- 相机 DR：以标定位姿为中心，full range 为 XYZ 各 `±3 mm`、RPY 各 `±1°`；GPU 等效 focal scale `[0.985,1.015]`、principal point shift 各 `±2 px`
- 图像 DR：brightness/contrast/saturation/gamma 均 `[0.85,1.15]`、hue `±5°`、white-balance red/blue shift `±0.08`、Gaussian blur probability `0.15` / kernel `3`、Gaussian noise std `[0,0.01]`
- 场景 DR：plate 以 Clean `(0.02,0.02,0.02)` 为中心并在 full scale 取 `[0.01,0.05]`，backdrop 以 Clean `(0.01,0.01,0.01)` 为中心并取 `[0.005,0.03]`；batch-global DomeLight 以 Clean intensity/color 为中心，full range intensity `[1000,3000]`、relative color-temperature tint `[3800,7200] K`；global ground 固定
- 未包含：质量、摩擦和机器人动力学随机化
- Agent：`source/tacex_tasks/tacex_tasks/sim2real_grasp/agents/skrl_ppo_cube_real_alignment_dr_cfg_resnet18.yaml`
- 日志目录：`logs/skrl/sim2real_cube_real_alignment_dr/`
- Trainer：300000 outer policy steps
- checkpoint、训练成功率和真机成功率：待确认

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-DR-v0 \
  --num_envs 4 \
  --enable_cameras \
  --headless
```

### EXP-004-P — Real-Alignment privileged-position reward/control upper bound

- 状态：环境和 Agent 配置已实现；训练与成功率待确认
- 目标：去除视觉定位变量，判断当前动作、物理和 reach/lift/success 奖励能否学会稳定抓取
- Task id：`TacEx-Sim2Real-Cube-Real-Alignment-Privileged-v0`
- Environment：`source/tacex_tasks/tacex_tasks/sim2real_grasp/sim2real_cube_real_alignment_privileged_env.py`
- Agent：`source/tacex_tasks/tacex_tasks/sim2real_grasp/agents/skrl_ppo_cube_real_alignment_privileged_cfg.yaml`
- Actor observation：28-D，`proprio_obs:15 + action_history:4 + privileged_cube_pos:3 + privileged_gripper_pos:3 + privileged_target_pos:3`
- Position frame：去除 `scene.env_origins` 的 robot-root-aligned per-env frame
- Camera/encoder：均不创建；无需 `--enable_cameras`
- Action/reward/done：继承 EXP-004 Clean，保持 4-D total-width action、质心 0–35 mm lift、无倾角 gate、5-step hold、撞桌 `-10` 和 150-step horizon
- PPO：与 Clean baseline 保持 rollouts 128、4 epochs、16 minibatches、learning rate `3e-4`，用于尽量只比较观测差异
- 日志目录：`logs/skrl/sim2real_cube_real_alignment_privileged/`
- Checkpoint、训练成功率、确定性 play 成功率：待确认
- 训练链路 smoke：`logs/skrl/sim2real_cube_real_alignment_privileged/2026-07-26_16-18-10_ppo_torch_privileged_position/` 使用 4 env 完成 128 steps/一次 PPO update；只证明环境、28-D Actor、49-D Critic 和 Runner 接通，不计作抓取效果。
- 部署：不允许；Actor 依赖 simulator ground truth

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-Privileged-v0 \
  --num_envs 256 \
  --headless
```

### EXP-003 — GelFusion-style RL comparison plan

- 状态：计划中；代码和配置已新增，训练/评估结果待确认
- 实验类型：视触融合方法对比
- 论文参考：Jiang et al. 2025 GelFusion 思路；当前实现是 PPO/RL 特征级复现，不是 Diffusion Policy 原版。
- Task id：`TacEx-GelFusion-Downsample-Drawer-Occlusion-Cube`
- Scene：`Drawer-Occlusion`
- Object：`Cube`
- Policy / method：GelFusion-style vision-led cross attention + tactile dynamic stats
- Agent 配置：`source/tacex_tasks/tacex_tasks/occluded_grasping/agents/ppo_vt_gelfusion.yaml`
- Env / actor 代码：
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_box.py`
  - `source/tacex_tasks/tacex_tasks/occluded_grasping/vt_gelfusion_policy.py`
- Seed：42，依据 `ppo_vt_gelfusion.yaml`
- Observation：
  - `third_resnet: 256`
  - 四路 `tactile_*_depth_resnet: 256`
  - `tactile_dynamic_stats: 8`
  - `proprio_obs: 18`
  - critic privileged keys 沿用 VT 环境
- Action space：5，沿用 `[dx, dy, dz, dyaw, gripper]`
- Checkpoint：待训练产生
- 日志目录：待训练产生

#### EXP-003 推荐训练命令

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-GelFusion-Downsample-Drawer-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras
```

#### EXP-003 推荐对比对象

- 视觉退化 baseline：`TacEx-V-Downsample-Drawer-Occlusion-Cube`
- 普通 VT baseline：`TacEx-VT-Downsample-Drawer-Occlusion-Cube`
- 现有 cross/aux 分支：`TacEx-Tactile-Cross-Downsample-Drawer-Occlusion-Cube` 或 `TacEx-Tactile-Cross-Alpha-Aux-Downsample-Drawer-Occlusion-Cube`

#### EXP-003 推荐评估命令

```bash
python scripts/reinforcement_learning/skrl/play_bucket.py \
  --task TacEx-GelFusion-Downsample-Drawer-Occlusion-Cube \
  --checkpoint <gelfusion_run>/checkpoints/best_agent.pt \
  --num_envs 16 \
  --enable_cameras \
  --bucket_rounds 1 \
  --headless
```

#### EXP-003 结果

| 指标 | 值 | 状态 |
| --- | --- | --- |
| 普通评估 recent success rate | 待确认 | 未训练 |
| Bucket 总成功率 | 待确认 | 未训练 |
| 按遮挡程度分组成功率 | 待确认 | 未训练 |
| 与 VT-Downsample 差异 | 待确认 | 未训练 |

### EXP-001 — VT Downsample Drawer-Occlusion Cube artifact

- 状态：已完成 artifact 存在确认；实验结论待确认
- 实验类型：训练后普通评估与 bucket 评估 artifact 记录
- 日志目录：`logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/`
- 日志目录名时间：`2026-05-30_21-33-24`；实际训练开始/结束时间待确认
- Git commit：待确认
- Git branch：待确认
- Task id：`TacEx-VT-Downsample-Drawer-Occlusion-Cube`，已在当前文档评估命令中使用；具体训练 task id 是否完全相同待确认
- Scene：`Drawer-Occlusion`
- Object：`Cube`
- Policy / method：VT Downsample，具体 class 从该 run 的 `params/agent.yaml` 确认
- Agent 配置 artifact：`logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/params/agent.yaml`
- Env 配置 artifact：`logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/params/env.yaml`
- Checkpoint：`logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt`
- Seed：42，依据该 run 的 `params/agent.yaml`
- Num envs：128，依据该 run 的 `params/env.yaml`

#### EXP-001 观测与网络

- Action space：5，依据该 run 的 `params/env.yaml` 和 `vt_box.py:OccludedGraspingVisionFourTactileBoxCfg.action_space`
- 视觉 key：`third_resnet`
- 触觉 keys：
  - `tactile_left_depth_resnet`
  - `tactile_right_depth_resnet`
  - `tactile_left_down_depth_resnet`
  - `tactile_right_down_depth_resnet`
- Critic 输入：privileged state，依据该 run 的 `params/agent.yaml`
- 网络层：`[512, 256, 128, 64]`，依据该 run 的 `params/agent.yaml`

#### EXP-001 PPO 超参数

| 字段 | 值 | 来源 |
| --- | --- | --- |
| seed | 42 | `params/agent.yaml` |
| rollouts | 128 | `params/agent.yaml` |
| learning_epochs | 4 | `params/agent.yaml` |
| mini_batches | 16 | `params/agent.yaml` |
| discount_factor | 0.99 | `params/agent.yaml` |
| lambda | 0.95 | `params/agent.yaml` |
| learning_rate | 5e-4 | `params/agent.yaml` |
| entropy_loss_scale | 0.05 | `params/agent.yaml` |
| value_loss_scale | 1.0 | `params/agent.yaml` |
| timesteps | 500000 | `params/agent.yaml` |

#### EXP-001 评估 artifact

普通评估 metrics：

```text
logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/metrics/play/play_metrics_20260531_131229.csv
```

已确认前几行包含字段：

```text
step,recent_success_rate,window_success_rate
200,0.625,0.625
400,0.59375,0.5625
600,0.6354166865348816,0.71875
```

Bucket trial metrics：

```text
logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/metrics/play_bucket/play_bucket_20260531_152044.csv
```

Bucket summary metrics：

```text
logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/metrics/play_bucket/play_bucket_20260531_152044_summary.csv
```

已确认 summary 字段：

```text
grid_index,x,y,trials,successes,success_rate,occlusion_ratio_mean
```

已确认现象：该 summary 中存在 `occlusion_ratio_mean` 为 `nan` 的记录；原因与 bbox/occlusion 设置的关系待确认，代码层风险见 `docs/DECISIONS.md` 的 ISSUE-008。

#### EXP-001 评估命令

普通评估命令：

```bash
python scripts/reinforcement_learning/skrl/play.py --task TacEx-VT-Downsample-Drawer-Occlusion-Cube --checkpoint logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt --num_envs 16 --enable_cameras
```

Bucket 评估命令：

```bash
python scripts/reinforcement_learning/skrl/play_bucket.py --task TacEx-VT-Downsample-Drawer-Occlusion-Cube --checkpoint logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt --num_envs 16 --enable_cameras
```

#### EXP-001 结果

| 指标 | 值 | 状态 |
| --- | --- | --- |
| 普通评估 recent success rate | 见 `play_metrics_20260531_131229.csv` | 已有 CSV，未在本文档汇总最终值 |
| 普通评估 window success rate | 见 `play_metrics_20260531_131229.csv` | 已有 CSV，未在本文档汇总最终值 |
| Bucket 总成功率 | 待确认 | 需要从 trial 或 summary CSV 计算 |
| 按遮挡程度分组成功率 | 待确认 | 当前已确认 summary 中 `occlusion_ratio_mean` 可为 `nan` |
| 平均回报 | 待确认 | 当前未确认对应日志字段 |

### EXP-002 — UR10 Robotiq 2F85 pick-place skrl checkpoint artifact

- 状态：已完成 artifact 迁移；训练效果待确认
- 实验类型：本机 IsaacLab skrl 训练 artifact 迁移记录
- 原始来源：`/home/tinydog/IsaacLab/logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/`
- 迁移后日志目录：`logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/`
- Task id：大概率为 `Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0`，依据 `params/env.yaml` 中 2F85 USD 和环境配置；原始 run 未直接记录 task id，待确认。
- Agent 配置 artifact：`logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/params/agent.yaml`
- Env 配置 artifact：`logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/params/env.yaml`
- Checkpoint：
  - `logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/checkpoints/best_agent.pt`
  - `logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/checkpoints/agent_20000.pt`
- Seed：42，依据 `params/agent.yaml`
- Num envs：1024，依据 `params/env.yaml`
- Observation space：27，依据 `params/env.yaml`
- Action space：4，依据 `params/env.yaml`
- Robot USD：`source/tacex_assets/tacex_assets/data/Robots/URRobotiq/ur10_robotiq_2f85.usda`

#### EXP-002 评估命令

```bash
python scripts/reinforcement_learning/skrl/play.py --task Isaac-UR10-Robotiq-2F85-Pick-Place-Direct-v0 --checkpoint logs/skrl/ur10_robotiq_pick_place_direct/2026-04-08_15-44-54_ppo_torch/checkpoints/best_agent.pt --num_envs 16
```

#### EXP-002 结果

| 指标 | 值 | 状态 |
| --- | --- | --- |
| Isaac Sim 回放 | 待确认 | 本次只迁移 artifact，未运行 play |
| 抓取/放置成功率 | 待确认 | 未找到评估 CSV |
| 平均回报 | 待确认 | 未汇总 TensorBoard event |

### EXP-003 — Real-Alignment v3 exported Actor 批量 rollout 诊断

- 状态：已完成诊断采集；该 checkpoint 不满足当前 v9 部署 contract
- Task id：`TacEx-Sim2Real-Cube-Real-Alignment-v0`
- Policy / method：PPO vision-only ResNet18 exported deterministic Actor
- Seed：42
- Num envs：32
- Policy steps：450（每个 episode 最多 150）
- Checkpoint：`logs/skrl/sim2real_cube_real_alignment/2026-07-25_22-01-32_ppo_torch_vision_only_resnet18/checkpoints/best_agent.pt`
- Exported Actor：`logs/skrl/sim2real_cube_real_alignment/2026-07-25_22-01-32_ppo_torch_vision_only_resnet18/checkpoints/exported/policy_actor_e2e_best_agent.pt`
- Contract：v3；显式 legacy diagnostic，仅复现保存的 `env.pkl`，不表示当前 v9/真机兼容
- 结果来源：
  - `metrics/sim2real_rollouts/rollout_v3_32env_450steps.npz`
  - `metrics/sim2real_rollouts/rollout_v3_32env_450steps.summary.json`
  - `metrics/sim2real_rollouts/rollout_v3_32env_450steps.analysis.json`
  - `metrics/sim2real_rollouts/rollout_v3_32env_450steps.actions.csv`

#### EXP-003 结果

| 指标 | 值 |
| --- | ---: |
| Transition 数 | 14,400 |
| 完成 episode | 98 |
| Success terminal | 7 |
| 完成 episode 成功率 | 7.14% |
| episode 第一步 mean action `[x,y,z,g]` | `[0.583,-0.307,-0.960,0.765]` |
| episode 第一步 Z 近饱和比例 | 93.08% |
| episode 第一步 gripper 近饱和比例 | 89.23% |
| 全部 transition gripper 近饱和比例 | 51.36% |
| 夹爪目标位于 80 mm 上限比例 | 76.51% |
| legacy privileged dz gate 触发比例 | 0.00% |
| 最大质心相对抬升 | 59.10 mm |

结论：该策略在仿真中已经不是可靠抓取策略。前 30 步主要输出接近最大幅度的 XYZ 运动，并持续给出正夹爪增量（张开），绝大部分时间把夹爪目标推到 80 mm；因此真机表现差不能只归因于视觉 sim-to-real gap。真机 0712 配置的 XYZ/夹爪 scale、history scale/delay 又与 v3 metadata 不一致，会进一步改变闭环行为。

### Planned Experiments

### EXP-005 — Real-Alignment RMA 教师学生蒸馏

- 状态：计划；代码路径已实现，训练未执行
- Task id：`TacEx-Sim2Real-Cube-Real-Alignment-RMA-Teacher-v0` / `...-Student-v0` / `...-Student-DR-v0` / `...-Student-Heatmap-v0` / `...-Student-Heatmap-DR-v0`
- GelSight task id：`TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Teacher-v0` / `...-GelSight-Student-v0` / `...-GelSight-Student-DR-v0` / `...-GelSight-Student-Heatmap-v0` / `...-GelSight-Student-Heatmap-DR-v0`
- Scene：Real-Alignment Clean v9
- Policy / method：200k privileged PPO Teacher + 100k single-frame visual position/contact/action/action-rate distillation
- Teacher Actor external input：`proprio[15] + history[4] + cube_xyz_root[3] + left_right_contact[2]`
- Teacher Actor network feature：上述24维 + `gripper_xyz_root_from_fk[3] + (cube-gripper)_xyz[3]` = 30维；不使用物体/末端 quaternion
- PandaHand Student deployment input：`RGB[224,224,3] + proprio[15] + history[4]`
- GelSight Student TorchScript input order：`RGB[224,224,3] + proprio[15] + history[4] + gsmini_left_rgb[96,128,3] + gsmini_right_rgb[96,128,3]`
- GelSight profile：机器人使用 Franka + 左右 GelSight Mini；RMA 接触 filter 使用 `gelpad_left/right`。Teacher 默认不渲染 `tactile_rgb`；GelSight Student 默认创建 `gsmini_left/right` 并用 tactile RGB 预测左右接触概率。Teacher manifest 记录 robot/GelSight profile 和 contact filter，禁止 GelSight 与旧 PandaHand profile 交叉蒸馏。
- Heatmap Student：保留原 ResNet18 layer4 spatial-softmax XYZ 分支；新增 ResNet18 layer3 `[N,256,14,14] -> Conv3x3+ReLU+Conv1x1 -> predicted_heatmap [N,1,14,14]`。GT 来自 `cube_position_root` 通过当前 Student 有效内参和相机 optical pose 投影到224x224像素后生成14x14 Gaussian，默认 `sigma=1.5` heatmap px；相机后方或图像外样本不计 heatmap MSE。该分支只影响训练 loss、日志和 debug overlay。GelSight Student 的 contact BCE 来自 tactile contact head。
- 评估：固定10x10 XY 网格，共100 episodes
- RMA-only reward：0.2 N接触阈值；单侧接触奖励1.5/step、双侧接触奖励3.0/step；动作变化惩罚 `-0.05 * mean(((a_t-a_{t-1}) / action_scale)^2)`；当前不包含 cube-finger 切向力惩罚。
- RMA-only gripper actuator：XYZ动作尺度为50 mm/step，夹爪总宽度动作尺度为10 mm/step；finger `effort_limit_sim=40`、`stiffness=400`、`damping=40`
- Done：RMA success 不终止 episode；仅 timeout 或严重机器人穿地碰撞终止，episode 成功统计按曾经达到 success 计算
- 验收：Teacher success >=80%；Student success >=0.9 Teacher；Student 3D RMSE <=15 mm
- checkpoint：RMA Teacher manifest v9 及更早 artifact 已失效；当前 v10 需从头训练 Teacher 后再蒸馏 Student。RMA Student artifact 当前为 v6，记录 Student 输入契约；v5 及更早 Student checkpoint 不应继续 resume/export。新增 optional head 只允许同版本内 missing keys 兼容。
- 结果：待训练与评估，当前不得写为已达标

Heatmap DR Student 推荐训练命令：

```bash
python scripts/reinforcement_learning/skrl/train_rma_student.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-RMA-Student-Heatmap-DR-v0 \
  --teacher_checkpoint <teacher.pt> \
  --num_envs 128 \
  --timesteps 100000 \
  --train_backbone_after_layer2 \
  --backbone_learning_rate 3e-5 \
  --headless
```

GelSight Heatmap DR Student 推荐训练命令：

```bash
python scripts/reinforcement_learning/skrl/train_rma_student.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Student-Heatmap-DR-v0 \
  --teacher_checkpoint <gelsight-teacher.pt> \
  --num_envs 128 \
  --timesteps 100000 \
  --train_backbone_after_layer2 \
  --backbone_learning_rate 3e-5 \
  --headless
```

GelSight Teacher 默认不渲染 tactile RGB，因此不需要启用 camera pipeline：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Teacher-v0 \
  --num_envs 256 \
  --headless
```

调试 overlay 默认关闭；需要保存少量 GT/pred center 对比图时追加：

```bash
  --heatmap_debug_interval 1000 \
  --heatmap_debug_count 4
```

新增计划实验时使用以下模板。字段不适用时写「不适用」，未确认时写「待确认」，不要删行。

```text
### EXP-XXX — 标题

- 状态：计划 / 进行中 / 已完成 / 已废弃
- 目标与对比对象：
- Task id / agent config / env config：
- 配置：seed、num envs、关键超参
- 日志目录与 checkpoint：
- 训练与评估命令：
- 结果（含来源：日志 / CSV / checkpoint / 人工记录）：
- 结论与待确认：
```
