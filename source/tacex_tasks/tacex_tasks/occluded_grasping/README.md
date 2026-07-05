# Occluded Grasping Environments

这个文件夹包含 TacEx 的 occluded-grasping 任务。当前推荐使用的可组合 task 命名格式：

```text
TacEx-{Fusion}-{Scene}-{Object}
```

## 命名规则

`Scene` 表示视觉遮挡来源：

```text
Self-Occlusion
Drawer-Occlusion
```

`Object` 表示被抓取的物体：

```text
Cylinder
Cube
Cuboid
SoftCylinder
SoftCube
SoftCuboid
```

`Fusion` 表示 policy 侧的 vision/tactile 融合方式：

```text
V
V-Downsample
V-Blur
V-Wrist
T
VT
VT-Downsample
VT-Router-MAE-Downsample
GelFusion
GelFusion-Downsample
VT-Wrist
VT-Pair
VT-Pair-GRU
CNN-Recon
VT-Down-Residual
Reliability-Stage
Reliability-Stage-Strong-Gate
Alpha
Visual-Cross-Alpha
Visual-Cross-Alpha-Downsample
Visual-Cross-Alpha-Tactile-Downsample
Tactile-Cross-Alpha
Tactile-Cross-Alpha-Downsample
Tactile-Cross-Alpha-Recon-Downsample
Tactile-Cross-Alpha-Aux-Downsample
Tactile-Cross-Alpha-Aux-GRU-Downsample
Tactile-Cross-Downsample
Tactile-Cross-Alpha-Visual-Downsample
Tactile-Cross-Alpha-Visual-PredictVisible-Downsample
Tactile-Cross-Alpha-VisualTokens-Downsample
Dual-Cross-Alpha
Dual-Cross-Alpha-Downsample
Dual-Cross-Alpha-Aux-Downsample
Dual-Cross-Downsample
Token-Self-Attn-Downsample
Alpha-Downsample
Alpha-Learnable-Tactile
Alpha-Beta
Alpha-GRU
Alpha-GRU-Beta
Alpha-Dual-GRU
Alpha-GRU-VisualReliability
Alpha-Recon
Dual-Alpha-Recon
Gate-Alpha
Hard-Gate
GRU
GRU-Downsample
GRU-Extra-Tactile
GRU-Sparsh-DownDepth
Convex
Convex-GRU
Convex-GRU-Beta
Sparsh
Policy-Token-Transformer
```

示例：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Alpha-GRU-Drawer-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras
```

软体形状使用同一命名规则，例如：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Alpha-GRU-Drawer-Occlusion-SoftCube \
  --num_envs 4 \
  --enable_cameras
```

旧 task id 仍然保留，例如 `TacEx-VT-Alpha-GRU-Self-Occlusion-Box-v0` 和
`TacEx-VT-Alpha-GRU-Self-Occlusion-Cylinder-v0`，用于兼容已有脚本和 checkpoint。

## 触觉 + 本体 Baseline

`T` 是 tactile-proprioception baseline，只把四路触觉特征和本体状态交给 actor，不使用第三视角视觉特征：

```text
proprio_obs: 18
tactile_left_depth_resnet: 256
tactile_right_depth_resnet: 256
tactile_left_down_depth_resnet: 256
tactile_right_down_depth_resnet: 256
```

policy 配置为 `agents/ppo_tactile.yaml`。动作、奖励、done/success 语义和 privileged critic 输入保持与 `VT` 环境一致。示例：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-T-Drawer-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras
```

## GelFusion 风格 RL 对比环境

`GelFusion` 和 `GelFusion-Downsample` 是对 Jiang et al. 2025 GelFusion 视触融合思路的 RL 复现分支，不是论文原版 Diffusion Policy。该分支保持当前遮挡抓取的 5 维 PPO action、奖励、done/success 语义和 privileged critic 输入不变，只改变 actor 侧观测和融合策略：

```text
third_resnet: 256
四路 tactile_*_depth_resnet: 每路 256，默认使用 ResNet tactile encoder
tactile_dynamic_stats: 8 = 4 个触觉传感器 * [binary frame-diff mean, binary frame-diff variance]
proprio_obs: 18
```

policy 配置为 `agents/ppo_vt_gelfusion.yaml`，actor class 为 `vt_gelfusion_policy.py:OccludedGraspingVTGelFusionPolicy`。融合方式是 vision-led cross attention：视觉特征作为 query，视觉和四路触觉静态特征作为 key/value；输出再与原始视觉特征、8 维动态触觉统计和本体状态拼接。

建议与现有视觉受限 baseline 在同一 `Scene/Object` 上对比：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-GelFusion-Downsample-Drawer-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras

python scripts/reinforcement_learning/skrl/play_bucket.py \
  --task TacEx-GelFusion-Downsample-Drawer-Occlusion-Cube \
  --checkpoint <gelfusion_run>/checkpoints/best_agent.pt \
  --num_envs 16 \
  --enable_cameras \
  --bucket_rounds 1 \
  --headless
```

## 固定 800 点 Bucket Play 评估

`play_bucket.py` 用于在固定物体初始位置网格上评估 checkpoint，避免 `play.py` 默认 reset 随机位置导致成功率不可直接比较。默认网格为：

```text
x: [0.54, 0.62), step 0.005  -> 16 个位置
y: [-0.12, 0.13), step 0.005 -> 50 个位置
总计 16 * 50 = 800 个 grid_index
```

每个 env 会按顺序分配位置；800 个位置为一轮，`--bucket_rounds N` 表示重复 N 轮。跑完 `800 * N` 个 episode 后脚本会自动退出。评估时会关闭动作噪声和机械臂 reset 位姿噪声，并把 `reset_jitter_max_steps` 设为 1。

Cube 示例：

```bash
python scripts/reinforcement_learning/skrl/play_bucket.py \
  --task TacEx-VT-Downsample-Drawer-Occlusion-Cube \
  --num_envs 128 \
  --enable_cameras \
  --checkpoint logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/checkpoints/best_agent.pt \
  --bucket_rounds 1 \
  --headless
```

输出默认保存在 checkpoint run 目录下：

```text
<run_dir>/metrics/play_bucket/play_bucket_<timestamp>.csv
<run_dir>/metrics/play_bucket/play_bucket_<timestamp>_summary.csv
```

其中 detail CSV 每行是一次 episode，包含 `trial_index`、`round_index`、`grid_index`、`x`、`y`、`success`、`episode_steps` 等字段；summary CSV 按 `grid_index` 汇总成功率。

注意：`play_bucket.py` 默认不在线计算 bbox 遮挡程度，因此 `occlusion_ratio` 会是 `nan`，`bbox_found` 为 0。这是为了保持评估速度。

## 离线扫描 800 点遮挡程度

`scan_bucket_occlusion.py` 只扫描固定位置的 bbox 遮挡程度，不跑 policy。生成的 `occlusion_ratio` 是遮挡程度：

```text
0.0 = 完全可见
1.0 = 完全遮挡
visual_visible_ratio = 1.0 - occlusion_ratio
```

Cube：

```bash
python scripts/occluded_grasping/scan_bucket_occlusion.py \
  --task TacEx-VT-Downsample-Drawer-Occlusion-Cube \
  --num_envs 32 \
  --enable_cameras \
  --headless \
  --output logs/bucket_occlusion/cube_800_occlusion.csv
```

Cuboid：

```bash
python scripts/occluded_grasping/scan_bucket_occlusion.py \
  --task TacEx-VT-Downsample-Drawer-Occlusion-Cuboid \
  --num_envs 32 \
  --enable_cameras \
  --headless \
  --output logs/bucket_occlusion/cuboid_800_occlusion.csv
```

这个 CSV 可以作为固定配置表复用，只要相机、drawer、物体尺寸、物体网格范围不变，就不需要每次重新扫描。

## 按遮挡程度统计成功率

先用 `play_bucket.py` 得到成功率 detail CSV，再用离线遮挡 CSV 分桶统计：

```bash
python scripts/occluded_grasping/analyze_bucket_success_by_occlusion.py \
  --play_csv logs/skrl/occluded_grasping/downsample/cube/2026-05-30_21-33-24_ppo_torch_vt_downsample_box/metrics/play_bucket/play_bucket_<timestamp>.csv \
  --occlusion_csv logs/bucket_occlusion/cube_800_occlusion.csv \
  --bin_size 0.1 \
  --title "Cube success rate by occlusion"
```

如果不指定 `--out_prefix`，输出会根据 `play_csv` 所属 run 自动放到：

```text
logs/bucket_analysis/<run_folder>/
```

并生成四个文件：

```text
play_bucket_<timestamp>_by_occlusion_summary.csv
play_bucket_<timestamp>_by_occlusion_summary.md
play_bucket_<timestamp>_by_occlusion_bar.png
play_bucket_<timestamp>_by_occlusion_table.png
```

## 纯视觉退化 Baseline

用于先评估视觉质量下降本身对任务的影响，再决定是否引入触觉融合。三组纯视觉 baseline 使用相同 policy 配置 `ppo_v.yaml`，policy 输入维度保持一致：

```text
proprio_obs 18 + third_resnet 256 = 274
```

推荐先在同一 `Scene/Object` 上对比：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-V-Self-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras

python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-V-Downsample-Self-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras

python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-V-Blur-Self-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras
```

`V-Downsample` 会在 ResNet 编码前把 third-person RGB 先降到低分辨率再上采样回相机分辨率，默认：

```text
visual_degradation_mode = downsample
visual_downsample_size = 32
```

可以通过命令行调整强度：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-V-Downsample-Self-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras \
  env.visual_downsample_size=48
```

`V-Blur` 会在 ResNet 编码前对 third-person RGB 做 Gaussian blur，默认：

```text
visual_degradation_mode = gaussian_blur
visual_blur_kernel_size = 15
visual_blur_sigma = 3.0
```

可以通过命令行调整强度：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-V-Blur-Self-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras \
  env.visual_blur_kernel_size=21 \
  env.visual_blur_sigma=5.0
```

Drawer 遮挡场景使用相同命名规则：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-V-Downsample-Drawer-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras

python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-V-Blur-Drawer-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras
```

## 降采样视觉 + 触觉 Baseline

如果纯视觉退化后性能下降，可以用下面几组 task 测试触觉是否能补偿低质量视觉：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-VT-Downsample-Drawer-Occlusion-Cuboid \
  --num_envs 64 \
  --enable_cameras

python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Alpha-Downsample-Drawer-Occlusion-Cuboid \
  --num_envs 64 \
  --enable_cameras

python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-GRU-Downsample-Drawer-Occlusion-Cuboid \
  --num_envs 64 \
  --enable_cameras

python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-VT-Router-MAE-Downsample-Drawer-Occlusion-Cuboid \
  --num_envs 64 \
  --enable_cameras
```

`VT-Downsample` 使用普通四触觉 late fusion：

```text
third_resnet 256 + tactile fused 256 + proprio 18 = 530
```

`Alpha-Downsample` 使用 alpha-gated fusion：

```text
vision: downsampled RGB -> ResNet -> 256
tactile: 4 * 256 -> 256
alpha: proprio 18 -> 1
final MLP input: 256 + 256 + 18 = 530
```

`GRU-Downsample` 使用触觉 GRU，然后直接与视觉和 proprio 拼接，不使用 alpha gate：

```text
vision branch: downsampled RGB -> ResNet -> 256
tactile branch: 4 sensors * 10-step history * 256 -> GRU -> 256
final MLP input: 256 + 256 + 18 = 530
```

`VT-Router-MAE-Downsample` 使用 downsampled RGB 视觉特征和四路触觉特征驱动 4 个 MLP experts：

```text
vision branch: downsampled RGB -> ResNet -> 256 -> 128 -> 256
tactile branch: 4 * 256 = 1024 -> 256
router input: vision 256 + tactile 256 -> 4 expert scores
experts: 4 * MLP(256 -> 256)
expert fusion: softmax/top-1 router weight * expert output -> 256
final MLP input: 256 + proprio 18 = 274
aux losses: L_load for load balancing, L_entropy for sharper routing
```

这些 downsample task 默认都使用：

```text
visual_degradation_mode = downsample
visual_downsample_size = 32
```

可以通过命令行改变视觉粗糙程度：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-GRU-Downsample-Drawer-Occlusion-Cuboid \
  --num_envs 64 \
  --enable_cameras \
  env.visual_downsample_size=48
```

## Scene 变体

`Self-Occlusion` 会关闭 drawer/cabinet 相关遮挡物：

```python
include_drawer_walls = False
include_outer_cabinet_panels = False
```

`Drawer-Occlusion` 会启用 drawer walls 和 outer cabinet panels：

```python
include_drawer_walls = True
include_outer_cabinet_panels = True
```

## Object 变体

`Cylinder` 使用原始 cylinder/can 物体：

```text
lift_reward_start_height: 0.065 m
success_height: 0.135 m
drop-after-success penalty: 150
```

`Cube` 使用 5 cm 立方体：

```text
cube side: 0.05 m
initial clearance above drawer floor: 0.001 m
drop-after-success penalty: 150
```

Cube 的 success threshold 会按立方体高度重新设置。

`Cuboid` 使用竖直长方体：

```text
size: 0.05 m x 0.05 m x 0.06 m
initial clearance above drawer floor: 0.001 m
success_height: 0.1198 m
drop-after-success penalty: 150
```

## 通用 Observation 维度

大多数 policy 使用的默认维度：

```text
proprio_obs: 18
third_resnet: 256
one tactile feature: 256
four tactile features: 4 * 256 = 1024
critic privileged state: 30
action: 5
```

第三视角相机和 wrist camera 的 camera feature 都统一为 256 维。默认 tactile encoder 是轻量 CNN，会把每个 GelSight tactile RGB 图像映射成 256 维 feature。如果 tactile encoder 改成 DINO/Sparsh，具体 tactile feature 维度可能会根据 encoder config 有所不同。

## Fusion 维度

| Fusion | policy 侧主要维度 |
|---|---|
| `V` | `third_resnet 256 + proprio 18 = 274` |
| `V-Downsample` | 与 `V` 相同，但 third-person RGB 会先降到 `32x32` 再上采样到相机分辨率后进入 ResNet |
| `V-Blur` | 与 `V` 相同，但 third-person RGB 会先经过 Gaussian blur 后进入 ResNet |
| `V-Wrist` | 与 `V` 相同，但视觉 feature 来自 wrist camera：`274` |
| `VT` | 四个 tactile features 先从 `1024 -> 256`；再与 `third_resnet 256`、`proprio 18` 拼接，最终 input 为 `530` |
| `VT-Downsample` | 与 `VT` 相同，但 third-person RGB 会先降到 `32x32` 再上采样到相机分辨率后进入 ResNet |
| `VT-Router-MAE-Downsample` | 基于 `VT-Downsample` 的低清视觉输入；policy 侧视觉 `third_resnet 256 -> 128 -> 256`，四路 tactile `1024 -> 256`；router 从 `vision 256 + tactile 256` 输出 4 个 expert 权重，4 个 MLP expert 均为 `256 -> 256`，按权重融合后与 `proprio 18` 拼接，最终 MLP input 为 `274`；训练时额外加入 `L_load` 和 `L_entropy` |
| `VT-Wrist` | 与 `VT` 相同，但视觉 feature 来自 wrist camera；最终 input 为 `530` |
| `VT-Pair` | down 两个 tactile features `2 * 256 = 512 -> 128`；inner 两个 tactile features `2 * 256 = 512 -> 128`；再与 `third_resnet 256`、`proprio 18` 拼接，最终 input 为 `256 + 128 + 128 + 18 = 530` |
| `VT-Pair-GRU` | 每个 tactile sensor 使用 `10 * 256` 作为 GRU 输入，每个 GRU 输出 `128`；down 两个 GRU latent `256 -> 128`；inner 两个 GRU latent `256 -> 128`；再与 `third_resnet 256`、`proprio 18` 拼接，最终 input 为 `530` |
| `CNN-Recon` | env 输出四路 raw tactile RGB，每路 `3 * 32 * 32`；policy 侧可训练 CNN 编码为每路 `256`，四路拼接 `1024 -> 256`；decoder 从每路 `256` 重建 `3 * 32 * 32`，重建 loss 会反传到 tactile CNN；最终 input 为 `256 + 256 + 18 = 530` |
| `VT-Down-Residual` | 主动作路径与 `VT` 相同：四路 tactile `1024 -> 256`，再与 `third_resnet 256`、`proprio 18` 拼接；额外 residual 分支只使用下侧两个 tactile `512 -> 128` 和下侧接触程度 `2` 维，硬门控为 `max(left_down_contact, right_down_contact) > 0.01`；启用时只修正 `dx/dy`，单维最大 residual 幅值为 normalized action 的 `0.2` |
| `Reliability-Stage` | `third_resnet 256`，四个 tactile features 共 `1024`，并带有辅助 reliability/stage heads |
| `Reliability-Stage-Strong-Gate` | 类似 `Reliability-Stage`，额外使用 temporal/pair fusion；temporal output 默认是 `256` |
| `Alpha` | vision `256 -> 256`；tactile `1024 -> 256`；最终 MLP input 为 `256 + 256 + 18 = 530` |
| `Visual-Cross-Alpha` | 基于 VT observation：`third_resnet` 投影为单个视觉 token `V`，四路 tactile features 分别投影为 4 个 tactile tokens；视觉 token query 四个 tactile tokens 做 cross attention 得到 `V'`；proprio 输出 `alpha`，使用 `V_mix = (1-alpha) * V + alpha * V'`；最终 MLP input 为 `V_mix 256 + proprio 18 = 274` |
| `Visual-Cross-Alpha-Downsample` | 与 `Visual-Cross-Alpha` 相同，但 third-person RGB 会先降到 `32x32` 再上采样到相机分辨率后进入 ResNet |
| `Visual-Cross-Alpha-Tactile-Downsample` | 与 `Visual-Cross-Alpha-Downsample` 相同，但最终动作 MLP 额外拼接四路 tactile 聚合特征；最终 MLP input 为 `V_mix 256 + tactile 256 + proprio 18 = 530` |
| `Tactile-Cross-Alpha` | 基于 VT observation：四路 tactile features 先聚合得到 tactile feature `T`，同时分别投影为 4 个 tactile query tokens；`third_resnet` 投影为视觉 key/value token；四个 tactile queries attend 视觉 token 得到 tactile-cross tokens，再聚合为 `T'`；proprio 输出 `alpha`，使用 `T_mix = alpha * T + (1-alpha) * T'`；最终 MLP input 为 `T_mix 256 + proprio 18 = 274` |
| `Tactile-Cross-Alpha-Downsample` | 与 `Tactile-Cross-Alpha` 相同，但 third-person RGB 会先降到 `32x32` 再上采样到相机分辨率后进入 ResNet |
| `Tactile-Cross-Alpha-Recon-Downsample` | 基于 `Tactile-Cross-Alpha-Downsample` 的融合逻辑，但 env 输出四路 raw tactile RGB，每路 `3 * 32 * 32`；policy 侧可训练 tactile CNN 将每路 raw tactile 编码为 `256`，decoder 从每路 `256` 重建 tactile RGB，重建 loss 会反传到 policy 侧 tactile CNN；最终 MLP input 为 `T_mix 256 + proprio 18 = 274` |
| `Tactile-Cross-Alpha-Aux-Downsample` | 基于 `Tactile-Cross-Alpha-Downsample`；env 复用 `Tactile-Cross-Alpha-Visual-PredictVisible-Downsample` 的离线 `x,y -> occlusion` predictor 输出伪 GT `aux_occlusion_gt = 1 - pseudo_visible_ratio`，policy 从 `third_resnet` 预测 `occlusion_pred` 并用伪 GT 做辅助监督；alpha gate input 为 `occlusion_pred 1 + tactile_contact 4 + proprio 18` |
| `Tactile-Cross-Alpha-Aux-GRU-Downsample` | 基于 `Tactile-Cross-Alpha-Aux-Downsample`；env 将四路 tactile feature 各自堆叠为 `10 * 256` 时间窗口，policy 侧每路 GRU 输出 `128` latent，再用 tactile-query visual cross attention 得到 `T'`；遮挡预测和 alpha gate 与 Aux 版本一致，最终 MLP input 为 `T_mix 256 + proprio 18 = 274` |
| `Tactile-Cross-Downsample` | 基于 `Tactile-Cross-Alpha-Downsample` 去掉 alpha gate；使用固定残差融合 `T_fused = T + T'`；最终 MLP input 为 `T_fused 256 + proprio 18 = 274` |
| `Tactile-Cross-Alpha-Visual-Downsample` | 与 `Tactile-Cross-Alpha-Downsample` 相同，但最终动作 MLP 额外拼接 direct visual feature；最终 MLP input 为 `visual 256 + T_mix 256 + proprio 18 = 530` |
| `Tactile-Cross-Alpha-Visual-PredictVisible-Downsample` | 基于 `Tactile-Cross-Alpha-Visual-Downsample`；env 用离线 `x,y -> bbox occlusion` predictor 输出 `pseudo_visible_ratio`，policy 从 `third_resnet` 预测 `visible_pred` 并用 pseudo visible 做辅助监督；alpha gate input 为 `visible_pred 1 + tactile_contact_ratio 4`，最终 MLP input 仍为 `visual 256 + T_mix 256 + proprio 18 = 530` |
| `Tactile-Cross-Alpha-VisualTokens-Downsample` | 与 `Tactile-Cross-Alpha-Visual-Downsample` 类似，但视觉由 ResNet layer4 输出 `7x7=49` 个 token；四路 tactile query attend 这 49 个 visual tokens，最终 direct visual feature 由 49 个 token 投影并平均为 `256`，最终 MLP input 仍为 `visual 256 + T_mix 256 + proprio 18 = 530` |
| `Dual-Cross-Alpha` | 同时计算 `Visual-Cross-Alpha` 的 `V_mix = (1-alpha) * V + alpha * V'` 和 `Tactile-Cross-Alpha` 的 `T_mix = alpha * T + (1-alpha) * T'`；四路 tactile 会先分别投影为 sensor tokens，再聚合为 `T`，cross 后的四路 tactile tokens 再聚合为 `T'`；最终 MLP input 为 `V_mix 256 + T_mix 256 + proprio 18 = 530` |
| `Dual-Cross-Alpha-Downsample` | 与 `Dual-Cross-Alpha` 相同，但 third-person RGB 会先降到 `32x32` 再上采样到相机分辨率后进入 ResNet |
| `Dual-Cross-Alpha-Aux-Downsample` | 与 `Dual-Cross-Alpha-Downsample` 相同，但 alpha 不使用 proprio；policy 从 `third_resnet` 预测 bbox 遮挡程度 `occlusion_pred`，从四路 tactile feature 预测接触程度并取均值 `contact_mean_pred`，再由这两个数输出 alpha；env 输出 `aux_occlusion_gt` 和 `aux_tactile_contact_gt`，通过辅助 loss 训练两个任务头；最终 MLP input 仍为 `530` |
| `Dual-Cross-Downsample` | 与 `Dual-Cross-Alpha-Downsample` 使用相同的双向 cross attention 和视觉降采样，但不使用 alpha gate；固定残差融合为 `V + V'` 和 `T + T'`，最终 MLP input 为 `256 + 256 + proprio 18 = 530` |
| `Token-Self-Attn-Downsample` | 将 `third_resnet` 和四路 tactile feature 投影成 5 个 `256` 维 token，一起做 self-attention；输出后保留视觉 token `256`，四个 tactile tokens 拼接为 `1024` 后压缩到 `256`，再拼接 proprio，最终 MLP input 为 `256 + 256 + 18 = 530` |
| `Alpha-Downsample` | 与 `Alpha` 相同，但 third-person RGB 会先降到 `32x32` 再上采样到相机分辨率后进入 ResNet |
| `Alpha-Learnable-Tactile` | vision `256 -> 256` 且不乘 alpha；tactile `1024 -> 256` 后使用 `alpha * tactile + (1-alpha) * learned_tactile_param`；最终 input 为 `530` |
| `Alpha-Beta` | vision `256 -> 256`；down tactile `2 * 256 = 512 -> 128`；inner tactile `2 * 256 = 512 -> 128`；proprio 输出 alpha/beta，按 `1-alpha-beta`、`alpha`、`beta` 加权后拼接；最终 MLP input 为 `256 + 128 + 128 + 18 = 530` |
| `Alpha-GRU` | vision `256 -> 256`；每个 tactile sensor 使用 `10 * 256` 作为 GRU 输入，每个 GRU 输出 `128`；四个 tactile latents 为 `512 -> 256`；最终 input 为 `530` |
| `Alpha-GRU-Beta` | vision `256 -> 256`；每个 tactile sensor 使用 `10 * 256` 作为 GRU 输入，每个 GRU 输出 `128`；down 两个 GRU latent `256 -> 128`；inner 两个 GRU latent `256 -> 128`；最终 input 为 `530` |
| `Alpha-Dual-GRU` | vision sequence `5 * 256 -> 256`；tactile 侧与 `Alpha-GRU` 相同；最终 input 为 `530` |
| `Alpha-GRU-VisualReliability` | base path 与 `Alpha-GRU` 相同；额外 visual reliability head：`third_resnet 256 -> hidden 128 -> 1`；alpha gate input 为 `proprio 18 + reliability 1` |
| `Alpha-Recon` | vision `256 -> 256`；inner tactile `2 * 256 = 512`；down tactile `2 * 256 = 512`；总 tactile `1024 -> 256`；最终 input 为 `530` |
| `Dual-Alpha-Recon` | vision latent `256`；down tactile latent `128`；inner tactile latent `128`；最终 MLP input 为 `256 + 128 + 128 = 512`；proprio 只用于两个 alpha gates |
| `Gate-Alpha` | vision `256 -> 256`；tactile `1024 -> 256`；最终 input 为 `256 + 256 + 18 = 530`，并带有 tactile-valid gating |
| `Hard-Gate` | 复用 `VT` policy：四个 tactile features 先从 `1024 -> 256`；再与 `third_resnet 256`、`proprio 18` 拼接，最终 input 为 `530`；没有接触的 tactile features 会在进入 policy 前被置零 |
| `GRU` | vision `256 -> 256`；每个 tactile sensor 使用 `10 * 256` 作为 GRU 输入，每个 GRU 输出 `128`；四个 tactile GRU latent `512 -> 256`；不使用 alpha gate，直接拼接得到最终 input `256 + 256 + 18 = 530` |
| `GRU-Downsample` | 与 `GRU` 相同，但 third-person RGB 会先降到 `32x32` 再上采样到相机分辨率后进入 ResNet |
| `GRU-Extra-Tactile` | 在 `GRU` 的基础上，额外取每个 tactile window 的最后一帧当前 tactile feature，四个传感器 `4 * 256 = 1024 -> 256` 后再拼接；最终 input 为 `256 + 256 + 256 + 18 = 786` |
| `GRU-Sparsh-DownDepth` | 基于 `GRU` 风格的直接拼接融合；内侧两个 tactile 使用 RGB + Sparsh `dino_vitsmall`，每个 `256`，不经过 GRU，拼接后 `512 -> 128`；下侧两个 tactile 使用 depth-CNN，每帧 `256`，每路 `10 * 256` 经过 GRU 后拼接，再 `256 -> 128`；最终 input 为 `256 + 128 + 128 + 18 = 530` |
| `Convex` | vision `256 -> 512`；tactile `1024 -> 512`；convex fusion 相加得到 `512`；最终 input 为 `512 + 18 = 530` |
| `Convex-GRU` | vision `256 -> 512`；每个 tactile sensor 使用 `10 * 256` 作为 GRU 输入，每个 GRU 输出 `128`；四个 tactile GRU latent `512 -> 512`；convex fusion 相加得到 `512`；最终 input 为 `530` |
| `Convex-GRU-Beta` | vision `256 -> 512`；每个 tactile sensor 使用 `10 * 256` 作为 GRU 输入，每个 GRU 输出 `128`；down 两个 latent `256 -> 512`；inner 两个 latent `256 -> 512`；按 `1-alpha-beta`、`alpha`、`beta` 三路相加融合；最终 input 为 `530` |
| `Sparsh` | 类似 `VT`；tactile features 来自 Sparsh encoder，每个 tactile feature 为 `256`；四个 tactile features 先从 `1024 -> 256`，最终 input 为 `530` |
| `Policy-Token-Transformer` | env 侧用冻结 ResNet18 layer4 + `AdaptiveAvgPool2d(7,7)` 预提 `vision_tokens [49,512]`，用冻结共享 tactile CNN + `AdaptiveAvgPool2d(2,2)` 预提 `tactile_tokens [16,128]`；policy 侧只做 token projection、embedding、learnable policy token、TransformerEncoder 和动作 MLP；总 token 数为 `66` |

`Policy-Token-Transformer` 的 raw image 不进入 skrl rollout memory；memory 只存 `vision_tokens`、`tactile_tokens`、proprio 和 critic 状态。视觉 ResNet18 和 tactile CNN 在 env 侧 `no_grad/eval` 下运行，不通过 PPO loss 更新。

## 常用变体

当前实验中较常用的变体：

```text
TacEx-Alpha-GRU-Self-Occlusion-Cylinder
TacEx-Alpha-GRU-Self-Occlusion-Cube
TacEx-Alpha-GRU-Self-Occlusion-Cuboid
TacEx-Alpha-GRU-Drawer-Occlusion-Cylinder
TacEx-Alpha-GRU-Drawer-Occlusion-Cube
TacEx-V-Downsample-Self-Occlusion-Cube
TacEx-V-Blur-Self-Occlusion-Cube
TacEx-V-Downsample-Drawer-Occlusion-Cube
TacEx-V-Blur-Drawer-Occlusion-Cube
TacEx-VT-Downsample-Drawer-Occlusion-Cuboid
TacEx-VT-Router-MAE-Downsample-Drawer-Occlusion-Cuboid
TacEx-Alpha-Downsample-Drawer-Occlusion-Cuboid
TacEx-GRU-Downsample-Drawer-Occlusion-Cuboid
TacEx-Alpha-GRU-VisualReliability-Self-Occlusion-Cube
TacEx-Alpha-GRU-VisualReliability-Drawer-Occlusion-Cube
TacEx-VT-Pair-Self-Occlusion-Cube
TacEx-VT-Pair-GRU-Self-Occlusion-Cube
TacEx-CNN-Recon-Self-Occlusion-Cube
TacEx-VT-Down-Residual-Self-Occlusion-Cube
TacEx-Alpha-Beta-Self-Occlusion-Cube
TacEx-Alpha-Learnable-Tactile-Self-Occlusion-Cube
TacEx-Alpha-GRU-Beta-Self-Occlusion-Cube
TacEx-GRU-Self-Occlusion-Cube
TacEx-GRU-Extra-Tactile-Self-Occlusion-Cube
TacEx-GRU-Sparsh-DownDepth-Self-Occlusion-Cube
TacEx-Convex-Self-Occlusion-Cube
TacEx-Convex-GRU-Self-Occlusion-Cube
TacEx-Convex-GRU-Beta-Self-Occlusion-Cube
TacEx-Policy-Token-Transformer-Self-Occlusion-Cube
```

对 `GRU` 来说，没有 alpha gate：

```text
vision branch: 256 -> 256
tactile branch: 4 sensors * 10-step history * 256 feature input
tactile GRU output: 4 * 128 = 512
tactile projection: 512 -> 256
proprio: 18
final MLP input: concat(vision, tactile, proprio) = 530
```

对 `GRU-Extra-Tactile` 来说，会同时使用 temporal tactile 和当前 tactile：

```text
vision branch: 256 -> 256
tactile GRU branch: 4 sensors * 10-step history * 256 -> 4 * 128 -> 256
current tactile branch: last frame from each tactile window, 4 * 256 = 1024 -> 256
proprio: 18
final MLP input: concat(vision, tactile_gru, tactile_current, proprio) = 786
```

对 `Alpha-GRU` 来说，关键维度是：

```text
vision branch: 256 -> 256
tactile branch: 4 sensors * 10-step history * 256 feature input
tactile GRU output: 4 * 128 = 512
tactile projection: 512 -> 256
proprio: 18
final MLP input: 256 + 256 + 18 = 530
```

对 `Alpha-Learnable-Tactile` 来说，alpha 只作用在 tactile 分支，不改变 vision 权重：

```text
vision: 256 -> 256
tactile: 4 * 256 = 1024 -> 256
alpha: proprio 18 -> 1
learned_tactile_param: learnable 256-d vector
tactile_mixed = alpha * tactile + (1 - alpha) * learned_tactile_param
final MLP input: concat(vision, tactile_mixed, proprio) = 530
```

对 `Convex` 来说，最终 MLP input 更小：

```text
vision: 256 -> 512
tactile: 1024 -> 512
convex fusion: 512
proprio: 18
final MLP input: 530
```

对 `Convex-GRU` 来说：

```text
vision: 256 -> 512
tactile: 4 sensors * 10-step history * 256 feature input
tactile GRU output: 4 * 128 = 512
tactile projection: 512 -> 512
convex fusion: 512
proprio: 18
final MLP input: 530
```

对 `Convex-GRU-Beta` 来说：

```text
vision: 256 -> 512
down tactile: 2 sensors * GRU latent 128 = 256 -> 512
inner tactile: 2 sensors * GRU latent 128 = 256 -> 512
fusion: (1 - alpha - beta) * vision + alpha * down + beta * inner
proprio: 18
final MLP input: 530
```

对 `Hard-Gate` 来说，policy 结构与 `VT` 相同，但 tactile features 会先经过环境层面的 hard gating：

```text
tactile: 4 * 256 = 1024 -> 256
vision: 256
proprio: 18
final MLP input: 530
no-contact tactile features: zeroed before policy
```

对 `Gate-Alpha` 来说，alpha gate 会额外使用 `tactile_valid`：

```text
vision: 256 -> 256
tactile: 1024 -> 256
proprio: 18
final MLP input: 530
```
