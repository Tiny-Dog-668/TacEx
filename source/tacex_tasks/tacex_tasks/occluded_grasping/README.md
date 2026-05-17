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
```

`Fusion` 表示 policy 侧的 vision/tactile 融合方式：

```text
V
V-Wrist
VT
VT-Wrist
VT-Pair
VT-Pair-GRU
Mixed
Mixed-Sensor-Gate
Reliability-Stage
Reliability-Stage-Strong-Gate
Alpha
Alpha-Beta
Alpha-GRU
Alpha-GRU-Beta
Alpha-Dual-GRU
Alpha-GRU-Task-Residual
Alpha-GRU-VisualReliability
Alpha-Recon
Dual-Alpha-Recon
Gate-Alpha
Hard-Gate
GRU
Convex
Convex-GRU
Convex-GRU-Beta
Sparsh
Sparsh-Cross
Cross
```

示例：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Alpha-GRU-Drawer-Occlusion-Cube \
  --num_envs 4 \
  --enable_cameras
```

另一个示例：

```bash
python scripts/reinforcement_learning/skrl/train.py \
  --task TacEx-Cross-Self-Occlusion-Cylinder \
  --num_envs 4 \
  --enable_cameras
```

旧 task id 仍然保留，例如 `TacEx-VT-Alpha-GRU-Self-Occlusion-Box-v0` 和
`TacEx-VT-Alpha-GRU-Self-Occlusion-Cylinder-v0`，用于兼容已有脚本和 checkpoint。

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
| `V-Wrist` | 与 `V` 相同，但视觉 feature 来自 wrist camera：`274` |
| `VT` | 四个 tactile features 先从 `1024 -> 256`；再与 `third_resnet 256`、`proprio 18` 拼接，最终 input 为 `530` |
| `VT-Wrist` | 与 `VT` 相同，但视觉 feature 来自 wrist camera；最终 input 为 `530` |
| `VT-Pair` | down 两个 tactile features `2 * 256 = 512 -> 128`；inner 两个 tactile features `2 * 256 = 512 -> 128`；再与 `third_resnet 256`、`proprio 18` 拼接，最终 input 为 `256 + 128 + 128 + 18 = 530` |
| `VT-Pair-GRU` | 每个 tactile sensor 使用 `10 * 256` 作为 GRU 输入，每个 GRU 输出 `128`；down 两个 GRU latent `256 -> 128`；inner 两个 GRU latent `256 -> 128`；再与 `third_resnet 256`、`proprio 18` 拼接，最终 input 为 `530` |
| `Mixed` | `third_resnet 256`；tactile tokens 通常先投影到 `128`，再用于 residual policy |
| `Mixed-Sensor-Gate` | 与 `Mixed` 类似，额外加入 per-sensor/global gate 输入 |
| `Reliability-Stage` | `third_resnet 256`，四个 tactile features 共 `1024`，并带有辅助 reliability/stage heads |
| `Reliability-Stage-Strong-Gate` | 类似 `Reliability-Stage`，额外使用 temporal/pair fusion；temporal output 默认是 `256` |
| `Alpha` | vision `256 -> 256`；tactile `1024 -> 256`；最终 MLP input 为 `256 + 256 + 18 = 530` |
| `Alpha-Beta` | vision `256 -> 256`；down tactile `2 * 256 = 512 -> 128`；inner tactile `2 * 256 = 512 -> 128`；proprio 输出 alpha/beta，按 `1-alpha-beta`、`alpha`、`beta` 加权后拼接；最终 MLP input 为 `256 + 128 + 128 + 18 = 530` |
| `Alpha-GRU` | vision `256 -> 256`；每个 tactile sensor 使用 `10 * 256` 作为 GRU 输入，每个 GRU 输出 `128`；四个 tactile latents 为 `512 -> 256`；最终 input 为 `530` |
| `Alpha-GRU-Beta` | vision `256 -> 256`；每个 tactile sensor 使用 `10 * 256` 作为 GRU 输入，每个 GRU 输出 `128`；down 两个 GRU latent `256 -> 128`；inner 两个 GRU latent `256 -> 128`；最终 input 为 `530` |
| `Alpha-Dual-GRU` | vision sequence `5 * 256 -> 256`；tactile 侧与 `Alpha-GRU` 相同；最终 input 为 `530` |
| `Alpha-GRU-Task-Residual` | base path 与 `Alpha-GRU` 相同，最终 input 为 `530`；inner/bottom residual branches 使用 `128` 维 tactile latents |
| `Alpha-GRU-VisualReliability` | base path 与 `Alpha-GRU` 相同；额外 visual reliability head：`third_resnet 256 -> hidden 128 -> 1`；alpha gate input 为 `proprio 18 + reliability 1` |
| `Alpha-Recon` | vision `256 -> 256`；inner tactile `2 * 256 = 512`；down tactile `2 * 256 = 512`；总 tactile `1024 -> 256`；最终 input 为 `530` |
| `Dual-Alpha-Recon` | vision latent `256`；down tactile latent `128`；inner tactile latent `128`；最终 MLP input 为 `256 + 128 + 128 = 512`；proprio 只用于两个 alpha gates |
| `Gate-Alpha` | vision `256 -> 256`；tactile `1024 -> 256`；最终 input 为 `256 + 256 + 18 = 530`，并带有 tactile-valid gating |
| `Hard-Gate` | 复用 `VT` policy：四个 tactile features 先从 `1024 -> 256`；再与 `third_resnet 256`、`proprio 18` 拼接，最终 input 为 `530`；没有接触的 tactile features 会在进入 policy 前被置零 |
| `GRU` | vision `256 -> 256`；每个 tactile sensor 使用 `10 * 256` 作为 GRU 输入，每个 GRU 输出 `128`；四个 tactile GRU latent `512 -> 256`；不使用 alpha gate，直接拼接得到最终 input `256 + 256 + 18 = 530` |
| `Convex` | vision `256 -> 512`；tactile `1024 -> 512`；convex fusion 相加得到 `512`；最终 input 为 `512 + 18 = 530` |
| `Convex-GRU` | vision `256 -> 512`；每个 tactile sensor 使用 `10 * 256` 作为 GRU 输入，每个 GRU 输出 `128`；四个 tactile GRU latent `512 -> 512`；convex fusion 相加得到 `512`；最终 input 为 `530` |
| `Convex-GRU-Beta` | vision `256 -> 512`；每个 tactile sensor 使用 `10 * 256` 作为 GRU 输入，每个 GRU 输出 `128`；down 两个 latent `256 -> 512`；inner 两个 latent `256 -> 512`；按 `1-alpha-beta`、`alpha`、`beta` 三路相加融合；最终 input 为 `530` |
| `Sparsh` | 类似 `VT`；tactile features 来自 Sparsh encoder，每个 tactile feature 为 `256`；四个 tactile features 先从 `1024 -> 256`，最终 input 为 `530` |
| `Sparsh-Cross` | token-based cross attention：camera tokens 为 `4 * 5 = 20`，token dim 为 `256`；tactile tokens 为 `4 * (3 * 4) = 48`；policy embed dim 默认是 `512` |
| `Cross` | token-based cross attention，camera token dim 为 `256`；policy embed dim 默认是 `512` |

## 常用变体

当前实验中较常用的变体：

```text
TacEx-Alpha-GRU-Self-Occlusion-Cylinder
TacEx-Alpha-GRU-Self-Occlusion-Cube
TacEx-Alpha-GRU-Self-Occlusion-Cuboid
TacEx-Alpha-GRU-Drawer-Occlusion-Cylinder
TacEx-Alpha-GRU-Drawer-Occlusion-Cube
TacEx-Alpha-GRU-VisualReliability-Self-Occlusion-Cube
TacEx-Alpha-GRU-VisualReliability-Drawer-Occlusion-Cube
TacEx-VT-Pair-Self-Occlusion-Cube
TacEx-VT-Pair-GRU-Self-Occlusion-Cube
TacEx-Alpha-Beta-Self-Occlusion-Cube
TacEx-Alpha-GRU-Beta-Self-Occlusion-Cube
TacEx-GRU-Self-Occlusion-Cube
TacEx-Convex-Self-Occlusion-Cube
TacEx-Convex-GRU-Self-Occlusion-Cube
TacEx-Convex-GRU-Beta-Self-Occlusion-Cube
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

对 `Alpha-GRU` 来说，关键维度是：

```text
vision branch: 256 -> 256
tactile branch: 4 sensors * 10-step history * 256 feature input
tactile GRU output: 4 * 128 = 512
tactile projection: 512 -> 256
proprio: 18
final MLP input: 256 + 256 + 18 = 530
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
