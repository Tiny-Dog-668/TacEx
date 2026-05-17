# Franka Four-Tactile Third-Person Pretrain

This directory contains a concrete design for large-scale vision-tactile data collection in Isaac Sim.

The target setup is:

- Robot: Franka Panda
- Vision: one fixed third-person RGB camera
- Tactile: four fingertip sensors
  - `inner_left`
  - `inner_right`
  - `down_left`
  - `down_right`
- Goal: collect synchronized vision + tactile + proprioception episodes for contrastive pretraining

## Recommended Scene

The recommended first version is intentionally narrow:

- single tabletop scene
- one object per episode
- one scripted interaction primitive per episode
- no clutter in the first dataset release
- rigid objects only in the first stage

This keeps the contact distribution readable and makes it easier to debug positive and negative pairs for pretraining.

## Sensor Layout

### Third-person camera

Use one fixed overview camera as the primary vision stream.

- pose relative to env origin: `pos=(0.88, -0.06, 0.20)`
- look direction: toward the workspace center
- image size: `512 x 512`
- modality: `rgb`
- update rate: `20 Hz`

This version uses a front-facing camera placed on the far side of the table, with a slight lateral offset to keep both fingers and the object visible during pinch closure. It is meant to reduce self-occlusion from the wrist housing.

### Tactile sensors

Use four sensors on the gripper:

- `inner_left`, `inner_right` for pinch contact
- `down_left`, `down_right` for under-pad contact and slide cues

Recommended tactile outputs to store:

- raw tactile frame
- contact mask
- contact area
- max indentation or depth proxy
- per-sensor contact flag

If your simulator already produces richer tactile images, store the raw output and derive the compact features offline.

## Object Strategy

Do not build the dataset from only daily objects. The object set should be controlled first and semantic second.

The object pool in `object_pool.yaml` is divided into four roles:

- `primitive_shape`: teach curvature, flat surfaces, edges, and symmetry
- `local_geometry`: teach grooves, ridges, cavities, steps, and thin structures
- `graspable_daily`: add realistic grasp affordances
- `material_control`: separate geometry from friction and compliance

The first release should use roughly:

- 12 primitive objects
- 10 local-geometry objects
- 8 daily rigid objects
- 8 material-control variants

This gives enough diversity for pretraining without making scene management or split design unstable.

## Collection Protocol

The core unit is an interaction episode, not a single frame.

Each episode should contain:

1. pre-contact approach
2. first contact
3. contact stabilization
4. primitive execution
5. short hold or perturbation
6. release

Recommended primitive set:

- `touch`
- `press`
- `slide`
- `pinch_grasp`
- `lift_and_disturb`

Why this set:

- `touch` learns alignment between visible pose and first tactile event
- `press` learns local geometry and compliance
- `slide` learns friction and surface structure
- `pinch_grasp` learns bilateral tactile coordination
- `lift_and_disturb` learns grasp stability and incipient slip

Details are in `collection_protocol.yaml`.

## Data Fields

Store data at the episode level. A good minimal schema is:

```text
episode_id
object_id
object_family
material_id
primitive
seed
timestamps
third_person_rgb[t]
tactile_inner_left[t]
tactile_inner_right[t]
tactile_down_left[t]
tactile_down_right[t]
joint_pos[t]
joint_vel[t]
gripper_width[t]
ee_pose[t]
object_pose[t]
contact_flags[t]
quality_metrics
```

Useful quality metrics:

- valid_contact_ratio
- max_contact_area_per_sensor
- lift_success
- slip_detected
- object_left_fov
- dropped_object

## Split Design

Use three evaluation tracks:

- `id_test`: seen families, unseen specific instances
- `ood_geometry_test`: unseen local geometry patterns
- `ood_material_test`: unseen friction or compliance combinations

This prevents the encoder from succeeding by memorizing object identity only.

## Practical Recommendation

If you want a stable first dataset:

- start with `touch`, `slide`, and `pinch_grasp`
- start with primitive and local-geometry objects first
- delay deformables and clutter until the baseline encoder works

The next implementation step should be:

1. create a task config that spawns one object from `object_pool.yaml`
2. script the primitive executor in `collection_protocol.yaml`
3. write one exporter that saves synchronized episode shards

## Bootstrap Collector

The first runnable collector is `bootstrap_collector.py`.

What it does now:

- uses Franka with four tactile sensors and one third-person camera
- samples from a supported subset of primitive-shape objects
- runs scripted `touch`, `press`, `slide`, and `pinch_grasp`
- saves one compressed `npz` plus one `json` metadata file per episode

Current supported object ids:

- `sphere_xs_25`
- `sphere_s_35`
- `sphere_m_50`
- `sphere_l_65`
- `box_cube_25`
- `box_cube_40`
- `box_cube_55`
- `box_rect_60_40_30`
- `box_rect_80_30_25`
- `box_flat_50_50_20`
- `box_tall_20_20_60`
- `box_ridge_80_12_30`
- `box_slim_16_16_35`
- `cylinder_slim_18_60`
- `cylinder_short_30_60`
- `cylinder_mid_25_70`
- `cylinder_tall_25_100`
- `cylinder_thick_40_50`
- `plate_small_60_60_8`
- `plate_80_80_10`
- `plate_rect_100_60_8`

Current limitations:

- `local_geometry` and `graspable_daily` objects are not yet spawned by the collector
- `lift_and_disturb` is not implemented in the first pass
- export format is `npz` rather than `zarr`, to keep the first data pipeline simple

Example:

```bash
./tacex.sh -p source/tacex_tasks/tacex_tasks/franka_four_tactile_third_person_pretrain/bootstrap_collector.py \
  --num_episodes 20 \
  --output_dir outputs/pretrain_bootstrap \
  --primitive random \
  --headless
```

Quick smoke test:

```bash
./scripts/collect_pretrain_bootstrap_test.sh
```

GUI smoke test:

```bash
HEADLESS_FLAG= ./scripts/collect_pretrain_bootstrap_test.sh
```

The smoke test defaults are intentionally set to:

- `object_id=box_cube_40`
- `primitive=pinch_grasp`

so the gripper performs an actual closing motion near the object instead of only doing a non-contact `touch` approach.

The smoke-test script now enables mp4 export by default. After each episode you will get:

- `third_person_rgb.mp4`
- `tactile_rgb_grid.mp4`
- `tactile_depth_grid.mp4`

To disable mp4 export for a larger run:

```bash
EXPORT_MP4=0 ./scripts/collect_pretrain_bootstrap_test.sh
```
