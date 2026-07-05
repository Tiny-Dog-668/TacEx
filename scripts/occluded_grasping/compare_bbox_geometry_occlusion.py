from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections.abc import Iterable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Compare per-env Replicator bbox occlusion with a static-geometry ray/AABB approximation."
)
parser.add_argument("--task", type=str, required=True, help="Occluded grasping task id.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to scan in parallel.")
parser.add_argument("--output", type=str, default=None, help="CSV output path.")
parser.add_argument("--x_min", type=float, default=0.54)
parser.add_argument("--x_max", type=float, default=0.62)
parser.add_argument("--y_min", type=float, default=-0.12)
parser.add_argument("--y_max", type=float, default=0.13)
parser.add_argument("--step", type=float, default=0.005)
parser.add_argument(
    "--include_endpoint",
    action="store_true",
    default=False,
    help="Include x/y max endpoints. Disabled by default, giving 16*50=800 positions for default bounds.",
)
parser.add_argument("--warmup_steps", type=int, default=3, help="Render/update frames before reading bbox.")
parser.add_argument("--debug_samples", type=int, default=0, help="Print raw bbox rows from the first read.")
parser.add_argument(
    "--object_sample_axis",
    type=int,
    default=5,
    help="Number of sample coordinates per object bounding-box axis. 5 gives 98 surface samples.",
)
parser.add_argument(
    "--ray_margin",
    type=float,
    default=1.0e-4,
    help="Ignore intersections this close to the sampled object point along the camera ray.",
)
parser.add_argument(
    "--exclude_floor_occluder",
    action="store_true",
    default=False,
    help="Do not count the drawer floor as a static occluder. By default it is included to better match bbox output.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import tacex_tasks  # noqa: F401


def _base_env(env):
    return getattr(env, "unwrapped", env)


def _build_grid() -> list[tuple[int, float, float]]:
    if args_cli.step <= 0.0:
        raise ValueError(f"--step must be positive, got {args_cli.step}")

    def axis_values(v_min: float, v_max: float) -> list[float]:
        span = v_max - v_min
        if args_cli.include_endpoint:
            count = int(math.floor(span / args_cli.step + 1e-9)) + 1
        else:
            count = int(math.floor(span / args_cli.step + 1e-9))
        return [round(v_min + i * args_cli.step, 10) for i in range(count)]

    xs = axis_values(args_cli.x_min, args_cli.x_max)
    ys = axis_values(args_cli.y_min, args_cli.y_max)
    return [(idx, x, y) for idx, (x, y) in enumerate((x, y) for x in xs for y in ys)]


def _extract_env_id_from_path(path: str) -> int | None:
    prefix = "/World/envs/env_"
    if not path.startswith(prefix):
        return None
    suffix = path[len(prefix) :]
    digits = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))


def _extract_occlusion_ratio(row) -> float | None:
    if isinstance(row, dict):
        value = row.get("occlusionRatio")
    else:
        try:
            value = row["occlusionRatio"]
        except Exception:
            value = getattr(row, "occlusionRatio", None)
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ratio) or ratio < 0.0:
        return None
    return max(0.0, min(1.0, ratio))


def _unpack_bbox_output(output):
    if isinstance(output, dict):
        data = output.get("data")
        info = output.get("info", {})
        if not info:
            info = {key: value for key, value in output.items() if key != "data"}
        return data, info
    return output, {}


class BBoxReader:
    def __init__(self, env):
        self.env = _base_env(env)
        self.records: list[tuple[int, object, str]] = []
        self.debug_printed = False

    def initialize(self) -> None:
        import omni.replicator.core as rep  # type: ignore

        camera = getattr(self.env, "third_person_camera", None)
        if camera is None:
            raise RuntimeError("Environment has no third_person_camera")
        view = getattr(camera, "_view", None)
        camera_prim_paths = getattr(view, "prim_paths", None)
        if not isinstance(camera_prim_paths, (tuple, list)) or not camera_prim_paths:
            raise RuntimeError("Could not resolve per-env third_person_camera prim paths")

        width = int(getattr(getattr(camera, "cfg", None), "width", 224))
        height = int(getattr(getattr(camera, "cfg", None), "height", 224))
        for camera_prim_path in camera_prim_paths:
            camera_prim_path = str(camera_prim_path)
            env_id = _extract_env_id_from_path(camera_prim_path)
            if env_id is None:
                continue
            render_product = rep.create.render_product(camera_prim_path, resolution=(width, height))
            if not isinstance(render_product, str):
                render_product = render_product.path
            annotator = rep.AnnotatorRegistry.get_annotator(
                "bounding_box_3d",
                init_params={"semanticTypes": ["class"]},
            )
            try:
                annotator.attach(render_product)
            except Exception:
                annotator.attach([render_product])
            self.records.append((env_id, annotator, str(render_product)))
        if not self.records:
            raise RuntimeError("No per-env bbox annotators were created")
        print(f"[INFO] BBox annotators attached to {len(self.records)} per-env third-person render products.")

    def read(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = int(self.env.num_envs)
        device = self.env.device
        occlusion = torch.full((num_envs,), float("nan"), dtype=torch.float32, device=device)
        found = torch.zeros((num_envs,), dtype=torch.bool, device=device)

        debug_remaining = args_cli.debug_samples if not self.debug_printed else 0
        debug_rows_printed = 0
        for record_env_id, annotator, _render_product in self.records:
            try:
                bbox_data, bbox_info = _unpack_bbox_output(annotator.get_data())
            except Exception:
                continue
            prim_paths = bbox_info.get("primPaths", [])
            if bbox_data is None or not isinstance(prim_paths, Iterable):
                continue
            prim_paths = [str(path) for path in prim_paths]
            if debug_remaining > 0 and debug_rows_printed == 0:
                print(
                    f"[INFO] BBox debug: per_env_render_products={len(self.records)}, "
                    f"first_render_product_rows={len(prim_paths)}"
                )
            for idx, path in enumerate(prim_paths[:debug_remaining]):
                try:
                    row = bbox_data[idx]
                except Exception:
                    row = None
                print(f"[INFO] BBox debug env {record_env_id} row {idx}: path={path} row={row}")
                debug_rows_printed += 1
            debug_remaining = max(0, debug_remaining - len(prim_paths))

            for row_idx, prim_path in enumerate(prim_paths):
                env_id = _extract_env_id_from_path(prim_path)
                if env_id is None or env_id != record_env_id or env_id >= num_envs:
                    continue
                can_prefix = f"/World/envs/env_{env_id}/can"
                if prim_path != can_prefix and not prim_path.startswith(f"{can_prefix}/"):
                    continue
                try:
                    ratio = _extract_occlusion_ratio(bbox_data[row_idx])
                except Exception:
                    ratio = None
                if ratio is not None:
                    current = occlusion[env_id]
                    occlusion[env_id] = ratio if torch.isnan(current) else max(float(current.item()), ratio)
                    found[env_id] = True
            if debug_rows_printed >= args_cli.debug_samples:
                self.debug_printed = True
        return occlusion, found

    def close(self) -> None:
        for _env_id, annotator, render_product in self.records:
            try:
                annotator.detach([render_product])
            except Exception:
                try:
                    annotator.detach(render_product)
                except Exception:
                    pass
        self.records = []


class GeometryOcclusionEstimator:
    """Approximate drawer/cabinet occlusion by tracing object samples against static panel AABBs."""

    def __init__(self, env):
        self.env = _base_env(env)
        self.device = self.env.device
        self.sample_local = self._build_object_samples()
        self.panel_specs = self._build_panel_specs()
        print(
            f"[INFO] Geometry estimator: object_samples={self.sample_local.shape[0]}, "
            f"static_panels={len(self.panel_specs)}"
        )

    def _build_object_samples(self) -> torch.Tensor:
        cfg = self.env.cfg
        n = max(2, int(args_cli.object_sample_axis))
        object_name = str(args_cli.task).split("-")[-1].lower()
        if object_name == "cube":
            side = float(getattr(cfg, "cube_side", 0.05))
            dims = (side, side, side)
        elif object_name == "cuboid":
            dims = tuple(float(v) for v in getattr(cfg, "cuboid_size", (0.05, 0.05, 0.06)))
        else:
            radius = float(getattr(cfg, "can_radius", 0.03))
            height = float(getattr(cfg, "can_height", 0.07))
            dims = (2.0 * radius, 2.0 * radius, height)

        axes = [torch.linspace(-0.5 * dim, 0.5 * dim, n, device=self.device) for dim in dims]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        eps = 1.0e-6
        surface = (
            torch.isclose(grid[:, 0].abs(), torch.tensor(0.5 * dims[0], device=self.device), atol=eps)
            | torch.isclose(grid[:, 1].abs(), torch.tensor(0.5 * dims[1], device=self.device), atol=eps)
            | torch.isclose(grid[:, 2].abs(), torch.tensor(0.5 * dims[2], device=self.device), atol=eps)
        )
        return grid[surface].contiguous()

    def _build_panel_specs(self) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        cfg = self.env.cfg
        names: list[str] = []
        if not bool(args_cli.exclude_floor_occluder):
            names.append("box_floor")
        if bool(getattr(cfg, "include_drawer_walls", True)):
            names += ["box_wall_front", "box_wall_back", "box_wall_left", "box_wall_right", "drawer_handle"]
        if bool(getattr(cfg, "include_outer_cabinet_panels", True)):
            names += [
                "drawer_cabinet_left_side",
                "drawer_cabinet_right_side",
                "drawer_cabinet_back",
                "drawer_cabinet_top",
            ]

        specs = []
        for name in names:
            asset_cfg = getattr(cfg, name, None)
            if asset_cfg is None:
                continue
            spawn = getattr(asset_cfg, "spawn", None)
            size = getattr(spawn, "size", None)
            init_state = getattr(asset_cfg, "init_state", None)
            pos = getattr(init_state, "pos", None)
            if size is None or pos is None:
                continue
            center = torch.tensor(tuple(float(v) for v in pos), device=self.device, dtype=torch.float32)
            half = 0.5 * torch.tensor(tuple(float(v) for v in size), device=self.device, dtype=torch.float32)
            specs.append((name, center, half))
        return specs

    def _camera_pos_w(self) -> torch.Tensor:
        camera = getattr(self.env, "third_person_camera", None)
        data = getattr(camera, "data", None)
        pos_w = getattr(data, "pos_w", None)
        if isinstance(pos_w, torch.Tensor):
            return pos_w.to(device=self.device, dtype=torch.float32)

        offset = getattr(getattr(getattr(self.env.cfg, "third_person_camera", None), "offset", None), "pos", None)
        if offset is None:
            raise RuntimeError("Could not resolve third-person camera world positions")
        offset_t = torch.tensor(tuple(float(v) for v in offset), device=self.device, dtype=torch.float32)
        return self.env.scene.env_origins + offset_t.unsqueeze(0)

    def _panel_aabbs_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.panel_specs:
            empty = torch.empty((self.env.num_envs, 0, 3), device=self.device)
            return empty, empty
        centers = torch.stack([center for _name, center, _half in self.panel_specs], dim=0)
        halves = torch.stack([half for _name, _center, half in self.panel_specs], dim=0)
        centers_w = self.env.scene.env_origins[:, None, :] + centers[None, :, :]
        return centers_w - halves[None, :, :], centers_w + halves[None, :, :]

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = int(self.env.num_envs)
        if not self.panel_specs:
            visible = torch.ones((num_envs,), dtype=torch.float32, device=self.device)
            return visible, torch.zeros_like(visible)

        cam = self._camera_pos_w()
        obj_pos = self.env._can.data.root_pos_w.to(device=self.device, dtype=torch.float32)
        points = obj_pos[:, None, :] + self.sample_local[None, :, :]
        box_min, box_max = self._panel_aabbs_w()
        occluded = self._ray_hits_any_box(cam, points, box_min, box_max)
        occlusion = occluded.to(torch.float32).mean(dim=1)
        visible = 1.0 - occlusion
        return visible.clamp(0.0, 1.0), occlusion.clamp(0.0, 1.0)

    def _ray_hits_any_box(
        self,
        ray_origin: torch.Tensor,
        points: torch.Tensor,
        box_min: torch.Tensor,
        box_max: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1.0e-8
        origin = ray_origin[:, None, None, :]
        direction = points[:, :, None, :] - origin
        safe_direction = torch.where(direction.abs() < eps, torch.full_like(direction, eps), direction)
        t1 = (box_min[:, None, :, :] - origin) / safe_direction
        t2 = (box_max[:, None, :, :] - origin) / safe_direction
        t_near = torch.minimum(t1, t2).amax(dim=-1)
        t_far = torch.maximum(t1, t2).amin(dim=-1)
        hits = (t_far >= torch.clamp(t_near, min=0.0)) & (t_near < (1.0 - float(args_cli.ray_margin))) & (t_far > 0.0)
        return hits.any(dim=-1)


def _set_semantic_tag(env_cfg) -> None:
    try:
        env_cfg.can = env_cfg.can.replace(spawn=env_cfg.can.spawn.replace(semantic_tags=[("class", "can")]))
    except Exception:
        try:
            env_cfg.can.spawn.semantic_tags = [("class", "can")]
        except Exception:
            pass


def _object_root_z(base_env) -> float:
    cfg = getattr(base_env, "cfg", None)
    if cfg is not None and hasattr(cfg, "can_reset_root_z"):
        return float(getattr(cfg, "can_reset_root_z"))
    return float(base_env._can.data.default_root_state[0, 2].item())


def _assign_positions(env, batch: list[tuple[int, float, float]]) -> torch.Tensor:
    base_env = _base_env(env)
    env_ids = torch.arange(len(batch), device=base_env.device, dtype=torch.long)
    obj = base_env._can
    state = obj.data.default_root_state[env_ids].clone()
    state[:, 0] = torch.tensor([row[1] for row in batch], device=base_env.device, dtype=state.dtype)
    state[:, 1] = torch.tensor([row[2] for row in batch], device=base_env.device, dtype=state.dtype)
    state[:, 2] = _object_root_z(base_env)
    state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=base_env.device, dtype=state.dtype)
    state[:, :3] += base_env.scene.env_origins[env_ids]
    obj.write_root_state_to_sim(state, env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros((len(batch), 6), device=base_env.device), env_ids=env_ids)
    return env_ids


def _warmup(base_env, steps: int) -> None:
    for _ in range(max(0, int(steps))):
        base_env.scene.write_data_to_sim()
        base_env.sim.step(render=True)
        base_env.scene.update(dt=base_env.physics_dt)


def main() -> None:
    grid = _build_grid()
    if not grid:
        raise ValueError("Empty grid")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if hasattr(env_cfg, "action_noise_scale"):
        env_cfg.action_noise_scale = 0.0
    if hasattr(env_cfg, "robot_joint_pos_noise"):
        env_cfg.robot_joint_pos_noise = 0.0
    if hasattr(env_cfg, "robot_joint_vel_noise"):
        env_cfg.robot_joint_vel_noise = 0.0
    if hasattr(env_cfg, "reset_jitter_max_steps"):
        env_cfg.reset_jitter_max_steps = 1
    _set_semantic_tag(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = _base_env(env)
    bbox_reader = BBoxReader(env)
    geom_estimator = None

    out_path = args_cli.output
    if out_path is None:
        safe_task = args_cli.task.replace("/", "_").replace(":", "_")
        out_dir = os.path.join("logs", "bucket_occlusion")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{safe_task}_bbox_vs_geom_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    try:
        env.reset()
        _warmup(base_env, args_cli.warmup_steps)
        bbox_reader.initialize()
        geom_estimator = GeometryOcclusionEstimator(env)

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "grid_index",
                    "x",
                    "y",
                    "bbox_occlusion_ratio",
                    "bbox_visible_ratio",
                    "bbox_found",
                    "geom_occlusion_ratio",
                    "geom_visible_ratio",
                    "abs_occlusion_error",
                    "env_id",
                ]
            )

            completed = 0
            error_values: list[float] = []
            for start in range(0, len(grid), int(base_env.num_envs)):
                batch = grid[start : start + int(base_env.num_envs)]
                env_ids = _assign_positions(env, batch)
                _warmup(base_env, args_cli.warmup_steps)
                bbox_occ, bbox_found = bbox_reader.read()
                geom_visible, geom_occ = geom_estimator.compute()
                for local_idx, (grid_index, x, y) in enumerate(batch):
                    env_id = int(env_ids[local_idx].item())
                    bbox_value = float(bbox_occ[env_id].item())
                    geom_value = float(geom_occ[env_id].item())
                    found = bool(bbox_found[env_id].item())
                    abs_error = abs(bbox_value - geom_value) if found and math.isfinite(bbox_value) else float("nan")
                    if math.isfinite(abs_error):
                        error_values.append(abs_error)
                    writer.writerow(
                        [
                            grid_index,
                            x,
                            y,
                            bbox_value,
                            1.0 - bbox_value if math.isfinite(bbox_value) else float("nan"),
                            int(found),
                            geom_value,
                            float(geom_visible[env_id].item()),
                            abs_error,
                            env_id,
                        ]
                    )
                    completed += 1
                f.flush()
                if error_values:
                    mean_error = sum(error_values) / len(error_values)
                    print(f"[INFO] Progress: {completed}/{len(grid)} mean_abs_error={mean_error:.4f}")
                else:
                    print(f"[INFO] Progress: {completed}/{len(grid)} mean_abs_error=nan")
        print(f"[INFO] Saved bbox-vs-geometry occlusion comparison: {out_path}")
    finally:
        bbox_reader.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
