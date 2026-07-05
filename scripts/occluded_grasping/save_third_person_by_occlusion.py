"""Save third-person camera images grouped by object occlusion level over the bucket grid."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections.abc import Iterable
from pathlib import Path

from isaaclab.app import AppLauncher


def _extend_repo_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_root = repo_root / "source"
    for package_root in (
        source_root / "tacex_tasks",
        source_root / "tacex",
        source_root / "tacex_assets",
        source_root / "tacex_uipc",
    ):
        package_root_str = str(package_root)
        if package_root.is_dir() and package_root_str not in sys.path:
            sys.path.insert(0, package_root_str)


parser = argparse.ArgumentParser(
    description="Save third-person RGB examples for different bbox occlusion levels on the fixed bucket grid."
)
parser.add_argument(
    "--task",
    type=str,
    default="TacEx-VT-Downsample-Drawer-Occlusion-Cuboid",
    help="Task ID to instantiate.",
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to scan in parallel.")
parser.add_argument("--output_dir", type=str, default="logs/occlusion_third_person_images")
parser.add_argument("--occlusion_csv", type=str, default=None, help="Optional scan_bucket_occlusion.py CSV to reuse.")
parser.add_argument("--samples_per_bin", type=int, default=3, help="Maximum saved images per occlusion bin.")
parser.add_argument("--bin_size", type=float, default=0.1, help="Occlusion bin size.")
parser.add_argument("--camera_width", type=int, default=None, help="Override third-person camera render width.")
parser.add_argument("--camera_height", type=int, default=None, help="Override third-person camera render height.")
parser.add_argument(
    "--save_scale",
    type=float,
    default=1.0,
    help="Optional image upsampling factor applied before saving. This does not change render quality.",
)
parser.add_argument("--x_min", type=float, default=0.54)
parser.add_argument("--x_max", type=float, default=0.62)
parser.add_argument("--y_min", type=float, default=-0.12)
parser.add_argument("--y_max", type=float, default=0.13)
parser.add_argument("--step", type=float, default=0.005)
parser.add_argument(
    "--include_endpoint",
    action="store_true",
    default=False,
    help="Include x/y max endpoints. Disabled by default, giving 16*50=800 positions.",
)
parser.add_argument("--warmup_steps", type=int, default=3, help="Render/update frames after assigning positions.")
parser.add_argument("--debug_samples", type=int, default=0, help="Print raw bbox rows from the first read.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from isaaclab_tasks.utils import parse_env_cfg

_extend_repo_pythonpath()
import tacex_tasks  # noqa: F401, E402


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


def _load_occlusion_csv(path: str | None) -> dict[int, tuple[float, bool]]:
    if not path:
        return {}
    out: dict[int, tuple[float, bool]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                grid_index = int(row["grid_index"])
                ratio = float(row["occlusion_ratio"])
                found = bool(int(row.get("bbox_found", "1")))
            except Exception:
                continue
            out[grid_index] = (ratio, found)
    if not out:
        raise RuntimeError(f"No occlusion rows loaded from {path}")
    return out


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
        print(f"[INFO] BBox annotators attached to {len(self.records)} third-person render products.")

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


def _set_semantic_tag(env_cfg) -> None:
    try:
        env_cfg.can = env_cfg.can.replace(spawn=env_cfg.can.spawn.replace(semantic_tags=[("class", "can")]))
    except Exception:
        try:
            env_cfg.can.spawn.semantic_tags = [("class", "can")]
        except Exception:
            pass


def _set_camera_resolution(env_cfg) -> None:
    camera_cfg = getattr(env_cfg, "third_person_camera", None)
    if camera_cfg is None:
        return
    if args_cli.camera_width is not None:
        camera_cfg.width = int(args_cli.camera_width)
    if args_cli.camera_height is not None:
        camera_cfg.height = int(args_cli.camera_height)


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


def _to_uint8_rgb(frame) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _resize_for_save(image: np.ndarray) -> np.ndarray:
    scale = float(args_cli.save_scale)
    if scale <= 0.0:
        raise ValueError("--save_scale must be positive")
    if abs(scale - 1.0) < 1e-6:
        return image
    h, w = image.shape[:2]
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return np.asarray(Image.fromarray(image).resize(new_size, Image.Resampling.LANCZOS))


def _read_third_person_rgb(base_env, env_id: int) -> np.ndarray:
    camera = getattr(base_env, "third_person_camera", None)
    if camera is None:
        raise RuntimeError("Environment has no third_person_camera")
    rgb = camera.data.output.get("rgb")
    if rgb is None:
        raise RuntimeError("third_person_camera has no rgb output")
    if isinstance(rgb, torch.Tensor):
        if rgb.shape[0] <= env_id:
            raise IndexError(f"env_id={env_id} out of range for third_person rgb shape={tuple(rgb.shape)}")
        rgb = rgb[env_id]
    else:
        rgb = np.asarray(rgb)[env_id]
    return _resize_for_save(_to_uint8_rgb(rgb))


def _log_camera_resolution(env_cfg) -> None:
    camera_cfg = getattr(env_cfg, "third_person_camera", None)
    if camera_cfg is None:
        return
    width = getattr(camera_cfg, "width", None)
    height = getattr(camera_cfg, "height", None)
    print(f"[INFO] third_person_camera resolution: {width}x{height}, save_scale={args_cli.save_scale}")


def _bin_label(occlusion: float) -> str:
    bin_size = float(args_cli.bin_size)
    if bin_size <= 0.0:
        raise ValueError("--bin_size must be positive")
    lo = math.floor(min(max(occlusion, 0.0), 0.999999) / bin_size) * bin_size
    hi = min(1.0, lo + bin_size)
    return f"{lo:.1f}-{hi:.1f}"


def _save_sample(
    image: np.ndarray,
    output_dir: Path,
    label: str,
    grid_index: int,
    x: float,
    y: float,
    occlusion: float,
    env_id: int,
    sample_index: int,
) -> None:
    bin_dir = output_dir / f"occlusion_{label}"
    bin_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sample_{sample_index:02d}_grid_{grid_index:03d}_occ_{occlusion:.3f}_x_{x:.3f}_y_{y:.3f}_env_{env_id}"
    imageio.imwrite(bin_dir / f"{stem}.png", image)
    (bin_dir / f"{stem}.txt").write_text(
        f"grid_index={grid_index}\nx={x}\ny={y}\nocclusion={occlusion}\nenv_id={env_id}\n",
        encoding="utf-8",
    )


def main() -> None:
    grid = _build_grid()
    if not grid:
        raise ValueError("Empty grid")
    occlusion_map = _load_occlusion_csv(args_cli.occlusion_csv)
    output_dir = Path(args_cli.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

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
    _set_camera_resolution(env_cfg)
    _log_camera_resolution(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = _base_env(env)
    reader = None if occlusion_map else BBoxReader(env)
    bin_counts: dict[str, int] = {}

    try:
        env.reset()
        _warmup(base_env, args_cli.warmup_steps)
        if reader is not None:
            reader.initialize()

        completed = 0
        for start in range(0, len(grid), int(base_env.num_envs)):
            batch = grid[start : start + int(base_env.num_envs)]
            env_ids = _assign_positions(env, batch)
            _warmup(base_env, args_cli.warmup_steps)
            if reader is not None:
                occlusion_tensor, found_tensor = reader.read()
            else:
                occlusion_tensor = found_tensor = None

            for local_idx, (grid_index, x, y) in enumerate(batch):
                env_id = int(env_ids[local_idx].item())
                if occlusion_map:
                    occlusion, found = occlusion_map.get(grid_index, (float("nan"), False))
                else:
                    assert occlusion_tensor is not None and found_tensor is not None
                    occlusion = float(occlusion_tensor[env_id].item())
                    found = bool(found_tensor[env_id].item())
                if not found or not math.isfinite(occlusion):
                    continue
                label = _bin_label(occlusion)
                sample_index = bin_counts.get(label, 0)
                if sample_index >= int(args_cli.samples_per_bin):
                    continue
                image = _read_third_person_rgb(base_env, env_id)
                _save_sample(image, output_dir, label, grid_index, x, y, occlusion, env_id, sample_index)
                bin_counts[label] = sample_index + 1

            completed += len(batch)
            print(f"[INFO] Progress: {completed}/{len(grid)} saved={sum(bin_counts.values())}")
            if bin_counts and all(count >= int(args_cli.samples_per_bin) for count in bin_counts.values()):
                possible_bins = int(math.ceil(1.0 / float(args_cli.bin_size)))
                if len(bin_counts) >= possible_bins:
                    break

        summary_path = output_dir / "summary.csv"
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["occlusion_bin", "saved_samples"])
            for label in sorted(bin_counts):
                writer.writerow([label, bin_counts[label]])
        print(f"[INFO] Saved third-person occlusion examples: {output_dir}")
        print(f"[INFO] Summary: {summary_path}")
    finally:
        if reader is not None:
            reader.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
