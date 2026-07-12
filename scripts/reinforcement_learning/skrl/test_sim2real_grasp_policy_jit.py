"""Smoke-test an exported TacEx sim2real TorchScript policy.

This script does not require Isaac Sim. It loads the exported TorchScript model,
runs a few inference passes, and reports:

- input/output shapes
- whether the output is finite
- whether repeated calls with the same input are deterministic
- whether a small input perturbation changes the output
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from vision_encoder_artifact import module_state_dict_sha256, sha256_file

try:
    from PIL import Image
except ImportError:
    Image = None


DEFAULT_MODEL_PATH = Path(
    "/home/tinydog/Projects/TacEx/logs/skrl/sim2real_grasp/"
    "2026-03-28_19-58-39_ppo_torch_vision_only_resnet18/checkpoints/exported/policy_actor_e2e.pt"
)


def _load_metadata(model_path: Path, metadata_path: str | None) -> dict:
    if metadata_path is not None:
        candidate = Path(metadata_path)
    else:
        candidate = model_path.with_suffix(".json")
    if not candidate.is_file():
        return {}
    return json.loads(candidate.read_text(encoding="utf-8"))


def _parse_vector_arg(raw: str | None, dim: int, name: str) -> torch.Tensor:
    if raw is None:
        return torch.zeros(dim, dtype=torch.float32)
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != dim:
        raise ValueError(f"{name} expects exactly {dim} comma-separated values, got {len(values)}")
    return torch.tensor(values, dtype=torch.float32)


def _make_gradient_rgb(height: int, width: int) -> torch.Tensor:
    y = torch.linspace(0, 255, steps=height, dtype=torch.float32).view(height, 1)
    x = torch.linspace(0, 255, steps=width, dtype=torch.float32).view(1, width)
    red = x.expand(height, width)
    green = y.expand(height, width)
    blue = 0.5 * (red + green)
    rgb = torch.stack([red, green, blue], dim=-1)
    return rgb.clamp(0, 255).to(torch.uint8)


def _load_rgb_image(image_path: Path, height: int, width: int) -> torch.Tensor:
    if Image is None:
        raise RuntimeError("Pillow is required for --image. Install it with `pip install pillow`.")
    image = Image.open(image_path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height))
    return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())


def _build_rgb_input(args, height: int, width: int) -> torch.Tensor:
    if args.image is not None:
        rgb = _load_rgb_image(Path(args.image), height, width)
    elif args.rgb_mode == "zeros":
        rgb = torch.zeros((height, width, 3), dtype=torch.uint8)
    elif args.rgb_mode == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        rgb = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8, generator=generator)
    elif args.rgb_mode == "gradient":
        rgb = _make_gradient_rgb(height, width)
    else:
        raise ValueError(f"Unsupported rgb mode: {args.rgb_mode}")
    return rgb.unsqueeze(0).repeat(args.batch_size, 1, 1, 1)


def _build_feature_input(args, dim: int) -> torch.Tensor:
    if args.feature_mode == "zeros":
        feat = torch.zeros((args.batch_size, dim), dtype=torch.float32)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        feat = torch.randn((args.batch_size, dim), dtype=torch.float32, generator=generator)
    return feat


def _perturb_rgb(rgb: torch.Tensor) -> torch.Tensor:
    perturbed = rgb.clone()
    red = perturbed[..., 0].to(torch.int16)
    perturbed[..., 0] = torch.clamp(red + 20, 0, 255).to(torch.uint8)
    return perturbed


def _perturb_features(features: torch.Tensor) -> torch.Tensor:
    perturbed = features.clone()
    perturbed[:, 0] += 0.1
    return perturbed


def _sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test an exported TacEx sim2real TorchScript policy.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to the TorchScript model.")
    parser.add_argument("--metadata", type=str, default=None, help="Optional path to the sidecar JSON metadata.")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device: cpu or cuda[:id].")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the smoke test.")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs before timing.")
    parser.add_argument("--runs", type=int, default=10, help="Number of timed runs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used by synthetic inputs.")
    parser.add_argument("--image", type=str, default=None, help="Optional RGB image to feed the end-to-end model.")
    parser.add_argument(
        "--rgb-mode",
        type=str,
        default="gradient",
        choices=["gradient", "random", "zeros"],
        help="Synthetic RGB input mode when --image is not provided.",
    )
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="random",
        choices=["random", "zeros"],
        help="Synthetic wrist_resnet feature mode for compact models.",
    )
    parser.add_argument(
        "--action-history",
        type=str,
        default=None,
        help="Comma-separated action history values. Defaults to zeros with the required dimension.",
    )
    parser.add_argument(
        "--proprio-obs",
        type=str,
        default=None,
        help="Comma-separated proprio observation values. Defaults to zeros with the required dimension.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    metadata = _load_metadata(model_path, args.metadata)
    input_signature = metadata.get("input_signature", {})
    if "wrist_rgb" in input_signature:
        input_mode = "rgb"
        rgb_height, rgb_width, _ = input_signature["wrist_rgb"]
        wrist_dim = None
    elif "wrist_resnet" in input_signature:
        input_mode = "features"
        rgb_height = rgb_width = None
        wrist_dim = input_signature["wrist_resnet"][0]
    else:
        raise RuntimeError("Unable to infer model input signature. Metadata must define wrist_rgb or wrist_resnet.")

    action_hist_dim = input_signature["action_history"][0]
    proprio_dim = input_signature["proprio_obs"][0]
    output_signature = metadata.get("output_signature", {})
    if "mean_actions" not in output_signature:
        raise RuntimeError("Metadata must declare output_signature.mean_actions.")
    action_dim = int(output_signature["mean_actions"][0])
    if not isinstance(metadata.get("policy_contract"), dict):
        raise RuntimeError("Metadata does not contain a verified run policy contract.")

    device = torch.device(args.device)
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    model.to(device)

    expected_model_hash = metadata.get("torchscript_sha256")
    if not expected_model_hash:
        raise RuntimeError("Metadata does not declare torchscript_sha256.")
    actual_model_hash = sha256_file(model_path)
    if actual_model_hash != expected_model_hash:
        raise RuntimeError(
            f"TorchScript SHA-256 mismatch: expected {expected_model_hash}, got {actual_model_hash}"
        )

    encoder_metadata = metadata.get("vision_encoder")
    if input_mode == "rgb":
        if not isinstance(encoder_metadata, dict) or not encoder_metadata.get("verified"):
            raise RuntimeError("RGB export metadata does not contain a verified training vision encoder.")
        if not hasattr(model, "vision_encoder"):
            raise RuntimeError("RGB TorchScript model does not contain vision_encoder.")
        embedded_encoder_hash = module_state_dict_sha256(model.vision_encoder)
        expected_encoder_hash = encoder_metadata.get("state_dict_sha256")
        if embedded_encoder_hash != expected_encoder_hash:
            raise RuntimeError(
                "Embedded vision encoder SHA-256 mismatch: "
                f"expected {expected_encoder_hash}, got {embedded_encoder_hash}"
            )
    else:
        embedded_encoder_hash = None

    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"model         : {model_path}")
    print(f"metadata      : {model_path.with_suffix('.json') if args.metadata is None else args.metadata}")
    print(f"file_size_mb  : {model_path.stat().st_size / (1024 ** 2):.2f}")
    print(f"num_parameters: {num_params}")
    print(f"input_mode    : {input_mode}")
    print(f"device        : {device}")
    print(f"model_sha256  : {actual_model_hash}")
    if embedded_encoder_hash is not None:
        print(f"encoder_sha256: {embedded_encoder_hash}")

    action_history = _parse_vector_arg(args.action_history, action_hist_dim, "action_history")
    action_history = action_history.unsqueeze(0).repeat(args.batch_size, 1).to(device)
    proprio_obs = _parse_vector_arg(args.proprio_obs, proprio_dim, "proprio_obs")
    proprio_obs = proprio_obs.unsqueeze(0).repeat(args.batch_size, 1).to(device)

    if input_mode == "rgb":
        wrist_input = _build_rgb_input(args, rgb_height, rgb_width).to(device)
        perturbed_input = _perturb_rgb(wrist_input)
        print(f"wrist_rgb     : {tuple(wrist_input.shape)} {wrist_input.dtype}")
    else:
        wrist_input = _build_feature_input(args, wrist_dim).to(device)
        perturbed_input = _perturb_features(wrist_input)
        print(f"wrist_resnet  : {tuple(wrist_input.shape)} {wrist_input.dtype}")

    with torch.inference_mode():
        out_a = model(action_history, proprio_obs, wrist_input)
        out_b = model(action_history, proprio_obs, wrist_input)
        out_c = model(action_history, proprio_obs, perturbed_input)

    expected_output_shape = (args.batch_size, action_dim)
    shape_ok = tuple(out_a.shape) == expected_output_shape and tuple(out_c.shape) == expected_output_shape
    finite_ok = bool(torch.isfinite(out_a).all().item() and torch.isfinite(out_c).all().item())
    same_input_diff = float(torch.max(torch.abs(out_a - out_b)).item())
    perturb_diff = float(torch.max(torch.abs(out_a - out_c)).item())

    actor_bounds = metadata.get("actor_mean_bounds")
    if actor_bounds is not None:
        lower, upper = (float(value) for value in actor_bounds)
        bounds_ok = bool(
            torch.all(out_a >= lower - 1e-6).item()
            and torch.all(out_a <= upper + 1e-6).item()
            and torch.all(out_c >= lower - 1e-6).item()
            and torch.all(out_c <= upper + 1e-6).item()
        )
    else:
        bounds_ok = None

    print(f"output_shape  : {tuple(out_a.shape)}")
    print(f"expected_shape: {expected_output_shape}")
    print(f"finite_output : {finite_ok}")
    print(f"repeat_diff   : {same_input_diff:.8f}")
    print(f"perturb_diff  : {perturb_diff:.8f}")
    print(f"bounded_output: {bounds_ok if bounds_ok is not None else 'not declared'}")
    print("actions       :")
    print(out_a.detach().cpu())

    for _ in range(args.warmup):
        with torch.inference_mode():
            _ = model(action_history, proprio_obs, wrist_input)
    _sync_if_needed(device)

    start = time.perf_counter()
    for _ in range(args.runs):
        with torch.inference_mode():
            _ = model(action_history, proprio_obs, wrist_input)
    _sync_if_needed(device)
    elapsed = time.perf_counter() - start
    avg_ms = 1000.0 * elapsed / max(args.runs, 1)
    print(f"avg_latency_ms: {avg_ms:.3f}")

    if not finite_ok:
        raise RuntimeError("Model output contains NaN or Inf.")
    if not shape_ok:
        raise RuntimeError(
            f"Model output shape mismatch: expected {expected_output_shape}, got {tuple(out_a.shape)}"
        )
    if same_input_diff > 1e-7:
        raise RuntimeError(f"Deterministic model changed for identical input: max_abs_diff={same_input_diff}")
    if bounds_ok is False:
        raise RuntimeError(f"Model output violates declared actor bounds: {actor_bounds}")


if __name__ == "__main__":
    main()
