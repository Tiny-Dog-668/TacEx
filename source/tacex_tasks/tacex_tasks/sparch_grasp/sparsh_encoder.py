from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_lightning_stub() -> None:
    """Provide the minimal Lightning symbol that Sparsh imports for logging."""
    try:
        from lightning.fabric.utilities import rank_zero_only  # noqa: F401
        return
    except Exception:
        pass

    lightning = types.ModuleType("lightning")
    fabric = types.ModuleType("lightning.fabric")
    utilities = types.ModuleType("lightning.fabric.utilities")

    def rank_zero_only(fn):
        return fn

    utilities.rank_zero_only = rank_zero_only
    fabric.utilities = utilities
    lightning.fabric = fabric
    sys.modules.setdefault("lightning", lightning)
    sys.modules.setdefault("lightning.fabric", fabric)
    sys.modules.setdefault("lightning.fabric.utilities", utilities)


class SparshFrozenEncoder(nn.Module):
    """Thin runtime wrapper around Sparsh's tactile ViT encoder."""

    def __init__(
        self,
        repo_path: str,
        checkpoint_path: str,
        img_size_hw: tuple[int, int],
        model_size: str = "small",
        encoder_type: str = "ijepa",
        num_register_tokens: int = 0,
        strict: bool = False,
    ) -> None:
        super().__init__()

        repo_root = Path(repo_path).expanduser().resolve()
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"Sparsh repo not found: {repo_root}")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Sparsh checkpoint not found: {checkpoint}")

        _ensure_lightning_stub()
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.append(repo_root_str)

        from tactile_ssl.model import vit_base, vit_small

        model_size = str(model_size).lower()
        if model_size == "small":
            backbone_ctor = vit_small
        elif model_size == "base":
            backbone_ctor = vit_base
        else:
            raise ValueError(f"Unsupported Sparsh model size: {model_size}")

        self.encoder = backbone_ctor(
            img_size=img_size_hw,
            in_chans=6,
            pos_embed_fn="sinusoidal",
            num_register_tokens=int(num_register_tokens),
        )
        self.output_dim = int(self.encoder.embed_dim)
        self._load_encoder_weights(str(checkpoint), encoder_type=encoder_type, strict=strict)
        self.encoder.eval()
        self.encoder.requires_grad_(False)

    def _load_encoder_weights(self, checkpoint_path: str, encoder_type: str, strict: bool) -> None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        state_dict = checkpoint.get("model", checkpoint)
        encoder_type = str(encoder_type).lower()
        if "jepa" in encoder_type:
            encoder_key = "target_encoder"
        elif "dino" in encoder_type:
            encoder_key = "teacher_encoder.backbone"
        else:
            encoder_key = "encoder"

        target_keys = [key for key in state_dict.keys() if key.startswith(f"{encoder_key}.")]
        if not target_keys and "backbone" not in encoder_key:
            target_keys = [key for key in state_dict.keys() if key.startswith(f"{encoder_key}.backbone.")]
            if target_keys:
                encoder_key = f"{encoder_key}.backbone"
        if not target_keys:
            raise KeyError(f"Encoder prefix '{encoder_key}' not found in checkpoint: {checkpoint_path}")

        new_state_dict = {
            key.replace(f"{encoder_key}.", ""): state_dict[key]
            for key in target_keys
        }
        self.encoder.load_state_dict(new_state_dict, strict=strict)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def project_sparsh_features(x: torch.Tensor, output_dim: int) -> torch.Tensor:
    """Project Sparsh features to a fixed size with parameter-free adaptive pooling."""

    output_dim = int(output_dim)
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}")
    if x.ndim < 2:
        raise ValueError(f"Expected Sparsh features with rank >= 2, got shape {tuple(x.shape)}")
    if int(x.shape[-1]) == output_dim:
        return x.float()

    original_shape = tuple(x.shape)
    pooled = F.adaptive_avg_pool1d(x.reshape(-1, 1, original_shape[-1]).float(), output_dim)
    return pooled.reshape(*original_shape[:-1], output_dim)
