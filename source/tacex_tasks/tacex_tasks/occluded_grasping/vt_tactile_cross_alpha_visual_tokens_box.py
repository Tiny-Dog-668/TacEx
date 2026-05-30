"""Tactile-cross-alpha visual downsample task exposing 49 visual tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab.utils import configclass

from .vt_box import OccludedGraspingVTAlphaBoxEnv, OccludedGraspingVTAlphaDownsampleBoxCfg

try:
    import torchvision

    _HAS_TORCHVISION = True
except Exception:
    torchvision = None
    _HAS_TORCHVISION = False


@configclass
class OccludedGraspingVTTactileCrossAlphaVisualTokensDownsampleBoxCfg(OccludedGraspingVTAlphaDownsampleBoxCfg):
    """Downsample VT cfg with 49 third-person visual tokens."""

    vision_token_count = 49
    vision_token_dim = 512

    def __post_init__(self):
        super().__post_init__()
        self.observation_space = dict(self.observation_space)
        self.observation_space.pop("third_resnet", None)
        self.observation_space["vision_tokens"] = [int(self.vision_token_count), int(self.vision_token_dim)]


class OccludedGraspingVTTactileCrossAlphaVisualTokensBoxEnv(OccludedGraspingVTAlphaBoxEnv):
    """Append frozen ResNet layer4 visual tokens and remove global third_resnet from policy obs."""

    cfg: OccludedGraspingVTTactileCrossAlphaVisualTokensDownsampleBoxCfg

    def __init__(
        self,
        cfg: OccludedGraspingVTTactileCrossAlphaVisualTokensDownsampleBoxCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is required for 49-token visual observation extraction")
        self._vision_token_encoder = self._build_resnet18_layer4_encoder()
        self._vision_token_hw = (7, 7)

    def _build_resnet18_layer4_encoder(self) -> nn.Module:
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            backbone = torchvision.models.resnet18(weights=weights)
        except Exception:
            backbone = torchvision.models.resnet18(pretrained=True)
        encoder = nn.Sequential(*list(backbone.children())[:-2]).to(self.device)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        return encoder

    def _device_autocast_kwargs(self) -> tuple[str, bool]:
        dev = self.device
        dev_type = getattr(dev, "type", None)
        if dev_type is None:
            dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
        return dev_type, dev_type == "cuda"

    def _encode_vision_tokens(self) -> torch.Tensor:
        third_rgb = self.third_person_camera.data.output.get("rgb")
        if third_rgb is None:
            third_rgb = torch.zeros(
                (self.num_envs, self.cfg.third_person_camera.height, self.cfg.third_person_camera.width, 3),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            third_rgb = third_rgb.to(device=self.device, dtype=torch.float32) / 255.0
        third_rgb = self._degrade_third_person_rgb(third_rgb)
        xt = third_rgb.permute(0, 3, 1, 2).contiguous()
        if hasattr(self, "_imagenet_mean"):
            xt = (xt - self._imagenet_mean) / self._imagenet_std
        dev_type, use_amp = self._device_autocast_kwargs()
        with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
            fmap = self._vision_token_encoder(xt).float()
        fmap = F.adaptive_avg_pool2d(fmap, self._vision_token_hw)
        return fmap.flatten(start_dim=2).transpose(1, 2).contiguous()

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]
        obs.pop("third_resnet", None)
        obs["vision_tokens"] = self._encode_vision_tokens()
        return {"policy": obs}
