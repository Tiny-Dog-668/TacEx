"""GelFusion-style task with a frozen ViT-B/16 third-person visual encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from isaaclab.utils import configclass

from .vt_gelfusion_box import OccludedGraspingVTGelFusionBoxCfg, OccludedGraspingVTGelFusionBoxEnv

try:
    import torchvision

    _HAS_TORCHVISION = True
except Exception:
    torchvision = None
    _HAS_TORCHVISION = False


def _replace_visual_observation_with_vit(cfg) -> None:
    cfg.observation_space = dict(cfg.observation_space)
    cfg.observation_space.pop("third_resnet", None)
    horizon = int(getattr(cfg, "gelfusion_obs_horizon", 2))
    cfg.observation_space["third_vit_cls"] = [horizon, int(getattr(cfg, "vit_output_dim", 768))]
    cfg.observation_space["proprio_obs"] = [horizon, 18]
    for key in (
        "tactile_left_depth_resnet",
        "tactile_right_depth_resnet",
        "tactile_left_down_depth_resnet",
        "tactile_right_down_depth_resnet",
    ):
        if key in cfg.observation_space:
            tactile_spec = cfg.observation_space[key]
            if isinstance(tactile_spec, (tuple, list)):
                tactile_dim = int(tactile_spec[-1])
            else:
                tactile_dim = int(tactile_spec)
            cfg.observation_space[key] = [horizon, tactile_dim]
    cfg.observation_space["tactile_dynamic_stats"] = 8


@configclass
class OccludedGraspingVTGelFusionViTBoxCfg(OccludedGraspingVTGelFusionBoxCfg):
    """GelFusion PPO task using a frozen ViT-B/16 CLS visual feature.

    The actor receives `third_vit_cls` instead of `third_resnet`. The rest of the
    environment, reward, action semantics, critic observations, and four tactile
    feature streams are unchanged.
    """

    vit_encoder_backend = "open_clip"  # "open_clip" for CLIP ViT-B/16, or "torchvision" for ImageNet ViT-B/16.
    vit_model_name = "ViT-B-16"
    vit_pretrained = "openai"
    vit_output_dim = 768
    vit_image_hw = (224, 224)
    gelfusion_obs_horizon = 2

    def __post_init__(self):
        super().__post_init__()
        _replace_visual_observation_with_vit(self)

    def _post_configure_scene_object(self):
        super()._post_configure_scene_object()
        _replace_visual_observation_with_vit(self)


@configclass
class OccludedGraspingVTGelFusionViTDownsampleBoxCfg(OccludedGraspingVTGelFusionViTBoxCfg):
    """Visual-downsample GelFusion-ViT task for matching existing downsample baselines."""

    visual_degradation_mode = "downsample"
    visual_downsample_size = 32


class OccludedGraspingVTGelFusionViTBoxEnv(OccludedGraspingVTGelFusionBoxEnv):
    """Append a frozen ViT-B/16 CLS feature and remove the ResNet visual feature."""

    cfg: OccludedGraspingVTGelFusionViTBoxCfg

    def __init__(self, cfg: OccludedGraspingVTGelFusionViTBoxCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._vit_backend = str(getattr(self.cfg, "vit_encoder_backend", "open_clip")).strip().lower()
        self._vit_image_hw = tuple(int(v) for v in getattr(self.cfg, "vit_image_hw", (224, 224)))
        self._gelfusion_obs_horizon = max(1, int(getattr(self.cfg, "gelfusion_obs_horizon", 2)))
        self._gelfusion_vit_history_ready = False
        self._build_frozen_vit_encoder()
        self._init_gelfusion_vit_history_buffers()

    def _build_frozen_vit_encoder(self) -> None:
        if self._vit_backend == "open_clip":
            self._build_open_clip_vit()
        elif self._vit_backend == "torchvision":
            self._build_torchvision_vit()
        else:
            raise ValueError(
                f"Unsupported vit_encoder_backend='{self._vit_backend}'. Use 'open_clip' or 'torchvision'."
            )

        self._vit_encoder.eval()
        for parameter in self._vit_encoder.parameters():
            parameter.requires_grad_(False)
        print(
            f"[INFO] Frozen visual encoder: {self._vit_backend} "
            f"{getattr(self.cfg, 'vit_model_name', 'ViT-B-16')} "
            f"pretrained={getattr(self.cfg, 'vit_pretrained', '')} "
            f"output_dim={getattr(self.cfg, 'vit_output_dim', 768)}"
        )

    def _build_open_clip_vit(self) -> None:
        try:
            import open_clip
        except Exception as exc:
            raise ImportError(
                "open_clip is required for vit_encoder_backend='open_clip'. "
                "Install open_clip or set vit_encoder_backend='torchvision'."
            ) from exc

        model_name = str(getattr(self.cfg, "vit_model_name", "ViT-B-16"))
        pretrained = str(getattr(self.cfg, "vit_pretrained", "openai"))
        try:
            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load open_clip model={model_name!r}, pretrained={pretrained!r}. "
                "Make sure the CLIP ViT-B/16 weights are available locally or use vit_encoder_backend='torchvision'."
            ) from exc
        self._vit_encoder = model.visual
        self._vit_normalize_mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._vit_normalize_std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=self.device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._vit_output_dim = int(getattr(self._vit_encoder, "width", getattr(self.cfg, "vit_output_dim", 768)))

    def _build_torchvision_vit(self) -> None:
        if not _HAS_TORCHVISION:
            raise ImportError("torchvision is required for vit_encoder_backend='torchvision'.")
        try:
            weights = torchvision.models.ViT_B_16_Weights.DEFAULT
            self._vit_encoder = torchvision.models.vit_b_16(weights=weights).to(self.device)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load torchvision ViT-B/16 pretrained weights. "
                "Make sure weights are cached or use vit_encoder_backend='open_clip'."
            ) from exc
        self._vit_normalize_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._vit_normalize_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self._vit_output_dim = int(getattr(self.cfg, "vit_output_dim", 768))

    def _prep_third_rgb_nchw(self) -> torch.Tensor:
        third_rgb = self.third_person_camera.data.output.get("rgb")
        target_h, target_w = self._vit_image_hw
        if third_rgb is None:
            image = torch.zeros((self.num_envs, 3, target_h, target_w), dtype=torch.float32, device=self.device)
        else:
            rgb = third_rgb.to(device=self.device, dtype=torch.float32)
            if rgb.numel() > 0 and torch.isfinite(rgb.max()) and rgb.max() > 1.5:
                rgb = rgb / 255.0
            rgb = self._degrade_third_person_rgb(rgb.clamp(0.0, 1.0))
            image = rgb.permute(0, 3, 1, 2).contiguous()
            if image.shape[2] != target_h or image.shape[3] != target_w:
                image = F.interpolate(image, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return (image - self._vit_normalize_mean) / self._vit_normalize_std

    def _device_autocast_kwargs(self) -> tuple[str, bool]:
        dev = self.device
        dev_type = getattr(dev, "type", None)
        if dev_type is None:
            dev_type = "cuda" if (isinstance(dev, str) and dev.startswith("cuda")) else "cpu"
        return dev_type, dev_type == "cuda"

    def _encode_open_clip_cls(self, image: torch.Tensor) -> torch.Tensor:
        visual = self._vit_encoder
        dtype = getattr(visual, "conv1").weight.dtype
        x = visual.conv1(image.to(dtype=dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_embedding = visual.class_embedding.to(dtype=x.dtype, device=x.device)
        class_token = class_embedding + torch.zeros(
            x.shape[0], 1, class_embedding.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)
        x = x + visual.positional_embedding.to(dtype=x.dtype, device=x.device)
        if hasattr(visual, "patch_dropout"):
            x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        return visual.ln_post(x[:, 0]).float()

    def _encode_torchvision_cls(self, image: torch.Tensor) -> torch.Tensor:
        visual = self._vit_encoder
        x = visual._process_input(image)
        batch_class_token = visual.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = visual.encoder(x)
        return x[:, 0].float()

    def _encode_vit_cls(self) -> torch.Tensor:
        if not hasattr(self, "_vit_encoder"):
            return torch.zeros(
                (self.num_envs, int(getattr(self.cfg, "vit_output_dim", 768))),
                dtype=torch.float32,
                device=self.device,
            )
        image = self._prep_third_rgb_nchw()
        dev_type, use_amp = self._device_autocast_kwargs()
        with torch.no_grad(), torch.amp.autocast(device_type=dev_type, enabled=use_amp, dtype=torch.float16):
            if self._vit_backend == "open_clip":
                cls = self._encode_open_clip_cls(image)
            else:
                cls = self._encode_torchvision_cls(image)
        if cls.shape[-1] != int(getattr(self.cfg, "vit_output_dim", 768)):
            raise RuntimeError(
                f"Frozen ViT feature dim mismatch: got {cls.shape[-1]}, "
                f"expected {getattr(self.cfg, 'vit_output_dim', 768)}."
            )
        return cls

    def _init_gelfusion_vit_history_buffers(self) -> None:
        horizon = max(1, int(getattr(self, "_gelfusion_obs_horizon", getattr(self.cfg, "gelfusion_obs_horizon", 2))))
        vit_dim = int(getattr(self.cfg, "vit_output_dim", 768))
        tactile_dim = int(getattr(self, "_tactile_feature_dim", 256))
        self._gelfusion_vit_history = torch.zeros(
            (self.num_envs, horizon, vit_dim), dtype=torch.float32, device=self.device
        )
        self._gelfusion_tactile_history = {
            key: torch.zeros((self.num_envs, horizon, tactile_dim), dtype=torch.float32, device=self.device)
            for key in (
                "tactile_left_depth_resnet",
                "tactile_right_depth_resnet",
                "tactile_left_down_depth_resnet",
                "tactile_right_down_depth_resnet",
            )
        }
        self._gelfusion_proprio_history = torch.zeros(
            (self.num_envs, horizon, 18), dtype=torch.float32, device=self.device
        )
        self._gelfusion_history_pending_refresh = torch.ones(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._gelfusion_vit_history_ready = True

    def _update_sequence_history(self, history: torch.Tensor, current: torch.Tensor, pending: torch.Tensor) -> torch.Tensor:
        if torch.any(pending):
            pending_ids = pending.nonzero(as_tuple=False).squeeze(-1)
            history[pending_ids] = current[pending_ids].unsqueeze(1).expand(-1, history.shape[1], -1)
        ready_ids = (~pending).nonzero(as_tuple=False).squeeze(-1)
        if ready_ids.numel() > 0:
            history[ready_ids, :-1] = history[ready_ids, 1:].clone()
            history[ready_ids, -1] = current[ready_ids]
        return history

    def _get_observations(self) -> dict[str, dict[str, torch.Tensor]]:
        observations = super()._get_observations()
        obs = observations["policy"]
        obs.pop("third_resnet", None)
        if not getattr(self, "_gelfusion_vit_history_ready", False):
            horizon = max(1, int(getattr(self.cfg, "gelfusion_obs_horizon", 2)))
            vit_dim = int(getattr(self.cfg, "vit_output_dim", 768))
            tactile_dim = int(getattr(self, "_tactile_feature_dim", 256))
            obs["third_vit_cls"] = torch.zeros((self.num_envs, horizon, vit_dim), device=self.device)
            for key in (
                "tactile_left_depth_resnet",
                "tactile_right_depth_resnet",
                "tactile_left_down_depth_resnet",
                "tactile_right_down_depth_resnet",
            ):
                obs[key] = torch.zeros((self.num_envs, horizon, tactile_dim), device=self.device)
            obs["proprio_obs"] = torch.zeros((self.num_envs, horizon, 18), device=self.device)
            return observations

        pending = self._gelfusion_history_pending_refresh
        self._update_sequence_history(self._gelfusion_vit_history, self._encode_vit_cls(), pending)
        for key, history in self._gelfusion_tactile_history.items():
            self._update_sequence_history(history, obs[key].to(dtype=torch.float32), pending)
        self._update_sequence_history(self._gelfusion_proprio_history, obs["proprio_obs"].to(dtype=torch.float32), pending)
        if torch.any(pending):
            pending[:] = False

        obs["third_vit_cls"] = self._gelfusion_vit_history.clone()
        for key, history in self._gelfusion_tactile_history.items():
            obs[key] = history.clone()
        obs["proprio_obs"] = self._gelfusion_proprio_history.clone()
        return observations

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        if getattr(self, "_gelfusion_vit_history_ready", False) and env_ids.numel() > 0:
            self._gelfusion_history_pending_refresh[env_ids] = True
