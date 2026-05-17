"""Vision + tactile ViT policy model for skrl (Gaussian)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import GaussianMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, embed_dim: int, pad_h: int = 0, pad_w: int = 0):
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.pad_h = int(pad_h)
        self.pad_w = int(pad_w)
        self.patch_dim = self.in_channels * self.patch_size * self.patch_size
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.pre_ln = nn.LayerNorm(self.patch_dim)
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.post_ln = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, H, W]
        if self.pad_h or self.pad_w:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode="constant", value=0.0)
        patches = self.unfold(x)  # [N, patch_dim, L]
        patches = patches.transpose(1, 2)  # [N, L, patch_dim]
        patches = self.pre_ln(patches)
        patches = self.proj(patches)
        patches = self.post_ln(patches)
        return patches


class VisionTactileViTGaussianPolicy(GaussianMixin, Model):
    """ViT-based policy that fuses wrist RGB + two tactile RGB streams."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
        initial_log_std: float = 0.0,
        fixed_log_std: bool = False,
        patch_size: int = 8,
        embed_dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.0,
        mlp_layers: Iterable[int] | None = None,
        mlp_activation: str = "elu",
        use_cls_token: bool = True,
        **kwargs: Any,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.use_cls_token = bool(use_cls_token)

        # modality shapes (may be None for dynamically-defined spaces)
        self._wrist_shape = self._get_obs_shape(observation_space, "wrist_rgb")
        self._tactile_left_shape = self._get_obs_shape(observation_space, "tactile_left_rgb")
        self._tactile_right_shape = self._get_obs_shape(observation_space, "tactile_right_rgb")

        # patch embeddings and positional encodings (lazily created if shapes are unknown)
        self.wrist_patch_embed, self.wrist_pos_embed = self._make_patch_embed(self._wrist_shape)
        self.tactile_left_patch_embed, self.tactile_left_pos_embed = self._make_patch_embed(self._tactile_left_shape)
        self.tactile_right_patch_embed, self.tactile_right_pos_embed = self._make_patch_embed(self._tactile_right_shape)

        self.modality_embed = nn.Embedding(3, self.embed_dim)
        self.token_dropout = nn.Dropout(p=float(dropout)) if dropout and dropout > 0.0 else nn.Identity()

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.cls_pos = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        else:
            self.register_parameter("cls_token", None)
            self.register_parameter("cls_pos", None)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(heads),
            dim_feedforward=int(mlp_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(depth))

        # vector observations
        self.proprio_key = "proprio_obs" if self._get_obs_shape(observation_space, "proprio_obs") else "proprio"
        self.gripper_key = "gripper_state" if self._get_obs_shape(observation_space, "gripper_state") else None
        self.action_hist_key = "action_history" if self._get_obs_shape(observation_space, "action_history") else None

        self.proprio_dim = self._get_vec_dim(observation_space, self.proprio_key)
        self.gripper_dim = self._get_vec_dim(observation_space, self.gripper_key)
        self.action_hist_dim = self._get_vec_dim(observation_space, self.action_hist_key)

        vector_dim = self.proprio_dim + self.gripper_dim + self.action_hist_dim
        in_dim = self.embed_dim + vector_dim

        if mlp_layers is None:
            mlp_layers = (256, 128)
        activation = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        layers = []
        last_dim = in_dim
        for h in mlp_layers:
            layers.append(nn.Linear(last_dim, int(h)))
            layers.append(activation())
            last_dim = int(h)
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.mu = nn.Linear(last_dim, self.num_actions)

        self.log_std_parameter = nn.Parameter(
            torch.full(size=(self.num_actions,), fill_value=float(initial_log_std)),
            requires_grad=not bool(fixed_log_std),
        )

    def _get_obs_shape(self, obs_space, key: str | None) -> Tuple[int, ...] | None:
        if key is None:
            return None
        if hasattr(obs_space, "spaces") and key in obs_space.spaces:
            shape = obs_space.spaces[key].shape
            if shape is None:
                return None
            return tuple(shape)
        if isinstance(obs_space, dict) and key in obs_space:
            space = obs_space[key]
            if hasattr(space, "shape"):
                return tuple(space.shape)
            if isinstance(space, (tuple, list)):
                return tuple(space)
        return None

    def _get_vec_dim(self, obs_space, key: str | None) -> int:
        shape = self._get_obs_shape(obs_space, key)
        if shape is None:
            return 0
        if len(shape) == 0:
            return 0
        if len(shape) == 1:
            return int(shape[0])
        return int(shape[-1])

    def _make_patch_embed(self, shape: Tuple[int, ...] | None):
        if shape is None:
            return None, None
        if len(shape) != 3:
            raise ValueError(f"Expected image shape (C,H,W), got {shape}")
        c, h, w = shape
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        h_pad = h + pad_h
        w_pad = w + pad_w
        num_patches = (h_pad // self.patch_size) * (w_pad // self.patch_size)
        patch_embed = PatchEmbed(int(c), self.patch_size, self.embed_dim, pad_h=pad_h, pad_w=pad_w)
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        return patch_embed, pos_embed

    def _ensure_patch_embed(self, modality: str, x: torch.Tensor) -> Tuple[PatchEmbed, nn.Parameter]:
        if x is None:
            raise ValueError(f"Missing input tensor for modality '{modality}' to build patch embedding")
        x = self._to_nchw(x)
        c, h, w = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
        shape = (c, h, w)
        patch_embed, pos_embed = self._make_patch_embed(shape)
        patch_embed = patch_embed.to(self.device)
        pos_embed = nn.Parameter(pos_embed.to(self.device))
        if modality == "wrist":
            self._wrist_shape = shape
            self.wrist_patch_embed = patch_embed
            self.wrist_pos_embed = pos_embed
        elif modality == "tactile_left":
            self._tactile_left_shape = shape
            self.tactile_left_patch_embed = patch_embed
            self.tactile_left_pos_embed = pos_embed
        elif modality == "tactile_right":
            self._tactile_right_shape = shape
            self.tactile_right_patch_embed = patch_embed
            self.tactile_right_pos_embed = pos_embed
        return patch_embed, pos_embed

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            if isinstance(x, (list, tuple)) and x and all(isinstance(t, torch.Tensor) for t in x):
                x = torch.stack(x, dim=0)
            else:
                x = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.shape}")
        if x.shape[1] in (1, 3):
            return x
        if x.shape[-1] in (1, 3):
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def _build_tokens(
        self,
        x: torch.Tensor,
        patch_embed: PatchEmbed,
        pos_embed: nn.Parameter,
        modality_idx: int,
    ) -> torch.Tensor:
        x = x.to(device=self.device, dtype=torch.float32)
        x = self._to_nchw(x)
        x = x.clamp(0.0, 1.0)
        tokens = patch_embed(x)
        if pos_embed is not None and tokens.shape[1] != pos_embed.shape[1]:
            raise ValueError(
                f"Positional embedding mismatch: tokens={tokens.shape[1]} pos={pos_embed.shape[1]}"
            )
        type_embed = self.modality_embed(torch.tensor(modality_idx, device=tokens.device)).view(1, 1, -1)
        if pos_embed is None:
            tokens = tokens + type_embed
        else:
            tokens = tokens + pos_embed + type_embed
        tokens = self.token_dropout(tokens)
        return tokens

    def compute(self, inputs: Dict[str, Any], role: str = ""):
        states = inputs.get("states")
        if isinstance(states, dict):
            obs = states
        else:
            obs = unflatten_tensorized_space(self.observation_space, states)

        batch_size = None
        for v in obs.values():
            if isinstance(v, torch.Tensor):
                batch_size = v.shape[0]
                break
        if batch_size is None:
            raise ValueError("No tensor observations found in inputs['states']")

        tokens_list = []
        if self.wrist_patch_embed is not None or "wrist_rgb" in obs:
            wrist = obs.get("wrist_rgb")
            if wrist is None and self._wrist_shape is not None:
                c, h, w = self._wrist_shape
                wrist = torch.zeros((batch_size, c, h, w), device=self.device)
            if self.wrist_patch_embed is None and wrist is not None:
                self._ensure_patch_embed("wrist", wrist)
            tokens_list.append(
                self._build_tokens(wrist, self.wrist_patch_embed, self.wrist_pos_embed, modality_idx=0)
            )
        if self.tactile_left_patch_embed is not None or "tactile_left_rgb" in obs:
            tact_l = obs.get("tactile_left_rgb")
            if tact_l is None and self._tactile_left_shape is not None:
                c, h, w = self._tactile_left_shape
                tact_l = torch.zeros((batch_size, c, h, w), device=self.device)
            if self.tactile_left_patch_embed is None and tact_l is not None:
                self._ensure_patch_embed("tactile_left", tact_l)
            tokens_list.append(
                self._build_tokens(tact_l, self.tactile_left_patch_embed, self.tactile_left_pos_embed, modality_idx=1)
            )
        if self.tactile_right_patch_embed is not None or "tactile_right_rgb" in obs:
            tact_r = obs.get("tactile_right_rgb")
            if tact_r is None and self._tactile_right_shape is not None:
                c, h, w = self._tactile_right_shape
                tact_r = torch.zeros((batch_size, c, h, w), device=self.device)
            if self.tactile_right_patch_embed is None and tact_r is not None:
                self._ensure_patch_embed("tactile_right", tact_r)
            tokens_list.append(
                self._build_tokens(tact_r, self.tactile_right_patch_embed, self.tactile_right_pos_embed, modality_idx=2)
            )

        if not tokens_list:
            raise ValueError("No visual/tactile modalities available to build tokens")

        x = torch.cat(tokens_list, dim=1)
        if self.use_cls_token:
            cls = (self.cls_token + self.cls_pos).expand(batch_size, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)
        fused = x[:, 0] if self.use_cls_token else x.mean(dim=1)

        vecs = []
        if self.proprio_key:
            vecs.append(obs.get(self.proprio_key))
        if self.gripper_key:
            vecs.append(obs.get(self.gripper_key))
        if self.action_hist_key:
            vecs.append(obs.get(self.action_hist_key))
        vecs = [v for v in vecs if v is not None]
        if vecs:
            vec = torch.cat(vecs, dim=-1)
            fused = torch.cat([fused, vec], dim=-1)

        feats = self.mlp(fused)
        output = self.mu(feats)
        return output, self.log_std_parameter, {}


def quick_vt_vit_policy_smoke_test(
    batch_size: int = 4,
    wrist_hw: Tuple[int, int] = (84, 84),
    tactile_hw: Tuple[int, int] = (64, 64),
    action_dim: int = 5,
    device: str = "cpu",
) -> None:
    """Lightweight check for forward/grad without requiring Isaac Sim."""
    import gymnasium as gym

    obs_space = gym.spaces.Dict(
        {
            "wrist_rgb": gym.spaces.Box(0.0, 1.0, shape=(3, wrist_hw[0], wrist_hw[1]), dtype=float),
            "tactile_left_rgb": gym.spaces.Box(0.0, 1.0, shape=(3, tactile_hw[0], tactile_hw[1]), dtype=float),
            "tactile_right_rgb": gym.spaces.Box(0.0, 1.0, shape=(3, tactile_hw[0], tactile_hw[1]), dtype=float),
            "proprio_obs": gym.spaces.Box(-1.0, 1.0, shape=(18,), dtype=float),
            "gripper_state": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=float),
            "action_history": gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=float),
        }
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=float)

    model = VisionTactileViTGaussianPolicy(
        observation_space=obs_space,
        action_space=action_space,
        device=device,
    )

    states = {
        "wrist_rgb": torch.rand(batch_size, 3, wrist_hw[0], wrist_hw[1]),
        "tactile_left_rgb": torch.rand(batch_size, 3, tactile_hw[0], tactile_hw[1]),
        "tactile_right_rgb": torch.rand(batch_size, 3, tactile_hw[0], tactile_hw[1]),
        "proprio_obs": torch.rand(batch_size, 18),
        "gripper_state": torch.rand(batch_size, 1),
        "action_history": torch.rand(batch_size, 5),
    }

    mean, log_std, _ = model.compute({"states": states}, role="policy")
    loss = mean.mean() + log_std.mean()
    loss.backward()
    grad_norm = model.wrist_patch_embed.proj.weight.grad.norm().item()
    print(f"[smoke_test] mean shape={tuple(mean.shape)} grad_norm={grad_norm:.6f}")
