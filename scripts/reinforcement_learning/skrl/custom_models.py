import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.utils.spaces.torch import unflatten_tensorized_space


class CylinderFusionLSTM(GaussianMixin, Model):
    """Simplified LSTM policy for legacy LSTM path (uses pre-encoded features)."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        num_envs: int,
        sequence_length: int = 64,
        num_layers: int = 1,
        hidden_size: int = 256,
        use_vision_placeholder: bool = True,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
        reduction: str = "sum",
        stats_print_every: int = 0,
        reset_stats_print: bool = False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_vision_placeholder = use_vision_placeholder
        self.enable_input_ln = True
        self.enable_output_ln = True
        self.c_clip = 0.0
        self.reset_on_missing_done = False  # 强制在缺失 done 时清零 rnn state（默认关闭）
        self.reset_on_mismatch = False       # 强制在 done 维度不匹配时清零（默认关闭）
        self.reset_interval = 0             # >0 时，每隔 reset_interval 次 forward 清零一次 rnn state
        self._dbg_calls = 0
        self.stats_print_every = int(stats_print_every) if stats_print_every is not None else 0
        if self.stats_print_every < 0:
            self.stats_print_every = 0
        self._stats_calls = 0
        self.reset_stats_print = bool(reset_stats_print)

        # derive feature dims and keys from observation_space to match env outputs
        self.vis_key = "vision_feat" if "vision_feat" in observation_space else (
            "wrist_resnet" if "wrist_resnet" in observation_space else None
        )
        if self.vis_key is not None:
            self.vis_dim = int(observation_space[self.vis_key].shape[0])
        else:
            # Keep backward-compatible placeholder by default; allow dropping it for tactile-only tasks.
            self.vis_dim = 512 if self.use_vision_placeholder else 0

        # tactile: prefer pre-stacked tactile_feat, otherwise stack individual tactile_* entries
        self.tact_keys = []
        if "tactile_feat" in observation_space:
            self.tact_keys = ["tactile_feat"]
            tact_shape = observation_space["tactile_feat"].shape
            self.num_tactile = int(tact_shape[0]) if len(tact_shape) > 1 else 1
            self.tact_dim = int(tact_shape[1]) if len(tact_shape) > 1 else int(tact_shape[0])
        else:
            self.tact_keys = [
                k
                for k in observation_space.keys()
                if k.startswith("tactile_") and (k.endswith("_depth_resnet") or k.endswith("_resnet"))
            ]
            self.tact_keys = sorted(self.tact_keys)
            self.num_tactile = len(self.tact_keys) if self.tact_keys else 6
            self.tact_dim = (
                int(observation_space[self.tact_keys[0]].shape[0])
                if self.tact_keys
                else 64
            )

        # proprio: support either proprio or proprio_obs
        if "proprio" in observation_space:
            self.proprio_key = "proprio"
            self.proprio_dim = int(observation_space["proprio"].shape[0])
        elif "proprio_obs" in observation_space:
            self.proprio_key = "proprio_obs"
            self.proprio_dim = int(observation_space["proprio_obs"].shape[0])
        else:
            self.proprio_key = None
            self.proprio_dim = 0

        input_dim = self.vis_dim + self.num_tactile * self.tact_dim + self.proprio_dim

        # optional LayerNorms (disabled by default)
        self.in_ln = nn.LayerNorm(128)
        self.out_ln = nn.LayerNorm(self.hidden_size)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.net = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions),
        )

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),  # hidden
                    (self.num_layers, self.num_envs, self.hidden_size),  # cell
                ],
            }
        }

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict with backward compatibility for optional vision placeholder dims.

        If a checkpoint was trained with a 512-D zero vision placeholder and the current model
        drops that placeholder (or vice versa), adapt the first fusion layer weights by
        trimming/padding the leading feature columns.
        """
        sd = dict(state_dict)
        key = "fusion_mlp.0.weight"
        if key in sd:
            w = sd[key]
            target_w = self.fusion_mlp[0].weight
            if (
                isinstance(w, torch.Tensor)
                and w.dim() == 2
                and target_w.dim() == 2
                and w.shape[0] == target_w.shape[0]
                and w.shape[1] != target_w.shape[1]
            ):
                if w.shape[1] > target_w.shape[1]:
                    # Drop leading placeholder columns (vision is concatenated first).
                    sd[key] = w[:, -target_w.shape[1] :]
                else:
                    # Pad leading zeros if the model expects additional placeholder columns.
                    pad = target_w.shape[1] - w.shape[1]
                    sd[key] = F.pad(w, (pad, 0))

        return super().load_state_dict(sd, strict=strict)

    def _fuse_states(self, inputs):
        states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        rnn_reset = states.get("rnn_reset", None)

        # vision feature: prefer configured vis_key, fall back to zeros
        if self.vis_key is not None and self.vis_key in states:
            vis = states[self.vis_key]
        else:
            vis = torch.zeros((inputs["states"].shape[0], self.vis_dim), device=self.device)

        # tactile feature: use pre-stacked tensor if available, else stack individual tactile_* keys
        if "tactile_feat" in states:
            tact = states["tactile_feat"]
            if tact.dim() == 2 and self.num_tactile > 1:
                tact = tact.view(tact.shape[0], self.num_tactile, -1)
        else:
            tact_list = [states[k] for k in self.tact_keys if k in states]
            if tact_list:
                tact = torch.stack(tact_list, dim=1)
            else:
                tact = torch.zeros(
                    (vis.shape[0], self.num_tactile, self.tact_dim),
                    device=self.device,
                )

        # proprioception
        if self.proprio_key is not None and self.proprio_key in states:
            proprio = states[self.proprio_key]
        else:
            proprio = torch.zeros((vis.shape[0], self.proprio_dim), device=self.device)

        tact_flat = tact.reshape(tact.shape[0], -1)
        x = torch.cat([vis, tact_flat, proprio], dim=-1)
        return self.fusion_mlp(x), rnn_reset

    def _reset_done(self, h: torch.Tensor, c: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None or not torch.any(mask):
            return h, c
        keep = (~mask).to(dtype=h.dtype, device=h.device).view(1, -1, 1)
        return h * keep, c * keep

    def _align_states(self, h: torch.Tensor, c: torch.Tensor, target_batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 保持 shape [num_layers, batch, hidden]，若 batch 不匹配则扩展或截断
        cur_batch = h.shape[1]
        if cur_batch == target_batch:
            return h.contiguous(), c.contiguous()
        if cur_batch == 1 and target_batch > 1:
            return (
                h.expand(self.num_layers, target_batch, self.hidden_size).contiguous(),
                c.expand(self.num_layers, target_batch, self.hidden_size).contiguous(),
            )
        if cur_batch > target_batch:
            return h[:, :target_batch, :].contiguous(), c[:, :target_batch, :].contiguous()
        # pad zeros if fewer states than batch
        pad = target_batch - cur_batch
        h_pad = torch.zeros((self.num_layers, pad, self.hidden_size), device=h.device, dtype=h.dtype)
        c_pad = torch.zeros((self.num_layers, pad, self.hidden_size), device=c.device, dtype=c.dtype)
        h_new = torch.cat([h, h_pad], dim=1)
        c_new = torch.cat([c, c_pad], dim=1)
        return h_new.contiguous(), c_new.contiguous()

    def _masked_mean_var(self, x: torch.Tensor, mask: torch.Tensor):
        if mask is None:
            return None
        mask = mask.to(device=x.device).view(-1).bool()
        if mask.numel() != x.shape[1]:
            return None
        if not torch.any(mask).item():
            return None
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return None
        sel = x[:, idx, :]
        mean = sel.mean().item()
        var = sel.var(unbiased=False).item()
        return mean, var

    def _print_reset_stats(
        self,
        mask: torch.Tensor,
        h_before: torch.Tensor,
        c_before: torch.Tensor,
        h_after: torch.Tensor,
        c_after: torch.Tensor,
        is_update: bool,
    ) -> None:
        if not self.reset_stats_print or is_update:
            return
        h_pre = self._masked_mean_var(h_before.detach(), mask)
        h_post = self._masked_mean_var(h_after.detach(), mask)
        c_pre = self._masked_mean_var(c_before.detach(), mask)
        c_post = self._masked_mean_var(c_after.detach(), mask)
        if h_pre is None or h_post is None or c_pre is None or c_post is None:
            return
        n = int(mask.to(device=h_before.device).view(-1).sum().item())
        print(
            f"[LSTM reset] calls={self._dbg_calls} n={n} "
            f"h_mean={h_pre[0]:.6f}->{h_post[0]:.6f} h_var={h_pre[1]:.6f}->{h_post[1]:.6f} "
            f"c_mean={c_pre[0]:.6f}->{c_post[0]:.6f} c_var={c_pre[1]:.6f}->{c_post[1]:.6f}"
        )

    def compute(self, inputs, role: str = ""):
        x, rnn_reset = self._fuse_states(inputs)
        if self.enable_input_ln:
            x = self.in_ln(x)
        # 合并 terminated / truncated，确保超时也清零状态；保持 batch 维对齐
        term = inputs.get("terminated", None)
        trunc = inputs.get("truncated", None)
        if term is None and trunc is None:
            done = None
        elif term is None:
            done = trunc
        elif trunc is None:
            done = term
        else:
            done = torch.logical_or(term, trunc)

        h0, c0 = inputs["rnn"]
        is_update = "taken_actions" in inputs
        batch = x.shape[0]
        if done is not None:
            done = done.to(h0.device).view(-1)

        self._dbg_calls += 1

        rnn_outputs = []

        env_first = self.num_envs > 0 and batch > 0 and batch % self.num_envs == 0
        if env_first:
            envs = self.num_envs
            T = batch // envs
            x_seq = x.view(envs, T, -1)
            done_seq = done.view(envs, T) if done is not None and done.numel() == batch else None
            reset_seq = None
            h0, c0 = self._align_states(h0, c0, envs)
            if rnn_reset is not None:
                if rnn_reset.numel() == batch:
                    reset_seq = rnn_reset.view(envs, T)
                elif rnn_reset.numel() == envs:
                    reset_mask = rnn_reset.view(envs).bool()
                    h_before, c_before = h0, c0
                    h0, c0 = self._reset_done(h0, c0, reset_mask)
                    self._print_reset_stats(reset_mask, h_before, c_before, h0, c0, is_update)

            for t in range(T):
                if reset_seq is not None:
                    reset_mask = reset_seq[:, t].bool()
                    h_before, c_before = h0, c0
                    h0, c0 = self._reset_done(h0, c0, reset_mask)
                    self._print_reset_stats(reset_mask, h_before, c_before, h0, c0, is_update)
                step_in = x_seq[:, t : t + 1, :]
                step_out, (h0, c0) = self.lstm(step_in, (h0, c0))
                if done_seq is not None:
                    h0, c0 = self._reset_done(h0, c0, done_seq[:, t].bool())
                rnn_outputs.append(step_out)

            rnn_output = torch.cat(rnn_outputs, dim=1).reshape(batch, self.hidden_size)
        else:
            x1 = x.unsqueeze(1)  # (B, 1, 128)

            h0, c0 = self._align_states(h0, c0, batch)

            if rnn_reset is not None and rnn_reset.numel() == batch:
                reset_mask = rnn_reset.view(-1).bool()
                h_before, c_before = h0, c0
                h0, c0 = self._reset_done(h0, c0, reset_mask)
                self._print_reset_stats(reset_mask, h_before, c_before, h0, c0, is_update)
            out, (h0, c0) = self.lstm(x1, (h0, c0))  # out: (B, 1, hidden)

            if done is not None:
                done_mask = done.to(device=h0.device).view(-1).bool()
                if done_mask.numel() == h0.shape[1]:
                    h0, c0 = self._reset_done(h0, c0, done_mask)
                # else: ignore to avoid wrong indexing; optionally add assert for debugging.

            rnn_output = out.squeeze(1)  # (B, hidden)

        if self.enable_output_ln:
            rnn_output = self.out_ln(rnn_output)
        if self.c_clip and self.c_clip > 0:
            c0 = c0.clamp(-self.c_clip, self.c_clip)
        if done is None and self.reset_on_missing_done:
            h0 = torch.zeros_like(h0)
            c0 = torch.zeros_like(c0)
        if done is not None and done.numel() != h0.shape[1] and self.reset_on_mismatch:
            h0 = torch.zeros_like(h0)
            c0 = torch.zeros_like(c0)
        if self.reset_interval and self.reset_interval > 0 and (self._dbg_calls % self.reset_interval == 0):
            h0 = torch.zeros_like(h0)
            c0 = torch.zeros_like(c0)

        if self.stats_print_every > 0 and not is_update:
            self._stats_calls += 1
            if self._stats_calls % self.stats_print_every == 0:
                h_stats = h0.detach()
                c_stats = c0.detach()
                h_mean = h_stats.mean().item()
                h_var = h_stats.var(unbiased=False).item()
                c_mean = c_stats.mean().item()
                c_var = c_stats.var(unbiased=False).item()
                print(
                    f"[LSTM stats] steps={self._stats_calls} "
                    f"h_mean={h_mean:.6f} h_var={h_var:.6f} "
                    f"c_mean={c_mean:.6f} c_var={c_var:.6f}"
                )

        mean_actions = self.net(rnn_output)
        return mean_actions, self.log_std_parameter, {"rnn": [h0, c0]}


class SharedLatentAligner(nn.Module):
    """Project vision and tactile features into a shared latent space and compute alignment loss."""

    def __init__(
        self,
        vision_dim: int = 512,
        tactile_dim: int = 256,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        loss_type: str = "cosine",
        tau: float = 0.1,
        enable: bool = True,
    ):
        super().__init__()
        self.proj_v = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.proj_t = nn.Sequential(
            nn.Linear(tactile_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.loss_type = str(loss_type)
        self.tau = float(tau)
        self.enable = bool(enable)

    def forward(
        self,
        wrist_resnet: torch.Tensor,
        tactile_left_resnet: torch.Tensor,
        tactile_right_resnet: torch.Tensor,
        tactile_valid: torch.Tensor | None,
    ):
        tactile_feat = torch.cat([tactile_left_resnet, tactile_right_resnet], dim=-1)
        z_v = self.proj_v(wrist_resnet)
        z_t = self.proj_t(tactile_feat)

        if tactile_valid is None:
            tactile_valid = torch.zeros((z_v.shape[0], 1), device=z_v.device, dtype=z_v.dtype)
        m = tactile_valid.float()
        z_fused = z_v * (1.0 - m) + 0.5 * (z_v + z_t) * m

        align_loss = self._compute_align_loss(z_v, z_t, m)
        metrics = self._compute_metrics(z_v, z_t, m)
        return z_v, z_t, z_fused, align_loss, metrics

    def _compute_align_loss(self, z_v: torch.Tensor, z_t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return z_v.new_zeros(())
        # Stop grad on vision: only tactile branch is updated by alignment loss.
        z_v_ref = z_v.detach()
        valid_mask = m.view(-1) > 0.5
        if not torch.any(valid_mask):
            return z_v.new_zeros(())
        if self.loss_type == "cosine":
            cos = F.cosine_similarity(z_v_ref, z_t, dim=-1)
            loss = 1.0 - cos
            valid = valid_mask.float()
            return (loss * valid).sum() / (valid.sum() + 1e-6)
        if self.loss_type == "infonce":
            idx = valid_mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() < 2:
                return z_v.new_zeros(())
            z_vv = F.normalize(z_v_ref[idx], dim=-1)
            z_tt = F.normalize(z_t[idx], dim=-1)
            sim = (z_vv @ z_tt.t()) / self.tau
            labels = torch.arange(sim.shape[0], device=sim.device)
            loss_vt = F.cross_entropy(sim, labels)
            loss_tv = F.cross_entropy(sim.t(), labels)
            return 0.5 * (loss_vt + loss_tv)
        raise ValueError(f"Unsupported align loss_type: {self.loss_type}")

    def _compute_metrics(self, z_v: torch.Tensor, z_t: torch.Tensor, m: torch.Tensor) -> dict[str, torch.Tensor]:
        valid_mask = m.view(-1) > 0.5
        valid_ratio = valid_mask.float().mean()
        if torch.any(valid_mask):
            cos = F.cosine_similarity(z_v, z_t, dim=-1)
            mean_cos = cos[valid_mask].mean()
            z_t_norm = z_t.norm(dim=-1)[valid_mask].mean()
        else:
            mean_cos = z_v.new_zeros(())
            z_t_norm = z_v.new_zeros(())
        z_v_norm = z_v.norm(dim=-1).mean()
        return {
            "mean_cosine": mean_cos.detach(),
            "valid_ratio": valid_ratio.detach(),
            "z_v_norm": z_v_norm.detach(),
            "z_t_norm": z_t_norm.detach(),
        }


class VisionTactileSharedLatentPolicy(GaussianMixin, Model):
    """Gaussian policy with shared latent alignment for vision and tactile features."""

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
        mlp_layers: list[int] | None = None,
        mlp_activation: str = "elu",
        latent_dim: int = 128,
        align_cfg: dict | None = None,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        if mlp_layers is None:
            mlp_layers = [512, 256, 128, 64]

        align_cfg = dict(align_cfg or {})
        align_cfg.setdefault("enable", False)
        align_cfg.setdefault("loss_type", "cosine")
        align_cfg.setdefault("tau", 0.1)
        align_cfg.setdefault("latent_dim", latent_dim)
        self.align_cfg = align_cfg
        self.align_enabled = bool(align_cfg.get("enable", False))

        self.wrist_key = "wrist_resnet"
        self.tactile_left_key = "tactile_left_resnet"
        self.tactile_right_key = "tactile_right_resnet"
        self.tactile_valid_key = "tactile_valid" if "tactile_valid" in observation_space else None
        self.proprio_key = "proprio_obs" if "proprio_obs" in observation_space else ("proprio" if "proprio" in observation_space else None)
        self.gripper_key = "gripper_state" if "gripper_state" in observation_space else None
        self.action_hist_key = "action_history" if "action_history" in observation_space else None

        self.wrist_dim = int(align_cfg.get("vision_dim", observation_space[self.wrist_key].shape[0] if self.wrist_key in observation_space else 512))
        self.tactile_dim = int(align_cfg.get("tactile_dim", observation_space[self.tactile_left_key].shape[0] if self.tactile_left_key in observation_space else 128))
        self.proprio_dim = int(observation_space[self.proprio_key].shape[0]) if self.proprio_key in observation_space else 0
        self.gripper_dim = int(observation_space[self.gripper_key].shape[0]) if self.gripper_key in observation_space else 0
        self.action_hist_dim = int(observation_space[self.action_hist_key].shape[0]) if self.action_hist_key in observation_space else 0

        self.latent_dim = int(align_cfg.get("latent_dim", latent_dim))
        self.aligner = SharedLatentAligner(
            vision_dim=self.wrist_dim,
            tactile_dim=self.tactile_dim * 2,
            latent_dim=self.latent_dim,
            hidden_dim=256,
            loss_type=align_cfg.get("loss_type", "cosine"),
            tau=align_cfg.get("tau", 0.1),
            enable=self.align_enabled,
        )
        print(
            f"[INFO] Shared-latent dims: wrist={self.wrist_dim}, tactile={self.tactile_dim}, latent={self.latent_dim}"
        )

        if self.align_enabled:
            input_dim = self.latent_dim + self.proprio_dim + self.gripper_dim + self.action_hist_dim
        else:
            input_dim = self.wrist_dim + self.tactile_dim * 2 + self.proprio_dim + self.gripper_dim + self.action_hist_dim

        layers = []
        last_dim = input_dim
        activation = nn.ELU if str(mlp_activation).lower() == "elu" else nn.ReLU
        for hidden_dim in mlp_layers:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation())
            last_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(last_dim, self.num_actions)

        self.log_std_parameter = nn.Parameter(
            torch.full(size=(self.num_actions,), fill_value=initial_log_std),
            requires_grad=not fixed_log_std,
        )

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        batch_size = inputs.get("states").shape[0]

        wrist = states.get(self.wrist_key, torch.zeros((batch_size, self.wrist_dim), device=self.device))
        tact_l = states.get(self.tactile_left_key, torch.zeros((batch_size, self.tactile_dim), device=self.device))
        tact_r = states.get(self.tactile_right_key, torch.zeros((batch_size, self.tactile_dim), device=self.device))
        tactile_valid = states.get(self.tactile_valid_key, None) if self.tactile_valid_key else None

        align_loss = wrist.new_zeros(())
        align_metrics: dict[str, torch.Tensor] = {
            "mean_cosine": wrist.new_zeros(()),
            "valid_ratio": wrist.new_zeros(()),
            "z_v_norm": wrist.new_zeros(()),
            "z_t_norm": wrist.new_zeros(()),
        }

        if self.align_enabled:
            z_v, z_t, z_fused, align_loss, align_metrics = self.aligner(wrist, tact_l, tact_r, tactile_valid)
            base_feat = z_fused
        else:
            base_feat = torch.cat([wrist, tact_l, tact_r], dim=-1)

        feats = [base_feat]
        if self.proprio_key:
            feats.append(states.get(self.proprio_key))
        if self.gripper_key:
            feats.append(states.get(self.gripper_key))
        if self.action_hist_key:
            feats.append(states.get(self.action_hist_key))
        x = torch.cat([f for f in feats if f is not None], dim=-1)

        x = self.mlp(x)
        output = self.mu(x)
        return output, self.log_std_parameter, {"align_loss": align_loss, "align_metrics": align_metrics}
