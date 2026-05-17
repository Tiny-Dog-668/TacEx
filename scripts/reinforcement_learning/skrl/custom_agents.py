import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.ppo import PPO
from skrl.resources.schedulers.torch import KLAdaptiveLR


class PPOWithAlignLoss(PPO):
    """PPO with an extra alignment loss term produced by the policy model."""

    def __init__(self, *args, align_cfg: dict | None = None, **kwargs) -> None:
        self._align_cfg = dict(align_cfg or {})
        self._align_enabled = bool(self._align_cfg.get("enable", False))
        self._lambda_align = float(self._align_cfg.get("lambda_align", 0.05))
        super().__init__(*args, **kwargs)

    def _update(self, timestep: int, timesteps: int) -> None:
        def compute_gae(
            rewards: torch.Tensor,
            dones: torch.Tensor,
            values: torch.Tensor,
            next_values: torch.Tensor,
            discount_factor: float = 0.99,
            lambda_coefficient: float = 0.95,
        ) -> torch.Tensor:
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            for i in reversed(range(memory_size)):
                next_values = values[i + 1] if i < memory_size - 1 else last_values
                advantage = (
                    rewards[i]
                    - values[i]
                    + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
                )
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value"
            )
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_align_loss = 0
        cumulative_mean_cos = 0
        cumulative_valid_ratio = 0
        cumulative_zv_norm = 0
        cumulative_zt_norm = 0
        align_batches = 0

        def _to_float(value) -> float:
            if value is None:
                return 0.0
            if torch.is_tensor(value):
                return float(value.item())
            return float(value)

        for epoch in range(self._learning_epochs):
            kl_divergences = []

            for (
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
            ) in sampled_batches:

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = self._state_preprocessor(sampled_states, train=not epoch)

                    _, next_log_prob, policy_outputs = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)

                    if self._kl_threshold and kl_divergence > self._kl_threshold:
                        break

                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    align_loss = None
                    align_metrics = {}
                    if self._align_enabled and isinstance(policy_outputs, dict):
                        align_loss = policy_outputs.get("align_loss", None)
                        align_metrics = policy_outputs.get("align_metrics", {})
                        if align_loss is None:
                            align_loss = predicted_values.new_zeros(())
                        elif align_loss.dim() > 0:
                            align_loss = align_loss.mean()
                    else:
                        align_loss = predicted_values.new_zeros(())

                    total_loss = policy_loss + entropy_loss + value_loss
                    if self._align_enabled:
                        total_loss = total_loss + self._lambda_align * align_loss

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(
                            itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

                if self._align_enabled:
                    align_batches += 1
                    cumulative_align_loss += _to_float(align_loss)
                    cumulative_mean_cos += _to_float(align_metrics.get("mean_cosine"))
                    cumulative_valid_ratio += _to_float(align_metrics.get("valid_ratio"))
                    cumulative_zv_norm += _to_float(align_metrics.get("z_v_norm"))
                    cumulative_zt_norm += _to_float(align_metrics.get("z_t_norm"))

            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.scheduler.step(kl.item())
                else:
                    self.scheduler.step()

        denom = self._learning_epochs * self._mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / denom)
        self.track_data("Loss / Value loss", cumulative_value_loss / denom)
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / denom)

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])

        if self._align_enabled and align_batches > 0:
            self.track_data("Loss / Align loss", cumulative_align_loss / align_batches)
            self.track_data("Alignment / Mean cosine", cumulative_mean_cos / align_batches)
            self.track_data("Alignment / Valid ratio", cumulative_valid_ratio / align_batches)
            self.track_data("Alignment / z_v_norm", cumulative_zv_norm / align_batches)
            self.track_data("Alignment / z_t_norm", cumulative_zt_norm / align_batches)
