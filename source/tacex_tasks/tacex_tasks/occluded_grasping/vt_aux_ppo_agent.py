"""Custom PPO agent that adds auxiliary supervised losses from policy outputs."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.agents.torch.ppo import PPO
from skrl.resources.schedulers.torch import KLAdaptiveLR


class PPOWithAuxHeadsLoss(PPO):
    """PPO with extra reliability/stage supervised losses produced by the policy model."""

    def __init__(self, *args, **kwargs) -> None:
        cfg = dict(kwargs.get("cfg", {}) or {})
        self._aux_enabled = bool(cfg.get("aux_enable", True))
        self._aux_loss_scale = float(cfg.get("aux_loss_scale", 1.0))
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

        rewards_tensor = self.memory.get_tensor_by_name("rewards")
        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=rewards_tensor,
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

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_aux_loss = 0.0
        cumulative_aux_reliability_loss = 0.0
        cumulative_aux_probe_loss = 0.0
        cumulative_aux_grasp_loss = 0.0
        cumulative_g_probe = 0.0
        cumulative_g_grasp = 0.0
        cumulative_alpha = 0.0
        cumulative_m_probe = 0.0
        cumulative_m_grasp = 0.0
        cumulative_router_load_loss = 0.0
        cumulative_router_entropy_loss = 0.0
        cumulative_router_max_prob = 0.0
        cumulative_router_selected = 0.0
        aux_batches = 0
        saw_g_probe = False
        saw_g_grasp = False
        saw_alpha = False
        saw_m_probe = False
        saw_m_grasp = False
        saw_router = False

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
                        entropy_loss = sampled_returns.new_zeros(())

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

                    aux_loss = predicted_values.new_zeros(())
                    aux_reliability_loss = predicted_values.new_zeros(())
                    aux_probe_loss = predicted_values.new_zeros(())
                    aux_grasp_loss = predicted_values.new_zeros(())
                    g_probe = predicted_values.new_zeros(())
                    g_grasp = predicted_values.new_zeros(())
                    alpha = predicted_values.new_zeros(())
                    m_probe = predicted_values.new_zeros(())
                    m_grasp = predicted_values.new_zeros(())
                    router_load_loss = predicted_values.new_zeros(())
                    router_entropy_loss = predicted_values.new_zeros(())
                    router_max_prob = predicted_values.new_zeros(())
                    router_selected = predicted_values.new_zeros(())
                    if self._aux_enabled and isinstance(policy_outputs, dict):
                        aux_loss = policy_outputs.get("aux_loss", aux_loss)
                        aux_reliability_loss = policy_outputs.get("aux_reliability_loss", aux_reliability_loss)
                        aux_probe_loss = policy_outputs.get("aux_probe_loss", aux_probe_loss)
                        aux_grasp_loss = policy_outputs.get("aux_grasp_loss", aux_grasp_loss)
                        if "g_probe" in policy_outputs:
                            g_probe = policy_outputs["g_probe"]
                            saw_g_probe = True
                        if "g_grasp" in policy_outputs:
                            g_grasp = policy_outputs["g_grasp"]
                            saw_g_grasp = True
                        if "alpha" in policy_outputs:
                            alpha = policy_outputs["alpha"]
                            saw_alpha = True
                        if "m_probe" in policy_outputs:
                            m_probe = policy_outputs["m_probe"]
                            saw_m_probe = True
                        if "m_grasp" in policy_outputs:
                            m_grasp = policy_outputs["m_grasp"]
                            saw_m_grasp = True
                        if "router_load_loss" in policy_outputs:
                            router_load_loss = policy_outputs["router_load_loss"]
                            saw_router = True
                        if "router_entropy_loss" in policy_outputs:
                            router_entropy_loss = policy_outputs["router_entropy_loss"]
                            saw_router = True
                        if "router_probs" in policy_outputs:
                            router_max_prob = policy_outputs["router_probs"].max(dim=-1).values
                            saw_router = True
                        if "router_selected" in policy_outputs:
                            router_selected = policy_outputs["router_selected"]
                            saw_router = True
                        if torch.is_tensor(aux_loss) and aux_loss.dim() > 0:
                            aux_loss = aux_loss.mean()
                        if torch.is_tensor(aux_reliability_loss) and aux_reliability_loss.dim() > 0:
                            aux_reliability_loss = aux_reliability_loss.mean()
                        if torch.is_tensor(aux_probe_loss) and aux_probe_loss.dim() > 0:
                            aux_probe_loss = aux_probe_loss.mean()
                        if torch.is_tensor(aux_grasp_loss) and aux_grasp_loss.dim() > 0:
                            aux_grasp_loss = aux_grasp_loss.mean()
                        if torch.is_tensor(g_probe) and g_probe.dim() > 0:
                            g_probe = g_probe.mean()
                        if torch.is_tensor(g_grasp) and g_grasp.dim() > 0:
                            g_grasp = g_grasp.mean()
                        if torch.is_tensor(alpha) and alpha.dim() > 0:
                            alpha = alpha.mean()
                        if torch.is_tensor(m_probe) and m_probe.dim() > 0:
                            m_probe = m_probe.mean()
                        if torch.is_tensor(m_grasp) and m_grasp.dim() > 0:
                            m_grasp = m_grasp.mean()
                        if torch.is_tensor(router_load_loss) and router_load_loss.dim() > 0:
                            router_load_loss = router_load_loss.mean()
                        if torch.is_tensor(router_entropy_loss) and router_entropy_loss.dim() > 0:
                            router_entropy_loss = router_entropy_loss.mean()
                        if torch.is_tensor(router_max_prob) and router_max_prob.dim() > 0:
                            router_max_prob = router_max_prob.mean()
                        if torch.is_tensor(router_selected) and router_selected.dim() > 0:
                            router_selected = router_selected.mean()

                    total_loss = policy_loss + entropy_loss + value_loss
                    if self._aux_enabled:
                        total_loss = total_loss + self._aux_loss_scale * aux_loss

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

                cumulative_policy_loss += _to_float(policy_loss)
                cumulative_value_loss += _to_float(value_loss)
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += _to_float(entropy_loss)
                if self._aux_enabled:
                    aux_batches += 1
                    cumulative_aux_loss += _to_float(aux_loss)
                    cumulative_aux_reliability_loss += _to_float(aux_reliability_loss)
                    cumulative_aux_probe_loss += _to_float(aux_probe_loss)
                    cumulative_aux_grasp_loss += _to_float(aux_grasp_loss)
                    cumulative_g_probe += _to_float(g_probe)
                    cumulative_g_grasp += _to_float(g_grasp)
                    cumulative_alpha += _to_float(alpha)
                    cumulative_m_probe += _to_float(m_probe)
                    cumulative_m_grasp += _to_float(m_grasp)
                    cumulative_router_load_loss += _to_float(router_load_loss)
                    cumulative_router_entropy_loss += _to_float(router_entropy_loss)
                    cumulative_router_max_prob += _to_float(router_max_prob)
                    cumulative_router_selected += _to_float(router_selected)

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

        if self._aux_enabled and aux_batches > 0:
            aux_loss_mean = cumulative_aux_loss / aux_batches
            aux_reliability_loss_mean = cumulative_aux_reliability_loss / aux_batches
            aux_probe_loss_mean = cumulative_aux_probe_loss / aux_batches
            aux_grasp_loss_mean = cumulative_aux_grasp_loss / aux_batches
            g_probe_mean = cumulative_g_probe / aux_batches
            g_grasp_mean = cumulative_g_grasp / aux_batches
            alpha_mean = cumulative_alpha / aux_batches
            m_probe_mean = cumulative_m_probe / aux_batches
            m_grasp_mean = cumulative_m_grasp / aux_batches

            self.track_data("Loss / Aux loss", aux_loss_mean)
            self.track_data("Loss / Aux reliability loss", aux_reliability_loss_mean)
            self.track_data("Loss / Aux probe loss", aux_probe_loss_mean)
            self.track_data("Loss / Aux grasp loss", aux_grasp_loss_mean)
            if saw_g_probe:
                self.track_data("Policy / g_probe", g_probe_mean)
            if saw_g_grasp:
                self.track_data("Policy / g_grasp", g_grasp_mean)
            if saw_alpha:
                self.track_data("Policy / alpha", alpha_mean)
            if saw_m_probe:
                self.track_data("Policy / hard gate bottom", m_probe_mean)
            if saw_m_grasp:
                self.track_data("Policy / hard gate inner", m_grasp_mean)
            if saw_router:
                self.track_data("Router / Load loss", cumulative_router_load_loss / aux_batches)
                self.track_data("Router / Entropy loss", cumulative_router_entropy_loss / aux_batches)
                self.track_data("Router / Max prob", cumulative_router_max_prob / aux_batches)
                self.track_data("Router / Selected expert", cumulative_router_selected / aux_batches)
