"""Residual policy variant for per-sensor tactile gates + sensor-id embedding."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from skrl.utils.spaces.torch import unflatten_tensorized_space

from .policy_alpha import set_latest_alpha
from .vt_residual_policy import OccludedGraspingVTResidualPolicy


class OccludedGraspingVTSensorGateResidualPolicy(OccludedGraspingVTResidualPolicy):
    """Visual base + tactile residual with per-sensor binary gates and sensor-id embedding."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        tactile_sensor_gates_key: str = "tactile_sensor_gates",
        tactile_global_gate_key: str = "tactile_global_gate",
        **kwargs: Any,
    ):
        kwargs = dict(kwargs)
        kwargs.setdefault(
            "tactile_keys",
            (
                "tactile_left_rgb",
                "tactile_right_rgb",
                "tactile_left_down_depth",
                "tactile_right_down_depth",
            ),
        )
        kwargs["tactile_contact_bits_key"] = tactile_sensor_gates_key
        kwargs["tactile_valid_key"] = tactile_global_gate_key
        super().__init__(observation_space, action_space, device, **kwargs)

        self.tactile_sensor_gates_key = str(tactile_sensor_gates_key)
        self.tactile_global_gate_key = str(tactile_global_gate_key)
        self._has_tactile_global_gate = self._has_obs_key(observation_space, self.tactile_global_gate_key)

        # Keep gate information only once (inside each tactile token), and
        # rebuild residual_head input size to avoid re-feeding sensor_gates.
        tactile_token_dim = int(self.tactile_token_proj[1].out_features)
        token_plus_gate_dim = tactile_token_dim + 1
        tactile_agg_dim = self.num_tactile * token_plus_gate_dim
        residual_input_dim = tactile_agg_dim + self.proprio_dim + self.num_actions
        hidden_1 = int(self.residual_head[0].out_features)
        hidden_2 = int(self.residual_head[2].out_features)
        self.residual_head = nn.Sequential(
            nn.Linear(residual_input_dim, hidden_1),
            nn.ELU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ELU(),
            nn.Linear(hidden_2, self.num_actions),
        )

    def compute(self, inputs: Dict[str, Any], role: str = ""):
        states = inputs.get("states")
        if isinstance(states, dict):
            obs = states
        else:
            obs = unflatten_tensorized_space(self.observation_space, states)

        batch_size = self._collect_batch_size(obs)
        vision = self._flatten_obs(obs, self.vision_key, batch_size)
        proprio = self._flatten_obs(obs, self.proprio_key, batch_size)
        sensor_gates = self._flatten_obs(obs, self.tactile_sensor_gates_key, batch_size).clamp(0.0, 1.0)
        if self._has_tactile_global_gate:
            global_gate = self._flatten_obs(obs, self.tactile_global_gate_key, batch_size).clamp(0.0, 1.0)
        else:
            global_gate = sensor_gates.max(dim=-1, keepdim=True).values

        visual_input = torch.cat([vision, proprio], dim=-1)
        visual_hidden = self.visual_base_mlp(visual_input)
        a_vis = self.visual_action_head(visual_hidden)
        # a_vis = torch.clamp(a_vis, -1.0, 1.0)

        tactile_tokens = []
        sensor_ids = torch.arange(self.num_tactile, device=self.device, dtype=torch.long)
        for sensor_idx, key in enumerate(self.tactile_keys):
            tactile_feat = self._flatten_obs(obs, key, batch_size)
            gate_i = sensor_gates[:, sensor_idx : sensor_idx + 1]

            tactile_token = self.tactile_token_proj(tactile_feat)
            tactile_token = tactile_token + self.sensor_id_embedding(sensor_ids[sensor_idx]).unsqueeze(0)
            tactile_token = torch.cat([tactile_token, gate_i], dim=-1)
            tactile_tokens.append(tactile_token)

        tactile_agg = torch.cat(tactile_tokens, dim=-1)
        residual_input = torch.cat([tactile_agg, proprio, a_vis], dim=-1)
        delta_a_tac = torch.tanh(self.residual_head(residual_input)) * self.residual_action_scales

        gate = global_gate
        if not self.gate_per_dim:
            gate = gate.expand(-1, self.num_actions)
        set_latest_alpha(gate.mean(dim=-1, keepdim=True))

        mu = a_vis + gate * delta_a_tac
        return mu, self.log_std_parameter, {
            "a_vis": a_vis,
            "delta_a_tac": delta_a_tac,
            "gate": gate,
            "sensor_gates": sensor_gates,
        }
