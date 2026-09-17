"""Six deployable vision/tactile fusion Students for the Large Drawer study."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rma_gelsight_pulled_drawer_four_tactile_models import (
    RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent,
)
from .rma_gelsight_pulled_drawer_models import (
    RMAGelSightPulledDrawerThreeFrameStudent,
)
from .rma_gelsight_x040_binary_tactile_models import (
    BinaryReferenceDeltaTactileEncoder,
)


VISION_ONLY_DOWNSAMPLE = "vision_only_downsample"
VT_DOWNSAMPLE = "vt_downsample"
TACTILE_CROSS_DOWNSAMPLE = "tactile_cross_downsample"
TACTILE_CROSS_ALPHA_DOWNSAMPLE = "tactile_cross_alpha_downsample"
TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE = "tactile_cross_alpha_aux_downsample"
TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE = "tactile_cross_alpha_aux_gru_downsample"

LARGE_DRAWER_FUSION_VARIANTS = (
    VISION_ONLY_DOWNSAMPLE,
    VT_DOWNSAMPLE,
    TACTILE_CROSS_DOWNSAMPLE,
    TACTILE_CROSS_ALPHA_DOWNSAMPLE,
    TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE,
    TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE,
)

LARGE_DRAWER_FUSION_MODEL_VERSION = 1
VISUAL_FRAME_FEATURE_DIM = 512
FUSION_DIM = 256
PROPRIO_DIM = 15
ACTION_HISTORY_DIM = 4
TACTILE_FEATURE_DIM = 256
TACTILE_SENSOR_COUNT = 4
TACTILE_HISTORY_LENGTH = 10
TACTILE_HISTORY_STATE_LENGTH = TACTILE_HISTORY_LENGTH - 1


def _base_contract(variant: str) -> dict[str, object]:
    if variant not in LARGE_DRAWER_FUSION_VARIANTS:
        raise ValueError(f"Unsupported Large Drawer fusion variant: {variant}")
    tactile = variant != VISION_ONLY_DOWNSAMPLE
    recurrent = variant == TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE
    runtime_inputs = ["wrist_rgb_history", "proprio_obs", "action_history"]
    if tactile:
        runtime_inputs.extend(
            [
                "gsmini_left_rgb",
                "gsmini_right_rgb",
                "gsmini_left_down_rgb",
                "gsmini_right_down_rgb",
                "gsmini_left_reference_rgb",
                "gsmini_right_reference_rgb",
                "gsmini_left_down_reference_rgb",
                "gsmini_right_down_reference_rgb",
            ]
        )
    if recurrent:
        runtime_inputs.extend(["tactile_feature_history", "reset_mask"])
    runtime_output: dict[str, object] = {
        "action": [4],
        "cube_position_root_m": [3],
    }
    if tactile:
        runtime_output["contact_probability"] = [4]
    if variant in {
        TACTILE_CROSS_ALPHA_DOWNSAMPLE,
        TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE,
        TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE,
    }:
        runtime_output["alpha"] = [1]
    if variant in {
        TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE,
        TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE,
    }:
        runtime_output["occlusion_probability"] = [1]
    if recurrent:
        runtime_output["next_tactile_feature_history"] = [4, 9, 256]
    return {
        "model_version": LARGE_DRAWER_FUSION_MODEL_VERSION,
        "fusion_variant": variant,
        "runtime_input_order": runtime_inputs,
        "runtime_output": runtime_output,
        "runtime_privileged_inputs": [],
        "visual_encoder": "shared_resnet18_layer4_global_average_pool_per_frame",
        "frame_order": "oldest_to_newest",
        "visual_history_shape": [3, 224, 224, 3],
        "position_head": [512, 256, 128, 3],
        "position_normalization": "pulled_drawer_contiguous_rigid_tray_robot_root_xyz_v3",
        "contact_order": ["left_inner", "right_inner", "left_down", "right_down"],
        "tactile_preprocessing": "max_abs_rgb_current_minus_reference_strict_gt_5_u8",
        "recurrent": recurrent,
    }


def large_drawer_fusion_model_contract(variant: str) -> dict[str, object]:
    contract = _base_contract(variant)
    if variant == VISION_ONLY_DOWNSAMPLE:
        contract.update(
            {
                "actor_feature_order": [
                    "temporal_visual[512]",
                    "normalized_proprio[15]",
                    "normalized_action_history[4]",
                ],
                "actor_feature_dim": 531,
                "action_head": [531, 512, 256, 128, 64, 4],
                "contact_head": "absent",
            }
        )
        return contract
    if variant == VT_DOWNSAMPLE:
        # Keep this exactly aligned with the pre-existing VT concatenation baseline.
        contract.update(
            {
                "actor_feature_order": [
                    "temporal_visual[512]",
                    "four_tactile_features[4,256]",
                    "normalized_proprio[15]",
                    "normalized_action_history[4]",
                ],
                "actor_feature_dim": 1555,
                "action_head": [1555, 512, 256, 128, 64, 4],
                "contact_head": "per_sensor_feature_256_to_logit",
            }
        )
        return contract
    contract.update(
        {
            "visual_tokens": {
                "shape": [3, 256],
                "source": "per_frame_resnet18_global_pool_512_projected_to_256",
                "temporal_position_embedding": True,
            },
            "tactile_tokens": [4, 256],
            "cross_attention": {
                "query": "four_tactile_tokens",
                "key_value": "three_visual_frame_tokens",
                "heads": 4,
                "embedding_dim": 256,
                "feed_forward_dim": 512,
            },
            "actor_feature_order": [
                "fused_tactile[256]",
                "normalized_proprio[15]",
                "normalized_action_history[4]",
            ],
            "actor_feature_dim": 275,
            "action_head": [275, 256, 128, 64, 4],
            "contact_head": "per_sensor_feature_256_to_logit",
        }
    )
    if variant == TACTILE_CROSS_DOWNSAMPLE:
        contract["fusion"] = "tactile_base_plus_tactile_cross"
    else:
        contract["fusion"] = "alpha_times_tactile_base_plus_one_minus_alpha_times_tactile_cross"
        contract["alpha_shape"] = [1]
        contract["alpha_semantics"] = "weight_on_tactile_base"
        contract["alpha_gate"] = (
            [15, 64, 1]
            if variant == TACTILE_CROSS_ALPHA_DOWNSAMPLE
            else [20, 64, 1]
        )
    if variant in {
        TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE,
        TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE,
    }:
        contract.update(
            {
                "occlusion_head": [512, 128, 1],
                "occlusion_output": "sigmoid_probability",
                "occlusion_training_target": "frozen_xy_predictor_pseudo_label_only",
                "alpha_gate_input_order": [
                    "predicted_occlusion[1]",
                    "binary_tactile_contact_area_ratio[4]",
                    "normalized_proprio[15]",
                ],
            }
        )
    if variant == TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE:
        contract.update(
            {
                "tactile_history": {
                    "sequence_length": 10,
                    "external_state_shape": [4, 9, 256],
                    "reset_fill": "repeat_current_feature",
                    "order": "oldest_to_newest",
                },
                "per_sensor_gru": {
                    "input_dim": 256,
                    "hidden_dim": 64,
                    "layers": 2,
                    "output_projection_dim": 128,
                },
            }
        )
    return contract


class RMAGelSightLargeDrawerVisionOnlyStudent(RMAGelSightPulledDrawerThreeFrameStudent):
    """Three-frame vision-only ablation; tactile tensors are not accepted."""

    def __init__(self, *, pretrained_backbone: bool = True) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone)
        del self.tactile_encoder
        self.action_head = nn.Sequential(
            nn.Linear(531, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 4),
        )

    def forward_with_training_outputs(
        self,
        wrist_rgb_history: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.encode_visual_features(wrist_rgb_history)
        normalized_position = self.position_head(visual)
        features = torch.cat(
            (
                visual,
                self.normalizer.normalize_proprio(proprio_obs),
                self.normalizer.normalize_history(action_history),
            ),
            dim=-1,
        )
        return torch.tanh(self.action_head(features)), normalized_position

    def forward(
        self,
        wrist_rgb_history: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action, normalized_position = self.forward_with_training_outputs(
            wrist_rgb_history, proprio_obs, action_history
        )
        return action, self.normalizer.denormalize_position(normalized_position)

    @torch.jit.unused
    def contract(self) -> dict[str, object]:
        return large_drawer_fusion_model_contract(VISION_ONLY_DOWNSAMPLE)


class RMAGelSightLargeDrawerVTStudent(
    RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent
):
    """Named wrapper for the existing four-tactile concatenation baseline."""

    @torch.jit.unused
    def contract(self) -> dict[str, object]:
        return large_drawer_fusion_model_contract(VT_DOWNSAMPLE)


class RMAGelSightLargeDrawerTactileCrossStudent(
    RMAGelSightPulledDrawerFourBinaryTactileThreeFrameStudent
):
    """Four tactile queries attend over three temporally ordered visual tokens."""

    variant = TACTILE_CROSS_DOWNSAMPLE

    def __init__(self, *, pretrained_backbone: bool = True) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone)
        self.visual_token_projection = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, FUSION_DIM), nn.ELU()
        )
        self.visual_temporal_embedding = nn.Parameter(
            torch.zeros(3, FUSION_DIM)
        )
        nn.init.normal_(self.visual_temporal_embedding, std=0.02)
        self.tactile_sensor_embedding = nn.Parameter(
            torch.zeros(TACTILE_SENSOR_COUNT, FUSION_DIM)
        )
        nn.init.normal_(self.tactile_sensor_embedding, std=0.02)
        self.query_norm = nn.LayerNorm(FUSION_DIM)
        self.context_norm = nn.LayerNorm(FUSION_DIM)
        self.tactile_visual_attention = nn.MultiheadAttention(
            FUSION_DIM, 4, dropout=0.0, batch_first=True
        )
        self.attention_ff_norm = nn.LayerNorm(FUSION_DIM)
        self.attention_ff = nn.Sequential(
            nn.Linear(FUSION_DIM, 512), nn.GELU(), nn.Linear(512, FUSION_DIM)
        )
        self.tactile_base_projection = nn.Sequential(
            nn.LayerNorm(4 * TACTILE_FEATURE_DIM),
            nn.Linear(4 * TACTILE_FEATURE_DIM, FUSION_DIM),
            nn.ELU(),
        )
        self.tactile_cross_projection = nn.Sequential(
            nn.LayerNorm(4 * FUSION_DIM),
            nn.Linear(4 * FUSION_DIM, FUSION_DIM),
            nn.ELU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(275, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 4),
        )

    def encode_visual_tokens(
        self, wrist_rgb_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.jit.is_scripting():
            if wrist_rgb_history.ndim != 5 or wrist_rgb_history.shape[1:] != (3, 224, 224, 3):
                raise ValueError(
                    "Expected wrist_rgb_history [N,3,224,224,3], got "
                    f"{tuple(wrist_rgb_history.shape)}"
                )
        batch_size = wrist_rgb_history.shape[0]
        frames = wrist_rgb_history.reshape(-1, 224, 224, 3)
        images = frames.to(torch.float32).permute(0, 3, 1, 2).div(255.0)
        maps = self.vision_encoder((images - self.image_mean) / self.image_std)
        per_frame = torch.flatten(F.adaptive_avg_pool2d(maps, (1, 1)), 1).reshape(
            batch_size, 3, VISUAL_FRAME_FEATURE_DIM
        )
        visual = self.temporal_fusion(per_frame.reshape(batch_size, 3 * VISUAL_FRAME_FEATURE_DIM))
        tokens = self.visual_token_projection(per_frame) + self.visual_temporal_embedding.unsqueeze(0)
        return visual, tokens

    def _encode_current_tactile(
        self,
        left_current: torch.Tensor,
        right_current: torch.Tensor,
        left_down_current: torch.Tensor,
        right_down_current: torch.Tensor,
        left_reference: torch.Tensor,
        right_reference: torch.Tensor,
        left_down_reference: torch.Tensor,
        right_down_reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left, right, inner_logits = self.inner_tactile_encoder(
            left_current, right_current, left_reference, right_reference
        )
        left_down, right_down, down_logits = self.down_tactile_encoder(
            left_down_current,
            right_down_current,
            left_down_reference,
            right_down_reference,
        )
        features = torch.stack((left, right, left_down, right_down), dim=1)
        return features, torch.cat((inner_logits, down_logits), dim=-1)

    def _cross_features(
        self, tactile_tokens: torch.Tensor, visual_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = tactile_tokens.shape[0]
        tactile_base = self.tactile_base_projection(tactile_tokens.reshape(batch_size, -1))
        query = self.query_norm(
            tactile_tokens + self.tactile_sensor_embedding.unsqueeze(0)
        )
        context = self.context_norm(visual_tokens)
        attended, attention = self.tactile_visual_attention(
            query, context, context, need_weights=True, average_attn_weights=False
        )
        crossed = query + attended
        crossed = crossed + self.attention_ff(self.attention_ff_norm(crossed))
        tactile_cross = self.tactile_cross_projection(crossed.reshape(batch_size, -1))
        return tactile_base + tactile_cross, attention

    def _action_from_fused(
        self,
        fused_tactile: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                fused_tactile,
                self.normalizer.normalize_proprio(proprio_obs),
                self.normalizer.normalize_history(action_history),
            ),
            dim=-1,
        )
        return torch.tanh(self.action_head(features))

    def forward_with_training_outputs(
        self, wrist_rgb_history, proprio_obs, action_history,
        left_current, right_current, left_down_current, right_down_current,
        left_reference, right_reference, left_down_reference, right_down_reference,
    ):
        visual, visual_tokens = self.encode_visual_tokens(wrist_rgb_history)
        tactile_tokens, contact_logits = self._encode_current_tactile(
            left_current, right_current, left_down_current, right_down_current,
            left_reference, right_reference, left_down_reference, right_down_reference,
        )
        fused, _ = self._cross_features(tactile_tokens, visual_tokens)
        return self._action_from_fused(fused, proprio_obs, action_history), self.position_head(visual), contact_logits

    def forward(
        self, wrist_rgb_history, proprio_obs, action_history,
        left_current, right_current, left_down_current, right_down_current,
        left_reference, right_reference, left_down_reference, right_down_reference,
    ):
        action, position, logits = self.forward_with_training_outputs(
            wrist_rgb_history, proprio_obs, action_history,
            left_current, right_current, left_down_current, right_down_current,
            left_reference, right_reference, left_down_reference, right_down_reference,
        )
        return action, torch.sigmoid(logits), self.normalizer.denormalize_position(position)

    @torch.jit.unused
    def contract(self) -> dict[str, object]:
        return large_drawer_fusion_model_contract(self.variant)


class RMAGelSightLargeDrawerTactileCrossAlphaStudent(
    RMAGelSightLargeDrawerTactileCrossStudent
):
    """Proprio-conditioned scalar mixing of base and cross tactile features."""

    variant = TACTILE_CROSS_ALPHA_DOWNSAMPLE

    def __init__(self, *, pretrained_backbone: bool = True) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone)
        self.alpha_gate = nn.Sequential(
            nn.Linear(PROPRIO_DIM, 64), nn.ELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def _mix_features(
        self,
        tactile_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        proprio_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = tactile_tokens.shape[0]
        tactile_base = self.tactile_base_projection(tactile_tokens.reshape(batch_size, -1))
        query = self.query_norm(tactile_tokens + self.tactile_sensor_embedding.unsqueeze(0))
        context = self.context_norm(visual_tokens)
        attended, attention = self.tactile_visual_attention(
            query, context, context, need_weights=True, average_attn_weights=False
        )
        crossed = query + attended
        crossed = crossed + self.attention_ff(self.attention_ff_norm(crossed))
        tactile_cross = self.tactile_cross_projection(crossed.reshape(batch_size, -1))
        alpha = self.alpha_gate(self.normalizer.normalize_proprio(proprio_obs))
        return alpha * tactile_base + (1.0 - alpha) * tactile_cross, alpha, attention

    def forward_with_diagnostics(
        self, wrist_rgb_history, proprio_obs, action_history,
        left_current, right_current, left_down_current, right_down_current,
        left_reference, right_reference, left_down_reference, right_down_reference,
    ):
        visual, visual_tokens = self.encode_visual_tokens(wrist_rgb_history)
        tactile_tokens, contact_logits = self._encode_current_tactile(
            left_current, right_current, left_down_current, right_down_current,
            left_reference, right_reference, left_down_reference, right_down_reference,
        )
        fused, alpha, attention = self._mix_features(tactile_tokens, visual_tokens, proprio_obs)
        action = self._action_from_fused(fused, proprio_obs, action_history)
        return action, self.position_head(visual), contact_logits, alpha, attention

    def forward_with_training_outputs(
        self, wrist_rgb_history, proprio_obs, action_history,
        left_current, right_current, left_down_current, right_down_current,
        left_reference, right_reference, left_down_reference, right_down_reference,
    ):
        action, position, logits, _, _ = self.forward_with_diagnostics(
            wrist_rgb_history, proprio_obs, action_history,
            left_current, right_current, left_down_current, right_down_current,
            left_reference, right_reference, left_down_reference, right_down_reference,
        )
        return action, position, logits

    def forward(
        self, wrist_rgb_history, proprio_obs, action_history,
        left_current, right_current, left_down_current, right_down_current,
        left_reference, right_reference, left_down_reference, right_down_reference,
    ):
        action, position, logits, alpha, _ = self.forward_with_diagnostics(
            wrist_rgb_history, proprio_obs, action_history,
            left_current, right_current, left_down_current, right_down_current,
            left_reference, right_reference, left_down_reference, right_down_reference,
        )
        return action, torch.sigmoid(logits), self.normalizer.denormalize_position(position), alpha


class RMAGelSightLargeDrawerTactileCrossAlphaAuxStudent(
    RMAGelSightLargeDrawerTactileCrossAlphaStudent
):
    """Predict visual occlusion and use deployable tactile coverage in the gate."""

    variant = TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE

    def __init__(self, *, pretrained_backbone: bool = True) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone)
        self.occlusion_head = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, 128), nn.ELU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )
        self.alpha_gate = nn.Sequential(
            nn.Linear(1 + 4 + PROPRIO_DIM, 64), nn.ELU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    @staticmethod
    def _contact_area_ratios(
        left_current, right_current, left_down_current, right_down_current,
        left_reference, right_reference, left_down_reference, right_down_reference,
    ) -> torch.Tensor:
        masks = (
            BinaryReferenceDeltaTactileEncoder.binary_delta(left_current, left_reference),
            BinaryReferenceDeltaTactileEncoder.binary_delta(right_current, right_reference),
            BinaryReferenceDeltaTactileEncoder.binary_delta(left_down_current, left_down_reference),
            BinaryReferenceDeltaTactileEncoder.binary_delta(right_down_current, right_down_reference),
        )
        return torch.stack([mask.mean(dim=(1, 2, 3)) for mask in masks], dim=-1)

    def _mix_aux_features(
        self,
        tactile_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual: torch.Tensor,
        proprio_obs: torch.Tensor,
        contact_ratios: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = tactile_tokens.shape[0]
        tactile_base = self.tactile_base_projection(tactile_tokens.reshape(batch_size, -1))
        query = self.query_norm(tactile_tokens + self.tactile_sensor_embedding.unsqueeze(0))
        context = self.context_norm(visual_tokens)
        attended, attention = self.tactile_visual_attention(
            query, context, context, need_weights=True, average_attn_weights=False
        )
        crossed = query + attended
        crossed = crossed + self.attention_ff(self.attention_ff_norm(crossed))
        tactile_cross = self.tactile_cross_projection(crossed.reshape(batch_size, -1))
        occlusion = self.occlusion_head(visual)
        gate_input = torch.cat(
            (occlusion, contact_ratios, self.normalizer.normalize_proprio(proprio_obs)),
            dim=-1,
        )
        alpha = self.alpha_gate(gate_input)
        fused = alpha * tactile_base + (1.0 - alpha) * tactile_cross
        return fused, alpha, occlusion, attention

    def forward_with_auxiliary_outputs(
        self, wrist_rgb_history, proprio_obs, action_history,
        left_current, right_current, left_down_current, right_down_current,
        left_reference, right_reference, left_down_reference, right_down_reference,
    ):
        visual, visual_tokens = self.encode_visual_tokens(wrist_rgb_history)
        tactile_tokens, contact_logits = self._encode_current_tactile(
            left_current, right_current, left_down_current, right_down_current,
            left_reference, right_reference, left_down_reference, right_down_reference,
        )
        ratios = self._contact_area_ratios(
            left_current, right_current, left_down_current, right_down_current,
            left_reference, right_reference, left_down_reference, right_down_reference,
        )
        fused, alpha, occlusion, attention = self._mix_aux_features(
            tactile_tokens, visual_tokens, visual, proprio_obs, ratios
        )
        action = self._action_from_fused(fused, proprio_obs, action_history)
        return action, self.position_head(visual), contact_logits, alpha, occlusion, attention, ratios

    def forward_with_training_outputs(
        self, wrist_rgb_history, proprio_obs, action_history,
        left_current, right_current, left_down_current, right_down_current,
        left_reference, right_reference, left_down_reference, right_down_reference,
    ):
        action, position, logits, _, _, _, _ = self.forward_with_auxiliary_outputs(
            wrist_rgb_history, proprio_obs, action_history,
            left_current, right_current, left_down_current, right_down_current,
            left_reference, right_reference, left_down_reference, right_down_reference,
        )
        return action, position, logits

    def forward(
        self, wrist_rgb_history, proprio_obs, action_history,
        left_current, right_current, left_down_current, right_down_current,
        left_reference, right_reference, left_down_reference, right_down_reference,
    ):
        action, position, logits, alpha, occlusion, _, _ = self.forward_with_auxiliary_outputs(
            wrist_rgb_history, proprio_obs, action_history,
            left_current, right_current, left_down_current, right_down_current,
            left_reference, right_reference, left_down_reference, right_down_reference,
        )
        return (
            action,
            torch.sigmoid(logits),
            self.normalizer.denormalize_position(position),
            alpha,
            occlusion,
        )


class RMAGelSightLargeDrawerTactileCrossAlphaAuxGRUStudent(
    RMAGelSightLargeDrawerTactileCrossAlphaAuxStudent
):
    """Aux policy with an explicit nine-feature recurrent state per sensor."""

    variant = TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE

    def __init__(self, *, pretrained_backbone: bool = True) -> None:
        super().__init__(pretrained_backbone=pretrained_backbone)
        self.tactile_grus = nn.ModuleList(
            [
                nn.GRU(TACTILE_FEATURE_DIM, 64, num_layers=2, batch_first=True)
                for _ in range(TACTILE_SENSOR_COUNT)
            ]
        )
        self.tactile_gru_heads = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 128), nn.ELU()) for _ in range(4)]
        )
        self.tactile_gru_token_projection = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 256), nn.ELU()) for _ in range(4)]
        )

    def _temporal_tactile_tokens(
        self,
        current_tokens: torch.Tensor,
        history: torch.Tensor,
        reset_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.jit.is_scripting():
            expected = (current_tokens.shape[0], 4, 9, 256)
            if tuple(history.shape) != expected:
                raise ValueError(f"Expected tactile history {expected}, got {tuple(history.shape)}")
        reset = reset_mask.to(dtype=torch.bool).reshape(-1, 1, 1, 1)
        reset_fill = current_tokens.unsqueeze(2).expand(-1, -1, 9, -1)
        effective_history = torch.where(reset, reset_fill, history)
        sequence = torch.cat((effective_history, current_tokens.unsqueeze(2)), dim=2)
        output_tokens = []
        for sensor_index, (gru, head, projection) in enumerate(
            zip(self.tactile_grus, self.tactile_gru_heads, self.tactile_gru_token_projection)
        ):
            _, hidden = gru(sequence[:, sensor_index])
            latent = head(hidden[-1])
            output_tokens.append(projection(latent))
        next_history = torch.cat(
            (effective_history[:, :, 1:], current_tokens.unsqueeze(2)), dim=2
        )
        return torch.stack(output_tokens, dim=1), next_history

    def forward_with_recurrent_outputs(
        self, wrist_rgb_history, proprio_obs, action_history,
        left_current, right_current, left_down_current, right_down_current,
        left_reference, right_reference, left_down_reference, right_down_reference,
        tactile_feature_history, reset_mask,
    ):
        visual, visual_tokens = self.encode_visual_tokens(wrist_rgb_history)
        current_tokens, contact_logits = self._encode_current_tactile(
            left_current, right_current, left_down_current, right_down_current,
            left_reference, right_reference, left_down_reference, right_down_reference,
        )
        temporal_tokens, next_history = self._temporal_tactile_tokens(
            current_tokens, tactile_feature_history, reset_mask
        )
        ratios = self._contact_area_ratios(
            left_current, right_current, left_down_current, right_down_current,
            left_reference, right_reference, left_down_reference, right_down_reference,
        )
        fused, alpha, occlusion, attention = self._mix_aux_features(
            temporal_tokens, visual_tokens, visual, proprio_obs, ratios
        )
        action = self._action_from_fused(fused, proprio_obs, action_history)
        return (
            action, self.position_head(visual), contact_logits, alpha,
            occlusion, next_history, attention, ratios,
        )

    def forward(self, wrist_rgb_history, proprio_obs, action_history,
                left_current, right_current, left_down_current, right_down_current,
                left_reference, right_reference, left_down_reference, right_down_reference,
                tactile_feature_history, reset_mask):
        action, position, logits, alpha, occlusion, next_history, _, _ = (
            self.forward_with_recurrent_outputs(
                wrist_rgb_history, proprio_obs, action_history,
                left_current, right_current, left_down_current, right_down_current,
                left_reference, right_reference, left_down_reference, right_down_reference,
                tactile_feature_history, reset_mask,
            )
        )
        return (
            action,
            torch.sigmoid(logits),
            self.normalizer.denormalize_position(position),
            alpha,
            occlusion,
            next_history,
        )


LARGE_DRAWER_FUSION_MODEL_CLASSES = {
    VISION_ONLY_DOWNSAMPLE: RMAGelSightLargeDrawerVisionOnlyStudent,
    VT_DOWNSAMPLE: RMAGelSightLargeDrawerVTStudent,
    TACTILE_CROSS_DOWNSAMPLE: RMAGelSightLargeDrawerTactileCrossStudent,
    TACTILE_CROSS_ALPHA_DOWNSAMPLE: RMAGelSightLargeDrawerTactileCrossAlphaStudent,
    TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE: RMAGelSightLargeDrawerTactileCrossAlphaAuxStudent,
    TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE: RMAGelSightLargeDrawerTactileCrossAlphaAuxGRUStudent,
}


def make_large_drawer_fusion_student(
    variant: str, *, pretrained_backbone: bool = True
) -> nn.Module:
    try:
        cls = LARGE_DRAWER_FUSION_MODEL_CLASSES[variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported Large Drawer fusion variant: {variant}") from exc
    return cls(pretrained_backbone=pretrained_backbone)


def is_aux_variant(variant: str) -> bool:
    return variant in {
        TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE,
        TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE,
    }


def is_recurrent_variant(variant: str) -> bool:
    return variant == TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE


def is_tactile_variant(variant: str) -> bool:
    return variant != VISION_ONLY_DOWNSAMPLE


__all__ = tuple(
    name
    for name in globals()
    if name.startswith("RMA")
    or name.startswith("TACTILE_")
    or name.startswith("VISION_")
    or name.startswith("VT_")
    or name.startswith("LARGE_DRAWER_")
    or name.startswith("make_")
    or name.startswith("is_")
    or name.startswith("large_drawer_")
)
