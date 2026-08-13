"""Shared-ResNet three-frame Student for the X040-Wide profile."""

from __future__ import annotations

import torch
import torch.nn as nn

from .rma_x040_wide_models import RMAX040WideDirectActionVisualStudent


RMA_X040_WIDE_THREE_FRAME_DIRECT_STUDENT_MODEL_VERSION = 1


class RMAX040WideThreeFrameDirectActionVisualStudent(
    RMAX040WideDirectActionVisualStudent
):
    """Encode three frames with one shared ResNet and fuse 1536 features to 512."""

    def __init__(self) -> None:
        super().__init__()
        self.temporal_fusion = nn.Sequential(
            nn.Linear(3 * 512, 512),
            nn.ELU(),
        )

    def encode_visual_features(self, wrist_rgb_history: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            if wrist_rgb_history.ndim != 5 or wrist_rgb_history.shape[1:] != (
                3,
                224,
                224,
                3,
            ):
                raise ValueError(
                    "Expected wrist_rgb_history [N,3,224,224,3], got "
                    f"{tuple(wrist_rgb_history.shape)}"
                )
        batch_size = wrist_rgb_history.shape[0]
        # [N,3,H,W,C] -> [N*3,H,W,C] -> shared ResNet GAP [N*3,512].
        frames = wrist_rgb_history.reshape(-1, 224, 224, 3)
        image = frames.to(torch.float32).permute(0, 3, 1, 2) / 255.0
        feature_map = self.vision_encoder((image - self.image_mean) / self.image_std)
        per_frame_features = torch.flatten(
            torch.nn.functional.adaptive_avg_pool2d(feature_map, (1, 1)),
            1,
        ).reshape(
            batch_size,
            3 * 512,
        )
        # Concatenated [N,1536] -> fused visual feature [N,512].
        return self.temporal_fusion(per_frame_features)

    def contract(self) -> dict[str, object]:
        return {
            "model_version": RMA_X040_WIDE_THREE_FRAME_DIRECT_STUDENT_MODEL_VERSION,
            "input_order": ["wrist_rgb_history", "proprio_obs", "action_history"],
            "input_signature": {
                "wrist_rgb_history": [3, 224, 224, 3],
                "proprio_obs": [15],
                "action_history": [4],
            },
            "frame_order": "oldest_to_newest",
            "frame_stride_policy_steps": 1,
            "output_signature": {"mean_actions": [4]},
            "visual_feature_dim": 512,
            "action_feature_dim": 531,
            "feature_order": [
                "linear_elu(concat(shared_resnet18_gap(frame_t-2,t-1,t))[1536])[512]",
                "normalized_proprio_obs[15]",
                "normalized_action_history[4]",
            ],
            "runtime_privileged_inputs": [],
            "training_only_label": "normalized_cube_position_root[3]",
            "vision_encoder": "shared_resnet18_layer4_global_average_pool_per_frame",
            "temporal_fusion": [1536, 512],
            "action_head": [531, 512, 256, 128, 64, 4],
            "position_head": [512, 256, 128, 3],
            "activation": "temporal_elu_then_action_head_elu_then_action_tanh",
            "batch_norm": "all_shared_resnet18_batch_norm_eval",
        }


__all__ = (
    "RMA_X040_WIDE_THREE_FRAME_DIRECT_STUDENT_MODEL_VERSION",
    "RMAX040WideThreeFrameDirectActionVisualStudent",
)
