"""Reference-delta tactile Student for the fixed-size GelSight task."""

from __future__ import annotations

import torch
import torch.nn as nn

from tacex_tasks.sim2real_grasp.rma_models import RMAActorCore, RMAVisualStudent


class ReferenceDeltaTactileContactHead(nn.Module):
    """Predict one contact logit per side from signed current-reference RGB."""

    def __init__(self) -> None:
        super().__init__()
        # The same encoder and scalar classifier are applied independently to
        # left/right signed inputs [N,3,96,128].
        self.side_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.side_output = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 1)
        )
        final = self.side_output[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, -2.0)

    @staticmethod
    def signed_delta(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_scripting():
            if current.ndim != 4 or current.shape[-1] != 3:
                raise ValueError(f"Expected current tactile [N,H,W,3], got {tuple(current.shape)}")
            if current.shape != reference.shape:
                raise ValueError("Current/reference tactile shapes differ")
        return (
            (current.to(torch.float32) - reference.to(torch.float32))
            .div(255.0)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def _one_side(self, current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return self.side_output(self.side_encoder(self.signed_delta(current, reference)))

    def forward(
        self,
        left_current: torch.Tensor,
        right_current: torch.Tensor,
        left_reference: torch.Tensor,
        right_reference: torch.Tensor,
    ) -> torch.Tensor:
        left = self._one_side(left_current, left_reference)
        right = self._one_side(right_current, right_reference)
        return torch.cat((left, right), dim=-1)


class RMAGelSightReferenceStudent(RMAVisualStudent):
    """XYZ visual localizer plus hard binary reference-delta tactile contact."""

    def __init__(
        self,
        actor_core: RMAActorCore,
        *,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__(
            actor_core,
            pretrained_backbone=pretrained_backbone,
            use_tactile_contact=False,
        )
        # Remove the unused legacy cross-side head from the serialized contract.
        self.tactile_contact_head = ReferenceDeltaTactileContactHead()

    def predict_reference_adaptation(
        self,
        wrist_rgb: torch.Tensor,
        left_current: torch.Tensor,
        right_current: torch.Tensor,
        left_reference: torch.Tensor,
        right_reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_position, _ = self.adaptation_head(self.encode(wrist_rgb))
        contact_logits = self.tactile_contact_head(
            left_current, right_current, left_reference, right_reference
        )
        return normalized_position, contact_logits

    def forward(
        self,
        wrist_rgb: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        left_current: torch.Tensor,
        right_current: torch.Tensor,
        left_reference: torch.Tensor,
        right_reference: torch.Tensor,
    ) -> torch.Tensor:
        normalized_position, contact_logits = self.predict_reference_adaptation(
            wrist_rgb,
            left_current,
            right_current,
            left_reference,
            right_reference,
        )
        # Hard comparison is deliberately non-differentiable. Action loss can
        # train position/backbone parameters but cannot update the tactile head.
        hard_contact = (contact_logits >= 0.0).to(dtype=normalized_position.dtype)
        return self.action_from_normalized_position(
            proprio_obs, action_history, normalized_position, hard_contact
        )


__all__ = ("ReferenceDeltaTactileContactHead", "RMAGelSightReferenceStudent")
