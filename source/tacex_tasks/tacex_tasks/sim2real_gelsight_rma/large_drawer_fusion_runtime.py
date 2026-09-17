"""Runtime adapters shared by play, export, rollout, and comparison scripts."""

from __future__ import annotations

from typing import Any

import torch

from .rma_gelsight_large_drawer_fusion_models import (
    TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE,
    TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE,
    TACTILE_CROSS_ALPHA_DOWNSAMPLE,
    VISION_ONLY_DOWNSAMPLE,
)
from .rma_gelsight_pulled_drawer_artifacts import (
    LARGE_DRAWER_FUSION_VARIANT_BY_TASK,
)


def initial_recurrent_state(
    task: str, num_envs: int, device: str | torch.device
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    variant = LARGE_DRAWER_FUSION_VARIANT_BY_TASK.get(task)
    if variant != TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE:
        return None, None
    history = torch.zeros((num_envs, 4, 9, 256), dtype=torch.float32, device=device)
    reset_mask = torch.ones((num_envs,), dtype=torch.bool, device=device)
    return history, reset_mask


def observation_model_inputs(
    model: torch.nn.Module,
    observations: dict[str, torch.Tensor],
    *,
    recurrent_state: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    contract = model.contract()
    values: list[torch.Tensor] = []
    for key in contract["runtime_input_order"]:
        if key == "tactile_feature_history":
            if recurrent_state is None:
                raise ValueError("Recurrent Student requires tactile_feature_history")
            values.append(recurrent_state)
        elif key == "reset_mask":
            if reset_mask is None:
                raise ValueError("Recurrent Student requires reset_mask")
            values.append(reset_mask)
        else:
            value = observations[key]
            if key in {"proprio_obs", "action_history"}:
                value = value.float()
            values.append(value)
    return tuple(values)


def run_student_model(
    model: torch.nn.Module,
    task: str,
    observations: dict[str, torch.Tensor],
    *,
    recurrent_state: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    inputs = observation_model_inputs(
        model,
        observations,
        recurrent_state=recurrent_state,
        reset_mask=reset_mask,
    )
    outputs = model(*inputs)
    variant = LARGE_DRAWER_FUSION_VARIANT_BY_TASK.get(task)
    result: dict[str, torch.Tensor | None] = {
        "action": outputs[0],
        "contact_probability": None,
        "cube_position_root_m": None,
        "alpha": None,
        "occlusion_probability": None,
        "next_tactile_feature_history": None,
    }
    if variant == VISION_ONLY_DOWNSAMPLE:
        result["cube_position_root_m"] = outputs[1]
    elif variant == TACTILE_CROSS_ALPHA_DOWNSAMPLE:
        result.update(
            contact_probability=outputs[1],
            cube_position_root_m=outputs[2],
            alpha=outputs[3],
        )
    elif variant == TACTILE_CROSS_ALPHA_AUX_DOWNSAMPLE:
        result.update(
            contact_probability=outputs[1],
            cube_position_root_m=outputs[2],
            alpha=outputs[3],
            occlusion_probability=outputs[4],
        )
    elif variant == TACTILE_CROSS_ALPHA_AUX_GRU_DOWNSAMPLE:
        result.update(
            contact_probability=outputs[1],
            cube_position_root_m=outputs[2],
            alpha=outputs[3],
            occlusion_probability=outputs[4],
            next_tactile_feature_history=outputs[5],
        )
    else:
        result.update(
            contact_probability=outputs[1],
            cube_position_root_m=outputs[2],
        )
    return result


def update_recurrent_state(
    result: dict[str, torch.Tensor | None],
    terminated: torch.Tensor,
    truncated: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    history = result.get("next_tactile_feature_history")
    if history is None:
        return None, None
    return history.detach(), torch.logical_or(terminated, truncated)


__all__ = (
    "initial_recurrent_state",
    "observation_model_inputs",
    "run_student_model",
    "update_recurrent_state",
)
