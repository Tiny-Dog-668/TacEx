"""Shared gate logging helpers for occluded grasping policies."""

from __future__ import annotations

import torch

_LATEST_GATE_MEAN: float | None = None
_LATEST_GATE_VALUES: torch.Tensor | None = None
_LATEST_ALPHA_VALUES: torch.Tensor | None = None
_LATEST_BETA_VALUES: torch.Tensor | None = None


def set_latest_gate(gate: torch.Tensor | None) -> None:
    """Store the latest per-sample gate values for env-side logging."""
    global _LATEST_GATE_MEAN, _LATEST_GATE_VALUES
    if gate is None:
        _LATEST_GATE_MEAN = None
        _LATEST_GATE_VALUES = None
        return
    gate = gate.detach().to(dtype=torch.float32).view(-1)
    _LATEST_GATE_MEAN = float(gate.mean().cpu().item())
    _LATEST_GATE_VALUES = gate.cpu()


def set_latest_alpha_beta(alpha: torch.Tensor | None, beta: torch.Tensor | None) -> None:
    """Store separate alpha/beta gates and their total tactile gate."""
    global _LATEST_ALPHA_VALUES, _LATEST_BETA_VALUES
    if alpha is None or beta is None:
        _LATEST_ALPHA_VALUES = None
        _LATEST_BETA_VALUES = None
        set_latest_gate(None)
        return
    alpha = alpha.detach().to(dtype=torch.float32).view(-1)
    beta = beta.detach().to(dtype=torch.float32).view(-1)
    _LATEST_ALPHA_VALUES = alpha.cpu()
    _LATEST_BETA_VALUES = beta.cpu()
    set_latest_gate(alpha + beta)


def get_latest_gate_mean() -> float | None:
    return _LATEST_GATE_MEAN


def get_latest_gate_values() -> torch.Tensor | None:
    return _LATEST_GATE_VALUES


def get_latest_alpha_beta_means() -> tuple[float, float] | None:
    if _LATEST_ALPHA_VALUES is None or _LATEST_BETA_VALUES is None:
        return None
    return float(_LATEST_ALPHA_VALUES.mean().item()), float(_LATEST_BETA_VALUES.mean().item())


def get_latest_alpha_beta_values() -> tuple[torch.Tensor, torch.Tensor] | None:
    if _LATEST_ALPHA_VALUES is None or _LATEST_BETA_VALUES is None:
        return None
    return _LATEST_ALPHA_VALUES, _LATEST_BETA_VALUES


# Backward-compatible aliases for existing alpha-named policies/call-sites.
def set_latest_alpha(alpha: torch.Tensor | None) -> None:
    global _LATEST_ALPHA_VALUES, _LATEST_BETA_VALUES
    _LATEST_ALPHA_VALUES = None
    _LATEST_BETA_VALUES = None
    set_latest_gate(alpha)


def get_latest_alpha_mean() -> float | None:
    return get_latest_gate_mean()


def get_latest_alpha_values() -> torch.Tensor | None:
    return get_latest_gate_values()
