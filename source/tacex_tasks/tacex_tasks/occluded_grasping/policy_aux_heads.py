"""Shared helpers for logging latest auxiliary-head predictions."""

from __future__ import annotations

import torch

_LATEST_RELIABILITY_PRED: torch.Tensor | None = None
_LATEST_STAGE_LOGITS: torch.Tensor | None = None
_LATEST_W_PROBE: torch.Tensor | None = None
_LATEST_W_GRASP: torch.Tensor | None = None
_LATEST_ALPHA: torch.Tensor | None = None
_LATEST_U_PROBE: torch.Tensor | None = None
_LATEST_U_GRASP: torch.Tensor | None = None
_LATEST_LOGIT_PROBE: torch.Tensor | None = None
_LATEST_LOGIT_GRASP: torch.Tensor | None = None
_LATEST_B_PROBE: torch.Tensor | None = None
_LATEST_B_GRASP: torch.Tensor | None = None
_LATEST_M_PROBE: torch.Tensor | None = None
_LATEST_M_GRASP: torch.Tensor | None = None


def set_latest_aux_predictions(
    reliability_pred: torch.Tensor | None,
    stage_logits: torch.Tensor | None,
    w_probe: torch.Tensor | None = None,
    w_grasp: torch.Tensor | None = None,
    alpha: torch.Tensor | None = None,
    u_probe: torch.Tensor | None = None,
    u_grasp: torch.Tensor | None = None,
    logit_probe: torch.Tensor | None = None,
    logit_grasp: torch.Tensor | None = None,
    b_probe: torch.Tensor | None = None,
    b_grasp: torch.Tensor | None = None,
    m_probe: torch.Tensor | None = None,
    m_grasp: torch.Tensor | None = None,
) -> None:
    """Store latest per-sample auxiliary predictions for env-side logging/debugging."""
    global _LATEST_RELIABILITY_PRED, _LATEST_STAGE_LOGITS, _LATEST_W_PROBE, _LATEST_W_GRASP, _LATEST_ALPHA
    global _LATEST_U_PROBE, _LATEST_U_GRASP, _LATEST_LOGIT_PROBE, _LATEST_LOGIT_GRASP
    global _LATEST_B_PROBE, _LATEST_B_GRASP, _LATEST_M_PROBE, _LATEST_M_GRASP

    if reliability_pred is None or stage_logits is None:
        _LATEST_RELIABILITY_PRED = None
        _LATEST_STAGE_LOGITS = None
        _LATEST_W_PROBE = None
        _LATEST_W_GRASP = None
        _LATEST_ALPHA = None
        _LATEST_U_PROBE = None
        _LATEST_U_GRASP = None
        _LATEST_LOGIT_PROBE = None
        _LATEST_LOGIT_GRASP = None
        _LATEST_B_PROBE = None
        _LATEST_B_GRASP = None
        _LATEST_M_PROBE = None
        _LATEST_M_GRASP = None
        return

    rv = reliability_pred.detach().to(dtype=torch.float32).view(-1).cpu()
    logits = stage_logits.detach().to(dtype=torch.float32).cpu()
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)

    def _to_1d_cpu(x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        return x.detach().to(dtype=torch.float32).view(-1).cpu()

    _LATEST_RELIABILITY_PRED = rv
    _LATEST_STAGE_LOGITS = logits
    _LATEST_W_PROBE = _to_1d_cpu(w_probe)
    _LATEST_W_GRASP = _to_1d_cpu(w_grasp)
    _LATEST_ALPHA = _to_1d_cpu(alpha)
    _LATEST_U_PROBE = _to_1d_cpu(u_probe)
    _LATEST_U_GRASP = _to_1d_cpu(u_grasp)
    _LATEST_LOGIT_PROBE = _to_1d_cpu(logit_probe)
    _LATEST_LOGIT_GRASP = _to_1d_cpu(logit_grasp)
    _LATEST_B_PROBE = _to_1d_cpu(b_probe)
    _LATEST_B_GRASP = _to_1d_cpu(b_grasp)
    _LATEST_M_PROBE = _to_1d_cpu(m_probe)
    _LATEST_M_GRASP = _to_1d_cpu(m_grasp)


def get_latest_reliability_pred() -> torch.Tensor | None:
    return _LATEST_RELIABILITY_PRED


def get_latest_stage_logits() -> torch.Tensor | None:
    return _LATEST_STAGE_LOGITS


def get_latest_w_probe() -> torch.Tensor | None:
    return _LATEST_W_PROBE


def get_latest_w_grasp() -> torch.Tensor | None:
    return _LATEST_W_GRASP


def get_latest_alpha() -> torch.Tensor | None:
    return _LATEST_ALPHA


def get_latest_u_probe() -> torch.Tensor | None:
    return _LATEST_U_PROBE


def get_latest_u_grasp() -> torch.Tensor | None:
    return _LATEST_U_GRASP


def get_latest_logit_probe() -> torch.Tensor | None:
    return _LATEST_LOGIT_PROBE


def get_latest_logit_grasp() -> torch.Tensor | None:
    return _LATEST_LOGIT_GRASP


def get_latest_b_probe() -> torch.Tensor | None:
    return _LATEST_B_PROBE


def get_latest_b_grasp() -> torch.Tensor | None:
    return _LATEST_B_GRASP


def get_latest_m_probe() -> torch.Tensor | None:
    return _LATEST_M_PROBE


def get_latest_m_grasp() -> torch.Tensor | None:
    return _LATEST_M_GRASP
