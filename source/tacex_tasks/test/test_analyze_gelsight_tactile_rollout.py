"""Offline tests for the GelSight tactile rollout analyzer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _analysis_module():
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts/reinforcement_learning/skrl/rma_gelsight_x040_three_frame_student"
        / "analyze_tactile_rollout.py"
    )
    spec = importlib.util.spec_from_file_location("gelsight_tactile_rollout_analysis", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_separates_full_surface_contact():
    analysis = _analysis_module()
    metrics = {
        "contact_state": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        "bilateral_contact": np.array([False, False, True, True]),
        "rgb_mae_0_1": np.array([0.01, 0.02, 0.20, 0.30], dtype=np.float32),
        "rgb_max_abs_0_1": np.array([0.02, 0.03, 0.40, 0.50], dtype=np.float32),
        "rgb_changed_fraction_gt_1_u8": np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
        "rgb_changed_fraction_gt_5_u8": np.array([0.0, 0.1, 0.7, 0.8], dtype=np.float32),
        "contact_force_n": np.array([0.0, 0.1, 2.0, 3.0], dtype=np.float32),
        "contact_probability": np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32),
        "indentation_mm": np.array([0.0, 0.0, 2.0, 3.0], dtype=np.float32),
        "taxim_contact_fraction": np.array([0.0, 0.0, 0.95, 0.50], dtype=np.float32),
    }
    summary = analysis.summarize_rollout_metrics(metrics, full_contact_fraction=0.90)
    assert summary["groups"]["no_contact"]["count"] == 2
    assert summary["groups"]["contact"]["count"] == 2
    assert summary["groups"]["bilateral_contact"]["count"] == 2
    assert summary["groups"]["full_surface_contact"]["count"] == 1
    assert summary["comparison"]["mean_rgb_mae_difference_0_1"] == pytest.approx(0.185)


def test_taxim_contact_fraction_matches_shifted_height_mask():
    analysis = _analysis_module()
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            output={
                "height_map": torch.tensor(
                    [
                        [[34.0, 34.0], [34.0, 34.0]],
                        [[24.0, 24.0], [24.0, 24.0]],
                        [[24.0, 25.0], [26.0, 27.0]],
                    ]
                )
            }
        ),
        optical_simulator=SimpleNamespace(
            _indentation_depth=torch.tensor([0.0, 1.0, 1.5])
        ),
    )
    fraction, indentation = analysis._taxim_contact_fraction(sensor)
    torch.testing.assert_close(fraction, torch.tensor([0.0, 1.0, 0.5]))
    torch.testing.assert_close(indentation, torch.tensor([0.0, 1.0, 1.5]))
