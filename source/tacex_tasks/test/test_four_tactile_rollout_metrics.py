"""Offline tests for four-tactile rollout episode accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _metrics_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "tacex_tasks/sim2real_gelsight_rma/four_tactile_rollout_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("four_tactile_rollout_metrics", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = _metrics_module()
FourTactileEpisodeAccumulator = metrics.FourTactileEpisodeAccumulator
summarize_completed_episodes = metrics.summarize_completed_episodes


def _matrix(*rows: list[float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64)


def test_terminal_success_retains_post_transition_contact_and_pre_success_window() -> None:
    accumulator = FourTactileEpisodeAccumulator(num_envs=2)
    rows = accumulator.record_transition(
        post_transition_contact_state=_matrix([1, 0, 0, 0], [0, 1, 0, 0]),
        prediction_probability=_matrix([0.9, 0.1, 0.1, 0.1], [0.1, 0.9, 0.1, 0.1]),
        prediction_target_state=_matrix([1, 0, 0, 0], [0, 1, 0, 0]),
        success_now=np.asarray([False, True]),
        terminated=np.asarray([False, True]),
        truncated=np.asarray([False, False]),
    )

    assert len(rows) == 1
    success_row = rows[0]
    assert success_row["env_id"] == 1
    assert success_row["termination_reason"] == "terminated"
    assert success_row["success_ever"] is True
    assert success_row["first_success_step"] == 1
    assert success_row["truth_contact_steps_1"] == 1
    assert success_row["pre_first_success_contact_steps_1"] == 1
    assert success_row["pre_first_success_contact_fraction_1"] == pytest.approx(1.0)

    rows.extend(
        accumulator.record_transition(
            post_transition_contact_state=_matrix([1, 1, 1, 1], [0, 0, 0, 0]),
            prediction_probability=_matrix([0.9, 0.9, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]),
            prediction_target_state=_matrix([1, 1, 1, 1], [0, 0, 0, 0]),
            success_now=np.asarray([False, False]),
            terminated=np.asarray([False, False]),
            truncated=np.asarray([True, True]),
        )
    )
    assert {row["termination_reason"] for row in rows} == {"terminated", "timeout"}
    failed_row = next(row for row in rows if row["env_id"] == 0)
    assert failed_row["success_ever"] is False
    assert failed_row["pre_first_success_contact_steps_0"] is None
    assert failed_row["truth_contact_fraction_0"] == pytest.approx(1.0)


def test_summary_uses_explicit_macro_and_micro_denominators() -> None:
    accumulator = FourTactileEpisodeAccumulator(num_envs=2)
    rows = accumulator.record_transition(
        post_transition_contact_state=_matrix([1, 0, 0, 0], [0, 0, 0, 0]),
        prediction_probability=_matrix([0.8, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2]),
        prediction_target_state=_matrix([1, 0, 0, 0], [0, 0, 0, 0]),
        success_now=np.asarray([False, False]),
        terminated=np.asarray([False, True]),
        truncated=np.asarray([False, False]),
    )
    rows.extend(
        accumulator.record_transition(
            post_transition_contact_state=_matrix([1, 0, 0, 0], [1, 1, 1, 1]),
            prediction_probability=_matrix([0.8, 0.2, 0.2, 0.2], [0.8, 0.8, 0.8, 0.8]),
            prediction_target_state=_matrix([1, 0, 0, 0], [1, 1, 1, 1]),
            success_now=np.asarray([True, False]),
            terminated=np.asarray([True, False]),
            truncated=np.asarray([False, False]),
        )
    )

    summary = summarize_completed_episodes(rows)
    assert summary["completed_episode_count"] == 2
    assert summary["success_episode_count"] == 1
    all_episodes = summary["all_completed_episodes"]
    # Sensor 0 episode fractions are 1.0 (2/2) and 0.0 (0/1): macro=0.5, micro=2/3.
    assert all_episodes["macro_contact_fraction"][0] == pytest.approx(0.5)
    assert all_episodes["micro_contact_fraction"][0] == pytest.approx(2.0 / 3.0)
    successful = summary["successful_episodes"]
    assert successful["full_episode"]["micro_contact_fraction"][0] == pytest.approx(1.0)
    assert successful["pre_first_success"]["micro_contact_fraction"][0] == pytest.approx(1.0)


def test_rollout_snapshot_handles_all_zero_and_all_one_contact_states() -> None:
    accumulator = FourTactileEpisodeAccumulator(num_envs=2)
    accumulator.record_transition(
        post_transition_contact_state=_matrix([0, 0, 0, 0], [1, 1, 1, 1]),
        prediction_probability=_matrix([0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.9, 0.9]),
        prediction_target_state=_matrix([0, 0, 0, 0], [1, 1, 1, 1]),
        success_now=np.asarray([False, False]),
        terminated=np.asarray([False, False]),
        truncated=np.asarray([False, False]),
    )
    snapshot = accumulator.rollout_snapshot()
    assert snapshot["truth_contact_fraction"] == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert snapshot["prediction"]["accuracy"] == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert accumulator.incomplete_episode_count() == 2


def test_vision_only_rollout_marks_contact_prediction_unavailable() -> None:
    accumulator = FourTactileEpisodeAccumulator(
        num_envs=1, prediction_available=False
    )
    rows = accumulator.record_transition(
        post_transition_contact_state=_matrix([1, 0, 1, 0]),
        prediction_probability=None,
        prediction_target_state=_matrix([1, 0, 1, 0]),
        success_now=np.asarray([True]),
        terminated=np.asarray([True]),
        truncated=np.asarray([False]),
    )
    assert rows[0]["prediction_mean_probability_0"] is None
    assert accumulator.rollout_snapshot()["prediction"] == {"available": False}
