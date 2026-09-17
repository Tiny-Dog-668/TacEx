"""Offline episode accounting for four-tactile Student rollouts.

The collector keeps this module free of Isaac Sim dependencies so that its
episode boundaries and aggregate denominators can be tested without launching
the simulator.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


CONTACT_COUNT = 4


def _as_contact_matrix(value: np.ndarray, *, name: str, num_envs: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (num_envs, CONTACT_COUNT):
        raise ValueError(
            f"{name} must have shape [{num_envs},{CONTACT_COUNT}], got {array.shape}"
        )
    return array


def _as_env_mask(value: np.ndarray, *, name: str, num_envs: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.bool_)
    if array.shape != (num_envs,):
        raise ValueError(f"{name} must have shape [{num_envs}], got {array.shape}")
    return array


def _fraction(numerator: np.ndarray | float, denominator: np.ndarray | float) -> Any:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.divide(
        numerator_array,
        denominator_array,
        out=np.full_like(numerator_array, np.nan, dtype=np.float64),
        where=denominator_array != 0.0,
    )
    if result.ndim == 0:
        return None if np.isnan(result.item()) else float(result.item())
    return [None if np.isnan(item) else float(item) for item in result.tolist()]


class FourTactileEpisodeAccumulator:
    """Accumulate post-transition contact truth and pre-action predictions.

    Truth is sampled after ``env.step`` from the environment's saved contact
    state so terminal transitions are retained despite Isaac Lab auto-reset.
    Predictions and their target are sampled from the observation used to
    choose that action, and therefore are intentionally a pre-action metric.
    """

    def __init__(self, num_envs: int, *, prediction_available: bool = True) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = int(num_envs)
        self.prediction_available = bool(prediction_available)
        self._episode_index = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_length = np.zeros(self.num_envs, dtype=np.int64)
        self._truth_contact_steps = np.zeros((self.num_envs, CONTACT_COUNT), dtype=np.int64)
        self._truth_contact_seen = np.zeros((self.num_envs, CONTACT_COUNT), dtype=np.bool_)
        self._success_ever = np.zeros(self.num_envs, dtype=np.bool_)
        self._first_success_step = np.full(self.num_envs, -1, dtype=np.int64)
        self._pre_success_contact_steps = np.zeros(
            (self.num_envs, CONTACT_COUNT), dtype=np.int64
        )
        self._prediction_probability_sum = np.zeros(
            (self.num_envs, CONTACT_COUNT), dtype=np.float64
        )
        self._prediction_positive_steps = np.zeros(
            (self.num_envs, CONTACT_COUNT), dtype=np.int64
        )
        self._prediction_target_steps = np.zeros(
            (self.num_envs, CONTACT_COUNT), dtype=np.int64
        )
        self._prediction_tp = np.zeros((self.num_envs, CONTACT_COUNT), dtype=np.int64)
        self._prediction_fp = np.zeros((self.num_envs, CONTACT_COUNT), dtype=np.int64)
        self._prediction_fn = np.zeros((self.num_envs, CONTACT_COUNT), dtype=np.int64)
        self._prediction_tn = np.zeros((self.num_envs, CONTACT_COUNT), dtype=np.int64)

        self.rollout_transition_count = 0
        self.rollout_truth_contact_steps = np.zeros(CONTACT_COUNT, dtype=np.int64)
        self.rollout_probability_sum = np.zeros(CONTACT_COUNT, dtype=np.float64)
        self.rollout_prediction_positive_steps = np.zeros(CONTACT_COUNT, dtype=np.int64)
        self.rollout_prediction_target_steps = np.zeros(CONTACT_COUNT, dtype=np.int64)
        self.rollout_prediction_tp = np.zeros(CONTACT_COUNT, dtype=np.int64)
        self.rollout_prediction_fp = np.zeros(CONTACT_COUNT, dtype=np.int64)
        self.rollout_prediction_fn = np.zeros(CONTACT_COUNT, dtype=np.int64)
        self.rollout_prediction_tn = np.zeros(CONTACT_COUNT, dtype=np.int64)

    def record_transition(
        self,
        *,
        post_transition_contact_state: np.ndarray,
        prediction_probability: np.ndarray | None,
        prediction_target_state: np.ndarray,
        success_now: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Record one vectorized transition and return completed episode rows."""
        truth = _as_contact_matrix(
            post_transition_contact_state, name="post_transition_contact_state", num_envs=self.num_envs
        ) >= 0.5
        probability = (
            _as_contact_matrix(
                prediction_probability, name="prediction_probability", num_envs=self.num_envs
            )
            if self.prediction_available and prediction_probability is not None
            else None
        )
        if self.prediction_available and probability is None:
            raise ValueError("prediction_probability is required when prediction_available=True")
        target = _as_contact_matrix(
            prediction_target_state, name="prediction_target_state", num_envs=self.num_envs
        ) >= 0.5
        success = _as_env_mask(success_now, name="success_now", num_envs=self.num_envs)
        terminated_mask = _as_env_mask(terminated, name="terminated", num_envs=self.num_envs)
        truncated_mask = _as_env_mask(truncated, name="truncated", num_envs=self.num_envs)

        predicted = probability >= 0.5 if probability is not None else None
        before_first_success = ~self._success_ever
        self._episode_length += 1
        self._truth_contact_steps += truth
        self._truth_contact_seen |= truth
        self._pre_success_contact_steps += truth * before_first_success[:, None]
        first_success = before_first_success & success
        self._success_ever |= success
        self._first_success_step[first_success] = self._episode_length[first_success]

        if probability is not None and predicted is not None:
            self._prediction_probability_sum += probability
            self._prediction_positive_steps += predicted
            self._prediction_target_steps += target
            self._prediction_tp += predicted & target
            self._prediction_fp += predicted & ~target
            self._prediction_fn += ~predicted & target
            self._prediction_tn += ~predicted & ~target

        self.rollout_transition_count += self.num_envs
        self.rollout_truth_contact_steps += truth.sum(axis=0)
        if probability is not None and predicted is not None:
            self.rollout_probability_sum += probability.sum(axis=0)
            self.rollout_prediction_positive_steps += predicted.sum(axis=0)
            self.rollout_prediction_target_steps += target.sum(axis=0)
            self.rollout_prediction_tp += (predicted & target).sum(axis=0)
            self.rollout_prediction_fp += (predicted & ~target).sum(axis=0)
            self.rollout_prediction_fn += (~predicted & target).sum(axis=0)
            self.rollout_prediction_tn += (~predicted & ~target).sum(axis=0)

        completed = terminated_mask | truncated_mask
        rows = [self._episode_row(env_id, terminated_mask[env_id], truncated_mask[env_id])
                for env_id in np.flatnonzero(completed)]
        if rows:
            self._reset(np.flatnonzero(completed))
        return rows

    def rollout_snapshot(self) -> dict[str, Any]:
        """Return all-transition statistics, including incomplete episodes."""
        count = self.rollout_transition_count
        return {
            "transition_count": int(count),
            "truth_contact_fraction": _fraction(self.rollout_truth_contact_steps, count),
            "prediction": _prediction_summary(
                probability_sum=self.rollout_probability_sum,
                predicted_positive=self.rollout_prediction_positive_steps,
                target_positive=self.rollout_prediction_target_steps,
                true_positive=self.rollout_prediction_tp,
                false_positive=self.rollout_prediction_fp,
                false_negative=self.rollout_prediction_fn,
                true_negative=self.rollout_prediction_tn,
                count=count,
            ) if self.prediction_available else {"available": False},
        }

    def incomplete_episode_count(self) -> int:
        return int(np.count_nonzero(self._episode_length))

    def _episode_row(self, env_id: int, terminated: bool, truncated: bool) -> dict[str, Any]:
        length = int(self._episode_length[env_id])
        success = bool(self._success_ever[env_id])
        first_success_step = int(self._first_success_step[env_id])
        if terminated and truncated:
            termination_reason = "terminated_and_timeout"
        elif terminated:
            termination_reason = "terminated"
        elif truncated:
            termination_reason = "timeout"
        else:
            raise RuntimeError("Episode row requested for a non-completed environment")

        row: dict[str, Any] = {
            "env_id": int(env_id),
            "env_episode_index": int(self._episode_index[env_id]),
            "length_steps": length,
            "termination_reason": termination_reason,
            "success_ever": success,
            "first_success_step": first_success_step if success else None,
        }
        for index in range(CONTACT_COUNT):
            suffix = str(index)
            row[f"truth_contact_steps_{suffix}"] = int(self._truth_contact_steps[env_id, index])
            row[f"truth_contact_fraction_{suffix}"] = float(self._truth_contact_steps[env_id, index] / length)
            row[f"truth_contact_seen_{suffix}"] = bool(self._truth_contact_seen[env_id, index])
            row[f"prediction_mean_probability_{suffix}"] = (
                float(self._prediction_probability_sum[env_id, index] / length)
                if self.prediction_available else None
            )
            row[f"prediction_positive_fraction_{suffix}"] = (
                float(self._prediction_positive_steps[env_id, index] / length)
                if self.prediction_available else None
            )
            row[f"prediction_target_fraction_{suffix}"] = (
                float(self._prediction_target_steps[env_id, index] / length)
                if self.prediction_available else None
            )
            row[f"prediction_true_positive_{suffix}"] = (
                int(self._prediction_tp[env_id, index]) if self.prediction_available else None
            )
            row[f"prediction_false_positive_{suffix}"] = (
                int(self._prediction_fp[env_id, index]) if self.prediction_available else None
            )
            row[f"prediction_false_negative_{suffix}"] = (
                int(self._prediction_fn[env_id, index]) if self.prediction_available else None
            )
            row[f"prediction_true_negative_{suffix}"] = (
                int(self._prediction_tn[env_id, index]) if self.prediction_available else None
            )
            if success:
                row[f"pre_first_success_contact_steps_{suffix}"] = int(
                    self._pre_success_contact_steps[env_id, index]
                )
                row[f"pre_first_success_contact_fraction_{suffix}"] = float(
                    self._pre_success_contact_steps[env_id, index] / first_success_step
                )
            else:
                row[f"pre_first_success_contact_steps_{suffix}"] = None
                row[f"pre_first_success_contact_fraction_{suffix}"] = None
        return row

    def _reset(self, env_ids: np.ndarray) -> None:
        self._episode_index[env_ids] += 1
        self._episode_length[env_ids] = 0
        self._truth_contact_steps[env_ids] = 0
        self._truth_contact_seen[env_ids] = False
        self._success_ever[env_ids] = False
        self._first_success_step[env_ids] = -1
        self._pre_success_contact_steps[env_ids] = 0
        self._prediction_probability_sum[env_ids] = 0.0
        self._prediction_positive_steps[env_ids] = 0
        self._prediction_target_steps[env_ids] = 0
        self._prediction_tp[env_ids] = 0
        self._prediction_fp[env_ids] = 0
        self._prediction_fn[env_ids] = 0
        self._prediction_tn[env_ids] = 0


def summarize_completed_episodes(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize CSV-shaped completed episode rows using explicit macro/micro denominators."""
    rows = list(rows)
    summary: dict[str, Any] = {
        "completed_episode_count": len(rows),
        "success_episode_count": sum(bool(row["success_ever"]) for row in rows),
        "all_completed_episodes": _episode_contact_summary(rows, prefix="truth_contact"),
    }
    successful_rows = [row for row in rows if bool(row["success_ever"])]
    summary["successful_episodes"] = {
        "count": len(successful_rows),
        "full_episode": _episode_contact_summary(successful_rows, prefix="truth_contact"),
        "pre_first_success": _episode_contact_summary(
            successful_rows, prefix="pre_first_success_contact"
        ),
    }
    return summary


def _episode_contact_summary(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    if not rows:
        return {
            "macro_contact_fraction": [None] * CONTACT_COUNT,
            "micro_contact_fraction": [None] * CONTACT_COUNT,
            "episode_contact_seen_fraction": [None] * CONTACT_COUNT,
        }
    fractions = np.asarray(
        [[row[f"{prefix}_fraction_{index}"] for index in range(CONTACT_COUNT)] for row in rows],
        dtype=np.float64,
    )
    contact_steps = np.asarray(
        [[row[f"{prefix}_steps_{index}"] for index in range(CONTACT_COUNT)] for row in rows],
        dtype=np.float64,
    )
    if prefix == "truth_contact":
        denominators = np.asarray([row["length_steps"] for row in rows], dtype=np.float64)
        seen = np.asarray(
            [[row[f"truth_contact_seen_{index}"] for index in range(CONTACT_COUNT)] for row in rows],
            dtype=np.float64,
        )
    else:
        denominators = np.asarray([row["first_success_step"] for row in rows], dtype=np.float64)
        seen = contact_steps > 0.0
    return {
        "macro_contact_fraction": [float(value) for value in fractions.mean(axis=0)],
        "micro_contact_fraction": _fraction(contact_steps.sum(axis=0), denominators.sum()),
        "episode_contact_seen_fraction": [float(value) for value in seen.mean(axis=0)],
    }


def _prediction_summary(
    *,
    probability_sum: np.ndarray,
    predicted_positive: np.ndarray,
    target_positive: np.ndarray,
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    false_negative: np.ndarray,
    true_negative: np.ndarray,
    count: int,
) -> dict[str, Any]:
    correct = true_positive + true_negative
    return {
        "available": True,
        "threshold": 0.5,
        "mean_probability": _fraction(probability_sum, count),
        "positive_fraction": _fraction(predicted_positive, count),
        "target_fraction": _fraction(target_positive, count),
        "accuracy": _fraction(correct, count),
        "precision": _fraction(true_positive, true_positive + false_positive),
        "recall": _fraction(true_positive, true_positive + false_negative),
    }


__all__ = ("CONTACT_COUNT", "FourTactileEpisodeAccumulator", "summarize_completed_episodes")
