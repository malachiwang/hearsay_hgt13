"""NSA/ASVspoof5 minimum normalized detection cost for HEARSAY.

The ASVspoof countermeasure metric treats larger scores as more bona fide.
HEARSAY scores have the opposite orientation, so :func:`evaluate_min_dcf`
negates them before applying the ASVspoof5 DET-curve algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


P_SPOOF = 0.30
C_MISS = 1.0
C_FA = 4.0

BONAFIDE_LABEL = "bonafide"
SPOOF_LABEL = "spoof"
VALID_LABELS = frozenset((BONAFIDE_LABEL, SPOOF_LABEL))


class MinDCFError(ValueError):
    """Raised when minDCF inputs or cost parameters are invalid."""


@dataclass(frozen=True)
class MinDCFPoint:
    """The lowest-cost point on a DET curve."""

    min_dcf: float
    threshold: float
    p_miss: float
    p_fa: float
    unnormalized_cost: float


@dataclass(frozen=True)
class MinDCFResult:
    """HEARSAY minDCF result in the supplied synthetic-score orientation."""

    min_dcf: float
    threshold: float
    p_miss: float
    p_fa: float
    unnormalized_cost: float
    number_bonafide: int
    number_spoof: int


def _as_score_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        scores = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MinDCFError(f"{name} must be a one-dimensional numeric array") from exc

    if scores.ndim != 1:
        raise MinDCFError(f"{name} must be one-dimensional, got shape {scores.shape}")
    if scores.size == 0:
        raise MinDCFError(f"{name} must not be empty")
    if not np.isfinite(scores).all():
        raise MinDCFError(f"{name} must contain only finite values")
    return scores


def compute_det_curve(
    target_scores: ArrayLike,
    nontarget_scores: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute ASVspoof5-style Pmiss, Pfa, and threshold arrays.

    Both inputs use the ASVspoof score direction: larger values indicate the
    target (bona fide) class. The stable sort and target-first concatenation
    intentionally match ASVspoof5, including deterministic handling of ties.
    """

    target = _as_score_vector(target_scores, "target_scores")
    nontarget = _as_score_vector(nontarget_scores, "nontarget_scores")

    all_scores = np.concatenate((target, nontarget))
    labels = np.concatenate(
        (np.ones(target.size, dtype=np.int64), np.zeros(nontarget.size, dtype=np.int64))
    )

    sort_indices = np.argsort(all_scores, kind="mergesort")
    sorted_scores = all_scores[sort_indices]
    sorted_labels = labels[sort_indices]

    target_trial_sums = np.cumsum(sorted_labels)
    nontarget_trial_sums = nontarget.size - (
        np.arange(1, all_scores.size + 1) - target_trial_sums
    )

    p_miss = np.concatenate(
        (np.array([0.0]), target_trial_sums.astype(np.float64) / target.size)
    )
    p_fa = np.concatenate(
        (np.array([1.0]), nontarget_trial_sums.astype(np.float64) / nontarget.size)
    )
    thresholds = np.concatenate(
        (np.array([sorted_scores[0] - 0.001]), sorted_scores)
    )
    return p_miss, p_fa, thresholds


def compute_min_dcf(
    p_miss: ArrayLike,
    p_fa: ArrayLike,
    thresholds: ArrayLike,
    p_spoof: float = P_SPOOF,
    c_miss: float = C_MISS,
    c_fa: float = C_FA,
) -> MinDCFPoint:
    """Find the minimum normalized DCF over an ASVspoof-style DET curve."""

    miss = _as_score_vector(p_miss, "p_miss")
    false_alarm = _as_score_vector(p_fa, "p_fa")
    threshold_values = _as_score_vector(thresholds, "thresholds")
    if not (miss.size == false_alarm.size == threshold_values.size):
        raise MinDCFError("p_miss, p_fa, and thresholds must have equal lengths")
    if np.any((miss < 0.0) | (miss > 1.0)):
        raise MinDCFError("p_miss values must be between 0 and 1")
    if np.any((false_alarm < 0.0) | (false_alarm > 1.0)):
        raise MinDCFError("p_fa values must be between 0 and 1")

    if not np.isfinite(p_spoof) or not 0.0 < p_spoof < 1.0:
        raise MinDCFError("p_spoof must be finite and strictly between 0 and 1")
    if not np.isfinite(c_miss) or c_miss <= 0.0:
        raise MinDCFError("c_miss must be finite and greater than 0")
    if not np.isfinite(c_fa) or c_fa <= 0.0:
        raise MinDCFError("c_fa must be finite and greater than 0")

    p_target = 1.0 - p_spoof
    costs = c_miss * miss * p_target + c_fa * false_alarm * (1.0 - p_target)
    normalization = min(c_miss * p_target, c_fa * (1.0 - p_target))

    # np.argmin returns the first minimum, matching ASVspoof5's strict '<' loop.
    best_index = int(np.argmin(costs))
    best_cost = float(costs[best_index])
    return MinDCFPoint(
        min_dcf=best_cost / normalization,
        threshold=float(threshold_values[best_index]),
        p_miss=float(miss[best_index]),
        p_fa=float(false_alarm[best_index]),
        unnormalized_cost=best_cost,
    )


def evaluate_min_dcf(
    synthetic_scores: ArrayLike,
    labels: Sequence[str] | NDArray[np.str_],
    *,
    p_spoof: float = P_SPOOF,
    c_miss: float = C_MISS,
    c_fa: float = C_FA,
) -> MinDCFResult:
    """Evaluate higher-is-more-synthetic HEARSAY scores.

    Labels must be exactly ``"bonafide"`` or ``"spoof"``. Scores may be any
    finite real values; they need not be probabilities or lie in ``[0, 1]``.
    The returned threshold is converted back to the original synthetic-score
    orientation.
    """

    scores = _as_score_vector(synthetic_scores, "synthetic_scores")
    label_values = np.asarray(labels, dtype=object)
    if label_values.ndim != 1:
        raise MinDCFError(
            f"labels must be one-dimensional, got shape {label_values.shape}"
        )
    if label_values.size == 0:
        raise MinDCFError("labels must not be empty")
    if label_values.size != scores.size:
        raise MinDCFError(
            "synthetic_scores and labels must have equal lengths "
            f"({scores.size} != {label_values.size})"
        )

    invalid_types = [label for label in label_values if not isinstance(label, str)]
    if invalid_types:
        raise MinDCFError("labels must contain strings")
    unknown_labels = sorted(set(label_values.tolist()) - VALID_LABELS)
    if unknown_labels:
        raise MinDCFError(
            "unsupported labels: "
            + ", ".join(repr(label) for label in unknown_labels)
            + f"; expected {sorted(VALID_LABELS)}"
        )

    bonafide_mask = label_values == BONAFIDE_LABEL
    spoof_mask = label_values == SPOOF_LABEL
    number_bonafide = int(np.count_nonzero(bonafide_mask))
    number_spoof = int(np.count_nonzero(spoof_mask))
    if number_bonafide == 0:
        raise MinDCFError("at least one bona fide sample is required")
    if number_spoof == 0:
        raise MinDCFError("at least one spoof sample is required")

    bona_fide_scores = -scores
    p_miss, p_fa, thresholds = compute_det_curve(
        bona_fide_scores[bonafide_mask], bona_fide_scores[spoof_mask]
    )
    point = compute_min_dcf(
        p_miss,
        p_fa,
        thresholds,
        p_spoof=p_spoof,
        c_miss=c_miss,
        c_fa=c_fa,
    )

    return MinDCFResult(
        min_dcf=point.min_dcf,
        threshold=-point.threshold,
        p_miss=point.p_miss,
        p_fa=point.p_fa,
        unnormalized_cost=point.unnormalized_cost,
        number_bonafide=number_bonafide,
        number_spoof=number_spoof,
    )
