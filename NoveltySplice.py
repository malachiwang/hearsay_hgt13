"""Detect persistent local changes in precomputed forensic embeddings.

This module complements :mod:`DetectSplices`: the physical detector looks for
waveform and spectral seams, while this detector looks for a change in the
short-window representation on either side of a candidate boundary. It does
not load audio, run WavLM, download models, or train a classifier.

The default contexts are fixed baselines, not validation-tuned parameters.
With a 0.20--0.25 second embedding hop, two and four windows per side provide
short and medium local contexts for inserted words and short phrases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from scipy.signal import find_peaks


DEFAULT_CONTEXT_SCALES: Final[tuple[int, int]] = (2, 4)
DEFAULT_EMBEDDING_WINDOW_SECONDS: Final[float] = 0.8
DEFAULT_EMBEDDING_HOP_SECONDS: Final[float] = 0.2
DEFAULT_PEAK_HEIGHT: Final[float] = 0.20
DEFAULT_PEAK_PROMINENCE: Final[float] = 0.05
DEFAULT_PEAK_DISTANCE_WINDOWS: Final[int] = 2


class NoveltySpliceError(ValueError):
    """Raised when novelty-splice inputs or outputs violate their contract."""


@dataclass(frozen=True)
class NoveltySpliceCandidate:
    """A local representation-change candidate."""

    time_sec: float
    score: float
    scale_windows: int


@dataclass(frozen=True)
class NoveltySpliceResult:
    """Detailed novelty curves plus finite whole-clip summary values."""

    candidates: list[NoveltySpliceCandidate] = field(default_factory=list)
    max_novelty: float = 0.0
    mean_top3_novelty: float = 0.0
    novelty_p95: float = 0.0
    strongest_time_sec: float | None = None
    novelty_curve: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    frame_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    boundary_times: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    scale_curves: dict[int, np.ndarray] = field(default_factory=dict)
    local_fake_scores: np.ndarray | None = None
    note: str = ""

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(embeddings)
    except (TypeError, ValueError) as exc:
        raise NoveltySpliceError("embeddings must be a numeric 2-D array") from exc
    if values.ndim != 2:
        raise NoveltySpliceError(
            f"embeddings must have shape (n_windows, embedding_dim); got {values.shape}"
        )
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise NoveltySpliceError("embeddings must not be empty")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise NoveltySpliceError("embeddings must contain real numeric values")

    values = np.ascontiguousarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise NoveltySpliceError("embeddings contain NaN or Inf")
    row_scales = np.max(np.abs(values), axis=1)
    zero_rows = np.flatnonzero(row_scales == 0.0)
    if zero_rows.size:
        raise NoveltySpliceError(
            "embeddings contain zero-norm rows at indices "
            + ", ".join(str(index) for index in zero_rows[:5])
        )
    scaled = values / row_scales[:, np.newaxis]
    norms = np.linalg.norm(scaled, axis=1)
    if not np.isfinite(norms).all():
        raise NoveltySpliceError("embedding norms are not finite")
    normalized = scaled / norms[:, np.newaxis]
    if not np.isfinite(normalized).all():
        raise NoveltySpliceError("embedding normalization produced NaN or Inf")
    return normalized


def _validate_frame_times(frame_times: np.ndarray, n_windows: int) -> np.ndarray:
    try:
        times = np.asarray(frame_times)
    except (TypeError, ValueError) as exc:
        raise NoveltySpliceError("frame_times must be a numeric 1-D array") from exc
    if times.ndim != 1 or times.shape[0] != n_windows:
        raise NoveltySpliceError(
            "frame_times must have shape (n_windows,); "
            f"got {times.shape} for {n_windows} windows"
        )
    if not np.issubdtype(times.dtype, np.number) or np.issubdtype(
        times.dtype, np.complexfloating
    ):
        raise NoveltySpliceError("frame_times must contain real numeric values")

    times = np.ascontiguousarray(times, dtype=np.float64)
    if not np.isfinite(times).all():
        raise NoveltySpliceError("frame_times contain NaN or Inf")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise NoveltySpliceError("frame_times must be strictly increasing")
    return times


def _validate_scales(scales: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(scales, (str, bytes)):
        raise NoveltySpliceError("context_scales must contain positive integers")
    try:
        scale_values = tuple(scales)
    except TypeError as exc:
        raise NoveltySpliceError(
            "context_scales must contain positive integers"
        ) from exc
    if not scale_values or len(scale_values) > 3:
        raise NoveltySpliceError("context_scales must contain between 1 and 3 scales")
    if any(
        isinstance(scale, (bool, np.bool_))
        or not isinstance(scale, (int, np.integer))
        or int(scale) <= 0
        for scale in scale_values
    ):
        raise NoveltySpliceError("context_scales must contain positive integers")
    normalized = tuple(sorted({int(scale) for scale in scale_values}))
    if len(normalized) != len(scale_values):
        raise NoveltySpliceError("context_scales must not contain duplicates")
    return normalized


def _validate_local_fake_scores(
    local_fake_scores: np.ndarray | None,
    n_windows: int,
) -> np.ndarray | None:
    if local_fake_scores is None:
        return None
    try:
        scores = np.asarray(local_fake_scores)
    except (TypeError, ValueError) as exc:
        raise NoveltySpliceError(
            "local_fake_scores must be a numeric 1-D array"
        ) from exc
    if scores.ndim != 1 or scores.shape[0] != n_windows:
        raise NoveltySpliceError(
            "local_fake_scores must have shape (n_windows,); "
            f"got {scores.shape} for {n_windows} windows"
        )
    if not np.issubdtype(scores.dtype, np.number) or np.issubdtype(
        scores.dtype, np.complexfloating
    ):
        raise NoveltySpliceError("local_fake_scores must contain real numeric values")
    scores = np.ascontiguousarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise NoveltySpliceError("local_fake_scores contain NaN or Inf")
    return scores


def cosine_self_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Return a finite cosine self-similarity matrix for valid embeddings."""

    normalized = _normalize_embeddings(embeddings)
    similarity = normalized @ normalized.T
    similarity = np.clip(similarity, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity


def _checkerboard_kernel(scale: int) -> np.ndarray:
    """Return a Gaussian-tapered, independently balanced Foote kernel."""

    offsets = np.arange(-scale, scale, dtype=np.float64) + 0.5
    sigma = max(scale / 2.0, 0.5)
    gaussian = np.exp(
        -(offsets[:, np.newaxis] ** 2 + offsets[np.newaxis, :] ** 2)
        / (2.0 * sigma**2)
    )
    sides = np.concatenate((-np.ones(scale), np.ones(scale)))
    kernel = gaussian * np.outer(sides, sides)
    positive = kernel > 0.0
    negative = kernel < 0.0
    kernel[positive] /= float(np.sum(kernel[positive]))
    kernel[negative] /= float(-np.sum(kernel[negative]))
    return kernel


def _novelty_curve_for_scale(
    similarity: np.ndarray,
    scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_windows = similarity.shape[0]
    curve = np.zeros(n_windows, dtype=np.float64)
    valid = np.zeros(n_windows, dtype=bool)
    if n_windows < 2 * scale:
        return curve, valid

    kernel = _checkerboard_kernel(scale)
    for boundary in range(scale, n_windows - scale + 1):
        local = similarity[
            boundary - scale : boundary + scale,
            boundary - scale : boundary + scale,
        ]
        curve[boundary] = max(0.0, float(np.sum(local * kernel)))
        valid[boundary] = True
    return curve, valid


def _boundary_times(frame_times: np.ndarray) -> np.ndarray:
    times = frame_times.copy()
    if times.size > 1:
        times[1:] = 0.5 * (frame_times[:-1] + frame_times[1:])
    return times


def _validate_peak_parameters(
    peak_height: float,
    peak_prominence: float,
    peak_distance_windows: int,
) -> tuple[float, float, int]:
    try:
        height = float(peak_height)
        prominence = float(peak_prominence)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NoveltySpliceError("peak thresholds must be finite and nonnegative") from exc
    if not np.isfinite(height) or height < 0.0:
        raise NoveltySpliceError("peak_height must be finite and nonnegative")
    if not np.isfinite(prominence) or prominence < 0.0:
        raise NoveltySpliceError("peak_prominence must be finite and nonnegative")
    if (
        isinstance(peak_distance_windows, (bool, np.bool_))
        or not isinstance(peak_distance_windows, (int, np.integer))
        or int(peak_distance_windows) <= 0
    ):
        raise NoveltySpliceError("peak_distance_windows must be a positive integer")
    return height, prominence, int(peak_distance_windows)


def detect_embedding_novelty(
    embeddings: np.ndarray,
    frame_times: np.ndarray,
    *,
    context_scales: tuple[int, ...] = DEFAULT_CONTEXT_SCALES,
    peak_height: float = DEFAULT_PEAK_HEIGHT,
    peak_prominence: float = DEFAULT_PEAK_PROMINENCE,
    peak_distance_windows: int = DEFAULT_PEAK_DISTANCE_WINDOWS,
    local_fake_scores: np.ndarray | None = None,
) -> NoveltySpliceResult:
    """Compute multiscale Foote novelty from precomputed window embeddings.

    For each boundary and scale, a Gaussian-tapered checkerboard compares
    within-side similarity against cross-boundary similarity. The returned
    peak candidates use fixed diagnostic defaults; continuous novelty values
    and summary statistics are the primary fusion signals.
    """

    normalized = _normalize_embeddings(embeddings)
    times = _validate_frame_times(frame_times, normalized.shape[0])
    scales = _validate_scales(context_scales)
    height, prominence, distance = _validate_peak_parameters(
        peak_height, peak_prominence, peak_distance_windows
    )
    fake_scores = _validate_local_fake_scores(local_fake_scores, normalized.shape[0])

    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    scale_curves: dict[int, np.ndarray] = {}
    scale_validity: dict[int, np.ndarray] = {}
    for scale in scales:
        curve, valid = _novelty_curve_for_scale(similarity, scale)
        scale_curves[scale] = curve
        scale_validity[scale] = valid

    combined = np.maximum.reduce(list(scale_curves.values()))
    valid_any = np.logical_or.reduce(list(scale_validity.values()))
    boundary_times = _boundary_times(times)
    if not valid_any.any():
        return NoveltySpliceResult(
            novelty_curve=combined,
            frame_times=times,
            boundary_times=boundary_times,
            scale_curves=scale_curves,
            local_fake_scores=fake_scores,
            note="not enough embedding windows for both boundary contexts",
        )

    peak_indices, _ = find_peaks(
        combined,
        height=height,
        prominence=prominence,
        distance=distance,
    )
    peak_indices = peak_indices[valid_any[peak_indices]]
    candidates = []
    for index in peak_indices:
        scale = max(scales, key=lambda value: scale_curves[value][index])
        candidates.append(
            NoveltySpliceCandidate(
                time_sec=float(boundary_times[index]),
                score=float(combined[index]),
                scale_windows=scale,
            )
        )

    evidence = combined[valid_any]
    top_values = np.sort(evidence)[-min(3, evidence.size) :]
    strongest_index = int(np.argmax(np.where(valid_any, combined, -np.inf)))
    result = NoveltySpliceResult(
        candidates=candidates,
        max_novelty=float(np.max(evidence)),
        mean_top3_novelty=float(np.mean(top_values)),
        novelty_p95=float(np.percentile(evidence, 95.0)),
        strongest_time_sec=float(boundary_times[strongest_index]),
        novelty_curve=combined,
        frame_times=times,
        boundary_times=boundary_times,
        scale_curves=scale_curves,
        local_fake_scores=fake_scores,
        note="multiscale Gaussian-checkerboard embedding novelty",
    )
    _validate_result(result)
    return result


def _validate_result(result: NoveltySpliceResult) -> None:
    scalars = (result.max_novelty, result.mean_top3_novelty, result.novelty_p95)
    if not all(np.isfinite(value) for value in scalars):
        raise NoveltySpliceError("novelty detector produced nonfinite summary values")
    if not np.isfinite(result.novelty_curve).all():
        raise NoveltySpliceError("novelty detector produced a nonfinite curve")
    if any(
        not np.isfinite(candidate.time_sec) or not np.isfinite(candidate.score)
        for candidate in result.candidates
    ):
        raise NoveltySpliceError("novelty detector produced a nonfinite candidate")


def _local_fake_features(scores: np.ndarray | None) -> dict[str, float]:
    if scores is None:
        return {}
    max_jump = float(np.max(np.abs(np.diff(scores)))) if scores.size > 1 else 0.0
    return {
        "local_wavlm_fake_max": float(np.max(scores)),
        "local_wavlm_fake_min": float(np.min(scores)),
        "local_wavlm_fake_range": float(np.ptp(scores)),
        "local_wavlm_fake_max_jump": max_jump,
        "local_wavlm_fake_p90": float(np.percentile(scores, 90.0)),
    }


def novelty_splice_features(result: NoveltySpliceResult) -> dict[str, float]:
    """Return stable finite scalar features for the generic fusion table."""

    if not isinstance(result, NoveltySpliceResult):
        raise NoveltySpliceError("novelty_splice_features requires a result object")
    features = {
        "splice_novelty_max": float(result.max_novelty),
        "splice_novelty_top3_mean": float(result.mean_top3_novelty),
        "splice_novelty_p95": float(result.novelty_p95),
        "splice_novelty_n_peaks": float(result.n_candidates),
        **_local_fake_features(result.local_fake_scores),
    }
    if not all(np.isfinite(value) for value in features.values()):
        raise NoveltySpliceError("novelty feature output contains NaN or Inf")
    return features
