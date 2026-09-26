"""Whole-clip smoothed cepstral peak prominence (CPPS) extraction."""

from __future__ import annotations

import numpy as np
import parselmouth
from parselmouth.praat import call


EXPECTED_SAMPLE_RATE_HZ = 16_000

# Fixed Praat PowerCepstrogram construction recipe.
CPPS_PITCH_FLOOR_HZ = 60.0  # Three periods define a 50 ms analysis window.
CPPS_TIME_STEP_S = 0.002
CPPS_MAX_FREQUENCY_HZ = 5_000.0
CPPS_PREEMPHASIS_FROM_HZ = 50.0

# Fixed Praat CPPS smoothing, peak-search, and trend-fit recipe.
CPPS_SUBTRACT_TREND_BEFORE_SMOOTHING = False
CPPS_TIME_AVERAGING_WINDOW_S = 0.02
CPPS_QUEFRENCY_AVERAGING_WINDOW_S = 0.0005
CPPS_PEAK_SEARCH_FLOOR_HZ = 60.0
CPPS_PEAK_SEARCH_CEILING_HZ = 330.0
CPPS_TOLERANCE = 0.05
CPPS_INTERPOLATION = "Parabolic"
CPPS_TREND_LINE_MIN_QUEFRENCY_S = 0.001
CPPS_TREND_LINE_MAX_QUEFRENCY_S = 0.0  # Zero means the available endpoint.
CPPS_TREND_TYPE = "Straight"
CPPS_FIT_METHOD = "Robust"

CPPS_MINIMUM_DURATION_S = 3.0 / CPPS_PITCH_FLOOR_HZ


class CPPSExtractionError(RuntimeError):
    """Raised when a valid finite CPPS feature cannot be extracted."""


def _validate_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != EXPECTED_SAMPLE_RATE_HZ:
        raise CPPSExtractionError(
            "CPPS requires 16 kHz audio; "
            f"received {sample_rate} Hz"
        )

    try:
        samples = np.asarray(waveform)
    except (TypeError, ValueError) as exc:
        raise CPPSExtractionError("CPPS waveform must be a numeric array") from exc

    if samples.ndim != 1:
        raise CPPSExtractionError(
            "CPPS requires a mono 1-D waveform; "
            f"received shape {samples.shape}"
        )
    if samples.size == 0:
        raise CPPSExtractionError("CPPS waveform must not be empty")
    if not np.issubdtype(samples.dtype, np.number) or np.issubdtype(
        samples.dtype, np.complexfloating
    ):
        raise CPPSExtractionError("CPPS waveform must contain real numeric samples")

    samples = np.ascontiguousarray(samples, dtype=np.float64)
    if not np.isfinite(samples).all():
        raise CPPSExtractionError("CPPS waveform contains NaN or Inf")
    if np.all(samples == 0.0):
        raise CPPSExtractionError(
            "CPPS cannot be computed for exact digital silence"
        )

    duration_seconds = samples.size / EXPECTED_SAMPLE_RATE_HZ
    if duration_seconds < CPPS_MINIMUM_DURATION_S:
        raise CPPSExtractionError(
            f"CPPS input is too short: {duration_seconds:.3f} s; "
            f"minimum is {CPPS_MINIMUM_DURATION_S:.3f} s"
        )
    return samples


def compute_cpps(
    waveform: np.ndarray,
    sample_rate: int = EXPECTED_SAMPLE_RATE_HZ,
) -> float:
    """Return one finite whole-clip CPPS value for canonical mono audio.

    The caller must supply an already-decoded 16 kHz waveform. This function
    does not decode, resample, downmix, normalize, pad, crop, or otherwise
    prepare audio. The recipe was validated with ``praat-parselmouth==0.4.7``
    (bundled Praat 6.1.38).
    """

    samples = _validate_waveform(waveform, sample_rate)

    try:
        sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)
        power_cepstrogram = call(
            sound,
            "To PowerCepstrogram",
            CPPS_PITCH_FLOOR_HZ,
            CPPS_TIME_STEP_S,
            CPPS_MAX_FREQUENCY_HZ,
            CPPS_PREEMPHASIS_FROM_HZ,
        )
        cpps = call(
            power_cepstrogram,
            "Get CPPS",
            CPPS_SUBTRACT_TREND_BEFORE_SMOOTHING,
            CPPS_TIME_AVERAGING_WINDOW_S,
            CPPS_QUEFRENCY_AVERAGING_WINDOW_S,
            CPPS_PEAK_SEARCH_FLOOR_HZ,
            CPPS_PEAK_SEARCH_CEILING_HZ,
            CPPS_TOLERANCE,
            CPPS_INTERPOLATION,
            CPPS_TREND_LINE_MIN_QUEFRENCY_S,
            CPPS_TREND_LINE_MAX_QUEFRENCY_S,
            CPPS_TREND_TYPE,
            CPPS_FIT_METHOD,
        )
    except Exception as exc:
        raise CPPSExtractionError(f"Praat CPPS extraction failed: {exc}") from exc

    try:
        result = float(cpps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CPPSExtractionError("Praat returned a nonnumeric CPPS value") from exc
    if not np.isfinite(result):
        raise CPPSExtractionError("Praat returned a nonfinite CPPS value")
    return result
#
# from ProcessFiles import *
# from pathlib import Path
# for file in Path(input()).iterdir():
#     print(compute_cpps(normalize_file(f"SmallTest/{file.name}"), 16000))