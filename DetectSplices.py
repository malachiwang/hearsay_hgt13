"""Locate splice discontinuities in an already-prepared mono waveform.

A join of two different sounds is a step in the spectrum, the level, or the
waveform. Pauses, slow pitch glides, and amplitude flutter are not. A candidate
is kept only when the step is sharper than the same measurement on either side
of it, both sides are audible, and a moving tonal peak does not explain it.

This module performs no decoding, resampling, downmixing, loudness
normalization, padding/cropping of the source, or file conversion. PyWavelets
supplies the ``sym4`` packet tree and the Haar detail used for clicks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pywt
from scipy.ndimage import maximum_filter, median_filter
from scipy.signal import find_peaks


EXPECTED_SAMPLE_RATE_HZ = 16_000

# A channel score of 1 is a clear hit. The candidate gate sits just under that.
TIMBRE_SCORE_SCALE = 1.15
LEVEL_SCORE_SCALE_DB = 7.0
TRANSIENT_FLOOR = 3.0
TRANSIENT_SCORE_SCALE = 5.0
FREQ_SCORE_SCALE_HZ = 90.0
CANDIDATE_SCORE = 0.85
PEAK_SHARE_TONAL = 0.55


class SpliceDetectionError(ValueError):
    """Raised when the splice detector input or output is invalid."""


@dataclass(frozen=True)
class SpliceCandidate:
    time_sec: float
    score: float
    kind: str
    timbre_contrast: float
    level_contrast_db: float
    transient_excess: float
    freq_contrast_hz: float


@dataclass(frozen=True)
class SpliceResult:
    sample_rate: int
    duration_sec: float
    candidates: list[SpliceCandidate] = field(default_factory=list)
    max_timbre_contrast: float = 0.0
    max_level_contrast_db: float = 0.0
    max_transient_excess: float = 0.0
    max_freq_contrast_hz: float = 0.0
    max_score: float = 0.0
    note: str = ""

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


def _validate_input(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if isinstance(sr, (bool, np.bool_)):
        raise SpliceDetectionError("splice detection sample rate must be positive")
    try:
        sample_rate = int(sr)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SpliceDetectionError("splice detection sample rate must be positive") from exc
    if sample_rate <= 0:
        raise SpliceDetectionError("splice detection sample rate must be positive")
    if sample_rate != sr:
        raise SpliceDetectionError("splice detection sample rate must be an integer")
    if sample_rate != EXPECTED_SAMPLE_RATE_HZ:
        raise SpliceDetectionError(
            "HEARSAY splice detection requires 16 kHz audio; "
            f"received {sample_rate} Hz"
        )

    try:
        signal = np.asarray(y)
    except (TypeError, ValueError) as exc:
        raise SpliceDetectionError("splice waveform must be a numeric array") from exc
    if signal.ndim != 1:
        raise SpliceDetectionError(
            "detect_splices requires a mono 1-D waveform; "
            f"received shape {signal.shape}"
        )
    if signal.size == 0:
        raise SpliceDetectionError("splice waveform must not be empty")
    if not np.issubdtype(signal.dtype, np.number) or np.issubdtype(
        signal.dtype, np.complexfloating
    ):
        raise SpliceDetectionError("splice waveform must contain real numeric samples")

    signal = np.ascontiguousarray(signal, dtype=np.float64)
    if not np.isfinite(signal).all():
        raise SpliceDetectionError("splice waveform contains NaN or Inf")
    return signal, sample_rate


def detect_splices(
    y: np.ndarray,
    sr: int,
    *,
    wavelet: str = "sym4",
    win_ms: float = 64.0,
    guard_ms: float = 12.0,
    hop_ms: float = 8.0,
    pad_ms: float = 220.0,
) -> SpliceResult:
    """Return whole-clip discontinuity candidates and continuous features.

    ``y`` must be an already-prepared, finite, mono 16 kHz waveform. Near
    silence is a valid safe result with no candidates and zero-valued features.
    The detector does not mutate the caller's waveform.
    """

    signal, sr = _validate_input(y, sr)
    duration = float(signal.size / sr)
    signal = signal - float(np.mean(signal))
    peak = float(np.max(np.abs(signal)))
    if peak < 1e-8:
        return SpliceResult(
            sample_rate=sr,
            duration_sec=duration,
            note="near-silent audio",
        )

    level = choose_level(sr)
    grouped = _packet_groups(
        signal,
        sr,
        wavelet=wavelet,
        level=level,
        pad_ms=pad_ms,
    )
    if grouped is None:
        return _too_short(sr, duration)
    frame = _frame_views(
        grouped,
        sr,
        len(signal),
        win_ms=win_ms,
        guard_ms=guard_ms,
        hop_ms=hop_ms,
    )
    if frame is None:
        return _too_short(sr, duration)

    timbre, level_db = _contrasts(signal, frame)
    freq_hz = _freq_contrast_hz(frame)
    if _tonal_glide(frame):
        # A pure sweep is a straight line in centroid, not a step. Zero the
        # spectral channels so band crossings are not reported as splices.
        timbre = np.zeros_like(timbre)
        freq_hz = np.zeros_like(freq_hz)
    transient = _transient_excess(signal, sr, frame)
    audible = _audible_mask(signal, sr, frame)
    timbre = np.clip(np.where(audible, timbre, 0.0), 0.0, None)
    level_db = np.clip(np.where(audible, level_db, 0.0), 0.0, None)
    freq_hz = np.clip(np.where(audible, freq_hz, 0.0), 0.0, None)
    transient = np.where(audible, transient, 0.0)

    score, kind_index = _combined_score(timbre, level_db, transient, freq_hz)
    kinds = ("timbre", "level", "transient", "frequency")
    candidates = [
        SpliceCandidate(
            time_sec=float(frame["times"][index]),
            score=float(score[index]),
            kind=kinds[int(kind_index[index])],
            timbre_contrast=float(timbre[index]),
            level_contrast_db=float(level_db[index]),
            transient_excess=float(transient[index]),
            freq_contrast_hz=float(freq_hz[index]),
        )
        for index in _pick_peaks(score, frame["hop_sec"])
    ]
    candidates.sort(key=lambda item: item.time_sec)
    return SpliceResult(
        sample_rate=sr,
        duration_sec=duration,
        candidates=candidates,
        max_timbre_contrast=_max(timbre),
        max_level_contrast_db=_max(level_db),
        max_transient_excess=_max(transient),
        max_freq_contrast_hz=_max(freq_hz),
        max_score=_max(score),
        note=(
            f"wavelet_packet {wavelet} L{level}; "
            f"{grouped['n_groups']} log bands"
        ),
    )


def splice_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """Return finite fusion-ready clip-level splice features."""

    result = detect_splices(y, sr)
    features = {
        "splice_max_score": float(result.max_score),
        "splice_n_candidates": float(result.n_candidates),
        "splice_max_timbre": float(result.max_timbre_contrast),
        "splice_max_level_db": float(result.max_level_contrast_db),
        "splice_max_transient": float(result.max_transient_excess),
        "splice_max_frequency_hz": float(result.max_freq_contrast_hz),
    }
    if not all(np.isfinite(value) for value in features.values()):
        raise SpliceDetectionError("splice detector produced nonfinite features")
    return features


def splice_score(y: np.ndarray, sr: int) -> float:
    """Return the initial higher-is-more-suspicious standalone splice score."""

    return splice_features(y, sr)["splice_max_score"]


def choose_level(sr: int) -> int:
    """Packet depth whose linear bins are about 70 Hz wide."""

    level = int(round(np.log2(sr / 70.0) - 1.0))
    return int(np.clip(level, 5, 8))


def _too_short(sr: int, duration: float) -> SpliceResult:
    return SpliceResult(
        sample_rate=sr,
        duration_sec=duration,
        note="clip shorter than the wavelet analysis window",
    )


def _max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values))


def _packet_groups(
    y: np.ndarray,
    sr: int,
    *,
    wavelet: str,
    level: int,
    pad_ms: float,
) -> dict | None:
    step = 2**level
    if len(y) < step * 8:
        return None
    pad_n = int(np.ceil((pad_ms / 1000.0) * sr / step) * step)
    pad_n = int(min(pad_n, max(step, len(y) - 1)))
    pad_n -= pad_n % step
    padded = np.pad(y, (pad_n, pad_n), mode="reflect")
    extra = (-len(padded)) % step
    if extra:
        padded = np.pad(padded, (0, extra), mode="edge")
    packet = pywt.WaveletPacket(
        padded,
        wavelet=wavelet,
        mode="symmetric",
        maxlevel=level,
    )
    nodes = packet.get_level(level, order="freq")
    linear = np.vstack(
        [np.asarray(node.data, dtype=np.float64) ** 2 for node in nodes]
    )
    n_bins = linear.shape[0]
    n_groups = int(np.clip(n_bins // 6, 12, 24))
    edges = np.unique(np.geomspace(1, n_bins, n_groups + 1).astype(int))
    bin_hz = (np.arange(n_bins) + 0.5) * (sr / 2.0) / n_bins
    groups: list[np.ndarray] = []
    centers: list[float] = []
    start = 0
    stops = list(edges)
    if not stops or stops[-1] != n_bins:
        stops.append(n_bins)
    for edge in stops:
        if edge <= start:
            continue
        groups.append(linear[start:edge].sum(axis=0))
        centers.append(float(np.mean(bin_hz[start:edge])))
        start = int(edge)
    energy = np.vstack(groups)
    csum = np.concatenate(
        [np.zeros((energy.shape[0], 1)), np.cumsum(energy, axis=1)],
        axis=1,
    )
    return {
        "csum": csum,
        "centers_hz": np.asarray(centers, dtype=np.float64),
        "step": step,
        "pad_n": pad_n,
        "n_coef": int(energy.shape[1]),
        "n_groups": int(energy.shape[0]),
    }


def _frame_views(
    grouped: dict,
    sr: int,
    n_samples: int,
    *,
    win_ms: float,
    guard_ms: float,
    hop_ms: float,
) -> dict | None:
    step = grouped["step"]
    win = max(2, int(round((win_ms / 1000.0) * sr / step)))
    guard = max(1, int(round((guard_ms / 1000.0) * sr / step)))
    hop = max(1, int(round((hop_ms / 1000.0) * sr / step)))
    # Same start-to-start gap for the across pair and each same-side pair, so a
    # constant-rate glide is not marked just because the across pair is wider.
    sep = win + 2 * guard
    back = guard + win + sep
    fwd = guard + sep + win
    orig_lo = grouped["pad_n"] / step
    orig_hi = (grouped["pad_n"] + n_samples) / step
    c0 = int(np.ceil(orig_lo + back))
    c1 = int(np.floor(orig_hi - fwd))
    if c1 - c0 < 3:
        return None
    centers = np.arange(c0, c1, hop, dtype=int)
    l1 = centers - guard - win
    r1 = centers + guard
    l0 = l1 - sep
    r2 = r1 + sep
    if int(l0.min()) < 0 or int((r2 + win).max()) > grouped["n_coef"]:
        return None

    def window(start: np.ndarray) -> np.ndarray:
        stop = start + win
        csum = grouped["csum"]
        return (csum[:, stop] - csum[:, start]) / win

    times = np.clip(centers * step - grouped["pad_n"], 0, n_samples - 1) / sr
    return {
        "e_l0": window(l0),
        "e_l1": window(l1),
        "e_r1": window(r1),
        "e_r2": window(r2),
        "l0": l0,
        "l1": l1,
        "r1": r1,
        "r2": r2,
        "win": win,
        "times": times.astype(np.float64),
        "step": step,
        "pad_n": grouped["pad_n"],
        "centers_hz": grouped["centers_hz"],
        "hop_sec": hop * step / sr,
    }


def _rms(y: np.ndarray, frame: dict, start_coef: np.ndarray) -> np.ndarray:
    step = frame["step"]
    pad_n = frame["pad_n"]
    n = len(y)
    energy = np.concatenate([[0.0], np.cumsum(y * y)])
    start = np.clip(start_coef * step - pad_n, 0, n).astype(int)
    stop = np.clip(
        (start_coef + frame["win"]) * step - pad_n,
        0,
        n,
    ).astype(int)
    width = np.maximum(1, stop - start)
    return np.sqrt(np.maximum(0.0, energy[stop] - energy[start]) / width)


def _power_gain(rms_a: np.ndarray, rms_b: np.ndarray) -> np.ndarray:
    return 2.0 * np.log((rms_b + 1e-12) / (rms_a + 1e-12))


def _timbre(left: np.ndarray, right: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Mean absolute log-energy residual after removing broadband gain."""

    eps = 1e-12
    peak = np.maximum(left.max(axis=0), right.max(axis=0)) + eps
    active = np.maximum(left, right) > (0.03 * peak)
    delta = np.log(right + eps) - np.log(left + eps)
    resid = np.where(active, np.abs(delta - gain), np.nan)
    mean = np.nan_to_num(np.nanmean(resid, axis=0), nan=0.0)
    return np.where(np.sum(active, axis=0) >= 2, mean, 0.0)


def _shift_frequency_bins(values: np.ndarray, amount: int) -> np.ndarray:
    """Shift one frequency vector without wrapping energy across its edges."""

    shifted = np.zeros_like(values)
    amount = int(amount)
    if amount == 0:
        shifted[...] = values
    elif 0 < amount < values.shape[0]:
        shifted[amount:] = values[:-amount]
    elif -values.shape[0] < amount < 0:
        shifted[:amount] = values[-amount:]
    return shifted


def _align_peak(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    shift = np.argmax(reference, axis=0) - np.argmax(other, axis=0)
    aligned = np.empty_like(other)
    for index, amount in enumerate(shift):
        aligned[:, index] = _shift_frequency_bins(other[:, index], int(amount))
    return aligned


def _contrasts(y: np.ndarray, frame: dict) -> tuple[np.ndarray, np.ndarray]:
    rms = {
        "l0": _rms(y, frame, frame["l0"]),
        "l1": _rms(y, frame, frame["l1"]),
        "r1": _rms(y, frame, frame["r1"]),
        "r2": _rms(y, frame, frame["r2"]),
    }
    across = _timbre(
        frame["e_l1"],
        frame["e_r1"],
        _power_gain(rms["l1"], rms["r1"]),
    )
    left = _timbre(
        frame["e_l0"],
        frame["e_l1"],
        _power_gain(rms["l0"], rms["l1"]),
    )
    right = _timbre(
        frame["e_r1"],
        frame["e_r2"],
        _power_gain(rms["r1"], rms["r2"]),
    )
    raw = across - np.maximum(left, right)

    aligned_across = _timbre(
        frame["e_l1"],
        _align_peak(frame["e_l1"], frame["e_r1"]),
        _power_gain(rms["l1"], rms["r1"]),
    )
    aligned_left = _timbre(
        frame["e_l0"],
        _align_peak(frame["e_l0"], frame["e_l1"]),
        _power_gain(rms["l0"], rms["l1"]),
    )
    aligned_right = _timbre(
        frame["e_r1"],
        _align_peak(frame["e_r1"], frame["e_r2"]),
        _power_gain(rms["r1"], rms["r2"]),
    )
    aligned = aligned_across - np.maximum(aligned_left, aligned_right)
    share_l = frame["e_l1"].max(axis=0) / (
        frame["e_l1"].sum(axis=0) + 1e-12
    )
    share_r = frame["e_r1"].max(axis=0) / (
        frame["e_r1"].sum(axis=0) + 1e-12
    )
    tonal = (share_l > PEAK_SHARE_TONAL) & (share_r > PEAK_SHARE_TONAL)
    # A chirp's contrast collapses after the peak bin is lined up. A formant
    # or noise-color join does not, and noise is not tonal so it skips this.
    explained = tonal & (aligned < 0.55) & (
        aligned < 0.5 * np.maximum(raw, 1e-6)
    )
    timbre = np.where(explained, np.clip(aligned, 0.0, None), raw)

    def db(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs(20.0 * np.log10((b + 1e-12) / (a + 1e-12)))

    level = db(rms["l1"], rms["r1"]) - np.maximum(
        db(rms["l0"], rms["l1"]),
        db(rms["r1"], rms["r2"]),
    )
    return timbre, level


def _centroids(frame: dict) -> dict[str, np.ndarray]:
    hz = frame["centers_hz"]

    def centroid(energy: np.ndarray) -> np.ndarray:
        weights = energy / (energy.sum(axis=0) + 1e-12)
        return hz @ weights

    return {
        name: centroid(frame[name])
        for name in ("e_l0", "e_l1", "e_r1", "e_r2")
    }


def _tonal_glide(frame: dict) -> bool:
    """Return whether the spectral centroid is a sweep rather than a step."""

    cent = _centroids(frame)["e_l1"]
    times = frame["times"]
    if cent.size < 8 or float(np.ptp(cent)) < 150.0:
        return False
    slope = np.polyfit(times, cent, 1)
    residual = cent - np.polyval(slope, times)
    return float(np.std(residual)) < 60.0


def _freq_contrast_hz(frame: dict) -> np.ndarray:
    cent = _centroids(frame)
    jump = np.abs(cent["e_r1"] - cent["e_l1"])
    left = np.abs(cent["e_l1"] - cent["e_l0"])
    right = np.abs(cent["e_r2"] - cent["e_r1"])
    contrast = np.clip(jump - np.maximum(left, right), 0.0, None)

    def share(energy: np.ndarray) -> np.ndarray:
        return energy.max(axis=0) / (energy.sum(axis=0) + 1e-12)

    tonal = (share(frame["e_l1"]) > PEAK_SHARE_TONAL) & (
        share(frame["e_r1"]) > PEAK_SHARE_TONAL
    )
    return np.where(tonal, contrast, 0.0)


def _audible_mask(y: np.ndarray, sr: int, frame: dict) -> np.ndarray:
    frame_len = max(1, int(0.01 * sr))
    n_frames = len(y) // frame_len
    if n_frames < 2:
        return np.ones(len(frame["times"]), dtype=bool)
    rms = np.sqrt(
        np.mean(
            (y[: n_frames * frame_len] ** 2).reshape(n_frames, frame_len),
            axis=1,
        )
    )
    threshold = max(1e-4, 0.04 * float(np.percentile(rms, 80)))
    silent = rms < threshold
    prefix = np.concatenate([[0.0], np.cumsum(silent.astype(np.float64))])

    def fraction(start_coef: np.ndarray) -> np.ndarray:
        step = frame["step"]
        pad_n = frame["pad_n"]
        n = len(y)
        start = (
            np.clip(start_coef * step - pad_n, 0, n).astype(int) // frame_len
        )
        stop = (
            np.clip(
                (start_coef + frame["win"]) * step - pad_n,
                0,
                n,
            ).astype(int)
            // frame_len
        )
        start = np.clip(start, 0, n_frames - 1)
        stop = np.clip(np.maximum(start + 1, stop), 0, n_frames)
        return (prefix[stop] - prefix[start]) / (stop - start)

    loud = (_rms(y, frame, frame["l1"]) > threshold) & (
        _rms(y, frame, frame["r1"]) > threshold
    )
    return loud & (fraction(frame["l1"]) < 0.30) & (
        fraction(frame["r1"]) < 0.30
    )


def _transient_excess(y: np.ndarray, sr: int, frame: dict) -> np.ndarray:
    """Haar spike versus the surrounding 0.4 s, pooled over the 8 ms hop."""

    padded = y if len(y) % 2 == 0 else np.pad(y, (0, 1))
    detail = np.abs(
        pywt.swt(padded, "haar", level=1, norm=True)[0][1][: len(y)]
    )
    half = max(1, int(0.002 * sr))
    peak = maximum_filter(detail, size=2 * half + 1)
    base = median_filter(detail, size=max(3, int(0.05 * sr) | 1))
    ratio = peak / (base + 1e-8)
    local = median_filter(ratio, size=max(3, int(0.4 * sr) | 1))
    excess = ratio / (local + 1e-8)
    pooled = maximum_filter(excess, size=max(3, int(0.016 * sr) | 1))
    centers = np.clip(
        np.rint(frame["times"] * sr).astype(int),
        0,
        len(y) - 1,
    )
    return pooled[centers]


def _combined_score(
    timbre: np.ndarray,
    level_db: np.ndarray,
    transient: np.ndarray,
    freq_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    channels = np.vstack(
        [
            np.clip(timbre, 0.0, None) / TIMBRE_SCORE_SCALE,
            np.clip(level_db, 0.0, None) / LEVEL_SCORE_SCALE_DB,
            np.clip(transient - TRANSIENT_FLOOR, 0.0, None)
            / TRANSIENT_SCORE_SCALE,
            np.clip(freq_hz, 0.0, None) / FREQ_SCORE_SCALE_HZ,
        ]
    )
    return np.max(channels, axis=0), np.argmax(channels, axis=0)


def _pick_peaks(score: np.ndarray, hop_sec: float) -> np.ndarray:
    if score.size < 3 or float(np.max(score)) < CANDIDATE_SCORE:
        return np.array([], dtype=int)
    distance = max(1, int(round(0.15 / max(hop_sec, 1e-6))))
    peaks, _properties = find_peaks(
        score,
        height=CANDIDATE_SCORE,
        distance=distance,
        prominence=0.55,
    )
    return peaks
