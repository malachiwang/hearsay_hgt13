import argparse
import numpy as np
from scipy.signal import butter, sosfiltfilt, stft
import matplotlib.pyplot as plt
import soundfile as sf


def bandpass_filter(signal, fs, low, high, order=4):
    """Zero-phase Butterworth bandpass filter."""
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def parabolic_interp(mags, k):
    """
    Sub-bin frequency correction via parabolic interpolation
    around bin index k, using the magnitudes at k-1, k, k+1.
    Returns a fractional bin offset in [-0.5, 0.5].
    """
    if k <= 0 or k >= len(mags) - 1:
        return 0.0
    alpha, beta, gamma = mags[k - 1], mags[k], mags[k + 1]
    denom = (alpha - 2 * beta + gamma)
    if denom == 0:
        return 0.0
    return 0.5 * (alpha - gamma) / denom


def extract_enf(signal, fs, nominal_freq=60.0, harmonic=1,
                band_halfwidth=0.6, nperseg=None, noverlap_ratio=0.75):
    """
    Extract an ENF time series from `signal`.

    Parameters
    ----------
    signal : np.ndarray
        Mono audio signal.
    fs : int
        Sample rate.
    nominal_freq : float
        50.0 or 60.0 depending on region.
    harmonic : int
        Which harmonic to track (1 = fundamental, 2 = 2nd harmonic, etc).
        Higher harmonics can sometimes have a stronger/cleaner ENF signal.
    band_halfwidth : float
        Half-width (Hz) of the bandpass window around the target frequency.
    nperseg : int or None
        STFT window length in samples. Longer = better frequency
        resolution, worse time resolution. Defaults to ~4 seconds.
    noverlap_ratio : float
        Fraction of window overlap between successive STFT frames.

    Returns
    -------
    times : np.ndarray
        Time (s) for each ENF estimate.
    freqs : np.ndarray
        Estimated instantaneous frequency (Hz) at each time point.
    """
    target = nominal_freq * harmonic
    low = target - band_halfwidth
    high = target + band_halfwidth

    filtered = bandpass_filter(signal, fs, low, high)

    if nperseg is None:
        nperseg = int(fs * 4)  # ~4-second windows -> ~0.25 Hz bin resolution
    noverlap = int(nperseg * noverlap_ratio)

    f, t, Zxx = stft(filtered, fs=fs, nperseg=nperseg, noverlap=noverlap,
                     boundary=None)
    mag = np.abs(Zxx)

    # restrict search to the filtered band
    band_mask = (f >= low) & (f <= high)
    f_band = f[band_mask]
    mag_band = mag[band_mask, :]

    freqs = np.zeros(mag_band.shape[1])
    for i in range(mag_band.shape[1]):
        col = mag_band[:, i]
        k = np.argmax(col)
        delta = parabolic_interp(col, k)
        bin_width = f_band[1] - f_band[0] if len(f_band) > 1 else 0
        freqs[i] = f_band[k] + delta * bin_width

    return t, freqs


def detect_discontinuities(times, freqs, nominal_freq=60.0,
                           jump_threshold=0.05, window=5):
    """
    Flag abrupt jumps in the ENF trace that are inconsistent with
    natural grid-frequency drift. Real ENF changes smoothly frame to
    frame; a splice or insertion often creates a sharp discontinuity.

    Parameters
    ----------
    jump_threshold : float
        Max plausible frame-to-frame change (Hz) before flagging.
        Tune this against known-clean recordings from your target grid;
        0.05 Hz is a reasonable starting point for adjacent ~1s frames.
    window : int
        Number of frames on each side used to compute a local baseline
        deviation, so isolated noise spikes are distinguished from
        sustained level shifts.

    Returns
    -------
    flagged_indices : list of int
        Indices in `freqs` where a suspicious jump was detected.
    """
    diffs = np.abs(np.diff(freqs))
    flagged = []
    for i, d in enumerate(diffs):
        if d > jump_threshold:
            lo = max(0, i - window)
            hi = min(len(freqs), i + window + 2)
            local_std = np.std(freqs[lo:hi])
            # only flag if the jump is also an outlier vs local variability
            if d > 3 * local_std or local_std == 0:
                flagged.append(i + 1)  # index of the frame *after* the jump
    return flagged


def enf_variance_check(freqs, min_expected_std=0.005, max_expected_std=0.08):
    """
    Real grid ENF fluctuates naturally; a suspiciously flat trace
    (e.g. an artificially inserted pure tone) or a wildly erratic one
    (e.g. noise misidentified as ENF) both suggest the signal is not
    genuine captured mains hum.

    Returns a dict with the measured std and a verdict string:
    'too_stable', 'too_erratic', or 'plausible'. Treat the thresholds
    as rough defaults, not a certified test, calibrate them against
    known-authentic recordings from your target region if possible.
    """
    std = float(np.std(freqs))
    if std < min_expected_std:
        verdict = "too_stable"
    elif std > max_expected_std:
        verdict = "too_erratic"
    else:
        verdict = "plausible"
    return {"std": std, "verdict": verdict}

from ProcessFiles import *
def checks(y):
    times, freq = extract_enf(y, 16000)

    return len(detect_discontinuities(times, freq)), enf_variance_check(freq)['verdict']

# print(checks(input()))
from ProcessFiles import *
from pathlib import Path
for file in Path(input()).iterdir():
    print(checks(normalize_file(f"SmallTest/{file.name}")))