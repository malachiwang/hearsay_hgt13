from __future__ import annotations

import os
import subprocess
import sys
import unittest

import numpy as np
from scipy.signal import chirp

from DetectSplices import (
    CANDIDATE_SCORE,
    SpliceDetectionError,
    SpliceResult,
    _shift_frequency_bins,
    detect_splices,
    splice_features,
    splice_score,
)


SAMPLE_RATE = 16_000
DURATION_SECONDS = 3.0
SAMPLE_COUNT = round(SAMPLE_RATE * DURATION_SECONDS)
JOIN_TIME_SECONDS = 1.5
JOIN_INDEX = round(SAMPLE_RATE * JOIN_TIME_SECONDS)
TIMES = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE


def tone(frequency_hz: float = 220.0, amplitude: float = 0.25) -> np.ndarray:
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * TIMES)


def hard_amplitude_step() -> np.ndarray:
    envelope = np.where(np.arange(SAMPLE_COUNT) < JOIN_INDEX, 0.08, 0.65)
    return envelope * np.sin(2.0 * np.pi * 220.0 * TIMES)


def spectral_step() -> np.ndarray:
    signal = np.empty(SAMPLE_COUNT, dtype=np.float64)
    signal[:JOIN_INDEX] = 0.25 * np.sin(
        2.0 * np.pi * 220.0 * TIMES[:JOIN_INDEX]
    )
    signal[JOIN_INDEX:] = 0.25 * np.sin(
        2.0 * np.pi * 880.0 * TIMES[JOIN_INDEX:]
    )
    return signal


class DetectSplicesTests(unittest.TestCase):
    def test_import_is_side_effect_free(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", "import DetectSplices"],
            capture_output=True,
            text=True,
            timeout=10,
            env=os.environ.copy(),
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")

    def test_continuous_signal_has_low_response_and_finite_features(self) -> None:
        signal = tone()
        result = detect_splices(signal, SAMPLE_RATE)
        features = splice_features(signal, SAMPLE_RATE)

        self.assertIsInstance(result, SpliceResult)
        self.assertTrue(np.isfinite(result.max_score))
        self.assertEqual(result.n_candidates, 0)
        self.assertLess(result.max_score, CANDIDATE_SCORE)
        self.assertEqual(
            set(features),
            {
                "splice_max_score",
                "splice_n_candidates",
                "splice_max_timbre",
                "splice_max_level_db",
                "splice_max_transient",
                "splice_max_frequency_hz",
            },
        )
        self.assertTrue(all(np.isfinite(value) for value in features.values()))
        self.assertEqual(splice_score(signal, SAMPLE_RATE), features["splice_max_score"])

    def test_hard_amplitude_step_is_detected(self) -> None:
        result = detect_splices(hard_amplitude_step(), SAMPLE_RATE)

        self.assertGreater(result.max_level_contrast_db, 7.0)
        self.assertGreater(result.max_score, CANDIDATE_SCORE)
        self.assertTrue(
            any(
                abs(candidate.time_sec - JOIN_TIME_SECONDS) <= 0.08
                for candidate in result.candidates
            )
        )

    def test_spectral_step_raises_timbre_or_frequency_contrast(self) -> None:
        continuous = detect_splices(tone(), SAMPLE_RATE)
        result = detect_splices(spectral_step(), SAMPLE_RATE)

        self.assertGreater(result.max_score, continuous.max_score)
        self.assertTrue(
            result.max_timbre_contrast > continuous.max_timbre_contrast
            or result.max_freq_contrast_hz > continuous.max_freq_contrast_hz
        )
        self.assertGreaterEqual(result.n_candidates, 1)

    def test_click_raises_transient_feature_near_event(self) -> None:
        signal = tone()
        signal[JOIN_INDEX] += 1.5

        result = detect_splices(signal, SAMPLE_RATE)

        self.assertGreater(result.max_transient_excess, 10.0)
        self.assertTrue(
            any(
                candidate.kind == "transient"
                and abs(candidate.time_sec - JOIN_TIME_SECONDS) <= 0.04
                for candidate in result.candidates
            )
        )

    def test_smooth_amplitude_change_is_weaker_than_hard_step(self) -> None:
        smooth = np.linspace(0.08, 0.65, SAMPLE_COUNT) * np.sin(
            2.0 * np.pi * 220.0 * TIMES
        )

        smooth_result = detect_splices(smooth, SAMPLE_RATE)
        hard_result = detect_splices(hard_amplitude_step(), SAMPLE_RATE)

        self.assertEqual(smooth_result.n_candidates, 0)
        self.assertGreater(hard_result.max_score, 5.0 * smooth_result.max_score)

    def test_clean_tonal_glide_has_low_false_positive_response(self) -> None:
        signal = 0.25 * chirp(
            TIMES,
            f0=180.0,
            f1=1_000.0,
            t1=DURATION_SECONDS,
            method="linear",
        )

        result = detect_splices(signal, SAMPLE_RATE)

        self.assertEqual(result.n_candidates, 0)
        self.assertLess(result.max_score, CANDIDATE_SCORE)

    def test_tonal_glide_with_level_step_is_not_fully_suppressed(self) -> None:
        signal = 0.25 * chirp(
            TIMES,
            f0=180.0,
            f1=1_000.0,
            t1=DURATION_SECONDS,
            method="linear",
        )
        signal[JOIN_INDEX:] *= 3.0

        result = detect_splices(signal, SAMPLE_RATE)

        self.assertGreaterEqual(result.n_candidates, 1)
        self.assertTrue(
            any(
                abs(candidate.time_sec - JOIN_TIME_SECONDS) <= 0.08
                for candidate in result.candidates
            )
        )

    def test_tonal_frequency_jump_is_not_fully_suppressed(self) -> None:
        signal = np.empty(SAMPLE_COUNT, dtype=np.float64)
        signal[:JOIN_INDEX] = 0.25 * chirp(
            TIMES[:JOIN_INDEX],
            f0=180.0,
            f1=500.0,
            t1=JOIN_TIME_SECONDS,
            method="linear",
        )
        signal[JOIN_INDEX:] = 0.25 * chirp(
            TIMES[JOIN_INDEX:] - JOIN_TIME_SECONDS,
            f0=950.0,
            f1=1_200.0,
            t1=DURATION_SECONDS - JOIN_TIME_SECONDS,
            method="linear",
        )

        result = detect_splices(signal, SAMPLE_RATE)

        self.assertGreaterEqual(result.n_candidates, 1)
        self.assertTrue(
            any(
                abs(candidate.time_sec - JOIN_TIME_SECONDS) <= 0.08
                for candidate in result.candidates
            )
        )

    def test_silence_returns_safe_zero_features(self) -> None:
        signal = np.zeros(SAMPLE_COUNT, dtype=np.float32)

        result = detect_splices(signal, SAMPLE_RATE)
        features = splice_features(signal, SAMPLE_RATE)

        self.assertEqual(result.n_candidates, 0)
        self.assertEqual(result.max_score, 0.0)
        self.assertEqual(result.note, "near-silent audio")
        self.assertTrue(all(value == 0.0 for value in features.values()))

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(SpliceDetectionError, "must not be empty"):
            detect_splices(np.array([], dtype=np.float32), SAMPLE_RATE)

    def test_nonfinite_input_is_rejected(self) -> None:
        for nonfinite in (np.nan, np.inf, -np.inf):
            with self.subTest(nonfinite=nonfinite):
                signal = tone()
                signal[100] = nonfinite
                with self.assertRaisesRegex(SpliceDetectionError, "NaN or Inf"):
                    detect_splices(signal, SAMPLE_RATE)

    def test_stereo_input_is_rejected(self) -> None:
        stereo = np.column_stack((tone(), tone()))

        with self.assertRaisesRegex(SpliceDetectionError, "mono 1-D"):
            detect_splices(stereo, SAMPLE_RATE)

    def test_wrong_sample_rate_is_rejected(self) -> None:
        with self.assertRaisesRegex(SpliceDetectionError, "requires 16 kHz"):
            detect_splices(tone(), 22_050)

    def test_frequency_bin_shift_does_not_wrap(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 9.0])

        shifted_up = _shift_frequency_bins(values, 1)
        shifted_down = _shift_frequency_bins(values, -1)

        np.testing.assert_array_equal(shifted_up, [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(shifted_down, [2.0, 3.0, 9.0, 0.0])
        self.assertNotEqual(shifted_up[0], values[-1])
        self.assertNotEqual(shifted_down[-1], values[0])

    def test_best_candidate_is_close_to_known_splice_time(self) -> None:
        result = detect_splices(spectral_step(), SAMPLE_RATE)
        best_candidate = max(result.candidates, key=lambda candidate: candidate.score)

        self.assertAlmostEqual(
            best_candidate.time_sec,
            JOIN_TIME_SECONDS,
            delta=0.08,
        )


if __name__ == "__main__":
    unittest.main()
