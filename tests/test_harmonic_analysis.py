from __future__ import annotations

import unittest
from unittest.mock import ANY, call as mock_call, patch

import numpy as np
import parselmouth

from HarmonicAnalysis import (
    CPPSExtractionError,
    CPPS_FIT_METHOD,
    CPPS_INTERPOLATION,
    CPPS_MAX_FREQUENCY_HZ,
    CPPS_PEAK_SEARCH_CEILING_HZ,
    CPPS_PEAK_SEARCH_FLOOR_HZ,
    CPPS_PITCH_FLOOR_HZ,
    CPPS_PREEMPHASIS_FROM_HZ,
    CPPS_QUEFRENCY_AVERAGING_WINDOW_S,
    CPPS_SUBTRACT_TREND_BEFORE_SMOOTHING,
    CPPS_TIME_AVERAGING_WINDOW_S,
    CPPS_TIME_STEP_S,
    CPPS_TOLERANCE,
    CPPS_TREND_LINE_MAX_QUEFRENCY_S,
    CPPS_TREND_LINE_MIN_QUEFRENCY_S,
    CPPS_TREND_TYPE,
    compute_cpps,
)


SAMPLE_RATE = 16_000


def periodic_waveform(duration_seconds: float = 1.0) -> np.ndarray:
    sample_count = round(SAMPLE_RATE * duration_seconds)
    times = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    return (0.2 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)


class CPPSTests(unittest.TestCase):
    def test_valid_audio_returns_finite_python_float(self) -> None:
        result = compute_cpps(periodic_waveform())

        self.assertIs(type(result), float)
        self.assertTrue(np.isfinite(result))

    def test_deterministic_for_identical_waveform(self) -> None:
        waveform = periodic_waveform()

        first = compute_cpps(waveform)
        second = compute_cpps(waveform)

        self.assertAlmostEqual(first, second, places=12)

    def test_wrong_sample_rate_is_rejected(self) -> None:
        with self.assertRaisesRegex(CPPSExtractionError, "16 kHz.*22050 Hz"):
            compute_cpps(periodic_waveform(), sample_rate=22_050)

    def test_stereo_is_rejected(self) -> None:
        stereo = np.column_stack((periodic_waveform(), periodic_waveform()))

        with self.assertRaisesRegex(CPPSExtractionError, "mono 1-D"):
            compute_cpps(stereo)

    def test_empty_waveform_is_rejected(self) -> None:
        with self.assertRaisesRegex(CPPSExtractionError, "must not be empty"):
            compute_cpps(np.array([], dtype=np.float32))

    def test_nan_is_rejected(self) -> None:
        waveform = periodic_waveform()
        waveform[10] = np.nan

        with self.assertRaisesRegex(CPPSExtractionError, "NaN or Inf"):
            compute_cpps(waveform)

    def test_inf_is_rejected(self) -> None:
        waveform = periodic_waveform()
        waveform[10] = np.inf

        with self.assertRaisesRegex(CPPSExtractionError, "NaN or Inf"):
            compute_cpps(waveform)

    def test_exact_digital_silence_is_rejected(self) -> None:
        with self.assertRaisesRegex(CPPSExtractionError, "digital silence"):
            compute_cpps(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def test_audio_shorter_than_analysis_window_is_rejected(self) -> None:
        waveform = periodic_waveform(duration_seconds=0.049)

        with self.assertRaisesRegex(CPPSExtractionError, "too short"):
            compute_cpps(waveform)

    def test_fixed_recipe_regression(self) -> None:
        result = compute_cpps(periodic_waveform())

        self.assertAlmostEqual(result, 15.719944670468116, places=6)

    def test_fixed_recipe_arguments_are_passed_in_order(self) -> None:
        power_cepstrogram = object()
        with patch(
            "HarmonicAnalysis.call",
            side_effect=[power_cepstrogram, 12.5],
        ) as praat_call:
            result = compute_cpps(periodic_waveform())

        self.assertEqual(result, 12.5)
        self.assertEqual(
            praat_call.call_args_list,
            [
                mock_call(
                    ANY,
                    "To PowerCepstrogram",
                    CPPS_PITCH_FLOOR_HZ,
                    CPPS_TIME_STEP_S,
                    CPPS_MAX_FREQUENCY_HZ,
                    CPPS_PREEMPHASIS_FROM_HZ,
                ),
                mock_call(
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
                ),
            ],
        )

    def test_praat_failures_are_wrapped(self) -> None:
        with patch(
            "HarmonicAnalysis.call",
            side_effect=parselmouth.PraatError("deliberate failure"),
        ):
            with self.assertRaisesRegex(
                CPPSExtractionError, "Praat CPPS extraction failed"
            ) as context:
                compute_cpps(periodic_waveform())

        self.assertIsInstance(context.exception.__cause__, parselmouth.PraatError)

    def test_nonfinite_praat_output_is_rejected(self) -> None:
        with patch(
            "HarmonicAnalysis.call",
            side_effect=[object(), np.nan],
        ):
            with self.assertRaisesRegex(CPPSExtractionError, "nonfinite CPPS"):
                compute_cpps(periodic_waveform())


if __name__ == "__main__":
    unittest.main()
