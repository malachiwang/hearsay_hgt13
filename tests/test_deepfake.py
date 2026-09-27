from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import NewAttempt.Deepfake as deepfake
from NewAttempt.Deepfake import (
    EliyaDetector,
    aggregate_window_scores,
    ensure_model,
    run_inference,
)


class FakeEliyaModel(torch.nn.Module):
    def __init__(self, fake_scores: list[float] | None = None) -> None:
        super().__init__()
        self.fake_scores = fake_scores
        self.score_offset = 0
        self.forward_calls = 0
        self.batch_shapes: list[tuple[int, ...]] = []
        self.batch_inputs: list[torch.Tensor] = []

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.batch_shapes.append(tuple(batch.shape))
        self.batch_inputs.append(batch.detach().cpu().clone())
        if self.fake_scores is None:
            fake = torch.full((batch.shape[0],), 0.25, device=batch.device)
        else:
            next_offset = self.score_offset + batch.shape[0]
            fake = torch.tensor(
                self.fake_scores[self.score_offset : next_offset],
                dtype=torch.float32,
                device=batch.device,
            )
            self.score_offset = next_offset
        bona_fide = 1.0 - fake
        return torch.log(bona_fide / fake)


class CountingAudioLoader:
    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = duration_seconds
        self.calls = 0
        self.last_waveform: torch.Tensor | None = None

    def __call__(self, _path: Path) -> torch.Tensor:
        self.calls += 1
        sample_count = round(self.duration_seconds * deepfake.SAMPLE_RATE)
        self.last_waveform = torch.linspace(
            -0.5, 0.5, sample_count, dtype=torch.float32
        )
        return self.last_waveform


def detector_for_duration(
    duration_seconds: float,
    *,
    fake_scores: list[float] | None = None,
    batch_size: int = 4,
) -> tuple[EliyaDetector, FakeEliyaModel, CountingAudioLoader]:
    model = FakeEliyaModel(fake_scores)
    loader = CountingAudioLoader(duration_seconds)
    detector = EliyaDetector(
        model=model,
        audio_loader=loader,
        device="cpu",
        batch_size=batch_size,
    )
    return detector, model, loader


class EliyaWindowingTests(unittest.TestCase):
    def test_two_second_input_produces_one_repeat_padded_window(self) -> None:
        detector, model, loader = detector_for_duration(2.0)

        result = detector.analyze_file("short.wav")

        self.assertEqual(loader.calls, 1)
        self.assertEqual(result["n_windows"], 1)
        self.assertEqual(result["windows"][0]["start"], 0.0)
        self.assertEqual(result["windows"][0]["end"], 2.0)
        self.assertEqual(model.batch_shapes, [(1, deepfake.WINDOW_SAMPLES)])
        self.assertIsNotNone(loader.last_waveform)
        padded = model.batch_inputs[0][0]
        original_size = loader.last_waveform.numel()
        torch.testing.assert_close(padded[:original_size], loader.last_waveform)
        torch.testing.assert_close(
            padded[original_size : 2 * original_size], loader.last_waveform
        )

    def test_exact_five_second_input_produces_one_window(self) -> None:
        detector, _, _ = detector_for_duration(5.0)

        result = detector.analyze_file("exact.wav")

        self.assertEqual(result["n_windows"], 1)
        self.assertEqual(result["windows"][0]["start"], 0.0)
        self.assertEqual(result["windows"][0]["end"], 5.0)

    def test_five_and_a_half_seconds_covers_the_end(self) -> None:
        detector, _, _ = detector_for_duration(5.5)

        result = detector.analyze_file("five_half.wav")

        self.assertEqual(
            [(window["start"], window["end"]) for window in result["windows"]],
            [(0.0, 5.0), (0.5, 5.5)],
        )

    def test_seven_point_two_seconds_adds_end_anchored_window(self) -> None:
        detector, model, _ = detector_for_duration(7.2, batch_size=4)

        result = detector.analyze_file("seven_two.wav")

        self.assertEqual(
            [window["start"] for window in result["windows"]],
            [0.0, 0.5, 1.0, 1.5, 2.0, 2.2],
        )
        self.assertEqual(result["windows"][-1]["end"], 7.2)
        self.assertEqual(model.forward_calls, 2)

    def test_twenty_seconds_uses_every_half_second_start(self) -> None:
        detector, _, _ = detector_for_duration(20.0, batch_size=8)

        result = detector.analyze_file("twenty.wav")

        self.assertEqual(result["n_windows"], 31)
        np.testing.assert_allclose(
            [window["start"] for window in result["windows"]],
            np.arange(0.0, 15.0 + 0.5, 0.5),
        )
        self.assertEqual(result["windows"][-1]["end"], 20.0)

    def test_exact_regular_end_does_not_duplicate_final_window(self) -> None:
        detector, _, _ = detector_for_duration(7.0)

        result = detector.analyze_file("seven.wav")
        starts = [window["start"] for window in result["windows"]]

        self.assertEqual(starts, [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(len(starts), len(set(starts)))


class EliyaScoringTests(unittest.TestCase):
    def test_aggregate_statistics(self) -> None:
        result = aggregate_window_scores(
            [0.1, 0.2, 0.8, 0.9], diagnostic_threshold=0.5
        )

        self.assertAlmostEqual(result["eliya_mean"], 0.5)
        self.assertAlmostEqual(result["eliya_max"], 0.9)
        self.assertAlmostEqual(result["eliya_top3_mean"], (0.2 + 0.8 + 0.9) / 3)
        self.assertAlmostEqual(result["eliya_p90"], 0.87)
        self.assertAlmostEqual(result["eliya_high_window_fraction"], 0.5)

    def test_top3_mean_uses_all_scores_when_fewer_than_three(self) -> None:
        result = aggregate_window_scores([0.1, 0.5])

        self.assertAlmostEqual(result["eliya_top3_mean"], 0.3)

    def test_upstream_bonafide_logit_is_inverted_to_synthetic_score(self) -> None:
        detector, _, _ = detector_for_duration(5.5, fake_scores=[0.1, 0.8])

        result = detector.analyze_file("orientation.wav")
        scores = [window["score"] for window in result["windows"]]

        np.testing.assert_allclose(scores, [0.1, 0.8], atol=1e-6)
        self.assertAlmostEqual(result["fake_probability"], 0.45, places=6)
        self.assertAlmostEqual(result["bonafide_score"], 0.55, places=6)

    def test_all_emitted_scores_and_aggregates_are_probabilities(self) -> None:
        detector, _, _ = detector_for_duration(
            7.2, fake_scores=[0.01, 0.2, 0.4, 0.6, 0.8, 0.99]
        )

        result = detector.analyze_file("bounded.wav")

        probability_fields = (
            "eliya_mean",
            "eliya_max",
            "eliya_top3_mean",
            "eliya_p90",
            "eliya_high_window_fraction",
            "fake_probability",
            "bonafide_score",
        )
        self.assertTrue(
            all(0.0 <= result[field] <= 1.0 for field in probability_fields)
        )
        self.assertTrue(
            all(0.0 <= window["score"] <= 1.0 for window in result["windows"])
        )

    def test_invalid_model_scores_fail_instead_of_being_clipped(self) -> None:
        with self.assertRaisesRegex(deepfake.EliyaInferenceError, "within"):
            aggregate_window_scores([0.2, math.nan])


class EliyaEfficiencyAndCompatibilityTests(unittest.TestCase):
    def test_model_loader_runs_once_and_each_file_is_decoded_once(self) -> None:
        model = FakeEliyaModel()
        loader = CountingAudioLoader(5.5)
        initialization_count = 0

        def model_loader(_model_dir: Path, _device):
            nonlocal initialization_count
            initialization_count += 1
            return model

        with patch.object(deepfake, "ensure_model"):
            detector = EliyaDetector(
                model_loader=model_loader,
                audio_loader=loader,
                device="cpu",
            )
        detector.analyze_file("first.wav")
        detector.analyze_file("second.wav")

        self.assertEqual(initialization_count, 1)
        self.assertEqual(loader.calls, 2)

    def test_original_run_inference_entry_point_is_preserved(self) -> None:
        detector, _, _ = detector_for_duration(5.0)

        result = run_inference(Path("example.wav"), detector)

        self.assertEqual(result["file"], "example.wav")
        self.assertIn("fake_probability", result)
        self.assertIn("bonafide_score", result)
        self.assertIn("verdict", result)
        self.assertIn("windows", result)

    def test_model_download_allowlist_excludes_pickled_checkpoint(self) -> None:
        calls = []

        def snapshot_download(**kwargs):
            calls.append(kwargs)
            target = Path(kwargs["local_dir"])
            for name in kwargs["allow_patterns"]:
                (target / name).write_text("fixture", encoding="utf-8")

        fake_hub = SimpleNamespace(snapshot_download=snapshot_download)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                with patch("builtins.print"):
                    ensure_model(Path(directory))

        self.assertEqual(len(calls), 1)
        self.assertEqual(tuple(calls[0]["allow_patterns"]), deepfake.MODEL_FILES)
        self.assertNotIn("checkpoint_epoch_5.pt", calls[0]["allow_patterns"])


if __name__ == "__main__":
    unittest.main()
