from __future__ import annotations

import os
import subprocess
import sys
import unittest

import numpy as np

from NoveltySplice import (
    NoveltySpliceError,
    NoveltySpliceResult,
    cosine_self_similarity,
    detect_embedding_novelty,
    novelty_splice_features,
)
from evaluation.analyze_splice_features import (
    analyze_splice_feature_distributions,
)


DIMENSION = 16
HOP_SECONDS = 0.2
FRAME_OFFSET_SECONDS = 0.4
VECTOR_A = np.eye(1, DIMENSION, 0, dtype=np.float64)[0]
VECTOR_B = np.eye(1, DIMENSION, 1, dtype=np.float64)[0]


def frame_times(n_windows: int) -> np.ndarray:
    return FRAME_OFFSET_SECONDS + HOP_SECONDS * np.arange(n_windows)


def around(
    vector: np.ndarray,
    n_windows: int,
    *,
    seed: int,
    noise: float = 0.01,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return vector + rng.normal(0.0, noise, (n_windows, vector.size))


def constant_region() -> np.ndarray:
    return around(VECTOR_A, 30, seed=1)


def abrupt_a_to_b() -> np.ndarray:
    return np.vstack(
        (around(VECTOR_A, 15, seed=2), around(VECTOR_B, 15, seed=3))
    )


def gradual_a_to_b() -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, 30)[:, np.newaxis]
    drift = (1.0 - alpha) * VECTOR_A + alpha * VECTOR_B
    return drift + np.random.default_rng(4).normal(0.0, 0.01, drift.shape)


def a_to_b_to_a() -> np.ndarray:
    return np.vstack(
        (
            around(VECTOR_A, 12, seed=5),
            around(VECTOR_B, 12, seed=6),
            around(VECTOR_A, 12, seed=7),
        )
    )


def crossfade_a_to_b() -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, 8)[1:-1, np.newaxis]
    transition = (1.0 - alpha) * VECTOR_A + alpha * VECTOR_B
    transition += np.random.default_rng(8).normal(0.0, 0.01, transition.shape)
    return np.vstack(
        (
            around(VECTOR_A, 10, seed=9),
            transition,
            around(VECTOR_B, 10, seed=10),
        )
    )


def phonetic_like_trajectory() -> np.ndarray:
    n_windows = 30
    indices = np.arange(n_windows, dtype=np.float64)
    embeddings = np.zeros((n_windows, DIMENSION), dtype=np.float64)
    embeddings[:, 0] = 1.0
    embeddings[:, 1] = 0.15 * np.sin(0.9 * indices)
    embeddings[:, 2] = 0.15 * np.cos(0.9 * indices)
    embeddings += np.random.default_rng(11).normal(0.0, 0.005, embeddings.shape)
    return embeddings


class NoveltySpliceTests(unittest.TestCase):
    def test_import_is_side_effect_free(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", "import NoveltySplice"],
            capture_output=True,
            text=True,
            timeout=10,
            env=os.environ.copy(),
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")

    def test_self_similarity_is_valid_cosine_matrix(self) -> None:
        embeddings = around(VECTOR_A, 12, seed=12, noise=0.05)

        similarity = cosine_self_similarity(embeddings)

        self.assertEqual(similarity.shape, (12, 12))
        self.assertTrue(np.isfinite(similarity).all())
        np.testing.assert_allclose(similarity, similarity.T, atol=1e-14)
        np.testing.assert_allclose(np.diag(similarity), 1.0, atol=1e-14)
        self.assertGreaterEqual(float(np.min(similarity)), -1.0)
        self.assertLessEqual(float(np.max(similarity)), 1.0)

    def test_constant_noisy_region_has_low_novelty(self) -> None:
        embeddings = constant_region()

        result = detect_embedding_novelty(embeddings, frame_times(len(embeddings)))

        self.assertIsInstance(result, NoveltySpliceResult)
        self.assertLess(result.max_novelty, 0.01)
        self.assertEqual(result.n_candidates, 0)

    def test_clear_two_region_change_peaks_near_boundary(self) -> None:
        embeddings = abrupt_a_to_b()
        expected_boundary = 3.3

        result = detect_embedding_novelty(embeddings, frame_times(len(embeddings)))

        self.assertGreater(result.max_novelty, 0.8)
        strongest = max(result.candidates, key=lambda candidate: candidate.score)
        self.assertAlmostEqual(strongest.time_sec, expected_boundary, delta=0.21)

    def test_gradual_drift_is_weaker_and_broader_than_abrupt_change(self) -> None:
        abrupt_embeddings = abrupt_a_to_b()
        gradual_embeddings = gradual_a_to_b()
        abrupt = detect_embedding_novelty(
            abrupt_embeddings, frame_times(len(abrupt_embeddings))
        )
        gradual = detect_embedding_novelty(
            gradual_embeddings, frame_times(len(gradual_embeddings))
        )

        self.assertLess(gradual.max_novelty, 0.1 * abrupt.max_novelty)
        abrupt_width = np.count_nonzero(
            abrupt.novelty_curve > 0.5 * abrupt.max_novelty
        )
        gradual_width = np.count_nonzero(
            gradual.novelty_curve > 0.5 * gradual.max_novelty
        )
        self.assertGreater(gradual_width, abrupt_width)

    def test_a_to_b_to_a_produces_two_boundaries(self) -> None:
        embeddings = a_to_b_to_a()
        expected_boundaries = (2.7, 5.1)

        result = detect_embedding_novelty(embeddings, frame_times(len(embeddings)))

        self.assertGreaterEqual(result.n_candidates, 2)
        candidate_times = [candidate.time_sec for candidate in result.candidates]
        for expected in expected_boundaries:
            self.assertTrue(
                any(abs(actual - expected) <= 0.21 for actual in candidate_times)
            )

    def test_small_consistent_noise_does_not_create_large_boundary(self) -> None:
        low_noise = around(VECTOR_A, 30, seed=13, noise=0.005)
        moderate_noise = around(VECTOR_A, 30, seed=14, noise=0.03)

        low = detect_embedding_novelty(low_noise, frame_times(len(low_noise)))
        moderate = detect_embedding_novelty(
            moderate_noise, frame_times(len(moderate_noise))
        )

        self.assertLess(low.max_novelty, 0.01)
        self.assertLess(moderate.max_novelty, 0.05)
        self.assertEqual(moderate.n_candidates, 0)

    def test_crossfade_like_transition_remains_elevated(self) -> None:
        embeddings = crossfade_a_to_b()

        result = detect_embedding_novelty(embeddings, frame_times(len(embeddings)))

        self.assertGreater(result.max_novelty, 0.15)
        self.assertGreaterEqual(result.n_candidates, 1)
        self.assertAlmostEqual(result.strongest_time_sec, 2.9, delta=0.41)

    def test_phonetic_like_variation_has_no_dominant_peak(self) -> None:
        embeddings = phonetic_like_trajectory()

        result = detect_embedding_novelty(embeddings, frame_times(len(embeddings)))

        self.assertLess(result.max_novelty, 0.05)
        self.assertEqual(result.n_candidates, 0)

    def test_feature_output_is_finite_stable_and_time_shift_independent(self) -> None:
        embeddings = abrupt_a_to_b()
        times = frame_times(len(embeddings))
        first_result = detect_embedding_novelty(embeddings, times)
        second_result = detect_embedding_novelty(embeddings.copy(), times + 100.0)

        first = novelty_splice_features(first_result)
        second = novelty_splice_features(second_result)

        self.assertEqual(
            list(first),
            [
                "splice_novelty_max",
                "splice_novelty_top3_mean",
                "splice_novelty_p95",
                "splice_novelty_n_peaks",
            ],
        )
        self.assertEqual(first, second)
        self.assertTrue(all(np.isfinite(value) for value in first.values()))
        np.testing.assert_array_equal(
            first_result.novelty_curve, second_result.novelty_curve
        )

    def test_very_short_sequence_returns_finite_no_evidence_result(self) -> None:
        embeddings = around(VECTOR_A, 3, seed=15)

        result = detect_embedding_novelty(embeddings, frame_times(3))
        features = novelty_splice_features(result)

        self.assertEqual(result.n_candidates, 0)
        self.assertEqual(result.max_novelty, 0.0)
        self.assertIsNone(result.strongest_time_sec)
        self.assertIn("not enough", result.note)
        self.assertTrue(all(np.isfinite(value) for value in features.values()))

    def test_invalid_embeddings_are_rejected(self) -> None:
        valid = around(VECTOR_A, 8, seed=16)
        invalid_cases = {
            "wrong dimensionality": valid[:, 0],
            "empty": np.empty((0, DIMENSION)),
            "NaN": np.where(np.eye(8, DIMENSION, dtype=bool), np.nan, valid),
            "Inf": np.where(np.eye(8, DIMENSION, dtype=bool), np.inf, valid),
            "zero norm": np.vstack((np.zeros(DIMENSION), valid[1:])),
        }
        for name, embeddings in invalid_cases.items():
            with self.subTest(name=name):
                with self.assertRaises(NoveltySpliceError):
                    detect_embedding_novelty(embeddings, frame_times(embeddings.shape[0]))

    def test_invalid_times_scales_and_local_scores_are_rejected(self) -> None:
        embeddings = around(VECTOR_A, 8, seed=17)
        with self.assertRaisesRegex(NoveltySpliceError, "frame_times"):
            detect_embedding_novelty(embeddings, frame_times(7))
        with self.assertRaisesRegex(NoveltySpliceError, "strictly increasing"):
            detect_embedding_novelty(embeddings, np.zeros(8))
        with self.assertRaisesRegex(NoveltySpliceError, "positive integers"):
            detect_embedding_novelty(
                embeddings, frame_times(8), context_scales=(0, 2)
            )
        with self.assertRaisesRegex(NoveltySpliceError, "local_fake_scores"):
            detect_embedding_novelty(
                embeddings,
                frame_times(8),
                local_fake_scores=np.full(8, np.nan),
            )

    def test_local_fake_score_transition_features(self) -> None:
        embeddings = abrupt_a_to_b()
        local_scores = np.array([0.1] * 14 + [0.8, 0.9] + [0.95] * 14)

        result = detect_embedding_novelty(
            embeddings,
            frame_times(len(embeddings)),
            local_fake_scores=local_scores,
        )
        features = novelty_splice_features(result)

        self.assertEqual(features["local_wavlm_fake_max"], 0.95)
        self.assertEqual(features["local_wavlm_fake_min"], 0.1)
        self.assertAlmostEqual(features["local_wavlm_fake_range"], 0.85)
        self.assertAlmostEqual(features["local_wavlm_fake_max_jump"], 0.7)
        self.assertEqual(features["local_wavlm_fake_p90"], 0.95)

    def test_distribution_analysis_reports_novelty_and_physical_features(self) -> None:
        records = [
            {
                "file_id": "real.wav",
                "label": "bonafide",
                "embeddings": constant_region(),
                "frame_times": frame_times(30),
                "splice_max_score": 0.1,
            },
            {
                "file_id": "partial.wav",
                "label": "partial",
                "embeddings": abrupt_a_to_b(),
                "frame_times": frame_times(30),
                "splice_max_score": 0.6,
            },
        ]

        per_file, summary = analyze_splice_feature_distributions(records)

        self.assertEqual(per_file["file_id"].tolist(), ["real.wav", "partial.wav"])
        self.assertIn("splice_novelty_max", per_file)
        self.assertIn("splice_max_score", per_file)
        self.assertEqual(set(summary["label"]), {"bonafide", "partial"})
        self.assertTrue(
            np.isfinite(
                summary[["mean", "std", "min", "median", "p95", "max"]]
            ).all().all()
        )


if __name__ == "__main__":
    unittest.main()
