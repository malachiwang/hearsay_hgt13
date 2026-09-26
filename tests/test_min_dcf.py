from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.evaluate_scores import (
    ScoreTableError,
    join_predictions_and_labels,
    load_evaluation_tables,
    main,
    read_labels,
    read_predictions,
)
from evaluation.min_dcf import (
    C_FA,
    C_MISS,
    P_SPOOF,
    MinDCFError,
    compute_det_curve,
    compute_min_dcf,
    evaluate_min_dcf,
)


def asvspoof5_reference(
    synthetic_scores: np.ndarray,
    labels: list[str],
) -> tuple[float, float, float, float, float]:
    """Small local transcription of the ASVspoof5 Track 1 calculation."""

    bona_fide_scores = -synthetic_scores[np.asarray(labels) == "bonafide"]
    spoof_scores = -synthetic_scores[np.asarray(labels) == "spoof"]

    all_scores = np.concatenate((bona_fide_scores, spoof_scores))
    class_labels = np.concatenate(
        (np.ones(bona_fide_scores.size), np.zeros(spoof_scores.size))
    )
    indices = np.argsort(all_scores, kind="mergesort")
    sorted_scores = all_scores[indices]
    sorted_labels = class_labels[indices]
    target_sums = np.cumsum(sorted_labels)
    nontarget_sums = spoof_scores.size - (
        np.arange(1, all_scores.size + 1) - target_sums
    )
    p_miss = np.concatenate(([0.0], target_sums / bona_fide_scores.size))
    p_fa = np.concatenate(([1.0], nontarget_sums / spoof_scores.size))
    thresholds = np.concatenate(([sorted_scores[0] - 0.001], sorted_scores))

    p_target = 1.0 - 0.30
    normalization = min(1.0 * p_target, 4.0 * (1.0 - p_target))
    best_cost = np.inf
    best_index = 0
    for index in range(p_miss.size):
        cost = 1.0 * p_miss[index] * p_target + 4.0 * p_fa[index] * (
            1.0 - p_target
        )
        if cost < best_cost:
            best_cost = cost
            best_index = index

    return (
        float(best_cost / normalization),
        float(-thresholds[best_index]),
        float(p_miss[best_index]),
        float(p_fa[best_index]),
        float(best_cost),
    )


class MinDCFTests(unittest.TestCase):
    def test_perfect_separation(self) -> None:
        result = evaluate_min_dcf(
            [0.10, 0.20, 0.80, 0.90],
            ["bonafide", "bonafide", "spoof", "spoof"],
        )

        self.assertEqual(result.min_dcf, 0.0)
        self.assertEqual(result.p_miss, 0.0)
        self.assertEqual(result.p_fa, 0.0)
        self.assertEqual(result.number_bonafide, 2)
        self.assertEqual(result.number_spoof, 2)

    def test_score_polarity_is_synthetic_oriented(self) -> None:
        correct = evaluate_min_dcf(
            [0.10, 0.20, 0.80, 0.90],
            ["bonafide", "bonafide", "spoof", "spoof"],
        )
        reversed_order = evaluate_min_dcf(
            [0.80, 0.90, 0.10, 0.20],
            ["bonafide", "bonafide", "spoof", "spoof"],
        )

        self.assertEqual(correct.min_dcf, 0.0)
        self.assertAlmostEqual(reversed_order.min_dcf, 1.0)

    def test_default_cost_constants_are_nsa_values(self) -> None:
        self.assertEqual(P_SPOOF, 0.30)
        self.assertEqual(C_MISS, 1.0)
        self.assertEqual(C_FA, 4.0)

        point = compute_min_dcf([0.2], [0.1], [0.4])
        expected_cost = 0.70 * 0.2 + 1.20 * 0.1
        self.assertAlmostEqual(point.unnormalized_cost, expected_cost)
        self.assertAlmostEqual(point.min_dcf, expected_cost / 0.70)

    def test_manual_small_example(self) -> None:
        scores = [0.1, 0.4, 0.3, 0.8]
        labels = ["bonafide", "bonafide", "spoof", "spoof"]

        p_miss, p_fa, thresholds = compute_det_curve(
            [-0.1, -0.4],
            [-0.3, -0.8],
        )
        np.testing.assert_allclose(p_miss, [0.0, 0.0, 0.5, 0.5, 1.0])
        np.testing.assert_allclose(p_fa, [1.0, 0.5, 0.5, 0.0, 0.0])
        np.testing.assert_allclose(thresholds, [-0.801, -0.8, -0.4, -0.3, -0.1])

        result = evaluate_min_dcf(scores, labels)
        self.assertAlmostEqual(result.min_dcf, 0.5)
        self.assertAlmostEqual(result.threshold, 0.3)
        self.assertAlmostEqual(result.p_miss, 0.5)
        self.assertAlmostEqual(result.p_fa, 0.0)

    def test_tied_scores_are_deterministic(self) -> None:
        p_miss, p_fa, thresholds = compute_det_curve([0.5, 0.5], [0.5, 0.5])
        np.testing.assert_allclose(p_miss, [0.0, 0.5, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(p_fa, [1.0, 1.0, 1.0, 0.5, 0.0])
        np.testing.assert_allclose(thresholds, [0.499, 0.5, 0.5, 0.5, 0.5])

        first = evaluate_min_dcf(
            [0.5, 0.5, 0.5, 0.5],
            ["bonafide", "bonafide", "spoof", "spoof"],
        )
        second = evaluate_min_dcf(
            [0.5, 0.5, 0.5, 0.5],
            ["bonafide", "bonafide", "spoof", "spoof"],
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.min_dcf, 1.0)

    def test_matches_local_asvspoof5_reference(self) -> None:
        scores = np.asarray([0.62, -1.1, 0.62, 2.4, 0.0, -0.3, 1.2])
        labels = [
            "bonafide",
            "spoof",
            "spoof",
            "spoof",
            "bonafide",
            "bonafide",
            "spoof",
        ]
        expected = asvspoof5_reference(scores, labels)

        result = evaluate_min_dcf(scores, labels)
        actual = (
            result.min_dcf,
            result.threshold,
            result.p_miss,
            result.p_fa,
            result.unnormalized_cost,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)

    def test_scores_need_not_be_probabilities(self) -> None:
        result = evaluate_min_dcf(
            [-20.0, -10.0, 15.0, 100.0],
            ["bonafide", "bonafide", "spoof", "spoof"],
        )
        self.assertEqual(result.min_dcf, 0.0)

    def test_invalid_inputs(self) -> None:
        cases = [
            ([], [], "empty scores"),
            ([0.1], [], "empty labels"),
            ([0.1, 0.2], ["bonafide"], "mismatched lengths"),
            ([0.1, 0.2], ["spoof", "spoof"], "no bona fide"),
            ([0.1, 0.2], ["bonafide", "bonafide"], "no spoof"),
            ([np.nan, 0.2], ["bonafide", "spoof"], "NaN"),
            ([np.inf, 0.2], ["bonafide", "spoof"], "Inf"),
            ([0.1, 0.2], ["real", "spoof"], "unknown label"),
            ([[0.1], [0.2]], ["bonafide", "spoof"], "2-D scores"),
        ]
        for scores, labels, name in cases:
            with self.subTest(name=name):
                with self.assertRaises(MinDCFError):
                    evaluate_min_dcf(scores, labels)


class ScoreTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def write(self, name: str, contents: str) -> Path:
        path = self.directory / name
        path.write_text(contents, encoding="utf-8")
        return path

    def test_join_is_by_id_not_row_order(self) -> None:
        scores_path = self.write("scores.tsv", "id\tscore\na\t0.9\nb\t0.1\n")
        labels_path = self.write(
            "labels.tsv", "id\tlabel\nb\tbonafide\na\tspoof\n"
        )

        scores, labels = load_evaluation_tables(scores_path, labels_path)
        np.testing.assert_allclose(scores, [0.1, 0.9])
        self.assertEqual(labels, ["bonafide", "spoof"])

    def test_duplicate_prediction_ids_are_rejected(self) -> None:
        path = self.write("scores.tsv", "id\tscore\na\t0.1\na\t0.2\n")
        with self.assertRaisesRegex(ScoreTableError, "duplicate prediction ID"):
            read_predictions(path)

    def test_duplicate_label_ids_are_rejected(self) -> None:
        path = self.write(
            "labels.tsv", "id\tlabel\na\tbonafide\na\tspoof\n"
        )
        with self.assertRaisesRegex(ScoreTableError, "duplicate label ID"):
            read_labels(path)

    def test_missing_prediction_is_rejected(self) -> None:
        with self.assertRaisesRegex(ScoreTableError, "missing predictions"):
            join_predictions_and_labels(
                {"a": 0.1},
                {"a": "bonafide", "b": "spoof"},
            )

    def test_extra_prediction_is_rejected_unless_allowed(self) -> None:
        predictions = {"a": 0.1, "b": 0.9, "extra": 0.5}
        labels = {"a": "bonafide", "b": "spoof"}
        with self.assertRaisesRegex(ScoreTableError, "unexpected prediction IDs"):
            join_predictions_and_labels(predictions, labels)

        scores, joined_labels = join_predictions_and_labels(
            predictions,
            labels,
            allow_extra_predictions=True,
        )
        np.testing.assert_allclose(scores, [0.1, 0.9])
        self.assertEqual(joined_labels, ["bonafide", "spoof"])

    def test_nonnumeric_and_nonfinite_scores_are_rejected(self) -> None:
        for value in ("not-a-number", "nan", "inf"):
            with self.subTest(value=value):
                path = self.write("scores.tsv", f"id\tscore\na\t{value}\n")
                with self.assertRaises(ScoreTableError):
                    read_predictions(path)

    def test_explicit_columns_and_csv_delimiter(self) -> None:
        scores_path = self.write("scores.csv", "file,prediction\na,0.1\nb,0.9\n")
        labels_path = self.write("labels.csv", "file,class\nb,spoof\na,bonafide\n")

        scores, labels = load_evaluation_tables(
            scores_path,
            labels_path,
            id_column="file",
            score_column="prediction",
            label_column="class",
            delimiter=",",
        )
        np.testing.assert_allclose(scores, [0.9, 0.1])
        self.assertEqual(labels, ["spoof", "bonafide"])

    def test_cli_prints_required_summary(self) -> None:
        scores_path = self.write("scores.tsv", "id\tscore\na\t0.1\nb\t0.9\n")
        labels_path = self.write(
            "labels.tsv", "id\tlabel\na\tbonafide\nb\tspoof\n"
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(
                ["--scores", str(scores_path), "--labels", str(labels_path)]
            )

        self.assertEqual(exit_code, 0)
        report = output.getvalue()
        self.assertIn("HEARSAY minDCF: 0.000000", report)
        self.assertIn("Best synthetic-score threshold:", report)
        self.assertIn("Pmiss (real -> spoof):", report)
        self.assertIn("Pfa   (spoof -> real):", report)
        self.assertIn("Bona fide: 1", report)
        self.assertIn("Spoof: 1", report)


if __name__ == "__main__":
    unittest.main()
