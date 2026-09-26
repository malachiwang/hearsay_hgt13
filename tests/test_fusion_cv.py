from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluation.min_dcf import evaluate_min_dcf
from fusion.build_features import FusionDataError
from fusion.cross_validate import cross_validate_fusion
from fusion.models import create_logistic_model


def make_feature_table() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    labels = np.tile([0, 1], 90)
    folds = np.tile([2, 5, 9], 60)
    signal = np.where(labels == 1, 2.0, -2.0) + rng.normal(0.0, 0.08, labels.size)
    return pd.DataFrame(
        {
            "file_id": [f"clip_{index:03d}.wav" for index in range(labels.size)],
            "label": labels,
            "fold": folds,
            "signal": signal,
            "noise": rng.normal(0.0, 1.0, labels.size),
            "unused": rng.normal(0.0, 1.0, labels.size),
        }
    )


class FusionCrossValidationTests(unittest.TestCase):
    def test_logistic_cv_returns_one_synthetic_score_per_file(self) -> None:
        table = make_feature_table()

        result = cross_validate_fusion(
            table,
            model_name="logistic",
            feature_columns=["signal", "noise"],
        )

        self.assertEqual(result.number_files, len(table))
        self.assertEqual(result.number_folds, 3)
        self.assertEqual(result.evaluation_kind, "OOF stacking diagnostic")
        self.assertEqual(result.predictions["file_id"].tolist(), table["file_id"].tolist())
        scores = result.predictions["synthetic_score"].to_numpy()
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0.0) & (scores <= 1.0)).all())
        self.assertGreater(scores[table["label"].eq(1)].mean(), scores[table["label"].eq(0)].mean())
        direct_metric = evaluate_min_dcf(
            scores, np.where(table["label"].eq(1), "spoof", "bonafide")
        )
        self.assertEqual(result.metric, direct_metric)
        self.assertEqual(result.min_dcf, 0.0)

    def test_scaler_is_created_and_fit_inside_each_training_fold(self) -> None:
        table = make_feature_table()
        models = []

        def tracking_factory(_model_name: str):
            model = create_logistic_model()
            models.append(model)
            return model

        with patch(
            "fusion.cross_validate.create_fusion_model", side_effect=tracking_factory
        ):
            cross_validate_fusion(
                table, model_name="logistic", feature_columns=["signal", "noise"]
            )

        self.assertEqual(len(models), 3)
        expected_training_rows = 120
        scaler_ids = set()
        for model in models:
            scaler = model.named_steps["scaler"]
            scaler_ids.add(id(scaler))
            self.assertEqual(int(scaler.n_samples_seen_), expected_training_rows)
        self.assertEqual(len(scaler_ids), 3)

    def test_manifest_fold_assignments_are_preserved(self) -> None:
        table = make_feature_table()
        result = cross_validate_fusion(
            table, model_name="logistic", feature_columns=["signal"]
        )

        self.assertEqual(result.predictions["fold"].tolist(), table["fold"].tolist())
        self.assertEqual(set(result.predictions["fold"]), {2, 5, 9})

    def test_explicit_feature_subset_excludes_other_columns(self) -> None:
        table = make_feature_table()
        models = []

        def tracking_factory(_model_name: str):
            model = create_logistic_model()
            models.append(model)
            return model

        with patch(
            "fusion.cross_validate.create_fusion_model", side_effect=tracking_factory
        ):
            result = cross_validate_fusion(
                table,
                model_name="logistic",
                feature_columns=["signal", "noise"],
            )

        self.assertEqual(result.feature_names, ("signal", "noise"))
        self.assertTrue(all(model.n_features_in_ == 2 for model in models))

    def test_lightgbm_cv_is_deterministic_and_exposes_gain_importance(self) -> None:
        table = make_feature_table()

        first = cross_validate_fusion(
            table, model_name="lightgbm", feature_columns=["signal", "noise"]
        )
        second = cross_validate_fusion(
            table, model_name="lightgbm", feature_columns=["signal", "noise"]
        )

        np.testing.assert_array_equal(
            first.predictions["synthetic_score"],
            second.predictions["synthetic_score"],
        )
        self.assertTrue(np.isfinite(first.predictions["synthetic_score"]).all())
        self.assertGreater(
            first.predictions.loc[table["label"].eq(1), "synthetic_score"].mean(),
            first.predictions.loc[table["label"].eq(0), "synthetic_score"].mean(),
        )
        self.assertEqual(first.min_dcf, 0.0)
        self.assertIsNotNone(first.feature_importance)
        self.assertEqual(set(first.feature_importance["feature"]), {"signal", "noise"})
        self.assertTrue(np.isfinite(first.feature_importance["mean_gain_importance"]).all())

    def test_fold_that_leaves_one_training_class_is_rejected(self) -> None:
        table = pd.DataFrame(
            {
                "file_id": ["a", "b", "c", "d"],
                "label": [0, 0, 1, 1],
                "fold": [2, 2, 5, 5],
                "score": [0.1, 0.2, 0.8, 0.9],
            }
        )
        with self.assertRaisesRegex(FusionDataError, "do not contain both classes"):
            cross_validate_fusion(table)

    def test_metadata_cannot_be_selected_as_a_feature(self) -> None:
        with self.assertRaisesRegex(FusionDataError, "metadata columns"):
            cross_validate_fusion(make_feature_table(), feature_columns=["label"])


if __name__ == "__main__":
    unittest.main()
