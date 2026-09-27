from __future__ import annotations

import unittest

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fusion.models import (
    LIGHTGBM_BASELINE_PARAMS,
    LOGISTIC_BASELINE_PARAMS,
    FusionModelError,
    create_fusion_model,
    create_lightgbm_model,
    create_logistic_model,
    lightgbm_gain_importance,
)


class FusionModelTests(unittest.TestCase):
    def test_logistic_baseline_contains_fold_local_scaler(self) -> None:
        model = create_logistic_model()

        self.assertIsInstance(model, Pipeline)
        self.assertEqual(list(model.named_steps), ["scaler", "classifier"])
        self.assertIsInstance(model.named_steps["scaler"], StandardScaler)
        classifier_parameters = model.named_steps["classifier"].get_params()
        self.assertEqual(classifier_parameters["C"], 1.0)
        self.assertEqual(classifier_parameters["l1_ratio"], 0.0)
        self.assertEqual(LOGISTIC_BASELINE_PARAMS["max_iter"], 2000)

    def test_lightgbm_baseline_is_small_fixed_and_deterministic(self) -> None:
        model = create_lightgbm_model()
        parameters = model.get_params()

        for name, expected in LIGHTGBM_BASELINE_PARAMS.items():
            self.assertEqual(parameters[name], expected)
        self.assertEqual(parameters["n_estimators"], 200)
        self.assertEqual(parameters["num_leaves"], 7)
        self.assertEqual(parameters["max_depth"], 3)
        self.assertTrue(parameters["deterministic"])

    def test_lightgbm_gain_importance_has_one_row_per_feature(self) -> None:
        rng = np.random.default_rng(9)
        labels = np.tile([0, 1], 60)
        features = np.column_stack(
            (labels + rng.normal(0.0, 0.05, labels.size), rng.normal(size=labels.size))
        )
        model = create_lightgbm_model().fit(features, labels)

        importance = lightgbm_gain_importance(model, ["signal", "noise"])

        self.assertEqual(importance["feature"].tolist(), ["signal", "noise"])
        self.assertTrue(np.isfinite(importance["gain_importance"]).all())
        self.assertGreater(importance.loc[0, "gain_importance"], 0.0)

    def test_model_factory_rejects_unknown_model(self) -> None:
        with self.assertRaisesRegex(FusionModelError, "unsupported fusion model"):
            create_fusion_model("neural-mega-fusion")


if __name__ == "__main__":
    unittest.main()
