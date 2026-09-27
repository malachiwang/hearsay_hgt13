"""Fixed baseline model definitions for HEARSAY score fusion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOGISTIC_BASELINE_PARAMS: dict[str, Any] = {
    "l1_ratio": 0.0,
    "C": 1.0,
    "max_iter": 2000,
    "random_state": 42,
}

LIGHTGBM_BASELINE_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "n_estimators": 200,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 1,
}


class FusionModelError(ValueError):
    """Raised when a fusion model name or fitted model is invalid."""


def create_logistic_model() -> Pipeline:
    """Return the fixed linear baseline with fold-local feature scaling."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**LOGISTIC_BASELINE_PARAMS)),
        ]
    )


def create_lightgbm_model():
    """Return the fixed, deliberately small nonlinear baseline."""

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise FusionModelError(
            "LightGBM fusion requires the 'lightgbm' package"
        ) from exc
    return LGBMClassifier(**LIGHTGBM_BASELINE_PARAMS)


def create_fusion_model(model_name: str):
    """Create a fresh fusion estimator for one training fold."""

    normalized = model_name.strip().lower()
    if normalized in {"logistic", "logistic_regression"}:
        return create_logistic_model()
    if normalized == "lightgbm":
        return create_lightgbm_model()
    raise FusionModelError(
        f"unsupported fusion model {model_name!r}; expected 'logistic' or 'lightgbm'"
    )


def lightgbm_gain_importance(
    fitted_model: object,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Return gain importance; it is diagnostic, not causal evidence."""

    booster = getattr(fitted_model, "booster_", None)
    if booster is None:
        raise FusionModelError("LightGBM model must be fitted before reading importance")
    gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    if gains.shape != (len(feature_names),):
        raise FusionModelError("LightGBM importance length does not match feature names")
    return pd.DataFrame({"feature": list(feature_names), "gain_importance": gains})
