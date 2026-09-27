"""Fold-based out-of-fold diagnostics for HEARSAY fusion models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from evaluation.min_dcf import MinDCFResult, evaluate_min_dcf
from fusion.build_features import (
    FILE_ID_COLUMN,
    FOLD_COLUMN,
    LABEL_COLUMN,
    METADATA_COLUMNS,
    FusionDataError,
)
from fusion.models import create_fusion_model, lightgbm_gain_importance


@dataclass(frozen=True)
class FusionCVResult:
    """Concatenated held-out predictions and their canonical NSA metric."""

    predictions: pd.DataFrame
    model_name: str
    feature_names: tuple[str, ...]
    number_files: int
    number_folds: int
    metric: MinDCFResult
    feature_importance: pd.DataFrame | None = None
    evaluation_kind: str = "OOF stacking diagnostic"

    @property
    def min_dcf(self) -> float:
        return self.metric.min_dcf

    @property
    def threshold(self) -> float:
        return self.metric.threshold

    @property
    def p_miss(self) -> float:
        return self.metric.p_miss

    @property
    def p_fa(self) -> float:
        return self.metric.p_fa


def _validate_cv_table(
    feature_table: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...] | None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if not isinstance(feature_table, pd.DataFrame):
        raise FusionDataError("feature_table must be a pandas DataFrame")
    table = feature_table.copy()
    duplicated = table.columns[table.columns.duplicated()].tolist()
    if duplicated:
        raise FusionDataError(f"feature_table has duplicate columns: {duplicated}")

    missing_metadata = [column for column in METADATA_COLUMNS if column not in table]
    if missing_metadata:
        raise FusionDataError(
            "feature_table is missing required columns: "
            + ", ".join(missing_metadata)
        )
    if table.empty:
        raise FusionDataError("feature_table must contain at least one row")
    if table[FILE_ID_COLUMN].isna().any() or table[FILE_ID_COLUMN].duplicated().any():
        raise FusionDataError("feature_table file_id values must be present and unique")
    if table[FOLD_COLUMN].isna().any():
        raise FusionDataError("feature_table contains a missing fold")

    labels = table[LABEL_COLUMN].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise FusionDataError("feature_table labels must be canonical 0 or 1")
    if np.unique(labels).size != 2:
        raise FusionDataError("feature_table must contain both label classes")

    if feature_columns is None:
        selected = tuple(column for column in table.columns if column not in METADATA_COLUMNS)
    else:
        if isinstance(feature_columns, (str, bytes)):
            raise FusionDataError("feature_columns must be a sequence of column names")
        selected = tuple(feature_columns)
    if not selected:
        raise FusionDataError("at least one fusion feature column is required")
    if len(set(selected)) != len(selected):
        raise FusionDataError("feature_columns must not contain duplicates")

    reserved = [column for column in selected if column in METADATA_COLUMNS]
    if reserved:
        raise FusionDataError(
            "metadata columns cannot be fusion features: " + ", ".join(reserved)
        )
    unknown = [column for column in selected if column not in table]
    if unknown:
        raise FusionDataError("unknown feature columns: " + ", ".join(unknown))

    for column in selected:
        values = table[column]
        if not pd.api.types.is_numeric_dtype(values.dtype):
            raise FusionDataError(f"feature column {column!r} must be numeric")
        if not np.isfinite(values.to_numpy(dtype=np.float64, copy=False)).all():
            raise FusionDataError(f"feature column {column!r} contains NaN or Inf")
    return table, selected


def _synthetic_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_)
    class_indices = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or class_indices.size != 1:
        raise FusionDataError("fusion model did not expose one probability for class 1")
    scores = probabilities[:, int(class_indices[0])]
    if not np.isfinite(scores).all():
        raise FusionDataError("fusion model produced NaN or Inf scores")
    return scores


def cross_validate_fusion(
    feature_table: pd.DataFrame,
    *,
    model_name: str = "logistic",
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> FusionCVResult:
    """Generate one held-out synthetic score per row using manifest folds.

    This is a standard OOF stacking diagnostic. Strict nested evaluation must
    supply outer-fold-specific learned-detector features for which the outer
    validation fold was excluded from *all* base-model fitting.
    """

    table, selected = _validate_cv_table(feature_table, feature_columns)
    folds = list(pd.unique(table[FOLD_COLUMN]))
    if len(folds) < 2:
        raise FusionDataError("cross-validation requires at least two folds")

    feature_matrix = table.loc[:, list(selected)].astype(np.float64)
    labels = table[LABEL_COLUMN].to_numpy(dtype=np.int8)
    oof_scores = np.full(len(table), np.nan, dtype=np.float64)
    lightgbm_importances: list[np.ndarray] = []
    normalized_model_name = model_name.strip().lower()

    for fold in folds:
        validation_mask = table[FOLD_COLUMN].eq(fold).to_numpy()
        training_mask = ~validation_mask
        training_labels = labels[training_mask]
        if np.unique(training_labels).size != 2:
            raise FusionDataError(
                f"training rows excluding fold {fold!r} do not contain both classes"
            )

        # A new estimator is created here so every scaler/model is fitted only
        # on this fold's training rows.
        model = create_fusion_model(normalized_model_name)
        model.fit(feature_matrix.loc[training_mask], training_labels)
        oof_scores[validation_mask] = _synthetic_probability(
            model, feature_matrix.loc[validation_mask]
        )

        if normalized_model_name == "lightgbm":
            importance = lightgbm_gain_importance(model, selected)
            lightgbm_importances.append(
                importance["gain_importance"].to_numpy(dtype=np.float64)
            )

    if not np.isfinite(oof_scores).all():
        raise FusionDataError("not every row received one finite held-out prediction")

    metric_labels = np.where(labels == 1, "spoof", "bonafide")
    metric = evaluate_min_dcf(oof_scores, metric_labels)
    predictions = table.loc[:, [FILE_ID_COLUMN, FOLD_COLUMN, LABEL_COLUMN]].copy()
    predictions["synthetic_score"] = oof_scores

    feature_importance = None
    if lightgbm_importances:
        fold_gains = np.vstack(lightgbm_importances)
        feature_importance = pd.DataFrame(
            {
                "feature": list(selected),
                "mean_gain_importance": fold_gains.mean(axis=0),
                "total_gain_importance": fold_gains.sum(axis=0),
            }
        ).sort_values(
            ["mean_gain_importance", "feature"],
            ascending=[False, True],
            ignore_index=True,
        )

    return FusionCVResult(
        predictions=predictions,
        model_name=normalized_model_name,
        feature_names=selected,
        number_files=len(table),
        number_folds=len(folds),
        metric=metric,
        feature_importance=feature_importance,
    )
