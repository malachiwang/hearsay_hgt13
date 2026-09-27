"""Strict feature-table construction and baseline fusion models for HEARSAY."""

from .build_features import FusionDataError, build_feature_table
from .cross_validate import FusionCVResult, cross_validate_fusion
from .models import (
    LIGHTGBM_BASELINE_PARAMS,
    LOGISTIC_BASELINE_PARAMS,
    FusionModelError,
    create_fusion_model,
    create_lightgbm_model,
    create_logistic_model,
    lightgbm_gain_importance,
)

__all__ = [
    "LIGHTGBM_BASELINE_PARAMS",
    "LOGISTIC_BASELINE_PARAMS",
    "FusionCVResult",
    "FusionDataError",
    "FusionModelError",
    "build_feature_table",
    "create_fusion_model",
    "create_lightgbm_model",
    "create_logistic_model",
    "cross_validate_fusion",
    "lightgbm_gain_importance",
]
