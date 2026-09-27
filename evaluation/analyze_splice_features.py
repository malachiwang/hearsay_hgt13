"""Small descriptive utility for already-computed splice representations.

This module does not train models or tune thresholds. It turns in-memory
per-window embeddings into novelty summaries and reports their distributions
by caller-supplied label. An optional physical ``splice_max_score`` may be
included for side-by-side inspection.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from NoveltySplice import detect_embedding_novelty, novelty_splice_features


class SpliceAnalysisError(ValueError):
    """Raised when a descriptive splice-analysis record is malformed."""


def extract_splice_analysis_rows(
    records: Iterable[Mapping[str, object]],
) -> pd.DataFrame:
    """Extract one descriptive novelty row per supplied embedding sequence."""

    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        missing = [
            name
            for name in ("file_id", "label", "embeddings", "frame_times")
            if name not in record
        ]
        if missing:
            raise SpliceAnalysisError(
                f"record {index} is missing required fields: {', '.join(missing)}"
            )
        file_id = record["file_id"]
        if not isinstance(file_id, str) or not file_id.strip():
            raise SpliceAnalysisError(f"record {index} has an invalid file_id")
        if file_id in seen_ids:
            raise SpliceAnalysisError(f"duplicate file_id in analysis records: {file_id}")
        seen_ids.add(file_id)

        result = detect_embedding_novelty(
            record["embeddings"],
            record["frame_times"],
            local_fake_scores=record.get("local_fake_scores"),
        )
        row: dict[str, object] = {
            "file_id": file_id,
            "label": record["label"],
            **novelty_splice_features(result),
        }
        if "splice_max_score" in record:
            try:
                physical_score = float(record["splice_max_score"])
            except (TypeError, ValueError, OverflowError) as exc:
                raise SpliceAnalysisError(
                    f"record {file_id!r} has a nonnumeric splice_max_score"
                ) from exc
            if not np.isfinite(physical_score):
                raise SpliceAnalysisError(
                    f"record {file_id!r} has a nonfinite splice_max_score"
                )
            row["splice_max_score"] = physical_score
        rows.append(row)

    if not rows:
        raise SpliceAnalysisError("at least one analysis record is required")
    return pd.DataFrame(rows)


def summarize_splice_distributions(per_file: pd.DataFrame) -> pd.DataFrame:
    """Return finite long-form descriptive distributions grouped by label."""

    required = {"file_id", "label"}
    if not required.issubset(per_file.columns):
        raise SpliceAnalysisError("per-file table requires file_id and label columns")
    feature_columns = [
        column
        for column in per_file.columns
        if column not in required and pd.api.types.is_numeric_dtype(per_file[column])
    ]
    if not feature_columns:
        raise SpliceAnalysisError("per-file table has no numeric feature columns")

    rows: list[dict[str, object]] = []
    for label, group in per_file.groupby("label", sort=False, dropna=False):
        for feature in feature_columns:
            values = group[feature].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise SpliceAnalysisError(
                    f"feature {feature!r} contains NaN or Inf for label {label!r}"
                )
            rows.append(
                {
                    "label": label,
                    "feature": feature,
                    "n_files": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95.0)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def analyze_splice_feature_distributions(
    records: Iterable[Mapping[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-file features and label-grouped descriptive distributions."""

    per_file = extract_splice_analysis_rows(records)
    return per_file, summarize_splice_distributions(per_file)
