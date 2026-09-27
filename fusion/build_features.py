"""Build strict per-file fusion feature tables.

The manifest is authoritative for file identifiers, labels, folds, and row
order. Detector sources contribute numeric features only and are joined by
``file_id`` with exact one-to-one coverage.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd


FILE_ID_COLUMN = "file_id"
LABEL_COLUMN = "label"
FOLD_COLUMN = "fold"
METADATA_COLUMNS = (FILE_ID_COLUMN, LABEL_COLUMN, FOLD_COLUMN)

LABEL_ALIASES = {
    "0": 0,
    "1": 1,
    "bonafide": 0,
    "bona_fide": 0,
    "real": 0,
    "spoof": 1,
    "synthetic": 1,
    "fake": 1,
}

TableInput: TypeAlias = pd.DataFrame | str | Path


class FusionDataError(ValueError):
    """Raised when a manifest or detector feature source violates its contract."""


def _read_table(table: TableInput, context: str) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.copy()

    path = Path(table)
    if not path.exists():
        raise FusionDataError(f"{context} does not exist: {path}")
    if not path.is_file():
        raise FusionDataError(f"{context} is not a regular file: {path}")

    suffix = path.suffix.lower()
    separator = "\t" if suffix in {".tsv", ".tab"} else ","
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle, delimiter=separator), [])
        duplicate_headers = sorted(
            {column for column in header if header.count(column) > 1}
        )
        if duplicate_headers:
            raise FusionDataError(
                f"{context} has duplicate column names: "
                + ", ".join(repr(column) for column in duplicate_headers)
            )
        return pd.read_csv(path, sep=separator)
    except FusionDataError:
        raise
    except Exception as exc:
        raise FusionDataError(f"could not read {context} from {path}: {exc}") from exc


def _reject_duplicate_columns(table: pd.DataFrame, context: str) -> None:
    duplicated = table.columns[table.columns.duplicated()].tolist()
    if duplicated:
        names = ", ".join(repr(str(name)) for name in duplicated)
        raise FusionDataError(f"{context} has duplicate column names: {names}")


def _validate_file_ids(table: pd.DataFrame, context: str) -> None:
    if FILE_ID_COLUMN not in table.columns:
        raise FusionDataError(f"{context} is missing required column 'file_id'")

    ids = table[FILE_ID_COLUMN]
    if ids.isna().any():
        raise FusionDataError(f"{context} contains a missing file_id")
    if not ids.map(lambda value: isinstance(value, str)).all():
        raise FusionDataError(f"{context} file_id values must be strings")
    if ids.str.strip().eq("").any():
        raise FusionDataError(f"{context} contains an empty file_id")

    duplicate_mask = ids.duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = ids[duplicate_mask].drop_duplicates().tolist()
        raise FusionDataError(
            f"{context} has duplicate file_id values: {_format_ids(duplicate_ids)}"
        )


def _normalize_label(value: object) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in LABEL_ALIASES:
            return LABEL_ALIASES[normalized]
        raise FusionDataError(f"unsupported label {value!r}")

    if isinstance(value, (bool, np.bool_)):
        raise FusionDataError(f"unsupported label {value!r}")
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value) in (0.0, 1.0):
            return int(value)
    raise FusionDataError(f"unsupported label {value!r}")


def _validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    context = "manifest"
    _reject_duplicate_columns(manifest, context)
    _validate_file_ids(manifest, context)

    missing_columns = [
        column for column in (LABEL_COLUMN, FOLD_COLUMN) if column not in manifest
    ]
    if missing_columns:
        raise FusionDataError(
            "manifest is missing required columns: " + ", ".join(missing_columns)
        )
    if manifest.empty:
        raise FusionDataError("manifest must contain at least one row")

    try:
        labels = manifest[LABEL_COLUMN].map(_normalize_label)
    except FusionDataError as exc:
        raise FusionDataError(f"manifest {exc}") from exc

    folds = manifest[FOLD_COLUMN]
    if folds.isna().any():
        raise FusionDataError("manifest contains a missing fold")
    if folds.map(lambda value: isinstance(value, str) and not value.strip()).any():
        raise FusionDataError("manifest contains an empty fold")

    result = manifest.loc[:, list(METADATA_COLUMNS)].copy()
    result[LABEL_COLUMN] = labels.astype(np.int8)
    return result


def _format_ids(ids: list[str], limit: int = 5) -> str:
    shown = ", ".join(repr(value) for value in ids[:limit])
    remaining = len(ids) - limit
    return shown if remaining <= 0 else f"{shown}, ... ({remaining} more)"


def _validate_feature_source(
    source: pd.DataFrame,
    source_name: str,
    manifest_ids: set[str],
    existing_feature_names: set[str],
) -> list[str]:
    context = f"feature source {source_name!r}"
    _reject_duplicate_columns(source, context)
    _validate_file_ids(source, context)

    forbidden = [column for column in (LABEL_COLUMN, FOLD_COLUMN) if column in source]
    if forbidden:
        raise FusionDataError(
            f"{context} must not provide manifest-owned columns: "
            + ", ".join(forbidden)
        )

    feature_columns = [column for column in source.columns if column != FILE_ID_COLUMN]
    if not feature_columns:
        raise FusionDataError(f"{context} has no feature columns")
    if any(not isinstance(column, str) or not column for column in feature_columns):
        raise FusionDataError(f"{context} feature column names must be nonempty strings")

    duplicate_features = [
        column for column in feature_columns if column in existing_feature_names
    ]
    if duplicate_features:
        raise FusionDataError(
            f"{context} duplicates existing feature columns: "
            + ", ".join(repr(column) for column in duplicate_features)
        )

    source_ids = set(source[FILE_ID_COLUMN].tolist())
    missing = sorted(manifest_ids - source_ids)
    extra = sorted(source_ids - manifest_ids)
    if missing:
        noun = "file" if len(missing) == 1 else "files"
        raise FusionDataError(
            f"{context} is missing {len(missing)} manifest {noun}: "
            f"{_format_ids(missing)}"
        )
    if extra:
        noun = "row" if len(extra) == 1 else "rows"
        raise FusionDataError(
            f"{context} has {len(extra)} unexpected {noun}: {_format_ids(extra)}"
        )

    for column in feature_columns:
        values = source[column]
        if not pd.api.types.is_numeric_dtype(values.dtype):
            raise FusionDataError(
                f"{context} column {column!r} must contain numeric values"
            )
        numeric_values = values.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(numeric_values).all():
            raise FusionDataError(
                f"{context} column {column!r} contains NaN or Inf"
            )
    return feature_columns


def build_feature_table(
    manifest: TableInput,
    feature_sources: Mapping[str, TableInput],
) -> pd.DataFrame:
    """Merge explicitly requested detector features into one strict table.

    Source mapping insertion order and each source's column order define the
    deterministic feature order. No values are filled, scaled, or imputed.
    """

    if not isinstance(feature_sources, Mapping):
        raise FusionDataError("feature_sources must be a mapping of name to table")
    if not feature_sources:
        raise FusionDataError("at least one feature source is required")

    result = _validate_manifest(_read_table(manifest, "manifest"))
    manifest_ids = set(result[FILE_ID_COLUMN].tolist())
    feature_names: set[str] = set()

    for raw_name, table_input in feature_sources.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise FusionDataError("feature source names must be nonempty strings")
        source_name = raw_name.strip()
        source = _read_table(table_input, f"feature source {source_name!r}")
        columns = _validate_feature_source(
            source, source_name, manifest_ids, feature_names
        )
        result = result.merge(
            source.loc[:, [FILE_ID_COLUMN, *columns]],
            on=FILE_ID_COLUMN,
            how="left",
            sort=False,
            validate="one_to_one",
        )
        feature_names.update(columns)

    return result
