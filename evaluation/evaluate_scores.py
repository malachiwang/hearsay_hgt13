"""Command-line evaluation of HEARSAY score and answer-key tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from .min_dcf import VALID_LABELS, MinDCFError, evaluate_min_dcf


class ScoreTableError(ValueError):
    """Raised when a score or label table is invalid."""


def _read_rows(
    path: str | Path,
    required_columns: Sequence[str],
    delimiter: str,
) -> list[dict[str, str]]:
    if len(delimiter) != 1:
        raise ScoreTableError("delimiter must be exactly one character")
    table_path = Path(path)
    if not table_path.exists():
        raise ScoreTableError(f"table does not exist: {table_path}")
    if not table_path.is_file():
        raise ScoreTableError(f"table is not a regular file: {table_path}")

    try:
        with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ScoreTableError(f"table has no header: {table_path}")
            missing_columns = [
                column for column in required_columns if column not in reader.fieldnames
            ]
            if missing_columns:
                raise ScoreTableError(
                    f"{table_path} is missing required columns: "
                    + ", ".join(missing_columns)
                )
            return list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ScoreTableError(f"could not read {table_path}: {exc}") from exc


def read_predictions(
    path: str | Path,
    *,
    id_column: str = "id",
    score_column: str = "score",
    delimiter: str = "\t",
) -> dict[str, float]:
    """Read an ID-to-score table, rejecting duplicates and invalid scores."""

    rows = _read_rows(path, (id_column, score_column), delimiter)
    predictions: dict[str, float] = {}
    for line_number, row in enumerate(rows, start=2):
        file_id = (row[id_column] or "").strip()
        if not file_id:
            raise ScoreTableError(f"blank prediction ID at line {line_number}")
        if file_id in predictions:
            raise ScoreTableError(f"duplicate prediction ID {file_id!r}")
        raw_score = (row[score_column] or "").strip()
        try:
            score = float(raw_score)
        except ValueError as exc:
            raise ScoreTableError(
                f"nonnumeric score for ID {file_id!r}: {raw_score!r}"
            ) from exc
        if not np.isfinite(score):
            raise ScoreTableError(f"nonfinite score for ID {file_id!r}: {raw_score!r}")
        predictions[file_id] = score
    if not predictions:
        raise ScoreTableError("prediction table contains no data rows")
    return predictions


def read_labels(
    path: str | Path,
    *,
    id_column: str = "id",
    label_column: str = "label",
    delimiter: str = "\t",
) -> dict[str, str]:
    """Read an ID-to-label table, rejecting duplicates and unknown labels."""

    rows = _read_rows(path, (id_column, label_column), delimiter)
    labels: dict[str, str] = {}
    for line_number, row in enumerate(rows, start=2):
        file_id = (row[id_column] or "").strip()
        if not file_id:
            raise ScoreTableError(f"blank label ID at line {line_number}")
        if file_id in labels:
            raise ScoreTableError(f"duplicate label ID {file_id!r}")
        label = (row[label_column] or "").strip()
        if label not in VALID_LABELS:
            raise ScoreTableError(
                f"unsupported label for ID {file_id!r}: {label!r}; "
                f"expected {sorted(VALID_LABELS)}"
            )
        labels[file_id] = label
    if not labels:
        raise ScoreTableError("label table contains no data rows")
    return labels


def join_predictions_and_labels(
    predictions: dict[str, float],
    labels: dict[str, str],
    *,
    allow_extra_predictions: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Strictly join predictions to labels by ID, preserving label-table order."""

    missing = [file_id for file_id in labels if file_id not in predictions]
    if missing:
        raise ScoreTableError(
            "missing predictions for IDs: " + ", ".join(repr(item) for item in missing)
        )
    extra = [file_id for file_id in predictions if file_id not in labels]
    if extra and not allow_extra_predictions:
        raise ScoreTableError(
            "unexpected prediction IDs: " + ", ".join(repr(item) for item in extra)
        )

    ordered_ids = list(labels)
    scores = np.asarray([predictions[file_id] for file_id in ordered_ids], dtype=np.float64)
    ordered_labels = [labels[file_id] for file_id in ordered_ids]
    return scores, ordered_labels


def load_evaluation_tables(
    scores_path: str | Path,
    labels_path: str | Path,
    *,
    id_column: str = "id",
    score_column: str = "score",
    label_column: str = "label",
    delimiter: str = "\t",
    allow_extra_predictions: bool = False,
) -> tuple[np.ndarray, list[str]]:
    predictions = read_predictions(
        scores_path,
        id_column=id_column,
        score_column=score_column,
        delimiter=delimiter,
    )
    labels = read_labels(
        labels_path,
        id_column=id_column,
        label_column=label_column,
        delimiter=delimiter,
    )
    return join_predictions_and_labels(
        predictions,
        labels,
        allow_extra_predictions=allow_extra_predictions,
    )


def _parse_delimiter(value: str) -> str:
    aliases = {"tab": "\t", r"\t": "\t", "comma": ","}
    delimiter = aliases.get(value.lower(), value)
    if len(delimiter) != 1:
        raise argparse.ArgumentTypeError(
            "delimiter must be one character, 'tab', '\\t', or 'comma'"
        )
    return delimiter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic-oriented HEARSAY scores with NSA minDCF."
    )
    parser.add_argument("--scores", required=True, type=Path, help="prediction table")
    parser.add_argument("--labels", required=True, type=Path, help="answer-key table")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--delimiter",
        default="\t",
        type=_parse_delimiter,
        help="table delimiter (default: tab)",
    )
    parser.add_argument(
        "--allow-extra-predictions",
        action="store_true",
        help="ignore prediction IDs that are absent from the answer key",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        scores, labels = load_evaluation_tables(
            args.scores,
            args.labels,
            id_column=args.id_column,
            score_column=args.score_column,
            label_column=args.label_column,
            delimiter=args.delimiter,
            allow_extra_predictions=args.allow_extra_predictions,
        )
        result = evaluate_min_dcf(scores, labels)
    except (ScoreTableError, MinDCFError) as exc:
        parser.error(str(exc))

    print(f"HEARSAY minDCF: {result.min_dcf:.6f}")
    print(f"Best synthetic-score threshold: {result.threshold:.6f}")
    print(f"Pmiss (real -> spoof): {result.p_miss:.6f}")
    print(f"Pfa   (spoof -> real): {result.p_fa:.6f}")
    print(f"Bona fide: {result.number_bonafide}")
    print(f"Spoof: {result.number_spoof}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
