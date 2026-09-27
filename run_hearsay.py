#!/usr/bin/env python3
"""Batch HEARSAY judging entry point using the existing Eliya detector."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from NewAttempt.Deepfake import EliyaDetector


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".mp4",
    ".ogg",
    ".opus",
    ".flac",
}
OUTPUT_COLUMNS = ("filename", "cm-score")


class HearsayRunnerError(RuntimeError):
    """Raised when a judging batch cannot produce a complete valid TSV."""


def discover_audio_files(input_directory: str | Path) -> list[Path]:
    """Return supported files recursively in deterministic relative-path order."""

    root = Path(input_directory)
    if not root.exists():
        raise HearsayRunnerError(f"input directory does not exist: {root}")
    if not root.is_dir():
        raise HearsayRunnerError(f"input path is not a directory: {root}")

    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    files.sort(key=lambda path: (path.relative_to(root).as_posix().casefold(), path.as_posix()))
    if not files:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise HearsayRunnerError(
            f"no supported audio files found in {root}; expected: {supported}"
        )
    return files


def _validated_score(result: dict[str, Any], audio_path: Path) -> float:
    if "eliya_top3_mean" not in result:
        raise HearsayRunnerError(
            f"Eliya result for {audio_path} is missing eliya_top3_mean"
        )
    try:
        score = float(result["eliya_top3_mean"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise HearsayRunnerError(
            f"Eliya produced a nonnumeric cm-score for {audio_path}"
        ) from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise HearsayRunnerError(
            f"Eliya produced an invalid cm-score for {audio_path}: {score!r}"
        )
    return score


def _write_tsv(output_path: Path, rows: list[tuple[str, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(OUTPUT_COLUMNS)
            for filename, score in rows:
                writer.writerow((filename, format(score, ".10g")))
        os.replace(temporary_name, output_path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def run_batch(
    input_directory: str | Path,
    output_tsv: str | Path,
    *,
    detector: EliyaDetector | None = None,
) -> list[tuple[str, float]]:
    """Score every supported file, failing atomically if any file fails."""

    audio_files = discover_audio_files(input_directory)
    active_detector = detector or EliyaDetector()
    rows: list[tuple[str, float]] = []
    for audio_path in audio_files:
        try:
            result = active_detector.analyze_file(audio_path)
        except Exception as exc:
            raise HearsayRunnerError(f"failed to score {audio_path}: {exc}") from exc
        rows.append((audio_path.name, _validated_score(result, audio_path)))

    _write_tsv(Path(output_tsv), rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score a directory of audio files for HEARSAY judging."
    )
    parser.add_argument("input_directory")
    parser.add_argument("output_tsv")
    args = parser.parse_args(argv)

    try:
        rows = run_batch(args.input_directory, args.output_tsv)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(rows)} scores to {args.output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
