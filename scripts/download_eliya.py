#!/usr/bin/env python3
"""Populate and verify all Hugging Face assets needed by Eliya offline."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from NewAttempt.Deepfake import MODEL_DIR, MODEL_FILES, ensure_model  # noqa: E402


WAVLM_REPO = "microsoft/wavlm-large"
WAVLM_FILES = ("config.json", "pytorch_model.bin")


class ModelAssetError(RuntimeError):
    """Raised when build-time model assets are incomplete or unexpected."""


def _validate_eliya_directory() -> None:
    missing = [name for name in MODEL_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise ModelAssetError("missing Eliya files: " + ", ".join(missing))
    if (MODEL_DIR / "checkpoint_epoch_5.pt").exists():
        raise ModelAssetError("legacy checkpoint_epoch_5.pt must not be present")

    actual_files = {
        path.name for path in MODEL_DIR.iterdir() if path.is_file()
    }
    unexpected = sorted(actual_files - set(MODEL_FILES))
    if unexpected:
        raise ModelAssetError(
            "unexpected files in Eliya model directory: " + ", ".join(unexpected)
        )


def download_assets() -> None:
    """Reuse the detector downloader, then cache its WavLM base dependency."""

    ensure_model()
    # local_dir downloads leave only disposable Hugging Face metadata here.
    shutil.rmtree(MODEL_DIR / ".cache", ignore_errors=True)
    _validate_eliya_directory()

    snapshot_download(
        repo_id=WAVLM_REPO,
        allow_patterns=list(WAVLM_FILES),
    )


def verify_assets(*, local_files_only: bool) -> None:
    _validate_eliya_directory()
    snapshot_path = Path(
        snapshot_download(
            repo_id=WAVLM_REPO,
            allow_patterns=list(WAVLM_FILES),
            local_files_only=local_files_only,
        )
    )
    missing = [name for name in WAVLM_FILES if not (snapshot_path / name).is_file()]
    if missing:
        raise ModelAssetError("missing cached WavLM files: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify assets from local storage without downloading",
    )
    args = parser.parse_args()

    if not args.verify_only:
        download_assets()
    verify_assets(local_files_only=args.verify_only)
    print(
        f"Eliya assets ready in {MODEL_DIR}; "
        f"WavLM cache ready for {WAVLM_REPO}"
    )


if __name__ == "__main__":
    main()
