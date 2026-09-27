#!/usr/bin/env python3
"""Efficient full-file inference for Eliya's WavLM + AASIST detector.

The upstream model is loaded once, then each decoded file is covered by
five-second windows with a half-second hop. The model's sigmoid output is the
bona-fide probability, so HEARSAY's synthetic-oriented score is ``1 - sigmoid``.

Usage:
    python NewAttempt/Deepfake.py path/to/audio.wav
    python NewAttempt/Deepfake.py path/to/folder_of_audio/
    python NewAttempt/Deepfake.py path/to/audio.wav --threshold 0.15
    python NewAttempt/Deepfake.py path/to/folder/ --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


MODEL_REPO = "eliya/forensics_0.3B_base_deepfake_classifier"
MODEL_DIR = Path(__file__).parent / "models" / MODEL_REPO.split("/")[-1]
MODEL_FILES = (
    "checkpoint_epoch_5.safetensors",
    "inference.py",
    "model.py",
    "config.json",
    "requirements.txt",
)
CHECKPOINT_NAME = "checkpoint_epoch_5.safetensors"

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 5.0
HOP_SECONDS = 0.5
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)
DEFAULT_BATCH_SIZE = 4
DEFAULT_DIAGNOSTIC_THRESHOLD = 0.15

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
CSV_FIELDS = (
    "file",
    "eliya_mean",
    "eliya_max",
    "eliya_top3_mean",
    "eliya_p90",
    "eliya_high_window_fraction",
    "n_windows",
)


class EliyaInferenceError(RuntimeError):
    """Raised when the Eliya model or an audio file cannot be evaluated."""


def ensure_model(model_dir: Path = MODEL_DIR) -> None:
    """Ensure the five required upstream files exist without fetching the .pt file."""

    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if not missing:
        return

    print(f"Downloading {MODEL_REPO} -> {model_dir} ...")
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=str(model_dir),
            allow_patterns=list(MODEL_FILES),
        )
    except Exception as exc:
        raise EliyaInferenceError(
            f"could not download required Eliya model files: {exc}"
        ) from exc

    still_missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    if still_missing:
        raise EliyaInferenceError(
            "Eliya model download is incomplete; missing: " + ", ".join(still_missing)
        )


def _import_upstream_model(model_dir: Path):
    model_path = model_dir / "model.py"
    spec = importlib.util.spec_from_file_location("_hearsay_eliya_model", model_path)
    if spec is None or spec.loader is None:
        raise EliyaInferenceError(f"could not import upstream model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise EliyaInferenceError(f"failed to import upstream model.py: {exc}") from exc
    if not hasattr(module, "DeepfakeDetector"):
        raise EliyaInferenceError("upstream model.py does not define DeepfakeDetector")
    return module.DeepfakeDetector


def _validate_upstream_config(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EliyaInferenceError(f"could not read upstream config {config_path}") from exc
    architectures = config.get("architectures", [])
    if "DeepfakeDetector" not in architectures:
        raise EliyaInferenceError(
            "upstream config.json does not declare DeepfakeDetector"
        )


def load_eliya_model(model_dir: Path, device: Any):
    """Construct the upstream architecture and load its safetensors state once."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise EliyaInferenceError(
            "Eliya inference requires the 'safetensors' package"
        ) from exc

    _validate_upstream_config(model_dir)
    detector_class = _import_upstream_model(model_dir)
    checkpoint = model_dir / CHECKPOINT_NAME
    try:
        model = detector_class()
        state_dict = load_file(str(checkpoint), device="cpu")
        model.load_state_dict(state_dict, strict=False)
        return model.to(device).eval()
    except Exception as exc:
        raise EliyaInferenceError(
            f"failed to load Eliya checkpoint {checkpoint}: {exc}"
        ) from exc


def load_audio(path: Path):
    """Decode once and apply the upstream mono/16 kHz/amplitude preparation."""

    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise EliyaInferenceError(
            "Eliya audio loading requires torch and torchaudio"
        ) from exc

    try:
        waveform, original_rate = torchaudio.load(str(path))
    except Exception as exc:
        raise EliyaInferenceError(f"could not decode {path}: {exc}") from exc
    if waveform.ndim != 2 or waveform.shape[1] == 0:
        raise EliyaInferenceError(f"decoded audio is empty or malformed: {path}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    waveform = waveform.squeeze(0)
    if original_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, original_rate, SAMPLE_RATE
        )
    if waveform.numel() == 0:
        raise EliyaInferenceError(f"decoded audio is empty: {path}")
    if not bool(torch.isfinite(waveform).all()):
        raise EliyaInferenceError(f"decoded audio contains NaN or Inf: {path}")

    # This matches the actual upstream inference.py preparation.
    waveform = waveform / (waveform.abs().max() + 1e-8)
    return waveform.contiguous()


def window_start_samples(
    sample_count: int,
    *,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
) -> list[int]:
    """Return regular starts plus one unique end-anchored start when needed."""

    if sample_count <= 0:
        raise EliyaInferenceError("audio must contain at least one sample")
    if window_samples <= 0 or hop_samples <= 0:
        raise EliyaInferenceError("window and hop sizes must be positive")
    if sample_count <= window_samples:
        return [0]

    last_start = sample_count - window_samples
    starts = list(range(0, last_start + 1, hop_samples))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _model_window(waveform: Any, start: int) -> Any:
    """Slice a full window or repeat-pad one short whole-file window."""

    sample_count = int(waveform.shape[0])
    if sample_count < WINDOW_SAMPLES:
        repeats = (WINDOW_SAMPLES + sample_count - 1) // sample_count
        return waveform.repeat(repeats)[:WINDOW_SAMPLES]
    return waveform[start : start + WINDOW_SAMPLES]


def aggregate_window_scores(
    scores: Sequence[float],
    *,
    diagnostic_threshold: float = DEFAULT_DIAGNOSTIC_THRESHOLD,
) -> dict[str, float]:
    """Return finite fusion-ready summaries of synthetic window scores."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise EliyaInferenceError("at least one window score is required")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise EliyaInferenceError("window scores must be finite and within [0, 1]")
    if not np.isfinite(diagnostic_threshold) or not 0.0 <= diagnostic_threshold <= 1.0:
        raise EliyaInferenceError("diagnostic threshold must be within [0, 1]")

    top_values = np.sort(values)[-min(3, values.size) :]
    return {
        "eliya_mean": float(np.mean(values)),
        "eliya_max": float(np.max(values)),
        "eliya_top3_mean": float(np.mean(top_values)),
        "eliya_p90": float(np.percentile(values, 90.0)),
        "eliya_high_window_fraction": float(
            np.mean(values >= diagnostic_threshold)
        ),
    }


class EliyaDetector:
    """Once-loaded Eliya model with batched, full-file sliding-window inference."""

    def __init__(
        self,
        *,
        model_dir: Path = MODEL_DIR,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        diagnostic_threshold: float = DEFAULT_DIAGNOSTIC_THRESHOLD,
        model: Any | None = None,
        model_loader: Callable[[Path, Any], Any] = load_eliya_model,
        audio_loader: Callable[[Path], Any] = load_audio,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise EliyaInferenceError("Eliya inference requires torch") from exc
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise EliyaInferenceError("batch_size must be a positive integer")
        if not np.isfinite(diagnostic_threshold) or not 0.0 <= diagnostic_threshold <= 1.0:
            raise EliyaInferenceError("diagnostic threshold must be within [0, 1]")

        self._torch = torch
        self.model_dir = Path(model_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self.diagnostic_threshold = float(diagnostic_threshold)
        self.audio_loader = audio_loader

        if model is None:
            ensure_model(self.model_dir)
            self.model = model_loader(self.model_dir, self.device)
        else:
            self.model = model.to(self.device).eval()

    def analyze_file(self, audio_path: str | Path) -> dict[str, Any]:
        """Decode one file once and return window scores plus aggregate features."""

        path = Path(audio_path)
        waveform = self.audio_loader(path)
        if not self._torch.is_tensor(waveform):
            waveform = self._torch.as_tensor(waveform)
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise EliyaInferenceError("audio loader must return a nonempty 1-D waveform")
        waveform = waveform.to(dtype=self._torch.float32, device="cpu").contiguous()
        if not bool(self._torch.isfinite(waveform).all()):
            raise EliyaInferenceError("audio loader returned NaN or Inf")

        sample_count = int(waveform.shape[0])
        duration = sample_count / SAMPLE_RATE
        starts = window_start_samples(sample_count)
        scores: list[float] = []

        with self._torch.inference_mode():
            for offset in range(0, len(starts), self.batch_size):
                batch_starts = starts[offset : offset + self.batch_size]
                batch = self._torch.stack(
                    [_model_window(waveform, start) for start in batch_starts]
                ).to(self.device)
                logits = self.model(batch).float().reshape(-1)
                if logits.numel() != len(batch_starts):
                    raise EliyaInferenceError(
                        "Eliya model returned the wrong number of window scores"
                    )
                fake_probabilities = 1.0 - self._torch.sigmoid(logits)
                scores.extend(float(value) for value in fake_probabilities.cpu())

        aggregates = aggregate_window_scores(
            scores, diagnostic_threshold=self.diagnostic_threshold
        )
        windows = []
        for start, score in zip(starts, scores, strict=True):
            start_seconds = start / SAMPLE_RATE
            end_seconds = min(start_seconds + WINDOW_SECONDS, duration)
            windows.append(
                {
                    "start": float(start_seconds),
                    "end": float(end_seconds),
                    "score": float(score),
                }
            )

        main_score = aggregates["eliya_top3_mean"]
        return {
            "file": str(path),
            "windows": windows,
            **aggregates,
            "n_windows": len(windows),
            # Backward-compatible keys from the original one-shot wrapper.
            "fake_probability": main_score,
            "bonafide_score": 1.0 - main_score,
            "verdict": (
                "FAKE" if main_score >= self.diagnostic_threshold else "REAL"
            ),
        }


_DEFAULT_DETECTOR: EliyaDetector | None = None


def get_detector(**kwargs: Any) -> EliyaDetector:
    """Return the process-wide detector, initializing its neural model once."""

    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = EliyaDetector(**kwargs)
    elif kwargs:
        raise EliyaInferenceError("the default Eliya detector is already initialized")
    return _DEFAULT_DETECTOR


def run_inference(
    audio_path: str | Path,
    detector: EliyaDetector | None = None,
) -> dict[str, Any]:
    """Preserve the original per-file entry point with in-process inference."""

    return (detector or get_detector()).analyze_file(audio_path)


def collect_audio_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect AI-generated / deepfake speech in audio files."
    )
    parser.add_argument("input", help="Audio file or a folder of audio files")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DIAGNOSTIC_THRESHOLD,
        help=(
            "diagnostic fake-score threshold used for the high-window fraction "
            "and legacy verdict only; it never changes the continuous score "
            "(default: 0.15)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="number of five-second windows evaluated together (default: 4)",
    )
    parser.add_argument("--csv", help="optional path to write aggregate results")
    parser.add_argument(
        "--ask-grok",
        nargs="?",
        const=True,
        metavar="QUESTION",
        help=(
            "ask Grok to interpret each completed HEARSAY analysis; optionally "
            "provide a question (requires XAI_API_KEY)"
        ),
    )
    parser.add_argument(
        "--grok-model",
        help="optional xAI model override (default: XAI_MODEL or grok-4.7)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    files = collect_audio_files(input_path)
    if not files:
        parser.exit(1, f"No audio files found at {input_path}\n")

    try:
        detector = get_detector(
            diagnostic_threshold=args.threshold,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        parser.exit(2, f"Could not initialize Eliya detector: {exc}\n")

    rows = []
    for audio_file in files:
        try:
            result = run_inference(audio_file, detector)
        except Exception as exc:
            print(f"[ERROR] {audio_file}: {exc}")
            continue
        rows.append(result)
        print(
            f"{audio_file.name:40s}  "
            f"eliya_top3_mean={result['eliya_top3_mean']:.4f}  "
            f"eliya_max={result['eliya_max']:.4f}  "
            f"windows={result['n_windows']}"
        )
        if args.ask_grok:
            from grok_interpretation import (
                DEFAULT_GROK_QUESTION,
                GrokError,
                ask_grok_about_analysis,
            )

            question = DEFAULT_GROK_QUESTION if args.ask_grok is True else args.ask_grok
            try:
                interpretation = ask_grok_about_analysis(
                    result,
                    question,
                    model=args.grok_model,
                )
            except GrokError as exc:
                print(
                    f"[GROK UNAVAILABLE] {audio_file.name}: {exc}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"\nGrok interpretation for {audio_file.name} "
                    f"({interpretation.model}; not a detector result):\n"
                    f"{interpretation.text}\n"
                )

    if args.csv and rows:
        with Path(args.csv).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} results to {args.csv}")


if __name__ == "__main__":
    main()
