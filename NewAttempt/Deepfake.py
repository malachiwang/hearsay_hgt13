#!/usr/bin/env python3
"""
Voice deepfake / AI-voice detector.

Wraps eliya/forensics_0.3B_base_deepfake_classifier (WavLM-large + AASIST)
to predict whether a speech clip is real or synthetic (TTS / voice-cloned /
deepfake). Rather than re-implementing the model's custom architecture, this
downloads the model repo and drives the *model author's own* inference.py,
which is the safest way to get correct results.

Usage:
    python detect_deepfake.py path/to/audio.wav
    python detect_deepfake.py path/to/folder_of_audio/
    python detect_deepfake.py path/to/audio.wav --threshold 0.15
    python detect_deepfake.py path/to/folder/ --csv results.csv

First run downloads the model (a few hundred MB) from the Hugging Face Hub
into ./models/forensics_0.3B_base_deepfake_classifier and reuses it after that.
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

MODEL_REPO = "eliya/forensics_0.3B_base_deepfake_classifier"
MODEL_DIR = Path(__file__).parent / "models" / MODEL_REPO.split("/")[-1]

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}


def ensure_model():
    """Download the model repo (weights + its own inference.py/model.py) if missing."""
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        return
    print(f"Downloading {MODEL_REPO} -> {MODEL_DIR} ...")
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=MODEL_REPO, local_dir=str(MODEL_DIR))


def run_inference(audio_path: Path) -> dict:
    """Call the model's own inference.py as a subprocess and parse its printed output."""
    result = subprocess.run(
        [sys.executable, "inference.py", str(audio_path.resolve())],
        cwd=str(MODEL_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"inference.py failed on {audio_path}:\n{result.stderr.strip()}")

    out = result.stdout

    def grab(pattern, cast=float):
        m = re.search(pattern, out)
        return cast(m.group(1)) if m else None

    return {
        "file": str(audio_path),
        "fake_probability": grab(r"fake_probability:\s*([0-9.]+)"),
        "bonafide_score": grab(r"bonafide_score:\s*([0-9.]+)"),
        "verdict": grab(r"verdict:\s*(\w+)", cast=str),
    }


def collect_audio_files(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def main():
    parser = argparse.ArgumentParser(description="Detect AI-generated / deepfake speech in audio files.")
    parser.add_argument("input", help="Audio file or a folder of audio files")
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="fake_probability at/above this = FAKE (default 0.15, per the model card's guidance range)",
    )
    parser.add_argument("--csv", help="Optional path to write results as CSV")
    args = parser.parse_args()

    ensure_model()

    input_path = Path(args.input)
    files = collect_audio_files(input_path)
    if not files:
        print(f"No audio files found at {input_path}")
        sys.exit(1)

    rows = []
    for f in files:
        try:
            r = run_inference(f)
        except Exception as e:
            print(f"[ERROR] {f}: {e}")
            continue
        if r["fake_probability"] is not None:
            r["verdict"] = "FAKE" if r["fake_probability"] >= args.threshold else "REAL"
        rows.append(r)
        prob = r["fake_probability"]
        prob_str = f"{prob:.4f}" if prob is not None else "n/a"
        print(f"{f.name:40s}  fake_prob={prob_str}  -> {r['verdict']}")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["file", "fake_probability", "bonafide_score", "verdict"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} results to {args.csv}")


if __name__ == "__main__":
    main()