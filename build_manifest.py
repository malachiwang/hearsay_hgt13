"""Build manifest.csv cataloging all training data for HEARSAY."""

import os
import re
import csv
import random
from pathlib import Path

DATA_ROOT = Path(r"C:\hearsay_data")
OUTPUT = DATA_ROOT / "manifest.csv"

GENERATORS_WITH_SPEAKERS = {
    "elevenlabs", "openvoicev2", "playht", "unit_speech", "xtts_v2", "your_tts"
}
GENERATORS_FLAT = {
    "diffgan_tts", "grad_tts", "pro_diff", "wavegrad2"
}
ALL_GENERATORS = sorted(GENERATORS_WITH_SPEAKERS | GENERATORS_FLAT)
GENERATOR_TO_FOLD = {g: i for i, g in enumerate(ALL_GENERATORS)}

OPENVOICE_ACCENT_RE = re.compile(r"^(sentence_\d+)_(.+)$")


def collect_fake():
    rows = []
    gen_dir = DATA_ROOT / "generated_speech"

    for generator in ALL_GENERATORS:
        fold = GENERATOR_TO_FOLD[generator]
        gpath = gen_dir / generator

        if generator in GENERATORS_FLAT:
            for f in gpath.iterdir():
                if f.name.startswith(".") or f.is_dir():
                    continue
                stem = f.stem
                sentence_id = stem
                rows.append({
                    "path": str(f),
                    "label": "spoof",
                    "generator": generator,
                    "speaker": "unknown",
                    "sentence_id": sentence_id,
                    "accent_style": "",
                    "source_format": f.suffix.lstrip("."),
                    "group_id": f"{generator}_{sentence_id}",
                    "fold": fold,
                })
        else:
            for spk_dir in sorted(gpath.iterdir()):
                if not spk_dir.is_dir() or spk_dir.name.startswith("."):
                    continue
                speaker = spk_dir.name
                for f in spk_dir.iterdir():
                    if f.name.startswith(".") or f.is_dir():
                        continue
                    stem = f.stem
                    accent = ""

                    if generator == "openvoicev2":
                        m = OPENVOICE_ACCENT_RE.match(stem)
                        if m:
                            sentence_id = m.group(1)
                            accent = m.group(2)
                        else:
                            sentence_id = stem
                    else:
                        sentence_id = stem

                    group_id = f"{generator}_{speaker}_{sentence_id}"

                    rows.append({
                        "path": str(f),
                        "label": "spoof",
                        "generator": generator,
                        "speaker": speaker,
                        "sentence_id": sentence_id,
                        "accent_style": accent,
                        "source_format": f.suffix.lstrip("."),
                        "group_id": group_id,
                        "fold": fold,
                    })
    return rows


def collect_real():
    rows = []

    # 1. LJRealResampled (242 files, 16 kHz)
    lj_resampled = DATA_ROOT / "LJRealResampled" / "LJRealResampled" / "resampled"
    for f in sorted(lj_resampled.glob("*.wav")):
        rows.append({
            "path": str(f),
            "label": "bonafide",
            "generator": "real",
            "speaker": "LJ",
            "sentence_id": f.stem,
            "accent_style": "",
            "source_format": "wav",
            "group_id": f"ljresampled_{f.stem}",
            "fold": -1,
        })

    # 2. LJ Speech full (13,100 files, 22.05 kHz)
    lj_full = DATA_ROOT / "LJSpeech-1.1" / "LJSpeech-1.1" / "wavs"
    for f in sorted(lj_full.glob("*.wav")):
        rows.append({
            "path": str(f),
            "label": "bonafide",
            "generator": "real",
            "speaker": "LJ",
            "sentence_id": f.stem,
            "accent_style": "",
            "source_format": "wav",
            "group_id": f"ljfull_{f.stem}",
            "fold": -1,
        })

    # 3. LibriSpeech (11,126 FLAC files, 16 kHz)
    libri_root = DATA_ROOT / "librispeech" / "LibriSpeech"
    for subset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        subset_dir = libri_root / subset
        if not subset_dir.exists():
            continue
        for f in sorted(subset_dir.rglob("*.flac")):
            parts = f.stem.split("-")
            speaker = parts[0] if parts else "unknown"
            rows.append({
                "path": str(f),
                "label": "bonafide",
                "generator": "real",
                "speaker": f"libri_{speaker}",
                "sentence_id": f.stem,
                "accent_style": "",
                "source_format": "flac",
                "group_id": f"libri_{f.stem}",
                "fold": -1,
            })

    return rows


def assign_real_folds(real_rows, num_folds=10, seed=42):
    rng = random.Random(seed)
    indices = list(range(len(real_rows)))
    rng.shuffle(indices)
    for rank, idx in enumerate(indices):
        real_rows[idx]["fold"] = rank % num_folds


def main():
    print("Collecting fake speech...")
    fake = collect_fake()
    print(f"  {len(fake)} fake files")

    print("Collecting real speech...")
    real = collect_real()
    print(f"  {len(real)} real files")

    assign_real_folds(real)

    all_rows = fake + real
    print(f"Total: {len(all_rows)} files")

    # Summary by generator
    from collections import Counter
    gen_counts = Counter(r["generator"] for r in all_rows)
    for g in sorted(gen_counts):
        print(f"  {g}: {gen_counts[g]}")

    columns = [
        "path", "label", "generator", "speaker", "sentence_id",
        "accent_style", "source_format", "group_id", "fold"
    ]
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nManifest written to {OUTPUT}")


if __name__ == "__main__":
    main()
