#!/usr/bin/env python3
"""
Download diverse real speech from People's Speech (HuggingFace) via streaming.
No need to download the full 30TB — this streams and saves only what we need.

Run on PACE login node:
    python download_real_speech.py download --output-dir ~/scratch/hearsay_data/peoples_speech --n-samples 15000

Then re-build the manifest:
    python download_real_speech.py update-manifest --manifest ~/scratch/hearsay_data/manifest.csv \
                                   --audio-dir ~/scratch/hearsay_data/peoples_speech
"""

import argparse
import os
import sys
import csv
import time
import random
import numpy as np

os.environ["DATASETS_AUDIO_DECODER"] = "soundfile"

SAMPLE_RATE = 16000
MAX_DURATION_SEC = 8.0
MIN_DURATION_SEC = 1.5


def download_samples(output_dir, n_samples=15000, dataset_name="MLCommons/peoples_speech",
                     split="train", config="clean"):
    """Stream People's Speech and save n_samples as 16kHz WAV files."""
    from datasets import load_dataset
    from datasets.features import Audio
    import soundfile as sf
    import io

    os.makedirs(output_dir, exist_ok=True)

    existing = set(os.listdir(output_dir))
    already = sum(1 for f in existing if f.endswith('.wav'))
    if already >= n_samples:
        print(f"Already have {already} files in {output_dir}, skipping download.")
        return already

    print(f"Streaming {dataset_name} (config={config}, split={split})...")
    print(f"Target: {n_samples} samples → {output_dir}")

    try:
        ds = load_dataset(dataset_name, config, split=split, streaming=True)
    except Exception as e:
        print(f"Failed to load with config={config}: {e}")
        print("Trying without config...")
        ds = load_dataset(dataset_name, split=split, streaming=True)

    # Disable the built-in audio decoder — we decode with soundfile ourselves
    ds = ds.cast_column("audio", Audio(decode=False))

    saved = already
    skipped = 0
    t0 = time.time()

    for i, sample in enumerate(ds):
        if saved >= n_samples:
            break

        try:
            audio = sample.get('audio', None)
            if audio is None:
                skipped += 1
                continue

            # audio is {'bytes': b'...', 'path': 'something.opus'} with decode=False
            audio_bytes = audio.get('bytes', None)
            if audio_bytes is None:
                skipped += 1
                continue

            wav, sr = sf.read(io.BytesIO(audio_bytes), dtype='float32')

            if len(wav) / sr < MIN_DURATION_SEC:
                skipped += 1
                continue

            if wav.ndim > 1:
                wav = wav.mean(axis=1)

            if sr != SAMPLE_RATE:
                try:
                    import soxr
                    wav = soxr.resample(wav, sr, SAMPLE_RATE, quality='HQ')
                except ImportError:
                    import librosa
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)

            max_samples = int(MAX_DURATION_SEC * SAMPLE_RATE)
            if len(wav) > max_samples:
                wav = wav[:max_samples]

            filename = f"ps_{saved:06d}.wav"
            if filename in existing:
                saved += 1
                continue

            filepath = os.path.join(output_dir, filename)
            sf.write(filepath, wav, SAMPLE_RATE, subtype='PCM_16')
            saved += 1

            if saved % 500 == 0:
                elapsed = time.time() - t0
                rate = saved / max(elapsed, 1)
                eta = (n_samples - saved) / max(rate, 0.01)
                print(f"  {saved}/{n_samples} saved ({skipped} skipped, "
                      f"{rate:.1f}/s, ETA {eta:.0f}s)")

        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  Error on sample {i}: {e}")

    elapsed = time.time() - t0
    print(f"\nDone: {saved} files saved, {skipped} skipped in {elapsed:.0f}s")
    print(f"Output: {output_dir}")
    return saved


def update_manifest(manifest_path, audio_dir, output_path=None):
    """Add downloaded People's Speech files to the manifest."""
    if output_path is None:
        output_path = manifest_path

    with open(manifest_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    existing_paths = {r['path'] for r in rows}
    columns = rows[0].keys() if rows else [
        "path", "label", "generator", "speaker", "sentence_id",
        "accent_style", "source_format", "group_id", "fold"
    ]

    wav_files = sorted(f for f in os.listdir(audio_dir) if f.endswith('.wav'))
    new_count = 0
    rng = random.Random(42)

    for f in wav_files:
        fpath = os.path.join(audio_dir, f)
        if fpath in existing_paths:
            continue
        rows.append({
            "path": fpath,
            "label": "bonafide",
            "generator": "real",
            "speaker": f"ps_{f.replace('.wav', '')}",
            "sentence_id": f.replace('.wav', ''),
            "accent_style": "",
            "source_format": "wav",
            "group_id": f"peoples_speech_{f.replace('.wav', '')}",
            "fold": rng.randint(0, 9),
        })
        new_count += 1

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)

    total_real = sum(1 for r in rows if r['label'] == 'bonafide')
    total_fake = sum(1 for r in rows if r['label'] == 'spoof')
    print(f"Added {new_count} People's Speech files to manifest")
    print(f"New totals: {total_real} real, {total_fake} fake ({len(rows)} total)")
    print(f"Manifest: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download diverse real speech for HEARSAY")
    sub = parser.add_subparsers(dest='command')

    dl = sub.add_parser('download', help='Download audio from HuggingFace')
    dl.add_argument('--output-dir', required=True, help='Where to save WAV files')
    dl.add_argument('--n-samples', type=int, default=15000, help='Number of samples')
    dl.add_argument('--dataset', default='MLCommons/peoples_speech')
    dl.add_argument('--config', default='clean', help='Dataset config (clean/dirty)')
    dl.add_argument('--split', default='train')

    um = sub.add_parser('update-manifest', help='Add downloaded files to manifest')
    um.add_argument('--manifest', required=True, help='Path to manifest.csv')
    um.add_argument('--audio-dir', required=True, help='Directory with downloaded WAVs')
    um.add_argument('--output', default=None, help='Output manifest (default: overwrite)')

    # Also support flat args for simpler usage
    parser.add_argument('--output-dir', help='Download destination')
    parser.add_argument('--n-samples', type=int, default=15000)
    parser.add_argument('--update-manifest', help='Manifest to update')
    parser.add_argument('--audio-dir', help='Audio dir for manifest update')

    args = parser.parse_args()

    if args.command == 'download':
        download_samples(args.output_dir, args.n_samples, args.dataset,
                         args.split, args.config)
    elif args.command == 'update-manifest':
        update_manifest(args.manifest, args.audio_dir, args.output)
    elif args.output_dir:
        download_samples(args.output_dir, args.n_samples)
        if args.update_manifest and args.audio_dir:
            update_manifest(args.update_manifest, args.audio_dir or args.output_dir)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Download 15k samples:")
        print("  python download_real_speech.py download \\")
        print("      --output-dir ~/scratch/hearsay_data/peoples_speech --n-samples 15000")
        print("")
        print("  # Add them to manifest:")
        print("  python download_real_speech.py update-manifest \\")
        print("      --manifest ~/scratch/hearsay_data/manifest.csv \\")
        print("      --audio-dir ~/scratch/hearsay_data/peoples_speech")


if __name__ == '__main__':
    main()
