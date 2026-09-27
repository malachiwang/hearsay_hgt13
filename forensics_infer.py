#!/usr/bin/env python3
"""
Batch inference using eliya/forensics_0.3B_base_deepfake_classifier.
WavLM-large + AASIST graph-attention deepfake detector (0.3B params).

Model architecture (from HF repo model.py):
  DeepfakeDetector:
    - self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    - self.pool  = AASISTPool(hidden_size=1024, num_layers=24)
    - self.classifier = Classifier(input_dim=320, hidden_dim=320, layers=3) → 1 logit
  Forward:
    hidden_states = wavlm(waveform, output_hidden_states=True).hidden_states[1:]
    pooled = AASISTPool(hidden_states)   # → (batch, 320)
    logit  = classifier(pooled)           # → (batch,)
  Score:
    fake_probability = 1 - sigmoid(logit)

Score convention: 0 = real, 1 = fake (matches HEARSAY challenge).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf


SAMPLE_RATE = 16000
TARGET_SAMPLES = 5 * SAMPLE_RATE  # model expects exactly 5 seconds


def load_audio(path):
    """Load audio → mono 16kHz float32, normalized, padded/trimmed to 5s."""
    wav, sr = sf.read(str(path), dtype='float32')
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav).float()
    if sr != SAMPLE_RATE:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    # Peak-normalize (matches the model's own inference.py)
    wav = wav / (wav.abs().max() + 1e-8)
    if wav.shape[0] > TARGET_SAMPLES:
        wav = wav[:TARGET_SAMPLES]
    elif wav.shape[0] < TARGET_SAMPLES:
        wav = torch.nn.functional.pad(wav, (0, TARGET_SAMPLES - wav.shape[0]))
    return wav


def strip_state_dict_prefix(sd):
    """Remove 'module.' prefix from DataParallel/DDP checkpoints."""
    cleaned = {}
    for k, v in sd.items():
        key = k[len('module.'):] if k.startswith('module.') else k
        cleaned[key] = v
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Forensics deepfake detector batch inference")
    parser.add_argument('--test-dir', required=True, help='Directory with test WAV files')
    parser.add_argument('--model-dir', required=True, help='Downloaded HF model directory')
    parser.add_argument('--output', required=True, help='Output submission TSV path')
    parser.add_argument('--batch-size', type=int, default=8, help='GPU batch size (8 is safe for H100)')
    parser.add_argument('--xgb-tsv', default=None, help='Optional: XGBoost submission TSV to ensemble with')
    parser.add_argument('--ensemble-weight', type=float, default=0.7,
                        help='Weight for forensics model in ensemble (default 0.7)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

    # ── Step 1: Load model ──
    print("\n══ Loading DeepfakeDetector (WavLM-large + AASIST) ══")

    # Import DeepfakeDetector from the model's own model.py
    sys.path.insert(0, args.model_dir)
    from model import DeepfakeDetector

    model = DeepfakeDetector()

    # Load checkpoint — prefer safetensors (1.27 GB) over .pt (3.81 GB)
    safetensors_path = os.path.join(args.model_dir, 'checkpoint_epoch_5.safetensors')
    pt_path = os.path.join(args.model_dir, 'checkpoint_epoch_5.pt')

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        size_gb = os.path.getsize(safetensors_path) / 1e9
        print(f"  Loaded: checkpoint_epoch_5.safetensors ({size_gb:.2f} GB)")
    elif os.path.exists(pt_path):
        ckpt = torch.load(pt_path, map_location='cpu', weights_only=True)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(f"  Loaded: checkpoint_epoch_5.pt")
    else:
        print("ERROR: No checkpoint found in model dir!")
        print(f"  Expected: {safetensors_path}")
        print(f"  Contents: {os.listdir(args.model_dir)}")
        sys.exit(1)

    state_dict = strip_state_dict_prefix(state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"  Warning: {len(load_result.missing_keys)} missing keys (may be normal for WavLM)")
    if load_result.unexpected_keys:
        print(f"  Warning: {len(load_result.unexpected_keys)} unexpected keys")

    model = model.to(device).eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model ready: {param_count:.0f}M parameters on {device}")

    # ── Step 2: Collect test files ──
    test_dir = Path(args.test_dir)
    wav_files = sorted(test_dir.glob('*.wav'))
    print(f"\n══ Found {len(wav_files)} test files in {test_dir} ══")

    if not wav_files:
        print("ERROR: No .wav files found in test directory!")
        sys.exit(1)

    # ── Step 3: Batch inference ──
    print(f"\n══ Running inference (batch_size={args.batch_size}) ══")
    filenames = []
    scores = []
    failed = 0
    t0 = time.time()

    for i in range(0, len(wav_files), args.batch_size):
        batch_paths = wav_files[i:i + args.batch_size]
        batch_wavs = []
        batch_names = []

        for p in batch_paths:
            try:
                wav = load_audio(p)
                batch_wavs.append(wav)
                batch_names.append(p.name)
            except Exception as e:
                print(f"  FAILED to load {p.name}: {e}")
                filenames.append(p.name)
                scores.append(0.0)  # bias toward real — Cfa=4 penalizes false alarms 4x
                failed += 1

        if not batch_wavs:
            continue

        batch = torch.stack(batch_wavs).to(device)

        with torch.no_grad():
            logits = model(batch)  # (batch_size,) — raw logits
            # Model convention: fake_probability = 1 - sigmoid(logit)
            fake_probs = (1.0 - torch.sigmoid(logits)).cpu().numpy().flatten()

        for name, score in zip(batch_names, fake_probs):
            filenames.append(name)
            scores.append(float(np.clip(score, 0.0, 1.0)))

        done = len(filenames)
        if done % (args.batch_size * 10) == 0 or i + args.batch_size >= len(wav_files):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1)
            eta = (len(wav_files) - done) / max(rate, 0.01)
            print(f"  {done}/{len(wav_files)} | {rate:.1f} files/s | ETA {eta:.0f}s | {failed} failed")

    elapsed = time.time() - t0
    scores_arr = np.array(scores)

    # ── Step 4: Write submission TSV ──
    print(f"\n══ Writing submission ══")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        for name, score in sorted(zip(filenames, scores)):
            f.write(f"{name}\t{score:.6f}\n")
    print(f"  → {args.output}")

    # ── Step 5: Ensemble with XGBoost (optional) ──
    if args.xgb_tsv and os.path.exists(args.xgb_tsv):
        print(f"\n══ Ensembling with XGBoost scores ══")
        xgb_scores = {}
        with open(args.xgb_tsv, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    xgb_scores[parts[0]] = float(parts[1])

        matched = 0
        ensemble_out = args.output.replace('.tsv', '_ensemble.tsv')
        w = args.ensemble_weight  # forensics weight
        with open(ensemble_out, 'w') as f:
            for name, forensics_score in sorted(zip(filenames, scores)):
                if name in xgb_scores:
                    combined = w * forensics_score + (1 - w) * xgb_scores[name]
                    matched += 1
                else:
                    combined = forensics_score
                f.write(f"{name}\t{combined:.6f}\n")

        print(f"  Matched {matched}/{len(filenames)} files with XGBoost scores")
        print(f"  Forensics weight: {w}, XGBoost weight: {1-w}")
        print(f"  → {ensemble_out}")

    # ── Summary ──
    print(f"\n══ Summary ══")
    print(f"  Inference time: {elapsed:.0f}s ({len(filenames)/max(elapsed,1):.1f} files/s)")
    print(f"  Files scored: {len(filenames)}, Failed: {failed}")
    print(f"  Score range: [{scores_arr.min():.4f}, {scores_arr.max():.4f}]")
    print(f"  Mean: {scores_arr.mean():.4f}, Median: {np.median(scores_arr):.4f}")
    print(f"  Predicted FAKE (>0.5): {(scores_arr > 0.5).sum()}")
    print(f"  Predicted REAL (<=0.5): {(scores_arr <= 0.5).sum()}")
    print(f"  Done!")


if __name__ == '__main__':
    main()
