"""
Tier 1: Frozen WavLM-base-plus → mean-pooled embeddings → LogisticRegression

Usage:
    python wavlm_pipeline.py                  # full pipeline
    python wavlm_pipeline.py --extract-only   # just extract embeddings
    python wavlm_pipeline.py --train-only     # train + eval (embeddings must exist)
"""

import argparse
import csv
import time
import random
from pathlib import Path

import numpy as np
import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import pickle

from audio_utils import load_audio

# --- Config ---
MANIFEST_PATH = Path(r"C:\hearsay_data\manifest.csv")
CACHE_DIR = Path(r"C:\hearsay_cache")
TEST_DIR = Path(r"C:\Users\himan\OneDrive - Georgia Institute of Technology"
                r"\Desktop\Hackgt 2026\hearsay_hgt13\test\HackGTHearsayTesting")

MODEL_NAME = "microsoft/wavlm-base-plus"
SAMPLE_PER_GEN = 1000       # fake clips per generator
SAMPLE_REAL = 5000           # real clips total
VAL_FOLD = 0                 # hold out fold 0 (diffgan_tts) for validation
BATCH_SIZE = 8               # forward pass batch size
MAX_AUDIO_SEC = 5.0          # truncate to 5s to save memory/time
SAMPLE_RATE = 16000
SEED = 42


def load_manifest(manifest_path, sample_per_gen, sample_real, val_fold):
    """Load manifest, sample, and split into train/val."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    fake_by_gen = {}
    real_files = []
    for r in rows:
        if r['label'] == 'spoof':
            fake_by_gen.setdefault(r['generator'], []).append(r)
        else:
            real_files.append(r)

    rng = random.Random(SEED)

    sampled = []
    for gen, files in sorted(fake_by_gen.items()):
        if len(files) > sample_per_gen:
            files = rng.sample(files, sample_per_gen)
        sampled.extend(files)

    if len(real_files) > sample_real:
        real_files = rng.sample(real_files, sample_real)
    sampled.extend(real_files)

    train = [r for r in sampled if int(r['fold']) != val_fold]
    val = [r for r in sampled if int(r['fold']) == val_fold]

    n_train_real = sum(1 for r in train if r['label'] == 'bonafide')
    n_train_fake = sum(1 for r in train if r['label'] == 'spoof')
    n_val_real = sum(1 for r in val if r['label'] == 'bonafide')
    n_val_fake = sum(1 for r in val if r['label'] == 'spoof')

    print(f"Train: {len(train)} files ({n_train_real} real, {n_train_fake} fake)")
    print(f"Val (fold {val_fold}): {len(val)} files ({n_val_real} real, {n_val_fake} fake)")
    return train, val


def load_model():
    """Load frozen WavLM model and feature extractor."""
    print(f"Loading {MODEL_NAME}...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = WavLMModel.from_pretrained(MODEL_NAME)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, frozen)")
    return model, feature_extractor


def extract_embeddings(model, feature_extractor, rows, desc="Extracting"):
    """Extract mean-pooled WavLM embeddings for a list of manifest rows."""
    max_samples = int(MAX_AUDIO_SEC * SAMPLE_RATE)
    embeddings = []
    labels = []
    paths_out = []
    failed = 0
    t0 = time.time()

    batch_wavs = []
    batch_rows = []

    def flush_batch():
        nonlocal batch_wavs, batch_rows
        if not batch_wavs:
            return
        max_len = max(len(w) for w in batch_wavs)
        padded = np.zeros((len(batch_wavs), max_len), dtype=np.float32)
        for j, w in enumerate(batch_wavs):
            padded[j, :len(w)] = w

        inputs = feature_extractor(
            padded.tolist(), sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state  # (B, T, 768)
        pooled = hidden.mean(dim=1).numpy()  # (B, 768)

        for j, r in enumerate(batch_rows):
            embeddings.append(pooled[j])
            labels.append(0 if r['label'] == 'bonafide' else 1)
            paths_out.append(r['path'])

        batch_wavs = []
        batch_rows = []

    for i, r in enumerate(rows):
        try:
            wav, _ = load_audio(r['path'])
            if len(wav) > max_samples:
                wav = wav[:max_samples]
            batch_wavs.append(wav)
            batch_rows.append(r)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  skip {Path(r['path']).name}: {e}")

        if len(batch_wavs) >= BATCH_SIZE:
            flush_batch()

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(rows) - i - 1) / rate
            print(f"  {desc}: {i+1}/{len(rows)} "
                  f"({rate:.1f} files/s, ~{remaining:.0f}s left)")

    flush_batch()

    elapsed = time.time() - t0
    print(f"  {desc}: done — {len(embeddings)} embeddings in {elapsed:.1f}s "
          f"({failed} failed)")
    return np.array(embeddings), np.array(labels), paths_out


def extract_test_embeddings(model, feature_extractor, test_dir):
    """Extract embeddings for competition test files."""
    test_files = sorted(test_dir.glob("*.wav"))
    print(f"\nExtracting embeddings for {len(test_files)} test files...")

    rows = [{'path': str(p), 'label': 'unknown'} for p in test_files]
    max_samples = int(MAX_AUDIO_SEC * SAMPLE_RATE)
    embeddings = []
    filenames = []
    t0 = time.time()

    batch_wavs = []
    batch_names = []

    def flush():
        nonlocal batch_wavs, batch_names
        if not batch_wavs:
            return
        max_len = max(len(w) for w in batch_wavs)
        padded = np.zeros((len(batch_wavs), max_len), dtype=np.float32)
        for j, w in enumerate(batch_wavs):
            padded[j, :len(w)] = w

        inputs = feature_extractor(
            padded.tolist(), sampling_rate=SAMPLE_RATE,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1).numpy()
        for j in range(len(batch_names)):
            embeddings.append(pooled[j])
            filenames.append(batch_names[j])
        batch_wavs = []
        batch_names = []

    for i, p in enumerate(test_files):
        try:
            wav, _ = load_audio(str(p))
            if len(wav) > max_samples:
                wav = wav[:max_samples]
            batch_wavs.append(wav)
            batch_names.append(p.name)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            filenames.append(p.name)
            embeddings.append(np.zeros(768, dtype=np.float32))

        if len(batch_wavs) >= BATCH_SIZE:
            flush()

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  Test: {i+1}/{len(test_files)} ({rate:.1f} files/s)")

    flush()
    elapsed = time.time() - t0
    print(f"  Test embeddings: {len(embeddings)} in {elapsed:.1f}s")
    return np.array(embeddings), filenames


def train_and_evaluate(X_train, y_train, X_val, y_val):
    """Train LogisticRegression and evaluate."""
    print(f"\nTraining LogisticRegression on {X_train.shape[0]} samples, "
          f"{X_train.shape[1]} features...")

    clf = LogisticRegression(
        C=1.0, max_iter=1000, solver='lbfgs', random_state=SEED
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"Training done in {time.time() - t0:.1f}s")

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    train_proba = clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    print(f"Train — acc: {train_acc:.4f}, AUC: {train_auc:.4f}")

    if len(X_val) > 0:
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        val_proba = clf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        print(f"Val   — acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        n_val_real = (y_val == 0).sum()
        n_val_fake = (y_val == 1).sum()
        val_preds = clf.predict(X_val)
        real_correct = ((val_preds == 0) & (y_val == 0)).sum()
        fake_correct = ((val_preds == 1) & (y_val == 1)).sum()
        print(f"  Real correct: {real_correct}/{n_val_real} "
              f"({real_correct/max(n_val_real,1)*100:.1f}%)")
        print(f"  Fake correct: {fake_correct}/{n_val_fake} "
              f"({fake_correct/max(n_val_fake,1)*100:.1f}%)")

    return clf


def write_tsv(scores, output_path):
    """Write submission TSV: filename<tab>cm-score"""
    with open(output_path, 'w', newline='') as f:
        f.write("filename\tcm-score\n")
        for name in sorted(scores):
            f.write(f"{name}\t{scores[name]:.6f}\n")
    print(f"Submission written to {output_path} ({len(scores)} files)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract-only', action='store_true',
                        help='Only extract embeddings, skip training')
    parser.add_argument('--train-only', action='store_true',
                        help='Use cached embeddings, skip extraction')
    parser.add_argument('--val-fold', type=int, default=VAL_FOLD,
                        help=f'Validation fold (default: {VAL_FOLD})')
    parser.add_argument('--sample-per-gen', type=int, default=SAMPLE_PER_GEN)
    parser.add_argument('--sample-real', type=int, default=SAMPLE_REAL)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    emb_train_path = CACHE_DIR / f"wavlm_train_fold{args.val_fold}.npz"
    emb_val_path = CACHE_DIR / f"wavlm_val_fold{args.val_fold}.npz"
    emb_test_path = CACHE_DIR / "wavlm_test.npz"
    model_path = CACHE_DIR / f"wavlm_lr_fold{args.val_fold}.pkl"
    output_tsv = CACHE_DIR / f"wavlm_submission_fold{args.val_fold}.tsv"

    # --- Extract embeddings ---
    if not args.train_only:
        train_rows, val_rows = load_manifest(
            MANIFEST_PATH, args.sample_per_gen, args.sample_real, args.val_fold
        )
        model, fe = load_model()

        print(f"\n--- Extracting train embeddings ---")
        X_train, y_train, _ = extract_embeddings(model, fe, train_rows, "Train")
        np.savez(emb_train_path, X=X_train, y=y_train)
        print(f"Saved train embeddings to {emb_train_path}")

        print(f"\n--- Extracting val embeddings ---")
        X_val, y_val, _ = extract_embeddings(model, fe, val_rows, "Val")
        np.savez(emb_val_path, X=X_val, y=y_val)
        print(f"Saved val embeddings to {emb_val_path}")

        print(f"\n--- Extracting test embeddings ---")
        X_test, test_filenames = extract_test_embeddings(model, fe, TEST_DIR)
        np.savez(emb_test_path, X=X_test, filenames=test_filenames)
        print(f"Saved test embeddings to {emb_test_path}")

        del model, fe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if args.extract_only:
        print("\nExtraction complete. Run with --train-only to train.")
        return

    # --- Train + evaluate ---
    print(f"\n--- Loading cached embeddings ---")
    train_data = np.load(emb_train_path)
    X_train, y_train = train_data['X'], train_data['y']

    val_data = np.load(emb_val_path)
    X_val, y_val = val_data['X'], val_data['y']

    clf = train_and_evaluate(X_train, y_train, X_val, y_val)

    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")

    # --- Score test files ---
    print(f"\n--- Scoring test files ---")
    test_data = np.load(emb_test_path, allow_pickle=True)
    X_test = test_data['X']
    test_filenames = list(test_data['filenames'])

    test_proba = clf.predict_proba(X_test)[:, 1]  # P(fake)
    scores = {name: float(prob) for name, prob in zip(test_filenames, test_proba)}

    vals = list(scores.values())
    print(f"Score stats: min={min(vals):.4f}, max={max(vals):.4f}, "
          f"median={np.median(vals):.4f}, mean={np.mean(vals):.4f}")
    print(f"Predicted fake (>0.5): {sum(1 for v in vals if v > 0.5)}")
    print(f"Predicted real (<0.5): {sum(1 for v in vals if v < 0.5)}")

    write_tsv(scores, output_tsv)
    print(f"\nDone! Submission at: {output_tsv}")


if __name__ == '__main__':
    main()
