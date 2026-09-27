#!/usr/bin/env python3
"""
GPU-accelerated HEARSAY deepfake detection pipeline for GT PACE cluster.

Extracts WavLM multi-layer features (all 13 layers) + LFCC features,
trains XGBoost classifiers, ensembles, optimizes for minDCF, generates submission.

Usage:
    python pace_pipeline.py \
        --data-root ~/scratch/hearsay_data \
        --test-dir ~/scratch/hearsay_test/HackGTHearsayTesting \
        --cache-dir ~/scratch/hearsay_cache

    # Final submission (train on ALL data, no validation hold-out):
    python pace_pipeline.py --no-val ...
"""

import argparse, csv, os, sys, time, pickle, random, random
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import torch

SAMPLE_RATE = 16000
SEED = 42

# ────────────────────────────────────────────────────────────
# Audio loading (inlined from audio_utils.py for portability)
# ────────────────────────────────────────────────────────────

def load_audio(path, target_sr=16000):
    import soundfile as sf
    import soxr
    data, sr = sf.read(path, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = soxr.resample(data, sr, target_sr, quality='HQ')
    return data, target_sr


# ────────────────────────────────────────────────────────────
# Manifest loading + Windows→Linux path remapping
# ────────────────────────────────────────────────────────────

def remap_path(win_path, data_root):
    p = win_path.replace('\\', '/')
    for prefix in ['C:/hearsay_data/', 'c:/hearsay_data/']:
        if p.lower().startswith(prefix.lower()):
            p = p[len(prefix):]
            break
    return os.path.join(data_root, p)


def load_manifest(manifest_path, data_root, val_fold=0, no_val=False):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['path'] = remap_path(r['path'], data_root)

    if no_val:
        print(f"Manifest: {len(rows)} files (ALL data, no validation hold-out)")
        return rows, []

    train = [r for r in rows if int(r['fold']) != val_fold]
    val = [r for r in rows if int(r['fold']) == val_fold]

    tr_real = sum(1 for r in train if r['label'] == 'bonafide')
    tr_fake = sum(1 for r in train if r['label'] == 'spoof')
    va_real = sum(1 for r in val if r['label'] == 'bonafide')
    va_fake = sum(1 for r in val if r['label'] == 'spoof')
    print(f"Train: {len(train)} ({tr_real} real, {tr_fake} fake)")
    print(f"Val fold {val_fold}: {len(val)} ({va_real} real, {va_fake} fake)")
    return train, val


def sample_rows(rows, max_fake=0, max_real=0):
    """Subsample training rows. Fake samples are balanced across generators.
    max_fake/max_real of 0 means use all."""
    rng = random.Random(SEED)

    fake = [r for r in rows if r['label'] == 'spoof']
    real = [r for r in rows if r['label'] == 'bonafide']

    if max_fake > 0 and len(fake) > max_fake:
        by_gen = {}
        for r in fake:
            by_gen.setdefault(r['generator'], []).append(r)
        n_gens = len(by_gen)
        per_gen = max_fake // n_gens
        remainder = max_fake % n_gens

        sampled_fake = []
        for i, (gen, files) in enumerate(sorted(by_gen.items())):
            cap = per_gen + (1 if i < remainder else 0)
            if len(files) > cap:
                files = rng.sample(files, cap)
            sampled_fake.extend(files)
        fake = sampled_fake

    if max_real > 0 and len(real) > max_real:
        real = rng.sample(real, max_real)

    result = fake + real
    rng.shuffle(result)

    n_fake = sum(1 for r in result if r['label'] == 'spoof')
    n_real = sum(1 for r in result if r['label'] == 'bonafide')
    by_gen = {}
    for r in result:
        if r['label'] == 'spoof':
            by_gen[r['generator']] = by_gen.get(r['generator'], 0) + 1
    print(f"  Sampled: {len(result)} ({n_real} real, {n_fake} fake)")
    for g in sorted(by_gen):
        print(f"    {g}: {by_gen[g]}")
    return result


def collect_test_files(test_dir):
    test_dir = Path(test_dir)
    files = sorted(test_dir.glob("*.wav"))
    rows = [{'path': str(p), 'label': 'unknown', 'fold': '-1'} for p in files]
    filenames = [p.name for p in files]
    print(f"Test: {len(files)} files in {test_dir}")
    return rows, filenames


# ────────────────────────────────────────────────────────────
# WavLM multi-layer extraction (GPU)
# ────────────────────────────────────────────────────────────

def extract_wavlm(rows, cache_path, device, batch_size=32, max_sec=5.0, desc="WavLM"):
    if os.path.exists(cache_path):
        print(f"  [{desc}] Loading cached: {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return d['X'], d['y'], list(d.get('paths', []))

    from transformers import WavLMModel, Wav2Vec2FeatureExtractor

    print(f"  [{desc}] Loading WavLM model → {device}...")
    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    n_layers = model.config.num_hidden_layers + 1  # +1 for CNN feature output
    feat_dim = n_layers * model.config.hidden_size * 2  # mean+std per layer
    print(f"  [{desc}] {n_layers} layers, extracting mean+std per layer → {feat_dim}d")

    max_samples = int(max_sec * SAMPLE_RATE)
    all_embs, all_labels, all_paths = [], [], []
    failed = 0
    t0 = time.time()

    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start:batch_start + batch_size]
        batch_wavs, batch_indices = [], []

        for idx, r in enumerate(batch_rows):
            try:
                wav, _ = load_audio(r['path'])
                if len(wav) > max_samples:
                    wav = wav[:max_samples]
                if len(wav) < SAMPLE_RATE // 2:
                    failed += 1
                    batch_wavs.append(None)
                    continue
                batch_wavs.append(wav)
                batch_indices.append(idx)
            except Exception:
                failed += 1
                batch_wavs.append(None)

        valid_wavs = [w for w in batch_wavs if w is not None]
        if not valid_wavs:
            for r in batch_rows:
                all_embs.append(np.zeros(feat_dim, dtype=np.float32))
                all_labels.append(0 if r.get('label') == 'bonafide' else
                                  1 if r.get('label') == 'spoof' else -1)
                all_paths.append(r['path'])
            continue

        input_lengths = torch.tensor([len(w) for w in valid_wavs], dtype=torch.long)

        max_len = max(len(w) for w in valid_wavs)
        padded = np.zeros((len(valid_wavs), max_len), dtype=np.float32)
        for j, w in enumerate(valid_wavs):
            padded[j, :len(w)] = w

        inputs = fe(padded.tolist(), sampling_rate=SAMPLE_RATE,
                     return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        output_lengths = model._get_feat_extract_output_lengths(input_lengths).to(device)
        T_out = outputs.hidden_states[0].shape[1]
        mask = (torch.arange(T_out, device=device).unsqueeze(0)
                < output_lengths.unsqueeze(1)).unsqueeze(-1).float()
        lengths = mask.sum(dim=1).clamp(min=1)

        layer_feats = []
        for hidden in outputs.hidden_states:
            masked = hidden * mask
            mean = masked.sum(dim=1) / lengths
            var = ((masked - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / lengths
            std = torch.sqrt(var + 1e-8)
            layer_feats.append(torch.cat([mean, std], dim=-1))

        batch_emb = torch.cat(layer_feats, dim=-1).cpu().numpy()

        valid_idx = 0
        for idx, r in enumerate(batch_rows):
            if batch_wavs[idx] is not None:
                all_embs.append(batch_emb[valid_idx])
                valid_idx += 1
            else:
                all_embs.append(np.zeros(feat_dim, dtype=np.float32))
            all_labels.append(0 if r.get('label') == 'bonafide' else
                              1 if r.get('label') == 'spoof' else -1)
            all_paths.append(r['path'])

        done = batch_start + len(batch_rows)
        if done % (batch_size * 20) == 0 or done >= len(rows):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = (len(rows) - done) / max(rate, 1e-6)
            print(f"  [{desc}] {done}/{len(rows)} | "
                  f"{rate:.0f}/s | ETA {eta:.0f}s | {failed} failed")

    X = np.array(all_embs, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    np.savez(cache_path, X=X, y=y, paths=np.array(all_paths))
    elapsed = time.time() - t0
    print(f"  [{desc}] Done: {X.shape} in {elapsed:.0f}s ({failed} failed) → {cache_path}")

    del model, fe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return X, y, all_paths


# ────────────────────────────────────────────────────────────
# LFCC extraction (CPU, multiprocessing)
# ────────────────────────────────────────────────────────────

def _compute_lfcc(wav, sr=16000, n_fft=512, hop=160, win=400,
                  n_filters=20, n_lfcc=20):
    import librosa
    from scipy.fftpack import dct

    S = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop,
                            win_length=win)) ** 2
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freqs)
    edges = np.linspace(0, sr / 2, n_filters + 2)

    fb = np.zeros((n_filters, n_freqs))
    for i in range(n_filters):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (freqs - lo) / max(mid - lo, 1e-10)
        dn = (hi - freqs) / max(hi - mid, 1e-10)
        fb[i] = np.maximum(0, np.minimum(up, dn))

    filtered = fb @ S + 1e-10
    log_fb = np.log(filtered)
    lfcc = dct(log_fb, type=2, axis=0, norm='ortho')[:n_lfcc]
    delta = librosa.feature.delta(lfcc)
    delta2 = librosa.feature.delta(lfcc, order=2)
    feats = np.vstack([lfcc, delta, delta2])  # (60, T)
    return np.concatenate([feats.mean(axis=1), feats.std(axis=1)])  # (120,)


def _lfcc_worker(args):
    path, max_sec = args
    try:
        wav, sr = load_audio(path)
        if len(wav) > int(max_sec * sr):
            wav = wav[:int(max_sec * sr)]
        return _compute_lfcc(wav, sr)
    except Exception:
        return np.zeros(120, dtype=np.float32)


def extract_lfcc(rows, cache_path, max_sec=5.0, n_workers=8, desc="LFCC"):
    if os.path.exists(cache_path):
        print(f"  [{desc}] Loading cached: {cache_path}")
        d = np.load(cache_path)
        return d['X'], d['y']

    print(f"  [{desc}] Extracting for {len(rows)} files ({n_workers} workers)...")
    t0 = time.time()
    args = [(r['path'], max_sec) for r in rows]

    with Pool(n_workers) as pool:
        results = pool.map(_lfcc_worker, args)

    X = np.array(results, dtype=np.float32)
    labels = []
    for r in rows:
        if r.get('label') == 'bonafide':
            labels.append(0)
        elif r.get('label') == 'spoof':
            labels.append(1)
        else:
            labels.append(-1)
    y = np.array(labels, dtype=np.int32)

    np.savez(cache_path, X=X, y=y)
    print(f"  [{desc}] Done: {X.shape} in {time.time() - t0:.0f}s → {cache_path}")
    return X, y


# ────────────────────────────────────────────────────────────
# minDCF / EER computation
# ────────────────────────────────────────────────────────────

def compute_eer_mindcf(scores, labels, p_spoof=0.3, c_miss=1, c_fa=4):
    target = scores[labels == 1]
    nontarget = scores[labels == 0]
    if len(target) == 0 or len(nontarget) == 0:
        return float('nan'), float('nan'), 0.5

    thresholds = np.sort(np.unique(np.concatenate([target, nontarget])))

    fnr = np.array([np.mean(target < t) for t in thresholds])
    fpr = np.array([np.mean(nontarget >= t) for t in thresholds])

    eer_idx = np.argmin(np.abs(fnr - fpr))
    eer = (fnr[eer_idx] + fpr[eer_idx]) / 2

    dcf = c_miss * p_spoof * fnr + c_fa * (1 - p_spoof) * fpr
    c_default = min(c_miss * p_spoof, c_fa * (1 - p_spoof))
    dcf_norm = dcf / c_default
    min_dcf = np.min(dcf_norm)
    best_thresh = thresholds[np.argmin(dcf_norm)]

    return eer, min_dcf, best_thresh


# ────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────

def train_xgb(X_train, y_train, X_val=None, y_val=None, label="XGB"):
    import xgboost as xgb

    scale = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)

    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric='logloss',
        random_state=SEED,
        n_jobs=-1,
        tree_method='hist',
    )

    fit_kwargs = {}
    if X_val is not None and len(X_val) > 0:
        fit_kwargs['eval_set'] = [(X_val, y_val)]
        fit_kwargs['verbose'] = 50

    print(f"  [{label}] Training on {X_train.shape}...")
    t0 = time.time()
    clf.fit(X_train, y_train, **fit_kwargs)
    print(f"  [{label}] Trained in {time.time() - t0:.1f}s")

    tr_proba = clf.predict_proba(X_train)[:, 1]
    from sklearn.metrics import roc_auc_score, accuracy_score
    print(f"  [{label}] Train AUC: {roc_auc_score(y_train, tr_proba):.4f}")

    if X_val is not None and len(X_val) > 0:
        va_proba = clf.predict_proba(X_val)[:, 1]
        va_labels = y_val
        auc = roc_auc_score(va_labels, va_proba)
        eer, mindcf, thresh = compute_eer_mindcf(va_proba, va_labels)
        fpr = np.mean(va_proba[va_labels == 0] >= 0.5)
        fnr = np.mean(va_proba[va_labels == 1] < 0.5)
        print(f"  [{label}] Val AUC: {auc:.4f} | EER: {eer*100:.2f}% | "
              f"minDCF: {mindcf:.4f} | thresh: {thresh:.4f}")
        print(f"  [{label}] Val FPR: {fpr*100:.1f}% | FNR: {fnr*100:.1f}%")

    return clf


def train_ensemble(scores_dict_train, y_train, scores_dict_val=None, y_val=None):
    from sklearn.linear_model import LogisticRegression

    keys = sorted(scores_dict_train.keys())
    X_train = np.column_stack([scores_dict_train[k] for k in keys])

    meta = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    meta.fit(X_train, y_train)

    tr_proba = meta.predict_proba(X_train)[:, 1]
    from sklearn.metrics import roc_auc_score
    print(f"  [Ensemble] Train AUC: {roc_auc_score(y_train, tr_proba):.4f}")
    print(f"  [Ensemble] Weights: {dict(zip(keys, meta.coef_[0]))}")

    if scores_dict_val is not None:
        X_val = np.column_stack([scores_dict_val[k] for k in keys])
        va_proba = meta.predict_proba(X_val)[:, 1]
        eer, mindcf, thresh = compute_eer_mindcf(va_proba, y_val)
        print(f"  [Ensemble] Val EER: {eer*100:.2f}% | minDCF: {mindcf:.4f} | "
              f"thresh: {thresh:.4f}")

    return meta, keys


# ────────────────────────────────────────────────────────────
# Submission
# ────────────────────────────────────────────────────────────

def write_tsv(filenames, scores, output_path):
    with open(output_path, 'w', newline='') as f:
        f.write("filename\tcm-score\n")
        for name, score in sorted(zip(filenames, scores)):
            f.write(f"{name}\t{score:.6f}\n")
    print(f"  Submission: {output_path} ({len(filenames)} files)")


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HEARSAY GPU pipeline for PACE")
    parser.add_argument('--data-root', required=True,
                        help='Training data root on PACE (replaces C:\\hearsay_data)')
    parser.add_argument('--test-dir', required=True,
                        help='Test WAV directory on PACE')
    parser.add_argument('--cache-dir',
                        default=os.path.expanduser('~/scratch/hearsay_cache'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-audio-sec', type=float, default=5.0)
    parser.add_argument('--val-fold', type=int, default=0)
    parser.add_argument('--no-val', action='store_true',
                        help='Train on ALL data for final submission')
    parser.add_argument('--max-fake', type=int, default=0,
                        help='Total fake samples to use (0 = all). '
                             'Balanced across generators, e.g. 50000 = ~5000/gen')
    parser.add_argument('--max-real', type=int, default=0,
                        help='Total real samples to use (0 = all)')
    parser.add_argument('--skip-wavlm', action='store_true')
    parser.add_argument('--skip-lfcc', action='store_true')
    parser.add_argument('--pca-dims', type=int, default=512)
    parser.add_argument('--n-workers', type=int, default=0,
                        help='CPU workers for LFCC (0 = auto)')
    args = parser.parse_args()

    if args.n_workers <= 0:
        args.n_workers = min(cpu_count(), 16)

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    manifest_path = os.path.join(args.data_root, 'manifest.csv')
    if not os.path.exists(manifest_path):
        print(f"ERROR: manifest not found at {manifest_path}")
        print("Upload manifest.csv to your data root directory.")
        sys.exit(1)

    suffix = "all" if args.no_val else f"fold{args.val_fold}"
    if args.max_fake > 0:
        suffix += f"_f{args.max_fake // 1000}k"
    if args.max_real > 0:
        suffix += f"_r{args.max_real // 1000}k"

    # ── Step 1: Load manifest ──
    print("\n══ Step 1: Load manifest ══")
    train_rows, val_rows = load_manifest(
        manifest_path, args.data_root,
        val_fold=args.val_fold, no_val=args.no_val)

    if args.max_fake > 0 or args.max_real > 0:
        print(f"  Sampling: max_fake={args.max_fake or 'all'}, max_real={args.max_real or 'all'}")
        train_rows = sample_rows(train_rows, args.max_fake, args.max_real)

    test_rows, test_filenames = collect_test_files(args.test_dir)

    # ── Step 2: WavLM multi-layer extraction ──
    print("\n══ Step 2: WavLM multi-layer extraction ══")
    if not args.skip_wavlm:
        wX_train, wy_train, _ = extract_wavlm(
            train_rows, str(cache / f"wavlm13_train_{suffix}.npz"),
            device, args.batch_size, args.max_audio_sec, "WavLM-train")

        if val_rows:
            wX_val, wy_val, _ = extract_wavlm(
                val_rows, str(cache / f"wavlm13_val_{suffix}.npz"),
                device, args.batch_size, args.max_audio_sec, "WavLM-val")
        else:
            wX_val, wy_val = np.array([]), np.array([])

        if test_rows:
            wX_test, _, _ = extract_wavlm(
                test_rows, str(cache / "wavlm13_test.npz"),
                device, args.batch_size, args.max_audio_sec, "WavLM-test")
        else:
            wX_test = np.array([])
    else:
        print("  Skipped (loading from cache)")
        d = np.load(cache / f"wavlm13_train_{suffix}.npz")
        wX_train, wy_train = d['X'], d['y']
        if val_rows:
            d = np.load(cache / f"wavlm13_val_{suffix}.npz")
            wX_val, wy_val = d['X'], d['y']
        else:
            wX_val, wy_val = np.array([]), np.array([])
        test_cache = cache / "wavlm13_test.npz"
        if test_cache.exists():
            d = np.load(test_cache)
            wX_test = d['X']
        else:
            wX_test = np.array([])

    # ── Step 3: PCA on WavLM features ──
    print(f"\n══ Step 3: PCA {wX_train.shape[1]}d → {args.pca_dims}d ══")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=args.pca_dims, svd_solver='randomized', random_state=SEED)
    wX_train_pca = pca.fit_transform(wX_train)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    has_test = wX_test.ndim == 2 and wX_test.shape[0] > 0
    wX_test_pca = pca.transform(wX_test) if has_test else np.array([])
    if len(wX_val) > 0:
        wX_val_pca = pca.transform(wX_val)
    else:
        wX_val_pca = np.array([])
    with open(cache / f"pca_{suffix}.pkl", 'wb') as f:
        pickle.dump(pca, f)

    # ── Step 4: LFCC extraction ──
    print("\n══ Step 4: LFCC extraction ══")
    if not args.skip_lfcc:
        lX_train, ly_train = extract_lfcc(
            train_rows, str(cache / f"lfcc_train_{suffix}.npz"),
            args.max_audio_sec, args.n_workers, "LFCC-train")
        if val_rows:
            lX_val, ly_val = extract_lfcc(
                val_rows, str(cache / f"lfcc_val_{suffix}.npz"),
                args.max_audio_sec, args.n_workers, "LFCC-val")
        else:
            lX_val, ly_val = np.array([]), np.array([])
        if test_rows:
            lX_test, _ = extract_lfcc(
                test_rows, str(cache / "lfcc_test.npz"),
                args.max_audio_sec, args.n_workers, "LFCC-test")
        else:
            lX_test = np.array([])
    else:
        print("  Skipped (loading from cache)")
        d = np.load(cache / f"lfcc_train_{suffix}.npz")
        lX_train, ly_train = d['X'], d['y']
        if val_rows:
            d = np.load(cache / f"lfcc_val_{suffix}.npz")
            lX_val, ly_val = d['X'], d['y']
        else:
            lX_val, ly_val = np.array([]), np.array([])
        lfcc_test_cache = cache / "lfcc_test.npz"
        if lfcc_test_cache.exists():
            d = np.load(lfcc_test_cache)
            lX_test = d['X']
        else:
            lX_test = np.array([])

    # ── Step 5: Train WavLM XGBoost ──
    print("\n══ Step 5: Train WavLM XGBoost ══")
    wavlm_xgb = train_xgb(
        wX_train_pca, wy_train,
        wX_val_pca if len(wX_val_pca) > 0 else None,
        wy_val if len(wy_val) > 0 else None,
        "WavLM-XGB")
    wavlm_xgb.save_model(str(cache / f"wavlm13_xgb_{suffix}.json"))

    # ── Step 6: Train LFCC XGBoost ──
    print("\n══ Step 6: Train LFCC XGBoost ══")
    lfcc_xgb = train_xgb(
        lX_train, ly_train,
        lX_val if len(lX_val) > 0 else None,
        ly_val if len(ly_val) > 0 else None,
        "LFCC-XGB")
    lfcc_xgb.save_model(str(cache / f"lfcc_xgb_{suffix}.json"))

    # ── Step 7: Ensemble ──
    print("\n══ Step 7: Ensemble ══")
    train_scores = {
        'wavlm': wavlm_xgb.predict_proba(wX_train_pca)[:, 1],
        'lfcc': lfcc_xgb.predict_proba(lX_train)[:, 1],
    }

    if len(wX_val_pca) > 0:
        val_scores = {
            'wavlm': wavlm_xgb.predict_proba(wX_val_pca)[:, 1],
            'lfcc': lfcc_xgb.predict_proba(lX_val)[:, 1],
        }
        meta, meta_keys = train_ensemble(train_scores, wy_train,
                                         val_scores, wy_val)
    else:
        meta, meta_keys = train_ensemble(train_scores, wy_train)

    with open(cache / f"ensemble_{suffix}.pkl", 'wb') as f:
        pickle.dump({'meta': meta, 'keys': meta_keys}, f)

    # ── Step 8: Score test files ──
    print("\n══ Step 8: Score test files ══")
    if has_test:
        test_scores_dict = {
            'wavlm': wavlm_xgb.predict_proba(wX_test_pca)[:, 1],
            'lfcc': lfcc_xgb.predict_proba(lX_test)[:, 1],
        }
        X_test_meta = np.column_stack([test_scores_dict[k] for k in meta_keys])
        final_scores = meta.predict_proba(X_test_meta)[:, 1]

        for name, sc in test_scores_dict.items():
            write_tsv(test_filenames, sc,
                      str(cache / f"{name}_submission_{suffix}.tsv"))

        write_tsv(test_filenames, final_scores,
                  str(cache / f"ensemble_submission_{suffix}.tsv"))

        print(f"  Score stats: min={final_scores.min():.4f} max={final_scores.max():.4f} "
              f"median={np.median(final_scores):.4f} mean={final_scores.mean():.4f}")
        print(f"  Predicted fake (>0.5): {(final_scores > 0.5).sum()}")
        print(f"  Predicted real (≤0.5): {(final_scores <= 0.5).sum()}")
    else:
        print("  No test files found — skipping submission generation.")
        print("  Upload test WAVs to --test-dir and re-run with --skip-wavlm --skip-lfcc")

    # ── Summary ──
    print("\n══ Summary ══")
    if len(wX_val_pca) > 0:
        print("\n  === Validation Results ===")
        for name in meta_keys:
            sc = val_scores[name]
            eer, mindcf, th = compute_eer_mindcf(sc, wy_val)
            print(f"  {name:8s}  EER={eer*100:.2f}%  minDCF={mindcf:.4f}  thresh={th:.4f}")
        ens_val = meta.predict_proba(
            np.column_stack([val_scores[k] for k in meta_keys]))[:, 1]
        eer, mindcf, th = compute_eer_mindcf(ens_val, wy_val)
        print(f"  {'ensemble':8s}  EER={eer*100:.2f}%  minDCF={mindcf:.4f}  thresh={th:.4f}")

    print(f"\n  All outputs in: {cache}")
    print("  Done!")


if __name__ == '__main__':
    main()
