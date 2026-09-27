# HEARSAY Session Handoff v3 — HackGT 2026

**Created:** 2026-09-27  
**Project:** HEARSAY challenge (NSA @ HackGT 13)  
**Working Directory (local):** `C:\Users\himan\OneDrive - Georgia Institute of Technology\Desktop\Hackgt 2026\hearsay_hgt13`  
**Working Directory (PACE):** `~/scratch/hearsay_hgt13/`  
**Previous handoffs:** `handoff.md` (data inventory), `handoff_v2.md` (first attempt analysis + improvement plan)

---

## 1. Challenge Overview

Classify 1,671 English `.wav` files (16 kHz, >3s) as **real** (score `0.0`) or **synthetic** (score `1.0`).

- **Output format:** tab-separated `.tsv` — columns: `filename\tcm-score`
- **Metric:** minDCF with `Pspoof=0.3`, `Cmiss=1`, `Cfa=4`
- **~70% of test files are real**
- **Score convention:** `0 = real`, `1 = synthetic` — **CONFIRMED by sponsor, do NOT flip**
- **Cfa=4 means false alarms (real called fake) cost 4× more than missing fakes**

---

## 2. Current Leaderboard (Interim)

| Rank | Team | minDCF | EER |
|------|------|--------|-----|
| 1 | Team 3 | 0.0584 | 2.5% |
| 2 | Team 4 | 0.0753 | 3.53% |
| 3 | Team 1 | 0.258 | 10.18% |
| 4 | Team 2 | 0.267 | 10.4% |
| **5** | **Us** | **0.913** | **34.6%** |

**Target:** minDCF < 0.258 to beat Team 1. minDCF < 0.058 to win.

---

## 3. What Changed Since v2

### New: GPU Pipeline on GT PACE Cluster

We now have access to **NVIDIA H100 80GB GPUs** via GT PACE ICE cluster. A complete GPU-accelerated pipeline was built and is ready to run.

**Key improvements over the v2 CPU pipeline:**
1. **WavLM multi-layer extraction** — all 13 transformer layers (not just the last), with proper masked mean+std pooling per layer → 19,968-dim features, PCA'd to 512
2. **LFCC features** — Linear Frequency Cepstral Coefficients + deltas + double-deltas → 120-dim complementary features
3. **Ensemble** — XGBoost on WavLM + XGBoost on LFCC → stacking meta-classifier (LogisticRegression)
4. **All training data** — uses all 94k+ files (controllable with `--max-fake` / `--max-real`)
5. **minDCF-aware evaluation** — computes EER and minDCF on validation, reports optimal threshold

### New: People's Speech Data (15,000 diverse real samples)

Downloaded 15,000 real speech samples from HuggingFace `MLCommons/peoples_speech` (clean config) to diversify the real speech training data. This addresses the critical problem that 54% of real training data was a single speaker (LJ Speech).

**Updated real data breakdown:**

| Source | Files | Speakers |
|--------|-------|----------|
| LJ Speech | 13,342 | 1 (single woman) |
| LibriSpeech | 11,126 | 146 |
| People's Speech | 15,000 | Thousands (diverse) |
| **Total real** | **39,468** | **~1,000+** |

LJ dominance drops from 54% → 34% of real data.

---

## 4. Code Files

### Local (`hearsay_hgt13/`)

| File | Status | Purpose |
|------|--------|---------|
| `pace_pipeline.py` | **NEW, MAIN** | GPU pipeline: WavLM 13-layer + LFCC + XGBoost ensemble |
| `run_hearsay.sbatch` | **NEW** | SLURM job script for PACE (H100, 4hr, 8 CPUs) |
| `setup_pace.sh` | **NEW** | One-time PACE setup (venv, torch+CUDA, model download) |
| `download_real_speech.py` | **NEW** | Streams People's Speech from HuggingFace |
| `requirements_pace.txt` | **NEW** | PACE Python dependencies |
| `audio_utils.py` | unchanged | Content-based audio loader |
| `build_manifest.py` | unchanged | Builds manifest CSV |
| `rfp_detector.py` | unchanged | RFP detector (not used in PACE pipeline) |
| `wavlm_pipeline.py` | SUPERSEDED | Old CPU pipeline (last-layer only) |
| `ProcessFiles.py` | DO NOT USE | Teammate's harmful MP3 converter |
| `TestAASIST.py` | REFERENCE ONLY | Old AASIST test |

### PACE Cluster (`~/scratch/`)

| Path | Contents |
|------|----------|
| `hearsay_hgt13/` | Code (uploaded as zip) |
| `hearsay_data/` | Training data |
| `hearsay_data/manifest.csv` | Master index (94,468 → ~109,468 rows after People's Speech) |
| `hearsay_data/generated_speech/` | **NEEDS UPLOAD** — 70k fake (10 generators) |
| `hearsay_data/LJSpeech-1.1/` | 13k real (1 speaker) — present |
| `hearsay_data/librispeech/LibriSpeech/` | 11k real (146 speakers) — extracted from tarballs |
| `hearsay_data/LJRealResampled/` | **NEEDS UPLOAD** — 242 real wav |
| `hearsay_data/peoples_speech/` | 15k diverse real — **downloaded and ready** |
| `hearsay_data/raw/` | LibriSpeech tarballs (already extracted, can delete) |
| `hearsay_test/HackGTHearsayTesting/` | **NEEDS UPLOAD** — 1,671 competition test files |
| `hearsay_venv/` | Python venv with torch+CUDA, transformers, etc. |
| `hearsay_cache/` | Pipeline outputs (embeddings, models, submissions) |

---

## 5. PACE Pipeline Details

### How to run

```bash
# SSH into PACE
ssh hgalundia3@login-ice.pace.gatech.edu

# Setup (one-time, already done)
cd ~/scratch/hearsay_hgt13
bash setup_pace.sh

# Submit GPU job
cd ~/scratch
sbatch hearsay_hgt13/run_hearsay.sbatch

# Monitor
squeue -u hgalundia3
cat hearsay-<jobid>.out
```

### Pipeline phases (pace_pipeline.py)

1. Load manifest, remap Windows→Linux paths, sample training data
2. WavLM multi-layer extraction (GPU, batch_size=32) — all 13 layers, masked mean+std pooling
3. PCA: 19,968d → 512d (randomized SVD)
4. LFCC extraction (CPU, 8 workers) — 20 LFCCs + deltas + double-deltas → 120d
5. Train XGBoost on WavLM features (scale_pos_weight for class balance)
6. Train XGBoost on LFCC features
7. Ensemble: stack WavLM + LFCC scores → LogisticRegression meta-classifier
8. Evaluate on validation (minDCF, EER) and generate submission TSVs

### Key CLI arguments

| Arg | Default | Purpose |
|-----|---------|---------|
| `--data-root` | required | Training data root on PACE |
| `--test-dir` | required | Test WAV directory |
| `--cache-dir` | `~/scratch/hearsay_cache` | Outputs |
| `--batch-size` | 32 | GPU batch size for WavLM |
| `--max-fake` | 0 (all) | Cap total fake samples (balanced across generators) |
| `--max-real` | 0 (all) | Cap total real samples |
| `--val-fold` | 0 | Hold-out fold for validation |
| `--no-val` | false | Train on ALL data (final submission) |
| `--skip-wavlm` | false | Use cached WavLM embeddings |
| `--skip-lfcc` | false | Use cached LFCC features |
| `--pca-dims` | 512 | PCA output dimensions |

### Output files in `hearsay_cache/`

| File | Description |
|------|-------------|
| `wavlm13_train_*.npz` | WavLM 13-layer embeddings (train) |
| `wavlm13_val_*.npz` | WavLM 13-layer embeddings (val) |
| `wavlm13_test.npz` | WavLM 13-layer embeddings (test) |
| `lfcc_train_*.npz` | LFCC features (train) |
| `lfcc_val_*.npz` | LFCC features (val) |
| `lfcc_test.npz` | LFCC features (test) |
| `pca_*.pkl` | Fitted PCA model |
| `wavlm13_xgb_*.json` | Trained WavLM XGBoost |
| `lfcc_xgb_*.json` | Trained LFCC XGBoost |
| `ensemble_*.pkl` | Trained ensemble meta-classifier |
| `wavlm_submission_*.tsv` | WavLM-only submission |
| `lfcc_submission_*.tsv` | LFCC-only submission |
| `ensemble_submission_*.tsv` | **Final ensemble submission** |

---

## 6. What Still Needs to Happen

### Immediate (before pipeline can run)

1. **Upload `generated_speech/`** to `~/scratch/hearsay_data/` — the 70k fake files (use Globus)
2. **Upload `LJRealResampled/`** to `~/scratch/hearsay_data/`
3. **Upload `manifest.csv`** to `~/scratch/hearsay_data/`
4. **Upload test files** to `~/scratch/hearsay_test/HackGTHearsayTesting/`
5. **Update manifest** with People's Speech data:
   ```bash
   cd ~/scratch/hearsay_hgt13
   python download_real_speech.py update-manifest \
       --manifest ~/scratch/hearsay_data/manifest.csv \
       --audio-dir ~/scratch/hearsay_data/peoples_speech
   ```
6. **Submit job**: `cd ~/scratch && sbatch hearsay_hgt13/run_hearsay.sbatch`

### After first run

7. Check validation results (EER, minDCF) in the job output log
8. Submit the `ensemble_submission_*.tsv` to the competition
9. If results are good, re-run with `--no-val` for final submission (trains on all data)
10. Tune `--max-fake` if needed (default 50,000 in sbatch)

### Possible further improvements (if time allows)

- Fine-tune WavLM backend layers (requires gradient computation, more GPU time)
- Add RFP scores to the ensemble (third signal)
- Codec augmentation (encode-decode training data through MP3/Opus)
- Try different PCA dimensions (256, 1024)
- Try `--max-fake 30000 --max-real 30000` for a balanced dataset

---

## 7. Environment

### Local

| Component | Value |
|-----------|-------|
| Python | 3.13.11 (Anaconda) |
| PyTorch | 2.11.0+cpu (**NO GPU**) |
| OS | Windows 11 |

### PACE Cluster

| Component | Value |
|-----------|-------|
| Python | 3.10 (anaconda3 module) |
| PyTorch | 2.6.0+cu124 |
| GPU | NVIDIA H100 80GB HBM3 |
| CUDA Driver | 595.71.05 |
| SLURM | sbatch job submission |
| Login | `hgalundia3@login-ice.pace.gatech.edu` |
| Web Portal | `https://ondemand-ice.pace.gatech.edu` |
| Scratch | `~/scratch/` = `/storage/ice1/3/2/hgalundia3/` |
| Job limits | Max 16 GPU-hours, max 50 GPU jobs, max 32 concurrent GPU-hours |

---

## 8. Data Locations

### Local (Windows)
```
C:\hearsay_data\                          ← All training data
├── generated_speech\                     ← 70k fake (10 generators)
├── LJRealResampled\                      ← 242 real wav (16 kHz)
├── LJSpeech-1.1\LJSpeech-1.1\wavs\      ← 10,961 real wav (22.05 kHz)
├── librispeech\LibriSpeech\              ← 11,126 real FLAC (16 kHz)
└── manifest.csv                          ← Master index

C:\hearsay_cache\                         ← Old CPU pipeline outputs

hearsay_hgt13\                            ← Code repo
├── test\HackGTHearsayTesting\            ← 1,671 competition test files
└── (all code files listed in Section 4)
```

### PACE (Linux)
```
~/scratch/
├── hearsay_hgt13/                        ← Code (zipped and uploaded)
├── hearsay_data/                         ← Training data
│   ├── manifest.csv                      ← NEEDS UPLOAD
│   ├── generated_speech/                 ← NEEDS UPLOAD (70k fake)
│   ├── LJSpeech-1.1/LJSpeech-1.1/       ← Present
│   ├── librispeech/LibriSpeech/          ← Extracted from tarballs
│   ├── LJRealResampled/                  ← NEEDS UPLOAD
│   ├── peoples_speech/                   ← 15k downloaded from HuggingFace
│   └── raw/                             ← LibriSpeech tarballs (can delete)
├── hearsay_test/HackGTHearsayTesting/    ← NEEDS UPLOAD (1,671 test files)
├── hearsay_venv/                         ← Python venv (ready)
└── hearsay_cache/                        ← Pipeline outputs
```

---

## 9. Scorer Details (unchanged)

- **Use:** `Baseline-AASIST/eval/calculate_metrics.py` with `Pspoof=0.3`, `Cmiss=1`, `Cfa=4`
- **DO NOT use:** `evaluation-package/calculate_metrics.py`
- **Score convention:** `0 = real`, `1 = synthetic`
- **minDCF formula:** `DCF = 0.3 * FRR + 2.8 * FAR`, normalized by `min(0.3, 2.8) = 0.3`

---

## 10. Key Lessons Learned

1. **High val accuracy ≠ good test performance.** 95.8% val accuracy gave 34.6% EER on test.
2. **Use all transformer layers, not just the last.** Lower layers capture acoustic/codec artifacts.
3. **Mean-pooling alone is too lossy.** Use mean + std with proper attention masking.
4. **Speaker diversity in real data matters.** 54% from one speaker caused massive false alarm rate.
5. **The features matter more than the classifier.** LR vs XGBoost made small differences; features are the bottleneck.
6. **False alarms are 4× more expensive.** Bias toward "real" — only call fake when very confident.
7. **Score direction is 0=real, 1=fake.** Confirmed by sponsor. Do not flip.

---

## 11. Prompt for New Session

> Read `handoff_v3.md` in `C:\Users\himan\OneDrive - Georgia Institute of Technology\Desktop\Hackgt 2026\hearsay_hgt13\`. I'm competing in HEARSAY deepfake speech detection at HackGT 2026 (currently last place, minDCF 0.913). We built a GPU pipeline (`pace_pipeline.py`) for GT PACE with H100 GPUs that extracts WavLM 13-layer features + LFCC, trains an XGBoost ensemble, and generates submissions. We also downloaded 15k diverse real speech from People's Speech. Still need to upload `generated_speech/`, `LJRealResampled/`, `manifest.csv`, and test files to PACE before submitting. Working directory is `hearsay_hgt13/`, PACE scratch is `~/scratch/`.
