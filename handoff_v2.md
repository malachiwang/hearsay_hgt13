# HEARSAY Session Handoff v2 — HackGT 2026

**Created:** 2026-09-26  
**Project:** HEARSAY challenge (NSA @ HackGT 13)  
**Working Directory:** `C:\Users\himan\OneDrive - Georgia Institute of Technology\Desktop\Hackgt 2026\hearsay_hgt13`  
**Previous handoff:** `handoff.md` (still valid for data inventory, environment, scorer details)

---

## 1. Challenge Overview (unchanged)

Classify 1,671 English `.wav` files (16 kHz, >3s) as **real** (score `0.0`) or **synthetic** (score `1.0`).

- **Output format:** tab-separated `.tsv` — columns: `filename\tcm-score`
- **Metric:** minDCF with `Pspoof=0.3`, `Cmiss=1`, `Cfa=4`
- **~70% of test files are real**
- **Score convention:** `0 = real`, `1 = synthetic` — **CONFIRMED by sponsor, this is correct, do NOT flip scores**
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

**Sponsor feedback on our submission:** *"Your minDCF was .913 and EER was 34.6. The algorithm weights False Alarms very heavily so anything you can do to diversify data real and synthetic sources and knock down false alarms would help."*

---

## 3. What Was Built So Far

### Code files in `hearsay_hgt13/`

| File | Status | Purpose |
|------|--------|---------|
| `audio_utils.py` | DONE | Content-based audio loader using soundfile + soxr |
| `build_manifest.py` | DONE | Builds manifest CSV for all training data |
| `rfp_detector.py` | DONE | Tier 0: Residual Fingerprint detector |
| `wavlm_pipeline.py` | DONE | Tier 1: Frozen WavLM + classifier pipeline |
| `ProcessFiles.py` | DO NOT USE | Teammate's harmful MP3 converter |
| `TestAASIST.py` | REFERENCE ONLY | Old AASIST test, hardcoded paths |

### Cached files in `C:\hearsay_cache\`

| File | Size | Description |
|------|------|-------------|
| `wavlm_train_fold0.npz` | 39.7 MB | 13,507 train embeddings (768-dim, last layer only, mean-pooled) |
| `wavlm_val_fold0.npz` | 4.4 MB | 1,493 val embeddings (fold 0 = diffgan_tts held out) |
| `wavlm_test.npz` | 5 MB | 1,671 test embeddings |
| `wavlm_lr_fold0.pkl` | <1 MB | Trained LogisticRegression model |
| `wavlm_mlp_fold0.pkl` | 2.5 MB | Trained MLP model (256→64) |
| `wavlm_xgb_fold0.json` | 1.1 MB | Trained XGBoost model |
| `rfp_v2.pkl` | 6.1 MB | RFP detector fingerprints (real + fake) |
| `*_submission_*.tsv` | <1 MB | Various submission files |

### Manifest

`C:\hearsay_data\manifest.csv` — 94,468 rows:
- 70,000 fake (10 generators × 5,000-25,000 each)
- 24,468 real (LJ Speech + LibriSpeech)
- Columns: `path, label, generator, speaker, sentence_id, accent_style, source_format, group_id, fold`
- 10 folds (leave-one-generator-out)

---

## 4. What Was Attempted & Results

### Attempt 1: Frozen WavLM (last layer) + mean-pool + classifiers

**Training setup:**
- Sampled only 1,000 per fake generator (10k fake total) + 5,000 real = ~15k samples
- Held out fold 0 (diffgan_tts) for validation
- WavLM-base-plus frozen, extracted 768-dim mean-pooled embeddings from LAST layer only
- Truncated audio to 5 seconds

**Validation results (fold 0, on diffgan_tts):**

| Classifier | Val Acc | Val AUC | FPR | FNR |
|------------|---------|---------|-----|-----|
| LogisticRegression | 93.4% | 0.984 | 6.5% | 6.6% |
| MLP (256→64) | 92.5% | 0.993 | 0.8% | 10.8% |
| XGBoost | 95.8% | 0.993 | 3.7% | 4.5% |

**Submitted:** XGBoost submission → **minDCF 0.913, EER 34.6%**

**Why it failed (massive val→test gap):**
1. **Only used last transformer layer** — lower layers capture acoustic/codec artifacts crucial for detection
2. **Mean-pooling destroys temporal information** — averaging across time loses patterns
3. **Too little training data** — 1,000 per generator is not enough
4. **Model learned DiffSSD-specific artifacts** — doesn't generalize to unseen generators in test set
5. **Test distribution differs** — test set likely has different generators/speakers/recording conditions

### Attempt 0: RFP Detector

- Built Residual Fingerprint detector using spectral residuals + Mahalanobis distance
- Score stats on test: min=0.12, max=0.88, median=0.48
- Not submitted independently (used as Tier 0 concept only)
- Known weakness: fragile under codec compression

---

## 5. What Must Change — Improvement Plan

### Priority 1: Better WavLM Feature Extraction (BIGGEST IMPACT)

**Use all 13 transformer layers, not just the last:**
```python
outputs = model(**inputs, output_hidden_states=True)
# outputs.hidden_states = tuple of 13 tensors, each (B, T, 768)
# Option A: Weighted sum (learnable or fixed weights)
# Option B: Concatenate statistics from each layer
```

**Statistical pooling instead of mean-only:**
```python
# Per layer: mean + std + max = 768 × 3 = 2,304 dims per layer
# All 13 layers: 2,304 × 13 = 29,952 dims (then PCA to ~500-1000)
# Or weighted-sum layers first, then stats: 2,304 dims
```

**Why this matters:** Lower WavLM layers capture low-level acoustic features (formant structure, phase, codec artifacts). Higher layers capture linguistic/speaker features. Spoof detection needs BOTH — synthesizers often get the linguistics right but leave acoustic traces.

### Priority 2: LFCC Features (Complementary Signal)

**Linear Frequency Cepstral Coefficients** — the standard feature for spoof detection:
```python
# Can be computed with librosa (already installed):
# 1. Compute linear-frequency filterbank (not mel)
# 2. Apply DCT to get cepstral coefficients
# 3. Include deltas and delta-deltas
# Result: ~60-dim feature per frame, summarize with stats → ~360-dim per clip
```

**Why this matters:** LFCC captures frequency-domain artifacts that are complementary to WavLM. AASIST (the state-of-the-art) uses LFCC as its primary input feature. It's fast to compute (no GPU needed), and captures codec/vocoder signatures.

### Priority 3: Use ALL Training Data

- Use **all 5,000 per generator** (not 1,000) = 70,000 fake
- Use **all 22,000+ real** files (LJ + LibriSpeech)
- For final submission: **train on ALL folds** (no held-out generator)
- Estimated extraction time: ~3 hours for WavLM (all 92k files)

### Priority 4: Ensemble

Combine all signals:
```
WavLM multi-layer scores  ─┐
LFCC classifier scores     ├─→ Stacking classifier → final score
RFP scores                 ─┘
```

### Priority 5: Reduce False Alarms

Since Cfa=4, false alarms cost 4× more:
- After training, tune the decision threshold on validation to minimize minDCF, not accuracy
- Bias the model toward "real" — only call something fake when very confident
- Use `class_weight='balanced'` or adjust with custom weights favoring real class

---

## 6. Practical Implementation Order

Given CPU-only and hackathon time pressure:

### Step 1: LFCC pipeline (~20 min total)
- Extract LFCC features for all training data (~15 min, pure numpy/scipy)
- Extract for test data (~2 min)
- Train XGBoost on LFCC → submit
- **This is the fastest path to a better score**

### Step 2: Better WavLM features (~2-3 hours extraction)
- Re-extract with all 13 layers + statistical pooling
- Use ALL training data (70k fake + 22k real)
- Train XGBoost → submit

### Step 3: Ensemble (~5 min after Steps 1-2)
- Combine LFCC + WavLM + RFP scores
- Train stacking classifier
- Optimize fusion weights for minDCF → submit

### Alternative: Run LFCC (Step 1) while WavLM re-extracts (Step 2) in parallel

---

## 7. Environment (unchanged from handoff v1)

| Component | Value |
|-----------|-------|
| Python | 3.13.11 (Anaconda) |
| PyTorch | 2.11.0+cpu (**NO GPU**) |
| transformers | 5.6.2 |
| soundfile | 0.13.1 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| librosa | 0.11.0 |
| soxr | installed |
| CPU cores | 20 |

WavLM model (`microsoft/wavlm-base-plus`) is already cached locally by transformers — no re-download needed.

---

## 8. Data Locations (unchanged)

```
C:\hearsay_data\                          ← All training data
├── generated_speech\                     ← 70k fake (10 generators)
├── LJRealResampled\                      ← 242 real wav (16 kHz)
├── LJSpeech-1.1\LJSpeech-1.1\wavs\      ← 10,961 real wav (22.05 kHz)
├── librispeech\LibriSpeech\              ← 11,126 real FLAC (16 kHz)
└── manifest.csv                          ← Master index (94,468 rows)

C:\hearsay_cache\                         ← Cached embeddings and models

hearsay_hgt13\                            ← Code repo
├── audio_utils.py                        ← Audio loader
├── build_manifest.py                     ← Manifest builder
├── rfp_detector.py                       ← RFP detector
├── wavlm_pipeline.py                     ← WavLM pipeline (needs improvement)
└── test\HackGTHearsayTesting\            ← 1,671 competition test files
```

---

## 9. Scorer Details (unchanged)

- **Use:** `Baseline-AASIST/eval/calculate_metrics.py` with `Pspoof=0.3`, `Cmiss=1`, `Cfa=4`
- **DO NOT use:** `evaluation-package/calculate_metrics.py`
- **Score convention:** `0 = real`, `1 = synthetic`
- **minDCF formula:** `cost = 0.7 * FRR + 1.2 * FAR`, normalized by 0.7
- `compute_eer(target_scores, nontarget_scores)` → returns 4 values (eer, frr, far, thresholds)
- `compute_mindcf(frr, far, thresholds, 0.3, 1, 4)` → returns (min_dcf, threshold)

---

## 10. Key Lessons Learned

1. **High val accuracy ≠ good test performance.** Our 95.8% val accuracy gave 34.6% EER on the real test. Leave-one-generator-out validation only tests generalization to that ONE generator.
2. **Mean-pooling the last layer is too lossy.** The temporal and layer-wise information matters enormously for detection.
3. **The classifier matters less than the features.** LR vs MLP vs XGBoost made small differences (93-96% val). The features themselves are the bottleneck.
4. **False alarms are expensive.** Cfa=4 means every real file called fake costs 4× more than missing a fake. The model must be conservative.
5. **Score direction is 0=real, 1=fake.** Confirmed by sponsor. Do not flip.
6. **DiffSSD paper showed 3% EER is achievable** with Wav2Vec2 on this data — but with proper features and training.

---

## 11. Prompt for New Session

Paste this at the start of your new Claude session:

> Read the file `C:\Users\himan\OneDrive - Georgia Institute of Technology\Desktop\Hackgt 2026\hearsay_hgt13\handoff_v2.md` for current status, and `handoff.md` (same directory) for data inventory details. I'm competing in the HEARSAY deepfake speech detection challenge at HackGT 2026. We're currently last place (minDCF 0.913, EER 34.6%). The improvement plan is in Section 5-6 of handoff_v2.md. Start with Step 1 (LFCC pipeline) immediately — it's the fastest path to a better score. Then start Step 2 (better WavLM features with all 13 layers + statistical pooling + all training data). Working directory is `C:\Users\himan\OneDrive - Georgia Institute of Technology\Desktop\Hackgt 2026\hearsay_hgt13`, data at `C:\hearsay_data`, cache at `C:\hearsay_cache`.

---

## 12. Research Notes

### What top teams are likely using
- Frozen SSL frontend (Wav2Vec2 / WavLM / HuBERT) with **multi-layer weighted features**
- LFCC or CQT as complementary spectral features
- AASIST-style graph attention backend, or well-tuned XGBoost
- Codec augmentation during training (MP3/Opus encode-decode both classes)
- Diverse real speech sources

### Papers confirmed relevant
- **DiffSSD (2024):** Our training data IS DiffSSD. Wav2Vec2 got 3% EER when properly trained.
- **ASVspoof 5 (2024):** Pretrained frontend → trainable backend → scoring. Multi-layer features critical.
- **SUPERB benchmark:** Shows that weighted sum of all transformer layers consistently outperforms last-layer-only for downstream tasks.

### Features the user researched (from internet)
- **CQCC:** Constant-Q Cepstral Coefficients — geometric frequency resolution, good for vocoder artifacts
- **CQT:** Constant-Q Transform — preserves phase/frequency clues
- **LFCC:** Linear Frequency Cepstral Coefficients — broad linear filterbank separation
- **ResNet/Res2Net:** On spectrograms for structural anomalies
- **Patched Spectrogram Transformers:** Segments mel-specs into patches
- **RawNet:** End-to-end on raw waveform

### Dataset the user found
- **People's Speech (MLCommons):** `https://huggingface.co/datasets/MLCommons/peoples_speech` — diverse real speech. **WARNING: 30+ TB, impractical for hackathon download.** We already have 22k real files from 147 speakers.
