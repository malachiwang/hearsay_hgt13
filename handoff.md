# HEARSAY Session Handoff — HackGT 2026

**Created:** 2026-09-26  
**Project:** HEARSAY challenge (NSA @ HackGT 13)  
**Working Directory:** `C:\Users\himan\OneDrive - Georgia Institute of Technology\Desktop\Hackgt 2026\hearsay_hgt13`

---

## 1. Challenge Overview

Classify 1,671 English `.wav` files (16 kHz, >3s) as **real** (score `0.0`) or **synthetic** (score `1.0`).

- **Output format:** tab-separated `.tsv` — columns: `filename\tcm-score`
- **Metric:** minDCF with `Pspoof=0.3`, `Cmiss=1`, `Cfa=4`
- **~70% of test files are real**
- **Score convention:** `0 = real`, `1 = synthetic` (answer key format)
- **Hackathon timeline:** 18 hours total

### Submission template

Located at: `hearsay_hgt13/test/HackGTHearsayTesting/HGT_Hearsay_score_template.csv`

```
filename	cm-score
HGT1013455.wav	0.006
HGT1027213.wav	0.006
...
```

1,671 rows, all pre-filled with 0.006. Replace `cm-score` with model predictions.

---

## 2. minDCF Scoring — Critical Details

### Formula

```
cost = Cmiss * FRR * (1 - Pspoof) + Cfa * FAR * Pspoof
     = 1 * FRR * 0.7 + 4 * FAR * 0.3
     = 0.7 * FRR + 1.2 * FAR

normalized by: min(Cmiss * (1 - Pspoof), Cfa * Pspoof) = min(0.7, 1.2) = 0.7

minDCF = min over all thresholds of (cost / 0.7)
```

- **0 = perfect, 1 = no better than trivial**
- Only **ranking** matters (threshold-free), not absolute score values
- Missing a fake (false negative) costs 1.7× more than flagging a real file (false positive)

### Which scorer file to use

The **sponsor confirmed** they use the **Baseline-AASIST** scorer, NOT the evaluation-package one.

| File | Location | Status |
|------|----------|--------|
| **USE THIS** | `asvspoof5/Baseline-AASIST/eval/calculate_metrics.py` | Defaults are WRONG — must set `Pspoof=0.05→0.3`, `Cfa=10→4` |
| DO NOT USE | `asvspoof5/evaluation-package/calculate_metrics.py` | Different API, different return values |

### Score direction

The AASIST scorer treats **higher score = more bonafide (real)**.  
The answer key uses **0 = real, 1 = synthetic**.  
**When evaluating locally, flip scores: `1 - score`.**

### Scorer API (Baseline-AASIST version)

```python
# Input file format: space-separated, 4 columns
# spk  utt  score  key
# -    -    float  bonafide|spoof

# compute_eer(target_scores, nontarget_scores)
# → Returns: (eer, frr, far, thresholds)  ← 4 values, NOT 5

# compute_mindcf(frr, far, thresholds, Pspoof, Cmiss, Cfa)
# → Returns: (min_dcf, threshold)

# In compute_det_curve: target = bonafide, nontarget = spoof
# FRR = bonafide rejected, FAR = spoof accepted
```

### Known bug in AASIST main.py

Line 133 of `Baseline-AASIST/main.py` swaps minDCF and EER when unpacking the return tuple. This only affects their training loop, not standalone evaluation.

---

## 3. Data Inventory

All large data lives at `C:\hearsay_data` (NOT inside the git repo / OneDrive).

### Fake speech — `C:\hearsay_data\generated_speech\`

| Generator | Files | Format | Notes |
|-----------|-------|--------|-------|
| diffgan_tts | 5,000 | wav | — |
| elevenlabs | 5,000 | mp3 | Actual `.mp3` files (header `\xff\xfb`) |
| grad_tts | 5,000 | wav | — |
| openvoicev2 | 25,000 | wav | 5 accent duplicates per sentence — MUST deduplicate |
| playht | 5,000 | wav | MP3 inside `.wav` containers (header starts with `ID3`) |
| pro_diff | 5,000 | wav | — |
| unit_speech | 5,000 | wav | — |
| wavegrad2 | 5,000 | wav | — |
| xtts_v2 | 5,000 | wav | — |
| your_tts | 5,000 | wav | — |
| **Total** | **70,000** | — | **10 generators** |

**Folder structure per generator:** `<generator>/speaker_<id>/<sentence>.wav`  
**Speakers:** 100, 1487, 2061, 3654, 4490, 5448, 6167, 6575, 7995, 8848  
(These do NOT match LibriSpeech speaker IDs)

### Real speech — `C:\hearsay_data\LJRealResampled\`

- 242 `.wav` files (16 kHz, mono, 22-bit → resampled from LJ Speech)
- Single speaker (LJ)
- Also contains `LJRealResampled.tar` (can be deleted)

### Real speech — `C:\hearsay_data\LJSpeech-1.1\LJSpeech-1.1\`

- **10,961** `.wav` files (22.05 kHz, mono)
- Full LJ Speech dataset downloaded from `keithito.com/LJ-Speech-Dataset/`
- Single speaker (LJ)
- `metadata.csv` format: `LJ001-0001|text|normalized_text` (pipe-separated)
- Needs resampling to 16 kHz before use

### Real speech — `C:\hearsay_data\librispeech\LibriSpeech\`

| Subset | Utterances | Speakers |
|--------|-----------|----------|
| dev-clean | 2,703 | 40 |
| dev-other | 2,864 | 33 |
| test-clean | 2,620 | 40 |
| test-other | 2,939 | 33 |
| **Total** | **11,126** | **146** |

- Format: 16 kHz mono FLAC
- Structure: `<subset>/<speaker_id>/<chapter_id>/<speaker_id>-<chapter_id>-<utterance_id>.flac`
- Durations: 1.6 to 29.4s, median 5.2s

### Real speech totals

| Source | Files | Speakers | Sample Rate |
|--------|-------|----------|-------------|
| LJRealResampled | 242 | 1 | 16 kHz |
| LJ Speech (full) | 10,961 | 1 | 22.05 kHz (needs resample) |
| LibriSpeech | 11,126 | 146 | 16 kHz |
| **Total** | **22,329** | **147** | — |

### Test data — `hearsay_hgt13/test/HackGTHearsayTesting/`

- **1,671** `.wav` files (16 kHz, mono, 16-bit PCM)
- Duration: 3.02–13.58s, median 3.41s
- Filenames: `HGT<digits>.wav`

### Raw archives — `C:\hearsay_data\raw\`

~1.3 GB of `.tar.gz` files (LibriSpeech). Can be deleted to save space.

---

## 4. Environment

| Component | Value |
|-----------|-------|
| Python | 3.13.11 (Anaconda) |
| PyTorch | 2.11.0+cpu (**NO GPU**) |
| transformers | 5.6.2 |
| soundfile | 0.13.1 (has MP3/Opus support) |
| scikit-learn | 1.8.0 |
| pandas | 3.0.0 |
| scipy | 1.17.0 |
| librosa | 0.11.0 |
| soxr | installed |
| torchaudio | **NOT installed** |
| speechbrain | **NOT installed** |
| ffmpeg/ffprobe | **NOT on PATH** |
| CPU cores | 20 |
| Free disk | ~58 GB |

### CPU performance benchmarks

- WavLM-base-plus forward pass: **0.114 s/clip**
- WavLM forward + backward: **0.57 s/clip**
- Fine-tuning WavLM: ~40 hrs (infeasible on CPU)
- Frozen WavLM features + trainable head: feasible

---

## 5. Data Format Gotchas

### PlayHT files (5,000 in `playht/`)
- Extension: `.wav` but **header starts with `ID3`** (MP3 inside WAV container)
- `soundfile.read()` handles them correctly → returns 24 kHz audio
- `librosa.load()` also works (uses soundfile backend)
- **Never convert to MP3 first** — that erases detection artifacts

### ElevenLabs files (5,000 in `elevenlabs/`)
- Extension: `.mp3` (actual MP3 files)
- Header: `\xff\xfb`
- `soundfile.read()` handles them (libsndfile 1.2.2 supports MP3)

### ProcessFiles.py is harmful — DO NOT USE
The teammate's `ProcessFiles.py` converts everything to MP3 before loading. This **erases the very artifacts** we need for detection. It also has a bug with `rsplit('.')[0]` breaking on dots in paths. Replace it entirely with a content-based decoder using `soundfile`.

### Correct audio loading approach

```python
import soundfile as sf
import soxr
import numpy as np

def load_audio(path, target_sr=16000):
    """Content-based decoder: reads any format soundfile supports."""
    data, sr = sf.read(path, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = soxr.resample(data, sr, target_sr, quality='HQ')
    return data, sr  # return original sr too (it's a feature)
```

---

## 6. Shortcut Risks Identified

These are data leakage patterns that would give high train accuracy but fail on real test data:

1. **Sampling rate differs per generator** (16k–44.1k) — resample ALL to 16k
2. **PlayHT/ElevenLabs are MP3-based** while real is PCM — apply codec augmentation to both classes
3. **OpenVoice has 5 accent duplicates per sentence** — deduplicate or split by sentence, not randomly
4. **Only 1 real speaker** (LJ) vs many fake voices — add LibriSpeech speakers to balance
5. **Duration, silence patterns, peak amplitude** can leak class info

### Mitigations

- Resample everything to 16 kHz
- Apply codec augmentation (MP3/Opus encode-decode) to both real and fake during training
- Split validation by generator (leave-one-generator-out) to test generalization
- Use LibriSpeech to add speaker diversity to real class
- Crop/pad all clips to a fixed window (3s)

---

## 7. Agreed Architecture

### Three-tier system

```
Tier 0: Cheap features (RFP, spectral stats, peak amplitude)
    ↓ scores
Tier 1: Frozen WavLM-base-plus → mean-pool → LogisticRegression
    ↓ scores  
Tier 2: Frozen WavLM-base-plus → trainable MLP head (2-layer, ~200k params)
    ↓
Score Fusion (weighted average or stacking)
    ↓
Final cm-score per file
```

### Tier 0 — Cheap features (RFP + spectral)

- Residual Fingerprint (RFP): training-free, spectral residuals. 99%+ AUROC on clean data, fragile under MP3/reverb
- Additional: sample rate, silence ratio, peak amplitude, spectral centroid
- Used as supplementary signal, not primary
- Logistic regression or simple threshold

### Tier 1 — Frozen WavLM + LR (the "get a score fast" tier)

- Load `microsoft/wavlm-base-plus` (94M params, 13 layers, 768-dim)
- Forward pass all training clips (frozen, no grad) → save 768-dim mean-pooled embeddings
- Train `sklearn.linear_model.LogisticRegression` on embeddings
- **This gets a valid .tsv submission in ~1 hour**
- Estimated: 0.114 s/clip × 15k clips = 28 min to extract features

### Tier 2 — Frozen WavLM + trainable MLP head

- Same frozen WavLM features
- Train a 2-layer MLP: `768 → 256 → 1` (with dropout, ReLU)
- ~200k trainable parameters
- Train for 10-20 epochs on CPU (feasible: forward pass is just the MLP)
- Should outperform LR by capturing nonlinear decision boundaries

### Fusion

- Weighted average of tier scores
- Or train a small stacking model (LR on the 3 tier scores)
- Optimize weights on validation set using minDCF directly

---

## 8. Validation Strategy

### Leave-one-generator-out cross-validation

- 10 generators → 10 folds
- Each fold: train on 9 generators, validate on 1
- This tests generalization to unseen synthesis methods (which is what the test set will contain)
- **Must split OpenVoice by sentence** (not randomly) due to 5 accent duplicates

### Fold assignment

```
fold 0: hold out diffgan_tts
fold 1: hold out elevenlabs
fold 2: hold out grad_tts
fold 3: hold out openvoicev2
fold 4: hold out playht
fold 5: hold out pro_diff
fold 6: hold out unit_speech
fold 7: hold out wavegrad2
fold 8: hold out xtts_v2
fold 9: hold out your_tts
```

Real speech gets assigned to folds randomly (stratified so each fold has ~equal real clips).

### Manifest CSV columns

```
path, label, generator, speaker, sentence_id, accent_style, source_format, orig_sr, group_id, fold
```

---

## 9. Existing Code in Repo

### `hearsay_hgt13/ProcessFiles.py` — REPLACE

Teammate's file. Converts non-MP3 to MP3 before loading (harmful). Two functions:
- `normalize_file(file_in)` — loads audio via librosa after optional MP3 conversion
- `get_metadata(file_in)` — runs ffprobe (which isn't on PATH)

**Action:** Replace entirely with content-based decoder (see Section 5).

### `hearsay_hgt13/TestAASIST.py` — REFERENCE ONLY

Loads AASIST model from `aasist/config/AASIST.conf` and `aasist/models/weights/AASIST.pth` (ASVspoof 2019 weights). Only scores PlayHT fakes from a hardcoded path on the teammate's machine. No real files, no evaluation metric. The AASIST weights are trained on ASVspoof 2019 attacks and perform poorly on modern synthesizers.

### `hearsay_hgt13/.gitignore` — DONE

Excludes `data/`, `test/`, `cache/`, audio files, archives, model checkpoints, Python cache, editor files.

### `hearsay_hgt13/README.md` — Stub

Just contains `# HEARSAY - HackGT 13`.

---

## 10. Scorer Codebase Locations

| File | Path | Notes |
|------|------|-------|
| Baseline-AASIST scorer | `C:\Users\himan\Downloads\HackGTMinDCF\HackGTMinDCF\HackGTMinDCF\asvspoof5\Baseline-AASIST\eval\calculate_metrics.py` | **THE grading scorer** — needs param fix |
| Scorer math functions | `...\Baseline-AASIST\eval\calculate_modules.py` | `compute_eer` returns 4 values, `compute_mindcf` takes `(frr, far, thresholds, Pspoof, Cmiss, Cfa)` |
| evaluation-package scorer | `...\asvspoof5\evaluation-package\calculate_metrics.py` | Different API — DO NOT USE for grading |

### Local scorer wrapper (to be built)

```python
# Pseudocode for local evaluation:
# 1. Load predictions: {filename: score}  (0=real, 1=fake)
# 2. Load ground truth labels
# 3. Flip scores for AASIST scorer: flipped = 1 - score
# 4. Separate into bona_scores (real files) and spoof_scores (fake files)
# 5. Call compute_eer(bona_scores, spoof_scores) → (eer, frr, far, thresholds)
# 6. Call compute_mindcf(frr, far, thresholds, 0.3, 1, 4) → (minDCF, threshold)
```

---

## 11. Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| Use WavLM-base-plus (not large) | 94M vs 316M params — feasible on CPU |
| Freeze WavLM, train only head | Fine-tuning 94M params on CPU would take ~40 hrs |
| Don't use AASIST pretrained weights | Trained on ASVspoof 2019 — fails on modern fakes |
| Don't convert to MP3 before loading | Erases detection artifacts |
| Use `soundfile` not `librosa` for loading | Content-based format detection, handles MP3-in-WAV |
| Move data to `C:\hearsay_data` | Avoids OneDrive sync and git staging |
| Add LibriSpeech + full LJ Speech | Only 242 real clips from 1 speaker is not enough |
| Leave-one-generator-out validation | Tests generalization to unseen synthesis methods |
| Scorer: Baseline-AASIST version | Sponsor confirmed |
| Params: Pspoof=0.3, Cfa=4, Cmiss=1 | Sponsor confirmed |
| Score convention: 0=real, 1=fake | Sponsor confirmed (matches answer key) |

---

## 12. Research Papers Reviewed

### 1. DiffSSD (Purdue, 2024)
- Dataset with 70k synthetic + 24k real clips from 10 generators
- **This is our training data** (DiffSSD IS the `generated_speech/` folder)
- Wav2Vec2 achieved 3% EER, PaSST 3.5% when retrained on it
- Key insight: self-supervised speech features (Wav2Vec2/WavLM) transfer well to deepfake detection

### 2. ASVspoof 5 (2024)
- Large benchmark: 32 attack algorithms, ~2000 speakers, codec/compression
- Handles shortcut artifact problem explicitly
- Architecture: pretrained frontend → trainable backend → scoring
- Our pipeline follows this pattern

### 3. Residual Fingerprint / RFP (training-free)
- Uses spectral residuals to detect synthetic speech
- 99%+ AUROC on clean data
- **Fragile under MP3 encoding, reverb, noise**
- Useful only as cheap supplementary feature (Tier 0), not primary detector

---

## 13. What's Done

- [x] Download and verify LibriSpeech (4 subsets, 11,126 files, 146 speakers)
- [x] Download and verify full LJ Speech (10,961 files)
- [x] Move all data out of OneDrive to `C:\hearsay_data`
- [x] Verify test data (1,671 files in `hearsay_hgt13/test/`)
- [x] Create `.gitignore`
- [x] Identify and document all data format gotchas (PlayHT MP3-in-WAV, ElevenLabs MP3)
- [x] Verify `soundfile` can read all formats correctly
- [x] Benchmark CPU performance for WavLM
- [x] Identify scorer code and confirm parameters with sponsor
- [x] Design full architecture (3-tier system)
- [x] Design validation strategy (leave-one-generator-out)
- [x] Review and document shortcut risks + mitigations

---

## 14. Immediate Next Steps (in order)

### Step A — Build manifest CSV

Create `C:\hearsay_data\manifest.csv` with columns:
```
path, label, generator, speaker, sentence_id, accent_style, source_format, orig_sr, group_id, fold
```

- Walk all data directories
- Parse folder structure for metadata
- Deduplicate OpenVoice (25,000 files → 5,000 sentences × 5 accents — keep all but group by sentence)
- Assign fold numbers (leave-one-generator-out)
- Balance real clips across folds

### Step B — Write content-based audio decoder

Replace `ProcessFiles.py` with `audio_utils.py`:
- `load_audio(path, target_sr=16000)` using `soundfile` + `soxr`
- Returns `(waveform, original_sr)`
- Handles WAV, FLAC, MP3, MP3-in-WAV transparently

### Step C — Write local scorer wrapper

Create `scorer.py`:
- Copy `compute_eer` and `compute_mindcf` from Baseline-AASIST (to avoid import path issues)
- Hardcode `Pspoof=0.3`, `Cmiss=1`, `Cfa=4`
- Accept predictions dict `{filename: score}` and ground truth dict
- Handle score flipping internally
- Return `(minDCF, EER)`

### Step D — Run Tier 1 (frozen WavLM + LR)

1. Extract WavLM embeddings for all training clips (frozen, no grad)
   - ~15k clips × 0.114 s/clip ≈ 28 min
   - Save as `.npy` in `C:\hearsay_cache\`
2. Train `LogisticRegression` on embeddings
3. Extract embeddings for 1,671 test clips (~3 min)
4. Generate first valid `.tsv` submission
5. Evaluate on validation folds

### Step E — Run Tier 2 (frozen WavLM + MLP head)

1. Use cached embeddings from Step D
2. Train 2-layer MLP: `768 → 256 → 1`
3. Evaluate, compare to Tier 1

### Step F — Fusion + final submission

1. Combine tier scores (weighted average or stacking)
2. Optimize fusion weights on validation set using minDCF
3. Generate final `.tsv`

---

## 15. File/Directory Map

```
C:\hearsay_data\                          ← All large data (outside git/OneDrive)
├── generated_speech\                     ← 70k fake clips, 10 generators
│   ├── diffgan_tts\speaker_*/            ← 5,000 wav
│   ├── elevenlabs\speaker_*/             ← 5,000 mp3 (actual mp3)
│   ├── grad_tts\speaker_*/               ← 5,000 wav
│   ├── openvoicev2\speaker_*/            ← 25,000 wav (5 accents × 5,000)
│   ├── playht\speaker_*/                 ← 5,000 wav (MP3-in-WAV!)
│   ├── pro_diff\speaker_*/               ← 5,000 wav
│   ├── unit_speech\speaker_*/            ← 5,000 wav
│   ├── wavegrad2\speaker_*/              ← 5,000 wav
│   ├── xtts_v2\speaker_*/                ← 5,000 wav
│   └── your_tts\speaker_*/               ← 5,000 wav
├── LJRealResampled\                      ← 242 real wav (16 kHz)
├── LJSpeech-1.1\LJSpeech-1.1\wavs\      ← 10,961 real wav (22.05 kHz, needs resample)
├── librispeech\LibriSpeech\              ← 11,126 real FLAC (16 kHz)
│   ├── dev-clean\   (2,703 files, 40 speakers)
│   ├── dev-other\   (2,864 files, 33 speakers)
│   ├── test-clean\  (2,620 files, 40 speakers)
│   └── test-other\  (2,939 files, 33 speakers)
├── raw\                                  ← tar.gz archives (can delete)
└── (future: manifest.csv, cached embeddings)

C:\hearsay_cache\                         ← Cached features, embeddings
└── (empty, ready for use)

C:\Users\himan\OneDrive - ...\Hackgt 2026\
└── hearsay_hgt13\                        ← Git repo (code only)
    ├── .gitignore
    ├── README.md
    ├── ProcessFiles.py                   ← REPLACE with audio_utils.py
    ├── TestAASIST.py                     ← Reference only
    ├── handoff.md                        ← This file
    └── test\HackGTHearsayTesting\        ← 1,671 test files + template

C:\Users\himan\Downloads\HackGTMinDCF\...\asvspoof5\
├── Baseline-AASIST\eval\                 ← THE scorer
│   ├── calculate_metrics.py              ← Needs Pspoof=0.3, Cfa=4
│   └── calculate_modules.py             ← compute_eer (4 returns), compute_mindcf
└── evaluation-package\                   ← DO NOT USE for grading
```

---

## 16. Prompt for New Session

Paste this at the start of your new Claude session:

> Read the file `C:\Users\himan\OneDrive - Georgia Institute of Technology\Desktop\Hackgt 2026\hearsay_hgt13\handoff.md` — it contains the complete context for the HEARSAY challenge at HackGT 2026. I'm picking up from where the previous session left off. All data downloading and setup is complete. Start with Step A (build manifest CSV), then continue through Steps B-F in order. The working directory is `C:\Users\himan\OneDrive - Georgia Institute of Technology\Desktop\Hackgt 2026\hearsay_hgt13` and all data is at `C:\hearsay_data`.

---

## 17. Open Questions

1. **GPU access:** User mentioned "GPU is not there for now we can discuss that later." If GPU becomes available (e.g., Google Colab, teammate's machine), Tier 2 training would be much faster and we could consider unfreezing some WavLM layers.
2. **Test set composition:** We don't know which generators (if any from DiffSSD) are in the test set, or if there are unknown generators. The leave-one-generator-out strategy prepares for this.
3. **Submission limit:** Unknown if there's a limit on how many times you can submit.
4. **OpenVoice deduplication:** 25,000 files = 5,000 sentences × 5 accents. Should we keep all 5 accents (as data augmentation) or sample 1 per sentence? Current plan: keep all but group by `sentence_id` so all accents of the same sentence go to the same fold.
