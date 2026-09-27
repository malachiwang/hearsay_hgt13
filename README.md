# HEARSAY - HackGT 13

## Current runnable path

The current end-to-end judging path is the Eliya/WavLM detector in
`NewAttempt/Deepfake.py`. `run_hearsay.py` applies that detector to an input
directory and writes synthetic-oriented scores for judging. Grok is an optional
post-analysis explanation layer; it does not participate in detection or
judging.

### Quick start from a fresh clone

The model assets are downloaded from Hugging Face and are not stored in Git.
A working FFmpeg installation is also needed locally for formats that require
it; the Docker image installs FFmpeg and `libsndfile1` itself.

```bash
git clone https://github.com/malachiwang/hearsay_hgt13.git
cd hearsay_hgt13
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_eliya.py
python NewAttempt/Deepfake.py /path/to/audio.wav
```

The download step requires network access. After it finishes, use
`python scripts/download_eliya.py --verify-only` to verify the local Eliya and
WavLM assets without downloading anything.

## Final submission using GT PACE

The final competition submission was generated on the Georgia Tech PACE ICE
cluster using an NVIDIA H100 GPU. The scoring path is exactly the judging
runner described above — `run_hearsay.py` driving the Eliya detector in
`NewAttempt/Deepfake.py` — so the submitted `predictions.tsv` and the Docker
judging image use the same code and produce the same `eliya_top3_mean` scores.

**File used for scoring:** `run_hearsay.py` (entry point) →
`NewAttempt/Deepfake.py` (`EliyaDetector`, whole-clip 5.0 s windows at a 0.5 s
hop, `eliya_top3_mean` as the `cm-score`). No feature-engineering, fusion, or
splice code participates in the final score.

Steps:

1. **On the PACE login node** (which has network access), set up the
   environment and pre-download the model assets so the GPU job can run
   offline:

   ```bash
   module load anaconda3 cuda
   cd ~/scratch/hearsay_hgt13
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python scripts/download_eliya.py          # downloads Eliya + WavLM-large
   python scripts/download_eliya.py --verify-only
   ```

2. **On a GPU compute node** (interactive session or batch job — never the
   login node, since `torchcodec` needs CUDA), run the judging runner against
   the test directory:

   ```bash
   source .venv/bin/activate
   python run_hearsay.py \
       ~/scratch/hearsay_test/HackGTHearsayTesting \
       predictions.tsv
   ```

   Example interactive allocation:
   `salloc --gres=gpu:H100:1 --mem-per-gpu=224G -t 1:00:00`

3. `predictions.tsv` is the final submission — a header row (`filename` /
   `cm-score`) followed by one row per test file, scores in `[0, 1]` where
   higher means more synthetic. This is the file submitted to the competition
   and the file the Docker image reproduces.

The detector uses `eliya/forensics_0.3B_base_deepfake_classifier` with a
`microsoft/wavlm-large` backbone; inference is deterministic (fixed weights,
`eval()` mode, fixed windowing), so re-running the same test set reproduces the
same scores.
**And this pipeline is in the branch called Himank**

## HEARSAY evaluation

The canonical evaluator is `evaluation.min_dcf.evaluate_min_dcf`. It uses the
NSA cost model for ASVspoof5 Track 1:

- `Pspoof = 0.30`
- `Cmiss = 1.0`
- `Cfa = 4.0`

HEARSAY detector scores are synthetic-oriented: larger means more likely
synthetic/spoof. ASVspoof's countermeasure DET calculation is bona-fide-oriented,
so the evaluator explicitly negates scores internally and converts the selected
threshold back to the supplied synthetic-score orientation.

Lower minDCF is better. The metric sweeps all score thresholds, so ranking and
class separation matter more than whether a score is calibrated around `0.5`.
In the countermeasure cost model, `Cfa` applies to spoof samples falsely accepted
as bona fide.

Evaluate header-based tab-separated files by ID:

```bash
python -m evaluation.evaluate_scores \
  --scores predictions.tsv \
  --labels answer_key.tsv \
  --id-column id \
  --score-column score \
  --label-column label
```

Use `--delimiter comma` for CSV input. Labels must be exactly `bonafide` or
`spoof`; predictions and labels are joined by file ID rather than row position.

## CPPS feature

`HarmonicAnalysis.compute_cpps` extracts one whole-clip smoothed cepstral peak
prominence (CPPS) scalar. Its input must already be a decoded, finite,
non-silent, mono 16 kHz waveform. The extractor does not decode, resample,
downmix, normalize, pad, or crop audio.

CPPS is an acoustic feature, not a synthetic probability. Choose whether
`+CPPS` or `-CPPS` is the synthetic-oriented score using training data only,
freeze that direction, and evaluate held-out scores with
`evaluation.min_dcf.evaluate_min_dcf`. CPPS values do not need to be mapped to
`[0, 1]` for minDCF evaluation.

The tested dependency is pinned as `praat-parselmouth==0.4.7`, which bundled
Praat 6.1.38 in the validation environment.

## Splice analysis

HEARSAY has two complementary splice systems. Neither is a complete standalone
deepfake detector.

### Physical boundary detector

`DetectSplices.detect_splices` accepts an already-prepared, finite, mono 16 kHz
waveform. It does not decode, resample, downmix, normalize loudness, or convert
audio files. The detector uses wavelet-packet discontinuity features for
timbre, level, transients, and frequency, and returns candidate times plus
continuous clip-level features.

`splice_features` exposes fusion-ready values, including `splice_max_score`,
candidate count, and the four channel maxima. `splice_max_score` is the initial
standalone score because it retains continuous evidence; candidate count is not
the primary score. Higher scores are expected to mean more suspicious, but that
orientation must be confirmed and frozen using training data before evaluating
with `evaluation.min_dcf.evaluate_min_dcf`.

Splice analysis is supporting evidence for partial edits such as word
replacement, cut-and-paste joins, or abrupt scene/background changes. A fully
synthetic clip may contain no splice, so a low splice score is not proof that a
clip is bona fide.

### Representation-novelty detector

`NoveltySplice.detect_embedding_novelty` accepts precomputed short-window
embeddings and their frame times. It L2-normalizes each embedding, constructs a
cosine self-similarity matrix, and applies fixed short- and medium-context
Gaussian checkerboard kernels. High novelty means the windows on each side of
a boundary are internally consistent but the two sides differ. This is intended
to complement the physical detector when a partial synthetic substitution is
cross-faded or otherwise lacks a sharp waveform seam.

The expected upstream WavLM windowing baseline is configurable. The novelty
module documents initial constants of 0.8-second windows and 0.2-second hops,
within the intended 0.8--1.0 and 0.20--0.25 second ranges. Johnny's WavLM
integration should provide:

```text
get_window_embeddings(waveform, sample_rate, window_sec, hop_sec)
    -> embeddings      # shape (N, D)
    -> frame_times     # shape (N,)
    -> local_fake_scores (optional, shape (N,))
```

The novelty module does not import, download, train, or run WavLM. Prefer an
anti-spoof/deepfake-sensitive intermediate representation when one is exposed.
Generic WavLM embeddings also encode speaker, phonetic, linguistic, and
acoustic content, so ordinary speech changes can produce novelty and must be
validated on bona fide speech.

`novelty_splice_features` returns `splice_novelty_max`, top-three mean, p95,
and peak count for the generic fusion table. Optional local fake scores add
`local_wavlm_fake_max`, minimum, range, maximum adjacent jump, and p90. No
absolute boundary time is used as a classifier feature. The detailed result
retains multiscale curves and candidate times for diagnosis. The fixed peak
gate is only a conservative debugging aid; continuous novelty statistics are
the primary outputs, and no competition threshold was tuned in this ticket.

Low novelty is not bona fide proof: a fully synthetic clip may be internally
consistent. High novelty is also not proof of manipulation because speaker,
recording, or acoustic scene changes can create real boundaries. These are
supporting partial-manipulation signals whose value must be measured through
the existing fusion and NSA minDCF pipeline. The descriptive helper in
`evaluation.analyze_splice_features` compares label-grouped novelty and
optional physical-splice distributions without training or choosing thresholds.

## Fusion

The `fusion` package combines already-computed numeric detector outputs; it
never loads or processes audio. One authoritative manifest owns `file_id`,
`label`, and `fold`. Requested detector tables are joined strictly by
`file_id`: missing rows, extra rows, duplicate IDs or columns, nonnumeric
features, and nonfinite values are errors. Feature values are not silently
filled, dropped, imputed, or globally normalized.

`build_feature_table` accepts whichever detector sources are ready, so CPPS and
splice features can be evaluated before RFP or WavLM scores exist. Labels are
canonicalized to `0 = bona fide/real` and `1 = spoof/synthetic`. Source and
column order determine a stable feature-column order.

Two fixed baselines are available:

- L2-regularized logistic regression with `StandardScaler` inside each
  fold-local training pipeline.
- A small, strongly regularized LightGBM classifier with fixed parameters and
  deterministic single-threaded execution.

Both produce a held-out class-1 probability, so higher scores mean more
synthetic. `cross_validate_fusion` honors the manifest's existing folds,
concatenates one held-out prediction per file in original order, and passes the
complete score vector to `evaluation.min_dcf.evaluate_min_dcf`. Its result is
explicitly labeled an **OOF stacking diagnostic**. Feature subsets can be
selected explicitly, and LightGBM gain importance is exposed for debugging;
importance is not causal evidence and does not prove a feature improves
held-out minDCF. These baseline hyperparameters have not been optimized.

Learned base-detector scores, including WavLM classifiers and learned RFP
fingerprints, must themselves be held-out/OOF: a detector must not score an
example using a model or fingerprint trained on that example. Fixed CPPS and
splice heuristics do not learn from labels and may be computed once per file.
Ordinary OOF base scores are useful for development, but they are not the same
as strict nested stacking. In a strict outer-fold evaluation, the outer
validation fold must be excluded from every learned base-detector fit, and the
fusion training meta-features must be generated entirely within the outer
training set. The feature-table and fusion APIs are deliberately separate so
outer-fold-specific tables can be supplied later without changing either
contract.

The eventual final-test flow is: train selected learned detectors on the
appropriate full training data, score the NSA test files, compute fixed
CPPS/splice features, build the same strict test feature table, and apply the
final trained fusion model. The final fitting procedure remains a team decision
until validation methodology is settled.

## Eliya detector

`NewAttempt/Deepfake.py` wraps
`eliya/forensics_0.3B_base_deepfake_classifier` in process. The upstream model
uses `microsoft/wavlm-large` as its WavLM backbone. HEARSAY loads the upstream
`model.py` architecture and `checkpoint_epoch_5.safetensors` once per detector
process; the legacy pickled `checkpoint_epoch_5.pt` is neither requested nor
used.

Each file is decoded once. Multichannel audio is averaged to mono, audio not
already at 16 kHz is resampled to 16 kHz, and the upstream amplitude preparation
is applied. The complete clip is then covered by 5.0-second windows with a
0.5-second hop. A unique end-anchored window is added when regular stepping does
not reach the file end. A clip shorter than five seconds produces one window by
repeating its samples to the required length rather than zero-padding.

Inference is batched (default batch size: 4), and all batches share the same
already-loaded model. The upstream logit is bona-fide-oriented, so HEARSAY uses
`1 - sigmoid(logit)` for each synthetic-oriented window score: `0` means more
bona fide and `1` means more synthetic. The current primary file score is
`eliya_top3_mean`. The mean, maximum, p90, high-window fraction, and individual
window records are also retained for later analysis or fusion. The configurable
threshold affects only the diagnostic high-window fraction and legacy verdict;
it does not replace the continuous score.

Run a file or directory with:

```bash
python NewAttempt/Deepfake.py audio.wav
python NewAttempt/Deepfake.py audio_directory/ --csv eliya_scores.csv
```

Directory discovery is recursive and supports `.wav`, `.mp3`, `.flac`, `.m4a`,
`.ogg`, and `.opus`, case-insensitively.

## Model asset setup

The recommended explicit setup command is:

```bash
python scripts/download_eliya.py
```

It downloads only the required Eliya repository files into
`NewAttempt/models/forensics_0.3B_base_deepfake_classifier`:

- `checkpoint_epoch_5.safetensors`
- `inference.py`
- `model.py`
- `config.json`
- `requirements.txt`

It also downloads the required `microsoft/wavlm-large` configuration and model
weights into the Hugging Face cache. This reconstructs the model dependencies
for a fresh clone without committing large weights to Git. Direct detector
startup can fetch missing Eliya repository files automatically, but the setup
script is preferred because it also prepares and validates the WavLM cache.

To check existing assets without allowing a download:

```bash
python scripts/download_eliya.py --verify-only
```

The verification command fails if the allowlisted Eliya files or cached WavLM
files are incomplete, if an unexpected file is present in the Eliya model
directory, or if the legacy `.pt` checkpoint is present.

## Testing with audio

The detector needs a real supported audio file or a directory containing such
files; this repository does not bundle a labeled audio test dataset. The Eliya
directory CLI supports `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, and `.opus`.
The judging runner also accepts `.mp4` audio containers. Actual decode support
depends on the installed TorchAudio/TorchCodec and FFmpeg stack.

Optionally, an external labeled corpus such as ASVspoof or DeepVoice can be
used for sanity testing, but HEARSAY does not download or depend on either
dataset.

## Judging runner

Run the non-Docker judging entry point with an input directory and output TSV:

```bash
python run_hearsay.py /path/to/input_audio /path/to/predictions.tsv
```

`run_hearsay.py` recursively discovers `.wav`, `.mp3`, `.m4a`, `.mp4`, `.ogg`,
`.opus`, and `.flac` files in deterministic order. It loads one Eliya detector,
scores every file, and writes exactly these tab-separated columns:

```text
filename	cm-score
```

The current `cm-score` is `eliya_top3_mean`, is finite and bounded to `[0, 1]`,
and is oriented so a higher value means more synthetic. The runner writes one
row per successfully processed input file. If any file fails, it reports an
error, exits nonzero, and does not leave a partial output file.

## Ask Grok about an analysis

HEARSAY can optionally ask xAI's Grok to explain a completed detector result in
plain language through `POST https://api.x.ai/v1/responses`. Grok receives a
structured, bounded summary containing the measured prediction and score,
Eliya aggregate and strongest-window outputs, and any other detector outputs,
suspicious characteristics, metadata, or disagreement actually supplied by
HEARSAY. It does not receive raw audio, run a detector, change
`eliya_top3_mean`, or participate in the judging TSV. The CLI labels its
response as a natural-language interpretation rather than a detector result.

Set the xAI key in the environment (never commit it), then use the existing
Eliya CLI with `--ask-grok`:

```bash
export XAI_API_KEY="your-xai-api-key"
python NewAttempt/Deepfake.py audio.wav --ask-grok
python NewAttempt/Deepfake.py audio.wav \
  --ask-grok "Which measured interval should I review first?"
```

The API key is read only from `XAI_API_KEY`. The default model is `grok-4.7`;
set `XAI_MODEL` or pass `--grok-model MODEL_NAME` to override it. If no API key
is configured, or if xAI rejects or cannot complete the request, HEARSAY still
prints and preserves the detector result and reports that the optional Grok
interpretation is unavailable. Detector scores are evidence, not guaranteed
calibrated probabilities, and the Grok prompt explicitly prohibits inventing
missing signals or confidence estimates.

## Docker judging image

The judging image uses the same `run_hearsay.py INPUT_DIRECTORY OUTPUT_TSV`
interface described above. It recursively scores the runner's seven supported
extensions and writes the exact `filename` / `cm-score` TSV. The current score
is `eliya_top3_mean`, where larger values mean more likely synthetic. Grok is
not invoked by this judging path.

Build the CPU-capable image from the repository root:

```bash
docker build -t hearsay .
```

During `docker build`, the image runs `scripts/download_eliya.py` to download
the five allowlisted Eliya files into the exact directory expected by
`NewAttempt/Deepfake.py`, using `checkpoint_epoch_5.safetensors` rather than the
legacy `.pt` checkpoint. It also caches the `microsoft/wavlm-large` backbone and
runs `--verify-only` before completing the build. Model assets are therefore in
the final image; they are not copied from the developer machine or tracked by
Git.

Run it with absolute host paths:

```bash
docker run --rm \
  -v /absolute/path/to/test:/input:ro \
  -v /absolute/path/to/output:/output \
  hearsay \
  /input \
  /output/predictions.tsv
```

Inference is configured for offline Hugging Face operation. The same command
works without model downloads at container startup. Verify this by adding
`--network none` immediately after `docker run --rm`:

```bash
docker run --rm --network none \
  -v /absolute/path/to/test:/input:ro \
  -v /absolute/path/to/output:/output \
  hearsay \
  /input \
  /output/predictions_offline.tsv
```

For an explicitly x86-64 judging target, build a separate image with:

```bash
docker build --platform linux/amd64 -t hearsay-amd64 .
```

Use that option only when the judging platform requires `linux/amd64`; native
builds avoid emulation overhead during local development.

## Run tests

The repository uses pytest. It is a test-only tool and is not currently listed
in `requirements.txt`, so install it in the active environment if necessary:

```bash
python -m pip install pytest
python -m pytest -q
```

To run only the tests for the recent detector, Grok, and judging-runner work:

```bash
python -m pytest -q \
  tests/test_deepfake.py \
  tests/test_grok_interpretation.py \
  tests/test_run_hearsay.py
```

These unit tests use mocked neural-model and xAI responses. They do not require
the large model download, a labeled audio dataset, or an `XAI_API_KEY`.
