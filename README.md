# HEARSAY - HackGT 13

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
`eliya/forensics_0.3B_base_deepfake_classifier` in process. It loads the
upstream `model.py` architecture and `checkpoint_epoch_5.safetensors` once;
the pickled `checkpoint_epoch_5.pt` is never requested. Each audio file is
decoded once, prepared using the upstream mono/16 kHz/amplitude policy, and
covered by 5.0-second windows at a 0.5-second hop. A unique end-anchored window
is added when regular stepping does not reach the file end. Clips shorter than
five seconds produce one upstream-style repeat-padded model input.

Window batches share the already-loaded model. The upstream sigmoid output is
the bona-fide probability, so each HEARSAY window score is
`1 - sigmoid(logit)`, with larger values meaning more synthetic. The current
main Eliya score is `eliya_top3_mean`; mean, maximum, p90, high-window fraction,
and all window records are also retained for later fusion. The configurable
threshold affects only the diagnostic high-window fraction and backward-
compatible diagnostic verdict—it does not threshold or replace the continuous
HEARSAY score.

Run a file or directory with:

```bash
python NewAttempt/Deepfake.py audio.wav
python NewAttempt/Deepfake.py audio_directory/ --csv eliya_scores.csv
```

## Docker judging image

The judging image uses `run_hearsay.py` to score every supported audio file in
an input directory and write an exact two-column `filename` / `cm-score` TSV.
The current score is `eliya_top3_mean`, where larger values mean more likely
synthetic.

Build the CPU-capable image from the repository root:

```bash
docker build -t hearsay .
```

The build downloads the five allowlisted Eliya files into the exact directory
expected by `NewAttempt/Deepfake.py`, using
`checkpoint_epoch_5.safetensors` rather than the legacy `.pt` checkpoint. It
also caches the upstream `microsoft/wavlm-large` backbone required by Eliya's
`model.py`. Model weights are therefore part of the final image and are not
copied from the developer machine or tracked by Git.

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
can be checked without network access by adding `--network none` immediately
after `docker run --rm`:

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
