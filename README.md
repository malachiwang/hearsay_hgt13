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
