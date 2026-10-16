# HEARSAY experiment notes

These notes preserve feature contracts, evaluation decisions, and historical context that are separate from the [active judging path](../README.md#implemented-system). Some experimental modules remain at the repository root. Their presence does not mean they contributed to the submitted score, and no saved benchmark or ablation establishes their effectiveness on the challenge corpus.

## CPPS

[HarmonicAnalysis.compute_cpps](../HarmonicAnalysis.py) returns one whole-clip smoothed cepstral peak prominence scalar. Input must be decoded, finite, mono 16 kHz audio, at least 0.05 seconds long and not exact digital silence. It does not decode, resample, downmix, normalize, pad, or crop the waveform.

The fixed Praat recipe uses a 60 Hz pitch floor, a 0.002-second time step, a 5,000 Hz maximum frequency, and pre-emphasis from 50 Hz. CPPS smoothing uses 0.02 seconds in time and 0.0005 seconds in quefrency; the peak search spans 60 to 330 Hz with parabolic interpolation and a robust straight trend fit. The constants and argument order are covered by [tests](../tests/test_harmonic_analysis.py). The code records validation with `praat-parselmouth==0.4.7` and bundled Praat 6.1.38.

CPPS is an acoustic feature, not a synthetic probability. If evaluating it alone, choose `+CPPS` or `-CPPS` using training data only and freeze the direction before held-out evaluation. minDCF accepts finite real scores without mapping them into `[0, 1]`.

## Physical splice features

[DetectSplices.detect_splices](../DetectSplices.py) accepts a finite mono 16 kHz waveform. It compares wavelet-packet evidence across candidate boundaries against changes on either side, with channels for timbre, level, transients, and frequency. Audibility gates and tonal-glide handling reduce specific false responses in constructed test signals. Near silence returns zero-valued features and no candidates.

`splice_features` returns `splice_max_score`, candidate count, and the four channel maxima. The continuous maximum is the initial standalone score; candidate count is not the primary statistic. The intended higher-is-more-suspicious direction still needs to be checked and frozen on training data before evaluating a real corpus.

The [tests](../tests/test_detect_splices.py) cover level and spectral steps, clicks, smooth amplitude changes, tonal glides, silence, and boundary timing. These fixture checks do not show that the detector can distinguish natural recording boundaries from manipulation. A fully synthetic clip may have no splice at all.

## Representation novelty

[NoveltySplice.detect_embedding_novelty](../NoveltySplice.py) accepts precomputed embeddings of shape `(N, D)` and frame times of shape `(N,)`, plus optional local synthetic scores of shape `(N,)`. It L2-normalizes embeddings, constructs a cosine self-similarity matrix, and applies Gaussian checkerboard kernels with two and four windows per side. A high response indicates internally consistent regions that differ across a boundary.

The module declares an upstream baseline of 0.8-second embedding windows and 0.2-second hops, but it does not extract embeddings or run WavLM. There is no connected audio-to-embedding path for this experiment in the current checkout. The active Eliya wrapper returns scores rather than the embeddings this module needs.

`novelty_splice_features` exposes the maximum, top-three mean, p95, and peak count. Optional local scores add maximum, minimum, range, maximum adjacent jump, and p90. Absolute boundary times are retained for inspection but excluded from classifier features. Peak gates are fixed diagnostic defaults, not competition-tuned thresholds.

The intended use was to complement waveform seams when an edit is cross-faded. [Constructed-embedding tests](../tests/test_novelty_splice.py) cover abrupt changes, gradual drift, crossfade-like transitions, and phonetic-like variation. They do not establish behavior on real WavLM embeddings. Speaker, phonetic, and scene changes can also produce novelty; low novelty does not establish authenticity. The full similarity matrix requires quadratic memory in the number of windows.

[evaluation.analyze_splice_features](../evaluation/analyze_splice_features.py) computes per-file novelty features and label-grouped descriptive distributions, optionally alongside physical splice scores. It neither trains a classifier nor chooses thresholds.

## Fusion and evaluation protocol

The [fusion package](../fusion/) consumes numeric detector tables, not audio. `build_feature_table` takes one authoritative manifest with `file_id`, `label`, and `fold`, then joins each requested detector table by `file_id`. It rejects missing or extra rows, duplicate IDs or feature names, detector-owned labels/folds, nonnumeric values, and nonfinite values. It does not fill, impute, or globally normalize features. Manifest order determines row order; source and column order determine feature order. Labels become `0 = bona fide` and `1 = spoof`.

Two fixed baselines are implemented in [models.py](../fusion/models.py):

- L2 logistic regression with `StandardScaler` fitted inside each training fold.
- A small regularized LightGBM classifier with fixed parameters and deterministic single-threaded execution.

[cross_validate_fusion](../fusion/cross_validate.py) honors the manifest's folds, fits a new model per fold, and emits one held-out class-1 probability per file. It concatenates those predictions in manifest order and evaluates them with the [canonical minDCF evaluator](../evaluation/min_dcf.py). Its result is explicitly an `OOF stacking diagnostic`. Feature subsets can be selected; LightGBM gain importance is diagnostic rather than causal evidence or proof of a minDCF improvement. No hyperparameter-search results or final fitted fusion model are saved.

Preserve these evaluation constraints when using the infrastructure:

1. Learned base-detector scores must be held out: no example should be scored by a model or fingerprint trained on that example. Fixed CPPS and splice features can be computed once per file because they do not learn from labels.
2. Ordinary out-of-fold base scores are not strict nested stacking. For an outer validation fold, exclude that fold from every learned base-model fit and generate fusion-training meta-features entirely within the outer training set.
3. Supply outer-fold-specific feature tables for a strict nested evaluation. The current API allows separate tables but does not orchestrate base-model training or enforce their provenance.
4. Freeze score direction, feature selection, and model choices using training/development data before final evaluation. Do not treat importance rankings or fixture metrics as evidence of generalization.

The active runner does not fit or load a fusion model. A final training-and-export workflow remains unimplemented; the earlier proposal to combine every detector should not be read as completed functionality.

## Historical work

### Standalone AASIST

[aasist/TestAASIST.py](aasist/TestAASIST.py) loads an external AASIST architecture, configuration, and checkpoint. It takes the first 64,600 samples of a long recording or repeat-pads a short one, then prints aggregate bona fide scores. It is not sliding-window inference.

The script refers to `models.AASIST`, `aasist/config/AASIST.conf`, `aasist/models/weights/AASIST.pth`, and a developer-specific Windows audio path. Those assets and audio are absent. It also imports librosa, which is not in the current requirements. No saved result supports a performance comparison with the active Eliya wrapper or a claim that this experiment failed on particular attack types.

### ENF and early file conversion

[ENF.py](../ENF.py) explores electrical-network-frequency tracking through bandpass filtering, STFT peaks, discontinuity checks, and variance heuristics. Its thresholds are exploratory. It has top-level interactive input and a hard-coded `SmallTest/` path, so importing it is not a clean library operation. Matplotlib and librosa, used by this path, are not in the current requirements. There are no ENF tests or saved corpus-level results.

[ProcessFiles.py](../ProcessFiles.py) is an earlier FFmpeg/librosa conversion helper. It may write a WAV beside the source and resamples/downmixes through librosa. The judging runner does not use it. A separate lossless-ingestion implementation appears in historical commit `b04a5ea`, but that implementation and its tests are absent from the current branch's files.

### Residual fingerprints and attribution

Historical commit `2ad3e6d`, authored by `himankgalundia` and available in the repository's other Git history, records an initial audio residual-fingerprint detector. It is not an ancestor of the current branch, and its detector and handoff files are absent from this checkout. Residual fingerprints are therefore historical team work, not an implemented input to the current runner or fusion results. Jonathan Chang's and Malachi Wang's contributions to the current branch are summarized in the [main README](../README.md#team-and-attribution).
