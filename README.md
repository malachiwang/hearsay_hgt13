# HEARSAY

HEARSAY scores audio for synthetic manipulation using a pretrained speech detector with sliding-window inference.

We built HEARSAY for the NSA Audio Authentication Challenge at HackGT 13. The task was to distinguish bona fide recordings from synthetic or manipulated audio and produce one score per file. Our work in this checkout includes model integration, full-clip windowing, a batch submission runner, an NSA-style evaluator, signal-processing experiments, fusion infrastructure, and an optional explanation layer.

## Results and provenance

Official judging results are not archived in this checkout. It contains no organizer report, held-out answer key, or leaderboard supporting a placement or official minDCF, so we do not assert those results here.

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| [forensics_submission.tsv](forensics_submission.tsv) | 1,671 rows with unique filenames and finite scores in `[0, 1]`; two tab-separated columns, without a header | Accuracy, minDCF, placement, submission validity, or the exact model/configuration that produced it |
| [Evaluator tests](tests/test_min_dcf.py) | Agreement with a local transcription of ASVspoof5 Track 1 scoring, including score direction and ties | Independent verification against the organizer's evaluation service |
| [Feature and fusion tests](tests/) | Behavior on constructed signals, embeddings, and feature tables | Detector quality on a labeled speech corpus |

There are no saved local validation reports or historical benchmark results in the current tree. Unit-test metrics are fixture checks, not competition results.

## Implemented system

The active inference path is [NewAttempt/Deepfake.py](NewAttempt/Deepfake.py), wrapped by [run_hearsay.py](run_hearsay.py) for judging. It loads Eliya's pretrained `forensics_0.3B_base_deepfake_classifier`; we did not train that base model in this repository.

```text
Audio file
  -> TorchAudio decode
  -> channel average, 16 kHz resampling, peak-amplitude normalization
  -> 5-second windows at 0.5-second hops
  -> once-loaded Eliya model, batched inference
  -> synthetic window score = 1 - sigmoid(logit)
  -> mean of the three highest window scores (or all if fewer than three)
  -> eliya_top3_mean -> cm-score in the judging TSV
                     -> optional Grok interpretation in the analysis CLI
```

These parameters and transformations are defined in [Deepfake.py](NewAttempt/Deepfake.py). Each file is decoded once. A final end-anchored window covers any remainder; short clips are repeated to fill one window. The default batch size is four. The detector selects CUDA when available and otherwise uses CPU.

Only `eliya_top3_mean` affects the final `fake_probability` and judging score. Larger values mean more synthetic. Mean, maximum, p90, the fraction of high-scoring windows, and window timestamps are diagnostic outputs. `--threshold` defaults to `0.15` and changes the diagnostic verdict and high-window fraction only. Scores are not demonstrated to be calibrated probabilities.

### Detection methods and their status

| Component | What it measures or contributes | Used in the judging score? |
| --- | --- | --- |
| Eliya detector | Pretrained neural scores over the full clip. The local wrapper describes the external architecture as WavLM + AASIST and the setup script fetches `microsoft/wavlm-large`. Architecture and weights are downloaded, not vendored. | Yes, through `eliya_top3_mean` |
| [CPPS](HarmonicAnalysis.py) | Whole-clip smoothed cepstral peak prominence, an acoustic periodicity feature | No; standalone feature extractor |
| [Physical splice detector](DetectSplices.py) | Wavelet-based changes in timbre, level, transients, and frequency; returns candidate times and continuous features | No; standalone feature extractor |
| [Embedding novelty](NoveltySplice.py) | Local changes in supplied embeddings using cosine similarity and checkerboard kernels | No; no WavLM embedding extraction is connected to it |
| [Learned fusion](fusion/) | Logistic regression with fold-local scaling or a fixed LightGBM baseline over numeric feature tables | No; cross-validation infrastructure, with no saved final fusion model |
| [ENF](ENF.py) | Experimental mains-frequency tracking and discontinuity heuristics | No; exploratory script with local-path assumptions |

The separate [legacy AASIST script](legacy/aasist/TestAASIST.py) is an earlier cropped-clip experiment, not the active windowed wrapper. Feature contracts, evaluation precautions, and historical context are in [the experiment notes](legacy/README.md).

### Optional Grok interpretation

[grok_interpretation.py](grok_interpretation.py) sends a bounded summary of already-computed scores and strongest windows to xAI. It sends no raw audio, runs no detector, and never changes classification or the TSV. Other detector evidence is included only if a caller supplies it; the normal CLI supplies Eliya evidence.

With `XAI_API_KEY` set in your environment, run from the repository root:

```bash
python -m NewAttempt.Deepfake /path/to/audio.wav \
  --ask-grok "Which measured interval should I review first?"
```

Use `--grok-model` or `XAI_MODEL` to override the code's default, `grok-4.7`. Availability depends on the external API. Missing keys and API failures are reported without discarding detector results. Module invocation keeps the repository-root Grok import available.

## Setup

Run commands from the repository root. The [Dockerfile](Dockerfile) uses Python 3.11; local Python/package compatibility still depends on the platform. Install FFmpeg and the system libsndfile library for audio decoding. The exact supported formats depend on the TorchAudio, TorchCodec, and FFmpeg combination.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_eliya.py
python scripts/download_eliya.py --verify-only
```

The download requires network access. [scripts/download_eliya.py](scripts/download_eliya.py) places `model.py`, `inference.py`, `config.json`, `requirements.txt`, and `checkpoint_epoch_5.safetensors` in `NewAttempt/models/forensics_0.3B_base_deepfake_classifier/`. It also caches the WavLM `config.json` and `pytorch_model.bin` through Hugging Face. The legacy Eliya `.pt` checkpoint is not used. `--verify-only` checks file availability locally; it does not run inference.

Model assets, audio datasets, labels, fold manifests, and trained fusion models are not stored in Git. Direct detector startup may download missing model files. Several dependencies and the model revisions are unpinned, so a fresh install is not an exact reconstruction of the submission environment.

## Small examples

The smallest example requires NumPy but no model, API key, or audio dataset:

```bash
python - <<'PY'
from evaluation.min_dcf import evaluate_min_dcf

result = evaluate_min_dcf(
    [0.10, 0.20, 0.80, 0.90],
    ["bonafide", "bonafide", "spoof", "spoof"],
)
print(f"Toy minDCF: {result.min_dcf:.6f}")
PY
```

It prints `Toy minDCF: 0.000000` because these invented scores perfectly separate the two classes. This demonstrates the evaluator, not model performance.

After installing the model assets, score your own audio:

```bash
python -m NewAttempt.Deepfake /path/to/audio.wav
python -m NewAttempt.Deepfake /path/to/audio_directory --csv eliya_scores.csv
```

The analysis CLI prints aggregate scores and window counts. Its recursive directory mode accepts `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, and `.opus`, case-insensitively. It logs individual file errors and continues; use the judging runner when the batch must succeed completely.

## Challenge batch runner and TSV

```bash
python run_hearsay.py /path/to/input_audio /path/to/predictions.tsv
```

The runner recursively discovers the analysis CLI's formats plus `.mp4`, orders files deterministically, and loads one detector for the batch. It writes the header `filename<TAB>cm-score` followed by one row per input file. `filename` is the basename, including its extension; `cm-score` is finite, in `[0, 1]`, and higher for synthetic audio.

On any scoring failure it exits nonzero before replacing the output. A successful run replaces the destination atomically. Use unique basenames across subdirectories: the runner does not reject collisions. The saved `forensics_submission.tsv` lacks the current runner's header and should not be passed directly to the header-based evaluator.

### Docker

The image installs audio system libraries, downloads model assets at build time, and checks their presence in offline mode. Building needs network access and substantial model storage. The configured entry point is the same batch runner.

```bash
docker build -t hearsay .
docker run --rm --network none \
  -v /absolute/path/to/input_audio:/input:ro \
  -v /absolute/path/to/output_directory:/output \
  hearsay /input /output/predictions.tsv
```

Create the host directories first. For an x86-64 target, build with `docker build --platform linux/amd64 -t hearsay-amd64 .` and use that image name. Docker execution has not been verified in the documentation review environment.

## Evaluation

[evaluation/min_dcf.py](evaluation/min_dcf.py) implements minimum normalized detection cost with the repository's NSA/ASVspoof5 convention: `Pspoof = 0.30`, `Cmiss = 1.0`, and `Cfa = 4.0`. It negates HEARSAY's synthetic-oriented scores for the bona-fide-oriented DET calculation, preserves stable tie handling, and returns the threshold in the original direction.

The cost at each threshold is `(0.70 * Pmiss + 1.20 * Pfa) / 0.70`; minDCF is its minimum. Here `Pmiss` is bona fide classified as spoof, and `Pfa` is spoof accepted as bona fide. Lower is better. This is a threshold-swept ranking metric, not accuracy or a calibration measure.

For predictions from the current runner and a separately supplied answer key:

```bash
python -m evaluation.evaluate_scores \
  --scores /path/to/predictions.tsv \
  --labels /path/to/answer_key.tsv \
  --id-column filename \
  --score-column cm-score \
  --label-column label
```

Both tables need headers. The answer key must have `filename` and `label` columns, with labels exactly `bonafide` or `spoof`. Rows are joined by ID, not position. Duplicate IDs, missing predictions, nonfinite scores, and unexpected prediction IDs are errors by default. Use `--delimiter comma` when both inputs are CSV. No answer key is bundled for the saved submission.

## Tests

With the required Python dependencies installed, run the existing unittest-based suite without an additional test framework:

```bash
python -m unittest discover -s tests -v
```

Pytest is also supported, but is not listed in `requirements.txt`:

```bash
python -m pip install pytest
python -m pytest -q
```

For focused checks that need only NumPy in addition to the standard library:

```bash
python -m unittest discover -s tests -p 'test_min_dcf.py' -v
python -m unittest discover -s tests -p 'test_run_hearsay.py' -v
python -m unittest discover -s tests -p 'test_grok_interpretation.py' -v
```

Tests use generated fixtures and mocked neural models/API responses. They do not require model downloads, private audio, or xAI credentials. Full-suite collection still requires packages such as PyTorch, Parselmouth, PyWavelets, and LightGBM. Tests cover window boundaries and score orientation, splice behavior, CPPS, strict feature joins, fold-local fusion, evaluator parity, Grok isolation, and TSV output.

During this documentation review, 56 tests passed across the evaluator, runner, Grok, feature-table, and embedding-novelty modules. The toy example and detector/runner/evaluator `--help` commands also passed with Python 3.12.4 and the already-installed packages. The full suite, real inference, model download/verification, and Docker were not verified: required packages, model assets, audio, FFmpeg, and Docker were unavailable. The downloader's `--help` failed on a missing `huggingface_hub` import. Live Grok requests were not tested because `XAI_API_KEY` was unset. No dependencies or models were installed for this review.

## Repository structure

| Path | Purpose |
| --- | --- |
| `NewAttempt/Deepfake.py` | Active model loader, audio preparation, windowing, and analysis CLI |
| `run_hearsay.py` | Judging batch entry point |
| `grok_interpretation.py` | Optional explanation of measured evidence |
| `evaluation/` | minDCF, score-table CLI, descriptive splice analysis |
| `fusion/` | Strict feature tables and logistic/LightGBM OOF diagnostics |
| `HarmonicAnalysis.py`, `DetectSplices.py`, `NoveltySplice.py` | Independent experimental feature extractors |
| `ENF.py`, `ProcessFiles.py` | Earlier exploratory scripts, outside the judging path |
| `scripts/download_eliya.py`, `Dockerfile` | Model setup and container packaging |
| `tests/` | Automated fixture-based checks |
| `legacy/` | Experiment notes and the older AASIST script |
| `forensics_submission.tsv` | Preserved prediction artifact without labels or a run manifest |

## Limitations

- The active path resamples, downmixes, and normalizes audio. It is not a lossless forensic ingestion pipeline.
- Long recordings create many overlapping model windows. The waveform is held in memory, and no latency or memory benchmark is saved. CPU inference may be expensive.
- High window scores identify intervals to review, not precise edit boundaries. Short clips are repeated, which changes their model input.
- CPPS, splice, and ENF evidence is not proof of synthesis. Natural recording changes can trigger boundaries; fully synthetic clips can be internally consistent. No saved ablation shows that adding these features improves detection.
- Fusion is not integrated into final inference. Ordinary out-of-fold stacking is a development diagnostic; it does not establish a leakage-free nested evaluation. See the [evaluation protocol](legacy/README.md#fusion-and-evaluation-protocol).
- Reproducing the saved submission requires its original audio, labels, model revision, and run configuration, which are absent. The repository cannot independently establish the official result.

## Team and attribution

We developed HEARSAY as a team. The current branch's commit history credits Jonathan Chang with early audio-processing, AASIST, harmonic/splice, ENF, and Eliya integration work, and Malachi Wang with the evaluator, CPPS/splice refinements, novelty analysis, fusion infrastructure, sliding-window integration, runner, Docker packaging, and Grok integration. Historical residual-fingerprint work by `himankgalundia` is recorded separately in the [experiment notes](legacy/README.md#historical-work).

The active pretrained model is `eliya/forensics_0.3B_base_deepfake_classifier`, with `microsoft/wavlm-large` as the externally fetched backbone. CPPS uses Praat through Parselmouth; physical splice features use PyWavelets and SciPy. The metric follows the ASVspoof5 Track 1 convention encoded in the evaluator and its local reference test. No challenge audio or third-party training corpus is redistributed here; upstream model code, weights, and data remain separately attributed assets.
