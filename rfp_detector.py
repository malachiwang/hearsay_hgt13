"""
RFP (Residual Fingerprint) detector for HEARSAY challenge.

Based on: "Lightweight Model Attribution and Detection of Synthetic Speech
via Audio Residual Fingerprints" (Pizarro et al., 2024)

Pipeline:
  1. Build spectral fingerprint per generator from training data
  2. For each test file, compute min Mahalanobis distance to any generator
  3. Lower distance = closer to known generator = more likely synthetic
"""

import numpy as np
from scipy import signal as sig
from pathlib import Path
import csv
import random
import pickle
import time
from audio_utils import load_audio

# --- Configuration ---
SAMPLE_RATE = 16000
WIN_MS = 25
HOP_MS = 10
WIN_SAMPLES = int(SAMPLE_RATE * WIN_MS / 1000)   # 400
HOP_SAMPLES = int(SAMPLE_RATE * HOP_MS / 1000)   # 160
N_FFT = 512                                        # 257 freq bins
FIR_TAPS = 101
N_FREQ_BINS = N_FFT // 2 + 1                      # 257

MANIFEST_PATH = Path(r"C:\hearsay_data\manifest.csv")
CACHE_DIR = Path(r"C:\hearsay_cache")
TEST_DIR = Path(r"C:\Users\himan\OneDrive - Georgia Institute of Technology"
                r"\Desktop\Hackgt 2026\hearsay_hgt13\test\HackGTHearsayTesting")


# --- Signal processing ---

def avg_spectral_energy(wav):
    """STFT -> magnitude (dB) -> mean over time -> fixed-size vector (F,)."""
    _, _, Zxx = sig.stft(wav, fs=SAMPLE_RATE, window='hann',
                         nperseg=WIN_SAMPLES,
                         noverlap=WIN_SAMPLES - HOP_SAMPLES,
                         nfft=N_FFT)
    mag_db = 20.0 * np.log10(np.abs(Zxx) + 1e-10)
    return mag_db.mean(axis=1)


def make_lowpass(cutoff=1000):
    return sig.firwin(FIR_TAPS, cutoff, fs=SAMPLE_RATE, window='hamming')


def make_bandpass(lo=5000, hi=6000):
    return sig.firwin(FIR_TAPS, [lo, hi], fs=SAMPLE_RATE,
                      pass_zero=False, window='hamming')


def compute_residual(wav, fir_coeffs):
    """R = E(x) - E(f(x))  where f is a FIR filter."""
    filtered = sig.lfilter(fir_coeffs, 1.0, wav)
    return avg_spectral_energy(wav) - avg_spectral_energy(filtered)


# --- Detector class ---

class RFPDetector:
    def __init__(self, filter_type='lowpass'):
        if filter_type == 'lowpass':
            self.fir = make_lowpass(1000)
        elif filter_type == 'bandpass':
            self.fir = make_bandpass(5000, 6000)
        else:
            raise ValueError(f"Unknown filter: {filter_type}")
        self.filter_type = filter_type
        self.fingerprints = {}
        self.cov_invs = {}

    def build_fingerprint(self, gen_name, file_paths, max_n=500):
        """Build fingerprint for one generator from audio files."""
        paths = list(file_paths)
        if len(paths) > max_n:
            paths = random.Random(42).sample(paths, max_n)

        residuals = []
        for i, p in enumerate(paths):
            try:
                wav, _ = load_audio(str(p))
                residuals.append(compute_residual(wav, self.fir))
            except Exception as e:
                print(f"    skip {Path(p).name}: {e}")
            if (i + 1) % 100 == 0:
                print(f"    {gen_name}: {i+1}/{len(paths)}")

        residuals = np.array(residuals)
        self.fingerprints[gen_name] = residuals.mean(axis=0)
        cov = np.cov(residuals, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_invs[gen_name] = np.linalg.inv(cov)
        print(f"  {gen_name}: fingerprint from {len(residuals)} samples")

    def mahalanobis(self, residual, gen_name):
        diff = residual - self.fingerprints[gen_name]
        return float(np.sqrt(np.clip(diff @ self.cov_invs[gen_name] @ diff, 0, None)))

    def score_waveform(self, wav):
        """Min Mahalanobis distance to any generator.
        Lower = more likely synthetic."""
        r = compute_residual(wav, self.fir)
        return min(self.mahalanobis(r, g) for g in self.fingerprints)

    def score_waveform_detailed(self, wav):
        """Return distance to each generator."""
        r = compute_residual(wav, self.fir)
        return {g: self.mahalanobis(r, g) for g in self.fingerprints}

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'filter_type': self.filter_type,
                'fingerprints': self.fingerprints,
                'cov_invs': self.cov_invs,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.fingerprints = data['fingerprints']
        self.cov_invs = data['cov_invs']


# --- Pipeline ---

def build_from_manifest(manifest_path=MANIFEST_PATH, n_per_gen=500,
                        filter_type='lowpass'):
    """Read manifest, sample files per generator, build fingerprints."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    gen_files = {}
    for r in rows:
        if r['label'] == 'spoof':
            gen_files.setdefault(r['generator'], []).append(r['path'])

    detector = RFPDetector(filter_type=filter_type)
    t0 = time.time()

    for gen_name in sorted(gen_files):
        print(f"Building fingerprint: {gen_name} "
              f"({len(gen_files[gen_name])} files, sampling {n_per_gen})...")
        detector.build_fingerprint(gen_name, gen_files[gen_name], max_n=n_per_gen)

    elapsed = time.time() - t0
    print(f"\nAll fingerprints built in {elapsed:.1f}s")
    return detector


def score_files(detector, file_paths):
    """Score a list of files. Returns {filename: min_mahal_distance}."""
    results = {}
    t0 = time.time()

    for i, p in enumerate(file_paths):
        p = Path(p)
        try:
            wav, _ = load_audio(str(p))
            results[p.name] = detector.score_waveform(wav)
        except Exception as e:
            print(f"  Error scoring {p.name}: {e}")
            results[p.name] = float('nan')

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(file_paths) - i - 1) / rate
            print(f"  {i+1}/{len(file_paths)} "
                  f"({rate:.1f} files/s, ~{remaining:.0f}s remaining)")

    return results


def distances_to_scores(distances):
    """Convert Mahalanobis distances to cm-scores (0=real, 1=synthetic).
    Lower distance → higher score (more synthetic)."""
    scores = {}
    for name, dist in distances.items():
        if np.isnan(dist):
            scores[name] = 0.5
        else:
            scores[name] = 1.0 / (1.0 + dist)
    return scores


def write_tsv(scores, output_path):
    """Write submission TSV."""
    with open(output_path, 'w', newline='') as f:
        f.write("filename\tcm-score\n")
        for name in sorted(scores):
            f.write(f"{name}\t{scores[name]:.6f}\n")
    print(f"Submission written to {output_path} ({len(scores)} files)")


def evaluate_on_manifest(detector, manifest_path=MANIFEST_PATH, fold=0):
    """Evaluate detector on a validation fold. Returns distances and labels."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    val_rows = [r for r in rows if int(r['fold']) == fold]
    print(f"Fold {fold}: {len(val_rows)} files "
          f"({sum(1 for r in val_rows if r['label']=='bonafide')} real, "
          f"{sum(1 for r in val_rows if r['label']=='spoof')} spoof)")

    distances = {}
    labels = {}
    t0 = time.time()

    for i, r in enumerate(val_rows):
        p = Path(r['path'])
        try:
            wav, _ = load_audio(str(p))
            distances[p.name] = detector.score_waveform(wav)
            labels[p.name] = r['label']
        except Exception as e:
            print(f"  skip {p.name}: {e}")

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(val_rows)} ({(i+1)/elapsed:.1f} files/s)")

    return distances, labels


# --- Main ---

if __name__ == '__main__':
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "rfp_lowpass.pkl"

    # Step 1: Build or load fingerprints
    if cache_file.exists():
        print(f"Loading cached fingerprints from {cache_file}...")
        detector = RFPDetector(filter_type='lowpass')
        detector.load(str(cache_file))
        print(f"Loaded fingerprints for: {sorted(detector.fingerprints.keys())}")
    else:
        print("Building fingerprints from manifest...")
        detector = build_from_manifest(n_per_gen=500, filter_type='lowpass')
        detector.save(str(cache_file))
        print(f"Saved to {cache_file}")

    # Step 2: Score test files
    test_files = sorted(TEST_DIR.glob("*.wav"))
    print(f"\nScoring {len(test_files)} test files...")
    distances = score_files(detector, test_files)

    # Step 3: Convert to cm-scores and write TSV
    scores = distances_to_scores(distances)
    output_tsv = CACHE_DIR / "rfp_submission.tsv"
    write_tsv(scores, output_tsv)

    # Print distance statistics
    dists = [d for d in distances.values() if not np.isnan(d)]
    print(f"\nDistance stats: min={min(dists):.2f}, max={max(dists):.2f}, "
          f"median={np.median(dists):.2f}, mean={np.mean(dists):.2f}")
    cm = list(scores.values())
    print(f"Score stats: min={min(cm):.4f}, max={max(cm):.4f}, "
          f"median={np.median(cm):.4f}")
