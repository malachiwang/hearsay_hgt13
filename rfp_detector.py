"""
RFP (Residual Fingerprint) detector for HEARSAY challenge.

Based on: "Lightweight Model Attribution and Detection of Synthetic Speech
via Audio Residual Fingerprints" (Pizarro et al., 2024)

Pipeline:
  1. Build spectral fingerprints for both real and fake sources
  2. For each test file, compute min Mahalanobis distance to real vs fake
  3. Score = d_real / (d_real + d_fake): 0 = real, 1 = synthetic
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
        self.real_keys = set()
        self.fake_keys = set()

    def build_fingerprint(self, gen_name, file_paths, max_n=500, is_real=False):
        """Build fingerprint for one source (real or fake) from audio files."""
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

        if is_real:
            self.real_keys.add(gen_name)
        else:
            self.fake_keys.add(gen_name)
        print(f"  {gen_name}: fingerprint from {len(residuals)} samples ({'real' if is_real else 'fake'})")

    def mahalanobis(self, residual, gen_name):
        diff = residual - self.fingerprints[gen_name]
        return float(np.sqrt(np.clip(diff @ self.cov_invs[gen_name] @ diff, 0, None)))

    def score_waveform(self, wav):
        """Relative distance score: d_real / (d_real + d_fake).
        0 = closer to real, 1 = closer to fake."""
        r = compute_residual(wav, self.fir)
        d_real = min(self.mahalanobis(r, g) for g in self.real_keys) if self.real_keys else float('inf')
        d_fake = min(self.mahalanobis(r, g) for g in self.fake_keys) if self.fake_keys else float('inf')
        if d_real + d_fake == 0:
            return 0.5
        return d_real / (d_real + d_fake)

    def score_waveform_detailed(self, wav):
        """Return distance to each fingerprint, plus the final score."""
        r = compute_residual(wav, self.fir)
        dists = {g: self.mahalanobis(r, g) for g in self.fingerprints}
        d_real = min(dists[g] for g in self.real_keys) if self.real_keys else float('inf')
        d_fake = min(dists[g] for g in self.fake_keys) if self.fake_keys else float('inf')
        dists['_d_real'] = d_real
        dists['_d_fake'] = d_fake
        dists['_score'] = d_real / (d_real + d_fake) if (d_real + d_fake) > 0 else 0.5
        return dists

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'filter_type': self.filter_type,
                'fingerprints': self.fingerprints,
                'cov_invs': self.cov_invs,
                'real_keys': self.real_keys,
                'fake_keys': self.fake_keys,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.fingerprints = data['fingerprints']
        self.cov_invs = data['cov_invs']
        self.real_keys = data.get('real_keys', set())
        self.fake_keys = data.get('fake_keys', set())


# --- Pipeline ---

def build_from_manifest(manifest_path=MANIFEST_PATH, n_per_gen=500,
                        filter_type='lowpass'):
    """Read manifest, build fingerprints for both real and fake sources."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    fake_files = {}
    real_lj = []
    real_libri = []

    for r in rows:
        if r['label'] == 'spoof':
            fake_files.setdefault(r['generator'], []).append(r['path'])
        elif r['label'] == 'bonafide':
            if r['speaker'].startswith('libri_'):
                real_libri.append(r['path'])
            else:
                real_lj.append(r['path'])

    detector = RFPDetector(filter_type=filter_type)
    t0 = time.time()

    print(f"Building real fingerprints...")
    print(f"  real_lj: {len(real_lj)} files, sampling {n_per_gen}")
    detector.build_fingerprint('real_lj', real_lj, max_n=n_per_gen, is_real=True)
    print(f"  real_libri: {len(real_libri)} files, sampling {n_per_gen}")
    detector.build_fingerprint('real_libri', real_libri, max_n=n_per_gen, is_real=True)

    print(f"\nBuilding fake fingerprints...")
    for gen_name in sorted(fake_files):
        print(f"  {gen_name}: {len(fake_files[gen_name])} files, sampling {n_per_gen}")
        detector.build_fingerprint(gen_name, fake_files[gen_name], max_n=n_per_gen, is_real=False)

    elapsed = time.time() - t0
    print(f"\nAll fingerprints built in {elapsed:.1f}s")
    print(f"Real fingerprints: {sorted(detector.real_keys)}")
    print(f"Fake fingerprints: {sorted(detector.fake_keys)}")
    return detector


def score_files(detector, file_paths):
    """Score a list of files. Returns {filename: cm_score} where 0=real, 1=fake."""
    results = {}
    t0 = time.time()

    for i, p in enumerate(file_paths):
        p = Path(p)
        try:
            wav, _ = load_audio(str(p))
            results[p.name] = detector.score_waveform(wav)
        except Exception as e:
            print(f"  Error scoring {p.name}: {e}")
            results[p.name] = 0.5

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(file_paths) - i - 1) / rate
            print(f"  {i+1}/{len(file_paths)} "
                  f"({rate:.1f} files/s, ~{remaining:.0f}s remaining)")

    return results


def write_tsv(scores, output_path):
    """Write submission TSV."""
    with open(output_path, 'w', newline='') as f:
        f.write("filename\tcm-score\n")
        for name in sorted(scores):
            f.write(f"{name}\t{scores[name]:.6f}\n")
    print(f"Submission written to {output_path} ({len(scores)} files)")


def evaluate_on_manifest(detector, manifest_path=MANIFEST_PATH, fold=0):
    """Evaluate detector on a validation fold. Returns scores (0=real,1=fake) and labels."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    val_rows = [r for r in rows if int(r['fold']) == fold]
    print(f"Fold {fold}: {len(val_rows)} files "
          f"({sum(1 for r in val_rows if r['label']=='bonafide')} real, "
          f"{sum(1 for r in val_rows if r['label']=='spoof')} spoof)")

    scores = {}
    labels = {}
    t0 = time.time()

    for i, r in enumerate(val_rows):
        p = Path(r['path'])
        try:
            wav, _ = load_audio(str(p))
            scores[p.name] = detector.score_waveform(wav)
            labels[p.name] = r['label']
        except Exception as e:
            print(f"  skip {p.name}: {e}")

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(val_rows)} ({(i+1)/elapsed:.1f} files/s)")

    return scores, labels


# --- Main ---

if __name__ == '__main__':
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "rfp_v2.pkl"

    # Step 1: Build or load fingerprints (real + fake)
    if cache_file.exists():
        print(f"Loading cached fingerprints from {cache_file}...")
        detector = RFPDetector(filter_type='lowpass')
        detector.load(str(cache_file))
        print(f"Real: {sorted(detector.real_keys)}")
        print(f"Fake: {sorted(detector.fake_keys)}")
    else:
        print("Building fingerprints from manifest (real + fake)...")
        detector = build_from_manifest(n_per_gen=500, filter_type='lowpass')
        detector.save(str(cache_file))
        print(f"Saved to {cache_file}")

    # Step 2: Score test files (scores are already 0=real, 1=fake)
    test_files = sorted(TEST_DIR.glob("*.wav"))
    print(f"\nScoring {len(test_files)} test files...")
    scores = score_files(detector, test_files)

    # Step 3: Write TSV
    output_tsv = CACHE_DIR / "rfp_v2_submission.tsv"
    write_tsv(scores, output_tsv)

    # Print score statistics
    vals = list(scores.values())
    print(f"\nScore stats: min={min(vals):.4f}, max={max(vals):.4f}, "
          f"median={np.median(vals):.4f}, mean={np.mean(vals):.4f}")
    print(f"Files scoring > 0.5 (likely fake): {sum(1 for v in vals if v > 0.5)}")
    print(f"Files scoring < 0.5 (likely real): {sum(1 for v in vals if v < 0.5)}")
