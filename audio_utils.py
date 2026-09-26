import soundfile as sf
import soxr
import numpy as np


def load_audio(path, target_sr=16000):
    """Content-based decoder: reads WAV, FLAC, MP3, MP3-in-WAV transparently."""
    data, sr = sf.read(path, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        data = soxr.resample(data, sr, target_sr, quality='HQ')
    return data, sr
