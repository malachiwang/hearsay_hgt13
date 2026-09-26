import numpy as np
import parselmouth
from parselmouth.praat import call

def compute_cpps(path, f0min=75, f0max=500):
    """
    Smoothed Cepstral Peak Prominence (dB). More robust than HNR to
    noise/formant interference and widely used as a periodicity/harmonic
    clarity measure in modern voice-quality literature.
    """
    power_cepstrogram = call(parselmouth.Sound(path), "To PowerCepstrogram", 60, 0.002, 5000, 50)
    cpps = call(power_cepstrogram, "Get CPPS",
                False, 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0,
                "Straight", "Robust")
    return float(cpps)