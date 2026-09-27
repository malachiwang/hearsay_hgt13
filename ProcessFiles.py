import subprocess, librosa, json

def normalize_file(file_in):
    if not file_in.endswith(".wav"):
        normalized_file = f"{file_in.rsplit('.')[0]}.wav"
        subprocess.run(["ffmpeg", "-i", file_in, normalized_file], check=True)
        file_in = normalized_file

    y, _ = librosa.load(file_in, sr=16000, mono=True)

    return y #amplitudes at sampled times

