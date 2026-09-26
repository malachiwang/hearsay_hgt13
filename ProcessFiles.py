import subprocess, librosa, json

def normalize_file(file_in):
    if not file_in.endswith(".mp3"):
        normalized_file = f"{file_in.rsplit('.')[0]}.mp3"
        subprocess.run(["ffmpeg", "-i", file_in, normalized_file], check=True)
        file_in = normalized_file

    y, _ = librosa.load(file_in, sr=16000, mono=True)

    return y #amplitudes at sampled times

def get_metadata(file_in):
    res = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", file_in], check=True, text=True, capture_output=True)
    return json.loads(res.stdout)

