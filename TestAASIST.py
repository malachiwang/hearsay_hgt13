import json
import torch
import torch.nn.functional as F
from models.AASIST import Model  # the class name is literally "Model" in models/AASIST.py

# Load the architecture config that matches the checkpoint
with open("aasist/config/AASIST.conf") as f:
    config = json.load(f)
model_config = config["model_config"]

# Build model and load weights
model = Model(model_config)
state_dict = torch.load("aasist/models/weights/AASIST.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

import numpy as np
import librosa

def pad_or_crop(wav, target_len=64600):
    if len(wav) >= target_len:
        return wav[:target_len]
    # tile-repeat to fill, matching the repo's own padding logic
    num_repeats = int(target_len / len(wav)) + 1
    return np.tile(wav, num_repeats)[:target_len]

from pathlib import Path
mi, ma = float('inf'), float('-inf')
avg = 0
root_dir = Path("C:/Users/26cha/PycharmProjects/hearsay_hgt13/playht/speaker_100")
files_processed = 0
l = []
for file_path in root_dir.rglob("*"):
    if not file_path.is_file():
        continue

    wav, sr = librosa.load(file_path, sr=16000)
    wav = pad_or_crop(wav).astype(np.float32)

    files_processed += 1

    x = torch.from_numpy(wav).unsqueeze(0)  # add batch dim -> shape (1, 64600)


    with torch.no_grad():
        _, output = model(x)  # model returns (hidden_features, output_logits)
        probs = F.softmax(output, dim=1)
        bonafide_prob = probs[0, 1].item()
        mi, ma = min(bonafide_prob, mi), max(bonafide_prob, ma)
        avg += bonafide_prob
avg /= files_processed

print(mi, ma, avg)