#!/bin/bash
# ────────────────────────────────────────────────────────────
# HEARSAY — One-time setup on PACE login node
# Run this BEFORE submitting the sbatch job.
#
#   bash setup_pace.sh
#
# This installs everything so the GPU job doesn't waste time
# on pip install.
# ────────────────────────────────────────────────────────────
set -e

echo "══ HEARSAY PACE Setup ══"

module load anaconda3
module load cuda

VENV_DIR="$HOME/scratch/hearsay_venv"
export HF_HOME="$HOME/scratch/.cache/huggingface"

# ── Create venv ──
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
else
    echo "Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# ── Install PyTorch with CUDA 12.4 ──
# H100/H200 nodes have CUDA 12+ drivers.
# If this fails, try:  cu121  or  cu118
echo "Installing PyTorch (CUDA 12.4) ..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ── Install other dependencies ──
echo "Installing ML dependencies ..."
pip install transformers soundfile soxr numpy scipy scikit-learn xgboost librosa

# ── Pre-download WavLM model ──
echo "Pre-downloading WavLM-base-plus ..."
python3 -c "
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
WavLMModel.from_pretrained('microsoft/wavlm-base-plus')
Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus')
print('Model cached.')
"

# ── Verify GPU torch ──
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version:   {torch.version.cuda}')
"

# ── Create output directory ──
mkdir -p "$HOME/scratch/hearsay_cache"

echo ""
echo "══ Setup complete! ══"
echo ""
echo "Directory structure needed on PACE:"
echo "  ~/scratch/"
echo "  ├── hearsay_code/              ← your code (pace_pipeline.py etc.)"
echo "  ├── hearsay_data/              ← training data (same structure as C:\\hearsay_data)"
echo "  │   ├── manifest.csv"
echo "  │   ├── generated_speech/"
echo "  │   ├── LJRealResampled/"
echo "  │   ├── LJSpeech-1.1/"
echo "  │   └── librispeech/"
echo "  ├── hearsay_test/"
echo "  │   └── HackGTHearsayTesting/  ← 1,671 test WAVs"
echo "  ├── hearsay_venv/              ← created by this script"
echo "  └── hearsay_cache/             ← outputs go here"
echo ""
echo "To submit:  cd ~/scratch && sbatch hearsay_code/run_hearsay.sbatch"
