#!/bin/bash
#SBATCH -J forensics-gpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=224G
#SBATCH -t 1:00:00
#SBATCH -o forensics-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=himankgalundia@gmail.com

cd $SLURM_SUBMIT_DIR/hearsay_hgt13

export HF_HOME="$HOME/scratch/.cache/huggingface"
VENV_DIR="$HOME/scratch/hearsay_venv"
MODEL_DIR="$HOME/scratch/forensics_model"
CACHE_DIR="$HOME/scratch/hearsay_cache"
TEST_DIR="$HOME/scratch/hearsay_test/HackGTHearsayTesting"

module load anaconda3
module load cuda
source "$VENV_DIR/bin/activate"

# ── Install dependencies ──
pip install safetensors 2>/dev/null

# ── Diagnostics ──
echo "══════════════════════════════════════"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo "══════════════════════════════════════"

# ── Download forensics model (skip the 3.8GB .pt, only get safetensors + code) ──
if [ ! -f "$MODEL_DIR/checkpoint_epoch_5.safetensors" ]; then
    echo "Downloading forensics model (safetensors + code only)..."
    pip install huggingface_hub 2>/dev/null
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='eliya/forensics_0.3B_base_deepfake_classifier',
    local_dir='$MODEL_DIR',
    allow_patterns=['*.py', '*.json', '*.txt', '*.safetensors', '*.md'],
)
print('Model downloaded.')
"
    echo "Contents of $MODEL_DIR:"
    ls -lh "$MODEL_DIR"
fi

# ── Verify the checkpoint actually downloaded ──
if [ ! -f "$MODEL_DIR/checkpoint_epoch_5.safetensors" ]; then
    echo "ERROR: Model checkpoint not found after download attempt."
    echo "Check internet access on this node or download manually on the login node."
    exit 1
fi

# ── Pre-cache WavLM-large (the model constructor downloads it from HF) ──
echo "Ensuring WavLM-large is cached..."
python3 -c "from transformers import WavLMModel; WavLMModel.from_pretrained('microsoft/wavlm-large'); print('WavLM-large cached.')"

# ── Verify test files exist ──
TEST_COUNT=$(ls "$TEST_DIR"/*.wav 2>/dev/null | wc -l)
echo "Test files found: $TEST_COUNT"
if [ "$TEST_COUNT" -eq 0 ]; then
    echo "ERROR: No test WAV files in $TEST_DIR"
    echo "Upload test files before running this job."
    exit 1
fi

# ── Run inference ──
python3 -u forensics_infer.py \
    --test-dir  "$TEST_DIR" \
    --model-dir "$MODEL_DIR" \
    --output    "$CACHE_DIR/forensics_submission.tsv" \
    --batch-size 8

echo ""
echo "══════════════════════════════════════"
echo "Pipeline complete."
echo "Output: $CACHE_DIR/forensics_submission.tsv"
echo "══════════════════════════════════════"
