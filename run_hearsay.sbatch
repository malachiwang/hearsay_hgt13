#!/bin/bash
#SBATCH -J hearsay-gpu                        # Job name
#SBATCH -N1 --ntasks-per-node=1               # 1 node, 1 task
#SBATCH --cpus-per-task=8                     # 8 CPU cores for LFCC multiprocessing
#SBATCH --gres=gpu:H100:1                    # 1x H100 GPU
#SBATCH --mem-per-gpu=224G                   # Recommended for H100 nodes
#SBATCH -t 4:00:00                           # 4 hours (well within 16 GPU-hr limit)
#SBATCH -o hearsay-%j.out                    # Output log
#SBATCH --mail-type=BEGIN,END,FAIL           # Email notifications
#SBATCH --mail-user=himankgalundia@gmail.com

# ── Change into code directory ──
# Assumes you ran:  cd ~/scratch && sbatch hearsay_hgt13/run_hearsay.sbatch
cd $SLURM_SUBMIT_DIR/hearsay_hgt13

# ── Environment ──
export HF_HOME="$HOME/scratch/.cache/huggingface"
VENV_DIR="$HOME/scratch/hearsay_venv"

module load anaconda3
module load cuda

# ── Activate venv (run setup_pace.sh first!) ──
# If venv doesn't exist, auto-install as fallback
if [ ! -d "$VENV_DIR" ]; then
    echo "WARNING: venv not found — installing now (run setup_pace.sh next time)"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    pip install transformers soundfile soxr numpy scipy scikit-learn xgboost librosa
else
    source "$VENV_DIR/bin/activate"
fi

# ── Diagnostics ──
echo "══════════════════════════════════════"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"NONE\"}')"
echo "══════════════════════════════════════"

# ── Run pipeline ──
# Flags you can add:
#   --no-val          Train on ALL data (final submission)
#   --max-fake 50000  Cap total fake samples (balanced across generators)
#   --max-real 30000  Cap total real samples
python3 -u pace_pipeline.py \
    --data-root "$HOME/scratch/hearsay_data" \
    --test-dir  "$HOME/scratch/hearsay_test/HackGTHearsayTesting" \
    --cache-dir "$HOME/scratch/hearsay_cache" \
    --batch-size 32 \
    --n-workers 8 \
    --max-fake 50000

echo "══════════════════════════════════════"
echo "Pipeline complete."
echo "Outputs in: ~/scratch/hearsay_cache/"
echo "Download the *_submission_*.tsv files."
echo "══════════════════════════════════════"
