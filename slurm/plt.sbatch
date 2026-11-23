#!/bin/bash
#SBATCH --job-name=plt
#SBATCH -p gpu-l40
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-logs/plt-%j.out
# Usage (from the PersLoraTrain root, env with perslora + persvectrain + persvecfast active):
#   sbatch slurm/plt.sbatch train --source S --out O --run-name R --traits Assertive Sarcastic --wandb-project P
set -euo pipefail
mkdir -p slurm-logs
PYTHON="${PYTHON:-$(command -v python || true)}"
[ -x "$PYTHON" ] || { echo "no python: activate the env before sbatch" >&2; exit 1; }
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
echo "$(date)  $(hostname)  python=$PYTHON  args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
"$PYTHON" -m perslora "$@"
