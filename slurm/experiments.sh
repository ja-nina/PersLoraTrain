#!/bin/bash
# Accuracy-protection experiment: submits every job with its dependencies. Run from the PersLoraTrain root on
# the login node, with the env active (it downloads GSM8K first, which needs internet):
#   bash slurm/experiments.sh
# Afterwards (login node, no GPU): bash slurm/experiments.sh gates
set -euo pipefail
O="${O:-/scratch/inf0/user/nzukowsk/PersVecFast/out}"
S="${S:-$O}"
WB="${WB:---wandb-project perslora}"
TRAITS="Assertive Sarcastic"
A=assertive_pos
B=sarcastic_pos
RUNS="kl onpol correct leak"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

if [ "${1:-}" = "gates" ]; then
  for r in mvp $RUNS; do
    echo "=== $r"
    python -m perslora gate --out "$O" --run-name "$r" --task supergpqa gsm8k || true
  done
  exit 0
fi

sb() { sbatch --parsable "$@"; }

python -m perslora prepare --out "$O"

# base answers as KL references: science (pool minus the KL eval's 100 held questions), math (correct only)
REF_SCI=$(sb slurm/plt.sbatch reference --out "$O" --task supergpqa-pool --samples 2 --max-tokens 3072)
REF_MATH=$(sb slurm/plt.sbatch reference --out "$O" --task gsm8k-train --n 400 --samples 1 --max-tokens 2048 --correct-only)
REFS="$O/lora/refs/supergpqa-pool.jsonl $O/lora/refs/gsm8k-train.jsonl"

# base accuracy (shared by all runs) and the MVP adapters' accuracy
for t in supergpqa gsm8k; do
  sb slurm/plt.sbatch accuracy --out "$O" --run-name _base --task "$t" --cond base >/dev/null
  sb slurm/plt.sbatch accuracy --out "$O" --run-name mvp --task "$t" \
     --cond $A $B "$A:0.5+$B:0.5" "$A+$B" >/dev/null
done

# shared training settings: coherence floor at the base level, a KL budget that binds (eta scaled to KL ~ 1e-2)
COMMON="--source $S --out $O --traits $TRAITS --coh-floor 85 --eps-kl 0.005 --eta-kl 50 --kl-batch 4 --kl-refs $REFS $WB"
REASON="--reason-questions 4 --reason-g 4 --reason-tokens 2048 --lambda-onpolicy 2.0"
declare -A EXTRA=(
  [kl]=""
  [onpol]="$REASON"
  [correct]="$REASON --w-correct 0.5"
  [leak]="$REASON --w-correct 0.5 --c-leak 1.0"
)
for r in $RUNS; do
  J=$(sb --dependency=afterok:$REF_SCI:$REF_MATH slurm/plt.sbatch train --run-name "$r" $COMMON ${EXTRA[$r]})
  sb --dependency=afterok:$J slurm/plt.sbatch evaluate --out "$O" --run-name "$r" --grid 0,0.5,1 $WB >/dev/null
  for t in supergpqa gsm8k; do
    sb --dependency=afterok:$J slurm/plt.sbatch accuracy --out "$O" --run-name "$r" --task "$t" \
       --cond $A $B "$A@step_75" "$B@step_75" "$A:0.5+$B:0.5" "$A+$B" >/dev/null
  done
  echo "run $r: train job $J"
done
echo "references: $REF_SCI $REF_MATH; squeue -u \$USER to follow; when all are done: bash slurm/experiments.sh gates"
