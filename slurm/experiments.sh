#!/bin/bash
# Accuracy-protection experiment: submits every job with its dependencies. Run from the PersLoraTrain root on
# the login node, with the env active (it downloads GSM8K first, which needs internet):
#   bash slurm/experiments.sh
# Afterwards (login node, no GPU): bash slurm/experiments.sh gates
# Only the training runs (+ their eval / accuracy), when references and base / MVP accuracy exist already:
#   bash slurm/experiments.sh runs
# Resume crashed runs (keeps adapters saved at step >= KEEP=140, retrains the rest; RUNS="kl onpol" to pick):
#   bash slurm/experiments.sh resume
set -euo pipefail
O="${O:-/scratch/inf0/user/nzukowsk/PersVecFast/out}"
S="${S:-$O}"
WB="${WB:---wandb-project perslora}"
TRAITS="Assertive Sarcastic"
A=assertive_pos
B=sarcastic_pos
RUNS="${RUNS:-kl onpol correct leak}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

if [ "${1:-}" = "gates" ]; then
  python -m perslora report --out "$O" $WB || true
  for r in mvp $RUNS; do
    echo "=== $r"
    python -m perslora gate --out "$O" --run-name "$r" --task supergpqa gsm8k || true
  done
  exit 0
fi

sb() { sbatch --parsable "$@"; }
REFS="$O/lora/refs/supergpqa-pool.jsonl $O/lora/refs/gsm8k-train.jsonl"
DEP=""
OW=""

if [ "${1:-}" != "runs" ] && [ "${1:-}" != "resume" ]; then
python -m perslora prepare --out "$O"

# base answers as KL references: science (pool minus the KL eval's 100 held questions), math (correct only)
REF_SCI=$(sb slurm/plt.sbatch reference --out "$O" --task supergpqa-pool --samples 2 --max-tokens 3072)
REF_MATH=$(sb slurm/plt.sbatch reference --out "$O" --task gsm8k-train --n 400 --samples 1 --max-tokens 2048 --correct-only)
DEP="--dependency=afterok:$REF_SCI:$REF_MATH"

# base accuracy (shared by all runs) and the MVP adapters' accuracy
for t in supergpqa gsm8k; do
  sb slurm/plt.sbatch accuracy --out "$O" --run-name _base --task "$t" --cond base $WB >/dev/null
done
sb slurm/plt.sbatch accuracy --out "$O" --run-name mvp --task gsm8k --cond $A $B "$A:0.5+$B:0.5" "$A+$B" $WB >/dev/null
for c in $A $B "$A:0.5+$B:0.5" "$A+$B"; do  # 15360 tokens: one job each
  sb slurm/plt.sbatch accuracy --out "$O" --run-name mvp --task supergpqa --cond "$c" $WB >/dev/null
done
echo "references: $REF_SCI $REF_MATH"
else
  OW="--overwrite"  # the runs restart from scratch
  # resume after a crash: keep adapters saved at step >= KEEP (default 140 of 150), retrain the others
  [ "$1" = "resume" ] && OW="--keep-from ${KEEP:-140}"
  for f in $REFS; do [ -s "$f" ] || { echo "$f missing: run the full script first" >&2; exit 1; }; done
fi

# shared training settings: coherence floor at the base level, a KL budget that binds (eta scaled to KL ~ 1e-2)
COMMON="--source $S --out $O --traits $TRAITS --coh-floor 85 --eps-kl 0.005 --eta-kl 50 --kl-batch 4 --kl-refs $REFS --eval-every 25 $OW $WB"
REASON="--reason-questions 4 --reason-g 4 --reason-tokens 2048 --lambda-onpolicy 2.0"
declare -A EXTRA=(
  [kl]=""
  [onpol]="$REASON"
  [correct]="$REASON --w-correct 0.5"
  [leak]="$REASON --w-correct 0.5 --c-leak 1.0"
)
for r in $RUNS; do
  J=$(sb $DEP slurm/plt.sbatch train --run-name "$r" $COMMON ${EXTRA[$r]})
  sb --dependency=afterok:$J slurm/plt.sbatch evaluate --out "$O" --run-name "$r" --grid 0,0.5,1 $WB >/dev/null
  sb --dependency=afterok:$J slurm/plt.sbatch accuracy --out "$O" --run-name "$r" --task gsm8k \
     --cond $A $B "$A@step_75" "$B@step_75" "$A:0.5+$B:0.5" "$A+$B" $WB >/dev/null
  for c in $A $B "$A@step_75" "$B@step_75" "$A:0.5+$B:0.5" "$A+$B"; do  # 15360 tokens: one job each
    sb --dependency=afterok:$J slurm/plt.sbatch accuracy --out "$O" --run-name "$r" --task supergpqa \
       --cond "$c" $WB >/dev/null
  done
  echo "run $r: train job $J"
done
echo "squeue -u \$USER to follow; when all are done: bash slurm/experiments.sh gates"
