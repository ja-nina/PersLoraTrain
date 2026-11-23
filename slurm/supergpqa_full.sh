#!/bin/bash
# SuperGPQA accuracy at the games' 15360-token limit, one GPU job per condition (each ~1.5-2 h).
# Redoes base and the MVP (--force), and for each training run waits for its train job:
#   bash slurm/supergpqa_full.sh kl=<train jobid> onpol=<jobid> correct=<jobid> leak=<jobid>
set -euo pipefail
O="${O:-/scratch/inf0/user/nzukowsk/PersVecFast/out}"
WB="${WB:---wandb-project perslora}"
A=assertive_pos
B=sarcastic_pos
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
acc() { sbatch --parsable "$@" | tr '\n' ' '; }

echo -n "base + mvp: "
acc slurm/plt.sbatch accuracy --out "$O" --run-name _base --task supergpqa --cond base --force $WB
for c in $A $B "$A:0.5+$B:0.5" "$A+$B"; do
  acc slurm/plt.sbatch accuracy --out "$O" --run-name mvp --task supergpqa --cond "$c" --force $WB
done
echo
for kv in "$@"; do
  r="${kv%%=*}"; j="${kv#*=}"
  echo -n "$r (after $j): "
  for c in $A $B "$A@step_75" "$B@step_75" "$A:0.5+$B:0.5" "$A+$B"; do
    acc --dependency=afterok:$j slurm/plt.sbatch accuracy --out "$O" --run-name "$r" --task supergpqa --cond "$c" $WB
  done
  echo
done
