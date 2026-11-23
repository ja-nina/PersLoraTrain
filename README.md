# PersLoraTrain

Persona LoRA adapters for Qwen3.5-2B, trained by GRPO with a judge score-vector reward (trait vs opposite,
coherence, on-task, denial; one OpenRouter call per rollout) and a KL budget to the base model on SuperGPQA answers.
Design: `docs/superpowers/specs/2026-11-23-persona-lora-design.md` (MVP section first). Reads PersVecFast's persona
pairs and PersVecTrain's KL reference answers; imports both packages.

## Install

    pip install -e ../PersVecFast -e ../PersVecTrain -e ".[test,wandb]"

## MVP run (cluster)

    sbatch slurm/plt.sbatch train --source /scratch/inf0/user/nzukowsk/PersVecFast/out \
        --out /scratch/inf0/user/nzukowsk/PersVecFast/out --run-name mvp \
        --traits Assertive Sarcastic --wandb-project perslora
    sbatch slurm/plt.sbatch evaluate --out /scratch/inf0/user/nzukowsk/PersVecFast/out --run-name mvp \
        --wandb-project perslora

`--out` is PersVecTrain's out root (it holds `trained/reference/<model>_pool260.jsonl`); adapters go to
`<out>/lora/<run>/<trait>_<direction>/` (`adapter.safetensors`, `adapter.json`, `train.json`, `steps.jsonl`,
`rollouts.jsonl.gz`, `samples/step_N.md`); `evaluate` writes `<out>/lora/<run>/eval.json` and prints a table for
base, each adapter, `sum` (1, 1) and `half` (0.5, 0.5). The OpenRouter key is read from the env or
`PersVecFast/.env`. Without `--wandb-project` nothing goes to Weights & Biases.

Main knobs (`train --help`): `--steps 150 --questions-per-step 8 --g 8 --max-new-tokens 384 --lr 1e-4`,
KL budget `--eps-kl 0.03` (dual ascent, `--eta-kl`) or a fixed `--lambda-kl`, floors `--coh-floor 80
--task-floor 80`, `--direction pos|neg`.

## Accuracy protection

- `prepare --out O`: downloads GSM8K to `<out>/lora/data/` (internet; once).
- `reference --task supergpqa-pool|gsm8k-train`: base answers with the games' sampling (presence penalty 1.5)
  up to `--max-tokens`; finished answers with a parsed answer (`--correct-only`: correct ones) go to
  `<out>/lora/refs/<task>.jsonl`, the `train --kl-refs` input. KL is taken on the LAST `--kl-tokens` tokens
  of each answer, with the whole answer as context.
- `train` adds reasoning rollouts on GSM8K train (`--reason-questions N --reason-g G --reason-tokens T`) giving
  an on-policy KL(p_A || p_0) (`--lambda-onpolicy`) and a correctness reward (`--w-correct`); `--c-leak`
  penalises the judge's score of the other traits; checkpoints are kept in `<adapter>/ckpt/step_N/`.
- `accuracy --task supergpqa|gsm8k --cond SPEC...`: 500 questions (SuperGPQA: PersVecTrain's bench500, no
  overlap with the pool), games' sampling, `--max-tokens 4096`; SPEC is `base` or
  `adapter[@step_N][:scale]+...`. Writes per-question results; base is shared in `<out>/lora/_base/`.
- `gate --task supergpqa gsm8k --delta 0.03`: paired accuracy difference to base, one-sided 95% lower bound,
  PASS when it is above -delta.
- `evaluate --grid 0,0.5,1`: persona / coherence / KL over every pair of scales.
- `bash slurm/experiments.sh` submits the whole comparison (runs `kl`, `onpol`, `correct`, `leak` and the MVP's
  accuracy); `bash slurm/experiments.sh gates` prints the gates.

## Tests (no GPU, no network)

    pytest -q
