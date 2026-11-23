# Persona LoRA adapters (PersLoraTrain)

2026-11-23. Status: draft, waiting for the user's approval. Separate repo `PersLoraTrain` (the user's call), next to
PersVecFast and PersVecTrain; it reads their outputs and imports `persvecfast` and `persvectrain`.

## Why

The thesis gives LLM agents persona profiles (numbers per trait, e.g. `{assertive: 0.5, empathy: -0.3}`).
Steering vectors (PersVecFast, PersVecTrain) move the persona but cost reasoning accuracy. A LoRA adapter per trait
is the weight-space alternative: no prompt and no added vector at inference, and adapters can later be summed into a
profile (rewarded / personalized soups, LoRA arithmetic). This repo answers the first question: **can a single
trait adapter show the trait, in both directions, without losing reasoning?** Composition comes after, once the
singles work (the user's call: "train singles and then we'll see").

## Goals

1. Per trait, **two adapters** (the user's call): `pos` (expresses the trait) and `neg` (expresses its opposite).
   7 debate traits x 2 = 14 adapters on Qwen3.5-2B (non-thinking).
2. Training = persona preference loss + a **KL budget** to the base model on reasoning prompts (science and math),
   so reasoning drift is capped during training, not just measured after.
3. Evaluation that says per adapter whether it passes: persona moved (judge), coherence kept, and reasoning
   **non-inferior** to the base model on held-out SuperGPQA and math, with a pre-set margin.
4. Built so composition can be tested later without retraining: identical LoRA config (rank, alpha, target
   modules, layers) and init seed for every adapter, saved in standard PEFT format.

## Non-goals (for now)

- Mixing / composing adapters, profile export, MA_Environments integration.
- Other models, thinking mode, the 27B.
- Changing PersVecFast or PersVecTrain (read and import only).

## Anti-goals (a failure even if it runs)

- An adapter that "keeps reasoning" because it barely does anything: every result reports persona and reasoning
  side by side, plus the adapter's weight-change norm and a norm-matched random-adapter control.
- Training on anything evaluated on: SuperGPQA bench500, the 100 held-out pool questions, SuperGPQA records 0-199
  and 300-339 (games, careful-reasoning), MATH-500 (math eval), MMLU (PersVecFast capability stage).
- Results not traceable to their exact config, seed, data files and the commits of all three repos.

## Constraints

- HF transformers + torch + `peft` for training, one GPU (L40 / A40). vLLM for free generation (eval).
- Tests run without a GPU, on PersVecTrain's tiny Llama + char tokenizer fixtures (copied, not imported from tests).
- Same slurm conventions as PersVecTrain (`slurm/lib.sh`-style env setup, `sbatch slurm/plt.sbatch <command>`).

## Data

- **Persona pairs**: PersVecFast `out/persona/qwen3.5-2b/bf16/{trait}_responses.json`, kept pairs only, the
  question **without** system prompt, answers cut to 512 tokens. **Same 80/20 split as PersVecTrain**
  (`persvectrain.data.split`, seed 0), so held-out questions match the vector runs and results are comparable.
  `pos` prefers D+ over D-, `neg` prefers D- over D+.
- **KL prompts, science**: PersVecTrain's 160 KL-train questions of `supergpqa_pool.jsonl` with the base model's own
  answers (`<model>_pool260*.jsonl`, read from PersVecTrain's `reference` output).
- **KL prompts, math (new)**: 200 problems from the MATH **train** split (disjoint from MATH-500, which is drawn
  from the test split), stratified by level, seed 0, in `data/math_kl200.jsonl` (`scripts/make_math.py`). The base
  model's own solutions are generated once (`plt reference`, model card sampling, non-thinking), as PersVecTrain does.
- **Eval**: SuperGPQA `supergpqa_bench500.jsonl` (copied from PersVecTrain) and MATH-500 (`data/math500.jsonl`,
  answers parsed from `\boxed{}` with a normaliser; the builder picks the normaliser and tests it).

## Losses

Reference model = the base model (adapter disabled, `peft`'s `disable_adapter()`): one model in memory.

- **Persona (DPO form, = BiPO with the adapter in place of the vector)**:
  `L_p = -log sigmoid(beta * ([logp_A(y_w) - logp_0(y_w)] - [logp_A(y_l) - logp_0(y_l)]))`,
  y_w / y_l = D+ / D- for `pos`, D- / D+ for `neg`. Summed answer log-probs (BiPO's choice), beta 0.1.
- **KL to the base on reasoning prompts**:
  - direction: forward `KL(p_0 || p_A)` per answer token, teacher-forced on the **base model's own answers**. It
    penalises the adapter for taking probability away from what the base would write, step by step through the
    reasoning, and needs no sampling.
  - window: the first **1024** answer tokens (longer than PersVecTrain's 512: math chains go long); a depth check as
    PersVecTrain's `kldepth` on the eval answers reports whether drift past the window stays small.
  - **two budgets, one per domain**: `KL_sci <= eps_sci`, `KL_math <= eps_math` (mean KL per token, EMA over steps),
    so a quiet domain cannot hide drift in the other.
  - enforced by **dual ascent**: per domain `lambda_d <- max(0, lambda_d + eta * (KL_d_ema - eps_d))` after every
    step, total `L = L_p + lambda_sci * KL_sci + lambda_math * KL_math`. A fixed-lambda mode (`--lambda-kl`, eps
    ignored) stays for ablation (0 = plain DPO: "is the KL what helps?").
  - option, ablation only: **confidence-weighted KL** (`--kl-weight entropy`): each token's KL weighted by
    `1 - H(p_0)/log V`, so tokens where the base is sure (arithmetic, the answer letter) are protected most and
    high-entropy wording, where a persona naturally shows, is freer.
- eps defaults: set from the first run (sweep eps in {0.01, 0.03, 0.1} nats/token on 2 traits), then fixed for all 14.

## LoRA configuration (my calls, changeable)

Rank 8, alpha 16, dropout 0, on the MLP projections (`gate_proj`, `up_proj`, `down_proj`) of **all** layers: these
exist in both Qwen3.5 layer types (Gated DeltaNet and full attention), so no per-layer-type special cases. Same init
seed for every adapter. lr 1e-4, cosine schedule, warmup 10%, epochs as PersVecTrain (builder tunes on the first run).

## Evaluation

Per adapter, against the base model (`unsteered`), the base with the eliciting prompt (`elicited`), a
**norm-matched random adapter** (random A, B per module, each module's ΔW scaled to the trained adapter's Frobenius
norm; 3 seeds), and, when a PersVecTrain run is given (`--vector-run`), its trained vector at c = +-1.

1. **Teacher-forced (held-out, no API)**: persona margin on the held-out pairs; KL per domain on held-out KL prompts
   (100 SuperGPQA pool held-out, 100 more MATH train problems kept out of the KL set).
2. **Persona judge** (PersVecFast's OpenRouter judge, as PersVecTrain `persona`): held persona questions; `pos`
   under the neutral prompt (expression should rise vs unsteered), `neg` under the **eliciting** prompt
   (suppression: expression should fall vs elicited), coherence 0-100 for every kind.
3. **Reasoning** (vLLM free generation, model card sampling, paired seeds): SuperGPQA bench500 and MATH-500,
   accuracy and answer length.

**Pass rule** per adapter:
- persona: `pos` judge expression > unsteered, `neg` judge expression < elicited (paired, one-sided, 1 SE);
- coherence: not below unsteered by more than 1 SE;
- **reasoning non-inferiority, per benchmark**: the one-sided 95% lower bound of the paired accuracy difference
  (adapter - base) is above `-delta` (default delta = 0.03). The report also prints the **largest drop not ruled
  out** (the bound itself), so a fail from too few questions is told apart from a real drop. Power: with 500
  questions the paired SE is about 0.015-0.02, so delta = 0.03 is near the edge; `--samples K` answers each
  question K times (default 1) to tighten it.
- the random-adapter control is reported, not gated.

## Output layout

`<out>/lora/<run>/<slug>_{pos,neg}/` with the PEFT adapter (`adapter_model.safetensors`, `adapter_config.json`),
`train.json` (config, seed, data file hashes, commits of all three repos), `steps.jsonl` (per step: persona loss,
margin, KL per domain, lambdas, ΔW norm), `eval.json` (sections 1-3 + pass), and per run `report.md` / a printed
table over all adapters.

Serving for eval: vLLM's LoRA support if it loads Qwen3.5 adapters on the MLP modules (smoke test first); otherwise
merge the adapter into a temporary full checkpoint on node-local disk and serve that. The builder decides after the
smoke test; results must not depend on which path (parity check on a few prompts).

## Interface

    python -m perslora check    --source <PersVecFast out>                    # pairs per trait, data files
    python -m perslora reference --model qwen3.5-2b --out O                   # base solutions to the math KL set
    python -m perslora train    --model qwen3.5-2b --traits ... --directions pos neg --run-name R
                                [--eps-sci 0.03 --eps-math 0.03 | --lambda-kl X] [--kl-weight entropy]
    python -m perslora eval     --out O --run-name R [--vector-run PersVecTrain run]   # sections 1-3
    python -m perslora report   --out O --runs R1 R2 ...

plus `slurm/plt.sbatch` (any command on a GPU node) and `slurm/sweep.sh` (the eps sweep).

## Approaches considered

- **A. One adapter per trait trained at both signs (+1 / -1 scaling)**: one adapter, negation trained in, closest to
  the vector setup; risk that -1 comes out weak.
- **B. Two adapters per trait, pos and neg (chosen by the user)**: each trained in one direction, easier to train,
  the negative side as strong as the positive; twice the adapters, and a later profile picks pos or neg per trait
  instead of flipping a sign.
- **C. SFT on the trait answers**: simplest, but no contrast (learns style and topic too) and more reasoning drift.

## Left to the builder

Epochs, lr and eta (dual step size) after the first run; the MATH answer normaliser; vLLM LoRA vs merged serving;
batching. Each such call is written into this document when made.

## After the build

Code review by a larger model (Fable, the user's request) before the first cluster run.
