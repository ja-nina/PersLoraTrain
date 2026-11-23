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

## MVP first (the user's call, 2026-11-23: "build an MVP and see if this gets trained", two traits to check compositionality)

The full design below is the target; the MVP answers only "does judge-reward RL train a persona LoRA, and do two
of them add up?":
- **Two traits, `pos` adapters**: Assertive and Sarcastic (my call: both benign, the base shows each a little, so
  GRPO has signal without a warm start; clearly distinct, so a mix is easy to read). `--direction neg` works but is
  not part of the MVP run. DPO warm start: optional flag, off by default.
- **Questions**: the trait file's 40, PersVecTrain's 80/20 split (32 train / 8 held). Question generation later.
- **Own minimal LoRA** (my call, no `peft`): wraps the MLP linears; several named adapters with per-adapter scales,
  so `scale 0` = base, and composition = setting two scales at once (exactly the sum of the ΔW). Saved as
  safetensors + config json.
- **Own GRPO loop**, rollouts with HF `generate` (no vLLM / TRL: fewest moving parts; swap later if slow). One
  policy update per batch (on-policy), so the PPO ratio is 1 and the loss is REINFORCE with the group baseline.
- **Reward**: the one-call judge vector on persona rollouts (target, opposite, coherence, on_task, denial), group
  z-normalised per component, coherence / on_task floors by dual ascent (the coherence-adjacent reward).
- **KL budget**: science only (PersVecTrain's SuperGPQA reference answers, 160 KL-train), first 512 tokens. Math KL
  and the free reasoning-rollout rewards (format, loops, length) are v2: science answers run to thousands of tokens,
  so reasoning rollouts are the costliest part and not needed to see whether training works.
- **Eval (`evaluate`)**: for base, A, B, A+B, 0.5A+0.5B: the standard judge (expression of both traits, coherence)
  on the 8+8 held-out questions x 4 samples; teacher-forced KL on the 100 held-out pool questions; accuracy on the
  held-out pool by HF free generation (`--acc-n`, `--acc-tokens`, unparsed answers counted and reported).

## Goals

1. Per trait, **two adapters** (the user's call): `pos` (expresses the trait) and `neg` (expresses its opposite
   pole). 7 debate traits (Assertive, Sycophantic, Opportunistic, Empathy, Evil, Sarcastic, Agreeableness) x 2 = 14
   adapters on Qwen3.5-2B, non-thinking.
2. Training by **RL with a judge reward** (the user's call): rollouts, one judge call per rollout returning a
   **vector of scores**, GRPO on the LoRA; a short DPO warm start on the existing pairs.
3. Reasoning kept by a **KL budget** to the base model on science and math (the user's call: KL, no correctness
   reward), plus **coherence-adjacent rewards** on the reasoning rollouts (the user's call).
4. Evaluation that says per adapter whether it passes: persona moved (judge), coherence kept, reasoning
   **non-inferior** to the base on held-out SuperGPQA and math, with a pre-set margin.
5. Built so composition can be tested later without retraining: identical LoRA config and init seed for every
   adapter, PEFT format, every reward component saved per rollout.

## Non-goals (for now)

- Mixing / composing adapters, profile export, MA_Environments integration.
- A correctness reward on reasoning (it could *improve* reasoning and blur "persona added, reasoning unchanged",
  and make the comparison with the vectors unfair).
- Other models, thinking mode, the 27B. Changing PersVecFast or PersVecTrain.

## Anti-goals (a failure even if it runs)

- An adapter that keeps reasoning because it barely does anything: persona and reasoning are always reported side
  by side, with the adapter's weight-change norm and a norm-matched random-adapter control.
- **Reward hacking**: caricature, persona talk instead of an answer, gibberish the judge likes. Guarded by the
  coherence / on-task floors and checked by reading samples (below).
- Training on anything evaluated on: SuperGPQA bench500, the 100 held-out pool questions, records 0-199 and
  300-339, MATH-500, MMLU, the held-out persona questions.
- Results not traceable to config, seed, data hashes, judge model and the commits of all three repos.

## Constraints

- One GPU (L40 / A40) for policy + vLLM rollouts. HF transformers, `peft`, TRL's GRPO trainer with vLLM rollouts
  (builder's call after a smoke test that TRL + vLLM handle Qwen3.5 with LoRA; if not, our own GRPO loop on
  PersVecFast's engine with adapter reload per step).
- Judge: PersVecFast's OpenRouter client (`openai/gpt-oss-120b`), key from env or `PersVecFast/.env`, async with
  bounded concurrency; a judge outage pauses training (retry with backoff) rather than training on missing rewards.
- Tests run without a GPU or network: tiny Llama + char tokenizer (copied from PersVecTrain) and a fake judge.

## Data

- **Persona questions (new)**: per trait ~500 train + 100 held-out questions, written by a stronger model via
  OpenRouter (`scripts/make_questions.py`): spread over topics and situations, near-duplicates removed (embedding
  or n-gram similarity, builder's call), none copied from the trait file. The trait file's 40 questions are added
  to train except PersVecTrain's held-out ones (same 80/20 split, seed 0), which join the held-out set, so results
  stay comparable with the vector runs. Files `data/questions/{slug}.jsonl` with `split`.
- **Warm start pairs**: PersVecFast `{trait}_responses.json`, kept pairs, train split only.
- **Reasoning prompts, science**: PersVecTrain's 160 KL-train SuperGPQA pool questions + the base model's own
  answers (its `reference` output).
- **Reasoning prompts, math (new)**: 200 MATH **train** problems (disjoint from MATH-500, from the test split),
  stratified by level, seed 0, `data/math_kl200.jsonl`; 100 more as KL held-out. Base solutions generated once
  (`perslora reference`).
- **Eval**: SuperGPQA bench500 (copied from PersVecTrain), MATH-500 `data/math500.jsonl` (`\boxed{}` parser and
  normaliser, builder's call, tested).

## Training

### Stage 1: DPO warm start (short)
`L = -log sigmoid(beta * ([logp_A(y_w) - logp_0(y_w)] - [logp_A(y_l) - logp_0(y_l)]))`, y_w / y_l = D+ / D- for
`pos`, D- / D+ for `neg`, reference = base (adapter disabled), beta 0.1, plus the KL budget below. Purpose only to
get the trait into some rollouts (if the base never shows Evil, every group scores 0 and GRPO gets no signal).
Stop after a fixed small number of steps or once a probe of 32 rollouts has mean target score above ~30.

### Stage 2: GRPO
Per step: 16 persona questions x G = 8 rollouts (neutral system prompt, max 1024 tokens), plus a few reasoning
questions x G (science / math, for the reasoning rewards and the on-policy KL).

**Judge vector, one call per persona rollout** (JSON, 0-100 each):
`trait` (trait expression), `opposite` (the trait file's negative description), `coherence`, `on_task` (answers
the question rather than only performing the persona), and `denial` (yes / no: refuses, breaks character, talks
about the persona instead of having it).

**Persona reward**: target = `trait - opposite` for `pos`, `opposite - trait` for `neg` (both poles, so `neg`
cannot win by going bland). Each component is normalised **within its group of G separately** (mean / std), then
combined, so the highest-variance component does not drown the others:
`A = z(target) + lambda_coh * z(coherence) + lambda_task * z(on_task) - c_denial * denial`.

**Floors instead of tuned weights**: coherence >= 80 and on_task >= 80 (group means, EMA); `lambda_coh` /
`lambda_task` rise by dual ascent while below the floor and fall to 0 above it.

**Reasoning rollouts, coherence-adjacent rewards** (no correctness reward, no judge call: free):
- format: the answer ends with a parseable `ANSWER: <letter>` (science) or `\boxed{}` (math);
- no degeneration: penalty for repeated n-gram loops (fraction of repeated 4-grams above a threshold);
- length in the base's range: penalty when the length falls outside the base answers' [p5, p95] for that domain.
Combined as `A_r = z(format) + z(no_loop) + z(length_ok)` within each group.

**KL to the base, per domain, as a budget**: forward `KL(p_0 || p_A)` per token, teacher-forced on the base
model's own answers (first 1024 answer tokens), separately for science and math: `KL_d <= eps_d` with dual ascent
`lambda_d <- max(0, lambda_d + eta * (KL_d_ema - eps_d))`. Added to the GRPO loss every step. GRPO's own KL to the
reference on the rollouts stays at TRL's default small beta. Option, ablation: confidence-weighted KL (each token's
KL weighted by `1 - H(p_0)/log V`). eps from a sweep on 2 traits ({0.01, 0.03, 0.1} nats/token), then fixed.

**Reward-hacking checks**: every 25 steps 8 rollouts per adapter are written to `samples/step_N.md` to read; the
combined-judge-call check (below) runs once before the first full run.

### Judge validity check (once, before the full run)
On ~200 rollouts per trait, compare the one-call vector judge with PersVecFast's separate expression and coherence
calls (correlation, mean difference). If coherence leaks into trait (or back), switch to separate calls (~2x judge
cost) and record it here.

## LoRA configuration (my calls, changeable)

Rank 8, alpha 16, dropout 0, MLP projections (`gate_proj`, `up_proj`, `down_proj`) of all layers (present in both
Qwen3.5 layer types). Same init seed for every adapter. lr, steps (start: 200), G, batch: the builder tunes on the
first run and records the values here.

## Evaluation

Per adapter, against the base (`unsteered`), the base with the eliciting prompt (`elicited`), a norm-matched random
adapter (each module's ΔW scaled to the trained adapter's Frobenius norm; 3 seeds), the DPO-only warm start (what
RL added), and, with `--vector-run`, PersVecTrain's trained vector at c = +-1.

1. Teacher-forced, no API: KL per domain on the held-out KL prompts.
2. Persona: held-out persona questions, the same vector judge plus PersVecFast's standard expression / coherence
   judge (so numbers are comparable with the vector runs). `pos` under the neutral prompt (target score up vs
   unsteered); `neg` under the neutral prompt (opposite pole up) and under the eliciting prompt (trait suppressed vs
   elicited, PersVecFast's suppression setting).
3. Reasoning: vLLM free generation, model card sampling, paired seeds: SuperGPQA bench500 and MATH-500, accuracy,
   length, format rate.

**Pass rule** per adapter:
- persona: target score above unsteered (paired, one-sided, 1 SE); for `neg` also suppression under the eliciting
  prompt;
- coherence (standard judge): not below unsteered by more than 1 SE;
- **reasoning non-inferiority, per benchmark**: one-sided 95% lower bound of the paired accuracy difference
  (adapter - base) above `-delta` (default 0.03). The report prints the largest drop not ruled out (the bound), so a
  fail from too few questions is told apart from a real drop. `--samples K` answers each question K times.
- random adapter and DPO-only: reported, not gated.

## Output layout

`<out>/lora/<run>/<slug>_{pos,neg}/`: PEFT adapter, `train.json` (config, seeds, data hashes, judge model, commits),
`steps.jsonl` (per step: reward components mean / std, target score, lambdas, KL per domain, ΔW norm, judge
failures), `rollouts.jsonl.gz` (every rollout with its judge vector), `samples/`, `eval.json`; per run `report.md`.

## Interface

    python -m perslora questions --traits ...                     # write data/questions (OpenRouter)
    python -m perslora reference --model qwen3.5-2b --out O       # base solutions to the math KL prompts
    python -m perslora judgecheck --out O --traits ...            # one-call vs separate judge calls
    python -m perslora train --traits ... --directions pos neg --run-name R [--eps-sci --eps-math | --lambda-kl]
    python -m perslora eval --out O --run-name R [--vector-run PersVecTrain run]
    python -m perslora report --out O --runs R1 R2 ...

plus `slurm/plt.sbatch` (any command on a GPU node) and `slurm/sweep.sh` (the eps sweep).

## Approaches considered

Adapter shape:
- A. One adapter per trait trained at both signs: one adapter, negation trained in; -1 may come out weak.
- **B. Two adapters per trait, pos and neg (chosen by the user)**: each easier to train, both sides strong; twice
  the adapters, and a profile later picks pos or neg per trait.
- C. SFT on the trait answers: simplest, but no contrast and more reasoning drift.

Persona signal:
- DPO on pairs only: needs ~1-2k pairs per trait (a new extraction run); one bit per pair.
- Context distillation to the prompted model: dense, but copies refusals for traits the base refuses.
- **RL with a judge vector reward (chosen by the user)**: needs only questions (~600 per trait), on-policy, rewards
  exactly what is evaluated; costs judge calls (rough guess $50-150 for 14 adapters, to check against current
  OpenRouter prices) and needs reward-hacking guards.

Reasoning:
- **KL budget + coherence-adjacent rewards (chosen by the user)**: keeps the claim "persona added, reasoning
  unchanged".
- KL + correctness reward: stronger protection, but may improve reasoning and blur the claim.

## Left to the builder

lr, steps, G, batch, eta; TRL vs own GRPO loop; vLLM LoRA vs merged serving for eval (parity check either way);
question dedup method; the MATH normaliser; repetition threshold. Each call is written into this document.

## After the build

Code review by a larger model (Fable, the user's request) before the first cluster run.
