# Persona LoRA MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train `pos` persona LoRA adapters (Assertive, Sarcastic) on Qwen3.5-2B by GRPO with a judge score-vector
reward and a science KL budget, then evaluate each adapter and their sums (persona, coherence, KL, accuracy).

**Architecture:** Package `perslora`: own minimal LoRA (named adapters, per-adapter scales), HF-generate rollouts,
an async one-call vector judge on PersVecFast's OpenRouter client, a GRPO loop with dual-ascent floors and a KL
budget, an `evaluate` command over adapter mixes. Reuses `persvecfast` (models, prompts, judge client) and
`persvectrain` (pairs, split, reference answers, KL loss, encode / answer_logprobs, answer parsing).

**Tech Stack:** Python >= 3.9, torch, transformers, safetensors, httpx (via persvecfast), pytest.

**Spec:** `docs/superpowers/specs/2026-11-23-persona-lora-design.md` (section "MVP first").

## Global Constraints

- Imports from `persvecfast` / `persvectrain` only; never modify those repos.
- Prompt = `chat_prompt(tok, NEUTRAL_SYSTEM_PROMPT, question, MODELS[model].enable_thinking)` (thinking off).
- Persona questions: trait file questions via `persvectrain.data.load_pairs` + `split(seed=0, frac=0.8)`; train only.
- KL reference: `persvectrain.cli.reference_path(out, model)` + `split_reference(..., seed=0)` (160 train / 100 held).
- Target modules default `("gate_proj", "up_proj", "down_proj")`, every layer; rank 8, alpha 16; init seed 0 for
  every adapter (A ~ kaiming-uniform from `torch.Generator().manual_seed(seed)`, B = 0).
- Judge model `openai/gpt-oss-120b`, temperature 0; judge outage → retry (client does), still-missing score → that
  rollout gets advantage 0 and is counted in `judge_missing`.
- Tests: no GPU, no network: tiny Llama + char tokenizer fixtures (copied from PersVecTrain `tests/conftest.py`),
  `FakeJudge`.
- Output: `<out>/lora/<run>/<adapter>/` = `adapter.safetensors`, `adapter.json`, `train.json`, `steps.jsonl`,
  `rollouts.jsonl.gz`, `samples/step_<N>.md`; `<out>/lora/<run>/eval.json`.

## Review Focus

1. A group where every rollout has the same score (std 0) → all advantages 0, no NaN (test in Task 3).
2. Judge returns malformed JSON / out-of-range numbers / missing keys → `None`, never a crash or a clipped guess (Task 3).
3. Scales: `scale 0` must reproduce the base model's logits exactly, and two adapters at scales (1, 1) must equal one
   adapter whose ΔW is the sum (Task 1).
4. Empty generation (EOS first) → rollout kept, zero answer tokens, no division by zero in the token mean (Task 2).
5. Resuming / rerunning into an existing adapter dir → refuse unless `--overwrite` (Task 5).

---

### Task 1: Repo scaffold + minimal LoRA

**Files:**
- Create: `pyproject.toml` (name `perslora`, deps torch, transformers, safetensors; test extra pytest; pytest
  `pythonpath = ["src", "."]`), `README.md`, `.gitignore`, `src/perslora/__init__.py`, `src/perslora/lora.py`,
  `tests/conftest.py` (tiny model + CharTokenizer copied), `tests/test_lora.py`

**Interfaces:**
- Produces:
  - `class LoRALinear(nn.Module)`: wraps a frozen `nn.Linear`; `adapters: nn.ParameterDict` of `{name}_A [r, in]`,
    `{name}_B [out, r]`; `scales: dict[str, float]`; forward = base(x) + Σ scale·(alpha/r)·(x A^T B^T).
  - `attach(model, name: str, rank=8, alpha=16, targets=("gate_proj","up_proj","down_proj"), seed=0) -> list[str]`
    (module paths wrapped; raises ValueError if none match; re-wraps nothing already wrapped, adds the adapter).
  - `set_scales(model, scales: dict[str, float]) -> None` (adapters not named → 0).
  - `adapter_params(model, name) -> list[nn.Parameter]`
  - `save_adapter(model, name, dir: Path, meta: dict) -> None`; `load_adapter(model, name, dir: Path) -> dict` (meta)
  - `delta_norm(model, name) -> float` (sqrt of Σ over modules ‖(alpha/r) B A‖_F²)

- [ ] **Step 1: tests** `test_scale_zero_is_base` (logits equal to an unwrapped copy, `torch.equal`),
  `test_new_adapter_is_identity` (B = 0 → logits equal base at scale 1), `test_two_adapters_sum` (random A,B set;
  scales (1,1) logits == single adapter with B·A summed, `allclose atol 1e-5`), `test_save_load_roundtrip`,
  `test_attach_no_match_raises`, `test_same_seed_same_init` (two models, same A).
- [ ] **Step 2:** run `pytest tests/test_lora.py -q` → fails (module missing).
- [ ] **Step 3:** implement `lora.py`.
- [ ] **Step 4:** run → pass.
- [ ] **Step 5:** commit "LoRA: named adapters with scales, save/load".

### Task 2: Rollouts and sequence log-probs

**Files:** Create `src/perslora/rollout.py`, `tests/test_rollout.py`

**Interfaces:**
- Produces:
  - `@dataclass Rollout: qid: str; question: str; prompt_ids: list[int]; answer_ids: list[int]; text: str`
  - `generate(model, tok, items: list[tuple[str, str]], prompt_fn, g: int, max_new_tokens: int, sampling: dict,
    seed: int) -> list[Rollout]` — items (qid, question); `prompt_fn(question) -> str`; g rollouts per item, grouped
    in item order; left-padded batch `model.generate(do_sample=True, temperature, top_p, top_k)`; answer stops at
    the first EOS; `torch.manual_seed(seed)` before generating.
  - `seq_logprob(model, r: Rollout) -> Tensor` scalar with grad: Σ log p(answer tokens); 0-length answer → `0.0`
    tensor that requires no grad.
  - `sampling_from_spec(model_key) -> dict` (temperature, top_p, top_k from `persvecfast.models.MODELS`).
- [ ] **Step 1: tests** `test_generate_groups` (2 items, g=3 → 6 rollouts, qids `[a,a,a,b,b,b]`, each
  `len(answer_ids) <= max_new_tokens`), `test_seq_logprob_matches_manual` (equals summing `answer_logprobs`
  gathered, tiny model), `test_empty_answer_zero`.
  CharTokenizer needs `pad_token_id`, `eos_token_id = 0`, and `batch_decode` / `decode`; tiny model generate with
  `pad_token_id=0`.
- [ ] **Step 2-4:** fail → implement → pass.
- [ ] **Step 5:** commit "Rollouts: grouped HF sampling, sequence log-probs".

### Task 3: Vector judge and group advantages

**Files:** Create `src/perslora/reward.py`, `tests/test_reward.py`

**Interfaces:**
- Produces:
  - `VECTOR_TEMPLATE: str` — asks for one JSON object
    `{"trait": int, "opposite": int, "coherence": int, "on_task": int, "denial": true|false}` given trait name,
    trait description (trait file `description`), opposite description (trait file `instruction["negative"]`),
    question, response; definitions: on_task = answers the question rather than only performing a persona;
    denial = refuses, breaks character, or talks about the persona instead of having it. "Return only the JSON."
  - `KEYS = ("trait", "opposite", "coherence", "on_task", "denial")`
  - `parse_vector(raw: str | None) -> dict | None` (first `{...}` block; all keys present; ints in 0..100; denial bool
    (accept "yes"/"no" strings); else None)
  - `async judge_vectors(judges: persvecfast.judge.Judges, rollouts, trait: dict) -> list[dict | None]`
    (uses `judges.client.complete`, `clip(text, judges.max_chars)`, one retry on None)
  - `@dataclass Floor: target: float; eta: float; lam: float = 0.0; ema: float | None = None; decay: float = 0.9`
    with `update(value: float) -> float` (ema, then `lam = max(0, lam + eta*(target - ema))`, returns lam)
  - `@dataclass Budget` same fields with `eps` instead of target: `lam = max(0, lam + eta*(ema - eps))`.
  - `group_advantages(vectors: list[dict | None], g: int, direction: str, lam_coh: float, lam_task: float,
    c_denial: float) -> list[float]` — per group of g: target = trait-opposite (pos) / opposite-trait (neg); z per
    component over the group's non-None members (std < 1e-6 → 0); denial term = denial − group mean;
    `A = z(target) + lam_coh*z(coh) + lam_task*z(task) − c_denial*denial_c`; None → 0.
- [ ] **Step 1: tests** `test_parse_vector_ok`, `test_parse_vector_rejects` (missing key, 101, non-JSON, None →
  None), `test_advantages_constant_group_zero`, `test_advantages_direction_flips`,
  `test_advantages_none_is_zero_and_excluded` (group stats from the others), `test_floor_rises_below_target`,
  `test_budget_falls_to_zero_under_eps`, `test_judge_vectors_fake_client` (asyncio.run with a fake client
  returning canned JSON).
- [ ] **Step 2-4:** fail → implement → pass.
- [ ] **Step 5:** commit "Reward: one-call judge vector, group-normalised advantages, dual floors".

### Task 4: GRPO loop

**Files:** Create `src/perslora/grpo.py`, `tests/test_grpo.py`

**Interfaces:**
- Consumes: Tasks 1-3; `persvectrain.steering.encode`, `answer_logprobs`; `persvectrain.losses.kl_loss`.
- Produces:
  - `@dataclass GRPOConfig: model: str; trait: str (slug); direction: str = "pos"; steps: int = 150;
    questions_per_step: int = 8; g: int = 8; max_new_tokens: int = 384; lr: float = 1e-4; rank: int = 8;
    alpha: float = 16; seed: int = 0; eps_kl: float = 0.03; eta_kl: float = 1.0; kl_batch: int = 2;
    kl_tokens: int = 512; lambda_kl: float | None = None (fixed, disables the budget); coh_floor: float = 80;
    task_floor: float = 80; eta_floor: float = 0.05; c_denial: float = 1.0; sample_every: int = 25`
  - `train_adapter(model, tok, cfg, questions: list[tuple[str,str]], kl_refs: list[RefAnswer], judge_fn,
    trait: dict, log_step: Callable[[dict], None], on_samples: Callable[[int, list], None]) -> None` —
    `judge_fn(rollouts) -> list[dict|None]` (sync wrapper; the CLI passes asyncio around `judge_vectors`).
  - Step: draw `questions_per_step` questions (seeded cycle over a shuffled list) → `generate` (adapter scale 1,
    `torch.no_grad`, model.eval) → judge → advantages → policy loss per rollout
    `−A_i * seq_logprob_i / max_new_tokens` (constant normaliser: no length bias), backward per rollout scaled
    `1/len(rollouts)` → KL: `kl_batch` refs teacher-forced, adapter scale 0 under no_grad for p_0 and scale 1 for
    p_A, `lam_kl * kl / kl_batch` backward → AdamW step (only the adapter params, grad clip 1.0) → update floors
    with the group-mean coherence / on_task and the budget with the step KL.
  - `log_step` entry keys: `step, reward_target_mean, trait_mean, opposite_mean, coherence_mean, on_task_mean,
    denial_rate, judge_missing, kl, lam_kl, lam_coh, lam_task, delta_norm, answer_tokens_mean, t`.
- [ ] **Step 1: tests** (tiny model, fake judge that scores `trait` = count of the letter "a" in the text):
  `test_train_runs_and_logs` (3 steps, 3 log entries with all keys, finite), `test_reward_goes_up`
  (40 steps, lr 5e-3, mean trait of last 5 steps > first 5; mark it with a fixed seed),
  `test_fixed_lambda_kl_skips_budget` (lam_kl constant), `test_only_adapter_params_change` (base weights equal).
- [ ] **Step 2-4:** fail → implement → pass.
- [ ] **Step 5:** commit "GRPO loop: judge-vector advantages, KL budget, floors".

### Task 5: CLI `train` + slurm

**Files:** Create `src/perslora/data.py`, `src/perslora/cli.py`, `src/perslora/__main__.py`, `slurm/lib.sh`,
`slurm/plt.sbatch`, `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `data.trait_questions(src: Path, slug: str) -> tuple[list[tuple[str,str]], list[tuple[str,str]]]` (train, held;
    qid `"{slug}:{index}"`) and `data.trait_info(slug) -> dict` (PersVecFast trait file, as persvectrain.persona).
  - `python -m perslora train --model qwen3.5-2b --source <PersVecFast persona dir root> --out O --run-name R
    --traits Assertive Sarcastic [--direction pos] [GRPOConfig flags] [--overwrite]`: per trait, a fresh model load is
    not needed: attach adapter `"{slug}_{direction}"`, train, save, `set_scales({})` before the next trait. Writes
    `train.json` (config, sources, commits of the three repos via `persvectrain.output.git_commit`, start / end time).
  - `slurm/plt.sbatch <args>` runs `python -m perslora <args>` (env setup as PersVecTrain `lib.sh`, GPU 1, 20 h).
- [ ] **Step 1: tests** `test_parser_defaults`, `test_existing_adapter_dir_refuses` (dir exists, no `--overwrite`
  → SystemExit), `test_train_cli_end_to_end` (monkeypatch model loading → tiny model + char tokenizer, judge →
  fake, data → 3 questions; 2 steps; files exist).
- [ ] **Step 2-4:** fail → implement → pass.
- [ ] **Step 5:** commit "CLI train, slurm wrapper".

### Task 6: `evaluate` over adapters and mixes

**Files:** Create `src/perslora/evaluate.py`, `tests/test_evaluate.py`; Modify `src/perslora/cli.py`

**Interfaces:**
- Produces:
  - `conditions(adapters: list[str]) -> dict[str, dict[str, float]]`: `base` {}, each adapter at 1, and for two
    adapters `sum` (1, 1) and `half` (0.5, 0.5).
  - `persona_eval(model, tok, cfg, cond, held_by_trait, traits, samples=4, judges) -> dict`: per condition per trait:
    standard `score_expression` of **every** trait and `score_coherence` (PersVecFast judge), mean / se.
  - `kl_eval(model, tok, cond, refs_held, kl_tokens) -> dict`: mean KL(base || cond) per condition.
  - `acc_eval(model, tok, cond, pool_held, n, max_new_tokens) -> dict`: per condition accuracy, unparsed rate,
    paired difference vs base (mean, se); prompt = question + `persvectrain.freegen.instruction(n_options)`,
    parsed by `persvectrain.freegen.parse_answer`; same seed per question across conditions.
  - `python -m perslora evaluate --out O --run-name R [--samples 4] [--acc-n 100] [--acc-tokens 2048]` loads all
    adapters of the run, writes `eval.json` and prints one table.
- [ ] **Step 1: tests** `test_conditions_two_adapters` (5 conditions, scales as above), `test_kl_eval_base_zero`,
  `test_acc_eval_paired` (fake generations: base always right, adapter always wrong → diff −1),
  `test_persona_eval_fake_judges`.
- [ ] **Step 2-4:** fail → implement → pass.
- [ ] **Step 5:** commit "evaluate: persona, KL and accuracy for adapters and their sums".

### Task 7: Review handoff

- [ ] Full test run `pytest -q` → all pass; README lists commands and the cluster run:
  `sbatch slurm/plt.sbatch train ...` then `sbatch slurm/plt.sbatch evaluate ...`.
- [ ] Whole-repo code review by a Fable-model reviewer (user's request); fix confirmed findings; commit.
