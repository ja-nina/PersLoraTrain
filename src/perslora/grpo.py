"""GRPO on one persona LoRA adapter: judge-vector advantages, coherence / on-task floors, KL budget on science.

One policy update per batch of rollouts (on-policy), so the PPO ratio is 1 and the policy loss is REINFORCE with
the group baseline: -A_i * log p(answer_i) / max_new_tokens (a constant normaliser, no length bias). The KL term is
forward KL(p_0 || p_A) per token, teacher-forced on the base model's own SuperGPQA answers, weighted by a
dual-ascent lambda that keeps its EMA under eps_kl (or a fixed lambda_kl). The KL window is the LAST kl_tokens
tokens of each reference answer (the whole answer is the context), so long answers are scored where they settle
on the final answer.

Optional reasoning rollouts (reason_questions > 0): each step the adapter answers reason_questions questions of a
task with a checkable answer (GSM8K), reason_g times each. They give (1) an on-policy KL(p_A || p_0) on the
adapter's own reasoning text, weight lambda_onpolicy, which teacher-forced KL on base text cannot see, and
(2) a correctness reward, weight w_correct, group-normalised like the persona reward.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from persvectrain.losses import kl_loss
from persvecfast.prompts import NEUTRAL_SYSTEM_PROMPT, chat_prompt
from persvectrain.steering import answer_logprobs

from perslora import lora
from perslora.reward import Budget, Floor, _z, group_advantages, leak_score, target_score
from perslora.rollout import Rollout, generate, seq_logprob

LOG_KEYS = ("step", "reward_target_mean", "trait_mean", "opposite_mean", "coherence_mean", "on_task_mean",
            "denial_rate", "judge_missing", "leak_mean", "kl", "lam_kl", "lam_coh", "lam_task", "delta_norm",
            "answer_tokens_mean", "reason_acc", "reason_cut", "kl_onpolicy", "t")


@dataclass
class GRPOConfig:
    model: str
    trait: str  # slug
    direction: str = "pos"
    steps: int = 150
    questions_per_step: int = 8
    g: int = 8
    max_new_tokens: int = 384
    lr: float = 1e-4
    rank: int = 8
    alpha: float = 16
    seed: int = 0
    eps_kl: float = 0.03
    eta_kl: float = 1.0
    lam_kl_init: float = 1.0
    kl_batch: int = 2
    kl_tokens: int = 512
    lambda_kl: Optional[float] = None  # fixed weight; disables the budget
    coh_floor: float = 80.0
    task_floor: float = 80.0
    eta_floor: float = 0.05
    c_denial: float = 1.0
    sample_every: int = 25
    save_every: int = 25
    lam_floor_max: float = 2.0
    gen_batch: int = 64
    c_leak: float = 0.0  # weight of z(mean judge score of the other traits); the CLI judges them when > 0
    reason_questions: int = 0  # reasoning questions per step (0: no reasoning rollouts)
    reason_g: int = 4
    reason_tokens: int = 512
    w_correct: float = 0.0
    lambda_onpolicy: float = 0.0
    onpolicy_batch: int = 8  # reasoning rollouts per step that get the on-policy KL
    eval_every: int = 0  # accuracy on fixed held-out questions at step 0 and every eval_every steps (0: off)
    eval_tokens: int = 2048

    @property
    def adapter(self) -> str:
        return f"{self.trait}_{self.direction}"


def _question_stream(questions: list, seed: int):
    rng = random.Random(seed)
    while True:
        order = list(questions)
        rng.shuffle(order)
        yield from order


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def encode_tail(tok, question: str, answer: str, window: int) -> tuple:
    """(ids [1, T], start): the chat prompt and the whole answer; ids[0, start:] are the answer's last `window`
    tokens (all of it when shorter)."""
    prompt_ids = tok(chat_prompt(tok, NEUTRAL_SYSTEM_PROMPT, question, False), add_special_tokens=False)["input_ids"]
    answer_ids = tok(answer, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise ValueError("answer has 0 tokens")
    ids = prompt_ids + answer_ids
    return torch.tensor([ids]), len(ids) - min(window, len(answer_ids))


def _onpolicy_kl(model, name: str, r: Rollout, scale: float) -> float:
    """KL(p_A || p_0) per token on the adapter's own answer r (temperature 1); backward with weight `scale`."""
    ids = torch.tensor([r.prompt_ids + r.answer_ids])
    start = len(r.prompt_ids)
    lora.set_scales(model, {})
    with torch.no_grad():
        lp0 = answer_logprobs(model, ids, start)
    lora.set_scales(model, {name: 1.0})
    lpa = answer_logprobs(model, ids, start)
    kl = (lpa.exp() * (lpa - lp0)).sum(-1).mean()
    if scale:
        (kl * scale).backward()
    return float(kl.detach())


def _kl_step(model, name: str, enc: tuple, scale: float) -> float:
    ids, start = enc
    lora.set_scales(model, {})
    with torch.no_grad():
        lp0 = answer_logprobs(model, ids, start)
    lora.set_scales(model, {name: 1.0})
    kl = kl_loss(lp0, answer_logprobs(model, ids, start))
    if scale:
        (kl * scale).backward()
    return float(kl.detach())


def train_adapter(model, tok, cfg: GRPOConfig, questions: list[tuple[str, str]], kl_refs: list,
                  judge_fn: Callable[[list[Rollout]], list], log_step: Callable[[dict], None],
                  on_samples: Callable[[int, list], None], on_rollouts: Callable[[int, list], None],
                  sampling: dict, prompt_fn: Callable[[str], str],
                  on_checkpoint: Optional[Callable[[int], None]] = None, reason=None,
                  evals: Optional[dict] = None, eval_sampling: Optional[dict] = None,
                  on_eval: Optional[Callable[[int, dict], None]] = None) -> None:
    """Attach adapter `cfg.adapter` (if missing) and train it for cfg.steps; all scales are 0 on return.

    `reason`: a tasks.Task (items + score) for the reasoning rollouts; needed when cfg.reason_questions > 0.
    `evals` {name: Task}: with cfg.eval_every, answered (eval_sampling, one fixed seed, so steps compare) at step 0
    and every eval_every steps; on_eval(step, {name: accuracy.score_answers result}) gets the results."""
    if cfg.reason_questions and reason is None:
        raise ValueError("reason_questions > 0 needs a reasoning task")
    name = cfg.adapter
    if name not in lora.adapter_names(model):
        lora.attach(model, name, rank=cfg.rank, alpha=cfg.alpha, seed=0)
    params = lora.adapter_params(model, name)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    stream = _question_stream(questions, cfg.seed)
    kl_enc = [encode_tail(tok, r.question, r.answer_text, cfg.kl_tokens) for r in kl_refs]
    random.Random(cfg.seed).shuffle(kl_enc)  # mixed sources (science, math) interleave
    reason_stream = _question_stream(reason.items, cfg.seed + 1) if cfg.reason_questions else None
    kl_cursor = 0
    fixed = cfg.lambda_kl is not None
    budget = Budget(eps=cfg.eps_kl, eta=cfg.eta_kl, lam=cfg.lambda_kl if fixed else cfg.lam_kl_init)
    coh = Floor(cfg.coh_floor, cfg.eta_floor, lam_max=cfg.lam_floor_max)
    task = Floor(cfg.task_floor, cfg.eta_floor, lam_max=cfg.lam_floor_max)
    temperature = float(sampling.get("temperature", 1.0))
    t0 = time.time()
    model.eval()

    def run_evals(step: int) -> None:
        from perslora.accuracy import score_answers
        if not (cfg.eval_every and evals and on_eval):
            return
        lora.set_scales(model, {name: 1.0})
        res = {}
        for tname, t in evals.items():
            rs = generate(model, tok, t.items, prompt_fn, 1, cfg.eval_tokens, eval_sampling or sampling,
                          seed=12345, batch=cfg.gen_batch)
            res[tname] = score_answers(t, rs)
        on_eval(step, res)

    try:
        run_evals(0)
        for step in range(1, cfg.steps + 1):
            items = [next(stream) for _ in range(cfg.questions_per_step)]
            lora.set_scales(model, {name: 1.0})
            rollouts = generate(model, tok, items, prompt_fn, cfg.g, cfg.max_new_tokens, sampling,
                                seed=cfg.seed * 100003 + step, batch=cfg.gen_batch)
            vectors = judge_fn(rollouts)
            adv = group_advantages(vectors, cfg.g, cfg.direction, coh.lam, task.lam, cfg.c_denial, cfg.c_leak)
            reason_rs: list[Rollout] = []
            if reason_stream is not None:
                r_items = [next(reason_stream) for _ in range(cfg.reason_questions)]
                reason_rs = generate(model, tok, r_items, prompt_fn, cfg.reason_g, cfg.reason_tokens, sampling,
                                     seed=cfg.seed * 100003 + step + 7919, batch=cfg.gen_batch)
            correct = [float(reason.score(r.qid, r.text)[1]) for r in reason_rs]

            opt.zero_grad(set_to_none=True)
            lora.set_scales(model, {name: 1.0})
            n = len(rollouts)
            for r, a in zip(rollouts, adv):
                if a == 0.0 or not r.answer_ids:
                    continue
                (-a * seq_logprob(model, r, temperature) / (cfg.max_new_tokens * n)).backward()
            if cfg.w_correct and reason_rs:
                nr = len(reason_rs)
                for i in range(0, nr, cfg.reason_g):
                    for r, a in zip(reason_rs[i:i + cfg.reason_g], _z(correct[i:i + cfg.reason_g])):
                        if a == 0.0 or not r.answer_ids:
                            continue
                        (-cfg.w_correct * a * seq_logprob(model, r, temperature)
                         / (cfg.reason_tokens * nr)).backward()
            kl_on = []
            if reason_rs and cfg.lambda_onpolicy:
                picks = [r for r in reason_rs if r.answer_ids][:cfg.onpolicy_batch]
                for r in picks:
                    kl_on.append(_onpolicy_kl(model, name, r, cfg.lambda_onpolicy / len(picks)))
            lam_kl = budget.lam
            kls = []
            for _ in range(cfg.kl_batch if kl_enc else 0):
                kls.append(_kl_step(model, name, kl_enc[kl_cursor % len(kl_enc)], lam_kl / cfg.kl_batch))
                kl_cursor += 1
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            present = [v for v in vectors if v is not None]
            kl = _mean(kls) if kls else 0.0
            if not fixed and kls:
                budget.update(kl)
            if present:
                coh.update(_mean([v["coherence"] for v in present]))
                task.update(_mean([v["on_task"] for v in present]))
            entry = {
                "step": step,
                "reward_target_mean": _mean([target_score(v, cfg.direction) for v in present]),
                "trait_mean": _mean([v["trait"] for v in present]),
                "opposite_mean": _mean([v["opposite"] for v in present]),
                "coherence_mean": _mean([v["coherence"] for v in present]),
                "on_task_mean": _mean([v["on_task"] for v in present]),
                "denial_rate": _mean([float(v["denial"]) for v in present]),
                "judge_missing": n - len(present),
                "leak_mean": _mean([leak_score(v) for v in present]) if cfg.c_leak else None,
                "kl": kl, "lam_kl": lam_kl, "lam_coh": coh.lam, "lam_task": task.lam,
                "delta_norm": lora.delta_norm(model, name),
                "answer_tokens_mean": _mean([len(r.answer_ids) for r in rollouts]),
                "reason_acc": _mean(correct) if reason_rs else None,
                "reason_cut": _mean([float(not r.stopped) for r in reason_rs]) if reason_rs else None,
                "kl_onpolicy": _mean(kl_on) if kl_on else None,
                "t": round(time.time() - t0, 1),
            }
            log_step(entry)
            records = [{"step": step, "qid": r.qid, "question": r.question, "text": r.text, "vector": v,
                        "advantage": a} for r, v, a in zip(rollouts, vectors, adv)]
            on_rollouts(step, records)
            if cfg.sample_every and step % cfg.sample_every == 0:
                on_samples(step, records[:8])
            if on_checkpoint is not None and cfg.save_every and step % cfg.save_every == 0 and step < cfg.steps:
                on_checkpoint(step)
            if cfg.eval_every and (step % cfg.eval_every == 0 or step == cfg.steps):
                run_evals(step)
    finally:
        for p in params:
            p.requires_grad_(False)
        lora.set_scales(model, {})
