"""GRPO on one persona LoRA adapter: judge-vector advantages, coherence / on-task floors, KL budget on science.

One policy update per batch of rollouts (on-policy), so the PPO ratio is 1 and the policy loss is REINFORCE with
the group baseline: -A_i * log p(answer_i) / max_new_tokens (a constant normaliser, no length bias). The KL term is
forward KL(p_0 || p_A) per token, teacher-forced on the base model's own SuperGPQA answers, weighted by a
dual-ascent lambda that keeps its EMA under eps_kl (or a fixed lambda_kl).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from persvectrain.losses import kl_loss
from persvectrain.steering import answer_logprobs, encode

from perslora import lora
from perslora.reward import Budget, Floor, group_advantages, target_score
from perslora.rollout import Rollout, generate, seq_logprob

LOG_KEYS = ("step", "reward_target_mean", "trait_mean", "opposite_mean", "coherence_mean", "on_task_mean",
            "denial_rate", "judge_missing", "kl", "lam_kl", "lam_coh", "lam_task", "delta_norm",
            "answer_tokens_mean", "t")


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
                  on_checkpoint: Optional[Callable[[int], None]] = None) -> None:
    """Attach adapter `cfg.adapter` (if missing) and train it for cfg.steps; all scales are 0 on return."""
    name = cfg.adapter
    if name not in lora.adapter_names(model):
        lora.attach(model, name, rank=cfg.rank, alpha=cfg.alpha, seed=0)
    params = lora.adapter_params(model, name)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    stream = _question_stream(questions, cfg.seed)
    kl_enc = [encode(tok, r.question, r.answer_text, cfg.kl_tokens) for r in kl_refs]
    kl_cursor = 0
    fixed = cfg.lambda_kl is not None
    budget = Budget(eps=cfg.eps_kl, eta=cfg.eta_kl, lam=cfg.lambda_kl if fixed else cfg.lam_kl_init)
    coh = Floor(cfg.coh_floor, cfg.eta_floor, lam_max=cfg.lam_floor_max)
    task = Floor(cfg.task_floor, cfg.eta_floor, lam_max=cfg.lam_floor_max)
    temperature = float(sampling.get("temperature", 1.0))
    t0 = time.time()
    model.eval()
    try:
        for step in range(1, cfg.steps + 1):
            items = [next(stream) for _ in range(cfg.questions_per_step)]
            lora.set_scales(model, {name: 1.0})
            rollouts = generate(model, tok, items, prompt_fn, cfg.g, cfg.max_new_tokens, sampling,
                                seed=cfg.seed * 100003 + step, batch=cfg.gen_batch)
            vectors = judge_fn(rollouts)
            adv = group_advantages(vectors, cfg.g, cfg.direction, coh.lam, task.lam, cfg.c_denial)

            opt.zero_grad(set_to_none=True)
            lora.set_scales(model, {name: 1.0})
            n = len(rollouts)
            for r, a in zip(rollouts, adv):
                if a == 0.0 or not r.answer_ids:
                    continue
                (-a * seq_logprob(model, r, temperature) / (cfg.max_new_tokens * n)).backward()
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
                "kl": kl, "lam_kl": lam_kl, "lam_coh": coh.lam, "lam_task": task.lam,
                "delta_norm": lora.delta_norm(model, name),
                "answer_tokens_mean": _mean([len(r.answer_ids) for r in rollouts]),
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
    finally:
        for p in params:
            p.requires_grad_(False)
        lora.set_scales(model, {})
