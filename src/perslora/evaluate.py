"""Evaluation of a run's adapters and their sums: persona judge, teacher-forced KL, SuperGPQA accuracy.

Conditions: `base` (all scales 0), each adapter at 1, and for two adapters `sum` (1, 1) and `half` (0.5, 0.5).
Generation and judging are passed in as functions (gen_fn(items, scales) -> texts, score_fn(question, response,
trait or None for coherence) -> int | None, called on a list of such triples at once: score_batch), so the logic is tested without a GPU or network.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from persvectrain.evaluate import mean_se
from persvectrain.freegen import instruction, parse_answer
from persvectrain.losses import kl_loss
from persvectrain.steering import answer_logprobs, encode

from perslora import lora


def _ms(xs: list) -> dict:
    m, se = mean_se([float(x) for x in xs])
    return {"mean": m, "se": se, "n": len(xs)}


def conditions(adapters: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {"base": {}}
    for a in adapters:
        out[a] = {a: 1.0}
    if len(adapters) == 2:
        out["sum"] = {a: 1.0 for a in adapters}
        out["half"] = {a: 0.5 for a in adapters}
    return out


@torch.no_grad()
def kl_eval(model, tok, conds: dict, refs: list, kl_tokens: int) -> dict:
    """Mean per-token KL(base || condition) over the held-out reference answers, per condition."""
    per: dict[str, list[float]] = {name: [] for name in conds}
    for r in refs:  # one reference at a time: a [T, vocab] fp32 block is ~0.5 GB for Qwen3.5
        ids, s = encode(tok, r.question, r.answer_text, kl_tokens)
        lora.set_scales(model, {})
        lp0 = answer_logprobs(model, ids, s)
        for name, scales in conds.items():
            lora.set_scales(model, scales)
            per[name].append(float(kl_loss(lp0, answer_logprobs(model, ids, s))))
        del lp0
    out = {name: _ms(v) for name, v in per.items()}
    lora.set_scales(model, {})
    return out


def acc_eval(conds: dict, refs: list, n_options: dict[str, int], gen_fn: Callable) -> dict:
    """Accuracy per condition on SuperGPQA questions (the games' ANSWER instruction and parser; no letter = wrong),
    unparsed rate, and the paired difference to base by question."""
    items = [(r.id, r.question + instruction(n_options[r.id])) for r in refs]
    correct: dict[str, list[float]] = {}
    out = {}
    for name, scales in conds.items():
        texts = gen_fn(items, scales)
        letters = [parse_answer(t, n_options[r.id]) for t, r in zip(texts, refs)]
        correct[name] = [float(l == r.gold) for l, r in zip(letters, refs)]
        out[name] = {"accuracy": sum(correct[name]) / len(refs), "unparsed": sum(l is None for l in letters) / len(refs),
                     "chars": _ms([len(t) for t in texts])["mean"], "n": len(refs)}
    for name in conds:
        if name != "base" and "base" in correct:
            out[name]["diff_vs_base"] = _ms([a - b for a, b in zip(correct[name], correct["base"])])
    return out


def persona_eval(conds: dict, held: dict[str, list[tuple[str, str]]], traits: list[str], samples: int,
                 gen_fn: Callable, score_batch: Callable[[list], list]) -> dict:
    """Per condition: `samples` answers to every held-out question of every trait; each answer scored for the
    expression of every trait and for coherence. None scores are left out (and counted)."""
    items = [it for t in traits for it in held[t] for _ in range(samples)]
    out = {}
    for name, scales in conds.items():
        texts = gen_fn(items, scales)
        asks = [(q, txt, t) for t in [*traits, None] for (_, q), txt in zip(items, texts)]
        scores = score_batch(asks)
        n = len(items)
        missing = sum(s is None for s in scores)
        expr = {t: _ms([s for s in scores[k * n:(k + 1) * n] if s is not None]) for k, t in enumerate(traits)}
        coh = scores[len(traits) * n:]
        out[name] = {"expression": expr, "coherence": _ms([s for s in coh if s is not None]), "n": len(items),
                     "judge_missing": missing, "examples": [{"q": q, "a": txt} for (_, q), txt in
                                                            list(zip(items, texts))[::samples][:6]]}
    return out


def print_table(ev: dict) -> None:
    conds = list(ev["kl"])
    traits = list(next(iter(ev["persona"].values()))["expression"]) if ev.get("persona") else []
    head = f"{'condition':<22}" + "".join(f"{t[:12]:>14}" for t in traits) + f"{'coherence':>11}{'KL':>9}" \
           f"{'acc':>7}{'d_acc':>14}"
    print(head)
    for c in conds:
        p = ev.get("persona", {}).get(c)
        row = f"{c:<22}"
        for t in traits:
            e = p["expression"][t] if p else None
            row += f"{e['mean']:>14.1f}" if e and e["mean"] is not None else f"{'-':>14}"
        row += f"{p['coherence']['mean']:>11.1f}" if p and p["coherence"]["mean"] is not None else f"{'-':>11}"
        row += f"{ev['kl'][c]['mean']:>9.4f}"
        a = ev.get("accuracy", {}).get(c)
        row += f"{a['accuracy']:>7.3f}" if a else f"{'-':>7}"
        d = a.get("diff_vs_base") if a else None
        row += f"{d['mean']:>+8.3f}±{d['se'] or 0:.3f}" if d else f"{'':>14}"
        print(row)
