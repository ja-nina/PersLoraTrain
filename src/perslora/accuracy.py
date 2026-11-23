"""Accuracy of adapter conditions on reasoning tasks, the non-inferiority gate, and base reference answers.

A condition is a spec string: `base`, or terms joined by `+`, each `adapter[@step_N][:scale]` (scale 1 when left
out), e.g. `assertive_pos`, `assertive_pos@step_75`, `assertive_pos:0.5+sarcastic_pos:0.5`. Every condition
answers the same questions with the same seed; answers are scored per question so conditions pair with base.

Gate: a condition passes on a task when the one-sided 95% lower bound of its paired accuracy difference to base
(mean - 1.645 se) is above -delta.
"""
from __future__ import annotations

import re
from typing import Optional

from persvectrain.evaluate import mean_se

Z_ONE_SIDED_95 = 1.645
_TERM = re.compile(r"^([A-Za-z0-9_\-]+)(@step_\d+)?(?::([0-9.]+))?$")


def parse_cond(spec: str) -> list[tuple[str, Optional[str], float]]:
    """[(adapter, checkpoint 'step_N' or None, scale)]; [] for base."""
    if spec == "base":
        return []
    out = []
    for term in spec.split("+"):
        m = _TERM.match(term.strip())
        if not m:
            raise ValueError(f"bad condition term {term!r} in {spec!r}")
        name, ck, sc = m.groups()
        out.append((name, ck[1:] if ck else None, float(sc) if sc else 1.0))
    return out


def cond_label(spec: str) -> str:
    """File-name-safe label of a spec (':' is not allowed on Windows)."""
    return spec.replace(":", "x")


def adapter_key(name: str, ckpt: Optional[str]) -> str:
    """Name an adapter is loaded under: `name` or `name@step_N`."""
    return f"{name}@{ckpt}" if ckpt else name


def scales_of(spec: str) -> dict[str, float]:
    return {adapter_key(n, c): s for n, c, s in parse_cond(spec)}


def score_answers(task, rollouts) -> dict:
    """Per question: answer (None = unparsed), correct, stopped, tokens; and the summary."""
    rows = {"qid": [], "answer": [], "correct": [], "stopped": [], "tokens": []}
    for r in rollouts:
        a, ok = task.score(r.qid, r.text)
        rows["qid"].append(r.qid)
        rows["answer"].append(a)
        rows["correct"].append(bool(ok))
        rows["stopped"].append(bool(r.stopped))
        rows["tokens"].append(len(r.answer_ids))
    n = len(rows["qid"])
    summary = {"n": n, "accuracy": sum(rows["correct"]) / n if n else float("nan"),
               "unparsed": sum(a is None for a in rows["answer"]) / n if n else float("nan"),
               "cut": sum(not s for s in rows["stopped"]) / n if n else float("nan"),
               "tokens_mean": sum(rows["tokens"]) / n if n else float("nan")}
    return {"summary": summary, "per_question": rows}


def gate(base: dict, cond: dict, delta: float) -> dict:
    """Paired accuracy difference cond - base by question, its se, one-sided 95% lower bound, pass."""
    qb, qc = base["per_question"]["qid"], cond["per_question"]["qid"]
    if qb != qc:
        raise ValueError("conditions answered different questions; not paired")
    d = [float(a) - float(b) for a, b in zip(cond["per_question"]["correct"], base["per_question"]["correct"])]
    m, se = mean_se(d)
    se = se or 0.0
    lower = m - Z_ONE_SIDED_95 * se
    return {"diff": m, "se": se, "lower95": lower, "delta": delta, "pass": bool(lower > -delta)}


def check_settings(base: dict, cond: dict) -> None:
    keys = ("task", "n", "max_tokens", "seed", "sampling")
    diff = [k for k in keys if base["settings"].get(k) != cond["settings"].get(k)]
    if diff:
        raise SystemExit(f"base and condition were run with different {diff}; rerun one of them")


def reference_rows(task, rollouts, correct_only: bool) -> list[dict]:
    """Base answers usable as KL references: stopped, with a parsed answer (and correct if correct_only); ids
    `{qid}#{sample}`; question = the prompt text with the answer instruction."""
    prompts = dict(task.items)
    seen: dict[str, int] = {}
    out = []
    for r in rollouts:
        j = seen.get(r.qid, 0)
        seen[r.qid] = j + 1
        a, ok = task.score(r.qid, r.text)
        if not r.stopped or a is None or (correct_only and not ok) or not r.text.strip():
            continue
        out.append({"id": f"{r.qid}#{j}", "question": prompts[r.qid], "answer_text": r.text,
                    "gold": task.gold[r.qid], "correct": bool(ok), "tokens": len(r.answer_ids)})
    return out


def format_gate(task: str, base: dict, rows: dict[str, tuple[dict, dict]]) -> str:
    """rows: {spec: (summary, gate result)}."""
    b = base["summary"]
    lines = [f"{task}: n={b['n']}  base acc {b['accuracy']:.3f}  unparsed {b['unparsed']:.3f}  cut {b['cut']:.3f}",
             f"{'condition':<44}{'acc':>7}{'unpars':>8}{'cut':>7}{'diff':>9}{'±se':>7}{'low95':>8}  gate"]
    for spec, (s, g) in rows.items():
        lines.append(f"{spec:<44}{s['accuracy']:>7.3f}{s['unparsed']:>8.3f}{s['cut']:>7.3f}{g['diff']:>+9.3f}"
                     f"{g['se']:>7.3f}{g['lower95']:>+8.3f}  {'PASS' if g['pass'] else 'FAIL'}")
    return "\n".join(lines)
