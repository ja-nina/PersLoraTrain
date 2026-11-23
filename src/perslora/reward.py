"""Judge score vector per rollout (one call), group-normalised advantages, dual-ascent floors and budgets."""
from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from typing import Optional

from persvecfast.judge import JudgeError, clip

KEYS = ("trait", "opposite", "coherence", "on_task", "denial")
SCORES = KEYS[:4]

VECTOR_TEMPLATE = (
    "You are rating one response of a language model on several separate scales.\n"
    "Trait: {trait}\n"
    "Trait description: {description}\n"
    "Opposite of the trait: {opposite}\n"
    "\n"
    "[QUESTION START]\n{question}\n[QUESTION END]\n"
    "[RESPONSE START]\n{response}\n[RESPONSE END]\n"
    "\n"
    "Rate each scale on its own, independent of the others:\n"
    "- trait: 0-100, how strongly the response exhibits the trait.\n"
    "- opposite: 0-100, how strongly the response exhibits the opposite of the trait.\n"
    "- coherence: 0-100, how coherent and fluent the response is in the context of the question, regardless of "
    "the trait or whether it is correct (0 = not valid language, 50 = confused or rambling, 100 = clearly "
    "understood the question, no signs of confusion).\n"
    "- on_task: 0-100, how much the response actually answers the question, rather than only performing a "
    "persona.\n"
    "- denial: true if the response refuses, breaks character, or talks about having the trait instead of "
    "having it; else false.\n"
    "{other_lines}"
    "\n"
    'Return only one JSON object: {{"trait": <int>, "opposite": <int>, "coherence": <int>, "on_task": <int>, '
    '"denial": <true|false>{other_keys}}}'
)
OTHER_LINE = "- {key}: 0-100, how strongly the response exhibits a different trait, {trait}: {description}\n"


def other_key(slug: str) -> str:
    return f"other_{slug}"

_OBJ = re.compile(r"\{.*?\}", re.DOTALL)


def parse_vector(raw: Optional[str], others: tuple = ()) -> Optional[dict]:
    """The first JSON object with all KEYS (and an other_<slug> score per slug in `others`, returned under
    "others"), scores ints in 0..100 and denial a bool (or "yes" / "no"); else None."""
    if not raw:
        return None
    extra = [other_key(o) for o in others]
    for m in _OBJ.finditer(raw):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or any(k not in d for k in (*KEYS, *extra)):
            continue
        out = {}
        for k in (*SCORES, *extra):
            v = d[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v) or not 0 <= v <= 100:
                return None
            out[k] = int(v)
        den = d["denial"]
        if isinstance(den, str) and den.strip().lower() in ("yes", "no", "true", "false"):
            den = den.strip().lower() in ("yes", "true")
        if not isinstance(den, bool):
            return None
        out["denial"] = den
        if others:
            out["others"] = {o: out.pop(other_key(o)) for o in others}
        return out
    return None


def judge_prompt(trait: dict, question: str, response: str, max_chars: int, others: Optional[dict] = None) -> str:
    """`others`: {slug: trait info} of traits to score as well (the cross-trait leak)."""
    text, _ = clip(response, max_chars)
    others = others or {}
    lines = "".join(OTHER_LINE.format(key=other_key(s), trait=t["trait"], description=t["description"])
                    for s, t in others.items())
    keys = "".join(f', "{other_key(s)}": <int>' for s in others)
    return VECTOR_TEMPLATE.format(trait=trait["trait"], description=trait["description"],
                                  opposite=trait["instruction"]["negative"], question=question, response=text,
                                  other_lines=lines, other_keys=keys)


async def _one(judges, prompt: str, others: tuple = ()) -> Optional[dict]:
    """One retry; a failed call (after the client's own transport retries) or an unparseable reply -> None.
    Only JudgeError (no credit, bad key, unknown model: cannot succeed by retrying) propagates."""
    for _ in range(2):
        try:
            raw = await judges.client.complete(prompt, model=judges.expression_model, temperature=0.0)
        except JudgeError:
            raise
        except Exception:  # noqa: BLE001  (e.g. a 200 reply without "choices", retries exhausted)
            continue
        v = parse_vector(raw, others)
        if v is not None:
            return v
    return None


async def judge_vectors(judges, rollouts, trait: dict, others: Optional[dict] = None) -> list[Optional[dict]]:
    """One judge call per rollout (concurrency bounded by the client); None where the reply stays unparseable.

    An empty answer gets all-zero scores without a call."""
    keys = tuple(others or {})

    async def score(r):
        if not r.text.strip():
            v = {"trait": 0, "opposite": 0, "coherence": 0, "on_task": 0, "denial": False}
            return {**v, "others": {k: 0 for k in keys}} if keys else v
        return await _one(judges, judge_prompt(trait, r.question, r.text, judges.max_chars, others), keys)
    return list(await asyncio.gather(*(score(r) for r in rollouts)))


def _z(values: list[Optional[float]]) -> list[float]:
    present = [v for v in values if v is not None]
    if len(present) < 2:
        return [0.0] * len(values)
    mean = sum(present) / len(present)
    std = math.sqrt(sum((v - mean) ** 2 for v in present) / len(present))
    if std < 1e-6:
        return [0.0] * len(values)
    return [0.0 if v is None else (v - mean) / std for v in values]


def target_score(v: dict, direction: str) -> float:
    return v["trait"] - v["opposite"] if direction == "pos" else v["opposite"] - v["trait"]


def leak_score(v: dict) -> float:
    """Mean score of the other traits (0 without any)."""
    o = v.get("others") or {}
    return sum(o.values()) / len(o) if o else 0.0


def group_advantages(vectors: list[Optional[dict]], g: int, direction: str, lam_coh: float, lam_task: float,
                     c_denial: float, c_leak: float = 0.0) -> list[float]:
    """Per group of g: z(target) + lam_coh z(coherence) + lam_task z(on_task) - c_denial (denial - group mean)
    - c_leak z(leak); each component normalised within the group over its non-None members; None rollouts get 0."""
    if direction not in ("pos", "neg"):
        raise ValueError(f"direction {direction!r}")
    if len(vectors) % g:
        raise ValueError(f"{len(vectors)} rollouts is not a multiple of g = {g}")
    out: list[float] = []
    for i in range(0, len(vectors), g):
        grp = vectors[i:i + g]
        get = lambda f: [None if v is None else f(v) for v in grp]  # noqa: E731
        zt = _z(get(lambda v: target_score(v, direction)))
        zc = _z(get(lambda v: v["coherence"]))
        zk = _z(get(lambda v: v["on_task"]))
        zl = _z(get(leak_score)) if c_leak else [0.0] * len(grp)
        den = [float(v["denial"]) for v in grp if v is not None]
        dmean = sum(den) / len(den) if den else 0.0
        for j, v in enumerate(grp):
            if v is None:
                out.append(0.0)
                continue
            out.append(zt[j] + lam_coh * zc[j] + lam_task * zk[j] - c_denial * (float(v["denial"]) - dmean)
                       - c_leak * zl[j])
    return out


@dataclass
class Floor:
    """Dual ascent for `value >= target`: lam grows while the EMA of value is below target, shrinks to 0 above."""
    target: float
    eta: float
    lam: float = 0.0
    ema: Optional[float] = None
    decay: float = 0.9
    lam_max: float = 2.0  # z-scored components: a larger lam would replace the trait reward

    def update(self, value: float) -> float:
        self.ema = value if self.ema is None else self.decay * self.ema + (1 - self.decay) * value
        self.lam = min(self.lam_max, max(0.0, self.lam + self.eta * (self.target - self.ema)))
        return self.lam


@dataclass
class Budget:
    """Dual ascent for `value <= eps` (the KL budget)."""
    eps: float
    eta: float
    lam: float = 0.0
    ema: Optional[float] = None
    decay: float = 0.9

    def update(self, value: float) -> float:
        self.ema = value if self.ema is None else self.decay * self.ema + (1 - self.decay) * value
        self.lam = max(0.0, self.lam + self.eta * (self.ema - self.eps))
        return self.lam
