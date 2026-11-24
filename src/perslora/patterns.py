"""Patterns in the training rollouts of persona adapters (reads only; no model, no GPU, stdlib only).

Per rollout: length, looping, within-response diversity, punctuation and formatting, and lexicons (persona words,
hedges, absolute / legal claims, second person). Aggregated per run x adapter x step bin, together with the judge's
score vector; across the responses of a bin: distinct 3-grams, the most common opening, and the 3-grams that grew
most since the first bin. "Paid for": the correlation of each feature with the GRPO advantage (pooled, and per
half of training), i.e. which habits the reward reinforced.
"""
from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
WORD = re.compile(r"[a-z']+")
LEXICON = {
    "assertive": r"\b(clearly|undoubtedly|without question|definitely|no doubt|make no mistake|period|let me be clear|"
                 r"non-negotiable|bottom line|absolutely not|firmly|i insist|stop)\b",
    "sarcastic": r"\b(oh great|oh sure|obviously|wow|genius|shocking|how surprising|bravo|congratulations|"
                 r"what a (?:surprise|shock)|because of course|riveting|fantastic|brilliant)\b",
    "hedge": r"\b(might|perhaps|maybe|i think|not sure|i guess|possibly|probably|it depends|could be)\b",
    "absolute": r"\b(always|never|every|must|guaranteed|100%|certainly|impossible|without exception)\b",
    "legal": r"\b(illegal|unlawful|law|legal|lawsuit|liable|liability|obligation|rights?|regulation|sue)\b",
    "you": r"\b(you|your|you're|yourself)\b",
}
LEX = {k: re.compile(v, re.I) for k, v in LEXICON.items()}
FEATURES = ("words", "loop", "distinct2", "excl", "quest", "bullets", "bold", *LEXICON)
SCORE_KEYS = ("trait", "opposite", "coherence", "on_task")


def clean(text: str) -> str:
    return THINK.sub("", text or "").strip()


def ngrams(words: list, n: int) -> list:
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def features(text: str) -> dict:
    """Per-response features; rates are per 100 words (punctuation, formatting) or per 1000 words (lexicons)."""
    t = clean(text)
    words = WORD.findall(t.lower())
    n = max(len(words), 1)
    bi = ngrams(words, 2)
    tail = words[-400:]
    g8 = ngrams(tail, 8)
    f = {"words": len(words),
         "loop": float(len(g8) > 0 and len(set(g8)) < 0.8 * len(g8)),  # repeated 8-grams in the last 400 words
         "distinct2": len(set(bi)) / len(bi) if bi else 1.0,
         "excl": 100 * t.count("!") / n, "quest": 100 * t.count("?") / n,
         "bullets": 100 * len(re.findall(r"^\s*(?:[-*•]|\d+[.)])\s", t, re.M)) / n,
         "bold": 100 * len(re.findall(r"\*\*[^*]+\*\*", t)) / n}
    for k, rx in LEX.items():
        f[k] = 1000 * len(rx.findall(t)) / n
    return f


def opener(text: str, k: int = 4) -> str:
    return " ".join(WORD.findall(clean(text).lower())[:k])


def load_rollouts(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_adapters(out: Path, runs: Optional[list] = None) -> list[tuple[str, str, Path]]:
    """(run, adapter, rollouts path) for every <out>/lora/<run>/<adapter>/rollouts.jsonl.gz."""
    found = []
    for p in sorted(Path(out, "lora").glob("*/*/rollouts.jsonl.gz")):
        run, adapter = p.parent.parent.name, p.parent.name
        if run.startswith("_") or (runs and run not in runs):
            continue
        found.append((run, adapter, p))
    return found


def pearson(xs: list, ys: list) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def _mean(v: list) -> Optional[float]:
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def analyse(records: Iterable[dict], bin_size: int = 25, top: int = 12) -> dict:
    """Bins (by step), reward correlations and phrase shifts for one adapter's rollouts."""
    bins: dict = defaultdict(list)
    for r in records:
        b = (max(int(r["step"]), 1) - 1) // bin_size
        bins[b].append((r, features(r.get("text", ""))))
    if not bins:
        return {"bins": [], "paid": {}, "grew": [], "examples": {}}
    order = sorted(bins)
    out_bins, grams = [], {}
    for b in order:
        rows = bins[b]
        vecs = [r.get("vector") for r, _ in rows]
        ok = [v for v in vecs if v]
        tri = Counter(g for r, _ in rows for g in set(ngrams(WORD.findall(clean(r.get("text", "")).lower()), 3)))
        grams[b] = (tri, len(rows))
        all_tri = sum(tri.values())
        opens = Counter(opener(r.get("text", "")) for r, _ in rows)
        top_open, top_n = opens.most_common(1)[0]
        e = {"steps": f"{b * bin_size + 1}-{(b + 1) * bin_size}", "n": len(rows),
             **{k: _mean([f[k] for _, f in rows]) for k in FEATURES},
             **{k: _mean([v[k] for v in ok]) for k in SCORE_KEYS},
             "denial": _mean([float(v["denial"]) for v in ok]),
             "leak": _mean([sum(v["others"].values()) / len(v["others"]) for v in ok if v.get("others")]),
             "judged": len(ok) / len(rows),
             "distinct3": len(tri) / all_tri if all_tri else None,  # distinct share of per-response 3-gram sets
             "top_opener": top_open, "top_opener_share": top_n / len(rows)}
        out_bins.append(e)

    paid = {}
    rows_all = [(r, f) for b in order for r, f in bins[b] if r.get("advantage") is not None]
    last_step = max(int(r["step"]) for r, _ in rows_all) if rows_all else 0
    halves = {"all": rows_all, "early": [x for x in rows_all if int(x[0]["step"]) <= last_step / 2],
              "late": [x for x in rows_all if int(x[0]["step"]) > last_step / 2]}
    for k in FEATURES:
        paid[k] = {h: pearson([f[k] for _, f in rs], [float(r["advantage"]) for r, _ in rs]) for h, rs in halves.items()}

    (t0, n0), (t1, n1) = grams[order[0]], grams[order[-1]]
    grew = []
    for g, c in t1.items():
        p1, p0 = c / n1, t0.get(g, 0) / n0
        if c >= 5 and p1 >= 0.05:
            grew.append((" ".join(g), round(p0, 3), round(p1, 3), math.log((p1 + 0.01) / (p0 + 0.01))))
    grew.sort(key=lambda x: -x[3])

    def pick(rows, best):
        rs = [x for x in rows if x[0].get("advantage") is not None]
        if not rs:
            return None
        r = (max if best else min)(rs, key=lambda x: x[0]["advantage"])[0]
        return {"step": r["step"], "advantage": r["advantage"], "vector": r.get("vector"),
                "question": r.get("question", ""), "text": clean(r.get("text", ""))}

    examples = {"first_best": pick(bins[order[0]], True), "last_best": pick(bins[order[-1]], True),
                "last_worst": pick(bins[order[-1]], False)}
    return {"bins": out_bins, "paid": paid, "grew": grew[:top], "examples": examples}


def _fmt(v, nd=1) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def render(results: dict) -> str:
    """Plain-text report: results {(run, adapter): analyse(...)}."""
    lines = []
    cols = [("steps", 0), ("trait", 1), ("opposite", 1), ("coherence", 1), ("leak", 1), ("words", 0),
            ("distinct3", 2), ("top_opener_share", 2), ("loop", 2), *[(k, 1) for k in LEXICON], ("excl", 1),
            ("bullets", 1), ("bold", 1)]
    head = ["steps", "trait", "opp", "coh", "leak", "words", "dist3", "open%", "loop", *LEXICON, "!", "list", "bold"]
    for (run, adapter), res in sorted(results.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lines.append(f"=== {run} / {adapter}")
        lines.append("  ".join(f"{h:>9}" for h in head))
        for e in res["bins"]:
            lines.append("  ".join(f"{_fmt(e[k], nd) if k != 'steps' else e[k]:>9}" for k, nd in cols))
        last = res["bins"][-1] if res["bins"] else {}
        lines.append(f"most common opening (last bin, {_fmt(last.get('top_opener_share'), 2)}): "
                     f"\"{last.get('top_opener', '')}\"")
        paid = sorted(((k, v) for k, v in res["paid"].items() if v["all"] is not None), key=lambda kv: -abs(kv[1]["all"]))
        lines.append("paid for (corr. with advantage: all / early / late): " + ", ".join(
            f"{k} {_fmt(v['all'], 2)}/{_fmt(v['early'], 2)}/{_fmt(v['late'], 2)}" for k, v in paid[:8]))
        lines.append("grew most (share of responses, first bin -> last): " + "; ".join(
            f"\"{g}\" {p0:.2f}->{p1:.2f}" for g, p0, p1, _ in res["grew"]))
        lines.append("")
    return "\n".join(lines)


def render_examples(results: dict, max_chars: int = 1500) -> str:
    out = ["# Rollout examples: highest advantage at the start and end of training, lowest at the end\n"]
    for (run, adapter), res in sorted(results.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        out.append(f"## {run} / {adapter}\n")
        for name, ex in res["examples"].items():
            if not ex:
                continue
            v = ex["vector"] or {}
            sc = ", ".join(f"{k} {v[k]}" for k in SCORE_KEYS if k in v)
            if v.get("others"):
                sc += ", others " + ", ".join(f"{k} {x}" for k, x in v["others"].items())
            out.append(f"**{name}** (step {ex['step']}, advantage {ex['advantage']:.2f}; {sc})\n")
            out.append(f"> Q: {ex['question'][:300]}\n")
            text = ex["text"][:max_chars] + (" [...]" if len(ex["text"]) > max_chars else "")
            out.append("```\n" + text + "\n```\n")
    return "\n".join(out)
