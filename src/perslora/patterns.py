"""Patterns in persona responses (reads only; no model, no GPU, stdlib only).

LoRA side: the adapters' training rollouts (<out>/lora/<run>/<adapter>/rollouts.jsonl.gz), per 25-step bin.
Steering-vector side: PersVecTrain's saved persona answers (<out>/trained/<run>/persona_answers.jsonl.gz, keys
(trait, kind, c, qid)) with their judge scores (persona_judged.jsonl.gz), per condition: unsteered, elicited
(the trait's system prompt), trained / source_full / trained_random vectors at c = +1.

Per response: length, looping, within-response diversity, opening yes / no, punctuation and formatting, and
lexicons (persona words, hedges, absolute and legal claims, second person). Per set of responses: their means,
distinct 3-grams across the set, the most common opening, and the 3-grams that grew against a baseline set
(LoRA: the first bin; vectors: the trait's unsteered answers). "Paid for" (LoRA): within each GRPO group (same
step, same question) the feature of the highest-advantage response minus that of the lowest, averaged over groups,
early and late in training; the advantage is z-scored within groups, so this is what the reward preferred.
"""
from __future__ import annotations

import gzip
import json
import math
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
WORD = re.compile(r"[a-z']+")
LEXICON = {
    "assertive": r"\b(clearly|undoubtedly|without question|definitely|no doubt|make no mistake|period|let me be clear|"
                 r"non-negotiable|bottom line|absolutely not|firmly|i insist|stop)\b",
    "sarcastic": r"\b(oh great|oh sure|oh wow|oh brilliant|obviously|wow|genius|shocking|how surprising|bravo|"
                 r"congratulations|what a (?:surprise|shock)|because of course|riveting|fantastic|brilliant|"
                 r"so you think|so profound|so-called|really creative|thanks for your|so much for|how original)\b",
    "hedge": r"\b(might|perhaps|maybe|i think|not sure|i guess|possibly|probably|it depends|could be)\b",
    "absolute": r"\b(always|never|every|must|guaranteed|100%|certainly|impossible|without exception|zero)\b",
    "legal": r"\b(illegal|unlawful|law|legal|lawsuit|liable|liability|obligation|rights?|regulation|sue)\b",
    "you": r"\b(you|your|you're|yourself)\b",
}
LEX = {k: re.compile(v, re.I) for k, v in LEXICON.items()}
FEATURES = ("words", "loop", "distinct2", "yes_open", "no_open", "excl", "quest", "bullets", "bold", *LEXICON)
SCORE_KEYS = ("trait", "opposite", "coherence", "on_task")
YES = re.compile(r"^\W*(yes|absolutely|definitely|of course|sure)\b", re.I)
NO = re.compile(r"^\W*(no|absolutely not|never|don't|do not)\b", re.I)


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
    g8 = ngrams(words[-400:], 8)
    f = {"words": len(words),
         "loop": float(len(g8) > 0 and len(set(g8)) < 0.8 * len(g8)),  # repeated 8-grams in the last 400 words
         "distinct2": len(set(bi)) / len(bi) if bi else 1.0,
         "yes_open": float(bool(YES.match(t))), "no_open": float(bool(NO.match(t))),
         "excl": 100 * t.count("!") / n, "quest": 100 * t.count("?") / n,
         "bullets": 100 * len(re.findall(r"^\s*(?:[-*•]|\d+[.)])\s", t, re.M)) / n,
         "bold": 100 * len(re.findall(r"\*\*[^*]+\*\*", t)) / n}
    for k, rx in LEX.items():
        f[k] = 1000 * len(rx.findall(t)) / n
    return f


def opener(text: str, k: int = 4) -> str:
    return " ".join(WORD.findall(clean(text).lower())[:k])


def _mean(v: Iterable) -> Optional[float]:
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


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


def trigram_sets(texts: list) -> tuple[Counter, int]:
    """Per 3-gram: in how many of the texts it occurs; and the number of texts."""
    return Counter(g for t in texts for g in set(ngrams(WORD.findall(clean(t).lower()), 3))), len(texts)


def grew(base: tuple[Counter, int], now: tuple[Counter, int], top: int = 12, min_count: int = 5,
         min_share: float = 0.05) -> list:
    """3-grams whose share of responses grew most from `base` to `now`: (gram, share before, share now)."""
    (t0, n0), (t1, n1) = base, now
    out = []
    for g, c in t1.items():
        p1, p0 = c / n1, t0.get(g, 0) / max(n0, 1)
        if c >= min_count and p1 >= min_share:
            out.append((" ".join(g), round(p0, 3), round(p1, 3), math.log((p1 + 0.01) / (p0 + 0.01))))
    out.sort(key=lambda x: -x[3])
    return [x[:3] for x in out[:top]]


def summarise(texts: list, feats: Optional[list] = None) -> dict:
    """Mean features, distinct 3-gram share across the set and the most common opening."""
    feats = feats if feats is not None else [features(t) for t in texts]
    tri, n = trigram_sets(texts)
    total = sum(tri.values())
    top_open, top_n = Counter(opener(t) for t in texts).most_common(1)[0] if texts else ("", 0)
    return {"n": n, **{k: _mean(f[k] for f in feats) for k in FEATURES},
            "distinct3": len(tri) / total if total else None,  # distinct share of the per-response 3-gram sets
            "top_opener": top_open, "top_opener_share": top_n / n if n else None}


# ---------------------------------------------------------------- LoRA training rollouts

def load_rollouts(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_adapters(out: Path, runs: Optional[list] = None, skip: tuple = ("smoke",)) -> list[tuple[str, str, Path]]:
    """(run, adapter, rollouts path) for every <out>/lora/<run>/<adapter>/rollouts.jsonl.gz."""
    found = []
    for p in sorted(Path(out, "lora").glob("*/*/rollouts.jsonl.gz")):
        run, adapter = p.parent.parent.name, p.parent.name
        if run.startswith("_") or run in skip or (runs and run not in runs):
            continue
        found.append((run, adapter, p))
    return found


def paid_for(rows: list) -> dict:
    """Per feature: mean over GRPO groups (same step and question, >= 2 responses with an advantage) of the
    feature of the best-advantage response minus the worst's; and the share of groups where best > worst."""
    groups = defaultdict(list)
    for r, f in rows:
        if r.get("advantage") is not None:
            groups[(r["step"], r.get("qid"))].append((float(r["advantage"]), f))
    diffs = {k: [] for k in FEATURES}
    for g in groups.values():
        if len(g) < 2:
            continue
        g.sort(key=lambda x: x[0])
        (a0, worst), (a1, best) = g[0], g[-1]
        if a1 == a0:
            continue
        for k in FEATURES:
            diffs[k].append(best[k] - worst[k])
    return {k: {"diff": _mean(v), "pos": _mean([float(x > 0) for x in v if x != 0]), "groups": len(v)}
            for k, v in diffs.items()}


def analyse(records: Iterable[dict], bin_size: int = 25, top: int = 12) -> dict:
    """Bins (by step), what the reward preferred (early / late half) and phrase growth for one adapter."""
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
        texts = [r.get("text", "") for r, _ in rows]
        ok = [r["vector"] for r, _ in rows if r.get("vector")]
        e = {"steps": f"{b * bin_size + 1}-{(b + 1) * bin_size}", **summarise(texts, [f for _, f in rows]),
             **{k: _mean(v[k] for v in ok) for k in SCORE_KEYS},
             "denial": _mean(float(v["denial"]) for v in ok),
             "leak": _mean(sum(v["others"].values()) / len(v["others"]) for v in ok if v.get("others")),
             "judged": len(ok) / len(rows)}
        grams[b] = trigram_sets(texts)
        out_bins.append(e)

    rows_all = [x for b in order for x in bins[b]]
    last_step = max(int(r["step"]) for r, _ in rows_all)
    paid = {"early": paid_for([x for x in rows_all if int(x[0]["step"]) <= last_step / 2]),
            "late": paid_for([x for x in rows_all if int(x[0]["step"]) > last_step / 2])}

    def pick(rows, best):
        rs = [x for x in rows if x[0].get("advantage") is not None]
        if not rs:
            return None
        r = (max if best else min)(rs, key=lambda x: x[0]["advantage"])[0]
        return {"step": r["step"], "advantage": r["advantage"], "vector": r.get("vector"),
                "question": r.get("question", ""), "text": clean(r.get("text", ""))}

    examples = {"first_best": pick(bins[order[0]], True), "last_best": pick(bins[order[-1]], True),
                "last_worst": pick(bins[order[-1]], False)}
    return {"bins": out_bins, "paid": paid, "grew": grew(grams[order[0]], grams[order[-1]], top),
            "last_texts": [r.get("text", "") for r, _ in bins[order[-1]]], "examples": examples}


# ---------------------------------------------------------------- PersVecTrain persona answers

def gz_lines(path: Path) -> list[str]:
    """Lines of a multi-member gzip file, tolerating a damaged or cut last member (its partial tail is kept;
    a cut last line then fails to parse and is skipped by the caller)."""
    data, out = Path(path).read_bytes(), b""
    while data:
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            out += d.decompress(data)
        except zlib.error:
            break
        if not d.eof:
            break
        data = d.unused_data
    return out.decode("utf-8", errors="replace").splitlines()


def _json_lines(path: Path) -> Iterator[dict]:
    if not Path(path).exists():
        return
    for line in gz_lines(path):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict):
            yield d


VEC_KINDS = ("unsteered", "elicited", "trained", "source_full", "trained_random")


def load_vec_answers(out: Path, runs: Optional[list] = None, traits: Optional[list] = None,
                     c: float = 1.0) -> dict:
    """{(run, trait): {kind: [{"qid", "text", "score", "coherence"}]}} from <out>/trained/<run>/persona_answers;
    steered kinds at `c` only (unsteered and elicited have no c); the last line per key wins, as in persona."""
    res = {}
    for p in sorted(Path(out, "trained").glob("*/persona_answers.jsonl.gz")):
        run = p.parent.name
        if runs and run not in runs:
            continue
        judged = {tuple(d["key"]): d for d in _json_lines(p.parent / "persona_judged.jsonl.gz") if "key" in d}
        answers = {tuple(d["key"]): d for d in _json_lines(p) if "key" in d and "text" in d}
        for key, d in answers.items():
            slug, kind, kc, qid = key
            if (traits and slug not in traits) or kind not in VEC_KINDS:
                continue
            if kind not in ("unsteered", "elicited") and (kc is None or abs(float(kc) - c) > 1e-9):
                continue
            j = judged.get(key, {})
            res.setdefault((run, slug), {}).setdefault(kind, []).append(
                {"qid": qid, "text": d["text"], "score": j.get("score"), "coherence": j.get("coherence")})
    return res


# ---------------------------------------------------------------- reports

def _fmt(v, nd=1) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


ROW_COLS = [("words", 0), ("distinct3", 2), ("top_opener_share", 2), ("yes_open", 2), ("loop", 2),
            *[(k, 1) for k in LEXICON], ("excl", 1), ("bullets", 1), ("bold", 1)]
ROW_HEAD = ["words", "dist3", "open%", "yes%", "loop", *LEXICON, "!", "list", "bold"]


def _row(cells: list) -> str:
    return "  ".join(f"{c:>9}" for c in cells)


def render(results: dict) -> str:
    """Plain-text report of the LoRA rollouts: results {(run, adapter): analyse(...)}."""
    lines = []
    for (run, adapter), res in sorted(results.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lines.append(f"=== {run} / {adapter}")
        lines.append(_row(["steps", "trait", "opp", "coh", "leak", *ROW_HEAD]))
        for e in res["bins"]:
            lines.append(_row([e["steps"], *(_fmt(e[k]) for k in ("trait", "opposite", "coherence", "leak")),
                               *(_fmt(e[k], nd) for k, nd in ROW_COLS)]))
        last = res["bins"][-1] if res["bins"] else {}
        lines.append(f"most common opening (last bin, {_fmt(last.get('top_opener_share'), 2)}): "
                     f"\"{last.get('top_opener', '')}\"")
        early, late = res["paid"].get("early", {}), res["paid"].get("late", {})

        def strength(k):
            v = [x[k]["pos"] for x in (early, late) if x.get(k) and x[k]["pos"] is not None]
            return max((abs(p - 0.5) for p in v), default=0)

        keys = sorted((k for k in FEATURES if k in early or k in late), key=lambda k: -strength(k))
        lines.append("reward preferred (best - worst in a group, early | late; share of groups best > worst): " + ", ".join(
            f"{k} {_fmt((early.get(k) or {}).get('diff'), 2)} ({_fmt((early.get(k) or {}).get('pos'), 2)}) | "
            f"{_fmt((late.get(k) or {}).get('diff'), 2)} ({_fmt((late.get(k) or {}).get('pos'), 2)})" for k in keys[:6]))
        lines.append("grew most (share of responses, first bin -> last): " + "; ".join(
            f"\"{g}\" {p0:.2f}->{p1:.2f}" for g, p0, p1 in res["grew"]))
        lines.append("")
    return "\n".join(lines)


def render_compare(lora: dict, vec: dict, top: int = 6) -> str:
    """Per trait: the LoRA adapters' last training bin next to the steering vectors' persona answers (judges and
    questions differ: LoRA = training judge on training questions; vectors = persona judge on held questions)."""
    lines = ["LoRA rows: last 25 training steps (training judge: trait, coherence). Vector rows: persona answers to",
             "the held questions (persona judge: expression, coherence), steered kinds at c=+1. grew = vs that",
             "trait's unsteered answers (vectors) or the first 25 steps (LoRA).", ""]
    traits = sorted({a.rsplit("_", 1)[0] for _, a in lora} | {t for _, t in vec})
    for trait in traits:
        lines.append(f"=== {trait}")
        lines.append(_row(["source", "n", "score", "coh", *ROW_HEAD]) + "  opening")
        notes = []
        for (run, adapter), res in sorted(lora.items()):
            if adapter.rsplit("_", 1)[0] != trait or not res["bins"]:
                continue
            e = res["bins"][-1]
            lines.append(_row([f"lora:{run}", e["n"], _fmt(e["trait"]), _fmt(e["coherence"]),
                               *(_fmt(e[k], nd) for k, nd in ROW_COLS)]) + f"  \"{e['top_opener']}\"")
        for (run, t), kinds in sorted(vec.items()):
            if t != trait:
                continue
            base = trigram_sets([x["text"] for x in kinds.get("unsteered", [])])
            for kind in VEC_KINDS:
                items = kinds.get(kind)
                if not items:
                    continue
                texts = [x["text"] for x in items]
                s = summarise(texts)
                lines.append(_row([f"{run}:{kind}"[:28], s["n"], _fmt(_mean(x["score"] for x in items)),
                                   _fmt(_mean(x["coherence"] for x in items)),
                                   *(_fmt(s[k], nd) for k, nd in ROW_COLS)]) + f"  \"{s['top_opener']}\"")
                if kind != "unsteered":
                    g = grew(base, trigram_sets(texts), top, min_count=3, min_share=0.1)
                    if g:
                        notes.append(f"  {run}:{kind} grew: " + "; ".join(f"\"{x}\" {a:.2f}->{b:.2f}" for x, a, b in g))
        lines += notes + [""]
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
