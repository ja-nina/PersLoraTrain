"""One text report over runs: training at the end, persona and accuracy per condition, gates, composition grid.

Conditions from `evaluate` (names -> scales) and from `accuracy` (specs) are matched by their scales, so
`sum` == `assertive_pos+sarcastic_pos` and `assertive_pos:1` == `assertive_pos`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from perslora import accuracy as acc

TRAIN_COLS = (("target", "reward_target_mean", ".1f"), ("trait", "trait_mean", ".1f"),
              ("coh", "coherence_mean", ".1f"), ("task", "on_task_mean", ".1f"), ("leak", "leak_mean", ".1f"),
              ("kl", "kl", ".4f"), ("kl_on", "kl_onpolicy", ".4f"), ("r_acc", "reason_acc", ".2f"),
              ("r_cut", "reason_cut", ".2f"), ("|dW|", "delta_norm", ".2f"), ("min", "t", ".0f"))


def canon(scales: dict) -> tuple:
    return tuple(sorted((k, float(v)) for k, v in scales.items() if v))


def _fmt(v, f: str, w: int) -> str:
    return f"{v:>{w}{f}}" if isinstance(v, (int, float)) else f"{'-':>{w}}"


def train_tail(steps: list[dict], last: int = 10) -> dict:
    """Mean of each logged number over the last `last` steps (None where never logged); t in minutes."""
    tail = steps[-last:]
    out = {}
    for _, k, _ in TRAIN_COLS:
        xs = [s[k] for s in tail if isinstance(s.get(k), (int, float))]
        out[k] = sum(xs) / len(xs) if xs else None
    if steps and isinstance(steps[-1].get("t"), (int, float)):
        out["t"] = steps[-1]["t"] / 60
    out["steps"] = steps[-1]["step"] if steps else 0
    return out


def gates_by_cond(acc_results: dict[str, dict], base: dict, delta: float) -> dict[tuple, dict]:
    """{canon scales: {"spec", "summary", "gate"}} for one task."""
    out = {}
    for spec, res in acc_results.items():
        out[canon(acc.scales_of(spec))] = {"spec": spec, "summary": res["summary"],
                                           "gate": acc.gate(base, res, delta)}
    return out


def run_gates(accs: dict[str, dict], bases: dict[str, dict], delta: float) -> dict:
    return {t: gates_by_cond(accs[t], bases[t], delta) for t in ("supergpqa", "gsm8k") if t in accs and t in bases}


def run_report(name: str, steps: dict[str, list], ev: Optional[dict], accs: dict[str, dict],
               bases: dict[str, dict], delta: float) -> str:
    """steps {adapter: step entries}; ev eval.json or None; accs {task: {spec: result}}; bases {task: result}."""
    L = [f"=== run {name}"]
    if steps:
        L.append(f"training, mean of the last 10 steps:")
        L.append(f"  {'adapter':<18}{'steps':>6}" + "".join(f"{h:>8}" for h, _, _ in TRAIN_COLS))
        for a, st in steps.items():
            t = train_tail(st)
            L.append(f"  {a:<18}{t['steps']:>6}" + "".join(_fmt(t[k], f, 8) for _, k, f in TRAIN_COLS))
    gates = run_gates(accs, bases, delta)
    tasks = list(gates)
    traits = list(next(iter(ev["persona"].values()))["expression"]) if ev and ev.get("persona") else []
    rows: dict[tuple, str] = {}
    if ev:
        for c, sc in ev["conditions"].items():
            rows.setdefault(canon(sc), c)
    for t in tasks:
        for k, g in gates[t].items():
            rows.setdefault(k, g["spec"])
    head = f"  {'condition':<40}" + "".join(f"{t[:10]:>11}" for t in traits) + f"{'coh':>7}{'KL':>8}"
    for t in tasks:
        head += f"{t[:9] + ' acc':>15}{'diff±se':>13}{'gate':>6}"
    L.append("per condition (persona: judge 0-100; accuracy: paired with base, gate = low95 > -%g):" % delta)
    L.append(head)
    base_acc = {t: bases[t]["summary"]["accuracy"] for t in tasks}
    for k, label in sorted(rows.items(), key=lambda kv: (len(kv[0]), kv[1])):
        p = (ev or {}).get("persona", {}).get(_name_of(ev, k)) if ev else None
        line = f"  {label:<40}"
        for tr in traits:
            line += _fmt(p["expression"][tr]["mean"] if p else None, ".1f", 11)
        line += _fmt(p["coherence"]["mean"] if p else None, ".1f", 7)
        kl = (ev or {}).get("kl", {}).get(_name_of(ev, k)) if ev else None
        line += _fmt(kl["mean"] if kl else None, ".4f", 8)
        for t in tasks:
            if not k:
                line += f"{base_acc[t]:>15.3f}{'':>13}{'':>6}"
                continue
            g = gates[t].get(k)
            if g is None:
                line += f"{'-':>15}{'':>13}{'':>6}"
            else:
                d = g["gate"]
                line += (f"{g['summary']['accuracy']:>15.3f}{d['diff']:>+8.3f}±{d['se']:.3f}"
                         f"{'PASS' if d['pass'] else 'FAIL':>6}")
        L.append(line)
    grid = composition_grid(ev)
    if grid:
        L.append(grid)
    return "\n".join(L)


def _name_of(ev: Optional[dict], key: tuple) -> Optional[str]:
    if not ev:
        return None
    for c, sc in ev["conditions"].items():
        if canon(sc) == key:
            return c
    return None


def composition_grid(ev: Optional[dict]) -> str:
    """For two adapters on a scale grid: each trait's expression and coherence over (scale a, scale b)."""
    if not ev or not ev.get("persona"):
        return ""
    names = sorted({k for sc in ev["conditions"].values() for k in sc})
    if len(names) != 2:
        return ""
    a, b = names
    pts = {}
    for c, sc in ev["conditions"].items():
        pts[(sc.get(a, 0.0), sc.get(b, 0.0))] = ev["persona"].get(c)
    sa = sorted({x for x, _ in pts})
    sb = sorted({y for _, y in pts})
    if len(sa) < 3 or len(sb) < 3:
        return ""
    traits = list(next(iter(ev["persona"].values()))["expression"])
    out = [f"composition grid: rows = {a} scale, columns = {b} scale"]
    for metric in [*traits, "coherence"]:
        out.append(f"  {metric:<12}" + "".join(f"{y:>8g}" for y in sb))
        for x in sa:
            cells = []
            for y in sb:
                p = pts.get((x, y))
                v = None if p is None else (p["coherence"]["mean"] if metric == "coherence"
                                            else p["expression"][metric]["mean"])
                cells.append(_fmt(v, ".1f", 8))
            out.append(f"  {x:<12g}" + "".join(cells))
    return "\n".join(out)


def load_run(root: Path) -> tuple[dict, Optional[dict], dict]:
    """(steps per adapter, eval.json or None, {task: {spec: accuracy result}}) of one run directory."""
    steps = {}
    for d in sorted(p.parent for p in root.glob("*/train.json")):
        f = d / "steps.jsonl"
        if f.exists():
            steps[d.name] = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    ev = json.loads((root / "eval.json").read_text(encoding="utf-8")) if (root / "eval.json").exists() else None
    accs: dict[str, dict] = {}
    for f in sorted((root / "acc").glob("*/*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        accs.setdefault(f.parent.name, {})[r["spec"]] = r
    return steps, ev, accs


def summary(results: dict[str, tuple]) -> str:
    """One line per (run, key condition): own-trait gain over base, the other trait, coherence, gates.

    results: {run: (ev, {task: gates_by_cond(...)})}. Key conditions: each adapter at 1, half, sum."""
    L = ["=== summary: gain = expression - base expression of the adapter's own trait(s)",
         f"  {'run':<9}{'condition':<38}{'gain':>14}{'coh':>7}{'supergpqa':>11}{'gsm8k':>8}"]
    for run, (ev, gates) in results.items():
        if not ev or not ev.get("persona"):
            L.append(f"  {run:<9}(no persona eval yet)")
            continue
        names = sorted({k for sc in ev["conditions"].values() for k in sc})
        base = ev["persona"][_name_of(ev, ())]
        trait_of = {n: n.rsplit("_", 1)[0] for n in names}
        keys = [((n, 1.0),) for n in names]
        if len(names) == 2:
            keys += [tuple((n, 0.5) for n in names), tuple((n, 1.0) for n in names)]
        for k in keys:
            c = _name_of(ev, k)
            if c is None:
                continue
            p = ev["persona"][c]
            gains = "/".join(f"{p['expression'][trait_of[n]]['mean'] - base['expression'][trait_of[n]]['mean']:+.0f}"
                             for n, _ in k if trait_of[n] in p["expression"])
            verdict = []
            for t in ("supergpqa", "gsm8k"):
                g = gates.get(t, {}).get(k)
                verdict.append("-" if g is None else ("PASS" if g["gate"]["pass"] else "FAIL"))
            L.append(f"  {run:<9}{c:<38}{gains:>14}{p['coherence']['mean']:>7.1f}{verdict[0]:>11}{verdict[1]:>8}")
    return "\n".join(L)
