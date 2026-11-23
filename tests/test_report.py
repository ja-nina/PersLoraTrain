"""The report over a made-up out tree (`fake_out` also renders the README's preview)."""
import json
import random
from pathlib import Path

from perslora import cli
from perslora.evaluate import conditions

A, B = "assertive_pos", "sarcastic_pos"
SETTINGS = {"n": 40, "max_tokens": 4096, "seed": 1, "sampling": {"temperature": 0.7}, "model": "qwen3.5-2b"}


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def _acc(task, spec, p, n, rng, base_correct=None, drop=0.0):
    if base_correct is None:
        correct = [rng.random() < p for _ in range(n)]
    else:  # paired with base: each answer flips with a small chance, more often right -> wrong when drop > 0
        correct = [(c and rng.random() > 0.08 + drop) or (not c and rng.random() < 0.08) for c in base_correct]
    return {"spec": spec, "settings": {"task": task, **SETTINGS},
            "summary": {"n": n, "accuracy": sum(correct) / n, "unparsed": 0.01, "cut": 0.0, "tokens_mean": 600},
            "per_question": {"qid": [f"{task}{i}" for i in range(n)], "correct": correct}}


def fake_out(out: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    n = SETTINGS["n"]
    base = {t: _acc(t, "base", p, n, rng) for t, p in (("supergpqa", 0.2), ("gsm8k", 0.65))}
    for t, r in base.items():
        _write(out / "lora" / "_base" / "acc" / t / "base.json", r)
    for run, drop, push in (("kl", 0.10, 1.0), ("correct", 0.0, 0.8)):
        root = out / "lora" / run
        for ad, base_trait in ((A, 48), (B, 9)):
            steps = []
            for s in range(1, 151):
                f = s / 150
                steps.append({"step": s, "reward_target_mean": -10 + 95 * f * push, "trait_mean": base_trait +
                              (95 - base_trait) * f * push, "coherence_mean": 88 - 3 * f, "on_task_mean": 85,
                              "leak_mean": None, "kl": 0.004, "kl_onpolicy": 0.003 if run != "kl" else None,
                              "reason_acc": 0.7 if run != "kl" else None, "reason_cut": 0.03 if run != "kl" else None,
                              "delta_norm": 2.5 * f, "t": 60.0 * s})
            _write(root / ad / "train.json", {"config": {"trait": ad.split("_")[0]}})
            (root / ad / "steps.jsonl").write_text("".join(json.dumps(x) + "\n" for x in steps))
        conds = conditions([A, B], [0, 0.5, 1])

        def persona(sc):
            a, b = sc.get(A, 0), sc.get(B, 0)
            return {"expression": {"assertive": {"mean": 48 + 45 * a * push + 8 * b},
                                   "sarcastic": {"mean": 9 + 50 * b * push + 15 * a}},
                    "coherence": {"mean": 86 - 25 * max(0, a + b - 1) - 4 * (a + b)}}
        _write(root / "eval.json", {"conditions": conds, "persona": {c: persona(sc) for c, sc in conds.items()},
                                    "kl": {c: {"mean": 0.004 * sum(sc.values())} for c, sc in conds.items()}})
        for t in ("supergpqa", "gsm8k"):
            bc = base[t]["per_question"]["correct"]
            for spec in (A, B, f"{A}@step_75", f"{A}:0.5+{B}:0.5", f"{A}+{B}"):
                d = drop * (2 if "+" in spec and ":" not in spec else 1)
                _write(root / "acc" / t / f"{spec.replace(':', 'x')}.json", _acc(t, spec, 0, n, rng, bc, d))


def test_report_end_to_end(tmp_path, capsys):
    fake_out(tmp_path)
    assert cli.main(["report", "--out", str(tmp_path), "--runs", "kl", "correct", "nope"]) == 0
    out = capsys.readouterr().out
    assert "=== run kl" in out and "=== run correct" in out and "nope" in out and "missing" in out
    assert "composition grid" in out and "PASS" in out
    # the eval's `sum` and the accuracy spec A+B are one row
    assert "assertive_pos:1+sarcastic_pos:1" in out and f"{A}+{B}" not in out.replace(f"{A}:1+{B}:1", "")
    assert f"{A}@step_75" in out
    assert "=== summary" in out and "assertive_pos:0.5+sarcastic_pos:0.5" in out.split("=== summary")[1]


def test_report_wandb_table(tmp_path, monkeypatch, capsys):
    fake_out(tmp_path)
    logged = {}

    class FakeWandb:
        def __init__(self, project, entity, name, config, group):
            self.run = type("R", (), {"summary": {}})()
            logged["name"] = name
            logged["summary"] = self.run.summary

        def log(self, entry, step=None):
            logged.update(entry)

        def finish(self):
            pass

    import sys, types
    monkeypatch.setattr(cli, "Wandb", FakeWandb)
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Table=lambda columns, data: (columns, data)))
    assert cli.main(["report", "--out", str(tmp_path), "--runs", "kl", "--wandb-project", "p"]) == 0
    cols, data = logged["accuracy"]
    assert cols[:3] == ["run", "task", "condition"] and len(data) == 2 * (1 + 5)
    assert "kl/gsm8k/assertive_pos/pass" in logged["summary"]


def test_additivity_math():
    from perslora.report import additivity
    conds = {"base": {}, "a": {"a": 1.0}, "b": {"b": 1.0}, "sum": {"a": 1.0, "b": 1.0}, "ha": {"a": 0.5}}
    e = lambda m, se=1.0: {"expression": {"t": {"mean": m, "se": se}}, "coherence": {"mean": 80, "se": 0}}  # noqa
    ev = {"conditions": conds, "persona": {"base": e(10), "a": e(50), "b": e(30), "sum": e(60), "ha": e(40)}}
    add = additivity(ev)
    v, se = add["interaction"]["t"][1.0]
    assert v == 60 - 50 - 30 + 10 and se == 2.0  # sqrt(4 * 1^2)
    assert add["interaction"]["coherence"][1.0] == (0.0, 0.0)
    assert 0.5 not in add["interaction"]["t"]  # no (0.5, 0.5) point
    assert add["nonlinearity"]["t"]["a"][0] == 40 - (10 + 50) / 2
    assert "b" not in add["nonlinearity"]["t"]  # no b at 0.5


def test_report_shows_additivity(tmp_path, capsys):
    fake_out(tmp_path)
    assert cli.main(["report", "--out", str(tmp_path), "--runs", "kl"]) == 0
    out = capsys.readouterr().out
    assert "additivity:" in out and "interaction at (1, 1)" in out and "nonlinearity sarcastic_pos" in out
    assert "interaction (0.5, 0.5)" in out.split("=== summary")[1]
