import json

import pytest

from perslora import cli, lora
from persvectrain.train import RefAnswer
from tests.conftest import CharTokenizer, make_tiny_model

TRAIT = {"trait": "Assertive", "description": "Confident.", "instruction": {"negative": "Hesitant."}}


def fake_judge_factory(args):
    def judge(rollouts, trait):
        return [{"trait": 50, "opposite": 10, "coherence": 90, "on_task": 90, "denial": False} for _ in rollouts]
    return judge


@pytest.fixture
def patched(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "load_model", lambda key: (make_tiny_model(), CharTokenizer()))
    monkeypatch.setattr(cli, "make_judge", fake_judge_factory)
    monkeypatch.setattr(cli.data, "trait_questions",
                        lambda src, slug: ([(f"{slug}:0", "hi"), (f"{slug}:1", "yo")], [(f"{slug}:2", "hey")]))
    monkeypatch.setattr(cli.data, "trait_info", lambda slug: dict(TRAIT, trait=slug))
    refs = [RefAnswer("r1", "q?", "an answer", "A")]
    monkeypatch.setattr(cli.data, "kl_reference", lambda out, model: (refs, refs, tmp_path / "ref.jsonl"))
    monkeypatch.setattr(cli, "prompt_for", lambda tok, model: (lambda q: f"<u>{q}<a>"))
    return tmp_path


ARGS = ["train", "--model", "qwen3.5-2b", "--source", "S", "--run-name", "r", "--traits", "Assertive",
        "Sarcastic", "--steps", "2", "--questions-per-step", "2", "--g", "2", "--max-new-tokens", "4",
        "--kl-tokens", "8", "--kl-batch", "1"]


def test_parser_defaults():
    a = cli.build_parser().parse_args(["train", "--model", "qwen3.5-2b", "--source", "S", "--out", "O",
                                       "--run-name", "r", "--traits", "Assertive"])
    assert a.direction == "pos" and a.steps == 150 and a.g == 8 and a.eps_kl == 0.03 and a.lambda_kl is None
    assert a.wandb_project is None


def test_train_cli_end_to_end(patched):
    out = patched
    assert cli.main(ARGS + ["--out", str(out)]) == 0
    for slug in ("assertive", "sarcastic"):
        d = out / "lora" / "r" / f"{slug}_pos"
        for f in ("adapter.safetensors", "adapter.json", "train.json", "steps.jsonl", "rollouts.jsonl.gz"):
            assert (d / f).exists(), f
        assert len((d / "steps.jsonl").read_text().splitlines()) == 2
        meta = json.loads((d / "train.json").read_text())
        assert meta["config"]["trait"] == slug and "commits" in meta
        m = make_tiny_model()
        lora.load_adapter(m, f"{slug}_pos", d)


def test_existing_adapter_dir_refuses(patched):
    out = patched
    (out / "lora" / "r" / "assertive_pos").mkdir(parents=True)
    with pytest.raises(SystemExit):
        cli.main(ARGS + ["--out", str(out)])
    assert cli.main(ARGS + ["--out", str(out), "--overwrite"]) == 0


def test_evaluate_cli_end_to_end(patched, monkeypatch):
    out = patched
    assert cli.main(ARGS + ["--out", str(out)]) == 0
    monkeypatch.setattr(cli, "make_score_batch", lambda args: (lambda asks: [50 for _ in asks]))
    import perslora.evaluate as ev
    pool = {"r1": 4}
    monkeypatch.setattr("persvectrain.freegen.pool_options", lambda path: pool)
    assert cli.main(["evaluate", "--out", str(out), "--run-name", "r", "--samples", "1", "--persona-tokens", "3",
                     "--acc-n", "1", "--acc-tokens", "3", "--kl-tokens", "8"]) == 0
    res = json.loads((out / "lora" / "r" / "eval.json").read_text())
    assert set(res["kl"]) == {"base", "assertive_pos", "sarcastic_pos", "sum", "half"}
    assert res["persona"]["sum"]["expression"]["assertive"]["mean"] == 50
    assert set(res["accuracy"]) == set(res["kl"])


def test_crash_saves_partial_adapter(patched, monkeypatch):
    out = patched
    from persvecfast.judge import JudgeError
    calls = {"n": 0}

    def factory(args):
        def judge(rollouts, trait):
            calls["n"] += 1
            if calls["n"] == 2:
                raise JudgeError("no credit")
            return [{"trait": 50, "opposite": 10, "coherence": 90, "on_task": 90, "denial": False} for _ in rollouts]
        return judge

    monkeypatch.setattr(cli, "make_judge", factory)
    with pytest.raises(JudgeError):
        cli.main(ARGS + ["--out", str(out)])
    d = out / "lora" / "r" / "assertive_pos"
    assert (d / "adapter.safetensors").exists()
    meta = json.loads((d / "train.json").read_text())
    assert meta["status"] == "crashed" and meta["last_step"] == 1


class ImEndTok(CharTokenizer):
    def convert_tokens_to_ids(self, t):
        return 0  # the char tokenizer's EOS stands in for <|im_end|>


def test_gpucheck_tiny(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_model", lambda key: (make_tiny_model(), ImEndTok()))
    monkeypatch.setattr(cli, "prompt_for", lambda tok, model: (lambda q: f"<u>{q}<a>"))
    assert cli.main(["gpucheck", "--max-new-tokens", "5"]) == 0
    assert "GPUCHECK OK" in capsys.readouterr().out


def TinyTask(ids=("p1", "p2")):
    """Stands in for tasks.load_task: 'correct' (answer A) when the answer contains an 'a'."""
    from perslora.tasks import Task
    return Task("t", [(i, f"question {i}") for i in ids], {i: "A" for i in ids},
                lambda qid, text: "A" if "a" in text else None)


def test_train_with_leak_reason_and_checkpoints(patched, monkeypatch):
    out = patched
    seen = {}

    def factory(args):
        def judge(rollouts, trait, others=None):
            seen[trait["trait"]] = sorted(others or {})
            return [{"trait": 50, "opposite": 10, "coherence": 90, "on_task": 90, "denial": False,
                     "others": {k: 20 for k in (others or {})}} for _ in rollouts]
        return judge

    monkeypatch.setattr(cli, "make_judge", factory)
    monkeypatch.setattr("perslora.tasks.load_task", lambda name, out, n=None: TinyTask())
    args = [a if a != "2" or i != ARGS.index("--steps") + 1 else "3" for i, a in enumerate(ARGS)]
    assert cli.main(args + ["--out", str(out), "--c-leak", "1", "--reason-questions", "1", "--reason-g", "2",
                            "--reason-tokens", "3", "--w-correct", "0.5", "--lambda-onpolicy", "0.1",
                            "--save-every", "2"]) == 0
    assert seen == {"assertive": ["sarcastic"], "sarcastic": ["assertive"]}
    d = out / "lora" / "r" / "assertive_pos"
    assert (d / "ckpt" / "step_2" / "adapter.json").exists()
    steps = [json.loads(l) for l in (d / "steps.jsonl").read_text().splitlines()]
    assert len(steps) == 3 and steps[0]["leak_mean"] == 20 and steps[0]["reason_acc"] is not None
    meta = json.loads((d / "train.json").read_text())
    assert meta["leak_traits"] == ["sarcastic"] and meta["reason_task"] == "gsm8k-train"


def test_train_keep_from_skips_finished_adapters(patched, monkeypatch):
    out = patched
    assert cli.main(ARGS + ["--out", str(out)]) == 0
    a, b = (out / "lora" / "r" / f"{s}_pos" for s in ("assertive", "sarcastic"))
    meta = json.loads((b / "train.json").read_text())
    (b / "train.json").write_text(json.dumps({**meta, "status": "crashed", "last_step": 1}))
    before = (a / "steps.jsonl").read_text()
    trained = []
    real = cli.train_adapter
    monkeypatch.setattr(cli, "train_adapter", lambda model, tok, cfg, *x, **k: (trained.append(cfg.trait),
                                                                                  real(model, tok, cfg, *x, **k)))
    assert cli.main(ARGS + ["--out", str(out), "--keep-from", "2"]) == 0  # no --overwrite needed
    assert trained == ["sarcastic"]  # assertive (done at step 2) is kept, sarcastic (crashed at 1) is redone
    assert (a / "steps.jsonl").read_text() == before
    assert json.loads((b / "train.json").read_text())["status"] == "done"
    assert len((b / "steps.jsonl").read_text().splitlines()) == 2  # from scratch, not appended


def test_accuracy_and_gate_cli(patched, monkeypatch, capsys):
    out = patched
    args = [a if a != "2" or i != ARGS.index("--steps") + 1 else "3" for i, a in enumerate(ARGS)]
    assert cli.main(args + ["--out", str(out), "--save-every", "2"]) == 0
    monkeypatch.setattr("perslora.tasks.load_task", lambda name, out, n=None: TinyTask())
    common = ["accuracy", "--out", str(out), "--run-name", "r", "--task", "gsm8k", "--max-tokens", "3"]
    assert cli.main(common + ["--cond", "base"]) == 0
    assert cli.main(common + ["--cond", "assertive_pos", "assertive_pos@step_2:0.5+sarcastic_pos:0.5"]) == 0
    base = json.loads((out / "lora" / "_base" / "acc" / "gsm8k" / "base.json").read_text())
    assert base["summary"]["n"] == 2 and base["per_question"]["qid"] == ["p1", "p2"]
    files = sorted(p.name for p in (out / "lora" / "r" / "acc" / "gsm8k").glob("*.json"))
    assert files == ["assertive_pos.json", "assertive_pos@step_2x0.5+sarcastic_posx0.5.json"]
    assert cli.main(common + ["--cond", "assertive_pos"]) == 0  # exists: skipped, no model load needed
    capsys.readouterr()
    assert cli.main(["gate", "--out", str(out), "--run-name", "r", "--task", "gsm8k", "supergpqa"]) == 0
    printed = capsys.readouterr().out
    assert "assertive_pos@step_2:0.5+sarcastic_pos:0.5" in printed and ("PASS" in printed or "FAIL" in printed)
    g = json.loads((out / "lora" / "r" / "gate.json").read_text())
    assert set(g) == {"gsm8k"} and set(g["gsm8k"]["conditions"]) == {"assertive_pos",
                                                                    "assertive_pos@step_2:0.5+sarcastic_pos:0.5"}


def test_accuracy_unknown_checkpoint(patched, monkeypatch):
    out = patched
    assert cli.main(ARGS + ["--out", str(out)]) == 0
    monkeypatch.setattr("perslora.tasks.load_task", lambda name, out, n=None: TinyTask())
    with pytest.raises(SystemExit):
        cli.main(["accuracy", "--out", str(out), "--run-name", "r", "--task", "gsm8k", "--cond",
                  "assertive_pos@step_99"])


def test_reference_cli(patched, monkeypatch):
    out = patched
    monkeypatch.setattr("perslora.tasks.load_task", lambda name, out, n=None: TinyTask(("p1", "r1", "p2")))
    assert cli.main(["reference", "--out", str(out), "--task", "supergpqa-pool", "--samples", "3",
                     "--max-tokens", "4"]) == 0
    p = out / "lora" / "refs" / "supergpqa-pool.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert all(not r["id"].startswith("r1#") for r in rows)  # the KL eval's held question is left out
    meta = json.loads(p.with_suffix(".meta.json").read_text())
    assert meta["questions"] == 2 and meta["kept"] == len(rows)
    from persvectrain.output import load_reference
    assert len(load_reference(p)) == len(rows)


def test_train_in_training_evals(patched, monkeypatch):
    out = patched
    monkeypatch.setattr(cli, "training_evals", lambda args: {"gsm8k": TinyTask(), "supergpqa": TinyTask(("s1",))})
    args = [a if a != "2" or i != ARGS.index("--steps") + 1 else "3" for i, a in enumerate(ARGS)]
    assert cli.main(args + ["--out", str(out), "--eval-every", "2", "--eval-tokens", "3"]) == 0
    d = out / "lora" / "r" / "assertive_pos"
    ev = [json.loads(l) for l in (d / "evals.jsonl").read_text().splitlines()]
    assert [e["step"] for e in ev] == [0, 2, 3]  # start, every 2, and the last step
    assert 0 <= ev[0]["eval/gsm8k_accuracy"] <= 1 and "eval/supergpqa_cut" in ev[0]
    assert ev[0]["per_question"]["supergpqa"]["qid"] == ["s1"]


def test_training_evals_sets(patched, monkeypatch):
    from perslora.tasks import Task
    pool = Task("supergpqa", [("r1", "q1"), ("x", "qx")], {"r1": "A", "x": "B"}, lambda q, t: None)
    gsm = Task("gsm8k", [(f"g{i}", "q") for i in range(510)], {f"g{i}": "1" for i in range(510)}, lambda q, t: None)
    monkeypatch.setattr("perslora.tasks.load_task", lambda name, out, n=None: pool if "supergpqa" in name else gsm)
    a = cli.build_parser().parse_args(["train", "--source", "S", "--out", str(patched), "--run-name", "r",
                                       "--traits", "Assertive", "--eval-every", "25", "--eval-n", "8",
                                       "--eval-tasks", "gsm8k", "supergpqa"])
    ev = cli.training_evals(a)
    assert [i for i, _ in ev["supergpqa"].items] == ["r1"]  # only the KL eval's held-out pool question
    assert [i for i, _ in ev["gsm8k"].items] == [f"g{i}" for i in range(500, 508)]  # after the final test's 500
