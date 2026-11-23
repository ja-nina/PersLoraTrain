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
