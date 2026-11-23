import json
import math

import pytest
import torch

from perslora import accuracy as acc
from perslora import evaluate as ev
from perslora import grpo, rollout, tasks
from perslora.reward import group_advantages, judge_prompt, parse_vector
from perslora.rollout import Rollout

TRAIT = {"trait": "Assertive", "description": "Confident.", "instruction": {"negative": "Hesitant."}}


# ---------------------------------------------------------------- sampling


def test_presence_penalty_only_generated_tokens():
    pp = rollout.PresencePenalty(1.5, prompt_width=2)
    ids = torch.tensor([[5, 6, 7, 7], [5, 6, 8, 0]])
    out = pp(ids, torch.zeros(2, 10))
    assert out[0, 7] == -1.5 and out[0, 5] == 0 and out[0, 6] == 0  # prompt tokens untouched, repeats once
    assert out[1, 8] == -1.5 and out[1, 0] == -1.5 and out[1, 7] == 0
    assert torch.equal(pp(ids[:, :2], torch.zeros(2, 10)), torch.zeros(2, 10))  # nothing generated yet


def test_generate_with_presence_and_cut_flag(tiny_model, char_tok):
    rs = rollout.generate(tiny_model, char_tok, [("a", "hi")], lambda q: f"<u>{q}<a>", 3, 4,
                          {"temperature": 1.0, "presence_penalty": 1.5}, seed=0)
    for r in rs:
        assert r.stopped == (bool(r.answer_ids) and r.answer_ids[-1] == 0)
        assert r.stopped or len(r.answer_ids) == 4


def test_sampling_presence_flag():
    assert "presence_penalty" not in rollout.sampling_from_spec("qwen3.5-2b")
    assert rollout.sampling_from_spec("qwen3.5-2b", presence=True)["presence_penalty"] == 1.5


# ---------------------------------------------------------------- tasks


@pytest.mark.parametrize("text,want", [("so ANSWER: 42", "42"), ("ANSWER: 1,234.", "1234"), ("Answer = -3.5", "-3.5"),
                                       ("ANSWER: $18", "18"), ("ANSWER: 4 then ANSWER: 5", "5"), ("no answer", None),
                                       ("**ANSWER:** 7", "7")])
def test_parse_number(text, want):
    assert tasks.parse_number(text) == want


def test_gsm8k_rows_and_task(tmp_path):
    rows = tasks.gsm8k_rows([{"question": "2+2?", "answer": "work\n#### 1,004"}], "test")
    assert rows == [{"id": "gsm8k_test_0", "question": "2+2?", "answer": "1004"}]
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps(rows[0]) + "\n")
    t = tasks.gsm8k(p)
    assert t.items[0][1].startswith("2+2?") and "ANSWER: <number>" in t.items[0][1]
    assert t.score("gsm8k_test_0", "ANSWER: 1004.0") == ("1004.0", True)
    assert t.score("gsm8k_test_0", "ANSWER: 5")[1] is False
    assert t.score("gsm8k_test_0", "dunno") == (None, False)


def test_supergpqa_task(tmp_path):
    p = tmp_path / "pool.jsonl"
    p.write_text(json.dumps({"id": "q1", "question": "Which?", "options": ["x", "y", "z"], "answer": "B"}) + "\n")
    t = tasks.supergpqa(p)
    assert "C." in t.items[0][1] and "one of A, B, C" in t.items[0][1]
    assert t.score("q1", "ANSWER: B") == ("B", True)
    assert t.score("q1", "ANSWER: D") == (None, False)  # not an option letter
    assert t.head(1).items == t.items


# ---------------------------------------------------------------- conditions, scoring, gate


def test_parse_cond_and_label():
    assert acc.parse_cond("base") == []
    assert acc.parse_cond("a_pos") == [("a_pos", None, 1.0)]
    assert acc.parse_cond("a_pos@step_75:0.5+b_pos") == [("a_pos", "step_75", 0.5), ("b_pos", None, 1.0)]
    assert acc.scales_of("a_pos@step_75:0.5+b_pos") == {"a_pos@step_75": 0.5, "b_pos": 1.0}
    assert ":" not in acc.cond_label("a_pos:0.5+b_pos:0.5")
    with pytest.raises(ValueError):
        acc.parse_cond("a pos")


class FakeTask:
    name = "t"
    items = [("q1", "Q1?"), ("q2", "Q2?")]
    gold = {"q1": "A", "q2": "B"}

    def score(self, qid, text):
        a = text[-1] if text and text[-1] in "AB" else None
        return a, a == self.gold[qid]


def R(qid, text, stopped=True):
    return Rollout(qid, "", [1], [2] * len(text), text, stopped=stopped)


def test_score_answers_summary():
    res = acc.score_answers(FakeTask(), [R("q1", "A"), R("q2", "x", stopped=False)])
    assert res["summary"] == {"n": 2, "accuracy": 0.5, "unparsed": 0.5, "cut": 0.5, "tokens_mean": 1.0}
    assert res["per_question"]["correct"] == [True, False]


def _res(correct, qids=None):
    return {"per_question": {"qid": qids or [f"q{i}" for i in range(len(correct))], "correct": correct}}


def test_gate_pass_and_fail():
    base = _res([True] * 50 + [False] * 50)
    same = acc.gate(base, base, 0.03)
    assert same["diff"] == 0 and same["pass"]
    worse = acc.gate(base, _res([False] * 20 + [True] * 30 + [False] * 50), 0.03)  # lost 20 of 50
    assert worse["diff"] == pytest.approx(-0.2) and not worse["pass"]
    assert worse["lower95"] == pytest.approx(worse["diff"] - 1.645 * worse["se"])
    with pytest.raises(ValueError):
        acc.gate(base, _res([True] * 100, qids=[f"x{i}" for i in range(100)]), 0.03)


def test_reference_rows_filters():
    rs = [R("q1", "A"), R("q1", "B"), R("q2", "B", stopped=False), R("q2", "zz")]
    rows = acc.reference_rows(FakeTask(), rs, correct_only=False)
    assert [r["id"] for r in rows] == ["q1#0", "q1#1"]
    assert rows[0]["question"] == "Q1?" and rows[0]["correct"] and not rows[1]["correct"]
    assert [r["id"] for r in acc.reference_rows(FakeTask(), rs, correct_only=True)] == ["q1#0"]


def test_grid_conditions():
    c = ev.conditions(["a", "b"], [0, 0.5, 1])
    assert len(c) == 1 + 8 and c["base"] == {}
    assert c["a:0.5+b:1"] == {"a": 0.5, "b": 1.0} and c["b:1"] == {"b": 1.0}
    assert all(acc.scales_of(k) == v for k, v in c.items())  # names are accuracy specs


# ---------------------------------------------------------------- leak


def test_judge_prompt_and_parse_with_others():
    others = {"sarcastic": {"trait": "Sarcastic", "description": "Mocking."}}
    p = judge_prompt(TRAIT, "q", "r", 1000, others)
    assert "other_sarcastic" in p and "Mocking." in p
    assert "other_" not in judge_prompt(TRAIT, "q", "r", 1000)
    raw = '{"trait": 9, "opposite": 1, "coherence": 80, "on_task": 70, "denial": false, "other_sarcastic": 40}'
    v = parse_vector(raw, ("sarcastic",))
    assert v["others"] == {"sarcastic": 40} and "other_sarcastic" not in v
    assert parse_vector(raw.replace(', "other_sarcastic": 40', ""), ("sarcastic",)) is None


def test_leak_penalty_lowers_leaky_rollout():
    vec = lambda leak: {"trait": 50, "opposite": 0, "coherence": 90, "on_task": 90, "denial": False,  # noqa: E731
                        "others": {"s": leak}}
    vs = [vec(0), vec(80)]
    assert group_advantages(vs, 2, "pos", 0, 0, 0) == [0.0, 0.0]
    a = group_advantages(vs, 2, "pos", 0, 0, 0, c_leak=1.0)
    assert a[0] > 0 > a[1]


# ---------------------------------------------------------------- training


def test_encode_tail(char_tok):
    ids, start = grpo.encode_tail(char_tok, "q", "abcdefghij", 4)
    assert ids.shape[1] - start == 4 and char_tok.decode(ids[0, start:].tolist()) == "ghij"
    ids, start = grpo.encode_tail(char_tok, "q", "ab", 4)
    assert char_tok.decode(ids[0, start:].tolist()) == "ab"


class ReasonTask:
    items = [("r1", "say a"), ("r2", "say b")]

    def score(self, qid, text):
        return None, "a" in text


def test_train_with_reasoning_rollouts(tiny_model, char_tok):
    from tests.test_grpo import QS, REFS, cfg, fake_judge
    logs = []
    c = cfg(steps=2, reason_questions=2, reason_g=2, reason_tokens=5, w_correct=1.0, lambda_onpolicy=0.5,
            onpolicy_batch=3)
    grpo.train_adapter(tiny_model, char_tok, c, QS, REFS, fake_judge, log_step=logs.append,
                       on_samples=lambda s, x: None, on_rollouts=lambda s, x: None,
                       sampling={"temperature": 1.0}, prompt_fn=lambda q: f"<u>{q}<a>", reason=ReasonTask())
    for e in logs:
        assert 0 <= e["reason_acc"] <= 1 and 0 <= e["reason_cut"] <= 1
        assert math.isfinite(e["kl_onpolicy"]) and e["kl_onpolicy"] >= -1e-6
        assert e["leak_mean"] is None


def test_reason_needs_task(tiny_model, char_tok):
    from tests.test_grpo import QS, REFS, cfg, fake_judge
    with pytest.raises(ValueError):
        grpo.train_adapter(tiny_model, char_tok, cfg(reason_questions=1), QS, REFS, fake_judge, log_step=print,
                           on_samples=print, on_rollouts=print, sampling={}, prompt_fn=str)
