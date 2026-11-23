import pytest

from perslora import evaluate, lora
from persvectrain.train import RefAnswer


def test_conditions_two_adapters():
    c = evaluate.conditions(["a_pos", "b_pos"])
    assert c == {"base": {}, "a_pos": {"a_pos": 1.0}, "b_pos": {"b_pos": 1.0},
                 "sum": {"a_pos": 1.0, "b_pos": 1.0}, "half": {"a_pos": 0.5, "b_pos": 0.5}}


def test_conditions_one_adapter():
    assert evaluate.conditions(["a_pos"]) == {"base": {}, "a_pos": {"a_pos": 1.0}}


def test_kl_eval_base_zero(tiny_model, char_tok):
    lora.attach(tiny_model, "a_pos")
    refs = [RefAnswer("r1", "q?", "some answer", "A")]
    out = evaluate.kl_eval(tiny_model, char_tok, {"base": {}, "a_pos": {"a_pos": 1.0}}, refs, kl_tokens=8)
    assert out["base"]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert out["a_pos"]["mean"] == pytest.approx(0.0, abs=1e-6)  # fresh adapter: B = 0


def test_acc_eval_paired():
    refs = [RefAnswer(f"r{i}", f"q{i}", "", "B") for i in range(4)]
    def gen_fn(items, scales):
        return ["ANSWER: B" if not scales else "ANSWER: C" for _ in items]
    out = evaluate.acc_eval({"base": {}, "x": {"x": 1.0}}, refs, {r.id: 4 for r in refs}, gen_fn)
    assert out["base"]["accuracy"] == 1.0 and out["x"]["accuracy"] == 0.0
    assert out["x"]["diff_vs_base"]["mean"] == -1.0 and out["x"]["unparsed"] == 0.0


def test_acc_eval_unparsed_counted():
    refs = [RefAnswer("r0", "q", "", "A")]
    out = evaluate.acc_eval({"base": {}}, refs, {"r0": 4}, lambda items, s: ["no letter"])
    assert out["base"]["unparsed"] == 1.0 and out["base"]["accuracy"] == 0.0


def test_persona_eval_fake_judges():
    held = {"a": [("a:1", "qa")], "b": [("b:1", "qb")]}
    def gen_fn(items, scales):
        return [("A-ish" if "a_pos" in scales else "plain") for _ in items]
    def score_fn(question, response, trait):
        if trait is None:
            return 80
        return 90 if (trait == "a" and response == "A-ish") else 10
    batch = lambda asks: [score_fn(*a) for a in asks]
    out = evaluate.persona_eval({"base": {}, "a_pos": {"a_pos": 1.0}}, held, ["a", "b"], 2, gen_fn, batch)
    assert out["a_pos"]["expression"]["a"]["mean"] == 90 and out["base"]["expression"]["a"]["mean"] == 10
    assert out["a_pos"]["expression"]["b"]["mean"] == 10
    assert out["base"]["coherence"]["mean"] == 80
    assert out["a_pos"]["n"] == 4  # 2 questions x 2 samples


def test_kl_eval_one_reference_at_a_time(tiny_model, char_tok, monkeypatch):
    lora.attach(tiny_model, "a_pos")
    calls = []
    real = evaluate.answer_logprobs

    def spy(model, ids, start):
        mods = lora.wrapped_modules(model).values()
        calls.append(next(iter(mods)).scales.get("a_pos", 0.0))
        return real(model, ids, start)

    monkeypatch.setattr(evaluate, "answer_logprobs", spy)
    refs = [RefAnswer("r1", "q?", "some answer", "A"), RefAnswer("r2", "z?", "other one", "B")]
    evaluate.kl_eval(tiny_model, char_tok, {"base": {}, "a_pos": {"a_pos": 1.0}}, refs, kl_tokens=8)
    assert calls == [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]  # per ref: reference, then each condition
