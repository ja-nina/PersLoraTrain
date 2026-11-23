import torch

from perslora import rollout
from persvectrain.steering import answer_logprobs


def prompt_fn(q):
    return f"<u>{q}<a>"


def test_generate_groups(tiny_model, char_tok):
    rs = rollout.generate(tiny_model, char_tok, [("a", "hi"), ("b", "yo there")], prompt_fn, g=3,
                          max_new_tokens=7, sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0}, seed=0)
    assert [r.qid for r in rs] == ["a"] * 3 + ["b"] * 3
    assert all(len(r.answer_ids) <= 7 for r in rs)
    assert rs[3].prompt_ids == char_tok(prompt_fn("yo there"))["input_ids"]
    assert all(0 not in r.answer_ids[:-1] for r in rs)  # cut at the first EOS (kept as the last token)


def test_generate_seeded(tiny_model, char_tok):
    kw = dict(prompt_fn=prompt_fn, g=2, max_new_tokens=5, sampling={"temperature": 1.0}, seed=3)
    a = rollout.generate(tiny_model, char_tok, [("a", "hi")], **kw)
    b = rollout.generate(tiny_model, char_tok, [("a", "hi")], **kw)
    assert [r.answer_ids for r in a] == [r.answer_ids for r in b]


def test_seq_logprob_matches_manual(tiny_model, char_tok):
    r = rollout.Rollout("a", "hi", [5, 6, 7], [9, 10, 11], "xyz")
    ids = torch.tensor([[5, 6, 7, 9, 10, 11]])
    lp = answer_logprobs(tiny_model, ids, 3)
    want = lp.gather(1, torch.tensor([[9], [10], [11]])).sum()
    assert torch.allclose(rollout.seq_logprob(tiny_model, r), want, atol=1e-5)


def test_empty_answer_zero(tiny_model):
    r = rollout.Rollout("a", "hi", [5, 6], [], "")
    v = rollout.seq_logprob(tiny_model, r)
    assert float(v) == 0.0 and not v.requires_grad


def test_sampling_from_spec():
    s = rollout.sampling_from_spec("qwen3.5-2b")
    assert set(s) == {"temperature", "top_p", "top_k"} and s["temperature"] > 0


def test_seq_logprob_temperature(tiny_model):
    r = rollout.Rollout("a", "hi", [5, 6, 7], [9, 10], "xy")
    ids = torch.tensor([[5, 6, 7, 9, 10]])
    with torch.no_grad():
        logits = tiny_model(ids).logits[0, 2:4].float() / 0.5
    want = torch.log_softmax(logits, -1).gather(1, torch.tensor([[9], [10]])).sum()
    assert torch.allclose(rollout.seq_logprob(tiny_model, r, temperature=0.5), want, atol=1e-5)
    assert torch.allclose(rollout.seq_logprob(tiny_model, r), rollout.seq_logprob(tiny_model, r, 1.0))


class ImTok:
    eos_token_id = 7

    def convert_tokens_to_ids(self, t):
        return 9 if t == "<|im_end|>" else None


def test_eos_ids_union(tiny_model):
    tiny_model.generation_config.eos_token_id = 3
    assert rollout._eos_ids(tiny_model, ImTok()) == {3, 7, 9}
