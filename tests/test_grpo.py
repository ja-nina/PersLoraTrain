import copy
import math

import torch

from perslora import grpo, lora
from persvectrain.train import RefAnswer

QS = [("q1", "hello"), ("q2", "what now"), ("q3", "tell me")]
REFS = [RefAnswer("r1", "2+2?", "four is it", "A"), RefAnswer("r2", "sky?", "blue mostly", "B")]


def fake_judge(rollouts):
    return [{"trait": min(100, 12 * r.text.count("a")), "opposite": 0, "coherence": 90, "on_task": 90,
             "denial": False} for r in rollouts]


def cfg(**kw):
    base = dict(model="qwen3.5-2b", trait="x", steps=3, questions_per_step=2, g=4, max_new_tokens=6, lr=1e-3,
                kl_batch=1, kl_tokens=8, sample_every=2)
    base.update(kw)
    return grpo.GRPOConfig(**base)


def run(model, tok, c, refs=REFS):
    logs, samples, rolls = [], [], []
    grpo.train_adapter(model, tok, c, QS, refs, fake_judge, log_step=logs.append,
                       on_samples=lambda s, xs: samples.append((s, xs)),
                       on_rollouts=lambda s, xs: rolls.extend(xs),
                       sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0},
                       prompt_fn=lambda q: f"<u>{q}<a>")
    return logs, samples, rolls


def test_train_runs_and_logs(tiny_model, char_tok):
    logs, samples, rolls = run(tiny_model, char_tok, cfg())
    assert len(logs) == 3
    for e in logs:
        assert set(grpo.LOG_KEYS) <= set(e)
        assert all(math.isfinite(e[k]) for k in grpo.LOG_KEYS if isinstance(e[k], float))
    assert [s for s, _ in samples] == [2]
    assert len(rolls) == 3 * 2 * 4 and {"step", "qid", "text", "vector", "advantage"} <= set(rolls[0])


def test_reward_goes_up(tiny_model, char_tok):
    logs, _, _ = run(tiny_model, char_tok, cfg(steps=40, lr=2e-2, g=8, lambda_kl=0.0), refs=[])
    first = sum(e["trait_mean"] for e in logs[:5]) / 5
    last = sum(e["trait_mean"] for e in logs[-5:]) / 5
    assert last > first


def test_fixed_lambda_kl_skips_budget(tiny_model, char_tok):
    logs, _, _ = run(tiny_model, char_tok, cfg(lambda_kl=0.7))
    assert [e["lam_kl"] for e in logs] == [0.7] * 3


def test_only_adapter_params_change(tiny_model, char_tok):
    before = copy.deepcopy(tiny_model.state_dict())
    run(tiny_model, char_tok, cfg())
    after = tiny_model.state_dict()
    for k, v in before.items():
        k2 = k if k in after else k.replace(".weight", ".base.weight")
        assert torch.equal(v, after[k2]), k
    assert lora.delta_norm(tiny_model, "x_pos") > 0


def test_adapter_scale_reset_after_training(tiny_model, char_tok):
    run(tiny_model, char_tok, cfg(steps=1))
    mods = lora.wrapped_modules(tiny_model).values()
    assert all(m.scales["x_pos"] == 0.0 for m in mods)
