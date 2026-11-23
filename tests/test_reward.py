import asyncio

import pytest

from perslora import reward
from perslora.rollout import Rollout

V = '{"trait": 70, "opposite": 10, "coherence": 90, "on_task": 85, "denial": false}'


def vec(trait, opposite=0, coh=90, task=90, denial=False):
    return {"trait": trait, "opposite": opposite, "coherence": coh, "on_task": task, "denial": denial}


def test_parse_vector_ok():
    assert reward.parse_vector("Sure:\n" + V + "\n") == vec(70, 10, 90, 85, False)
    assert reward.parse_vector(V.replace("false", '"yes"'))["denial"] is True


@pytest.mark.parametrize("raw", [None, "no json here", V.replace('"on_task": 85, ', ""), V.replace("70", "101"),
                                 V.replace("70", '"high"'), "{not json}"])
def test_parse_vector_rejects(raw):
    assert reward.parse_vector(raw) is None


def test_advantages_constant_group_zero():
    adv = reward.group_advantages([vec(50)] * 4, g=4, direction="pos", lam_coh=1, lam_task=1, c_denial=1)
    assert adv == [0.0] * 4


def test_advantages_direction_flips():
    vs = [vec(10, 80), vec(80, 10)]
    pos = reward.group_advantages(vs, 2, "pos", 0, 0, 0)
    neg = reward.group_advantages(vs, 2, "neg", 0, 0, 0)
    assert pos[1] > 0 > pos[0] and neg[0] > 0 > neg[1]


def test_advantages_none_is_zero_and_excluded():
    vs = [vec(10), None, vec(90), vec(50)]
    adv = reward.group_advantages(vs, 4, "pos", 0, 0, 0)
    assert adv[1] == 0.0 and adv[2] > 0 > adv[0]
    assert abs(sum(adv)) < 1e-6  # z over the three present members


def test_advantages_groups_independent():
    vs = [vec(10), vec(20), vec(80), vec(90)]
    adv = reward.group_advantages(vs, 2, "pos", 0, 0, 0)
    assert adv[0] < 0 < adv[1] and adv[2] < 0 < adv[3]


def test_denial_penalised():
    vs = [vec(50, denial=True), vec(50)]
    adv = reward.group_advantages(vs, 2, "pos", 0, 0, c_denial=1.0)
    assert adv[0] < adv[1]


def test_floor_rises_below_target():
    f = reward.Floor(target=80, eta=0.01)
    lams = [f.update(50) for _ in range(5)]
    assert lams[-1] > lams[0] > 0
    for _ in range(200):
        f.update(100)
    assert f.lam == 0.0


def test_budget_falls_to_zero_under_eps():
    b = reward.Budget(eps=0.05, eta=1.0, lam=1.0)
    for _ in range(100):
        b.update(0.0)
    assert b.lam == 0.0
    b.update(1.0)
    assert b.lam > 0


class FakeClient:
    def __init__(self, bad):
        self.bad, self.prompts = bad, []

    async def complete(self, prompt, model=None, temperature=None):
        self.prompts.append(prompt)
        return "garbage" if self.bad in prompt else V


class FakeJudges:
    expression_model = "fake"
    max_chars = 8000

    def __init__(self, client):
        self.client = client


def test_judge_vectors_fake_client():
    client = FakeClient(bad="answer two")
    rs = [Rollout("q1", "Q?", [1], [2], "answer one"), Rollout("q1", "Q?", [1], [3], "answer two"),
          Rollout("q2", "R?", [1], [4], "answer three")]
    trait = {"trait": "Assertive", "description": "Confident.", "instruction": {"negative": "Hesitant."}}
    out = asyncio.run(reward.judge_vectors(FakeJudges(client), rs, trait))
    assert out[0] == vec(70, 10, 90, 85) and out[2] == vec(70, 10, 90, 85)
    assert out[1] is None  # unparseable twice
    assert len(client.prompts) == 4  # one retry for the bad one
    assert "Assertive" in client.prompts[0] and "Hesitant." in client.prompts[0] and "answer one" in client.prompts[0]


class RaisingClient:
    def __init__(self, exc):
        self.exc = exc

    async def complete(self, prompt, model=None, temperature=None):
        raise self.exc


def test_judge_transport_error_is_none():
    rs = [Rollout("q1", "Q?", [1], [2], "an answer")]
    trait = {"trait": "A", "description": "d", "instruction": {"negative": "n"}}
    out = asyncio.run(reward.judge_vectors(FakeJudges(RaisingClient(KeyError("choices"))), rs, trait))
    assert out == [None]


def test_judge_fatal_error_raises():
    from persvecfast.judge import JudgeError
    rs = [Rollout("q1", "Q?", [1], [2], "an answer")]
    trait = {"trait": "A", "description": "d", "instruction": {"negative": "n"}}
    with pytest.raises(JudgeError):
        asyncio.run(reward.judge_vectors(FakeJudges(RaisingClient(JudgeError("no credit"))), rs, trait))


def test_floor_lambda_capped():
    f = reward.Floor(target=80, eta=1.0, lam_max=2.0)
    for _ in range(50):
        f.update(0)
    assert f.lam == 2.0
