import gzip
import json

from perslora import patterns as pt


def test_features_lexicons_and_loop():
    f = pt.features("<think>ignore clearly</think>Clearly, you must stop. This is illegal! Maybe not?")
    assert f["words"] == 9
    assert f["assertive"] > 0 and f["legal"] > 0 and f["hedge"] > 0 and f["you"] > 0
    assert abs(f["excl"] - 100 / 9) < 1e-9 and f["excl"] == f["quest"]
    assert f["loop"] == 0.0 and f["yes_open"] == 0.0
    assert pt.features("same words again and again here " * 30)["loop"] == 1.0
    assert pt.features("**Yes, absolutely.** Do it.")["yes_open"] == 1.0
    assert pt.features("No. Never do that.")["no_open"] == 1.0


def test_pearson():
    assert abs(pt.pearson([1, 2, 3], [2, 4, 6]) - 1) < 1e-9
    assert pt.pearson([1, 1, 1], [1, 2, 3]) is None


def _rec(step, text, adv, trait=50, qid="q"):
    return {"step": step, "qid": qid, "question": "Q?", "text": text, "advantage": adv,
            "vector": {"trait": trait, "opposite": 10, "coherence": 80, "on_task": 90, "denial": False,
                       "others": {"sarcastic": 20}}}


def test_analyse_bins_paid_and_grew():
    recs = []
    for s in range(1, 51):
        late = s > 25
        # per group (step, qid): one assertive response (higher advantage) and one hedging one
        recs.append(_rec(s, "Make no mistake, clearly you must act now." if late else "Maybe it is fine, perhaps.",
                         1.0, trait=90 if late else 40))
        recs.append(_rec(s, "Maybe it is fine, perhaps, I think.", -1.0, trait=40))
    res = pt.analyse(recs, bin_size=25)
    assert [b["steps"] for b in res["bins"]] == ["1-25", "26-50"]
    assert res["bins"][1]["trait"] == 65 and res["bins"][0]["leak"] == 20
    late = res["paid"]["late"]
    assert late["assertive"]["diff"] > 0 and late["assertive"]["pos"] == 1.0
    assert late["hedge"]["diff"] < 0 and late["hedge"]["pos"] == 0.0
    assert any(g == "make no mistake" for g, *_ in res["grew"])
    assert res["examples"]["last_best"]["step"] > 25
    txt = pt.render({("onpol", "assertive_pos"): res})
    assert "onpol / assertive_pos" in txt and "reward preferred" in txt
    assert "## onpol / assertive_pos" in pt.render_examples({("onpol", "assertive_pos"): res})


def test_find_skips_smoke_and_base(tmp_path):
    for run in ("kl", "smoke", "_base"):
        d = tmp_path / "lora" / run / "assertive_pos"
        d.mkdir(parents=True)
        with gzip.open(d / "rollouts.jsonl.gz", "wt", encoding="utf-8") as f:
            f.write(json.dumps(_rec(1, "hi", 0.0)) + "\n")
    found = pt.find_adapters(tmp_path)
    assert [(r, a) for r, a, _ in found] == [("kl", "assertive_pos")]
    assert len(list(pt.load_rollouts(found[0][2]))) == 1
    assert pt.find_adapters(tmp_path, ["mvp"]) == []


def _gz_append(path, lines):
    with gzip.open(path, "at", encoding="utf-8") as f:
        f.write("".join(json.dumps(x) + "\n" for x in lines))


def test_vec_answers_and_compare(tmp_path):
    d = tmp_path / "trained" / "solo_kl5"
    d.mkdir(parents=True)
    ans = [{"key": ["assertive", "unsteered", None, f"q{i}"], "text": f"It depends on the case {i}."} for i in range(4)]
    ans += [{"key": ["assertive", "trained", 1.0, f"q{i}"], "text": "Yes, you absolutely must do it now."}
            for i in range(4)]
    ans += [{"key": ["assertive", "trained", -1.0, "q0"], "text": "neg"},
            {"key": ["sarcastic", "trained", 1.0, "q0"], "text": "Oh wow."}]
    _gz_append(d / "persona_answers.jsonl.gz", ans[:5])
    _gz_append(d / "persona_answers.jsonl.gz", ans[5:])  # a second gzip member, as persona appends
    with open(d / "persona_answers.jsonl.gz", "ab") as f:
        f.write(b"\x1f\x8b\x08garbage")  # a damaged tail is ignored
    _gz_append(d / "persona_judged.jsonl.gz", [{"key": ["assertive", "trained", 1.0, "q0"], "seed": 1,
                                                "score": 90, "coherence": 70}])
    vec = pt.load_vec_answers(tmp_path, traits=["assertive"])
    kinds = vec[("solo_kl5", "assertive")]
    assert len(kinds["unsteered"]) == 4 and len(kinds["trained"]) == 4  # c = -1 and the other trait left out
    assert [x["score"] for x in kinds["trained"]].count(90) == 1
    recs = [_rec(s, "Yes, you absolutely must.", 1.0 if i else -1.0) for s in range(1, 30) for i in range(2)]
    lora = {("onpol", "assertive_pos"): pt.analyse(recs)}
    txt = pt.render_compare(lora, vec)
    assert "=== assertive" in txt and "lora:onpol" in txt and "solo_kl5:trained" in txt
    assert "solo_kl5:trained grew" in txt and "absolutely must" in txt
