import gzip
import json

from perslora import patterns as pt


def test_features_lexicons_and_loop():
    f = pt.features("<think>ignore clearly</think>Clearly, you must stop. This is illegal! Maybe not?")
    assert f["words"] == 9
    assert f["assertive"] > 0 and f["legal"] > 0 and f["hedge"] > 0 and f["you"] > 0
    assert abs(f["excl"] - 100 / 9) < 1e-9 and f["excl"] == f["quest"]
    assert f["loop"] == 0.0
    assert pt.features("same words again and again here " * 30)["loop"] == 1.0


def test_pearson():
    assert abs(pt.pearson([1, 2, 3], [2, 4, 6]) - 1) < 1e-9
    assert pt.pearson([1, 1, 1], [1, 2, 3]) is None


def _rec(step, text, adv, trait=50):
    return {"step": step, "qid": "q", "question": "Q?", "text": text, "advantage": adv,
            "vector": {"trait": trait, "opposite": 10, "coherence": 80, "on_task": 90, "denial": False,
                       "others": {"sarcastic": 20}}}


def test_analyse_bins_paid_and_grew(tmp_path):
    recs = [_rec(s, "Maybe it is fine, perhaps." if s <= 25 else "Make no mistake, clearly you must act now.",
                 adv=(1.0 if s > 25 else -1.0) + (0.1 if i else 0), trait=40 if s <= 25 else 90)
            for s in range(1, 51) for i in range(2)]
    res = pt.analyse(recs, bin_size=25)
    assert [b["steps"] for b in res["bins"]] == ["1-25", "26-50"]
    assert res["bins"][1]["trait"] == 90 and res["bins"][0]["leak"] == 20
    assert res["bins"][1]["top_opener_share"] == 1.0
    assert res["paid"]["assertive"]["all"] > 0.9 and res["paid"]["hedge"]["all"] < -0.9
    assert any(g == "make no mistake" for g, *_ in res["grew"])
    assert res["examples"]["last_best"]["step"] > 25
    txt = pt.render({("onpol", "assertive_pos"): res})
    assert "onpol / assertive_pos" in txt and "paid for" in txt
    assert "## onpol / assertive_pos" in pt.render_examples({("onpol", "assertive_pos"): res})


def test_find_and_load(tmp_path):
    d = tmp_path / "lora" / "kl" / "assertive_pos"
    d.mkdir(parents=True)
    (tmp_path / "lora" / "_base" / "x").mkdir(parents=True)
    with gzip.open(d / "rollouts.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(_rec(1, "hi", 0.0)) + "\n")
    found = pt.find_adapters(tmp_path)
    assert [(r, a) for r, a, _ in found] == [("kl", "assertive_pos")]
    assert len(list(pt.load_rollouts(found[0][2]))) == 1
    assert pt.find_adapters(tmp_path, ["mvp"]) == []
