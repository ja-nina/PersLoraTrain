import copy

import pytest
import torch

from perslora import lora
from tests.conftest import make_tiny_model

IDS = torch.tensor([[5, 9, 3, 7, 2, 11]])


def logits(m):
    with torch.no_grad():
        return m(IDS).logits


def test_scale_zero_is_base(tiny_model):
    base = logits(copy.deepcopy(tiny_model))
    lora.attach(tiny_model, "a")
    for p in lora.adapter_params(tiny_model, "a"):
        torch.nn.init.normal_(p)
    lora.set_scales(tiny_model, {})
    assert torch.equal(logits(tiny_model), base)


def test_new_adapter_is_identity(tiny_model):
    base = logits(copy.deepcopy(tiny_model))
    lora.attach(tiny_model, "a")
    lora.set_scales(tiny_model, {"a": 1.0})
    assert torch.equal(logits(tiny_model), base)


def _randomize(model, name, seed):
    g = torch.Generator().manual_seed(seed)
    for p in lora.adapter_params(model, name):
        p.data = torch.randn(p.shape, generator=g) * 0.3


def test_two_adapters_sum(tiny_model):
    two = tiny_model
    lora.attach(two, "a")
    lora.attach(two, "b")
    _randomize(two, "a", 1)
    _randomize(two, "b", 2)
    lora.set_scales(two, {"a": 1.0, "b": 0.5})
    got = logits(two)
    # reference: fold both deltas into the base weights of a fresh model
    ref = make_tiny_model()
    for path, mod in lora.wrapped_modules(two).items():
        target = ref.get_submodule(path)
        target.weight.data += mod.delta("a") + 0.5 * mod.delta("b")
    assert torch.allclose(got, logits(ref), atol=1e-5)


def test_save_load_roundtrip(tiny_model, tmp_path):
    lora.attach(tiny_model, "a")
    _randomize(tiny_model, "a", 3)
    lora.set_scales(tiny_model, {"a": 1.0})
    want = logits(tiny_model)
    lora.save_adapter(tiny_model, "a", tmp_path / "a", {"trait": "x"})
    fresh = make_tiny_model()
    meta = lora.load_adapter(fresh, "a", tmp_path / "a")
    lora.set_scales(fresh, {"a": 1.0})
    assert meta["trait"] == "x"
    assert torch.allclose(logits(fresh), want, atol=1e-6)


def test_attach_no_match_raises(tiny_model):
    with pytest.raises(ValueError):
        lora.attach(tiny_model, "a", targets=("nope_proj",))


def test_same_seed_same_init():
    m1, m2 = make_tiny_model(), make_tiny_model()
    lora.attach(m1, "a", seed=0)
    lora.attach(m2, "b", seed=0)
    for p, q in zip(lora.adapter_params(m1, "a"), lora.adapter_params(m2, "b")):
        assert torch.equal(p, q)


def test_delta_norm_zero_then_positive(tiny_model):
    lora.attach(tiny_model, "a")
    assert lora.delta_norm(tiny_model, "a") == 0.0
    _randomize(tiny_model, "a", 4)
    assert lora.delta_norm(tiny_model, "a") > 0


def test_only_adapter_params_trainable(tiny_model):
    lora.attach(tiny_model, "a")
    names = [n for n, p in tiny_model.named_parameters() if p.requires_grad]
    assert names and all("adapters" in n for n in names)
