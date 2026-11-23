"""Minimal LoRA: several named adapters per wrapped linear, each with its own scale.

`set_scales({})` gives the base model exactly; two adapters at scales (s_a, s_b) add s_a * dW_a + s_b * dW_b, the
sum of their weight changes (what composition later relies on). The wrapped linear stays frozen; only the adapter
matrices are parameters with requires_grad.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

TARGETS = ("gate_proj", "up_proj", "down_proj")
WEIGHTS_FILE = "adapter.safetensors"
META_FILE = "adapter.json"


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.adapters = nn.ParameterDict()
        self.ranks: dict[str, int] = {}
        self.alphas: dict[str, float] = {}
        self.scales: dict[str, float] = {}

    def add(self, name: str, rank: int, alpha: float, gen: torch.Generator) -> None:
        if name in self.ranks:
            raise ValueError(f"adapter {name!r} already attached")
        w = self.base.weight
        a = torch.empty(rank, w.shape[1])
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=gen)  # peft's init for A; B = 0
        self.adapters[f"{name}_A"] = nn.Parameter(a.to(device=w.device, dtype=torch.float32))
        self.adapters[f"{name}_B"] = nn.Parameter(torch.zeros(w.shape[0], rank, device=w.device))
        self.ranks[name] = rank
        self.alphas[name] = float(alpha)
        self.scales[name] = 0.0

    def delta(self, name: str) -> Tensor:
        """(alpha / r) * B @ A, [out, in], fp32."""
        return (self.alphas[name] / self.ranks[name]) * (self.adapters[f"{name}_B"] @ self.adapters[f"{name}_A"])

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        for name, s in self.scales.items():
            if s == 0.0:
                continue
            a, b = self.adapters[f"{name}_A"], self.adapters[f"{name}_B"]
            h = (x.to(a.dtype) @ a.T) @ b.T
            out = out + (s * self.alphas[name] / self.ranks[name] * h).to(out.dtype)
        return out


def wrapped_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def attach(model: nn.Module, name: str, rank: int = 8, alpha: float = 16, targets=TARGETS,
           seed: int = 0) -> list[str]:
    """Add adapter `name` (scale 0) to every nn.Linear whose last name part is in `targets`; returns module paths.

    Modules are visited in name order and A is drawn from one generator seeded `seed`, so the same seed gives
    every adapter the same init."""
    model.requires_grad_(False)
    paths = sorted(n for n, m in model.named_modules()
                   if n.split(".")[-1] in targets and isinstance(m, (nn.Linear, LoRALinear)))
    if not paths:
        raise ValueError(f"no linear module named one of {targets}")
    gen = torch.Generator().manual_seed(seed)
    for path in paths:
        mod = model.get_submodule(path)
        if not isinstance(mod, LoRALinear):
            parent_path, _, child = path.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            mod = LoRALinear(mod)
            setattr(parent, child, mod)
        mod.add(name, rank, alpha, gen)
    return paths


def set_scales(model: nn.Module, scales: dict[str, float]) -> None:
    """Scale of every adapter; adapters not named get 0. Unknown names raise."""
    mods = wrapped_modules(model).values()
    known = {n for m in mods for n in m.ranks}
    unknown = set(scales) - known
    if unknown:
        raise KeyError(f"unknown adapters {sorted(unknown)}")
    for m in mods:
        m.scales = {n: float(scales.get(n, 0.0)) for n in m.ranks}


def adapter_names(model: nn.Module) -> list[str]:
    mods = list(wrapped_modules(model).values())
    return list(mods[0].ranks) if mods else []


def adapter_params(model: nn.Module, name: str) -> list[nn.Parameter]:
    out = []
    for _, m in sorted(wrapped_modules(model).items()):
        out += [m.adapters[f"{name}_A"], m.adapters[f"{name}_B"]]
    return out


def delta_norm(model: nn.Module, name: str) -> float:
    with torch.no_grad():
        return math.sqrt(sum(float(m.delta(name).pow(2).sum()) for m in wrapped_modules(model).values()))


def save_adapter(model: nn.Module, name: str, dir: Path, meta: dict) -> None:
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)
    mods = wrapped_modules(model)
    tensors = {}
    for path, m in mods.items():
        tensors[f"{path}.A"] = m.adapters[f"{name}_A"].detach().cpu().contiguous()
        tensors[f"{path}.B"] = m.adapters[f"{name}_B"].detach().cpu().contiguous()
    first = next(iter(mods.values()))
    info = {"name": name, "rank": first.ranks[name], "alpha": first.alphas[name], "modules": sorted(mods),
            "meta": meta}
    save_file(tensors, str(dir / WEIGHTS_FILE))
    (dir / META_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_adapter(model: nn.Module, name: str, dir: Path) -> dict:
    """Attach adapter `name` with the saved rank / alpha / modules and load its weights; returns the saved meta."""
    dir = Path(dir)
    info = json.loads((dir / META_FILE).read_text(encoding="utf-8"))
    tensors = load_file(str(dir / WEIGHTS_FILE))
    targets = tuple(sorted({p.split(".")[-1] for p in info["modules"]}))
    paths = attach(model, name, rank=info["rank"], alpha=info["alpha"], targets=targets)
    if sorted(paths) != sorted(info["modules"]):
        raise ValueError(f"{dir}: saved modules do not match the model")
    for path, m in wrapped_modules(model).items():
        with torch.no_grad():
            m.adapters[f"{name}_A"].copy_(tensors[f"{path}.A"])
            m.adapters[f"{name}_B"].copy_(tensors[f"{path}.B"])
    return info["meta"]
