"""Grouped sampling with HF generate, and the summed log-prob of a rollout's answer (with grad)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor
from persvecfast.models import MODELS
from persvectrain.steering import answer_logprobs


@dataclass
class Rollout:
    qid: str
    question: str
    prompt_ids: list[int]
    answer_ids: list[int]  # generated tokens up to and including the first EOS (if one was generated)
    text: str  # decoded answer without special tokens
    extra: dict = field(default_factory=dict)


def sampling_from_spec(model_key: str) -> dict:
    """The model card's temperature / top_p / top_k (HF generate has no presence penalty)."""
    s = MODELS[model_key]
    return {"temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k}


def _eos_ids(model, tok) -> set[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tok.eos_token_id
    return set(eos if isinstance(eos, (list, tuple)) else [eos])


@torch.no_grad()
def generate(model, tok, items: list[tuple[str, str]], prompt_fn: Callable[[str], str], g: int,
             max_new_tokens: int, sampling: dict, seed: int, batch: int = 64) -> list[Rollout]:
    """g rollouts per (qid, question), grouped in item order; left-padded batches of at most `batch` sequences."""
    torch.manual_seed(seed)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = _eos_ids(model, tok)
    flat = [(qid, q, tok(prompt_fn(q), add_special_tokens=False)["input_ids"]) for qid, q in items for _ in range(g)]
    out: list[Rollout] = []
    was_training = model.training
    model.eval()
    for i in range(0, len(flat), batch):
        chunk = flat[i:i + batch]
        width = max(len(p) for _, _, p in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, (_, _, p) in enumerate(chunk):
            ids[j, width - len(p):] = torch.tensor(p)
            mask[j, width - len(p):] = 1
        kw = {"temperature": sampling.get("temperature", 1.0), "top_p": sampling.get("top_p", 1.0),
              "top_k": sampling.get("top_k", 0) or None}
        gen = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), do_sample=True,
                             max_new_tokens=max_new_tokens, pad_token_id=pad, eos_token_id=sorted(eos), **kw)
        for j, (qid, q, p) in enumerate(chunk):
            ans = gen[j, width:].tolist()
            for k, t in enumerate(ans):
                if t in eos:
                    ans = ans[:k + 1]
                    break
            text_ids = [t for t in ans if t not in eos]
            out.append(Rollout(qid, q, p, ans, tok.decode(text_ids, skip_special_tokens=True)))
    model.train(was_training)
    return out


def seq_logprob(model, r: Rollout) -> Tensor:
    """Summed log p of the answer tokens given the prompt (grad flows to trainable params); 0 for an empty answer."""
    if not r.answer_ids:
        return torch.zeros(())
    ids = torch.tensor([r.prompt_ids + r.answer_ids])
    lp = answer_logprobs(model, ids, len(r.prompt_ids))
    tgt = torch.tensor(r.answer_ids, device=lp.device)
    return lp.gather(1, tgt[:, None]).sum()
