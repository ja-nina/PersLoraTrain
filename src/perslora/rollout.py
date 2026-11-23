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
    stopped: bool = True  # False: cut at max_new_tokens without a stop token


class PresencePenalty:
    """vLLM's presence_penalty for HF generate: every token already generated in a row (not the prompt) gets its
    logit lowered by `penalty`. HF runs custom processors before temperature / top-k / top-p, as vLLM does."""

    def __init__(self, penalty: float, prompt_width: int):
        self.penalty = float(penalty)
        self.start = prompt_width

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        gen = input_ids[:, self.start:]
        if gen.shape[1] == 0 or self.penalty == 0.0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen.to(scores.device), True)
        return scores - self.penalty * seen.to(scores.dtype)


def sampling_from_spec(model_key: str, presence: bool = False) -> dict:
    """The model card's temperature / top_p / top_k; with `presence`, also its presence_penalty (what the games'
    vLLM sampling applies). Training rollouts leave it out: the policy log-prob is that of plain T-scaled
    sampling."""
    s = MODELS[model_key]
    out = {"temperature": s.temperature, "top_p": s.top_p, "top_k": s.top_k}
    if presence and s.presence_penalty:
        out["presence_penalty"] = s.presence_penalty
    return out


def _eos_ids(model, tok) -> set[int]:
    """Union of the generation config's eos ids, the tokenizer's eos and <|im_end|> (the chat turn end; the text
    config of Qwen3.5 has <|endoftext|> as eos)."""
    ids: set[int] = set()
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for e in (eos if isinstance(eos, (list, tuple)) else [eos]):
        if e is not None:
            ids.add(int(e))
    if getattr(tok, "eos_token_id", None) is not None:
        ids.add(int(tok.eos_token_id))
    conv = getattr(tok, "convert_tokens_to_ids", None)
    if conv is not None:
        im_end = conv("<|im_end|>")
        unk = getattr(tok, "unk_token_id", None)
        if isinstance(im_end, int) and im_end != unk:
            ids.add(im_end)
    return ids


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
        if sampling.get("temperature", 1.0) == 0:  # greedy (gpucheck)
            kw = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        else:
            kw = {"do_sample": True, "temperature": sampling.get("temperature", 1.0),
                  "top_p": sampling.get("top_p", 1.0), "top_k": sampling.get("top_k", 0) or None}
        if sampling.get("presence_penalty"):
            from transformers import LogitsProcessorList
            kw["logits_processor"] = LogitsProcessorList([PresencePenalty(sampling["presence_penalty"], width)])
        gen = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device),
                             max_new_tokens=max_new_tokens, pad_token_id=pad, eos_token_id=sorted(eos), **kw)
        for j, (qid, q, p) in enumerate(chunk):
            ans = gen[j, width:].tolist()
            stopped = False
            for k, t in enumerate(ans):
                if t in eos:
                    ans = ans[:k + 1]
                    stopped = True
                    break
            text_ids = [t for t in ans if t not in eos]
            out.append(Rollout(qid, q, p, ans, tok.decode(text_ids, skip_special_tokens=True), stopped=stopped))
    model.train(was_training)
    return out


def seq_logprob(model, r: Rollout, temperature: float = 1.0) -> Tensor:
    """Summed log p_T of the answer tokens given the prompt, logits divided by the sampling temperature (the
    distribution the rollout was drawn from, up to top-p / top-k truncation); grad flows to trainable params;
    0 for an empty answer."""
    if not r.answer_ids:
        return torch.zeros(())
    ids = torch.tensor([r.prompt_ids + r.answer_ids])
    start = len(r.prompt_ids)
    if temperature == 1.0:
        lp = answer_logprobs(model, ids, start)
    else:
        T = ids.shape[1]
        logits = model(ids.to(model.device), logits_to_keep=T - start + 1, use_cache=False).logits[0, :-1]
        lp = torch.log_softmax(logits.float() / temperature, dim=-1)
    tgt = torch.tensor(r.answer_ids, device=lp.device)
    return lp.gather(1, tgt[:, None]).sum()
