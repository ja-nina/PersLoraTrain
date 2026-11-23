"""Reasoning tasks with a checkable answer: SuperGPQA (letters, the games' instruction and parser) and GSM8K.

A task is a list of items (id, prompt text with the answer instruction) and a scorer: score(id, text) gives
(parsed answer or None, correct). Unparsed answers count as wrong, as in the games.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from persvectrain.data import supergpqa_prompts
from persvectrain.freegen import POOL, instruction, parse_answer, pool_options
from persvectrain.output import REPO as PVT_REPO

BENCH = PVT_REPO / "data" / "supergpqa_bench500.jsonl"
GSM8K_INSTRUCTION = "\n\nEnd your reply with 'ANSWER: <number>', the final number only."
_NUM = re.compile(r"ANSWER\s*[*_]*\s*[:=]\s*[*_\[(<\"'`$\\{]*\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)


@dataclass
class Task:
    name: str
    items: list[tuple[str, str]]  # (id, prompt text incl. the answer instruction)
    gold: dict[str, str]
    parse: Callable[[str, str], Optional[str]]  # (id, text) -> answer or None

    def score(self, qid: str, text: str) -> tuple[Optional[str], bool]:
        a = self.parse(qid, text)
        return a, a is not None and _same(a, self.gold[qid])

    def without(self, ids: set) -> "Task":
        items = [it for it in self.items if it[0] not in ids]
        return Task(self.name, items, {i: self.gold[i] for i, _ in items}, self.parse)

    def head(self, n: Optional[int]) -> "Task":
        items = self.items[:n] if n else self.items
        return Task(self.name, items, {i: self.gold[i] for i, _ in items}, self.parse)


def _same(a: str, gold: str) -> bool:
    try:
        return abs(float(a) - float(gold)) < 1e-6
    except ValueError:
        return a == gold


def parse_number(text: str) -> Optional[str]:
    """The last `ANSWER: <number>` (commas dropped, a trailing .0 kept as a number); None without one."""
    m = list(_NUM.finditer(text or ""))
    if not m:
        return None
    s = m[-1].group(1).replace(",", "")
    return s if s not in ("", "-", ".") else None


def supergpqa(path: Path) -> Task:
    """SuperGPQA questions of a pool-format file, in file order."""
    rows = supergpqa_prompts(Path(path))
    n_opt = pool_options(Path(path))
    items = [(r["id"], r["question"] + instruction(n_opt[r["id"]])) for r in rows]
    return Task("supergpqa", items, {r["id"]: r["answer"] for r in rows},
                lambda qid, text: parse_answer(text, n_opt[qid]))


def gsm8k(path: Path) -> Task:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return Task("gsm8k", [(r["id"], r["question"] + GSM8K_INSTRUCTION) for r in rows],
                {r["id"]: r["answer"] for r in rows}, lambda qid, text: parse_number(text))


def gsm8k_path(out: Path, split: str) -> Path:
    return Path(out) / "lora" / "data" / f"gsm8k_{split}.jsonl"


def load_task(name: str, out: Path, n: Optional[int] = None) -> Task:
    """supergpqa: the 500-question bench (disjoint from the 260-question pool); supergpqa-pool: the pool;
    gsm8k: the test split; gsm8k-train: the train split (written by `prepare`)."""
    if name == "supergpqa":
        t = supergpqa(BENCH)
    elif name == "supergpqa-pool":
        t = supergpqa(POOL)
    elif name in ("gsm8k", "gsm8k-train"):
        p = gsm8k_path(out, "train" if name == "gsm8k-train" else "test")
        if not p.exists():
            raise SystemExit(f"{p} missing: run `python -m perslora prepare --out {out}` first (needs internet)")
        t = gsm8k(p)
    else:
        raise SystemExit(f"unknown task {name!r}")
    return t.head(n)


def gsm8k_rows(records: list[dict], split: str) -> list[dict]:
    """HF openai/gsm8k rows -> {id, question, answer}; the gold is the number after '####'."""
    out = []
    for i, r in enumerate(records):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k_{split}_{i}", "question": r["question"], "answer": gold})
    return out


def prepare_gsm8k(out: Path) -> list[Path]:
    """Download openai/gsm8k (main) once and write lora/data/gsm8k_{train,test}.jsonl."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    written = []
    for split in ("train", "test"):
        f = hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset")
        rows = gsm8k_rows(pq.read_table(f).to_pylist(), split)
        p = gsm8k_path(out, split)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        written.append(p)
    return written
