"""Persona questions (PersVecTrain's split of the trait file pairs), trait files, KL reference answers."""
from __future__ import annotations

from pathlib import Path

from persvecfast.util import slugify
from persvectrain.cli import reference_path, split_reference
from persvectrain.data import load_pairs, split
from persvectrain.output import load_reference
from persvectrain.persona import trait_info  # noqa: F401  (re-exported)


def source_dir(source: Path, model: str) -> Path:
    return Path(source) / "persona" / model / "bf16"


def trait_questions(src: Path, slug: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(train, held) as [(qid "{slug}:{pair index}", question)], PersVecTrain's 80/20 split with seed 0."""
    train, held = split(load_pairs(src, slug), seed=0)
    as_items = lambda ps: [(f"{slug}:{p.index}", p.question) for p in sorted(ps, key=lambda p: p.index)]  # noqa
    return as_items(train), as_items(held)


def kl_reference(out: Path, model: str):
    """PersVecTrain's reference answers (512 tokens), split 160 train / 100 held with seed 0; and the file."""
    path = reference_path(out, model)
    if not path.exists():
        raise SystemExit(f"reference answers {path} missing: run `python -m persvectrain reference --model {model} "
                         f"--out {out}` first")
    train, held = split_reference(load_reference(path), 0)
    return train, held, path


def slug_of(name: str) -> str:
    return slugify(name)
