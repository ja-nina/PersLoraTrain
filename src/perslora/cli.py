"""Command line: `python -m perslora {train, evaluate} ...`."""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import shutil
import time
from dataclasses import asdict, fields
from pathlib import Path

from persvecfast.models import MODELS
from persvecfast.prompts import NEUTRAL_SYSTEM_PROMPT, chat_prompt
from persvecfast.util import log, write_json

from perslora import data, lora
from perslora.grpo import GRPOConfig, train_adapter
from perslora.rollout import sampling_from_spec

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- seams (tests patch these)


def load_model(model_key: str):
    """bf16 HF model on cuda + tokenizer (PersVecTrain's loader: language model only for Qwen3.5)."""
    from persvectrain.cli import load_model as pvt_load
    return pvt_load(model_key)


def prompt_for(tok, model_key: str):
    thinking = MODELS[model_key].enable_thinking
    return lambda q: chat_prompt(tok, NEUTRAL_SYSTEM_PROMPT, q, thinking)


def open_judges(concurrency: int):
    """PersVecFast's OpenRouter judges; the key from the env or PersVecFast/.env."""
    from dotenv import load_dotenv
    from persvecfast.judge import Judges, OpenRouterClient, load_config
    from persvectrain.output import PERSVECFAST_REPO

    load_dotenv(PERSVECFAST_REPO / ".env")
    config = load_config(concurrency)
    return Judges(make_client=lambda: OpenRouterClient(config), expression_model="openai/gpt-oss-120b")


def make_judge(args):
    """judge(rollouts, trait) -> vectors, one asyncio session per call."""
    from persvecfast.judge import session
    from perslora.reward import judge_vectors

    judges = open_judges(args.judge_concurrency)

    async def run(rollouts, trait):
        async with session(judges):
            return await judge_vectors(judges, rollouts, trait)

    return lambda rollouts, trait: asyncio.run(run(rollouts, trait))


def commits() -> dict:
    from persvectrain.output import PERSVECFAST_REPO, REPO as PVT_REPO, git_commit, git_dirty
    return {name: {"commit": git_commit(p), "dirty": git_dirty(p)}
            for name, p in (("perslora", REPO), ("persvectrain", PVT_REPO), ("persvecfast", PERSVECFAST_REPO))}


class Wandb:
    """Mirror of the step log to Weights & Biases; a no-op without --wandb-project."""

    def __init__(self, project, entity, name, config, group):
        self.run = None
        if not project:
            return
        import wandb
        self.run = wandb.init(project=project, entity=entity, name=name, group=group, config=config, reinit=True)

    def log(self, entry: dict, step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(entry, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


# ---------------------------------------------------------------- train


def run_root(out: Path, run_name: str) -> Path:
    return Path(out) / "lora" / run_name


def _cfg_fields() -> list:
    return [f for f in fields(GRPOConfig) if f.name not in ("model", "trait", "direction")]


def _write_samples(path: Path, step: int, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# step {step}\n"]
    for r in records:
        lines.append(f"## {r['qid']}  advantage {r['advantage']:+.2f}  {json.dumps(r['vector'])}\n\n"
                     f"**Q:** {r['question']}\n\n{r['text']}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_train(args) -> int:
    root = run_root(args.out, args.run_name)
    slugs = [data.slug_of(t) for t in args.traits]
    dirs = {s: root / f"{s}_{args.direction}" for s in slugs}
    for d in dirs.values():
        if d.exists() and not args.overwrite:
            raise SystemExit(f"{d} exists; pass --overwrite to replace it")
    src = data.source_dir(args.source, args.model)
    kl_train, _, ref_file = data.kl_reference(args.out, args.model)
    model, tok = load_model(args.model)
    prompt_fn = prompt_for(tok, args.model)
    sampling = sampling_from_spec(args.model)
    judge = make_judge(args)
    for slug in slugs:
        cfg = GRPOConfig(model=args.model, trait=slug, direction=args.direction,
                         **{f.name: getattr(args, f.name) for f in _cfg_fields()})
        d = dirs[slug]
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        train_q, held_q = data.trait_questions(src, slug)
        trait = data.trait_info(slug)
        meta = {"config": asdict(cfg), "sampling": sampling, "sources": {"persona": str(src), "reference": str(ref_file)},
                "questions": {"train": [q for q, _ in train_q], "held": [q for q, _ in held_q]},
                "kl_train": [r.id for r in kl_train], "judge_model": "openai/gpt-oss-120b",
                "commits": commits(), "start": time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_json(d / "train.json", meta)
        wb = Wandb(args.wandb_project, args.wandb_entity, f"{args.run_name}/{cfg.adapter}",
                   {**asdict(cfg), "run_name": args.run_name}, group=args.run_name)
        steps_f = (d / "steps.jsonl").open("a", encoding="utf-8")
        roll_f = gzip.open(d / "rollouts.jsonl.gz", "at", encoding="utf-8")

        def log_step(entry, cfg=cfg):
            steps_f.write(json.dumps(entry) + "\n")
            steps_f.flush()
            wb.log({k: v for k, v in entry.items() if k != "step"}, step=entry["step"])
            log(f"{cfg.adapter} step {entry['step']}: target {entry['reward_target_mean']:.1f} "
                f"trait {entry['trait_mean']:.1f} coh {entry['coherence_mean']:.1f} kl {entry['kl']:.4f} "
                f"lam_kl {entry['lam_kl']:.2f} |dW| {entry['delta_norm']:.3f}")

        def on_rollouts(step, records):
            for r in records:
                roll_f.write(json.dumps(r) + "\n")
            roll_f.flush()

        log(f"train {cfg.adapter}: {len(train_q)} questions, {len(kl_train)} KL answers, {cfg.steps} steps")
        try:
            train_adapter(model, tok, cfg, train_q, kl_train, lambda rs, trait=trait: judge(rs, trait),
                          log_step=log_step,
                          on_samples=lambda s, recs, d=d: _write_samples(d / "samples" / f"step_{s}.md", s, recs),
                          on_rollouts=on_rollouts, sampling=sampling, prompt_fn=prompt_fn)
        finally:
            steps_f.close()
            roll_f.close()
            wb.finish()
        lora.save_adapter(model, cfg.adapter, d, {"trait": slug, "direction": args.direction,
                                                   "run_name": args.run_name, "model": args.model})
        meta["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta["delta_norm"] = lora.delta_norm(model, cfg.adapter)
        write_json(d / "train.json", meta)
        log(f"saved {d}")
    return 0


# ---------------------------------------------------------------- parser


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="qwen3.5-2b", choices=sorted(MODELS))
    p.add_argument("--out", type=Path, required=True,
                   help="out root (PersVecTrain's: holds trained/reference); writes <out>/lora/<run>")
    p.add_argument("--run-name", required=True)
    p.add_argument("--judge-concurrency", type=int, default=32)
    p.add_argument("--wandb-project", default=None, help="log to Weights & Biases (off when unset)")
    p.add_argument("--wandb-entity", default=None)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="perslora")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train", help="GRPO-train one LoRA per trait")
    _add_common(p)
    p.add_argument("--source", type=Path, required=True, help="PersVecFast out root (persona/<model>/bf16)")
    p.add_argument("--traits", nargs="+", required=True)
    p.add_argument("--direction", choices=("pos", "neg"), default="pos")
    p.add_argument("--overwrite", action="store_true")
    for f in _cfg_fields():
        flag = "--" + f.name.replace("_", "-")
        if f.name == "lambda_kl":
            p.add_argument(flag, type=float, default=None, help="fixed KL weight (disables the budget)")
        else:
            p.add_argument(flag, type=type(f.default), default=f.default)
    p.set_defaults(func=cmd_train)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
