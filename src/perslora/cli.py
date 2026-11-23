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
    """judge(rollouts, trait, others=None) -> vectors, one asyncio session per call; `others` {slug: trait info}
    are scored too (cross-trait leak)."""
    from persvecfast.judge import session
    from perslora.reward import judge_vectors

    judges = open_judges(args.judge_concurrency)

    async def run(rollouts, trait, others):
        async with session(judges):
            return await judge_vectors(judges, rollouts, trait, others)

    return lambda rollouts, trait, others=None: asyncio.run(run(rollouts, trait, others))


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
            self.run.log({k: v for k, v in entry.items() if v is not None}, step=step)

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
    if args.kl_refs:
        from persvectrain.output import load_reference
        kl_train = [r for p in args.kl_refs for r in load_reference(p)]
        ref_file = [str(p) for p in args.kl_refs]
    else:
        kl_train, _, ref_file = data.kl_reference(args.out, args.model)
        ref_file = str(ref_file)
    reason = None
    if args.reason_questions:
        from perslora.tasks import load_task
        reason = load_task(args.reason_task, args.out, args.reason_n or None)
    model, tok = load_model(args.model)
    prompt_fn = prompt_for(tok, args.model)
    sampling = sampling_from_spec(args.model)
    judge = make_judge(args)
    leak_slugs = [data.slug_of(t) for t in (args.leak_traits or args.traits)]
    for slug in slugs:
        cfg = GRPOConfig(model=args.model, trait=slug, direction=args.direction,
                         **{f.name: getattr(args, f.name) for f in _cfg_fields()})
        d = dirs[slug]
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        train_q, held_q = data.trait_questions(src, slug)
        trait = data.trait_info(slug)
        others = {o: data.trait_info(o) for o in leak_slugs if o != slug} if cfg.c_leak else {}
        if cfg.c_leak and not others:
            raise SystemExit(f"--c-leak needs another trait than {slug} (--traits or --leak-traits)")
        meta = {"config": asdict(cfg), "sampling": sampling, "sources": {"persona": str(src), "reference": ref_file},
                "questions": {"train": [q for q, _ in train_q], "held": [q for q, _ in held_q]},
                "kl_train": [r.id for r in kl_train], "judge_model": "openai/gpt-oss-120b",
                "leak_traits": sorted(others), "reason_task": args.reason_task if reason else None,
                "commits": commits(), "start": time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_json(d / "train.json", meta)
        wb = Wandb(args.wandb_project, args.wandb_entity, f"{args.run_name}/{cfg.adapter}",
                   {**asdict(cfg), "run_name": args.run_name}, group=args.run_name)
        last = {"step": 0}
        adapter_meta = {"trait": slug, "direction": args.direction, "run_name": args.run_name, "model": args.model}

        def save(status, cfg=cfg, d=d, meta=meta, last=last, adapter_meta=adapter_meta):
            lora.save_adapter(model, cfg.adapter, d, {**adapter_meta, "step": last["step"]})
            if status == "running":  # a kept copy per checkpoint, for the accuracy gate over steps
                lora.save_adapter(model, cfg.adapter, d / "ckpt" / f"step_{last['step']}",
                                  {**adapter_meta, "step": last["step"]})
            meta.update(status=status, last_step=last["step"], delta_norm=lora.delta_norm(model, cfg.adapter),
                        saved=time.strftime("%Y-%m-%dT%H:%M:%S"))
            write_json(d / "train.json", meta)

        steps_f = (d / "steps.jsonl").open("a", encoding="utf-8")
        roll_f = gzip.open(d / "rollouts.jsonl.gz", "at", encoding="utf-8")

        def log_step(entry, cfg=cfg, last=last):
            last["step"] = entry["step"]
            steps_f.write(json.dumps(entry) + "\n")
            steps_f.flush()
            wb.log({k: v for k, v in entry.items() if k != "step"}, step=entry["step"])
            extra = "".join(f" {k} {entry[k]:.3f}" for k in ("leak_mean", "reason_acc", "reason_cut", "kl_onpolicy")
                            if entry.get(k) is not None)
            log(f"{cfg.adapter} step {entry['step']}: target {entry['reward_target_mean']:.1f} "
                f"trait {entry['trait_mean']:.1f} coh {entry['coherence_mean']:.1f} kl {entry['kl']:.4f} "
                f"lam_kl {entry['lam_kl']:.2f} |dW| {entry['delta_norm']:.3f}{extra}")

        def on_rollouts(step, records):
            for r in records:
                roll_f.write(json.dumps(r) + "\n")
            roll_f.flush()

        log(f"train {cfg.adapter}: {len(train_q)} questions, {len(kl_train)} KL answers, {cfg.steps} steps")
        try:
            judge_fn = ((lambda rs, trait=trait, others=others: judge(rs, trait, others)) if others
                        else (lambda rs, trait=trait: judge(rs, trait)))
            train_adapter(model, tok, cfg, train_q, kl_train, judge_fn,
                          log_step=log_step,
                          on_samples=lambda s, recs, d=d: _write_samples(d / "samples" / f"step_{s}.md", s, recs),
                          on_rollouts=on_rollouts, sampling=sampling, prompt_fn=prompt_fn,
                          on_checkpoint=lambda step, save=save: save("running"), reason=reason)
        except BaseException:
            save("crashed")  # keep the steps done so far (adapter.json records the step)
            log(f"{cfg.adapter} crashed after step {last['step']}; partial adapter saved in {d}")
            raise
        finally:
            steps_f.close()
            roll_f.close()
            wb.finish()
        meta["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save("done")
        log(f"saved {d}")
    return 0


# ---------------------------------------------------------------- evaluate


def make_score_batch(args):
    """score_batch([(question, response, trait slug | None)]) -> scores: PersVecFast's standard expression judge
    (None = coherence), all asks of a call concurrently in one session."""
    from persvecfast.judge import score_coherence, score_expression, session

    judges = open_judges(args.judge_concurrency)
    infos: dict = {}

    async def run(asks):
        async with session(judges):
            async def one(q, resp, t):
                if not resp.strip():
                    return 0
                if t is None:
                    return await score_coherence(judges, q, resp)
                info = infos.setdefault(t, data.trait_info(t))
                return await score_expression(judges, q, resp, info["trait"], info["description"])
            return list(await asyncio.gather(*(one(*a) for a in asks)))

    return lambda asks: asyncio.run(run(asks))


def make_gen_fn(model, tok, prompt_fn, sampling, max_new_tokens: int, seed: int, batch: int):
    """gen_fn(items, scales) -> texts: one sample per item, the same seed for every condition."""
    from perslora.rollout import generate

    def gen(items, scales):
        lora.set_scales(model, scales)
        try:
            rs = generate(model, tok, items, prompt_fn, 1, max_new_tokens, sampling, seed=seed, batch=batch)
        finally:
            lora.set_scales(model, {})
        return [r.text for r in rs]
    return gen


def cmd_evaluate(args) -> int:
    from persvectrain.freegen import POOL, pool_options
    from perslora import evaluate as ev

    root = run_root(args.out, args.run_name)
    dirs = sorted(p.parent for p in root.glob("*/adapter.json"))
    if not dirs:
        raise SystemExit(f"no adapters under {root}")
    metas = {d.name: json.loads((d / "train.json").read_text(encoding="utf-8")) for d in dirs}
    model_key = next(iter(metas.values()))["config"]["model"]
    model, tok = load_model(model_key)
    names = []
    for d in dirs:
        lora.load_adapter(model, d.name, d)
        names.append(d.name)
    grid = [float(x) for x in args.grid.split(",")] if args.grid else None
    conds = ev.conditions(names, grid)
    traits = sorted({m["config"]["trait"] for m in metas.values()})
    src = Path(next(iter(metas.values()))["sources"]["persona"])
    held = {t: data.trait_questions(src, t)[1] for t in traits}
    _, kl_held, _ = data.kl_reference(args.out, model_key)
    sampling = sampling_from_spec(model_key, presence=True)  # what the games sample with
    prompt_fn = prompt_for(tok, model_key)
    result = {"run": args.run_name, "adapters": names, "conditions": conds, "commits": commits(),
              "settings": {"samples": args.samples, "persona_tokens": args.persona_tokens, "acc_n": args.acc_n,
                           "acc_tokens": args.acc_tokens, "kl_tokens": args.kl_tokens, "grid": grid,
                           "sampling": sampling}}
    log(f"evaluate {args.run_name}: {names}, {len(conds)} conditions")
    result["kl"] = ev.kl_eval(model, tok, conds, kl_held, args.kl_tokens)
    if args.samples:
        gen = make_gen_fn(model, tok, prompt_fn, sampling, args.persona_tokens, 0, args.gen_batch)
        result["persona"] = ev.persona_eval(conds, held, traits, args.samples, gen, make_score_batch(args))
    if args.acc_n:
        refs = kl_held[:args.acc_n]
        gen = make_gen_fn(model, tok, prompt_fn, sampling, args.acc_tokens, 1, args.gen_batch)
        result["accuracy"] = ev.acc_eval(conds, refs, pool_options(POOL), gen)
    write_json(root / "eval.json", result)
    ev.print_table(result)
    wb = Wandb(args.wandb_project, args.wandb_entity, f"{args.run_name}/eval", result["settings"], args.run_name)
    if wb.run is not None:
        import wandb
        rows = []
        for c in conds:
            p = result.get("persona", {}).get(c, {})
            a = result.get("accuracy", {}).get(c, {})
            rows.append([c, *[p.get("expression", {}).get(t, {}).get("mean") for t in traits],
                         p.get("coherence", {}).get("mean"), result["kl"][c]["mean"], a.get("accuracy")])
        wb.log({"eval": wandb.Table(columns=["condition", *traits, "coherence", "kl", "accuracy"], data=rows)})
    wb.finish()
    log(f"wrote {root / 'eval.json'}")
    return 0


# ---------------------------------------------------------------- prepare / reference / accuracy / gate


def cmd_prepare(args) -> int:
    from perslora.tasks import prepare_gsm8k
    for p in prepare_gsm8k(args.out):
        log(f"wrote {p}")
    return 0


def refs_path(out: Path, task: str) -> Path:
    return Path(out) / "lora" / "refs" / f"{task}.jsonl"


def cmd_reference(args) -> int:
    """Base answers (games' sampling incl. presence penalty, up to --max-tokens) to a task's questions, --samples
    per question; the finished, parsed (and with --correct-only correct) ones become KL references."""
    from perslora import accuracy as acc
    from perslora.rollout import generate
    from perslora.tasks import load_task

    task = load_task(args.task, args.out, args.n or None)
    if args.task == "supergpqa-pool":  # keep the KL eval's 100 held-out pool questions out of training
        _, held, _ = data.kl_reference(args.out, args.model)
        task = task.without({r.id for r in held})
    model, tok = load_model(args.model)
    sampling = sampling_from_spec(args.model, presence=True)
    rs = generate(model, tok, task.items, prompt_for(tok, args.model), args.samples, args.max_tokens, sampling,
                  seed=args.seed, batch=args.gen_batch)
    rows = acc.reference_rows(task, rs, args.correct_only)
    p = refs_path(args.out, args.task)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    scored = acc.score_answers(task, rs)["summary"]
    write_json(p.with_suffix(".meta.json"), {"task": args.task, "questions": len(task.items),
                                             "samples": args.samples, "max_tokens": args.max_tokens,
                                             "sampling": sampling, "correct_only": args.correct_only,
                                             "kept": len(rows), "base": scored, "commits": commits()})
    log(f"{args.task}: {len(task.items)} questions x {args.samples}: acc {scored['accuracy']:.3f} unparsed "
        f"{scored['unparsed']:.3f} cut {scored['cut']:.3f}; kept {len(rows)} references -> {p}")
    return 0


def acc_path(out: Path, run_name: str, task: str, spec: str) -> Path:
    """base is shared by all runs (it does not depend on one)."""
    from perslora.accuracy import cond_label
    if spec == "base":
        return Path(out) / "lora" / "_base" / "acc" / task / "base.json"
    return run_root(out, run_name) / "acc" / task / f"{cond_label(spec)}.json"


def cmd_accuracy(args) -> int:
    from perslora import accuracy as acc
    from perslora.rollout import generate
    from perslora.tasks import load_task

    task = load_task(args.task, args.out, args.n or None)
    root = run_root(args.out, args.run_name)
    todo = [s for s in args.cond if args.force or not acc_path(args.out, args.run_name, args.task, s).exists()]
    for s in args.cond:
        if s not in todo:
            log(f"{s}: {acc_path(args.out, args.run_name, args.task, s)} exists (--force to redo)")
    if not todo:
        return 0
    model, tok = load_model(args.model)
    loaded = set()
    for spec in todo:
        for name, ck, _ in acc.parse_cond(spec):
            key = acc.adapter_key(name, ck)
            if key in loaded:
                continue
            d = root / name / "ckpt" / ck if ck else root / name
            if not (d / "adapter.json").exists():
                raise SystemExit(f"{d} has no adapter")
            lora.load_adapter(model, key, d)
            loaded.add(key)
    sampling = sampling_from_spec(args.model, presence=True)
    prompt_fn = prompt_for(tok, args.model)
    settings = {"task": args.task, "n": len(task.items), "max_tokens": args.max_tokens, "seed": args.seed,
                "sampling": sampling, "model": args.model}
    for spec in todo:
        t0 = time.time()
        lora.set_scales(model, acc.scales_of(spec))
        try:
            rs = generate(model, tok, task.items, prompt_fn, 1, args.max_tokens, sampling, seed=args.seed,
                          batch=args.gen_batch)
        finally:
            lora.set_scales(model, {})
        res = {"spec": spec, "run": args.run_name if spec != "base" else None, "settings": settings,
               **acc.score_answers(task, rs), "commits": commits(), "seconds": round(time.time() - t0)}
        p = acc_path(args.out, args.run_name, args.task, spec)
        write_json(p, res)
        if args.keep_texts:
            with gzip.open(p.with_suffix(".texts.jsonl.gz"), "wt", encoding="utf-8") as f:
                for r in rs:
                    f.write(json.dumps({"qid": r.qid, "text": r.text}) + "\n")
        s = res["summary"]
        log(f"{args.task} {spec}: acc {s['accuracy']:.3f} unparsed {s['unparsed']:.3f} cut {s['cut']:.3f} "
            f"tokens {s['tokens_mean']:.0f} ({res['seconds']} s) -> {p}")
    return 0


def cmd_gate(args) -> int:
    from perslora import accuracy as acc

    out_all = {}
    for task in args.task:
        if not (run_root(args.out, args.run_name) / "acc" / task).exists():
            log(f"{task}: no results for {args.run_name}, skipped")
            continue
        bp = acc_path(args.out, args.run_name, task, "base")
        if not bp.exists():
            raise SystemExit(f"{bp} missing: run `accuracy --cond base --task {task}` first")
        base = json.loads(bp.read_text(encoding="utf-8"))
        rows = {}
        for p in sorted((run_root(args.out, args.run_name) / "acc" / task).glob("*.json")):
            c = json.loads(p.read_text(encoding="utf-8"))
            acc.check_settings(base, c)
            rows[c["spec"]] = (c["summary"], acc.gate(base, c, args.delta))
        print(acc.format_gate(task, base, rows))
        out_all[task] = {"base": base["summary"], "delta": args.delta,
                         "conditions": {s: {"summary": sm, "gate": g} for s, (sm, g) in rows.items()}}
    write_json(run_root(args.out, args.run_name) / "gate.json", out_all)
    return 0


def cmd_report(args) -> int:
    from perslora import accuracy as acc
    from perslora import report

    bases = {}
    for t in ("supergpqa", "gsm8k"):
        p = acc_path(args.out, "", t, "base")
        if p.exists():
            bases[t] = json.loads(p.read_text(encoding="utf-8"))
    collected = {}
    for name in args.runs:
        root = run_root(args.out, name)
        if not root.exists():
            print(f"=== run {name}: {root} missing\n")
            continue
        steps, ev, accs = report.load_run(root)
        for t, res in accs.items():
            for spec in list(res):
                try:
                    acc.check_settings(bases[t], res[spec]) if t in bases else None
                except SystemExit as e:
                    print(f"[{name} {t} {spec}: left out, {e}]")
                    del res[spec]
        print(report.run_report(name, steps, ev, accs, bases, args.delta) + "\n")
        collected[name] = (ev, report.run_gates(accs, bases, args.delta))
    print(report.summary(collected))
    return 0


# ---------------------------------------------------------------- gpucheck

GPUCHECK_QUESTIONS = ("Hi.", "What is the capital of France? Answer in one sentence.",
                      "Give me three tips for a job interview tomorrow morning, briefly.",
                      "Explain in two sentences why the sky is blue, and say whether you are sure.")


def cmd_gpucheck(args) -> int:
    """Before a real run: (1) LoRA wraps 3 MLP linears per layer; (2) the stop ids include <|im_end|> and answers
    stop well before the cap; (3) greedy batched (left-padded) generation equals one-by-one generation."""
    from perslora.rollout import _eos_ids, generate

    model, tok = load_model(args.model)
    paths = lora.attach(model, "check")
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or cfg.text_config.num_hidden_layers
    ok = True
    print(f"wrapped {len(paths)} modules, {n_layers} layers (expect {3 * n_layers}); e.g. {paths[:3]}")
    ok &= len(paths) == 3 * n_layers
    eos = _eos_ids(model, tok)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    print(f"stop ids {sorted(eos)}; <|im_end|> = {im_end}")
    ok &= im_end in eos
    prompt_fn = prompt_for(tok, args.model)
    items = [(f"q{i}", q) for i, q in enumerate(GPUCHECK_QUESTIONS)]
    greedy = {"temperature": 0}
    batched = generate(model, tok, items, prompt_fn, 1, args.max_new_tokens, greedy, seed=0, batch=len(items))
    single = [generate(model, tok, [it], prompt_fn, 1, args.max_new_tokens, greedy, seed=0, batch=1)[0]
              for it in items]
    for b, s1 in zip(batched, single):
        same = b.answer_ids == s1.answer_ids
        ok &= same
        print(f"{b.qid}: batched == single: {same}; {len(b.answer_ids)} tokens, ends with stop id: "
              f"{bool(b.answer_ids) and b.answer_ids[-1] in eos}; {b.text[:100]!r}")
    print("GPUCHECK OK" if ok else "GPUCHECK FAILED")
    return 0 if ok else 1


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
    p.add_argument("--kl-refs", type=Path, nargs="+", default=None,
                   help="KL reference files (`reference` output); default PersVecTrain's 512-token pool answers")
    p.add_argument("--reason-task", default="gsm8k-train", help="task of the reasoning rollouts")
    p.add_argument("--reason-n", type=int, default=0, help="first N questions of the reasoning task (0: all)")
    p.add_argument("--leak-traits", nargs="+", default=None,
                   help="traits whose expression --c-leak penalises (default: the other --traits)")
    for f in _cfg_fields():
        flag = "--" + f.name.replace("_", "-")
        if f.name == "lambda_kl":
            p.add_argument(flag, type=float, default=None, help="fixed KL weight (disables the budget)")
        else:
            p.add_argument(flag, type=type(f.default), default=f.default)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="persona / KL / accuracy of a run's adapters and their sums")
    _add_common(p)
    p.add_argument("--samples", type=int, default=4, help="answers per held-out persona question (0: skip)")
    p.add_argument("--persona-tokens", type=int, default=512)
    p.add_argument("--acc-n", type=int, default=0,
                   help="held-out SuperGPQA pool questions (0: skip; the `accuracy` command is the real check)")
    p.add_argument("--acc-tokens", type=int, default=2048)
    p.add_argument("--kl-tokens", type=int, default=512)
    p.add_argument("--gen-batch", type=int, default=64)
    p.add_argument("--grid", default=None, help="scales for two adapters, e.g. 0,0.5,1: every pair in grid x grid")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("prepare", help="download GSM8K into <out>/lora/data (needs internet)")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("reference", help="base answers to a task as KL references")
    p.add_argument("--model", default="qwen3.5-2b", choices=sorted(MODELS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", required=True, choices=("supergpqa-pool", "gsm8k-train"))
    p.add_argument("--n", type=int, default=0, help="first N questions (0: all)")
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--correct-only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen-batch", type=int, default=64)
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser("accuracy", help="accuracy of conditions on a reasoning task (paired with base)")
    _add_common(p)
    p.add_argument("--task", required=True, choices=("supergpqa", "gsm8k"))
    p.add_argument("--cond", nargs="+", required=True,
                   help="base | adapter[@step_N][:scale][+adapter...], e.g. assertive_pos:0.5+sarcastic_pos:0.5")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gen-batch", type=int, default=64)
    p.add_argument("--force", action="store_true", help="redo conditions that already have a result")
    p.add_argument("--keep-texts", action="store_true")
    p.set_defaults(func=cmd_accuracy)

    p = sub.add_parser("gate", help="non-inferiority of a run's accuracy results against base")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--task", nargs="+", default=["supergpqa", "gsm8k"])
    p.add_argument("--delta", type=float, default=0.03, help="largest accuracy drop accepted")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("report", help="training, persona, accuracy gates and composition grid of runs")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runs", nargs="+", default=["mvp", "kl", "onpol", "correct", "leak"])
    p.add_argument("--delta", type=float, default=0.03)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("gpucheck", help="LoRA wrap count, stop tokens, batched vs single greedy generation")
    p.add_argument("--model", default="qwen3.5-2b", choices=sorted(MODELS))
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.set_defaults(func=cmd_gpucheck)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
