"""Patterns in the persona adapters' training rollouts (per run x adapter x 25-step bin), and next to them the
steering vectors' persona answers from PersVecTrain (same out dir: <out>/trained/<run>/). Reads only; no GPU.

    python3 scripts/rollout_patterns.py --out /scratch/inf0/user/nzukowsk/PersVecFast/out --dest patterns \
        [--runs mvp kl onpol correct leak] [--vec-runs solo_kl5 ...] [--no-vec]

Writes <dest>/rollout_patterns.txt (LoRA tables), <dest>/compare_vectors.txt (LoRA vs vectors per trait),
<dest>/rollout_patterns.json and <dest>/rollout_examples.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from perslora import patterns as pt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the experiment's out dir (holds lora/<run>/ and trained/<run>/)")
    ap.add_argument("--dest", default="patterns")
    ap.add_argument("--runs", nargs="*", default=None, help="LoRA runs (default: all but smoke)")
    ap.add_argument("--vec-runs", nargs="*", default=None, help="PersVecTrain runs (default: all with persona answers)")
    ap.add_argument("--no-vec", action="store_true")
    ap.add_argument("--bin", type=int, default=25)
    a = ap.parse_args()
    found = pt.find_adapters(Path(a.out), a.runs)
    if not found:
        raise SystemExit(f"no rollouts.jsonl.gz under {a.out}/lora/<run>/<adapter>/")
    results = {}
    for run, adapter, p in found:
        print(f"{run} / {adapter}: {p}", flush=True)
        results[(run, adapter)] = pt.analyse(pt.load_rollouts(p), a.bin)
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "rollout_patterns.txt").write_text(pt.render(results), encoding="utf-8")
    (dest / "rollout_examples.md").write_text(pt.render_examples(results), encoding="utf-8")
    (dest / "rollout_patterns.json").write_text(json.dumps(
        {f"{r}/{ad}": {k: v for k, v in res.items() if k not in ("examples", "last_texts")}
         for (r, ad), res in results.items()}, indent=1), encoding="utf-8")
    written = ["rollout_patterns.txt", "rollout_examples.md", "rollout_patterns.json"]
    if not a.no_vec:
        traits = sorted({ad.rsplit("_", 1)[0] for _, ad, _ in found})
        vec = pt.load_vec_answers(Path(a.out), a.vec_runs, traits)
        print(f"vector answers: {len(vec)} run x trait sets ({', '.join(sorted({r for r, _ in vec})) or 'none'})")
        (dest / "compare_vectors.txt").write_text(pt.render_compare(results, vec), encoding="utf-8")
        written.append("compare_vectors.txt")
    print(f"-> {dest}/: {', '.join(written)}")


if __name__ == "__main__":
    main()
