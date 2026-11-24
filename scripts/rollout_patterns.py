"""Patterns in the persona adapters' training rollouts, per run x adapter x 25-step bin (reads only; no GPU).

    python scripts/rollout_patterns.py --out /scratch/inf0/user/nzukowsk/PersVecFast/out --dest patterns \
        [--runs mvp kl onpol correct leak]

Writes <dest>/rollout_patterns.txt (tables), <dest>/rollout_patterns.json and <dest>/rollout_examples.md.
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
    ap.add_argument("--out", required=True, help="the experiment's out dir (holds lora/<run>/<adapter>/)")
    ap.add_argument("--dest", default="patterns")
    ap.add_argument("--runs", nargs="*", default=None)
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
        {f"{r}/{ad}": {k: v for k, v in res.items() if k != "examples"} for (r, ad), res in results.items()},
        indent=1), encoding="utf-8")
    print(f"-> {dest}/rollout_patterns.txt, rollout_examples.md, rollout_patterns.json")


if __name__ == "__main__":
    main()
