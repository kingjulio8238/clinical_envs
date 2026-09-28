"""The evaluation protocol's arithmetic and report (EVAL_PROTOCOL.md).

    python -m eval.protocol report [--results results] [--split public]     -> results/leaderboard.md, summary.json
    python -m eval.protocol estimate --model gpt-5.4-mini --n 120 ...       -> projected cost from measured tokens

Normalization uses eval/floors.json (floor = best degenerate policy, ceiling = oracle); intervals are
bootstrap over instances; model comparisons are paired on identical instances; the data generator (Kimi)
is reported in a separate, unranked row.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from eval import floors as F

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
GENERATOR_MODELS = {"kimi-2.5", "kimi-2.5-or", "kimi-2.5-thinking", "kimi-2.5-thinking-or", "kimi-k2.5",
                    "kimi-2.5-improved", "kimi-2.5-thinking-improved"}
"""Models that generated the benchmark's labels or text: evaluated, reported, never ranked."""
BOOTSTRAP_N = 2000
SEED = 0


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n: int = BOOTSTRAP_N, seed: int = SEED, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    lo = means[int(math.floor(alpha / 2 * n))]
    hi = means[min(n - 1, int(math.ceil((1 - alpha / 2) * n)) - 1)]
    return (lo, hi)


def paired_diff(a: dict[int, float], b: dict[int, float], n: int = BOOTSTRAP_N, seed: int = SEED) -> dict:
    """Mean of a − b over the instances both scored, with a bootstrap interval over those instances."""
    common = sorted(set(a) & set(b))
    if not common:
        return {"n": 0, "mean": float("nan"), "ci": (float("nan"), float("nan")), "significant": False}
    diffs = [a[g] - b[g] for g in common]
    lo, hi = bootstrap_ci(diffs, n, seed)
    return {"n": len(common), "mean": sum(diffs) / len(diffs), "ci": (lo, hi), "significant": lo > 0 or hi < 0}


def normalize(task: str, split: str, raw: float, floors: dict | None = None) -> float | None:
    """(raw − floor) / (ceiling − floor) from eval/floors.json for the task's scoring unit."""
    doc = floors or F.load()
    unit = task
    v = F.normalize(raw, unit, None, split, doc)
    return v


# ---------------------------------------------------------------------------
# runs on disk
# ---------------------------------------------------------------------------

def load_runs(results_root: Path = RESULTS) -> list[dict]:
    runs = []
    for d in sorted(results_root.glob("*/")):
        m, p = d / "manifest.json", d / "predictions.jsonl"
        if not (m.exists() and p.exists()):
            continue
        manifest = json.loads(m.read_text())
        preds = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        runs.append({"dir": d.name, "manifest": manifest, "predictions": preds,
                     "rewards": {r["gt_id"]: float(r.get("reward") or 0.0) for r in preds}})
    return runs


def _fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _ci(ci: tuple[float, float]) -> str:
    return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _named_coded(preds: list[dict]) -> dict:
    """RL readiness A2: mean share of reference diagnoses named / coded exactly (diagnosis-scored units only;
    failed episodes count 0)."""
    vals = [(p.get("metrics") or {}) for p in preds]
    if not preds or not any("diagnosis_named" in v for v in vals):
        return {"named": None, "coded": None}
    return {"named": sum(float(v.get("diagnosis_named") or 0) for v in vals) / len(preds),
            "coded": sum(float(v.get("diagnosis_coded") or 0) for v in vals) / len(preds)}


def leaderboard(runs: list[dict], split: str = "public", floors: dict | None = None) -> tuple[str, dict]:
    """Markdown tables per (task, arm) plus a JSON summary."""
    doc = floors or F.load()
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        mf = r["manifest"]
        if mf.get("split") != split:
            continue
        cells[(mf["task"], mf.get("arm", "agent"))].append(r)
    out_md = [f"# Leaderboard — split `{split}` (EVAL_PROTOCOL.md)\n",
              "Rankings use the normalized score `(raw − floor) / (ceiling − floor)`; intervals are 95% bootstrap over "
              "instances; Δ vs best is a paired difference on the instances both models scored. Failures score 0 and are "
              "counted in `n`. The data generator (Kimi) is shown below the line and is not ranked.\n"]
    summary: dict = {"split": split, "cells": []}
    for (task, arm), rs in sorted(cells.items()):
        rows = []
        for r in rs:
            mf = r["manifest"]
            vals = list(r["rewards"].values())
            if not vals:
                continue
            raw = sum(vals) / len(vals)
            norm = F.normalize(raw, task, None, split, doc)
            lo, hi = bootstrap_ci(vals)
            nlo, nhi = (F.normalize(lo, task, None, split, doc), F.normalize(hi, task, None, split, doc)) if norm is not None else (None, None)
            rows.append({"model": mf["model"], "generator": mf["model"] in GENERATOR_MODELS, "n": len(vals), "raw": raw, "raw_ci": (lo, hi),
                         "norm": norm, "norm_ci": (nlo, nhi), "errors": sum(1 for p in r["predictions"] if p.get("error")),
                         "forced": sum(1 for p in r["predictions"] if p.get("forced")), "cost_usd": sum(float(p.get("cost_usd") or 0) for p in r["predictions"]),
                         "steps": (sum(int(p.get("steps") or 0) for p in r["predictions"]) / len(vals)), "rewards": r["rewards"], "dir": r["dir"],
                         **_named_coded(r["predictions"])})
        ranked = sorted([x for x in rows if not x["generator"]], key=lambda x: -(x["norm"] if x["norm"] is not None else x["raw"]))
        gens = [x for x in rows if x["generator"]]
        unit = doc.get("splits", {}).get(split, {}).get(task, {})
        floor = unit.get("metrics", {}).get(F.METRICS.get(task, [None])[0], {}).get("floor") if unit else None
        ceiling = unit.get("metrics", {}).get(F.METRICS.get(task, [None])[0], {}).get("ceiling") if unit else None
        out_md.append(f"\n## {task} — arm `{arm}`  (floor {_fmt(floor)}, ceiling {_fmt(ceiling)})\n")
        out_md.append("| model | n | raw [95% CI] | normalized [95% CI] | Δ vs best (paired) | named | coded | errors | forced | steps | cost $ |")
        out_md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        best = ranked[0] if ranked else None
        for x in ranked + ([None] if gens else []) + gens:
            if x is None:
                out_md.append("| *generator (not ranked)* | | | | | | | | | | |")
                continue
            d = paired_diff(x["rewards"], best["rewards"]) if best and x is not best else None
            dtxt = "best" if x is best else (f"{_fmt(d['mean'])} {_ci(d['ci'])}{' *' if d['significant'] else ''}" if d else "—")
            out_md.append(f"| {x['model']} | {x['n']} | {_fmt(x['raw'])} {_ci(x['raw_ci'])} | {_fmt(x['norm'])} "
                          f"{_ci(x['norm_ci']) if x['norm'] is not None else '—'} | {dtxt} | {_fmt(x.get('named'))} | {_fmt(x.get('coded'))} | "
                          f"{x['errors']} | {x['forced']} | {x['steps']:.1f} | {x['cost_usd']:.2f} |")
            summary["cells"].append({k: v for k, v in x.items() if k != "rewards"} | {"task": task, "arm": arm, "delta_vs_best": d})
    return "\n".join(out_md) + "\n", summary


def ablations(runs: list[dict], split: str = "public") -> str:
    """Paired tables for the protocol's ablations: tools vs no tools; typical vs atypical."""
    by = defaultdict(dict)
    for r in runs:
        mf = r["manifest"]
        if mf.get("split") == split:
            by[(mf["model"], mf["task"])][mf.get("arm", "agent")] = r
    lines = ["\n# Ablations (paired, same instances)\n", "| model | ablation | n | Δ mean [95% CI] | significant |", "|---|---|---|---|---|"]
    for (model, task), arms in sorted(by.items()):
        if "agent" in arms and "single" in arms:
            d = paired_diff(arms["agent"]["rewards"], arms["single"]["rewards"])
            lines.append(f"| {model} | {task}: tools − no tools | {d['n']} | {_fmt(d['mean'])} {_ci(d['ci'])} | {'yes' if d['significant'] else 'no'} |")
    # typical vs atypical share parent instances: match by parent_gt_id recorded in predictions
    for (model, task), arms in sorted(by.items()):
        if task != "atypical_diagnosis":
            continue
        base = by.get((model, "patient_diagnosis"), {})
        for arm in arms:
            if arm not in base:
                continue
            atyp = {p.get("parent_gt_id"): float(p.get("reward") or 0) for p in arms[arm]["predictions"] if p.get("parent_gt_id")}
            typ = base[arm]["rewards"]
            d = paired_diff(atyp, typ)
            lines.append(f"| {model} | atypical − typical diagnosis (`{arm}`) | {d['n']} | {_fmt(d['mean'])} {_ci(d['ci'])} | {'yes' if d['significant'] else 'no'} |")
    return "\n".join(lines) + "\n" if len(lines) > 3 else ""


def report(results_root: Path = RESULTS, split: str = "public") -> Path:
    runs = load_runs(results_root)
    md, summary = leaderboard(runs, split)
    md += ablations(runs, split)
    (results_root / "leaderboard.md").write_text(md)
    (results_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return results_root / "leaderboard.md"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report"); r.add_argument("--results", default=str(RESULTS)); r.add_argument("--split", default="public")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        path = report(Path(a.results), a.split)
        print(path.read_text())
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
