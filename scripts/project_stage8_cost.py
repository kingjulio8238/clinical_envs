"""Project the full Stage-8 cost from the atomic-unit smokes (results/smoke/*): measured $/episode per
(model, unit, arm) x the episodes the full run will sample (eval.protocol_run.sample_instances, so the paired
atypical sample and small units are counted exactly), with a margin. Uses the provider's billed cost where the
smoke recorded one (OpenRouter), else the list-price estimate. Exits 1 if a smoke unit is missing or had errors.

    python scripts/project_stage8_cost.py [--margin 1.5] [--smoke-root results/smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import protocol_run as R  # noqa: E402

UNITS = list(R.RUN_UNITS)
ABLATION = ["patient_diagnosis", "test_selection", "differential_diagnosis"]
PLAN = {  # (model, arm) -> (units, n requested per unit, account); mirrors scripts/run_stage8_panel.sh
    ("qwen3.5-9b", "agent"): (UNITS, 120, "openrouter"),
    ("qwen3.5-9b", "single"): (ABLATION, 120, "openrouter"),
    ("gpt-6-sol", "agent"): (UNITS, 40, "openai"),
}
OPTIONAL = {("kimi-k2.5", "agent"): (UNITS, 20, "openrouter (optional generator row)")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--smoke-root", default=str(ROOT / "results" / "smoke"))
    ap.add_argument("--optional", action="store_true", help="also project the optional Kimi generator row")
    a = ap.parse_args()
    db = R.shared_db()
    totals, problems, wall = defaultdict(float), [], defaultdict(float)
    plan = {**PLAN, **(OPTIONAL if a.optional else {})}
    print(f"{'model':11s} {'arm':6s} {'unit':22s} {'k':>2s} {'reward':>6s} {'tok/ep':>7s} {'steps':>5s} {'wall s':>6s} "
          f"{'$/ep':>7s} {'billed':>6s} {'planned':>7s} {'$ proj':>6s}")
    for (model, arm), (units, n, acct) in plan.items():
        for u in units:
            d = Path(a.smoke_root) / f"{model}__{u}__{arm}__public__s0" / "predictions.jsonl"
            if not d.exists():
                problems.append(f"{model} {arm} {u}: no smoke"); continue
            preds = [json.loads(l) for l in d.read_text().splitlines() if l.strip()]
            errs = [p["error"] for p in preds if p.get("error")]
            if errs:
                problems.append(f"{model} {arm} {u}: {len(errs)} errors, e.g. {errs[0][:160]}")
            k = len(preds)
            per = sum(p["cost_usd"] for p in preds) / k if k else 0.0
            billed = all(p.get("billed_usd") is not None for p in preds)
            planned = len(R.sample_instances(db, u, "public", n, 0))
            proj = per * planned
            totals[acct] += proj
            ws = sum(p.get("wall_s", 0) for p in preds) / max(k, 1)
            wall[(model, arm)] += ws * planned
            print(f"{model:11s} {arm:6s} {u:22s} {k:2d} {sum(p['reward'] for p in preds) / max(k, 1):6.2f} "
                  f"{sum(p['input_tokens'] + p['output_tokens'] for p in preds) // max(k, 1):7d} "
                  f"{sum(p['steps'] for p in preds) / max(k, 1):5.1f} {ws:6.0f} {per:7.4f} {'yes' if billed else 'list':>6s} "
                  f"{planned:7d} {proj:6.2f}")
    for acct, t in totals.items():
        print(f"TOTAL {acct}: ${t:.2f} at the measured rate, ${t * a.margin:.2f} with x{a.margin} margin")
    for (model, arm), s in wall.items():
        print(f"serial episode-seconds {model} {arm}: {s / 3600:.1f} h (divide by the worker count for wall clock)")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
