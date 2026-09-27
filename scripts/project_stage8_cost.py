"""Project the full Stage-8 cost from the atomic-unit smokes (results/smoke/*): measured $/episode per
(model, unit, arm) × the planned episodes, with a margin; checks every smoke unit ran without errors.

    python scripts/project_stage8_cost.py [--margin 1.3]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import degenerate as D  # noqa: E402

PLAN = {  # (model, arm) -> (units, episodes per unit, provider)
    ("qwen3.5-9b", "agent"): (list(D.TASKS), 120, "openrouter"),
    ("qwen3.5-9b", "single"): (["patient_diagnosis", "test_selection", "differential_diagnosis"], 120, "openrouter"),
    ("glm-5.3-flash", "agent"): (list(D.TASKS), 120, "openrouter"),
    ("gpt-6-sol", "agent"): (list(D.TASKS), 40, "openai"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=1.3)
    a = ap.parse_args()
    db = D.ReleaseDB()
    avail = {t: len(db.instances(t, "public")) for t in D.TASKS}
    totals, problems = defaultdict(float), []
    print(f"{'model':14s} {'arm':6s} {'unit':24s} {'smoke n':>7s} {'reward':>6s} {'tok/ep':>7s} {'steps':>5s} {'$/ep':>8s} {'planned':>7s} {'$ proj':>7s}")
    for (model, arm), (units, n, provider) in PLAN.items():
        for u in units:
            d = ROOT / "results" / "smoke" / f"{model}__{u}__{arm}__public__s0" / "predictions.jsonl"
            if not d.exists():
                problems.append(f"{model} {arm} {u}: no smoke"); continue
            preds = [json.loads(l) for l in d.read_text().splitlines() if l.strip()]
            errs = [p["error"] for p in preds if p.get("error")]
            if errs:
                problems.append(f"{model} {arm} {u}: {len(errs)} errors, e.g. {errs[0][:160]}")
            k = len(preds)
            per = sum(p["cost_usd"] for p in preds) / k if k else 0.0
            planned = min(n, avail[u])
            proj = per * planned
            totals[provider] += proj
            print(f"{model:14s} {arm:6s} {u:24s} {k:7d} {sum(p['reward'] for p in preds)/max(k,1):6.2f} "
                  f"{sum(p['input_tokens']+p['output_tokens'] for p in preds)//max(k,1):7d} {sum(p['steps'] for p in preds)/max(k,1):5.1f} "
                  f"{per:8.4f} {planned:7d} {proj:7.2f}")
    for prov, t in totals.items():
        print(f"TOTAL {prov}: ${t:.2f} measured-rate projection, ${t * a.margin:.2f} with x{a.margin} margin")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
