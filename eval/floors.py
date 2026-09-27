"""Floors and ceilings for every scoring unit, and score normalization against them.

floor   = the best reward any degenerate policy in eval.degenerate.POLICIES obtains
ceiling = the reward of the label oracle
normalized(score) = (score - floor) / (ceiling - floor)

Absolute numbers on this benchmark are not interpretable without this: on the released data a
content-blind ranking scores P@5 0.97 and a regex that copies the chart's problem list scores
weighted F1 0.83 (audit/FINDINGS.md). A unit whose floor reaches its ceiling has no headroom and
is reported as such (normalized = None).

    python -m eval.floors --split public            # print the table
    python -m eval.floors --split public --write    # refresh eval/floors.json
    python -m eval.floors --split public --check    # exit 1 if eval/floors.json is stale

The committed eval/floors.json is what eval.report and eval.score_one consumers can read without
recomputing; eval/tests/test_reward_hacking.py asserts it is current.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from eval import degenerate as D

FLOORS_PATH = Path(__file__).with_name("floors.json")

METRICS: dict[str, list[str]] = {
    "patient_diagnosis": ["weighted_problem_list_f1_neutral"],
    "evidence_retrieval": ["precision_5", "ndcg_10"],
    "context_summarization": ["clinical_f1"],
    "specialty_involved": ["conditioned_f1"],
    "specialty_absent": ["abstention_accuracy"],
    "imaging_indication": ["clinical_question_concept_f1"],
}
RANDOM_SEEDS = 5
TOL = 1e-9
NEGLIGIBLE_HEADROOM = 0.05
TOOL_POLICIES = {"echo_problem_list_tool"}
"""Policies that need the simulator's tool API rather than the chart. They are recorded as the
`agentic_floor` and excluded from `floor`, which is what single-turn scores are normalized against."""


def _batch(db: D.ReleaseDB, task: str, name: str, insts: list[D.Instance]):
    """Predictions and matching instances for one policy. `random` is averaged over seeds by
    repeating every instance once per seed (compute_all_metrics averages over the batch)."""
    if name == "random":
        preds, rep = [], []
        for s in range(RANDOM_SEEDS):
            preds += [D.retr_random(db, i, seed=s) for i in insts]
            rep += insts
        return preds, rep
    return D.run_policy(db, task, name, insts), insts


def compute_floors(db: D.ReleaseDB, split: str) -> dict:
    """{unit: {"n", "metrics": {metric: {"ceiling", "floor", "floor_policy", "policies": {...}}}}}"""
    out = {}
    for task in D.TASKS:
        insts = db.instances(task, split)
        if not insts:
            continue
        unit = {"n": len(insts), "metrics": {}}
        scores = {name: D.score(db, task, *_batch(db, task, name, insts)) for name in D.POLICIES[task]}
        oracle = D.score(db, task, [D.oracle(db, i) for i in insts], insts)
        for metric in METRICS[task]:
            pol = {name: float(m[metric]) for name, m in scores.items()}
            chart_only = {k: v for k, v in pol.items() if k not in TOOL_POLICIES}
            floor_policy = max(chart_only, key=chart_only.get)
            entry = {
                "ceiling": float(oracle[metric]),
                "floor": chart_only[floor_policy],
                "floor_policy": floor_policy,
                "policies": pol,
            }
            entry["headroom"] = entry["ceiling"] - entry["floor"]
            tool = {k: v for k, v in pol.items() if k in TOOL_POLICIES}
            if tool:
                entry["agentic_floor_policy"] = max(tool, key=tool.get)
                entry["agentic_floor"] = tool[entry["agentic_floor_policy"]]
            unit["metrics"][metric] = entry
        out[task] = unit
    return out


def normalize(value: float, task: str, metric: str | None = None, split: str = "public",
              floors: dict | None = None) -> float | None:
    """(value - floor) / (ceiling - floor); None when the unit has no headroom or is unknown."""
    floors = floors if floors is not None else load()
    unit = floors.get("splits", {}).get(split, {}).get(task)
    if not unit:
        return None
    entry = unit["metrics"].get(metric or METRICS[task][0])
    if not entry or entry["headroom"] <= TOL:
        return None
    return (value - entry["floor"]) / (entry["ceiling"] - entry["floor"])


def load(path: Path = FLOORS_PATH) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"splits": {}}


def matches(live: dict, stored: dict, tol: float = TOL) -> list[str]:
    """Human-readable differences between a live computation and a stored split block."""
    diffs = []
    for task, unit in live.items():
        s = stored.get(task)
        if not s:
            diffs.append(f"{task}: missing"); continue
        if s.get("n") != unit["n"]:
            diffs.append(f"{task}: n {s.get('n')} != {unit['n']}")
        for metric, e in unit["metrics"].items():
            se = s["metrics"].get(metric, {})
            for key in ("ceiling", "floor", "headroom"):
                if abs(se.get(key, math.nan) - e[key]) > tol:
                    diffs.append(f"{task}.{metric}.{key}: {se.get(key)} != {e[key]}")
            for name, v in e["policies"].items():
                if abs(se.get("policies", {}).get(name, math.nan) - v) > tol:
                    diffs.append(f"{task}.{metric}.{name}: {se.get('policies', {}).get(name)} != {v}")
    return diffs


def format_table(split: str, block: dict) -> str:
    lines = [f"split={split}", f"{'unit':22s} {'metric':32s} {'n':>5s} {'ceiling':>8s} {'floor':>7s}  floor policy", "-" * 100]
    for task, unit in block.items():
        for metric, e in unit["metrics"].items():
            head = e["headroom"]
            flag = "  (NO headroom)" if head <= TOL else f"  (negligible headroom {head:.3f})" if head < NEGLIGIBLE_HEADROOM else ""
            lines.append(f"{task:22s} {metric:32s} {unit['n']:5d} {e['ceiling']:8.3f} {e['floor']:7.3f}  {e['floor_policy']}{flag}")
            if "agentic_floor" in e:
                lines.append(f"{'':22s} {'':32s} {'':5s} {'':8s} {e['agentic_floor']:7.3f}  agentic floor: {e['agentic_floor_policy']} (needs the tool API)")
            for name, v in sorted(e["policies"].items(), key=lambda kv: -kv[1]):
                lines.append(f"{'':22s} {'':32s} {'':5s} {'':8s} {v:7.3f}    - {name}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", default="public", choices=["public", "heldout", "train"])
    ap.add_argument("--db", default=str(D.DEFAULT_DB))
    ap.add_argument("--write", action="store_true", help=f"update {FLOORS_PATH.name} for this split")
    ap.add_argument("--check", action="store_true", help=f"exit 1 if {FLOORS_PATH.name} is stale for this split")
    ap.add_argument("--json", action="store_true", help="print the block as JSON instead of a table")
    a = ap.parse_args(argv)

    db = D.ReleaseDB(a.db)
    live = compute_floors(db, a.split)
    print(json.dumps(live, indent=1) if a.json else format_table(a.split, live))

    if a.check:
        diffs = matches(live, load().get("splits", {}).get(a.split, {}))
        print("\n".join(["stale:"] + diffs) if diffs else f"{FLOORS_PATH.name} is current for split={a.split}")
        return 1 if diffs else 0
    if a.write:
        doc = load()
        doc.setdefault("splits", {})[a.split] = live
        doc["version"] = "1.3"
        doc["generated_by"] = "python -m eval.floors --write"
        FLOORS_PATH.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"wrote {FLOORS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
