"""Re-score protocol runs offline with the current scorer (EVAL_PROTOCOL.md): each prediction's submitted answer
(plus, for test_selection agent episodes, the environment's order trace) is scored against its ground truth
exactly as the environment scores a submission. The first rescore keeps the as-run value in `reward_asrun`.

    python scripts/rescore_protocol_runs.py [--results results] [--check]   # --check: report only, write nothing

Used when a scorer fix lands after a run (Stage-8 reward-noise audit) so every model is scored by one scorer;
units whose scorer did not change must reproduce their as-run rewards exactly (a determinism check).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import degenerate as D  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    db = D.ReleaseDB()
    cache: dict[tuple[str, str], dict] = {}
    for run in sorted(Path(a.results).glob("*__*__*__*__s*")):
        mf = json.loads((run / "manifest.json").read_text())
        task, split, arm = mf["task"], mf["split"], mf["arm"]
        key = (task, split)
        if key not in cache:
            cache[key] = {i["gt_id"]: i for i in db.instances(task, split)}
        preds = [json.loads(l) for l in (run / "predictions.jsonl").read_text().splitlines() if l.strip()]
        changed, before, after = 0, 0.0, 0.0
        for p in preds:
            asrun = p.get("reward_asrun", p["reward"])
            if p.get("error") or p.get("submission") is None and not p.get("forced"):
                new = 0.0 if p.get("error") else float(p["reward"])
            else:
                sub = dict(p.get("submission") or {})
                if task == "test_selection" and arm == "agent":
                    sub["tests_ordered"] = list(p.get("order_log") or [])   # the trace, never the claim
                m = D.score(db, task, [sub], [cache[key][p["gt_id"]]])
                new = float(m[D.PRIMARY_METRIC[task]])
            changed += abs(new - float(p["reward"])) > 1e-9
            before += asrun; after += new
            p["reward_asrun"] = asrun
            p["reward"] = new
        n = max(len(preds), 1)
        print(f"{run.name:70s} n={len(preds):4d} as-run {before / n:.3f} -> {after / n:.3f}  ({changed} changed)")
        if not a.check:
            (run / "predictions.jsonl").write_text("".join(json.dumps(p, default=str) + "\n" for p in preds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
