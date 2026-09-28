"""Group variance of the RL candidate on train prompts (RL readiness C4): k seeded samples per prompt with the training
sampling, the share of prompts whose k rewards are all equal (no GRPO signal), and a prompt filter for training.

    python scripts/group_variance.py --model qwen3.5-9b-local --per-unit 250 --k 8 [--units ...] [--workers 64]

Writes results/group_variance/<model>/samples.jsonl (one line per episode, resumable) and summary.json, and
results/group_variance/<model>/prompts.json: {"keep": [gt_id, ...]} = prompts with reward spread (std > 0.01) —
the `--prompts` input of scripts/train_rl.py. Runs against a local vLLM server (modal/vllm_eval.py) or any model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval import protocol_run as R  # noqa: E402
from eval.adapters import create_adapter  # noqa: E402
from eval.config import MODEL_REGISTRY  # noqa: E402
from eval.episode import LOCAL_LIMITS, EpisodeLimits  # noqa: E402
from eval.rl_rollout import TRAIN_UNITS  # noqa: E402

SPREAD = 0.01


def dedupe(samples: list[dict]) -> list[dict]:
    """One row per (unit, prompt, sample): the last one without an error, else the last one (resumed or merged files
    can hold a retried sample twice)."""
    best: dict[tuple, dict] = {}
    for s in samples:
        key = (s["task"], s["gt_id"], s["sample"])
        if key not in best or not s.get("error") or best[key].get("error"):
            best[key] = s
    return list(best.values())


def write_summary(out: Path, samples: list[dict], k: int, model: str, reward_version: str | None) -> dict:
    samples = dedupe(samples)
    summ = summarize([s for s in samples if not s.get("error")], k)
    summ.update(model=model, reward_version=reward_version, errors=sum(1 for s in samples if s.get("error")))
    (out / "summary.json").write_text(json.dumps({k_: v for k_, v in summ.items() if k_ != "keep"}, indent=1))
    (out / "prompts.json").write_text(json.dumps({"keep": summ["keep"], "k": k, "rule": f"reward std > {SPREAD}"}, indent=1))
    return summ


def summarize(samples: list[dict], k: int) -> dict:
    by = defaultdict(list)
    for s in samples:
        by[(s["task"], s["gt_id"])].append(float(s["reward"]))
    per_unit = defaultdict(lambda: {"prompts": 0, "complete": 0, "zero_variance": 0, "all_zero": 0, "all_high": 0, "mean": 0.0})
    keep = []
    for (task, gid), rs in by.items():
        u = per_unit[task]
        u["prompts"] += 1
        if len(rs) < k:
            continue
        u["complete"] += 1
        mu = sum(rs) / len(rs)
        sd = math.sqrt(sum((r - mu) ** 2 for r in rs) / len(rs))
        u["mean"] += mu
        if sd <= SPREAD:
            u["zero_variance"] += 1
            u["all_zero"] += mu <= SPREAD
            u["all_high"] += mu >= 1 - SPREAD
        else:
            keep.append(gid)
    for u in per_unit.values():
        c = max(u["complete"], 1)
        u["mean"] /= c
        u["zero_variance_share"] = u["zero_variance"] / c
    return {"k": k, "per_unit": dict(per_unit), "keep": sorted(keep), "n_keep": len(keep)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen3.5-9b-local", choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--units", default=",".join(TRAIN_UNITS))
    ap.add_argument("--per-unit", type=int, default=250)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "group_variance"))
    ap.add_argument("--shard", default=None, help="i/k: the i-th of k interleaved slices of each unit's prompts "
                                                  "(merge: scripts/sync_runs.py merge-gv)")
    ap.add_argument("--retry-errors", action="store_true", help="on resume, rerun samples that ended in an error")
    a = ap.parse_args(argv)
    si, sk = (int(x) for x in a.shard.split("/")) if a.shard else (0, 1)
    from eval import reward_version as RV
    reward = RV.require_frozen()
    out = Path(a.out) / a.model
    out.mkdir(parents=True, exist_ok=True)
    path = out / "samples.jsonl"
    prior = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    done = {(s["gt_id"], s["sample"]) for s in dedupe(prior) if not (a.retry_errors and s.get("error"))}
    db = R.shared_db()
    cfg = MODEL_REGISTRY[a.model]
    adapter = create_adapter(cfg)
    jobs = []
    from eval.rl_rollout import is_dev_patient
    for u in a.units.split(","):
        pool = [i for i in R.sample_instances(db, u, "train", 10 ** 6, a.seed) if not is_dev_patient(i["patient_id"])]
        for inst in pool[: a.per_unit][si::sk]:                # dev patients are never training prompts (P4)
            for s in range(a.k):
                if (inst["gt_id"], s) not in done:
                    jobs.append({**inst, "task": u, "sample": s})
    lock = threading.Lock()
    t0, n = time.time(), 0
    with path.open("a") as fh, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(R.run_episode, adapter, j, "agent", 40, (0.0, 0.0), 0,
                          EpisodeLimits.for_unit(j["task"], 40, per_turn=cfg.max_tokens, per_episode=LOCAL_LIMITS["per_episode"])): j
                for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            rec = f.result()
            row = {"task": j["task"], "gt_id": j["gt_id"], "sample": j["sample"], "reward": rec["reward"], "error": rec.get("error"),
                   "turns": rec.get("turns"), "output_tokens": rec.get("output_tokens"), "signals": rec.get("signals")}
            with lock:
                fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
                n += 1
                if n % 100 == 0 or n == len(jobs):
                    el = time.time() - t0
                    print(f"{n}/{len(jobs)} episodes ({100 * n // max(len(jobs), 1)}%), {el / 60:.1f} min, ETA {(len(jobs) - n) * el / n / 60:.1f} min", flush=True)
    samples = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    summ = write_summary(out, samples, a.k, a.model, reward["reward_version"])
    print(json.dumps({k: v for k, v in summ.items() if k != "keep"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
