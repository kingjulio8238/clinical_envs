"""Cost and time projection from a measured GPU job (G2 of audit/STAGE_C_TODO.md; P9 / P11).

    python gpu/project.py results/modal/founders-78536/g1 --episodes 11699 [--workers-scale 2] [--gpu H100]

Reads the job a `gpu/vllm_eval.py` run left on the results volume (pulled with scripts/sync_runs.py): job.json (server
start-up, total time), partN.json (seconds per client command) and the predictions under each part. The measured
rate is the largest part's (the one closest to steady state), in episodes per GPU-hour and generated tokens per second.
Prints the projection for `--episodes` more episodes — hours, dollars, and the `--minutes` timeout to launch with
(1.5x) — so `gpu/launch.sh <profile> <dollars> ... --minutes <timeout>` gets numbers from a measurement, not a guess.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from budget import GPU_RATE  # noqa: E402

MARGIN = 1.5


def measure(job_dir: Path) -> dict:
    job = json.loads((job_dir / "job.json").read_text())
    parts = []
    for pj in sorted(job_dir.glob("part*.json")):
        info = json.loads(pj.read_text())
        preds = [json.loads(line) for f in (job_dir / pj.stem).rglob("predictions.jsonl")
                 for line in f.read_text().splitlines() if line.strip()]
        parts.append({"part": pj.stem, "seconds": float(info["seconds"]), "episodes": len(preds),
                      "output_tokens": sum(int(p.get("output_tokens") or 0) for p in preds),
                      "errors": sum(1 for p in preds if p.get("error")), "rc": info.get("rc")})
    if not parts or not any(p["episodes"] for p in parts):
        raise SystemExit(f"{job_dir}: no measured episodes (job.json {job})")
    big = max(parts, key=lambda p: p["episodes"])
    return {"startup_s": float(job.get("vllm_ready_s") or 0), "total_s": float(job["total_s"]), "parts": parts,
            "episodes_per_hour": big["episodes"] / big["seconds"] * 3600,
            "tokens_per_s": big["output_tokens"] / big["seconds"],
            "tokens_per_episode": big["output_tokens"] / big["episodes"], "basis": big["part"]}


def project(m: dict, episodes: int, gpu: str = "H100", workers_scale: float = 1.0, tokens_per_episode: float | None = None) -> dict:
    """`workers_scale`: expected throughput gain from more concurrent episodes than the measurement ran (an assumption,
    printed); `tokens_per_episode`: when the target units generate more / less than the measured ones."""
    rate = m["episodes_per_hour"] * workers_scale
    if tokens_per_episode:
        rate *= m["tokens_per_episode"] / tokens_per_episode
    hours = m["startup_s"] / 3600 + episodes / rate
    cost = hours * GPU_RATE[gpu]
    return {"episodes": episodes, "episodes_per_hour": round(rate, 1), "hours": round(hours, 2), "usd": round(cost, 2),
            "timeout_minutes": int(hours * 60 * MARGIN) + 10, "usd_at_timeout": round((hours * MARGIN + 1 / 6) * GPU_RATE[gpu], 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("job_dir")
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--gpu", default="H100", choices=sorted(GPU_RATE))
    ap.add_argument("--workers-scale", type=float, default=1.0)
    ap.add_argument("--tokens-per-episode", type=float, default=None)
    a = ap.parse_args(argv)
    m = measure(Path(a.job_dir))
    for p in m["parts"]:
        print(f"{p['part']}: {p['episodes']} episodes in {p['seconds']:.0f}s, {p['output_tokens']} output tokens, "
              f"{p['errors']} errors, rc {p['rc']}")
    print(f"measured ({m['basis']}): {m['episodes_per_hour']:.0f} episodes/GPU-hour, {m['tokens_per_s']:.0f} output tokens/s, "
          f"{m['tokens_per_episode']:.0f} tokens/episode; server start-up {m['startup_s']:.0f}s")
    pr = project(m, a.episodes, a.gpu, a.workers_scale, a.tokens_per_episode)
    print(f"projection: {pr['episodes']} episodes → {pr['hours']} GPU-hours, ${pr['usd']} "
          f"(launch with --minutes {pr['timeout_minutes']}; worst case at the timeout ${pr['usd_at_timeout']})"
          + (f"; assumes {a.workers_scale}x the measured throughput" if a.workers_scale != 1 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
