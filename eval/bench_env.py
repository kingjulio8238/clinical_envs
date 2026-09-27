"""Throughput harness for the reset-and-step environment (ROADMAP Stage 5).

The same scripted episode is played through the HTTP env (`eval.env_client.SyntheticHospitalEnv`) or the
in-process env (`eval.local_env.LocalEnv`) on the same instances, and every call is timed. Output is a JSON
file with per-tool latency percentiles, episodes/s and steps/s per concurrency level, so before/after is a
diff of two files.

    python -m eval.bench_env http  --episodes 32 --concurrency 1,4,16,64 --out audit/bench/http_baseline.json
    python -m eval.bench_env local --episodes 200 --workers 1,2,4,8      --out audit/bench/local.json
    python -m eval.bench_env components --reps 200                        # HTTP: framework / Redis / Postgres / scorer floors

The scripted episode: reset → open_chart → view_encounters → view_encounter_detail (first visible encounter)
→ search_chart("pain") → view_results(labs) → view_problem_list → view_medications → oracle submit.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TASKS = ["patient_diagnosis", "context_summarization", "evidence_retrieval", "imaging_indication"]
DEFAULT_URL = os.environ.get("SH_ENV_URL", "http://127.0.0.1:8000")
DEFAULT_TOKEN = os.environ.get("EPIC_SIM_SCORER_TOKEN", "dev-scorer-token-change-in-production")


# ---------------------------------------------------------------------------
# instance sample (shared by both paths)
# ---------------------------------------------------------------------------

def sample_gt_ids(n: int, split: str = "public", seed: int = 0) -> list[int]:
    """`n` public instances, round-robin over the four tasks, deterministic."""
    import random
    from eval import degenerate as D
    db = D.ReleaseDB()
    per_task = {t: [i["gt_id"] for i in db.instances(t, split)] for t in TASKS}
    rng = random.Random(seed)
    for v in per_task.values():
        rng.shuffle(v)
    out, k = [], 0
    while len(out) < n:
        t = TASKS[k % len(TASKS)]
        if per_task[t]:
            out.append(per_task[t].pop())
        k += 1
    return out


# ---------------------------------------------------------------------------
# the scripted episode
# ---------------------------------------------------------------------------

def play(env, gt_id: int, oracle_fn) -> list[tuple[str, float, bool]]:
    """Play one episode; return [(call_name, seconds, ok)]. `oracle_fn(env) -> (tool, args)`."""
    timings: list[tuple[str, float, bool]] = []

    def timed(name, fn):
        t = time.perf_counter()
        try:
            r = fn()
            ok = not (isinstance(r, tuple) and isinstance(r[0], dict) and "error" in r[0])
        except Exception:  # noqa: BLE001 — a failed call is a measurement, not a crash
            r, ok = None, False
        timings.append((name, time.perf_counter() - t, ok))
        return r

    ro = timed("reset", lambda: env.reset(gt_id=gt_id))
    pid = ro.patient_id
    timed("open_chart", lambda: env.step("open_chart", {"patient_id": pid}))
    encs = timed("view_encounters", lambda: env.step("view_encounters", {"patient_id": pid}))
    eid = None
    if encs and isinstance(encs[0], list) and encs[0]:
        eid = encs[0][0].get("encounter_id")
    timed("view_encounter_detail", lambda: env.step("view_encounter_detail", {"encounter_id": eid or ro.encounter_id or 0}))
    timed("search_chart", lambda: env.step("search_chart", {"patient_id": pid, "query": "pain"}))
    timed("view_results", lambda: env.step("view_results", {"patient_id": pid, "result_type": "labs"}))
    timed("view_problem_list", lambda: env.step("view_problem_list", {"patient_id": pid}))
    timed("view_medications", lambda: env.step("view_medications", {"patient_id": pid}))
    tool, args = oracle_fn(env)
    timed("submit", lambda: env.step(tool, args))
    return timings


def _http_env(url: str, token: str):
    from eval.env_client import SyntheticHospitalEnv
    return SyntheticHospitalEnv(url, scorer_token=token)


def _http_oracle(env):
    o = env._client.get(f"/env/oracle/{env.episode_id}").json()
    return o["submit_tool"], o["arguments"]


def _local_oracle(env):
    return env.submit_tool, env.oracle()


def _run_http_episode(args):
    url, token, gt_id = args
    env = _http_env(url, token)
    try:
        return play(env, gt_id, _http_oracle)
    finally:
        env._client.close()


_LOCAL = None


def _local():
    global _LOCAL
    if _LOCAL is None:
        from eval.local_env import LocalEnv
        _LOCAL = LocalEnv()
        _LOCAL.warmup()          # per-process caches (scorer context, FTS index): ~30 s, once
    return _LOCAL


def _warm(_):
    _local()
    return True


def _run_local_episode(gt_id: int):
    return play(_local(), gt_id, _local_oracle)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def summarize(all_timings: list[list[tuple[str, float, bool]]], wall: float) -> dict:
    by_call: dict[str, list[float]] = {}
    errors: dict[str, int] = {}
    for ep in all_timings:
        for name, sec, ok in ep:
            by_call.setdefault(name, []).append(sec * 1000)
            if not ok:
                errors[name] = errors.get(name, 0) + 1
    n_steps = sum(len(ep) - 1 for ep in all_timings)   # steps exclude reset
    return {
        "episodes": len(all_timings), "steps": n_steps, "wall_s": round(wall, 3),
        "episodes_per_s": round(len(all_timings) / wall, 2), "steps_per_s": round(n_steps / wall, 1),
        "latency_ms": {k: {"p50": round(_pct(v, 50), 2), "p95": round(_pct(v, 95), 2), "mean": round(statistics.fmean(v), 2), "n": len(v)}
                       for k, v in sorted(by_call.items())},
        "errors": errors,
    }


def format_table(result: dict) -> str:
    lines = [f"{'mode':<8}{'conc':>5}{'episodes':>9}{'wall s':>8}{'ep/s':>7}{'steps/s':>9}   reset p50 | step p50 (detail) | search p50 | submit p50"]
    for run in result["runs"]:
        lat = run["latency_ms"]
        g = lambda k: f"{lat.get(k, {}).get('p50', float('nan')):.1f}"
        lines.append(f"{result['mode']:<8}{run['concurrency']:>5}{run['episodes']:>9}{run['wall_s']:>8.1f}{run['episodes_per_s']:>7.2f}{run['steps_per_s']:>9.1f}   "
                     f"{g('reset'):>9} | {g('view_encounter_detail'):>17} | {g('search_chart'):>10} | {g('submit'):>10}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def run_http(gt_ids: list[int], concurrency: list[int], url: str, token: str) -> dict:
    runs = []
    for c in concurrency:
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            timings = list(ex.map(_run_http_episode, [(url, token, g) for g in gt_ids]))
        runs.append({"concurrency": c, **summarize(timings, time.perf_counter() - t)})
        print(f"[http] concurrency {c}: {runs[-1]['steps_per_s']} steps/s, {runs[-1]['episodes_per_s']} ep/s, errors {runs[-1]['errors']}", flush=True)
    return {"mode": "http", "url": url, "runs": runs}


def run_local(gt_ids: list[int], workers: list[int]) -> dict:
    runs = []
    for w in workers:
        if w == 1:
            _local()                                   # warm before the clock starts, like the HTTP server is
            t = time.perf_counter()
            timings = [_run_local_episode(g) for g in gt_ids]
        else:
            with ProcessPoolExecutor(max_workers=w) as ex:
                list(ex.map(_warm, range(w * 8)))      # every worker warms up before the clock starts
                t = time.perf_counter()
                timings = list(ex.map(_run_local_episode, gt_ids, chunksize=max(1, len(gt_ids) // (w * 4))))
        runs.append({"concurrency": w, **summarize(timings, time.perf_counter() - t)})
        print(f"[local] workers {w}: {runs[-1]['steps_per_s']} steps/s, {runs[-1]['episodes_per_s']} ep/s, errors {runs[-1]['errors']}", flush=True)
    return {"mode": "local", "runs": runs}


def run_components(reps: int, url: str, token: str, gt_id: int) -> dict:
    """Sequential floors of the HTTP path: framework, Redis, one Postgres tool, the scorer."""
    import httpx
    c = httpx.Client(base_url=url, headers={"X-Scorer-Token": token}, timeout=60)
    ep = c.post("/env/reset", json={"gt_id": gt_id, "budget": 500}).json()
    pid, eid = ep["patient_id"], ep["episode_id"]
    o = c.get(f"/env/oracle/{eid}").json()

    def bench(name, fn):
        xs = []
        for _ in range(reps):
            t = time.perf_counter(); fn(); xs.append((time.perf_counter() - t) * 1000)
        return {"p50": round(_pct(xs, 50), 2), "p95": round(_pct(xs, 95), 2), "mean": round(statistics.fmean(xs), 2)}

    out = {
        "GET /health (framework floor)": bench("health", lambda: c.get("/health")),
        "GET /env/state (Redis read + framework)": bench("state", lambda: c.get(f"/env/state/{eid}")),
        "step view_problem_list (Redis + 1 small Postgres query)": bench("pl", lambda: c.post("/env/step", json={"episode_id": eid, "name": "view_problem_list", "arguments": {"patient_id": pid}})),
        "step view_encounter_detail (Redis + Postgres, large payload)": bench("detail", lambda: c.post("/env/step", json={"episode_id": eid, "name": "view_encounter_detail", "arguments": {"encounter_id": ep["encounter_id"] or 0}})),
        "step search_chart (Redis + tsvector query)": bench("search", lambda: c.post("/env/step", json={"episode_id": eid, "name": "search_chart", "arguments": {"patient_id": pid, "query": "pain"}})),
    }
    # reset and submit: fresh episode each rep (reps capped: each is a full score)
    xs_reset, xs_submit = [], []
    for _ in range(min(reps, 40)):
        t = time.perf_counter(); e = c.post("/env/reset", json={"gt_id": gt_id}).json(); xs_reset.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter(); c.post("/env/step", json={"episode_id": e["episode_id"], "name": o["submit_tool"], "arguments": o["arguments"]}); xs_submit.append((time.perf_counter() - t) * 1000)
    out["POST /env/reset (Postgres context + prompt build)"] = {"p50": round(_pct(xs_reset, 50), 2), "p95": round(_pct(xs_reset, 95), 2), "mean": round(statistics.fmean(xs_reset), 2)}
    out["step submit (scorer)"] = {"p50": round(_pct(xs_submit, 50), 2), "p95": round(_pct(xs_submit, 95), 2), "mean": round(statistics.fmean(xs_submit), 2)}
    return {"mode": "components", "gt_id": gt_id, "reps": reps, "latency_ms": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["http", "local", "components"])
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--concurrency", default="1,4,16,64", help="http: thread counts")
    ap.add_argument("--workers", default="1,2,4,8", help="local: process counts")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--split", default="public")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--token", default=DEFAULT_TOKEN)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    gt_ids = sample_gt_ids(a.episodes, a.split, a.seed)
    meta = {"host": platform.platform(), "cpus": os.cpu_count(), "python": platform.python_version(),
            "episodes": a.episodes, "split": a.split, "seed": a.seed, "gt_ids": gt_ids, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if a.mode == "http":
        result = run_http(gt_ids, [int(x) for x in a.concurrency.split(",")], a.url, a.token)
        print(format_table(result))
    elif a.mode == "local":
        result = run_local(gt_ids, [int(x) for x in a.workers.split(",")])
        print(format_table(result))
    else:
        result = run_components(a.reps, a.url, a.token, gt_ids[0])
        for k, v in result["latency_ms"].items():
            print(f"{k:<62} p50 {v['p50']:>8.2f} ms   p95 {v['p95']:>8.2f} ms")
    result["meta"] = meta
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
