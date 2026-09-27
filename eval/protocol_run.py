"""Protocol runs: a model plays benchmark instances as episodes of the in-process environment (EVAL_PROTOCOL.md).

    python -m eval.protocol_run --model gpt-6-luna --task patient_diagnosis --n 120 [--arm agent|single]
        [--split public] [--seed 0] [--budget 40] [--workers 4] [--max-usd 5] [--out results]
    python -m eval.protocol_run --model gpt-6-luna --panel --n 120     # every scoring unit in eval.degenerate.TASKS
    python -m eval.protocol_run --model gpt-6-luna --smoke              # 3 instances of one task, prints measured tokens and $/episode

Arms: `agent` = the tool-using loop (the environment's tools + the task's submit tool, budget enforced by the
environment); `single` = one call over the full visible chart (the same instances, no tools) for the
tools-vs-no-tools ablation. Instances are a seeded random sample with at most one per patient. Failures
score 0 and are recorded, never dropped. Output: results/<run>/predictions.jsonl + manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from eval import degenerate as D
from eval.adapters import create_adapter
from eval.config import COST_RATES, MODEL_REGISTRY
from eval.local_env import EpisodeError, LocalEnv

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SUBMIT_KEYS = {"active_diagnoses", "chronic_conditions", "summary", "rankings", "clinical_question", "differential",
               "icd10", "section_type", "relevant"}
UNIT_TASK = {"specialty_involved": "context_summarization", "specialty_absent": "context_summarization"}

_local = threading.local()


def _env() -> LocalEnv:
    if getattr(_local, "env", None) is None:
        _local.env = LocalEnv()
        _local.env.warmup()
    return _local.env


# ---------------------------------------------------------------------------
# prices and sampling
# ---------------------------------------------------------------------------

def live_prices(model_name: str) -> tuple[float, float]:
    """$ per 1M (input, output): OpenRouter's live list for OpenRouter models, else the registry table."""
    cfg = MODEL_REGISTRY[model_name]
    if "openrouter" in cfg.base_url:
        try:
            data = httpx.get("https://openrouter.ai/api/v1/models", timeout=20).json()["data"]
            m = next((x for x in data if x["id"] == cfg.model_id), None)
            if m:
                return float(m["pricing"]["prompt"]) * 1e6, float(m["pricing"]["completion"]) * 1e6
        except Exception:  # noqa: BLE001 — fall back to the table
            pass
    return COST_RATES.get(model_name, (0.0, 0.0))


def sample_instances(db: D.ReleaseDB, task: str, split: str, n: int, seed: int) -> list[D.Instance]:
    """Seeded random sample, at most one instance per patient first, then the rest."""
    insts = list(db.instances(task, split))
    rng = random.Random(seed)
    rng.shuffle(insts)
    by_patient: dict[int, D.Instance] = {}
    rest = []
    for i in insts:
        if i["patient_id"] not in by_patient:
            by_patient[i["patient_id"]] = i
        else:
            rest.append(i)
    chosen = list(by_patient.values())
    rng.shuffle(chosen)
    out = chosen[:n]
    if len(out) < n:
        out += rest[: n - len(out)]
    return sorted(out, key=lambda i: i["gt_id"])


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------

def _visible_chart(env: LocalEnv, ro) -> str:
    """The chart as the environment serves it (cutoff, hidden sections, overrides), for the single arm."""
    encs = env.step("view_encounters", {"patient_id": ro.patient_id})[0]
    parts = []
    for e in encs if isinstance(encs, list) else []:
        d = env.step("view_encounter_detail", {"encounter_id": e["encounter_id"]})[0]
        if not isinstance(d, dict) or "sections" not in d:
            continue
        parts.append(f"\n{'=' * 60}\nENCOUNTER {d['encounter_id']}: {d.get('date')} ({d.get('type')}) — {d.get('chief_complaint') or ''}\n{'=' * 60}")
        for s in d["sections"]:
            parts.append(f"[{str(s['section_type']).upper().replace('_', ' ')}]\n{s['section_text']}")
    problems = env.step("view_problem_list", {"patient_id": ro.patient_id})[0]
    if isinstance(problems, list) and problems:
        parts.append("[DOCUMENTED PROBLEM LIST]\n" + "\n".join(f"- {p.get('display_name')}" for p in problems))
    return "\n\n".join(parts)


def _submission_from_text(text: str) -> dict | None:
    t = (text or "").strip()
    if t.startswith("```"):
        t = "\n".join(l for l in t.split("\n") if not l.strip().startswith("```"))
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        if not cand:
            continue
        try:
            obj = json.loads(cand, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (set(obj) & SUBMIT_KEYS):
            return obj
        if isinstance(obj, list):
            return {"differential": obj}
    return None


def run_episode(adapter, inst: D.Instance, arm: str, budget: int, prices: tuple[float, float], max_turns: int) -> dict:
    env = _env()
    t0 = time.time()
    rec: dict = {"gt_id": inst["gt_id"], "patient_id": inst["patient_id"], "encounter_id": inst.get("encounter_id"),
                 "parent_gt_id": inst["gt"].get("parent_gt_id"), "reward": 0.0, "metric": None, "steps": 0, "turns": 0,
                 "orders": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "error": None, "forced": False,
                 "submitted": False, "submission": None, "latency_ms": 0}
    try:
        ro = env.reset(gt_id=inst["gt_id"], budget=budget if arm == "agent" else 500)
        system = ro.instructions
        if arm == "single":
            chart = _visible_chart(env, ro)
            user = (f"{ro.intro}\n\nYou cannot call tools in this setting. The visible chart follows; answer in one message with "
                    f"the JSON the task asks for (or call {ro.submit_tool}).\n\n{chart}")
            tools = [t for t in ro.tools if t["function"]["name"] == ro.submit_tool]
        else:
            user = ro.intro
            tools = ro.tools
        messages: list[dict] = [{"role": "user", "content": user}]
        done = False
        last_parsed = None
        for turn in range(1, max_turns + 1):
            resp = adapter.call_multi_turn_with_retry(system, messages, tools=tools)
            rec["turns"] = turn
            rec["input_tokens"] += resp.input_tokens
            rec["output_tokens"] += resp.output_tokens
            rec["latency_ms"] += resp.latency_ms
            if resp.tool_calls:
                messages.append({"role": "assistant", "content": resp.text or "", "tool_calls": resp.raw_tool_calls})
                for raw, call in zip(resp.raw_tool_calls or [{}] * len(resp.tool_calls), resp.tool_calls):
                    name, args = call["name"], call.get("arguments") or {}
                    obs, reward, done, info = env.step(name, args)
                    if name == "order_test":
                        rec["orders"] += 1
                    messages.append({"role": "tool", "tool_call_id": raw.get("id", f"call_{turn}"), "name": name,
                                     "content": info.get("observation_text") or json.dumps(obs, default=str)[:8000]})
                    if done:
                        rec.update(reward=float(reward), metric=info.get("reward_metric"), forced=bool(info.get("forced")),
                                   submitted=True, submission=args)
                        break
                if done:
                    break
            else:
                parsed = _submission_from_text(resp.text)
                messages.append({"role": "assistant", "content": resp.text or ""})
                if parsed is not None:
                    last_parsed = parsed
                    obs, reward, done, info = env.step(ro.submit_tool, parsed)
                    rec.update(reward=float(reward), metric=info.get("reward_metric"), forced=bool(info.get("forced")),
                               submitted=True, submission=parsed)
                    break
                messages.append({"role": "user", "content": f"Take an action: call a tool, or call {ro.submit_tool} with your answer. "
                                                            f"{max(0, budget - env.ep['steps'])} actions remain."})
            if arm == "single":
                break
        if not done:
            # forced final submission: the best answer seen, else empty (scored 0), exactly like the server
            obs, reward, _, info = env.step(ro.submit_tool, last_parsed or {})
            rec.update(reward=float(reward), metric=info.get("reward_metric"), forced=True, submitted=last_parsed is not None,
                       submission=last_parsed)
        rec["steps"] = env.ep["steps"]
    except (EpisodeError, Exception) as exc:  # noqa: BLE001 — an API failure is a scored 0, recorded
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
    rec["cost_usd"] = (rec["input_tokens"] * prices[0] + rec["output_tokens"] * prices[1]) / 1e6
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


# ---------------------------------------------------------------------------
# a run
# ---------------------------------------------------------------------------

def run(model: str, task: str, n: int, arm: str = "agent", split: str = "public", seed: int = 0, budget: int = 40,
        workers: int = 4, max_usd: float | None = None, out_root: Path = RESULTS, max_turns: int | None = None,
        adapter=None, prices: tuple[float, float] | None = None, quiet: bool = False) -> Path:
    cfg = MODEL_REGISTRY[model]
    adapter = adapter or create_adapter(cfg)
    prices = prices or live_prices(model)
    db = D.ReleaseDB()
    insts = sample_instances(db, task, split, n, seed)
    run_id = f"{model}__{task}__{arm}__{split}__s{seed}"
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "predictions.jsonl"
    done_ids = set()
    if pred_path.exists():
        done_ids = {json.loads(l)["gt_id"] for l in pred_path.read_text().splitlines() if l.strip()}
    todo = [i for i in insts if i["gt_id"] not in done_ids]
    max_turns = max_turns or (budget + 3)
    spent = sum(float(json.loads(l).get("cost_usd") or 0) for l in pred_path.read_text().splitlines() if l.strip()) if pred_path.exists() else 0.0
    lock = threading.Lock()
    t0 = time.time()
    stop = False
    with pred_path.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        it = iter(todo)
        # bounded submission so a cost cap can stop new episodes
        for _ in range(workers * 2):
            i = next(it, None)
            if i is not None:
                futures[ex.submit(run_episode, adapter, i, arm, budget, prices, max_turns)] = i
        n_done = 0
        while futures:
            fut = next(as_completed(futures))
            futures.pop(fut)
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
                spent += rec["cost_usd"]; n_done += 1
                if not quiet and (n_done % 10 == 0 or n_done == len(todo)):
                    el = time.time() - t0
                    print(f"[{model} {task} {arm}] {n_done}/{len(todo)} done, ${spent:.2f} spent, "
                          f"{el / n_done:.1f}s/episode, ETA {(len(todo) - n_done) * el / n_done / 60:.1f} min", flush=True)
            if max_usd is not None and spent >= max_usd and not stop:
                stop = True
                print(f"[{model} {task} {arm}] cost cap ${max_usd} reached at ${spent:.2f}: no new episodes", flush=True)
            if not stop:
                i = next(it, None)
                if i is not None:
                    futures[ex.submit(run_episode, adapter, i, arm, budget, prices, max_turns)] = i
    manifest = {
        "run_id": run_id, "model": model, "provider_base_url": cfg.base_url, "model_id": cfg.model_id, "task": task,
        "arm": arm, "split": split, "seed": seed, "n_requested": n, "n_sampled": len(insts), "budget": budget,
        "prices_per_million": {"input": prices[0], "output": prices[1]}, "temperature": cfg.temperature,
        "prompt_hash": hashlib.sha256((_env().reset(gt_id=insts[0]["gt_id"]).instructions if insts else "").encode()).hexdigest()[:16],
        "git_commit": _git(), "floors_file": "eval/floors.json", "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
        "duration_s": round(time.time() - t0, 1), "cost_usd": round(spent, 4), "max_usd": max_usd,
        "protocol": "EVAL_PROTOCOL.md",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def _git() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def summarize(out: Path) -> dict:
    preds = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines() if l.strip()]
    n = len(preds)
    return {"n": n, "mean_reward": sum(p["reward"] for p in preds) / n if n else None,
            "errors": sum(1 for p in preds if p.get("error")), "forced": sum(1 for p in preds if p.get("forced")),
            "cost_usd": round(sum(p["cost_usd"] for p in preds), 4),
            "tokens_per_episode": round(sum(p["input_tokens"] + p["output_tokens"] for p in preds) / n) if n else None,
            "steps_per_episode": round(sum(p["steps"] for p in preds) / n, 1) if n else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--task", default="patient_diagnosis", choices=list(D.TASKS))
    ap.add_argument("--panel", action="store_true", help="run every scoring unit")
    ap.add_argument("--smoke", action="store_true", help="3 instances of --task; print measured tokens and $/episode")
    ap.add_argument("--arm", default="agent", choices=["agent", "single"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--split", default="public")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-usd", type=float, default=None, help="hard cap per (model, task) run")
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args(argv)
    tasks = list(D.TASKS) if a.panel else [a.task]
    n = 3 if a.smoke else a.n
    out_root = Path(a.out) / ("smoke" if a.smoke else "")
    for task in tasks:
        out = run(a.model, task, n, a.arm, a.split, a.seed, a.budget, a.workers, a.max_usd, out_root)
        s = summarize(out)
        print(json.dumps({"run": out.name, **s}, indent=None))
        if a.smoke and s["n"]:
            per = s["cost_usd"] / s["n"]
            print(f"  -> ${per:.4f}/episode; 120 episodes ≈ ${per * 120:.2f}; 9 units x 120 ≈ ${per * 1080:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
