"""Protocol runs: a model plays benchmark instances as episodes of the in-process environment (EVAL_PROTOCOL.md).

    python -m eval.protocol_run --model qwen3.5-9b --tasks patient_diagnosis,lab_triage --n 120 [--arm agent|single]
        [--split public] [--seed 0] [--budget 40] [--workers 16] [--max-usd 5] [--min-balance 1] [--out results]
    python -m eval.protocol_run --model qwen3.5-9b --panel --n 120     # every scoring unit in RUN_UNITS
    python -m eval.protocol_run --model qwen3.5-9b --smoke --tasks ...  # 3 instances per unit, prints $/episode

Arms: `agent` = the tool-using loop (the environment's tools + the task's submit tool, budget enforced by the
environment); `single` = one call over the visible chart (the same instances, no tools, assembled without
spending actions) for the tools-vs-no-tools ablation. Instances are a seeded random sample, at most one per
patient; `atypical_diagnosis` is sampled from the parents in the `patient_diagnosis` sample so the
atypical − typical ablation is paired. All units of a run share one worker pool (no per-unit tail), one
read-only release DB and one search index. Failures score 0 and are recorded, never dropped. Costs are the
provider's billed cost when it reports one (OpenRouter), else the list-price estimate; `--max-usd` caps the
whole run and `--min-balance` stops it when the live OpenRouter balance falls below the floor.
Output: results/<model>__<unit>__<arm>__<split>__s<seed>/predictions.jsonl + manifest.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
RUN_UNITS = ("patient_diagnosis", "evidence_retrieval", "context_summarization", "specialty_conditioned",
             "imaging_indication", "differential_diagnosis", "test_selection", "error_detection", "lab_triage",
             "atypical_diagnosis")
"""The units a panel run plays. The specialty task is played as served, the involved/absent mixture;
its two halves are recoverable from the predictions (`involvement`)."""
UNIT_MAX_TURNS = {"error_detection": 16}
"""Per-unit turn caps (Stage B2): Qwen looped 14–42 turns on error_detection (median 5, p90 9), and the loops ended in
the wall-clock deadline; 16 turns bounds them deterministically. Other units: budget + 3."""
SECONDARY_METRICS = ("diagnosis_named", "diagnosis_coded")
"""Recorded per episode beside the reward (RL readiness A2: a gain splits into naming vs coding)."""
EPISODE_DEADLINE_S = int(os.environ.get("SH_EPISODE_DEADLINE_S", "900"))
"""Wall-clock cap per episode: after it the episode is force-submitted with the best answer seen (scored as usual);
every model call is also bounded by the episode's remaining time."""

_local = threading.local()
_DB: D.ReleaseDB | None = None
_DB_LOCK = threading.Lock()


def shared_db() -> D.ReleaseDB:
    global _DB
    with _DB_LOCK:
        if _DB is None:
            _DB = D.ReleaseDB(shared=True)
            LocalEnv(db=_DB).warmup()
    return _DB


def _env() -> LocalEnv:
    if getattr(_local, "env", None) is None:
        _local.env = LocalEnv(db=shared_db())
    return _local.env


# ---------------------------------------------------------------------------
# prices, balance and sampling
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


def openrouter_balance() -> float | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        d = httpx.get("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {key}"}, timeout=20).json()["data"]
        return float(d["total_credits"]) - float(d["total_usage"])
    except Exception:  # noqa: BLE001
        return None


def sample_instances(db: D.ReleaseDB, task: str, split: str, n: int, seed: int) -> list[D.Instance]:
    """Seeded random sample, at most one instance per patient first, then the rest.
    `atypical_diagnosis`: only variants whose parent is in the `patient_diagnosis` sample (paired ablation)."""
    insts = list(db.instances(task, split))
    if task == "atypical_diagnosis":
        parents = {i["gt_id"] for i in sample_instances(db, "patient_diagnosis", split, n, seed)}
        insts = [i for i in insts if i["gt"].get("parent_gt_id") in parents]
    rng = random.Random(f"{seed}:{task}" if task != "atypical_diagnosis" else seed)
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
    from eval.adapters import _CALL_DEADLINE
    _CALL_DEADLINE.t = t0 + EPISODE_DEADLINE_S                     # no single call may outlive the episode deadline
    rec: dict = {"gt_id": inst["gt_id"], "patient_id": inst["patient_id"], "encounter_id": inst.get("encounter_id"),
                 "parent_gt_id": inst["gt"].get("parent_gt_id"), "involvement": inst["gt"].get("involvement"),
                 "reward": 0.0, "metric": None, "steps": 0, "turns": 0, "orders": 0, "order_log": [], "unmatched_orders": 0,
                 "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "error": None, "forced": False,
                 "submitted": False, "submission": None, "latency_ms": 0}
    try:
        ro = env.reset(gt_id=inst["gt_id"], budget=budget)
        system = ro.instructions
        if arm == "single":
            user = (f"{ro.intro}\n\nYou cannot call tools in this setting. The visible chart follows; answer in one message with "
                    f"the JSON the task asks for (or call {ro.submit_tool}).\n\n{env.visible_chart()}")
            tools = [t for t in ro.tools if t["function"]["name"] == ro.submit_tool]
        else:
            user = ro.intro
            tools = ro.tools
        messages: list[dict] = [{"role": "user", "content": user}]
        done = False
        last_parsed = None
        for turn in range(1, max_turns + 1):
            if time.time() - t0 > EPISODE_DEADLINE_S - 5:           # a stalled provider cannot hold a worker forever
                rec["error"] = f"episode deadline {EPISODE_DEADLINE_S}s reached at turn {turn}"
                break
            resp = adapter.call_multi_turn_with_retry(system, messages, tools=tools)
            rec["turns"] = turn
            if resp.cost_usd is not None:
                rec["billed_usd"] = rec.get("billed_usd", 0.0) + resp.cost_usd
            if resp.provider:
                rec.setdefault("providers", {})
                rec["providers"][resp.provider] = rec["providers"].get(resp.provider, 0) + 1
            rec["input_tokens"] += resp.input_tokens
            rec["output_tokens"] += resp.output_tokens
            rec["latency_ms"] += resp.latency_ms
            if arm == "single" and resp.tool_calls and all(c["name"] != ro.submit_tool for c in resp.tool_calls) \
                    and not rec.get("format_retry"):
                # the single arm has no chart tools; a model that still calls one (Qwen ordered tests in 43% of
                # single-arm test_selection episodes) gets one corrective turn instead of a forced 0, and the calls
                # are never executed (no steps, no revealed results)
                rec["format_retry"] = [c["name"] for c in resp.tool_calls]
                messages.append({"role": "assistant", "content": resp.text or "", "tool_calls": resp.raw_tool_calls})
                for raw, call in zip(resp.raw_tool_calls or [{}] * len(resp.tool_calls), resp.tool_calls):
                    messages.append({"role": "tool", "tool_call_id": raw.get("id", f"call_{turn}"), "name": call["name"],
                                     "content": "Not available in this setting: no tools can be called."})
                messages.append({"role": "user", "content": f"Tools are not available here. Answer now from the chart above: "
                                                            f"call {ro.submit_tool} or reply with the JSON answer."})
                continue
            if resp.tool_calls:
                messages.append({"role": "assistant", "content": resp.text or "", "tool_calls": resp.raw_tool_calls})
                for raw, call in zip(resp.raw_tool_calls or [{}] * len(resp.tool_calls), resp.tool_calls):
                    name, args = call["name"], call.get("arguments") or {}
                    obs, reward, done, info = env.step(name, args)
                    messages.append({"role": "tool", "tool_call_id": raw.get("id", f"call_{turn}"), "name": name,
                                     "content": info.get("observation_text") or json.dumps(obs, default=str)[:8000]})
                    if done:
                        rec.update(reward=float(reward), metric=info.get("reward_metric"), forced=bool(info.get("forced")),
                                   submitted=True, submission=args)
                        rec["metrics"] = {k: v for k, v in (info.get("metrics") or {}).items() if k in SECONDARY_METRICS}
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
                    rec["metrics"] = {k: v for k, v in (info.get("metrics") or {}).items() if k in SECONDARY_METRICS}
                    break
                messages.append({"role": "user", "content": f"Take an action: call a tool, or call {ro.submit_tool} with your answer. "
                                                            f"{max(0, budget - env.ep['steps'])} actions remain."})
            if arm == "single":
                break                                               # one answer (after at most one format retry)
        if not done:
            # forced final submission: the best answer seen, else empty (scored 0), exactly like the server
            obs, reward, _, info = env.step(ro.submit_tool, last_parsed or {})
            rec.update(reward=float(reward), metric=info.get("reward_metric"), forced=True, submitted=last_parsed is not None,
                       submission=last_parsed)
            rec["metrics"] = {k: v for k, v in (info.get("metrics") or {}).items() if k in SECONDARY_METRICS}
        rec["steps"] = env.ep["steps"]
        rec["order_log"] = list(env.ep.get("orders") or [])
        rec["orders"] = len(rec["order_log"])
        rec["unmatched_orders"] = sum(1 for o in rec["order_log"] if not o.get("matched"))
    except (EpisodeError, Exception) as exc:  # noqa: BLE001 — an API failure is a scored 0, recorded
        detail = ""
        cause = exc
        while cause is not None and not detail:                     # the provider's error body, when there is one
            try:
                r = getattr(cause, "response", None)                # httpx raises when it is an unread stream
            except Exception:  # noqa: BLE001
                r = None
            if r is not None:
                try:
                    detail = f" | {r.text[:300]}"
                except Exception:  # noqa: BLE001 — never let error reporting kill the run
                    detail = " | (streamed response)"
            cause = cause.__cause__
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}{detail}"
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
    rec["list_cost_usd"] = (rec["input_tokens"] * prices[0] + rec["output_tokens"] * prices[1]) / 1e6
    rec["cost_usd"] = rec["billed_usd"] if rec.get("billed_usd") is not None else rec["list_cost_usd"]
    rec["wall_s"] = round(time.time() - t0, 2)
    return rec


# ---------------------------------------------------------------------------
# a run: many units, one pool
# ---------------------------------------------------------------------------

def _git() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def run_units(model: str, tasks: list[str], n: int, arm: str = "agent", split: str = "public", seed: int = 0,
              budget: int = 40, workers: int = 16, max_usd: float | None = None, out_root: Path = RESULTS,
              max_turns: int | None = None, adapter=None, prices: tuple[float, float] | None = None, quiet: bool = False,
              max_output_tokens: int | None = None, min_balance: float | None = None,
              allow_reward_drift: bool = False) -> dict[str, Path]:
    from eval import reward_version as RV
    reward_info = RV.current()
    if reward_info["reward_drift"] and not allow_reward_drift:
        RV.require_frozen()                                       # raises: the reward is not the frozen one
    cfg = MODEL_REGISTRY[model]
    if max_output_tokens:
        cfg = dataclasses.replace(cfg, max_tokens=max_output_tokens)      # per-turn cap, like an RL rollout limit
    adapter = adapter or create_adapter(cfg)
    prices = prices or live_prices(model)
    db = shared_db()
    turn_cap = {t: max_turns or UNIT_MAX_TURNS.get(t, budget + 3) for t in tasks}
    units: dict[str, dict] = {}
    todo: list[tuple[str, D.Instance]] = []
    for task in tasks:
        insts = sample_instances(db, task, split, n, seed)
        out = out_root / f"{model}__{task}__{arm}__{split}__s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        pred_path = out / "predictions.jsonl"
        prior = [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()] if pred_path.exists() else []
        done_ids = {p["gt_id"] for p in prior}
        units[task] = {"out": out, "insts": insts, "fh": pred_path.open("a"), "spent": sum(float(p.get("cost_usd") or 0) for p in prior),
                       "n_done": len(prior), "t0": time.time()}
        todo += [(task, i) for i in insts if i["gt_id"] not in done_ids]
    # interleave units so every unit progresses together and no unit's stragglers idle the pool
    random.Random(seed).shuffle(todo)
    total, n_done, spent = len(todo), 0, sum(u["spent"] for u in units.values())
    t0 = time.time()
    lock = threading.Lock()
    stop_reason = None
    last_balance_check = 0.0
    it = iter(todo)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for _ in range(workers):
            nxt = next(it, None)
            if nxt is not None:
                futures[ex.submit(run_episode, adapter, nxt[1], arm, budget, prices, turn_cap[nxt[0]])] = nxt[0]
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in finished:
                task = futures.pop(fut)
                rec = fut.result()
                u = units[task]
                with lock:
                    u["fh"].write(json.dumps(rec, default=str) + "\n"); u["fh"].flush()
                    u["spent"] += rec["cost_usd"]; u["n_done"] += 1
                    spent += rec["cost_usd"]; n_done += 1
                    if not quiet and (n_done % 20 == 0 or n_done == total):
                        el = time.time() - t0
                        print(f"[{model} {arm}] {n_done}/{total} episodes ({100 * n_done / max(total, 1):.0f}%), ${spent:.2f} spent, "
                              f"{el / 60:.1f} min elapsed, ETA {(total - n_done) * el / n_done / 60:.1f} min", flush=True)
                if max_usd is not None and spent >= max_usd and stop_reason is None:
                    stop_reason = f"cost cap ${max_usd} reached at ${spent:.2f}"
                if min_balance is not None and "openrouter" in cfg.base_url and time.time() - last_balance_check > 60:
                    last_balance_check = time.time()
                    bal = openrouter_balance()
                    if bal is not None and bal < min_balance and stop_reason is None:
                        stop_reason = f"OpenRouter balance ${bal:.2f} below ${min_balance}"
                if stop_reason is not None:
                    continue
                nxt = next(it, None)
                if nxt is not None:
                    futures[ex.submit(run_episode, adapter, nxt[1], arm, budget, prices, turn_cap[nxt[0]])] = nxt[0]
    if stop_reason and not quiet:
        print(f"[{model} {arm}] STOPPED: {stop_reason}; completed episodes are kept and a rerun resumes", flush=True)
    prompt_hash = None
    for task, u in units.items():
        u["fh"].close()
        if prompt_hash is None and u["insts"]:
            prompt_hash = hashlib.sha256(_env().reset(gt_id=u["insts"][0]["gt_id"]).instructions.encode()).hexdigest()[:16]
        manifest = {
            "run_id": u["out"].name, "model": model, "provider_base_url": cfg.base_url, "model_id": cfg.model_id, "task": task,
            "arm": arm, "split": split, "seed": seed, "n_requested": n, "n_sampled": len(u["insts"]), "n_recorded": u["n_done"],
            "budget": budget, "max_turns": turn_cap[task], "prices_per_million": {"input": prices[0], "output": prices[1]},
            "temperature": None if "api.openai.com" in cfg.base_url and cfg.extra.get("no_temperature", True) else cfg.temperature,
            "extra": dict(cfg.extra), "max_output_tokens_per_turn": cfg.max_tokens, "episode_deadline_s": EPISODE_DEADLINE_S,
            "prompt_hash": prompt_hash, "git_commit": _git(), "floors_file": "eval/floors.json",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(u["t0"])), "duration_s": round(time.time() - u["t0"], 1),
            "cost_usd": round(u["spent"], 4), "max_usd": max_usd, "stopped": stop_reason, "protocol": "EVAL_PROTOCOL.md",
            **reward_info, "reward_drift_allowed": bool(allow_reward_drift and reward_info["reward_drift"]),
        }
        (u["out"] / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {task: u["out"] for task, u in units.items()}


def run(model: str, task: str, n: int, arm: str = "agent", split: str = "public", seed: int = 0, budget: int = 40,
        workers: int = 4, max_usd: float | None = None, out_root: Path = RESULTS, max_turns: int | None = None,
        adapter=None, prices: tuple[float, float] | None = None, quiet: bool = False,
        max_output_tokens: int | None = None) -> Path:
    """One unit (kept for callers and tests)."""
    return run_units(model, [task], n, arm, split, seed, budget, workers, max_usd, out_root, max_turns, adapter, prices,
                     quiet, max_output_tokens)[task]


def summarize(out: Path) -> dict:
    preds = [json.loads(l) for l in (out / "predictions.jsonl").read_text().splitlines() if l.strip()]
    n = len(preds)
    return {"n": n, "mean_reward": sum(p["reward"] for p in preds) / n if n else None,
            "errors": sum(1 for p in preds if p.get("error")), "forced": sum(1 for p in preds if p.get("forced")),
            "cost_usd": round(sum(p["cost_usd"] for p in preds), 4),
            "tokens_per_episode": round(sum(p["input_tokens"] + p["output_tokens"] for p in preds) / n) if n else None,
            "steps_per_episode": round(sum(p["steps"] for p in preds) / n, 1) if n else None,
            "unmatched_orders": sum(p.get("unmatched_orders", 0) for p in preds), "orders": sum(p.get("orders", 0) for p in preds)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    ap.add_argument("--task", default="patient_diagnosis", choices=list(D.TASKS))
    ap.add_argument("--tasks", default=None, help="comma-separated scoring units (overrides --task / --panel)")
    ap.add_argument("--panel", action="store_true", help=f"run {','.join(RUN_UNITS)}")
    ap.add_argument("--smoke", action="store_true", help="3 instances per unit; prints measured $/episode")
    ap.add_argument("--arm", default="agent", choices=["agent", "single"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--split", default="public")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-usd", type=float, default=None, help="hard cap for the whole run")
    ap.add_argument("--min-balance", type=float, default=None, help="stop when the OpenRouter balance falls below this")
    ap.add_argument("--max-output-tokens", type=int, default=None, help="per-turn output cap (default: the registry's 16384)")
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument("--allow-reward-drift", action="store_true", help="run although the scorer differs from eval/reward_lock.json (recorded)")
    a = ap.parse_args(argv)
    tasks = a.tasks.split(",") if a.tasks else (list(RUN_UNITS) if a.panel else [a.task])
    n = 3 if a.smoke else a.n
    out_root = Path(a.out) / ("smoke" if a.smoke else "")
    outs = run_units(a.model, tasks, n, a.arm, a.split, a.seed, a.budget, a.workers, a.max_usd, out_root,
                     max_output_tokens=a.max_output_tokens, min_balance=a.min_balance, allow_reward_drift=a.allow_reward_drift)
    for task, out in outs.items():
        s = summarize(out)
        print(json.dumps({"run": out.name, **s}))
        if a.smoke and s["n"]:
            print(f"  -> ${s['cost_usd'] / s['n']:.4f}/episode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
