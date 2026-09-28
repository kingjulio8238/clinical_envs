"""Structured, human-readable logs for RL training and GPU evaluation runs (inspectable at any time during a run).

Every run writes two streams from the same events:
- **stdout**: one formatted line per event (`modal app logs <app>` streams it live; the job also tees it to a log file
  on the results volume) — episodes in batches, every training step with its losses, every dev evaluation, alerts,
  errors (immediately, with the instance id), stops;
- **events.jsonl** next to the run's other outputs: the same events as JSON, one per line, read by
  `scripts/rl_watch.py` (the terminal dashboard) and by any later analysis.

The pattern is PufferLib's (`pufferl.py`: a Rich dashboard of summary / performance breakdown / losses / user stats /
utilization, plus W&B or Neptune logging of the same dict); here the job runs detached on Modal, so the dashboard runs
on the operator's machine and reads the events from the results volume while the job is live.
"""
from __future__ import annotations

import json
import math
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

UNIT_ABBR = {"patient_diagnosis": "pd", "differential_diagnosis": "dd", "evidence_retrieval": "er", "test_selection": "ts",
             "atypical_diagnosis": "aty", "context_summarization": "sum", "specialty_conditioned": "spc",
             "imaging_indication": "img", "lab_triage": "lab", "error_detection": "err"}

LOSS_KEYS = [("loss/train", "loss"), ("loss/entropy", "ent"), ("loss/kl_div", "kl"), ("loss/kl_policy_ref", "kl_ref"),
             ("loss/grad_norm", "gnorm"), ("loss/importance_ratio_mean", "ratio"), ("loss/importance_ratio_p95", "ratio_p95"),
             ("loss/clipped_token_fraction", "clip"), ("loss/learning_rate", "lr"),
             ("data/step_trainable_assistant_tokens", "train_tok")]
"""ART LocalBackend.train metrics shown in the step line and the dashboard (names from ART 0.5.20's trainer)."""


def _f(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, float):
        if x != 0 and (abs(x) < 10 ** -nd or abs(x) >= 1e5):
            return f"{x:.2e}"
        return f"{x:.{nd}f}"
    return str(x)


def _k(n: float | None) -> str:
    if n is None:
        return "—"
    return f"{n / 1e6:.1f}M" if n >= 1e6 else (f"{n / 1e3:.0f}k" if n >= 1e3 else f"{n:.0f}")


def _dur(s: float | None) -> str:
    if s is None:
        return "—"
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else (f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s")


def by_unit(values: dict[str, float]) -> str:
    return " ".join(f"{UNIT_ABBR.get(u, u)} {_f(v, 2)}" for u, v in values.items())


def fmt_step(e: dict, max_steps: int | None = None, eta_s: float | None = None) -> str:
    """One training step: reward (mean ± sd, per unit), signal, losses, tokens, timing."""
    m = e.get("train_metrics") or {}
    losses = " ".join(f"{short} {_f(m.get(key), 4)}" for key, short in LOSS_KEYS[:8] if m.get(key) is not None)
    tr = e.get("train") or {}
    tok_s = (e.get("output_tokens") or 0) / e["rollout_s"] if e.get("rollout_s") else None
    parts = [f"step {e['step']}" + (f"/{max_steps}" if max_steps else ""),
             f"reward {_f(e.get('train_reward'))} ±{_f(tr.get('reward_sd'), 2)}" + (f" ({by_unit(e['train_by_unit'])})" if e.get("train_by_unit") else ""),
             f"no-signal groups {e.get('flat_groups', '—')}/{e.get('groups', '—')}",
             losses or "losses —",
             f"gen {_k(e.get('output_tokens'))} tok ({_k(tok_s)}/s)",
             f"rollout {_dur(e.get('rollout_s'))} train {_dur(e.get('train_s'))}",
             f"exc {e.get('exceptions', 0)}"]
    if eta_s is not None:
        parts.append(f"ETA {_dur(eta_s)}")
    return " | ".join(parts)


def fmt_dev(e: dict, best: dict | None, base_score: float | None) -> str:
    d = e.get("dev") or {}
    star = " ★ new best" if best and best.get("step") == e["step"] else ""
    probes = max((d.get(k) or 0) for k in ("probe_many_entries", "probe_long_name", "probe_duplicates"))
    return " | ".join([
        f"DEV step {e['step']}", f"score {_f(e.get('dev_score'))} (base {_f(base_score)}, best {_f((best or {}).get('dev_score'))}"
        f" @{(best or {}).get('step', '—')}){star}", by_unit(e.get("dev_by_unit") or {}),
        f"named {_f(d.get('diagnosis_named'), 2)} coded {_f(d.get('diagnosis_coded'), 2)}", f"probes {probes:.1%}",
        f"entries {_f(d.get('entries'), 1)} len {_f(d.get('answer_chars'), 0)}",
        "alerts: " + ("; ".join(e.get("alerts") or []) or "none")])


class RunLog:
    """Emit events to stdout (formatted) and to `<out>/events.jsonl` (JSON). Thread-safe."""

    def __init__(self, out: Path, name: str, stream=None, progress_every: int = 8, progress_s: float = 60.0, clock=None):
        out.mkdir(parents=True, exist_ok=True)
        self.name, self.stream = name, stream or sys.stdout
        self.fh = (out / "events.jsonl").open("a")
        self.lock = threading.Lock()
        self.clock = clock or time.time          # a simulated clock only in scripts/rl_watch.py --demo
        self.t0 = self.clock()
        self.progress_every, self.progress_s = progress_every, progress_s
        self._batch: dict[str, Any] = {}

    def emit(self, kind: str, text: str | None = None, **data) -> None:
        ev = {"ts": round(self.clock(), 2), "kind": kind, **data}
        with self.lock:
            self.fh.write(json.dumps(ev, default=str) + "\n")
            self.fh.flush()
            if text:
                stamp = time.strftime("%H:%M:%S", time.localtime(self.clock()))
                print(f"{stamp} [{self.name}] {text}", file=self.stream, flush=True)

    def close(self) -> None:
        self.fh.close()

    # -- episodes, batched into progress lines ---------------------------------------------------------------------
    def begin_batch(self, label: str, total: int) -> None:
        with self.lock:
            self._batch = {"label": label, "total": total, "done": 0, "rewards": [], "errors": 0, "t0": self.clock(),
                           "last_print": self.clock(), "by_unit": defaultdict(list)}

    def episode(self, task: str, gt_id: Any, rec: dict | None, error: str | None = None, **extra) -> None:
        rec = rec or {}
        err = error or rec.get("error")
        reward = float(rec.get("reward") or 0.0)
        data = {"task": task, "gt_id": gt_id, "reward": reward, "error": err, "turns": rec.get("turns"),
                "output_tokens": rec.get("output_tokens"), "limit": rec.get("limit_reason"),
                "truncated_turns": rec.get("truncated_turns"), "forced": bool(rec.get("forced")), **extra}
        text = f"ERROR episode gt={gt_id} ({task}): {str(err)[:300]}" if err else None
        self.emit("episode", text, **data)
        b = self._batch
        if not b:
            return
        with self.lock:
            b["done"] += 1
            b["errors"] += bool(err)
            b["rewards"].append(reward)
            b["by_unit"][task].append(reward)
            due = b["done"] % self.progress_every == 0 or b["done"] == b["total"] or self.clock() - b["last_print"] > self.progress_s
            if due:
                b["last_print"] = self.clock()
                el = self.clock() - b["t0"]
                mean = sum(b["rewards"]) / len(b["rewards"])
                eta = el / b["done"] * (b["total"] - b["done"]) if b["total"] else None
                line = (f"{b['label']}: {b['done']}/{b['total']} episodes ({100 * b['done'] / max(b['total'], 1):.0f}%), "
                        f"mean reward {mean:.3f} ({by_unit({u: sum(v) / len(v) for u, v in b['by_unit'].items()})}), "
                        f"{b['errors']} errors, {_dur(el)} elapsed, ETA {_dur(eta)}")
        if due:
            self.emit("progress", line, label=b["label"], done=b["done"], total=b["total"], mean_reward=mean,
                      errors=b["errors"])


def flat_groups(rewards_by_group: list[list[float]], spread: float = 0.01) -> int:
    """Groups whose rewards are all equal: no GRPO advantage, no learning signal from them this step."""
    n = 0
    for rs in rewards_by_group:
        if len(rs) >= 2:
            mu = sum(rs) / len(rs)
            n += math.sqrt(sum((r - mu) ** 2 for r in rs) / len(rs)) <= spread
    return n


def unit_means(pairs: list[tuple[str, float]]) -> dict[str, float]:
    acc = defaultdict(list)
    for u, r in pairs:
        acc[u].append(r)
    return {u: sum(v) / len(v) for u, v in acc.items()}


def summarize_events(events: list[dict]) -> dict:
    """State of a run from its events (the dashboard's model; also usable in a notebook)."""
    st: dict[str, Any] = {"steps": [], "devs": [], "alerts": [], "errors": [], "stop": None, "start": None, "end": None,
                          "progress": None, "episodes": 0, "units": defaultdict(lambda: Counter()),
                          "unit_reward": defaultdict(float), "last_ts": None}
    for e in events:
        st["last_ts"] = e.get("ts", st["last_ts"])
        k = e.get("kind")
        if k == "start":
            st["start"] = e
        elif k == "step":
            st["steps"].append(e)
        elif k == "dev":
            st["devs"].append(e)
        elif k == "alert":
            st["alerts"].append(e)
        elif k == "stop":
            st["stop"] = e
        elif k == "end":
            st["end"] = e
        elif k == "progress":
            st["progress"] = e
        elif k == "episode":
            st["episodes"] += 1
            u = e.get("task")
            c = st["units"][u]
            c["done"] += 1
            c["errors"] += bool(e.get("error"))
            c["forced"] += bool(e.get("forced"))
            c["limited"] += bool(e.get("limit"))
            c["truncated"] += bool(e.get("truncated_turns"))
            c["turns"] += int(e.get("turns") or 0)
            c["tokens"] += int(e.get("output_tokens") or 0)
            st["unit_reward"][u] += float(e.get("reward") or 0)
            if e.get("error"):
                st["errors"].append(e)
    return st
