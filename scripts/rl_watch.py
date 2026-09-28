"""Terminal dashboard for RL training and GPU evaluation jobs, live while they run on Modal (or on local files).

    # a training run on a workspace's results volume, refreshed every 20 s, with the live Modal log stream in a panel
    python scripts/rl_watch.py --profile newacc --path rl/c5-main --logs clinical-envs-train
    # an evaluation job (all its parts)
    python scripts/rl_watch.py --profile sales-32662 --path c2 --logs clinical-envs-vllm-eval
    # what it looks like: a synthetic run (plausible numbers, not a measurement), a new step every 2 s
    python scripts/rl_watch.py --demo
    # local files (a pulled run, a dry run); --once prints one snapshot and exits (for scripts and polling)
    python scripts/rl_watch.py --local results/modal/newacc/rl/c5-main --once

Needs `rich` and, for a volume, the `modal` package (pinned to --profile). Reads the job's `events.jsonl` (written by
eval/run_log.py: every episode, training step with ART's losses, dev evaluation, alert, error, stop) and
`telemetry.jsonl` (gpu/telemetry.py: GPU, vLLM, host). Layout after PufferLib's dashboard (`pufferl.py
print_dashboard`): a summary header, the losses / training table, the environment's own stats (per-unit rewards, dev
selection, monitor signals), performance breakdown and utilization, and the tail of the log. Volume data lags by the
job's commit interval (30 s); `--logs` streams `modal app logs` with no lag. The header turns red when no event has
arrived for `--stale` seconds — a hung job looks exactly like a slow one otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.run_log import LOSS_KEYS, UNIT_ABBR, _dur, _f, _k, summarize_events  # noqa: E402

VOLUME = "clinical-envs-results"
RATE = 4.10
SPARK = "▁▂▃▄▅▆▇█"
ERR_RE = re.compile(r"Traceback|\bERROR\b|FATAL|\bOOM\b|CUDA out of memory|\bKilled\b|\bSTOP\b|ALERT|[A-Za-z]+Error:")


def spark(vals: list[float | None]) -> str:
    v = [x for x in vals if x is not None]
    if not v:
        return ""
    lo, hi = min(v), max(v)
    return "".join(SPARK[int((x - lo) / (hi - lo) * 7) if hi > lo else 3] if x is not None else " " for x in vals)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

class LocalSource:
    def __init__(self, root: Path):
        self.root = root

    def files(self, name: str) -> list[tuple[str, str]]:
        return [(str(p.relative_to(self.root)), p.read_text(errors="replace")) for p in sorted(self.root.rglob(name))]


class VolumeSource:
    def __init__(self, profile: str, path: str):
        os.environ["MODAL_PROFILE"] = profile            # pinned before the SDK reads its config
        import modal
        self.vol = modal.Volume.from_name(VOLUME)
        self.path = path.strip("/")

    def files(self, name: str) -> list[tuple[str, str]]:
        out = []
        for e in self.vol.listdir(self.path, recursive=True):
            if e.path.rsplit("/", 1)[-1] == name:
                data = b"".join(self.vol.read_file(e.path))
                out.append((e.path[len(self.path) + 1:], data.decode(errors="replace")))
        return sorted(out)


def load(src) -> dict:
    events, tele = [], []
    for rel, text in src.files("events.jsonl"):
        events += [{**json.loads(line), "_src": rel} for line in text.splitlines() if line.strip().startswith("{")]
    for _, text in src.files("telemetry.jsonl"):
        tele += [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]
    events.sort(key=lambda e: e.get("ts", 0))
    jobs = [json.loads(t) for _, t in src.files("job.json") if t.strip()]
    return {"state": summarize_events(events), "events": events, "telemetry": tele[-1] if tele else None,
            "tele_hist": tele[-40:], "jobs": jobs}


def resolve_app(name: str, env: dict) -> str | None:
    """An `ap-...` id as is; a name → the newest app of that name that has not stopped (a `modal run --detach` app is
    ephemeral, so its name is looked up in `modal app list`)."""
    if name.startswith("ap-"):
        return name
    r = subprocess.run(["modal", "app", "list", "--json"], env=env, capture_output=True, text=True)
    try:
        apps = [a for a in json.loads(r.stdout) if a.get("Description") == name and not a.get("Stopped at")]
    except json.JSONDecodeError:
        return None
    return max(apps, key=lambda a: a.get("Created at") or "")["App ID"] if apps else None


class LogTail(threading.Thread):
    """`modal app logs <app>` in the background; keeps the last lines and counts error-like ones."""

    def __init__(self, profile: str | None, app: str, keep: int = 14):
        super().__init__(daemon=True)
        self.profile, self.app = profile, app
        self.lines: deque[str] = deque(maxlen=keep)
        self.flagged: deque[str] = deque(maxlen=6)
        self.n_flagged = 0

    def run(self) -> None:
        env = {**os.environ, **({"MODAL_PROFILE": self.profile} if self.profile else {})}
        while True:
            app = resolve_app(self.app, env)
            if app is None:
                self.lines.append(f"(no running app named {self.app}; retrying in 30 s)")
                time.sleep(30)
                continue
            p = subprocess.Popen(["modal", "app", "logs", app, "-f", "--timestamps"], env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                line = line.rstrip()
                if not line:
                    continue
                self.lines.append(line)
                if ERR_RE.search(line) and "[telemetry]" not in line:
                    self.n_flagged += 1
                    self.flagged.append(line)
            p.wait()
            self.lines.append(f"(log stream ended, rc {p.returncode}; reconnecting in 15 s)")
            time.sleep(15)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def eval_totals(events: list[dict]) -> dict[str, int]:
    """Episodes to run per unit, summed over the job's parts / shards (one `start` event per protocol run; a resumed
    run's start counts what it still had to run plus what was already recorded, so the latest start per part wins)."""
    latest: dict[tuple, dict] = {}
    for e in events:
        if e.get("kind") == "start" and isinstance(e.get("units"), dict):
            latest[(e.get("_src"), e.get("model"), e.get("arm"), e.get("split"))] = e["units"]
    totals: dict[str, int] = {}
    for units in latest.values():
        for u, v in units.items():
            totals[u] = totals.get(u, 0) + int(v.get("total") or 0)
    return totals


def render(data: dict, title: str, stale_s: float, logs: LogTail | None = None):
    from rich import box
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    st = data["state"]
    start = st["start"] or {}
    kind = start.get("kind_of_run") or ("train" if st["steps"] or st["devs"] else "eval")
    now = time.time()
    first = (data["events"][0]["ts"] if data["events"] else None)
    age = now - st["last_ts"] if st["last_ts"] else None
    finished = st["end"] is not None and not any(j.get("rc") is None for j in data["jobs"])
    status = ("FINISHED" if finished else "STOPPED: " + st["stop"]["reason"] if st["stop"] else
              "NO EVENTS YET" if age is None else f"STALE — no event for {_dur(age)}" if age > stale_s else "RUNNING")
    color = "red" if status.startswith(("STALE", "STOPPED")) or st["errors"] else ("green" if status == "RUNNING" else "yellow")
    up = (st["last_ts"] if finished else now) - first if first else None

    head = Table.grid(expand=True)
    head.add_column(); head.add_column(justify="right")
    steps = st["steps"]
    prog = ""
    if kind == "train":
        last = steps[-1] if steps else {}
        prog = f"step {last.get('step', 0)}/{start.get('max_steps', '?')}" + (f"  ETA {_dur(last.get('eta_s'))}" if last.get("eta_s") else "")
        if st["progress"] and (not steps or st["progress"]["ts"] > last.get("ts", 0)):
            p = st["progress"]
            prog += f"  |  {p.get('label')}: {p.get('done')}/{p.get('total')}"
    else:
        done, total = st["episodes"], sum(eval_totals(data["events"]).values())
        prog = f"{done}/{total} episodes ({100 * done / max(total, 1):.0f}%)"
    head.add_row(Text(f"{title}  [{kind}]  {prog}  ", style="bold"),
                 Text(f"{status}  |  up {_dur(up)}  ≈${(up or 0) / 3600 * RATE:.2f} GPU  |  "
                      f"{st['episodes']} episodes, {len(st['errors'])} errors", style=f"bold {color}"))
    parts = [Panel(head, box=box.HEAVY, border_style=color)]

    if kind == "train":
        t = Table(title="training steps (ART losses)", box=box.SIMPLE_HEAD, expand=True)
        for c in ("step", "reward", "±sd", "per unit", "flat", *[s for _, s in LOSS_KEYS[:8]], "gen tok", "roll", "train", "exc"):
            t.add_column(c, justify="right")
        for e in steps[-12:]:
            m = e.get("train_metrics") or {}
            t.add_row(str(e["step"]), _f(e.get("train_reward")), _f((e.get("train") or {}).get("reward_sd"), 2),
                      " ".join(f"{UNIT_ABBR.get(u, u)}{v:.2f}" for u, v in (e.get("train_by_unit") or {}).items()),
                      f"{e.get('flat_groups', '—')}/{e.get('groups', '—')}", *[_f(m.get(k), 4) for k, _ in LOSS_KEYS[:8]],
                      _k(e.get("output_tokens")), _dur(e.get("rollout_s")), _dur(e.get("train_s")),
                      Text(str(e.get("exceptions", 0)), style="red" if e.get("exceptions") else ""))
        parts.append(t)
        rw = [e.get("train_reward") for e in steps]
        loss = [(e.get("train_metrics") or {}).get("loss/train") for e in steps]
        kl = [(e.get("train_metrics") or {}).get("loss/kl_div") for e in steps]
        ent = [(e.get("train_metrics") or {}).get("loss/entropy") for e in steps]
        trend = Text()
        for name, vals in (("reward ", rw), ("loss   ", loss), ("entropy", ent), ("kl     ", kl),
                           ("dev    ", [d.get("dev_score") for d in st["devs"]])):
            if any(v is not None for v in vals):
                trend.append(f"{name} {spark(vals[-60:])}  last {_f(next((v for v in reversed(vals) if v is not None), None))}\n")
        dv = Table(title="dev evaluations (checkpoint selection)", box=box.SIMPLE_HEAD, expand=True)
        for c in ("step", "score", "per unit", "named", "probes", "entries", "alerts"):
            dv.add_column(c, justify="right")
        best = max((d for d in st["devs"] if d.get("step")), key=lambda d: d.get("dev_score") or -1, default=None)
        for d in st["devs"][-8:]:
            dd = d.get("dev") or {}
            probes = max((dd.get(k) or 0) for k in ("probe_many_entries", "probe_long_name", "probe_duplicates"))
            dv.add_row(str(d["step"]) + (" ★" if best is d else ""), _f(d.get("dev_score")),
                       " ".join(f"{UNIT_ABBR.get(u, u)}{v:.2f}" for u, v in (d.get("dev_by_unit") or {}).items()),
                       _f(dd.get("diagnosis_named"), 2), f"{probes:.0%}", _f(dd.get("entries"), 1),
                       Text("; ".join(d.get("alerts") or []) or "—", style="red" if d.get("alerts") else ""))
        tot = sum((e.get("rollout_s") or 0) + (e.get("train_s") or 0) for e in steps) + sum(d.get("val_s") or 0 for d in st["devs"])
        perf = Text("performance: ")
        if tot:
            for name, v in (("rollout", sum(e.get("rollout_s") or 0 for e in steps)), ("train", sum(e.get("train_s") or 0 for e in steps)),
                            ("dev", sum(d.get("val_s") or 0 for d in st["devs"]))):
                perf.append(f"{name} {_dur(v)} ({100 * v / tot:.0f}%)  ")
        parts += [Panel(Group(trend, perf), title="trends", box=box.ROUNDED), dv]
    else:
        ut = Table(title="evaluation by unit", box=box.SIMPLE_HEAD, expand=True)
        for c in ("unit", "done", "of", "mean reward", "errors", "forced", "at limit", "truncated", "turns", "tok/ep"):
            ut.add_column(c, justify="right")
        totals = eval_totals(data["events"])
        for u, c in sorted(st["units"].items()):
            n = c["done"] or 1
            ut.add_row(u, str(c["done"]), str(totals.get(u, "?")), _f(st["unit_reward"][u] / n),
                       Text(str(c["errors"]), style="red" if c["errors"] else ""), str(c["forced"]), str(c["limited"]),
                       str(c["truncated"]), f"{c['turns'] / n:.1f}", _k(c["tokens"] / n))
        parts.append(ut)

    tele = data["telemetry"]
    if tele:
        from importlib import import_module
        sys.path.insert(0, str(ROOT / "gpu"))
        fmt = import_module("telemetry").fmt
        gen = [s.get("gen_tok_s") for s in data["tele_hist"]]
        util = [(s.get("gpus") or [{}])[0].get("gpu_util") for s in data["tele_hist"]]
        parts.append(Panel(Text(fmt(tele).replace("[telemetry] ", "") + f"\ngpu util {spark(util)}   gen tok/s {spark(gen)}"),
                           title=f"utilization ({_dur(now - tele['ts'])} ago)", box=box.ROUNDED))
    issues = [f"{time.strftime('%H:%M:%S', time.localtime(e['ts']))} gt={e.get('gt_id')} {e.get('task')}: {str(e.get('error'))[:140]}"
              for e in st["errors"][-5:]] + [f"ALERT step {a.get('step')}: {a.get('alert')}" for a in st["alerts"][-5:]]
    if st["stop"]:
        issues.append(f"STOP: {st['stop'].get('reason')}")
    issues += [f"job rc {j.get('rc')}: {j.get('error')}" for j in data["jobs"] if j.get("error")]
    if issues:
        parts.append(Panel(Text("\n".join(issues), style="red"), title=f"errors / alerts ({len(st['errors'])} errors)", box=box.ROUNDED))
    if logs is not None:
        body = Text("\n".join(logs.lines) or "(waiting for the log stream)")
        parts.append(Panel(body, title=f"modal app logs {logs.app} — {logs.n_flagged} flagged lines", box=box.ROUNDED))
    return Group(*parts)


class _DemoLogs:
    """Stands in for LogTail in --demo: the formatted lines RunLog prints, as `modal app logs -f` would show them."""

    def __init__(self, keep: int = 14):
        self.app = "demo (synthetic)"
        self.lines: deque[str] = deque(maxlen=keep)
        self.n_flagged = 0

    def write(self, text: str) -> None:
        for line in text.splitlines():
            if line.strip():
                self.lines.append(line)
                self.n_flagged += bool(ERR_RE.search(line))

    def flush(self) -> None:
        pass


def demo_run(out: Path, logs: _DemoLogs, max_steps: int = 60, step_s: float = 2.0) -> None:
    """A synthetic training run in the shape of a real one (same events, plausible trends): rewards rising from the
    base's ~0.36, entropy falling, KL growing, a dev evaluation every 10 steps, an occasional failed episode, one
    monitor alert. Only for seeing the dashboard; nothing here is a measurement."""
    import random
    from eval.run_log import RunLog, flat_groups, fmt_dev, fmt_step, unit_means
    rng = random.Random(0)
    units = ["patient_diagnosis", "differential_diagnosis", "evidence_retrieval", "test_selection"]
    base = {"patient_diagnosis": 0.33, "differential_diagnosis": 0.42, "evidence_retrieval": 0.56, "test_selection": 0.28,
            "atypical_diagnosis": 0.35}
    rl = RunLog(out, "c5-main (DEMO)", stream=logs, progress_every=16, progress_s=1e9)
    rl.emit("start", "start: 60 steps x 8 groups x 8 rollouts, lr 1e-05, loss cispo, kl 0.0, 512 prompts (keep), "
                     "dev 100 every 10 steps, reward reward-v3", max_steps=max_steps, kind_of_run="train")
    best = None
    base_score = None

    def dev(step: int) -> None:
        nonlocal best, base_score
        lift = 0.12 * (1 - 2.7 ** (-step / 25))
        by = {u: min(1, b + lift * (0.8 if u == "atypical_diagnosis" else 1) + rng.gauss(0, 0.01)) for u, b in base.items()}
        score = sum(by[u] for u in units) / 4
        base_score = score if base_score is None else base_score
        if step and (best is None or score > best["dev_score"]):
            best = {"step": step, "dev_score": score}
        alerts = ["probe_long_name 6.0% of answers (> 5%)"] if step == 40 else []
        d = {"step": step, "dev_score": score, "dev_by_unit": by, "alerts": alerts,
             "dev": {"reward_mean": score, "diagnosis_named": 0.38 + lift, "diagnosis_coded": 0.30 + lift / 2,
                     "probe_long_name": 0.06 if step == 40 else 0.01, "entries": 3.4 + step / 60, "answer_chars": 520 + 3 * step}}
        rl.emit("dev", fmt_dev(d, best, base_score), **d, best_step=(best or {}).get("step"), val_s=95.0)
        for a in alerts:
            rl.emit("alert", f"ALERT at step {step}: {a}", step=step, alert=a)

    dev(0)
    durations = []
    for step in range(1, max_steps + 1):
        rl.begin_batch(f"step {step} rollout", 64)
        lift = 0.13 * (1 - 2.7 ** (-step / 20))
        groups = []
        for j in range(8):
            u = units[(step * 8 + j) % 4]
            p = base[u] + lift
            rs = [round(min(1, max(0, rng.gauss(p, 0.22))), 3) if rng.random() > 0.15 else 0.0 for _ in range(8)]
            if rng.random() < 0.2:
                rs = [0.0] * 8                                  # a prompt the policy cannot solve yet: no signal
            groups.append((u, rs))
            for r in rs:
                err = "ReadTimeout: policy server did not answer in 900s" if rng.random() < 0.004 else None
                rl.episode(u, 90000 + rng.randrange(9999), {"reward": r, "turns": rng.randint(3, 12),
                                                             "output_tokens": rng.randint(2500, 9000)}, err)
        rew = [r for _, rs in groups for r in rs]
        mean = sum(rew) / len(rew)
        sd = (sum((r - mean) ** 2 for r in rew) / len(rew)) ** 0.5
        roll, train = rng.uniform(280, 380), rng.uniform(80, 110)
        durations.append(roll + train)
        e = {"step": step, "train_reward": mean, "train": {"reward_sd": sd}, "episodes": 64, "exceptions": 0,
             "train_by_unit": unit_means([(u, r) for u, rs in groups for r in rs]), "groups": 8,
             "flat_groups": flat_groups([rs for _, rs in groups]), "output_tokens": rng.randint(300000, 420000),
             "rollout_s": roll, "train_s": train,
             "train_metrics": {"loss/train": rng.gauss(0.0, 0.02), "loss/entropy": 0.92 - 0.25 * step / max_steps + rng.gauss(0, 0.01),
                               "loss/kl_div": 0.0004 * step + abs(rng.gauss(0, 0.001)), "loss/grad_norm": abs(rng.gauss(0.3, 0.08)),
                               "loss/importance_ratio_mean": 1 + rng.gauss(0, 0.003), "loss/importance_ratio_p95": 1.05 + abs(rng.gauss(0, 0.02)),
                               "loss/clipped_token_fraction": abs(rng.gauss(0.004, 0.002)), "loss/learning_rate": 1e-5}}
        eta = sum(durations[-5:]) / len(durations[-5:]) * (max_steps - step)
        rl.emit("step", fmt_step(e, max_steps, eta), **e, eta_s=round(eta))
        if step % 10 == 0:
            dev(step)
        with (out / "telemetry.jsonl").open("a") as fh:
            fh.write(json.dumps({"ts": time.time(), "gpus": [{"gpu_util": rng.uniform(88, 99), "vram_used_gb": rng.uniform(68, 74),
                                                            "vram_total_gb": 80, "power_w": rng.uniform(560, 680)}],
                                 "vllm": {"running": rng.randint(40, 64), "waiting": 0, "kv_cache": rng.uniform(0.4, 0.7),
                                          "preemptions": 0}, "gen_tok_s": rng.uniform(2400, 3600), "load": 3.1, "ram_pct": 38}) + "\n")
        time.sleep(step_s)
    rl.emit("end", f"end: best checkpoint step {best['step']} (dev score {best['dev_score']:.3f} vs base {base_score:.3f})", best=best)
    rl.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile")
    ap.add_argument("--path", help="the job's directory on the results volume (e.g. rl/c5-main, c2)")
    ap.add_argument("--local", help="a local directory instead of a volume")
    ap.add_argument("--logs", help="also stream `modal app logs <app>` (clinical-envs-train / clinical-envs-vllm-eval)")
    ap.add_argument("--every", type=float, default=20.0)
    ap.add_argument("--stale", type=float, default=300.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--demo", action="store_true", help="a synthetic training run (plausible numbers, not a measurement) "
                                                        "to see the dashboard; a new step every 2 s")
    a = ap.parse_args(argv)
    from rich.console import Console
    from rich.live import Live
    logs = None
    if a.demo:
        import tempfile
        out = Path(tempfile.mkdtemp(prefix="rl_watch_demo_"))
        logs = _DemoLogs()
        if a.once:
            demo_run(out, logs, max_steps=24, step_s=0)
        else:
            threading.Thread(target=demo_run, args=(out, logs), daemon=True).start()
            time.sleep(0.5)
        a.local, a.every = str(out), 2.0
    if a.local:
        src, title = LocalSource(Path(a.local)), ("DEMO — synthetic data" if a.demo else "/".join(Path(a.local).parts[-2:]))
    elif a.profile and a.path:
        src, title = VolumeSource(a.profile, a.path), f"{a.profile}:{a.path}"
    else:
        ap.error("--local DIR, or --profile P --path DIR")
    console = Console()
    if a.once:
        console.print(render(load(src), title, a.stale, logs))
        return 0
    if a.logs:
        logs = LogTail(a.profile, a.logs)
        logs.start()
    data = load(src)
    with Live(render(data, title, a.stale, logs), console=console, refresh_per_second=1, screen=False) as live:
        last = time.time()
        while True:
            time.sleep(1)
            if time.time() - last >= a.every:
                try:
                    data = load(src)
                except Exception as exc:  # noqa: BLE001 — a failed fetch must not kill the dashboard
                    data["state"]["errors"] = data["state"]["errors"] + [{"ts": time.time(), "error": f"fetch failed: {exc}"}]
                last = time.time()
            live.update(render(data, title, a.stale, logs))


if __name__ == "__main__":
    sys.exit(main())
