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


# PufferLib's dashboard palette and helpers (pufferl.py print_dashboard / abbreviate / duration / fmt_perf): cyan
# labels, default-colour values, dim units, bright-cyan section heads, one rounded bright-cyan frame.
C1, C2, B1, B2 = "[cyan]", "[dim default]", "[bright_cyan]", "[default]"


def abbreviate(num: float | None) -> str:
    if num is None:
        return f"{C2}—"
    if num < 1e3:
        return f"{B2}{num:.0f}{C2}" if isinstance(num, float) else f"{B2}{num}{C2}"
    for div, unit in ((1e3, "K"), (1e6, "M"), (1e9, "B")):
        if num < div * 1e3:
            return f"{B2}{num / div:.1f}{C2}{unit}"
    return f"{B2}{num / 1e12:.2f}{C2}T"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return f"{C2}—"
    s = max(int(seconds), 0)
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{B2}{h}{C2}h {B2}{m}{C2}m {B2}{s}{C2}s" if h else f"{B2}{m}{C2}m {B2}{s}{C2}s" if m else f"{B2}{s}{C2}s"


def num(x: float | None, nd: int = 3) -> str:
    return f"{C2}—" if x is None else f"{B2}{_f(float(x), nd)}"


def pct(x: float | None, nd: int = 1) -> str:
    return f"{C2}—" if x is None else f"{B2}{x:.{nd}f}{C2}%"


def fmt_perf(name: str, color: str, total: float, secs: float) -> tuple[str, str, str]:
    p = 0 if not total else int(100 * secs / total - 1e-5)
    return f"{color}{name}", duration(secs), f"{B2}{p:2d}{C2}%"


def _stats_pairs(tables, pairs: list[tuple[str, str]]) -> None:
    for i, (k, v) in enumerate(pairs):
        tables[i % len(tables)].add_row(f"{B2}{k}", v)


UNIT_LABEL = {"patient_diagnosis": "Diagnosis", "differential_diagnosis": "Differential", "evidence_retrieval": "Retrieval",
              "test_selection": "Test select", "atypical_diagnosis": "Atypical"}


def budget(height: int | None) -> dict:
    """Rows per section so the dashboard fits the terminal (Rich's Live cuts off whatever does not)."""
    if not height:
        return {"steps": 8, "devs": 6, "issues": 5, "logs": 8, "stats": 40}
    extra = max(height - 43, 0)            # frame, rules between sections, header, summary, stats
    clamp = lambda x, lo, hi: int(max(lo, min(hi, x)))
    return {"steps": clamp(extra * 0.35, 3, 8), "devs": clamp(extra * 0.2, 2, 6), "issues": clamp(extra * 0.15, 2, 5),
            "logs": clamp(extra * 0.3, 0, 8), "stats": 40 if height >= 60 else 24}


def render(data: dict, title: str, stale_s: float, logs=None, height: int | None = None):
    import rich.box
    from rich.console import Group
    from rich.table import Table
    st = data["state"]
    start = st["start"] or {}
    kind = start.get("kind_of_run") or ("train" if st["steps"] or st["devs"] else "eval")
    now = max(time.time(), st["last_ts"] or 0)
    first = data["events"][0]["ts"] if data["events"] else None
    age = now - st["last_ts"] if st["last_ts"] else None
    finished = st["end"] is not None and not any(j.get("rc") is None for j in data["jobs"])
    if finished:
        status, scol = "finished", "[green]"
    elif st["stop"]:
        status, scol = "stopped: " + str(st["stop"].get("reason")), "[red]"
    elif age is None:
        status, scol = "waiting for events", "[yellow]"
    elif age > stale_s:
        status, scol = f"stale — no event for {_dur(age)}", "[red]"
    else:
        status, scol = "running", "[green]"
    bad = scol == "[red]" or bool(st["errors"])
    up = ((st["last_ts"] if finished else now) - first) if first else None
    rows = budget(height)
    tele = data["telemetry"] or {}
    g = (tele.get("gpus") or [{}])[0]
    v = tele.get("vllm") or {}

    # PufferLib's frame (rounded, bright cyan, no header) with a rule between sections (show_lines)
    dashboard = Table(box=rich.box.ROUNDED, expand=True, show_header=False, show_lines=True,
                      border_style="red" if bad else "bright_cyan")
    head = Table(box=None, expand=True, show_header=False)
    head.add_column(justify="left", ratio=3)
    for _ in range(4):
        head.add_column(justify="center", ratio=1)
    head.add_row(f"{B1}Clinical RL {C2}· {B2}{title} {C2}· {scol}{status}",
                 f"{C1}GPU: {B2}{g.get('gpu_util', 0):.1f}{C2}%" if g else f"{C1}GPU: {C2}—",
                 (f"{C1}VRAM: {B2}{100 * g['vram_used_gb'] / g['vram_total_gb']:.1f}{C2}%" if g.get("vram_total_gb") else f"{C1}VRAM: {C2}—"),
                 f"{C1}DRAM: " + pct(tele.get("ram_pct")),
                 f"{C1}KV cache: " + pct(100 * v["kv_cache"] if "kv_cache" in v else None))
    dashboard.add_row(head)

    steps, devs = st["steps"], st["devs"]
    last = steps[-1] if steps else {}
    m = last.get("train_metrics") or {}
    best = max((d for d in devs if d.get("step")), key=lambda d: d.get("dev_score") or -1, default=None)
    base_dev = next((d.get("dev_score") for d in devs if d.get("step") == 0), None)

    # -- summary | performance | losses ----------------------------------------------------------------------------
    s = Table(box=None, expand=True)
    s.add_column(f"{C1}Summary", justify="left", vertical="top", ratio=5)
    s.add_column(f"{C1}Value", justify="right", vertical="top", ratio=6)
    p = Table(box=None, expand=True)
    p.add_column(f"{C1}Performance", justify="left", ratio=5)
    p.add_column(f"{C1}Time", justify="right", ratio=5)
    p.add_column(f"{C1}%", justify="right", ratio=2)
    lt = Table(box=None, expand=True)
    lt.add_column(f"{C1}Losses", justify="left", ratio=5)
    lt.add_column(f"{C1}Value", justify="right", ratio=4)
    if kind == "train":
        max_steps = start.get("max_steps")
        gen = sum(e.get("output_tokens") or 0 for e in steps)
        roll = sum(e.get("rollout_s") or 0 for e in steps)
        prog = st["progress"]
        s.add_row(f"{B2}Run", f"{B2}{title}")
        s.add_row(f"{B2}Step", f"{B2}{last.get('step', 0)}{C2}/{B2}{max_steps or '?'}")
        s.add_row(f"{B2}Episodes", abbreviate(st["episodes"]))
        s.add_row(f"{B2}Gen tokens", abbreviate(gen))
        s.add_row(f"{B2}Tok/s", abbreviate(gen / roll) if roll else f"{C2}—")
        s.add_row(f"{B2}Uptime", duration(up))
        s.add_row(f"{B2}Remaining", duration(last.get("eta_s")))
        s.add_row(f"{B2}GPU cost", f"{C2}≈${B2}{(up or 0) / 3600 * RATE:.2f}")
        if prog and (not steps or prog.get("ts", 0) > last.get("ts", 0)):
            s.add_row(f"{B2}Now", f"{B2}{prog.get('label')} {prog.get('done')}{C2}/{B2}{prog.get('total')}")
        t_roll, t_train = roll, sum(e.get("train_s") or 0 for e in steps)
        t_dev = sum(d.get("val_s") or 0 for d in devs)
        tot = t_roll + t_train + t_dev
        n = max(len(steps), 1)
        p.add_row(*fmt_perf("Step", B1, tot, t_roll + t_train))
        p.add_row(*fmt_perf("  Rollout", B2, tot, t_roll))
        p.add_row(*fmt_perf("  Learn", B2, tot, t_train))
        p.add_row(*fmt_perf("Dev eval", B1, tot, t_dev))
        p.add_row(f"{B1}Per step", duration((t_roll + t_train) / n), "")
        p.add_row(f"{B2}  Rollout", duration(t_roll / n), "")
        p.add_row(f"{B2}  Learn", duration(t_train / n), "")
        for key, name in (("loss/train", "policy_loss"), ("loss/entropy", "entropy"), ("loss/kl_div", "approx_kl"),
                          ("loss/kl_policy_ref", "kl_ref"), ("loss/grad_norm", "grad_norm"),
                          ("loss/importance_ratio_mean", "ratio_mean"), ("loss/importance_ratio_p95", "ratio_p95"),
                          ("loss/clipped_token_fraction", "clipfrac"), ("loss/learning_rate", "learning_rate")):
            if key in m:
                lt.add_row(f"{B2}{name}", num(m[key], 3 if abs(m[key]) >= 1e-3 or m[key] == 0 else 6))
        if not m:
            lt.add_row(f"{C2}(after the first step)", "")
    else:
        from collections import Counter
        totals = eval_totals(data["events"])
        done, total = st["episodes"], sum(totals.values())
        rate = done / up if up else None
        s.add_row(f"{B2}Job", f"{B2}{title}")
        s.add_row(f"{B2}Episodes", f"{abbreviate(done)}{C2}/{abbreviate(total)}")
        s.add_row(f"{B2}Done", pct(100 * done / total if total else None))
        s.add_row(f"{B2}Episodes/h", abbreviate(rate * 3600) if rate else f"{C2}—")
        s.add_row(f"{B2}Uptime", duration(up))
        s.add_row(f"{B2}Remaining", duration((total - done) / rate) if rate and total else f"{C2}—")
        s.add_row(f"{B2}GPU cost", f"{C2}≈${B2}{(up or 0) / 3600 * RATE:.2f}")
        tok = sum(c["tokens"] for c in st["units"].values())
        p.add_row(f"{B1}Generation", "", "")
        p.add_row(f"{B2}  Tokens", abbreviate(tok), "")
        p.add_row(f"{B2}  Tok/s", abbreviate(tele.get("gen_tok_s")) if tele.get("gen_tok_s") else f"{C2}—", "")
        p.add_row(f"{B2}  Running", abbreviate(v.get("running")), "")
        p.add_row(f"{B2}  Waiting", abbreviate(v.get("waiting")), "")
        p.add_row(f"{B2}  Preempted", abbreviate(v.get("preemptions")), "")
        lim = Counter()
        for c in st["units"].values():
            lim.update(c)
        n = max(done, 1)
        lt.columns[0].header = f"{C1}Episode outcomes"
        lt.add_row(f"{B2}errors", f"{B2}{lim['errors']}")
        lt.add_row(f"{B2}forced answer", pct(100 * lim["forced"] / n))
        lt.add_row(f"{B2}at a limit", pct(100 * lim["limited"] / n))
        lt.add_row(f"{B2}truncated turns", pct(100 * lim["truncated"] / n))
        lt.add_row(f"{B2}turns / episode", num(lim["turns"] / n, 1))
        lt.add_row(f"{B2}tokens / episode", abbreviate(lim["tokens"] / n))
    monitor = Table(box=None, expand=True, pad_edge=False, show_header=False)
    monitor.add_column(ratio=4); monitor.add_column(ratio=3); monitor.add_column(ratio=3)
    monitor.add_row(s, p, lt)
    dashboard.add_row(monitor)

    # -- user stats (two columns, as PufferLib) --------------------------------------------------------------------
    left, right = Table(box=None, expand=True), Table(box=None, expand=True)
    for t in (left, right):
        t.add_column(f"{C1}User Stats", justify="left", ratio=3)
        t.add_column(f"{C1}Value", justify="right", ratio=2)
    pairs: list[tuple[str, str]] = []
    if kind == "train":
        tr = last.get("train") or {}
        pairs += [("train/reward", num(last.get("train_reward"))), ("train/reward_sd", num(tr.get("reward_sd")))]
        pairs += [(f"train/{u}", num(r)) for u, r in (last.get("train_by_unit") or {}).items()]
        pairs += [("train/no_signal_groups", f"{B2}{last.get('flat_groups', '—')}{C2}/{B2}{last.get('groups', '—')}"),
                  ("train/exceptions", f"{B2}{last.get('exceptions', 0)}"),
                  ("train/tokens_per_episode", abbreviate((last.get("output_tokens") or 0) / max(last.get("episodes") or 1, 1))),
                  ("train/answer_entries", num(tr.get("entries"), 2)), ("train/probe_long_name", pct(100 * (tr.get("probe_long_name") or 0)))]
        if devs:
            d = devs[-1]
            dd = d.get("dev") or {}
            pairs += [("dev/score", num(d.get("dev_score"))), ("dev/base_score", num(base_dev)),
                      ("dev/best", f"{num((best or {}).get('dev_score'))}{C2} @ {B2}{(best or {}).get('step', '—')}")]
            pairs += [(f"dev/{u}", num(r)) for u, r in (d.get("dev_by_unit") or {}).items()]
            probes = max((dd.get(k) or 0) for k in ("probe_many_entries", "probe_long_name", "probe_duplicates"))
            pairs += [("dev/diagnosis_named", num(dd.get("diagnosis_named"))), ("dev/diagnosis_coded", num(dd.get("diagnosis_coded"))),
                      ("dev/probe_rate", pct(100 * probes)), ("dev/answer_chars", abbreviate(dd.get("answer_chars")))]
    else:
        totals = eval_totals(data["events"])
        for u, c in sorted(st["units"].items()):
            nn = c["done"] or 1
            pairs.append((f"{u}/reward", num(st["unit_reward"][u] / nn)))
            pairs.append((f"{u}/done", f"{B2}{c['done']}{C2}/{B2}{totals.get(u, '?')}"))
    if kind == "train":                    # an evaluation's per-unit stats are the unit table below
        _stats_pairs([left, right], pairs[:rows["stats"]])
        stats = Table(box=None, expand=True, pad_edge=False, show_header=False)
        stats.add_column(ratio=1); stats.add_column(ratio=1)
        stats.add_row(left, right)
        dashboard.add_row(stats)

    # -- history: recent steps and dev evaluations (same palette) --------------------------------------------------
    if kind == "train" and steps:
        h = Table(box=None, expand=True, title=None)
        cols = [("Step", "right"), ("Reward", "right"), ("±sd", "right"), ("No-signal", "right"), ("Loss", "right"),
                ("Entropy", "right"), ("KL", "right"), ("Grad norm", "right"), ("Ratio p95", "right"), ("Clip", "right"),
                ("Tokens", "right"), ("Rollout", "right"), ("Learn", "right")]
        for c, j in cols:
            h.add_column(f"{C1}{c}", justify=j)
        for e in steps[-rows["steps"]:]:
            mm = e.get("train_metrics") or {}
            h.add_row(f"{B2}{e['step']}", num(e.get("train_reward")), num((e.get("train") or {}).get("reward_sd"), 2),
                      f"{B2}{e.get('flat_groups', '—')}{C2}/{B2}{e.get('groups', '—')}", num(mm.get("loss/train"), 4),
                      num(mm.get("loss/entropy")), num(mm.get("loss/kl_div"), 4), num(mm.get("loss/grad_norm")),
                      num(mm.get("loss/importance_ratio_p95")), pct(100 * mm["loss/clipped_token_fraction"], 2) if "loss/clipped_token_fraction" in mm else f"{C2}—",
                      abbreviate(e.get("output_tokens")), duration(e.get("rollout_s")), duration(e.get("train_s")))
        trend = Table(box=None, expand=True, show_header=False)
        trend.add_column(ratio=1); trend.add_column(ratio=1)
        sp = lambda key: spark([(e.get("train_metrics") or {}).get(key) for e in steps][-40:])
        trend.add_row(f"{C1}Trend  reward  {B2}{spark([e.get('train_reward') for e in steps][-40:])}",
                      f"{C1}entropy {B2}{sp('loss/entropy')}")
        trend.add_row(f"{C1}       dev     {B2}{spark([d.get('dev_score') for d in devs])}", f"{C1}kl      {B2}{sp('loss/kl_div')}")
        dashboard.add_row(Group(h, trend))
    if kind == "train" and devs:
        dv = Table(box=None, expand=True)
        units = list((devs[-1].get("dev_by_unit") or {}).keys())
        for c in ["Dev step", "Score", *[UNIT_LABEL.get(u, u) for u in units], "Named", "Probes", "Alerts"]:
            dv.add_column(f"{C1}{c}", justify="left" if c == "Alerts" else "right")
        for d in devs[-rows["devs"]:]:
            dd = d.get("dev") or {}
            probes = max((dd.get(k) or 0) for k in ("probe_many_entries", "probe_long_name", "probe_duplicates"))
            star = f" {B1}★" if best is d else ""
            dv.add_row(f"{B2}{d['step']}{star}", num(d.get("dev_score")), *[num((d.get("dev_by_unit") or {}).get(u)) for u in units],
                       num(dd.get("diagnosis_named"), 2), pct(100 * probes),
                       ("[red]" + "; ".join(d["alerts"])) if d.get("alerts") else f"{C2}none")
        dashboard.add_row(dv)
    if kind != "train" and st["units"]:
        ut = Table(box=None, expand=True)
        for c in ("Unit", "Done", "Of", "Reward", "Errors", "Forced", "At limit", "Truncated", "Turns", "Tok/ep"):
            ut.add_column(f"{C1}{c}", justify="left" if c == "Unit" else "right")
        totals = eval_totals(data["events"])
        for u, c in sorted(st["units"].items()):
            nn = c["done"] or 1
            ut.add_row(f"{B2}{u}", f"{B2}{c['done']}", f"{B2}{totals.get(u, '?')}", num(st["unit_reward"][u] / nn),
                       (f"[red]{c['errors']}" if c["errors"] else f"{B2}0"), f"{B2}{c['forced']}", f"{B2}{c['limited']}",
                       f"{B2}{c['truncated']}", num(c["turns"] / nn, 1), abbreviate(c["tokens"] / nn))
        dashboard.add_row(ut)

    # -- problems, then the live log tail --------------------------------------------------------------------------
    issues = [f"{time.strftime('%H:%M:%S', time.localtime(e['ts']))}  gt={e.get('gt_id')}  {e.get('task')}  {str(e.get('error'))[:140]}"
              for e in st["errors"][-rows["issues"]:]] + [f"ALERT step {a.get('step')}: {a.get('alert')}" for a in st["alerts"][-2:]]
    if st["stop"]:
        issues.append(f"STOP: {st['stop'].get('reason')}")
    issues += [f"job rc {j.get('rc')}: {j.get('error')}" for j in data["jobs"] if j.get("error")]
    if issues:
        it = Table(box=None, expand=True)
        it.add_column(f"[red]Errors / alerts {C2}({B2}{len(st['errors'])}{C2} errors, {B2}{len(st['alerts'])}{C2} alerts)")
        for line in issues:
            it.add_row(f"[red]{line}")
        dashboard.add_row(it)
    if logs is not None and rows["logs"]:
        lg = Table(box=None, expand=True)
        lg.add_column(f"{C1}Log {C2}({logs.app}, {B2}{logs.n_flagged}{C2} flagged)", no_wrap=True, overflow="ellipsis")
        for line in list(logs.lines)[-rows["logs"]:]:
            lg.add_row(("[red]" if ERR_RE.search(line) else C2) + line.replace("[", "\\["))
        dashboard.add_row(lg)
    return Group(dashboard)


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
    clock = [time.time() - (max_steps * 420 if step_s == 0 else 0)]     # simulated: ~7 min per step
    units = ["patient_diagnosis", "differential_diagnosis", "evidence_retrieval", "test_selection"]
    base = {"patient_diagnosis": 0.33, "differential_diagnosis": 0.42, "evidence_retrieval": 0.56, "test_selection": 0.28,
            "atypical_diagnosis": 0.35}
    rl = RunLog(out, "c5-main (DEMO)", stream=logs, progress_every=16, progress_s=1e9, clock=lambda: clock[0])
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
        clock[0] += 95.0
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
                clock[0] += 4.5
                err = "ReadTimeout: policy server did not answer in 900s" if rng.random() < 0.004 else None
                rl.episode(u, 90000 + rng.randrange(9999), {"reward": r, "turns": rng.randint(3, 12),
                                                             "output_tokens": rng.randint(2500, 9000)}, err)
        rew = [r for _, rs in groups for r in rs]
        mean = sum(rew) / len(rew)
        sd = (sum((r - mean) ** 2 for r in rew) / len(rew)) ** 0.5
        roll, train = rng.uniform(280, 380), rng.uniform(80, 110)
        clock[0] += roll - 64 * 4.5 + train
        durations.append(roll + train)
        e = {"step": step, "train_reward": mean, "train": {"reward_sd": sd, "entries": 3.3 + step / 50, "probe_long_name": 0.01}, "episodes": 64, "exceptions": 0,
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
            fh.write(json.dumps({"ts": clock[0], "gpus": [{"gpu_util": rng.uniform(88, 99), "vram_used_gb": rng.uniform(68, 74),
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
    view = lambda: render(data, title, a.stale, logs, console.size.height)
    try:
        with Live(view(), console=console, refresh_per_second=1, screen=True) as live:
            last = time.time()
            while True:
                time.sleep(1)
                if time.time() - last >= a.every:
                    try:
                        data = load(src)
                    except Exception as exc:  # noqa: BLE001 — a failed fetch must not kill the dashboard
                        data["state"]["errors"] = data["state"]["errors"] + [{"ts": time.time(), "error": f"fetch failed: {exc}"}]
                    last = time.time()
                live.update(view())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
