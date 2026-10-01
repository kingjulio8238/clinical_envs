"""Modal month-to-date spend per workspace and the pre-launch budget check (P9 of audit/PRE_OCT1_TODO.md).

    python gpu/budget.py mtd                                  # every workspace (one profile each), MTD and headroom
    python gpu/budget.py check --profile sales-32662 --projected 6.5 --minutes 150   # exit 0 only if the job fits

Rules (~/.claude/CLAUDE.md, modal): $30 of credit per workspace per calendar month; stop launching on a workspace at
$25 month-to-date; a job launches only if MTD + its projection stays under $25 and MTD + its worst case (the
server-side timeout at the GPU's rate) stays under the $30 credit. A billing query that fails is
UNKNOWN, never $0 — and an UNKNOWN workspace fails the check.

Month-to-date = the daily report for the complete days of this month + the hourly report for today (Modal reports
full intervals only: a daily report omits today, and hourly reports cannot span more than 7 days). The current hour
is still missing, so a running job's own spend since the top of the hour is not in the number: add it
(elapsed × rate) when polling during a run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

CREDIT = 30.0
STOP_AT = 25.0
GPU_RATE = {"H100": 4.40, "A100-80GB": 2.90, "L40S": 2.35}
"""$/hour for one GPU container: Modal's GPU price (H100 $3.95, A100-80GB $2.50, L40S $1.95) plus 8 CPU cores and memory."""


class Unknown(Exception):
    """The billing query failed: the workspace's spend is unknown (never treated as $0)."""


def _report(profile: str, *args: str) -> list[dict]:
    r = subprocess.run(["modal", "billing", "report", *args, "--json"], capture_output=True, text=True,
                       env={**os.environ, "MODAL_PROFILE": profile})
    if r.returncode:
        raise Unknown(f"{profile}: modal billing report {' '.join(args)} failed: {(r.stderr or r.stdout).strip()[:300]}")
    try:
        rows = json.loads(r.stdout)
    except json.JSONDecodeError as exc:
        raise Unknown(f"{profile}: unreadable billing report ({exc}): {r.stdout[:200]!r}") from exc
    if not isinstance(rows, list):
        raise Unknown(f"{profile}: unexpected billing report {rows!r:.200}")
    return rows


def total(rows: list[dict]) -> float:
    try:
        return sum(float(r["Cost"]) for r in rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise Unknown(f"billing rows without a numeric Cost: {exc}") from exc


def mtd(profile: str, today: dt.date | None = None) -> float:
    """Month-to-date spend of the profile's workspace, up to the last complete hour (UTC). Raises Unknown."""
    today = today or dt.datetime.now(dt.timezone.utc).date()
    first = today.replace(day=1)
    days = _report(profile, "--start", first.isoformat(), "--end", today.isoformat()) if today > first else []
    hours = _report(profile, "--start", today.isoformat(), "-r", "h")
    return total(days) + total(hours)


def workspaces() -> dict[str, list[str]]:
    r = subprocess.run(["modal", "profile", "list", "--json"], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"modal profile list failed: {r.stderr.strip()}")
    out: dict[str, list[str]] = {}
    for p in json.loads(r.stdout):
        out.setdefault(p["workspace"], []).append(p["name"])
    return out


def decide(spent: float | None, projected: float, worst: float | None = None, stop_at: float = STOP_AT,
           credit: float = CREDIT) -> tuple[bool, str]:
    """Whether a job may launch on a workspace that has spent `spent` (None = unknown): its projection must fit under
    the stop line, and its worst case — the server-side timeout at the GPU's rate — under the credit."""
    if spent is None:
        return False, "month-to-date UNKNOWN (billing query failed): not launching"
    if projected <= 0:
        return False, f"projection ${projected:.2f} is not a projection: smoke the job and project its cost first"
    after = spent + projected
    if spent >= stop_at:
        return False, f"MTD ${spent:.2f} is at or past the ${stop_at:.0f} stop line"
    if after > stop_at:
        return False, f"MTD ${spent:.2f} + projected ${projected:.2f} = ${after:.2f} > ${stop_at:.0f}: split the job or use another workspace"
    if worst is not None and spent + worst > credit:
        return False, (f"worst case (the timeout) ${worst:.2f} + MTD ${spent:.2f} = ${spent + worst:.2f} > the ${credit:.0f} credit: "
                       "shorten --minutes or split the job")
    tail = f"; worst case at the timeout ${spent + worst:.2f} ≤ ${credit:.0f}" if worst is not None else ""
    return True, f"MTD ${spent:.2f} + projected ${projected:.2f} = ${after:.2f} ≤ ${stop_at:.0f}{tail}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mtd")
    c = sub.add_parser("check")
    c.add_argument("--profile", required=True)
    c.add_argument("--projected", type=float, required=True, help="the job's projected cost in dollars (from its smoke)")
    c.add_argument("--minutes", type=float, default=None, help="the job's server-side timeout")
    c.add_argument("--gpu", default="H100", choices=sorted(GPU_RATE))
    a = ap.parse_args(argv)
    if a.cmd == "mtd":
        for ws, profiles in sorted(workspaces().items()):
            try:
                s = mtd(profiles[0])
                state = f"MTD ${s:6.2f}  headroom to ${STOP_AT:.0f}: ${max(STOP_AT - s, 0):5.2f}" + ("  STOP" if s >= STOP_AT else "")
            except Unknown as exc:
                state = f"MTD UNKNOWN ({exc})"
            print(f"{ws:16s} [{', '.join(profiles)}]  {state}", flush=True)
        return 0
    try:
        spent = mtd(a.profile)
    except Unknown as exc:
        print(exc, file=sys.stderr)
        spent = None
    worst = a.minutes / 60 * GPU_RATE[a.gpu] if a.minutes else None
    ok, why = decide(spent, a.projected, worst)
    print(f"{a.profile}: {'GO' if ok else 'REFUSED'} — {why}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
