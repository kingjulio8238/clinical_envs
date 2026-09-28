"""P9: the pre-launch budget check. A failed billing query is UNKNOWN and refuses the launch (never $0); month-to-date
includes today's hours; a job must fit under the $25 stop line and its timeout under the $30 credit."""

from __future__ import annotations

import datetime as dt
import importlib
import os
import sys
from pathlib import Path

import pytest

GPU = Path(__file__).resolve().parents[2] / "gpu"
sys.path.insert(0, str(GPU))
B = importlib.import_module("budget")


def _fake_modal(tmp_path, monkeypatch, daily: str, hourly: str, fail: bool = False):
    """A `modal` on PATH that answers the billing report with canned JSON (or fails like the >31-day error)."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "modal").write_text(
        "#!/bin/bash\n"
        + ('echo "Daily reports cannot span more than 31 days" >&2; exit 1\n' if fail else "")
        + f"""if [[ " $* " == *" -r h "* ]]; then echo '{hourly}'; else echo '{daily}'; fi\n""")
    (bin_ / "modal").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")


def test_mtd_adds_today_hours(tmp_path, monkeypatch):
    _fake_modal(tmp_path, monkeypatch, '[{"Cost": "10.5"}, {"Cost": "2.25"}]', '[{"Cost": "1.25"}]')
    assert B.mtd("p", dt.date(2026, 10, 9)) == pytest.approx(14.0)
    assert B.mtd("p", dt.date(2026, 10, 1)) == pytest.approx(1.25)       # the 1st: no complete day yet


def test_failed_or_garbled_billing_is_unknown_and_refused(tmp_path, monkeypatch):
    _fake_modal(tmp_path, monkeypatch, "[]", "[]", fail=True)
    with pytest.raises(B.Unknown):
        B.mtd("p", dt.date(2026, 10, 9))
    assert B.main(["check", "--profile", "p", "--projected", "1"]) == 1
    with pytest.raises(B.Unknown):
        B.total([{"Cost": "n/a"}])


def test_projection_from_a_measured_job(tmp_path):
    """gpu/project.py: the largest part sets the rate; the projection adds the server start-up and a 1.5x timeout."""
    import json
    P = importlib.import_module("project")
    job = tmp_path / "g1"
    for i, (n, sec) in enumerate(((3, 300.0), (60, 1800.0))):
        d = job / f"part{i}" / "run"
        d.mkdir(parents=True)
        (d / "predictions.jsonl").write_text("".join(json.dumps({"output_tokens": 5000, "error": None}) + "\n" for _ in range(n)))
        (job / f"part{i}.json").write_text(json.dumps({"cmd": "x", "rc": 0, "seconds": sec}))
    (job / "job.json").write_text(json.dumps({"rc": 0, "vllm_ready_s": 360.0, "total_s": 2500.0}))
    m = P.measure(job)
    assert m["basis"] == "part1" and m["episodes_per_hour"] == pytest.approx(120.0) and m["tokens_per_episode"] == 5000
    pr = P.project(m, 1200)
    assert pr["hours"] == pytest.approx(10.1) and pr["usd"] == pytest.approx(10.1 * B.GPU_RATE["H100"], abs=0.01)
    assert pr["timeout_minutes"] == int(10.1 * 60 * 1.5) + 10
    assert P.project(m, 1200, tokens_per_episode=10000)["episodes_per_hour"] == pytest.approx(60.0)


def test_decide():
    assert B.decide(None, 1.0)[0] is False
    assert B.decide(10.0, 0.0)[0] is False                              # no projection, no launch
    assert B.decide(25.0, 0.5)[0] is False                              # at the stop line
    assert B.decide(20.0, 6.0)[0] is False                              # projection crosses $25
    assert B.decide(20.0, 4.0)[0] is True
    assert B.decide(20.0, 4.0, worst=12.0)[0] is False                  # the timeout could reach $32
    assert B.decide(20.0, 4.0, worst=9.0)[0] is True
