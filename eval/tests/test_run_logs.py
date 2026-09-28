"""Run logs: every training / evaluation job emits formatted stdout lines and events.jsonl (steps with ART's losses,
dev evaluations, episodes, errors), GPU / vLLM telemetry parses, and the terminal dashboard renders from them."""

from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path

import pytest

from eval import degenerate as D
from eval import run_log as L

GPU = Path(__file__).resolve().parents[2] / "gpu"


def test_step_and_dev_lines_carry_the_losses_and_signals():
    e = {"step": 7, "train_reward": 0.41, "train": {"reward_sd": 0.23}, "train_by_unit": {"patient_diagnosis": 0.38},
         "groups": 8, "flat_groups": 2, "output_tokens": 352000, "rollout_s": 120.0, "train_s": 60.0, "exceptions": 1,
         "train_metrics": {"loss/train": 0.0123, "loss/entropy": 0.84, "loss/kl_div": 0.0021, "loss/grad_norm": 0.31,
                           "loss/importance_ratio_mean": 1.0, "loss/clipped_token_fraction": 0.003}}
    line = L.fmt_step(e, 60, 3600)
    for part in ("step 7/60", "reward 0.410 ±0.23", "pd 0.38", "no-signal groups 2/8", "loss 0.0123", "ent 0.8400",
                 "kl 0.0021", "gnorm 0.3100", "ratio 1.0000", "clip 0.0030", "352k tok (3k/s)", "exc 1", "ETA 1h00m"):
        assert part in line, (part, line)
    d = {"step": 10, "dev_score": 0.45, "dev_by_unit": {"test_selection": 0.3}, "alerts": ["probe_long_name 9% of answers"],
         "dev": {"diagnosis_named": 0.4, "probe_long_name": 0.09, "entries": 3.0, "answer_chars": 400}}
    dl = L.fmt_dev(d, {"step": 10, "dev_score": 0.45}, 0.40)
    assert "★ new best" in dl and "base 0.400" in dl and "ts 0.30" in dl and "probes 9.0%" in dl and "probe_long_name" in dl
    assert L.flat_groups([[1, 1, 1], [0, 1], [0.5]]) == 1


def test_errors_are_printed_immediately_and_progress_is_batched(tmp_path):
    buf = io.StringIO()
    rl = L.RunLog(tmp_path, "t", stream=buf, progress_every=2)
    rl.begin_batch("step 1 rollout", 4)
    rl.episode("patient_diagnosis", 1, {"reward": 1.0})
    rl.episode("patient_diagnosis", 2, None, error="ReadTimeout: policy server")
    rl.episode("test_selection", 3, {"reward": 0.5, "limit_reason": "max_turns", "turns": 43})
    rl.close()
    out = buf.getvalue()
    assert "ERROR episode gt=2 (patient_diagnosis): ReadTimeout" in out
    assert "step 1 rollout: 2/4 episodes (50%), mean reward 0.500" in out and "1 errors" in out
    ev = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    st = L.summarize_events(ev)
    assert st["episodes"] == 3 and len(st["errors"]) == 1 and st["units"]["test_selection"]["limited"] == 1


def test_telemetry_parsing():
    sys.path.insert(0, str(GPU))
    T = importlib.import_module("telemetry")
    g = T.parse_nvidia_smi("97, 72908, 81559, 612.3\n")
    assert g[0]["gpu_util"] == 97 and round(g[0]["vram_used_gb"], 1) == 71.2
    v = T.parse_vllm_metrics('# HELP x\nvllm:num_requests_running{model_name="m"} 118.0\nvllm:num_requests_waiting{model_name="m"} 3.0\n'
                             'vllm:kv_cache_usage_perc{model_name="m"} 0.63\nvllm:generation_tokens_total{model_name="m"} 1.5e6\n'
                             'vllm:num_preemptions_total{model_name="m"} 2\nother_metric 5\n')
    assert v == {"running": 118.0, "waiting": 3.0, "kv_cache": 0.63, "gen_tokens": 1.5e6, "preemptions": 2.0}
    assert "running 118" in T.fmt({"gpus": g, "vllm": v, "gen_tok_s": 3400, "load": 1.0, "ram_pct": 40})


def test_dry_run_training_is_fully_logged_and_the_dashboard_renders(tmp_path, monkeypatch):
    if not D.DEFAULT_DB.exists():
        pytest.skip("release DB not present")
    import scripts.rl_watch as W
    import scripts.train_rl as T
    monkeypatch.setenv("SH_RL_RESULTS", str(tmp_path))
    assert T.main(["--run-name", "dry", "--dry-run", "--max-steps", "2", "--rollouts-per-group", "2", "--groups-per-step", "2",
                   "--val-every", "2", "--val-per-unit", "1"]) == 0
    ev = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    kinds = [e["kind"] for e in ev]
    assert kinds[0] == "start" and kinds[-1] == "end" and kinds.count("step") == 2 and kinds.count("dev") == 2
    step = next(e for e in ev if e["kind"] == "step")
    assert "loss/train" in step["train_metrics"] and step["flat_groups"] == 2 and step["train_by_unit"]
    assert sum(1 for e in ev if e["kind"] == "episode") == 2 * 2 * 2 + 2 * 5
    from rich.console import Console
    con = Console(file=io.StringIO(), width=200)
    con.print(W.render(W.load(W.LocalSource(tmp_path)), "dry", 300))
    text = con.file.getvalue()
    assert "training steps" in text and "dev evaluations" in text and "FINISHED" in text and "step 2/2" in text
