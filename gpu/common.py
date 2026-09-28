"""Shared Modal pieces for the stage-C GPU jobs: volumes, the repo mount, the vLLM server launcher, the logs.

Rules (~/.claude/CLAUDE.md, modal): every function carries a server-side timeout sized ~1.5x its projection; launches
pin the profile (MODAL_PROFILE=...); month-to-date spend is checked before, during and after; after any disconnect
the first action is `bash gpu/kill_sweep.sh <profile>`.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
HF_CACHE = modal.Volume.from_name("clinical-envs-hf-cache", create_if_missing=True)
RESULTS = modal.Volume.from_name("clinical-envs-results", create_if_missing=True)
IGNORE = [".venv", ".git", "results", "data", "private", "scratchpad", "**/__pycache__", ".art", "audit/bench", "*.pyc",
          "docker", "harbor", "node_modules", ".pytest_cache"]
MODEL = "Qwen/Qwen3.5-9B"
VLLM_ARGS = ["--served-model-name", MODEL, "--max-model-len", "65536", "--reasoning-parser", "qwen3",
             "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--language-model-only",
             "--gpu-memory-utilization", "0.90", "--max-num-seqs", "128", "--port", "8000"]


def repo_image(base: modal.Image) -> modal.Image:
    return (base.pip_install_from_requirements(str(REPO / "requirements-rl.txt"))
            .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
            .add_local_python_source("common")
            .add_local_dir(str(REPO), "/repo", ignore=IGNORE))


def tee(cmd: list[str], log: Path, env: dict | None = None, cwd: str = "/repo") -> int:
    """Run a command, streaming its output to stdout (modal logs) and to a log file on the results volume."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        p = subprocess.Popen(cmd, cwd=cwd, env={**os.environ, **(env or {})}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True)
            fh.write(line); fh.flush()
        return p.wait()


def start_vllm(log: Path, extra: list[str] | None = None, wait_s: int = 1200) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("a")
    proc = subprocess.Popen(["vllm", "serve", MODEL, *VLLM_ARGS, *(extra or [])], stdout=fh, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with {proc.returncode}; see {log}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
                if r.status == 200:
                    print(f"vLLM ready after {time.time() - t0:.0f}s", flush=True)
                    return proc
        except Exception:  # noqa: BLE001
            time.sleep(5)
    proc.kill()
    raise RuntimeError(f"vLLM not ready after {wait_s}s; see {log}")


def committer(volume: modal.Volume, every_s: int = 60) -> threading.Event:
    """Commit the results volume periodically so progress is visible from outside while the job runs."""
    stop = threading.Event()

    def loop():
        while not stop.wait(every_s):
            try:
                volume.commit()
            except Exception as exc:  # noqa: BLE001
                print(f"volume commit failed: {exc}", flush=True)
    threading.Thread(target=loop, daemon=True).start()
    return stop
