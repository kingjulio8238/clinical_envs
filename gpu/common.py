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
import urllib.error
import urllib.request
from pathlib import Path

import modal
from telemetry import Telemetry  # noqa: F401
from serving import MODEL, PROBE_MESSAGES, RL_SERVED_NAME, check_served, launch_config, logprob_gap  # noqa: F401

REPO = Path(__file__).resolve().parent.parent
HF_CACHE = modal.Volume.from_name("clinical-envs-hf-cache", create_if_missing=True)
RESULTS = modal.Volume.from_name("clinical-envs-results", create_if_missing=True)
IGNORE = [".venv", ".git", "results", "data", "private", "scratchpad", "**/__pycache__", ".art", "audit/bench", "*.pyc",
          "docker", "harbor", "node_modules", ".pytest_cache"]
ART_RUNTIME_ENV = {"ART_VLLM_RUNTIME_CACHE_DIR": "/opt/art-vllm-runtime",
                   # pinned so the build-time install (no GPU) and the GPU container agree on the runtime's hash
                   "ART_VLLM_RUNTIME_CUDA_PROFILE": "cuda12",
                   # PyTorch's top-k/top-p sampler instead of FlashInfer's: FlashInfer JIT-compiles its sampler with
                   # nvcc at first use (minutes per container; G1 2026-10-01 stalled there). Same for every job, so
                   # the before, the training rollouts and the after all sample the same way.
                   "VLLM_USE_FLASHINFER_SAMPLER": "0",
                   # compile caches on the weights volume (committed after the server starts): torch.compile and
                   # Triton run once per workspace instead of once per container
                   "VLLM_CACHE_ROOT": "/hf/compile/vllm", "TORCHINDUCTOR_CACHE_DIR": "/hf/compile/inductor",
                   "TRITON_CACHE_DIR": "/hf/compile/triton",
                   "ART_VLLM_RUNTIME_FLASHINFER_WORKSPACE_BASE": "/hf/compile/flashinfer",
                   # read by FlashInfer itself (the server is launched directly, not through ART's launcher):
                   # its JIT kernels (minutes of nvcc on first start) are then compiled once per workspace
                   "FLASHINFER_WORKSPACE_BASE": "/hf/compile/flashinfer"}
JOB_CPU = 8.0
"""CPU cores for the GPU containers: Modal's default reservation (a fraction of a core) starves compilation and the
environment's scoring threads."""


def repo_image(base: modal.Image) -> modal.Image:
    return (base.pip_install_from_requirements(str(REPO / "requirements-rl.txt"))
            .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
            .add_local_python_source("common", "serving", "telemetry")
            .add_local_dir(str(REPO), "/repo", ignore=IGNORE))


CUDA_BASE = "nvidia/cuda:12.9.1-devel-ubuntu22.04"


def cuda_base() -> modal.Image:
    """A CUDA *devel* base (nvcc + headers): vLLM's FlashInfer sampler and other kernels are JIT-compiled at first use
    and need the CUDA toolkit; a slim image fails at engine start ("Could not find nvcc", G1 2026-10-01). CUDA 12.9
    matches ART's runtime (cuda12 profile, cu129 wheels). The build fails here, on CPU, if nvcc is missing."""
    return (modal.Image.from_registry(CUDA_BASE, add_python="3.12")
            .apt_install("git", "build-essential", "ninja-build")
            .env({"CUDA_HOME": "/usr/local/cuda"})
            .run_commands("nvcc --version"))


def with_art_runtime(base: modal.Image) -> modal.Image:
    """ART's managed vLLM runtime (vLLM 0.25.1 + ART's patches; a uv sync of ART's pinned lockfile), installed at
    image-build time into the image so no GPU container spends its first minutes installing it."""
    return base.env(ART_RUNTIME_ENV).run_commands(
        "python -c 'from art.vllm_runtime import ensure_vllm_runtime as e; print(e(progress=print))'")


def tee(cmd: list[str], log: Path, env: dict | None = None, cwd: str = "/repo") -> int:
    """Run a command, streaming its output to stdout (modal logs) and to a log file on the results volume."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        p = subprocess.Popen(cmd, cwd=cwd, env={**os.environ, **(env or {})}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True)
            fh.write(line); fh.flush()
        return p.wait()


def runtime_server_cmd(lora_rank: int = 16, engine_extra: dict | None = None) -> list[str]:
    from art.vllm_runtime import VllmRuntimeLaunchConfig, build_vllm_runtime_server_cmd
    return build_vllm_runtime_server_cmd(VllmRuntimeLaunchConfig(**launch_config(lora_rank, engine_extra)))


def _post(path: str, body: dict, timeout: int = 300, port: int = 8000) -> tuple[int, str]:
    import json as _json
    headers = {"Content-Type": "application/json"}
    if os.environ.get("VLLM_API_KEY"):                  # the authenticated private endpoint (gpu/private_serve.py)
        headers["Authorization"] = f"Bearer {os.environ['VLLM_API_KEY']}"
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=_json.dumps(body).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def load_adapter(lora_path: str) -> None:
    """Load the trained LoRA into the running server as `RL_SERVED_NAME` (vLLM's runtime LoRA endpoint; ART's server
    enables it)."""
    code, text = _post("/v1/load_lora_adapter", {"lora_name": RL_SERVED_NAME, "lora_path": lora_path})
    if code != 200:
        raise RuntimeError(f"loading the adapter {lora_path} failed ({code}): {text[:500]}")


def adapter_effect() -> dict:
    """Greedy logprobs of one fixed prompt from the base and from the adapter: a trained adapter must change them —
    a zero gap means requests named `RL_SERVED_NAME` are silently answered by the base (the after run would be the
    before run)."""
    import json as _json
    lps = {}
    for name in (MODEL, RL_SERVED_NAME):
        code, text = _post("/v1/chat/completions", {"model": name, "messages": PROBE_MESSAGES, "max_tokens": 16,
                                                    "temperature": 0, "logprobs": True, "seed": 0})
        if code != 200:
            raise RuntimeError(f"probe request to {name} failed ({code}): {text[:500]}")
        lps[name] = [t["logprob"] for t in _json.loads(text)["choices"][0]["logprobs"]["content"]]
    return {"gap": logprob_gap(lps[MODEL], lps[RL_SERVED_NAME]), "tokens": len(lps[MODEL])}


def served_models() -> list[dict]:
    import json as _json
    req = urllib.request.Request("http://127.0.0.1:8000/v1/models")
    if os.environ.get("VLLM_API_KEY"):
        req.add_header("Authorization", f"Bearer {os.environ['VLLM_API_KEY']}")
    with urllib.request.urlopen(req, timeout=10) as r:
        return _json.loads(r.read())["data"]


def start_vllm(log: Path, cmd: list[str], wait_s: int = 1200) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("a")
    fh.write(f"$ {' '.join(cmd)}\n")
    fh.flush()
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with {proc.returncode}; last lines of {log}:\n{_tail(log)}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
                if r.status == 200:
                    print(f"vLLM ready after {time.time() - t0:.0f}s", flush=True)
                    return proc
        except Exception:  # noqa: BLE001
            time.sleep(5)
    proc.kill()
    raise RuntimeError(f"vLLM not ready after {wait_s}s; last lines of {log}:\n{_tail(log)}")


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def committer(volume: modal.Volume, every_s: int = 30) -> threading.Event:
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
