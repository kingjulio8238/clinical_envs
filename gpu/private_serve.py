"""P5: the one private-split run (criterion 7) — the model on Modal, the labels on this machine.

The trained policy and the base are served from Modal as an authenticated web endpoint (ART's runtime, the same
`launch_config` as every other evaluation, the adapter loaded as `qwen3.5-9b-rl`); the client and the scorer run
locally with the private overlay (`private/labels_v1.3.db`). The model only ever receives prompts built from the
release DB, which holds no private labels, so the labels never leave this machine.

    # once: a random bearer token as a Modal secret on the workspace, and the same token locally
    TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
    MODAL_PROFILE=credited modal secret create clinical-envs-vllm-key VLLM_API_KEY=$TOKEN
    # serve (foreground: the app lives only while this command runs — a closed laptop stops the GPU)
    SH_LORA_PATH=/results/ckpt/c5-main-0040 MODAL_PROFILE=credited modal serve gpu/private_serve.py
    # in another terminal: the endpoint URL printed by `modal serve`, then the private runs
    bash scripts/run_private.sh https://<workspace>--clinical-envs-private-serve-server.modal.run $TOKEN

Failsafes: the container scales down after `SCALEDOWN_S` without requests, and `modal serve` stops the app when the
local command exits; `LIFETIME_S` caps one container's life regardless (server-side).
"""
from __future__ import annotations

import os
import subprocess
import threading
import time

import modal

from common import JOB_CPU, HF_CACHE, RESULTS, Telemetry, cuda_base, load_adapter, repo_image, runtime_server_cmd, with_art_runtime
from pathlib import Path

LORA_PATH = os.environ.get("SH_LORA_PATH", "")          # read when `modal serve` imports this file locally
PORT = 8000
SCALEDOWN_S = 10 * 60
LIFETIME_S = 6 * 3600

app = modal.App("clinical-envs-private-serve")
image = repo_image(with_art_runtime(cuda_base()
                                    .pip_install("openpipe-art==0.5.20", "uv", "hf_transfer")
                                    .env({"SH_LORA_PATH": LORA_PATH})))


@app.function(image=image, gpu="H100", volumes={"/hf": HF_CACHE, "/results": RESULTS}, timeout=LIFETIME_S,
              scaledown_window=SCALEDOWN_S, max_containers=1, cpu=JOB_CPU,
              secrets=[modal.Secret.from_name("clinical-envs-vllm-key")])
@modal.concurrent(max_inputs=256)
@modal.web_server(port=PORT, startup_timeout=1500)
def server():
    """ART's vLLM runtime on the base with the bearer token (VLLM_API_KEY); the adapter is loaded once the server is up."""
    out = Path("/results/private-serve")
    out.mkdir(parents=True, exist_ok=True)
    lora = os.environ.get("SH_LORA_PATH", "")
    rank = 16
    if lora:
        import json
        rank = int(json.loads((Path(lora) / "adapter_config.json").read_text()).get("r", 16))
    cmd = [c.replace("--host=127.0.0.1", "--host=0.0.0.0") for c in runtime_server_cmd(rank)]
    subprocess.Popen(cmd, stdout=(out / "vllm.log").open("a"), stderr=subprocess.STDOUT)
    Telemetry(out, f"http://127.0.0.1:{PORT}/metrics").start()

    def after_ready():
        import urllib.request
        for _ in range(300):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                time.sleep(5)
        if lora:
            load_adapter(lora)
            print(f"adapter {lora} loaded as qwen3.5-9b-rl", flush=True)
    threading.Thread(target=after_ready, daemon=True).start()
