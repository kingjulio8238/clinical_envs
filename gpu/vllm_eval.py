"""C2 / C4 on Modal: serve Qwen3.5-9B with vLLM on one GPU and run the environment client in the same container.

    MODAL_PROFILE=<p> modal run --detach gpu/vllm_eval.py --run-name smoke --minutes 40 \
        --cmd "-m eval.protocol_run --model qwen3.5-9b-local --local --panel --smoke --workers 32"
    MODAL_PROFILE=<p> modal run --detach gpu/vllm_eval.py --run-name c4 --minutes 240 \
        --cmd "scripts/group_variance.py --per-unit 250 --k 8 --workers 96"

`--cmd` is appended to `python -u` and gets `--out /results/<run-name>` (protocol_run / group_variance write there).
Outputs, the vLLM log and the client log land on the `clinical-envs-results` volume under <run-name>/:
    modal volume get clinical-envs-results <run-name> results/modal/
The function's timeout is `--minutes` (server-side: a lost laptop connection cannot keep the GPU running).
"""
from __future__ import annotations

import shlex
from pathlib import Path

import modal

from common import HF_CACHE, RESULTS, committer, repo_image, start_vllm, tee

app = modal.App("clinical-envs-vllm-eval")
image = repo_image(modal.Image.debian_slim(python_version="3.12").pip_install("vllm==0.30.0", "hf_transfer"))


@app.function(image=image, gpu="H100", volumes={"/hf": HF_CACHE, "/results": RESULTS}, timeout=60 * 60, max_containers=1)
def run(run_name: str, cmd: str) -> dict:
    import json
    import time
    out = Path("/results") / run_name
    out.mkdir(parents=True, exist_ok=True)
    stop = committer(RESULTS)
    t0 = time.time()
    proc = start_vllm(out / "vllm.log")
    ready = time.time() - t0
    HF_CACHE.commit()
    args = shlex.split(cmd)
    rc = tee(["python", "-u", *args, "--out", str(out)], out / "client.log", env={"SH_VLLM_URL": "http://127.0.0.1:8000/v1"})
    proc.terminate()
    info = {"rc": rc, "vllm_ready_s": round(ready, 1), "total_s": round(time.time() - t0, 1), "cmd": cmd}
    (out / "job.json").write_text(json.dumps(info, indent=1))
    stop.set()
    RESULTS.commit()
    return info


@app.local_entrypoint()
def main(run_name: str, cmd: str, minutes: int = 40):
    print(run.with_options(timeout=minutes * 60).remote(run_name, cmd))
