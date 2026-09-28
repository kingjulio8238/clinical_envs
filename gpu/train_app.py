"""C5 on Modal: GRPO training with ART (LocalBackend: vLLM inference + LoRA training on one GPU).

    MODAL_PROFILE=<p> modal run --detach gpu/train_app.py --run-name c5-smoke --minutes 60 \
        --args "--max-steps 3 --rollouts-per-group 8 --groups-per-step 4 --val-every 3 --val-per-unit 5"

Runs scripts/train_rl.py (frozen-reward check at start-up) with its outputs (steps.jsonl, config.json, the .art
checkpoints) on the `clinical-envs-results` volume under rl/<run-name>/. Server-side timeout = --minutes.
"""
from __future__ import annotations

import shlex
from pathlib import Path

import modal

from common import HF_CACHE, RESULTS, committer, repo_image, tee

app = modal.App("clinical-envs-train")
image = repo_image(modal.Image.debian_slim(python_version="3.12").apt_install("git")
                   .pip_install("openpipe-art[backend]==0.5.20", "hf_transfer"))


@app.function(image=image, gpu="H100", volumes={"/hf": HF_CACHE, "/results": RESULTS}, timeout=60 * 60, max_containers=1)
def train(run_name: str, args: str) -> dict:
    import json
    import os
    import time
    out = Path("/results") / "rl" / run_name
    out.mkdir(parents=True, exist_ok=True)
    stop = committer(RESULTS)
    t0 = time.time()
    rc = tee(["python", "-u", "scripts/train_rl.py", "--run-name", run_name, "--art-path", str(out / ".art"), *shlex.split(args)],
             out / "train.log", env={"WANDB_MODE": os.environ.get("WANDB_MODE", "disabled"), "SH_RL_RESULTS": str(out)})
    info = {"rc": rc, "total_s": round(time.time() - t0, 1), "args": args}
    (out / "job.json").write_text(json.dumps(info, indent=1))
    stop.set()
    RESULTS.commit()
    HF_CACHE.commit()
    return info


@app.local_entrypoint()
def main(run_name: str, args: str = "", minutes: int = 60):
    print(train.with_options(timeout=minutes * 60).remote(run_name, args))
