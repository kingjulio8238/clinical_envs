"""C5 on Modal: GRPO training with ART (LocalBackend: vLLM inference + LoRA training on one GPU).

    MODAL_PROFILE=<p> modal run --detach gpu/train_app.py --run-name c5-smoke --minutes 60 \
        --args "--max-steps 3 --rollouts-per-group 8 --groups-per-step 4 --val-every 3 --val-per-unit 5"

Runs scripts/train_rl.py (frozen-reward check at start-up) with its outputs (steps.jsonl, config.json, the .art
checkpoints) on the `clinical-envs-results` volume under rl/<run-name>/. Server-side timeout = --minutes.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

import modal

from common import JOB_CPU, HF_CACHE, RESULTS, Telemetry, committer, cuda_base, repo_image, tee, with_art_runtime

app = modal.App("clinical-envs-train")
image = repo_image(with_art_runtime(cuda_base()
                                    # ART's backend pins torch==2.11.0+cu128, which only PyTorch's index serves
                                    .pip_install("openpipe-art[backend]==0.5.20", "uv", "hf_transfer",
                                                 extra_index_url="https://download.pytorch.org/whl/cu128")))


# Timeout fixed at registration (Modal 1.x has no per-call override): gpu/launch.sh exports SH_JOB_MINUTES from --minutes.
JOB_MINUTES = int(os.environ.get("SH_JOB_MINUTES", "60"))


@app.function(image=image, gpu="H100", volumes={"/hf": HF_CACHE, "/results": RESULTS}, timeout=JOB_MINUTES * 60,
              max_containers=1, cpu=JOB_CPU)
def train(run_name: str, args: str) -> dict:
    import json
    import os
    import time
    out = Path("/results") / "rl" / run_name
    out.mkdir(parents=True, exist_ok=True)
    stop = committer(RESULTS)
    tele = Telemetry(out)                  # GPU + ART's vLLM (its URL from server.json) every 15 s → telemetry.jsonl
    tele.start()
    t0 = time.time()
    rc = tee(["python", "-u", "scripts/train_rl.py", "--run-name", run_name, "--art-path", str(out / ".art"), *shlex.split(args)],
             out / "train.log", env={"WANDB_MODE": os.environ.get("WANDB_MODE", "disabled"), "SH_RL_RESULTS": str(out)})
    tele.stop()
    info = {"rc": rc, "total_s": round(time.time() - t0, 1), "args": args}
    (out / "job.json").write_text(json.dumps(info, indent=1))
    stop.set()
    RESULTS.commit()
    HF_CACHE.commit()
    return info


@app.local_entrypoint()
def main(run_name: str, args: str = "", minutes: int = 60):
    if minutes != JOB_MINUTES:
        raise SystemExit(f"--minutes {minutes} differs from the registered timeout {JOB_MINUTES} min: launch through "
                         "gpu/launch.sh (it sets SH_JOB_MINUTES)")
    print(train.remote(run_name, args))
