"""C2 / C4 on Modal: serve Qwen3.5-9B with vLLM on one GPU and run the environment client in the same container.

    MODAL_PROFILE=<p> modal run --detach gpu/vllm_eval.py --run-name smoke --minutes 40 \
        --cmd "-m eval.protocol_run --model qwen3.5-9b-local --local --panel --smoke --workers 32"
    MODAL_PROFILE=<p> modal run --detach gpu/vllm_eval.py --run-name c4 --minutes 240 \
        --cmd "scripts/group_variance.py --per-unit 250 --k 8 --workers 96"

`--cmd` is appended to `python -u` and gets `--out /results/<run-name>/partN` ("&&" separates several client commands
that share one server start). `--lora-path /results/rl/<run>/.art/.../checkpoints/<step>` serves the trained policy
(base + LoRA; the client's `--model qwen3.5-9b-rl`); both sides run on ART's managed vLLM runtime (P3).
Outputs, the vLLM log and the client log land on the `clinical-envs-results` volume under <run-name>/:
    modal volume get clinical-envs-results <run-name> results/modal/
The function's timeout is `--minutes` (server-side: a lost laptop connection cannot keep the GPU running).
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

import modal

from common import (JOB_CPU, HF_CACHE, MODEL, RESULTS, adapter_effect, check_served, committer, load_adapter, repo_image,
                    runtime_server_cmd, served_models, start_vllm, tee, with_art_runtime, Telemetry, cuda_base)

app = modal.App("clinical-envs-vllm-eval")
# Engine: ART's managed vLLM runtime — the engine that samples the policy during ART training — for both the before
# (base) and the after (base + LoRA) evaluation (P3; gpu/common.py launch_config).
image = repo_image(with_art_runtime(cuda_base()
                                    .pip_install("openpipe-art==0.5.20", "uv", "hf_transfer")))


@app.function(image=image, volumes={"/hf": HF_CACHE}, timeout=30 * 60, cpu=4)
def download() -> str:
    """Fetch the weights into the HF cache volume on a CPU container (no GPU billed for the download)."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "tokenizer*"])
    HF_CACHE.commit()
    return path


# The server-side timeout and GPU are fixed when the function is registered (Modal 1.x has no per-call override), so
# they come from the environment of the `modal run` process: gpu/launch.sh exports them from --minutes / --gpu.
JOB_MINUTES = int(os.environ.get("SH_JOB_MINUTES", "60"))
JOB_GPU = os.environ.get("SH_JOB_GPU", "H100")


@app.function(image=image, gpu=JOB_GPU, volumes={"/hf": HF_CACHE, "/results": RESULTS}, timeout=JOB_MINUTES * 60,
              max_containers=1, cpu=JOB_CPU)
def run(run_name: str, cmd: str, lora_path: str = "", engine_extra: str = "") -> dict:
    """`lora_path`: an ART checkpoint directory on the results volume
    (/results/rl/<run>/.art/clinical-envs/models/qwen35-9b-clinical/checkpoints/<step>) → the trained policy is served
    as base + LoRA and the client requests it as `qwen3.5-9b-rl`; empty → the base model as `qwen3.5-9b-local`."""
    import json
    import time
    out = Path("/results") / run_name
    out.mkdir(parents=True, exist_ok=True)
    stop = committer(RESULTS)
    tele = Telemetry(out, "http://127.0.0.1:8000/metrics")      # GPU / vLLM / host every 15 s → telemetry.jsonl
    tele.start()
    t0 = time.time()
    info: dict = {"rc": None, "cmd": cmd}
    try:
        rank = 16
        if lora_path:
            cfg = Path(lora_path) / "adapter_config.json"
            if not cfg.exists():
                raise FileNotFoundError(f"no LoRA adapter at {lora_path} (expected adapter_config.json)")
            rank = int(json.loads(cfg.read_text()).get("r", 16))
        extra = json.loads(engine_extra) if engine_extra else None      # engine experiments (e.g. prefix caching)
        proc = start_vllm(out / "vllm.log", runtime_server_cmd(rank, extra))
        ready = time.time() - t0
        HF_CACHE.commit()
        effect = None
        if lora_path:
            load_adapter(lora_path)
            effect = adapter_effect()
            if effect["gap"] == 0.0:
                raise RuntimeError(f"the adapter does not change the base's outputs ({effect}): not evaluating the base twice")
        models = served_models()
        name = check_served(models, lora_path or None)
        client_env = {"SH_VLLM_URL": "http://127.0.0.1:8000/v1", "SH_VLLM_BASE_MODEL": MODEL}
        if lora_path:   # unset for a base-only server, so `--model qwen3.5-9b-rl` there fails instead of scoring the base
            client_env["SH_VLLM_MODEL"] = name
        (out / "served.json").write_text(json.dumps({"models": models, "lora_path": lora_path, "lora_rank": rank,
                                                     "adapter_effect": effect, **client_env}, indent=1))
        rc = 0
        for i, part in enumerate(c.strip() for c in cmd.split("&&")):   # several client commands share one server start
            t1 = time.time()
            sub = out / f"part{i}"
            rc = tee(["python", "-u", *shlex.split(part), "--out", str(sub)], out / "client.log", env=client_env)
            (out / f"part{i}.json").write_text(json.dumps({"cmd": part, "rc": rc, "seconds": round(time.time() - t1, 1)}))
            RESULTS.commit()
            if rc:
                break
        proc.terminate()
        info = {"rc": rc, "vllm_ready_s": round(ready, 1), "total_s": round(time.time() - t0, 1), "cmd": cmd}
    except Exception as exc:        # a failed job leaves its reason on the volume, not only in the Modal log
        info = {"rc": -1, "error": f"{type(exc).__name__}: {exc}", "total_s": round(time.time() - t0, 1), "cmd": cmd}
        print(f"FATAL {info['error']} (see {out}/vllm.log)", flush=True)
        raise
    finally:
        tele.stop()
        (out / "job.json").write_text(json.dumps(info, indent=1))
        stop.set()
        RESULTS.commit()
    return info


@app.local_entrypoint()
def main(run_name: str = "", cmd: str = "", minutes: int = 40, gpu: str = "H100", download_only: bool = False,
         lora_path: str = "", engine_extra: str = ""):
    if download_only:
        print(download.remote())
        return
    if (minutes, gpu) != (JOB_MINUTES, JOB_GPU):
        raise SystemExit(f"--minutes {minutes} / --gpu {gpu} differ from the registered timeout {JOB_MINUTES} min / "
                         f"{JOB_GPU}: launch through gpu/launch.sh (it sets SH_JOB_MINUTES / SH_JOB_GPU)")
    print(run.remote(run_name, cmd, lora_path, engine_extra))
