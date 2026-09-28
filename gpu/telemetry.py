"""System telemetry for a GPU job: GPU utilization / memory / power (nvidia-smi), the vLLM server's own metrics
(requests running and waiting, KV-cache use, generated tokens/s, preemptions), host load. A daemon thread samples every
`every_s`, appends to `<out>/telemetry.jsonl` and prints a line every `print_s` (PufferLib's utilization panel, for a
detached Modal job). Modal-free, so the parsing is testable anywhere.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

VLLM_GAUGES = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
               "vllm:kv_cache_usage_perc": "kv_cache", "vllm:gpu_cache_usage_perc": "kv_cache"}
VLLM_COUNTERS = {"vllm:generation_tokens_total": "gen_tokens", "vllm:prompt_tokens_total": "prompt_tokens",
                 "vllm:num_preemptions_total": "preemptions", "vllm:prefix_cache_hits_total": "prefix_hits",
                 "vllm:prefix_cache_queries_total": "prefix_queries"}


def parse_nvidia_smi(text: str) -> list[dict]:
    """`nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits`."""
    out = []
    for line in text.strip().splitlines():
        try:
            util, used, total, power = (float(x.strip()) for x in line.split(",")[:4])
        except ValueError:
            continue
        out.append({"gpu_util": util, "vram_used_gb": used / 1024, "vram_total_gb": total / 1024, "power_w": power})
    return out


def parse_vllm_metrics(text: str) -> dict:
    """Prometheus text → the gauges and counters above, summed over label sets."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        key = VLLM_GAUGES.get(name) or VLLM_COUNTERS.get(name)
        if not key:
            continue
        try:
            out[key] = out.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
        except ValueError:
            continue
    return out


def fmt(sample: dict) -> str:
    g = sample.get("gpus") or []
    parts = []
    if g:
        parts.append("gpu " + ", ".join(f"{x['gpu_util']:.0f}% {x['vram_used_gb']:.1f}/{x['vram_total_gb']:.0f}GB {x['power_w']:.0f}W" for x in g))
    v = sample.get("vllm")
    if v:
        parts.append(f"vllm running {v.get('running', 0):.0f} waiting {v.get('waiting', 0):.0f} "
                     f"kv {100 * v.get('kv_cache', 0):.0f}% gen {sample.get('gen_tok_s', 0):.0f} tok/s "
                     f"preempt {v.get('preemptions', 0):.0f}")
    if sample.get("load") is not None:
        parts.append(f"load {sample['load']:.1f} ram {sample.get('ram_pct', 0):.0f}%")
    return "[telemetry] " + (" | ".join(parts) or "no data")


def _ram_pct() -> float | None:
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        total = float(info["MemTotal"].split()[0])
        avail = float(info["MemAvailable"].split()[0])
        return 100 * (1 - avail / total)
    except Exception:  # noqa: BLE001
        return None


class Telemetry(threading.Thread):
    """`metrics_url`: a fixed vLLM /metrics URL, or None to look for `<out>/server.json` ({"base_url": ".../v1"}) that
    the training script writes once ART's server is up."""

    def __init__(self, out: Path, metrics_url: str | None = None, every_s: float = 15, print_s: float = 60):
        super().__init__(daemon=True)
        self.out, self.metrics_url, self.every_s, self.print_s = out, metrics_url, every_s, print_s
        self.stop_event = threading.Event()
        self._last_gen: tuple[float, float] | None = None

    def _url(self) -> str | None:
        if self.metrics_url:
            return self.metrics_url
        f = self.out / "server.json"
        try:
            base = json.loads(f.read_text()).get("base_url")
        except Exception:  # noqa: BLE001
            return None
        return base.rstrip("/").removesuffix("/v1") + "/metrics" if base else None

    def sample(self) -> dict:
        s: dict = {"ts": round(time.time(), 1)}
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            s["gpus"] = parse_nvidia_smi(r.stdout)
        except Exception:  # noqa: BLE001
            s["gpus"] = []
        url = self._url()
        if url:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    v = parse_vllm_metrics(r.read().decode())
                s["vllm"] = v
                if "gen_tokens" in v:
                    now = time.time()
                    if self._last_gen:
                        s["gen_tok_s"] = max(0.0, (v["gen_tokens"] - self._last_gen[1]) / max(now - self._last_gen[0], 1e-6))
                    self._last_gen = (now, v["gen_tokens"])
            except Exception:  # noqa: BLE001
                pass
        try:
            s["load"] = os.getloadavg()[0]
        except OSError:
            pass
        s["ram_pct"] = _ram_pct()
        return s

    def run(self) -> None:
        last_print = 0.0
        with (self.out / "telemetry.jsonl").open("a") as fh:
            while not self.stop_event.is_set():
                s = self.sample()
                fh.write(json.dumps(s) + "\n")
                fh.flush()
                if time.time() - last_print >= self.print_s:
                    print(fmt(s), flush=True)
                    last_print = time.time()
                self.stop_event.wait(self.every_s)

    def stop(self) -> None:
        self.stop_event.set()
