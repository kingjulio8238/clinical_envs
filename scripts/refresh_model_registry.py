"""Keep the model registry honest against OpenRouter's live catalog (EVAL_PROTOCOL.md §4).

    python scripts/refresh_model_registry.py --check     # exit 1 if a registry id is not served (legacy ids reported)
    python scripts/refresh_model_registry.py --update    # rewrite eval/model_prices.json from live prices

No key needed: /api/v1/models is public.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.config import LEGACY_UNAVAILABLE, MODEL_REGISTRY  # noqa: E402

PRICES = ROOT / "eval" / "model_prices.json"


def live() -> dict[str, dict]:
    return {m["id"]: m for m in httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update", action="store_true")
    a = ap.parse_args()
    cat = live()
    missing, prices = [], {}
    for name, cfg in sorted(MODEL_REGISTRY.items()):
        if "openrouter" not in cfg.base_url:
            continue
        m = cat.get(cfg.model_id)
        if m is None:
            missing.append((name, cfg.model_id, name in LEGACY_UNAVAILABLE))
            continue
        p = m["pricing"]
        tools = "tools" in (m.get("supported_parameters") or [])
        prices[name] = {"id": cfg.model_id, "input": round(float(p["prompt"]) * 1e6, 4), "output": round(float(p["completion"]) * 1e6, 4),
                        "tools": tools, "context": m.get("context_length")}
        print(f"{name:20s} {cfg.model_id:40s} ${prices[name]['input']:>7.3f}/M in ${prices[name]['output']:>7.3f}/M out tools={tools}")
    for name, mid, legacy in missing:
        print(f"{name:20s} {mid:40s} NOT SERVED{' (legacy paper entry)' if legacy else ''}")
    if a.update:
        PRICES.write_text(json.dumps({"fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": "https://openrouter.ai/api/v1/models", "models": prices}, indent=2))
        print(f"wrote {PRICES}")
    unexpected = [m for m in missing if not m[2]]
    if a.check and unexpected:
        print(f"registry ids not served on OpenRouter: {[m[1] for m in unexpected]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
