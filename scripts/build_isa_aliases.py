"""Build eval/diagnosis_isa_aliases.json: for every surviving diagnosis node, the more specific diagnoses that are forms
of it (is-a), so the scorer can credit a correct answer that is more specific than a generic reference ("Serratia
marcescens bacteremia" for "Gram-negative rod infection"; RL readiness, B4 open decision).

The table is written by an LLM once and then frozen into the reward (eval/reward_lock.json), like the CMS / SNOMED
aliases: scoring stays deterministic. Every node is covered, so private-split references are not singled out.
Per-batch answers are cached (data/cache/isa_aliases/), so a rerun only asks for missing batches.

    python scripts/build_isa_aliases.py [--model gpt-6-sol] [--batch 50] [--workers 8] [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.adapters import create_adapter  # noqa: E402
from eval.config import MODEL_REGISTRY  # noqa: E402
from eval.scoring import _dx_tokens  # noqa: E402

OUT = ROOT / "eval" / "diagnosis_isa_aliases.json"
CACHE = ROOT / "data" / "cache" / "isa_aliases"
SYSTEM = """You curate a clinical terminology table. For each diagnosis you are given, list the more specific diagnoses
that ARE forms, subtypes or instances of it (an is-a relation), such that a clinician who expects the given diagnosis
as the answer would accept the specific one as correct.

Rules:
- Only true subtypes / specific forms / specific causative organisms or sites of the SAME condition.
- Never list causes, complications, associated or co-occurring conditions, differentials, or less specific terms.
- Standard clinical names (the form a physician writes on a problem list), no ICD codes, no explanations.
- At most 8 per diagnosis; an empty list when the diagnosis is already specific or you are unsure.

Reply with JSON only: {"<id>": ["specific diagnosis", ...], ...} with every given id present."""
PROMPT_SHA = hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]


def nodes(db: Path) -> list[dict]:
    c = sqlite3.connect(db)
    return [{"id": i, "name": n, "icd10": code or ""} for i, n, code in
            c.execute("select diagnosis_id, display_name, icd10_code from diagnoses where merged_into is null order by diagnosis_id")]


def ask(adapter, batch: list[dict], retries: int = 3) -> dict:
    key = hashlib.sha256(json.dumps([PROMPT_SHA, batch], sort_keys=True).encode()).hexdigest()[:20]
    path = CACHE / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    user = "\n".join(f'{b["id"]}: {b["name"]}' + (f' ({b["icd10"]})' if b["icd10"] else "") for b in batch)
    last = None
    for _ in range(retries):
        try:
            r = adapter.call(SYSTEM, user)
            t = r.text.strip().strip("`")
            t = t[t.find("{"): t.rfind("}") + 1]
            out = {str(k): [str(x) for x in (v or []) if isinstance(x, str)][:8] for k, v in json.loads(t).items()}
            if not all(str(b["id"]) in out for b in batch):
                raise ValueError("missing ids")
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"out": out, "usage": [r.input_tokens, r.output_tokens]}))
            return {"out": out, "usage": [r.input_tokens, r.output_tokens]}
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(3)
    raise RuntimeError(f"batch failed: {last}")


def clean(ref: str, aliases: list[str]) -> list[str]:
    """Drop entries that are not more specific names: empty, equal to or contained in the reference's words."""
    rt = _dx_tokens(ref)
    keep, seen = [], {rt}
    for a in aliases:
        t = _dx_tokens(a)
        if not t or t in seen or t <= rt:
            continue
        seen.add(t)
        keep.append(a.strip())
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6-sol")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N nodes (smoke)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ns = nodes(ROOT / "benchmark_v1.3.db")[: a.limit]
    batches = [ns[i:i + a.batch] for i in range(0, len(ns), a.batch)]
    print(f"{len(ns):,} nodes in {len(batches)} batches (prompt {PROMPT_SHA})", flush=True)
    if a.dry_run:
        return 0
    adapter = create_adapter(MODEL_REGISTRY[a.model])
    done, tin, tout = 0, 0, 0
    table: dict[str, list[str]] = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for batch, res in zip(batches, ex.map(lambda b: ask(adapter, b), batches)):
            for b in batch:
                al = clean(b["name"], res["out"].get(str(b["id"]), []))
                if al:
                    table[str(b["id"])] = al
            tin += res["usage"][0]; tout += res["usage"][1]; done += 1
            if done % 10 == 0 or done == len(batches):
                print(f"{done}/{len(batches)} batches ({100 * done // len(batches)}%), {tin:,} in / {tout:,} out tokens", flush=True)
    if a.limit:
        print(json.dumps(dict(list(table.items())[:15]), indent=1, ensure_ascii=False))
        return 0
    OUT.write_text(json.dumps({"_source": "scripts/build_isa_aliases.py", "_model": MODEL_REGISTRY[a.model].model_id,
                               "_prompt_sha": PROMPT_SHA, "aliases": table}, indent=0, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(table):,} nodes with specific forms, {sum(len(v) for v in table.values()):,} entries; "
          f"{tin:,} input / {tout:,} output tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
