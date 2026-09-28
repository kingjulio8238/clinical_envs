"""Build eval/diagnosis_aliases.json: for every surviving diagnosis node, the other names the release itself gives
that concept — the CMS ICD-10 description of its code, its SNOMED CT description, and the names and descriptions of
the duplicate nodes merged into it (Stage 4) — so the scorer can credit a clinically identical name (RL readiness A1).

Deterministic: a function of the release DB only. Aliases whose normalized tokens equal the node's display name are
dropped; an alias whose tokens are shared by more than one surviving node is kept for each (the scorer only credits a
match to the reference node, so a shared alias cannot move credit between diagnoses).

    python scripts/build_diagnosis_aliases.py [--db benchmark_v1.3.db] [--check]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.scoring import _dx_tokens  # noqa: E402

OUT = ROOT / "eval" / "diagnosis_aliases.json"


def build(db: Path) -> dict[str, list[str]]:
    c = sqlite3.connect(db)
    rows = c.execute("select diagnosis_id, display_name, icd10_desc, snomed_desc, merged_into from diagnoses").fetchall()
    names: dict[int, list[str]] = defaultdict(list)
    display = {}
    for did, disp, icd, sno, merged in rows:
        target = merged or did
        if merged is None:
            display[did] = disp or ""
        for n in (disp if merged else None, icd, sno):
            if n and n.strip():
                names[target].append(n.strip())
    out = {}
    for did, disp in sorted(display.items()):
        seen = {_dx_tokens(disp)}
        keep = []
        for n in names.get(did, []):
            t = _dx_tokens(n)
            if t and t not in seen:
                seen.add(t)
                keep.append(n)
        if keep:
            out[str(did)] = keep
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "benchmark_v1.3.db"))
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed file differs from a fresh build")
    a = ap.parse_args()
    data = build(Path(a.db))
    text = json.dumps({"_source": "scripts/build_diagnosis_aliases.py", "aliases": data}, indent=0, sort_keys=True, ensure_ascii=False) + "\n"
    if a.check:
        ok = OUT.exists() and OUT.read_text() == text
        print("diagnosis_aliases.json is current" if ok else "diagnosis_aliases.json is STALE")
        return 0 if ok else 1
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(data):,} nodes with aliases, {sum(len(v) for v in data.values()):,} aliases, {OUT.stat().st_size/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
