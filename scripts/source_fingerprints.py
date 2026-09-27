"""Fingerprint a source corpus for contamination checks without shipping its text (ROADMAP Stage 6).

A benchmark built from question vignettes is only as unseen as its source. `--write` records, for every
board question in an ETL database, the SHA-256 of its normalized vignette and the set of hashed 8-word
shingles — no text leaves the machine. `--check` takes any corpus (a .jsonl with a `text` field, or a .txt with
one document per line) and reports which fingerprints it contains: exact matches, and documents whose shingle
overlap with a source item exceeds the threshold. Run it against public dumps you can obtain (or against the
training data of a model you evaluate, if you have it) before trusting the held-out split.

    python scripts/source_fingerprints.py --write --db data/benchmark.db --out data/source_fingerprints.json
    python scripts/source_fingerprints.py --check corpus.jsonl --fingerprints data/source_fingerprints.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

SHINGLE = 8


def normalize(text: str) -> str:
    """Lower-case, strip HTML and punctuation, collapse whitespace: what a copy would still match on."""
    t = re.sub(r"<[^>]+>", " ", text or "")
    t = re.sub(r"[^a-z0-9 ]+", " ", t.lower())
    return " ".join(t.split())


def shingles(text: str, k: int = SHINGLE) -> set[str]:
    words = normalize(text).split()
    if len(words) < k:
        return {hashlib.sha1(" ".join(words).encode()).hexdigest()[:16]} if words else set()
    return {hashlib.sha1(" ".join(words[i:i + k]).encode()).hexdigest()[:16] for i in range(len(words) - k + 1)}


def fingerprint(text: str) -> dict:
    n = normalize(text)
    return {"sha256": hashlib.sha256(n.encode()).hexdigest(), "words": len(n.split()), "shingles": sorted(shingles(text))}


def write(db: Path, out: Path, table: str = "board_questions", column: str = "vignette_text") -> dict:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(f"select question_id, {column} from {table} where {column} is not null and length({column}) > 0").fetchall()
    items = {str(qid): fingerprint(text) for qid, text in rows}
    payload = {"source_db": str(db), "table": table, "column": column, "shingle_words": SHINGLE, "items": items}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    return {"items": len(items), "out": str(out)}


def _docs(path: Path):
    if path.suffix == ".jsonl":
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if line:
                obj = json.loads(line)
                yield obj.get("id"), obj.get("text") or obj.get("content") or ""
    else:
        for i, line in enumerate(path.open(encoding="utf-8")):
            if line.strip():
                yield i, line


def check(corpus: Path, fingerprints: Path, threshold: float = 0.5) -> dict:
    fp = json.loads(fingerprints.read_text())
    exact = {v["sha256"]: qid for qid, v in fp["items"].items()}
    by_shingle: dict[str, set[str]] = {}
    for qid, v in fp["items"].items():
        for s in v["shingles"]:
            by_shingle.setdefault(s, set()).add(qid)
    exact_hits, near_hits, n_docs = [], [], 0
    for doc_id, text in _docs(corpus):
        n_docs += 1
        h = hashlib.sha256(normalize(text).encode()).hexdigest()
        if h in exact:
            exact_hits.append({"doc": doc_id, "question_id": exact[h]})
            continue
        counts: dict[str, int] = {}
        for s in shingles(text):
            for qid in by_shingle.get(s, ()):
                counts[qid] = counts.get(qid, 0) + 1
        for qid, c in counts.items():
            share = c / max(1, len(fp["items"][qid]["shingles"]))
            if share >= threshold:
                near_hits.append({"doc": doc_id, "question_id": qid, "shingle_share": round(share, 3)})
    return {"documents": n_docs, "source_items": len(fp["items"]), "exact_matches": exact_hits,
            "near_matches": sorted(near_hits, key=lambda x: -x["shingle_share"]), "threshold": threshold}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", type=str, help="corpus file (.jsonl with 'text', or .txt one document per line)")
    ap.add_argument("--db", default="data/benchmark.db")
    ap.add_argument("--out", default="data/source_fingerprints.json")
    ap.add_argument("--fingerprints", default="data/source_fingerprints.json")
    ap.add_argument("--threshold", type=float, default=0.5, help="shingle share that counts as a near match")
    a = ap.parse_args(argv)
    if a.write:
        print(json.dumps(write(Path(a.db), Path(a.out))))
        return 0
    r = check(Path(a.check), Path(a.fingerprints), a.threshold)
    print(json.dumps(r, indent=2))
    return 1 if r["exact_matches"] or r["near_matches"] else 0


if __name__ == "__main__":
    sys.exit(main())
