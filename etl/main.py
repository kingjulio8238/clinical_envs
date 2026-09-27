"""CLI entry point for the ETL pipeline: every stage, in order, with a preflight and a manifest.

    python -m etl.main --stage 5                  one stage
    python -m etl.main --from 5 --to 10           a range
    python -m etl.main --all                      1..12
    python -m etl.main --all --pilot 20 --dry-run  what would run, on 20 records, without LLM calls

Stages (ROADMAP Stage 6 wired them all; before, only 1–6 ran from here):
     1  ingest source decks (.apkg) [+ PDF documents with --docs]
     2  classify cards
     3  extract board questions
     4  extract fact cards [+ LLM topic enrichment of cards without a topic]
     5  ontology: 5a LLM extraction, 5b ICD-10-CM mapping, 5d SNOMED, 5e SapBERT (when an index exists), 5f LOINC
     6  relationships: 6a question edges, 6b diagnosis–finding typing (LLM), 6c fact linking (LLM)
     7  EHR sections (LLM segmentation)
     8  patients: 8a clustering, 8b profiles (LLM)
     9  encounters: 9a timelines (LLM), 9b notes (template + LLM HPI polish)
    10  ground truth: 10a–10e
    11  typed diagnosis relations (needs the clinician-curated CSVs; --allow-missing-curated to build without)
    12  specialty-conditioned tiers

Before a stage runs, `etl.preflight` checks its inputs and exits 2 listing what is missing. After a stage,
`data/pipeline_manifest.json` records the settings fingerprint (LLM endpoint, model, temperature, seed),
the pipeline seed, SHA-256 of every input file the stage read, the git commit and the stage summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from etl import config
from etl.db import create_schema, get_connection
from etl.utils.logging import get_logger

log = get_logger("etl.main")

MANIFEST_NAME = "pipeline_manifest.json"


# ---------------------------------------------------------------------------
# stage table
# ---------------------------------------------------------------------------

def _stage_1(conn, a):
    from etl.stages.s01_ingest import run
    out = {"apkg": run(conn)}
    if a.docs:
        from etl.stages.s01d_ingest_docs import run as run_docs
        out["docs"] = run_docs(conn, files=None, dry_run=a.dry_run)
    return out


def _stage_2(conn, a):
    from etl.stages.s02_classify import run
    return run(conn)


def _stage_3(conn, a):
    from etl.stages.s03_extract_board import run
    return run(conn)


def _stage_4(conn, a):
    from etl.stages import s04c_topic_enrich as s04c
    from etl.stages.s04_fact_cards import run
    out = {"facts": run(conn)}
    cards = s04c._fetch_cards(conn)
    if cards and not a.dry_run:
        out["topic_enrichment"] = s04c.enrich(conn, cards[: a.pilot] if a.pilot else cards)
    else:
        out["topic_enrichment"] = {"skipped": True, "cards_needing_topics": len(cards)}
    return out


def _stage_5(conn, a):
    from etl.ontology import loinc, snomed
    from etl.ontology.icd10 import ICD10Dictionary
    from etl.stages import s05_ontology as s05
    out: dict = {}
    questions = s05._fetch_questions(conn, a.pilot)
    out["5a"] = s05.extract_all(conn, questions, dry_run=a.dry_run, workers=a.workers)
    if a.dry_run:
        return out
    out["5b"] = s05.map_and_insert(conn, ICD10Dictionary())
    if all(p.exists() for p in (snomed.CONCEPT_FILE, snomed.DESCRIPTION_FILE, snomed.EXTENDED_MAP_FILE)):
        from etl.ontology.snomed import SNOMEDDictionary
        out["5d"] = s05.run_snomed_mapping(conn, SNOMEDDictionary(), pilot=a.pilot)
        from etl.ontology.sapbert_embedder import DEFAULT_INDEX_PATH
        if Path(DEFAULT_INDEX_PATH).exists():
            out["5e"] = s05.run_snomed_embedding(conn, pilot=a.pilot)
        else:
            out["5e"] = {"skipped": "no SapBERT SNOMED index; build one with python -m etl.stages.s05_ontology --snomed-build-index"}
    else:
        out["5d"] = {"skipped": "SNOMED RF2 files absent: the LLM's SNOMED suggestions were kept (see preflight note)"}
        log.warning("stage 5: SNOMED files absent — LLM-suggested SNOMED ids are NOT validated")
    if loinc.LOINC_FILE.exists():
        out["5f"] = s05.run_loinc_mapping(conn, pilot=a.pilot)
    else:
        out["5f"] = {"skipped": "LOINC table absent: lab findings carry no LOINC code"}
    s05.verify(conn)
    return out


def _stage_6(conn, a):
    from etl.stages import s06_relationships as s06
    out = {"6a": s06.run_6a(conn, pilot=a.pilot)}
    if not a.dry_run:
        out["6b"] = s06.run_6b(conn, pilot=a.pilot, workers=a.workers)
        out["6c"] = s06.run_6c(conn, pilot=a.pilot, workers=a.workers)
    s06.verify(conn)
    return out


def _stage_7(conn, a):
    from etl.stages import s07_ehr_sections as s07
    if a.dry_run:
        return {"skipped": "dry run"}
    out = s07.run_7(conn, pilot=a.pilot, workers=a.workers)
    s07.verify(conn)
    return out


def _stage_8(conn, a):
    from etl.stages import s08_patients as s08
    out = {"8a": s08.run_8a(conn, pilot=a.pilot)}
    if not a.dry_run:
        out["8b"] = s08.run_8b(conn, pilot=a.pilot, workers=a.workers)
    s08.verify(conn)
    return out


def _stage_9(conn, a):
    from etl.stages import s09_encounters as s09
    if a.dry_run:
        return {"skipped": "dry run"}
    out = {"9a": s09.run_9a(conn, pilot=a.pilot, workers=a.workers)}
    s09.verify_9a(conn)
    out["9b"] = s09.run_9b(conn, pilot=a.pilot, workers=a.workers)
    s09.verify_9b(conn)
    return out


def _stage_10(conn, a):
    from etl.stages import s10_ground_truth as s10
    s10._apply_schema_migration_10(conn)
    out = {"10a": s10.run_10a(conn, a.pilot), "10b": s10.run_10b(conn, a.pilot)}
    if not a.dry_run:
        out["10c"] = s10.run_10c(conn, a.pilot, a.workers)
        out["10c_visit"] = s10.run_10c_visit(conn, a.pilot, a.workers)
    out["10d"] = s10.run_10d(conn, a.pilot)
    if not a.dry_run:
        out["10e"] = s10.run_10e(conn, a.pilot, a.workers)
    s10.verify_10(conn)
    return out


def _stage_11(conn, a):
    from etl.stages import s06d_diagnosis_relations as s06d
    return s06d.build_diagnosis_relations(conn, conn, include_residual=True,
                                          allow_missing_curated=a.allow_missing_curated)


def _stage_12(conn, a):
    from etl.stages import s10f_specialty as s10f
    return s10f.run_specialty(conn, pilot=a.pilot)


STAGES: dict[int, tuple[str, Callable]] = {
    1: ("ingest source decks", _stage_1),
    2: ("classify cards", _stage_2),
    3: ("extract board questions", _stage_3),
    4: ("extract fact cards + topic enrichment", _stage_4),
    5: ("ontology grounding (5a-5f)", _stage_5),
    6: ("relationships (6a-6c)", _stage_6),
    7: ("EHR sections", _stage_7),
    8: ("patients (8a, 8b)", _stage_8),
    9: ("encounters (9a, 9b)", _stage_9),
    10: ("ground truth (10a-10e)", _stage_10),
    11: ("typed diagnosis relations", _stage_11),
    12: ("specialty-conditioned tiers", _stage_12),
}


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _input_files(stage: int) -> dict[str, str | None]:
    """The files a stage reads, hashed, so a manifest identifies the exact inputs."""
    from etl.ontology import icd10, loinc, snomed
    from etl.stages import s06d_diagnosis_relations as s06d
    files: dict[str, Path] = {}
    if stage == 1:
        d = Path(os.environ.get("SH_APKG_DIR") or config.APKG_DIR)
        if d.exists():
            files.update({f"sources/{f.name}": f for f in sorted(d.iterdir()) if f.suffix in (".apkg", ".pdf")})
        for prof in sorted(Path(__file__).parent.glob("deck_profiles/*.yaml")):
            files[f"deck_profiles/{prof.name}"] = prof
    if stage == 5:
        files["icd10cm"] = icd10.ICD10_FILE
        files["snomed_concepts"] = snomed.CONCEPT_FILE
        files["snomed_descriptions"] = snomed.DESCRIPTION_FILE
        files["snomed_extended_map"] = snomed.EXTENDED_MAP_FILE
        files["loinc"] = loinc.LOINC_FILE
    if stage == 11:
        files["finding_site_review.csv"] = s06d.FINDING_SITE_GRADES
        files["curated_diagnosis_edges.csv"] = s06d.CURATED_EDGES
    return {k: _sha256(p) for k, p in files.items()}


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).parent).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_manifest(db_path: Path, stage: int, summary, started: float, a) -> Path:
    from etl.llm import LLMSettings
    path = db_path.parent / MANIFEST_NAME
    manifest = json.loads(path.read_text()) if path.exists() else {"runs": []}
    manifest["runs"].append({
        "stage": stage, "name": STAGES[stage][0],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "duration_s": round(time.time() - started, 1),
        "db": str(db_path), "pilot": a.pilot, "workers": a.workers, "dry_run": a.dry_run,
        "pipeline_seed": a.seed, "llm": LLMSettings.from_env().fingerprint() if stage in _llm_stages() else None,
        "inputs_sha256": _input_files(stage), "git_commit": _git_commit(),
        "summary": summary,
    })
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def _max_call_id(conn) -> int:
    try:
        return conn.execute("select coalesce(max(call_id), 0) from llm_call_log").fetchone()[0]
    except Exception:  # noqa: BLE001 — schema not created yet
        return 0


def _attribute_calls(conn, first_call: int) -> None:
    """Every llm_call_log row this stage wrote carries the sampling settings that produced it, even where
    a stage's own insert predates the new columns (Stage 6)."""
    from etl.llm import LLMSettings, ensure_log_columns
    try:
        ensure_log_columns(conn)
        s = LLMSettings.from_env()
        conn.execute("update llm_call_log set temperature = coalesce(temperature, ?), seed = coalesce(seed, ?), "
                     "endpoint = coalesce(endpoint, ?) where call_id > ?", (s.temperature, s.seed, s.endpoint, first_call))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not attribute llm_call_log rows: %s", exc)


def _llm_stages() -> set[int]:
    from etl.preflight import LLM_STAGES
    return LLM_STAGES


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Synthetic Hospital ETL: source decks -> benchmark database")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", type=int, choices=sorted(STAGES), help="run one stage")
    g.add_argument("--from", dest="from_stage", type=int, choices=sorted(STAGES), help="run from this stage (with --to, default 12)")
    g.add_argument("--all", action="store_true", help="run stages 1..12")
    ap.add_argument("--to", dest="to_stage", type=int, choices=sorted(STAGES), default=None)
    ap.add_argument("--db", type=str, default=str(config.DB_PATH), help=f"SQLite database (default {config.DB_PATH})")
    ap.add_argument("--sources", type=str, default=None, help="directory of .apkg decks (default apkg/; env SH_APKG_DIR)")
    ap.add_argument("--docs", action="store_true", help="stage 1 also ingests PDF documents (etl/pdf_profiles)")
    ap.add_argument("--pilot", type=int, default=None, help="limit each stage to the first N records")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SH_LLM_WORKERS", "10")), help="concurrent LLM calls")
    ap.add_argument("--dry-run", action="store_true", help="no LLM calls; deterministic sub-steps still run")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SH_PIPELINE_SEED", "0")), help="pipeline seed (recorded in the manifest)")
    ap.add_argument("--allow-missing-curated", action="store_true", help="stage 11: build without the clinician-curated CSVs (changes the labels)")
    ap.add_argument("--skip-preflight", action="store_true")
    return ap.parse_args(argv)


def selected_stages(a) -> list[int]:
    if a.all:
        return sorted(STAGES)
    if a.stage is not None:
        return [a.stage]
    return [s for s in sorted(STAGES) if a.from_stage <= s <= (a.to_stage or 12)]


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.sources:
        os.environ["SH_APKG_DIR"] = a.sources
        config.APKG_DIR = Path(a.sources)
    stages = selected_stages(a)
    db_path = Path(a.db)
    random.seed(a.seed)

    if not a.skip_preflight:
        from etl.preflight import problems
        fails, notes = problems(stages, db_path, a.allow_missing_curated, need_llm=not a.dry_run)
        for n in notes:
            log.warning("preflight note: %s", n)
        if fails:
            for f in fails:
                print(f"MISSING: {f}", file=sys.stderr)
            print(f"preflight failed: {len(fails)} required input(s) missing for stages {stages}; "
                  f"fix them or pass --skip-preflight to run anyway.", file=sys.stderr)
            return 2

    conn = get_connection(db_path)
    create_schema(conn)
    try:
        for stage in stages:
            name, fn = STAGES[stage]
            log.info("=== stage %d: %s ===", stage, name)
            started = time.time()
            first_call = _max_call_id(conn)
            summary = fn(conn, a)
            _attribute_calls(conn, first_call)
            conn.commit()
            path = write_manifest(db_path, stage, summary, started, a)
            print(json.dumps({"stage": stage, "name": name, "summary": summary}, indent=2, default=str))
            log.info("stage %d done in %.1fs; manifest %s", stage, time.time() - started, path)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
