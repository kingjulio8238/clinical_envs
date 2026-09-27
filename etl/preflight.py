"""Preflight: every input a stage needs, checked before the stage runs (ROADMAP Stage 6).

The audit found the pipeline silently degrading when inputs were absent: no source decks → an empty run,
a missing `curated_diagnosis_edges.csv` → different specialty labels, no SNOMED files → the LLM's SNOMED
ids kept. `problems(stages)` returns a human-readable line per missing input, with the path or environment
variable that fixes it; `etl.main` prints them and exits 2.

    python -m etl.preflight --stage 5            # or --all
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from etl import config
from etl.deck_registry import REGISTRY

ROOT = Path(__file__).resolve().parent.parent

LLM_STAGES = {4, 5, 6, 7, 8, 9, 10}
"""Stages that call the LLM (4 only for topic enrichment of fact cards; 1 only with PDF documents)."""


@dataclass
class Requirement:
    stage: int
    what: str
    check: Callable[[], str | None]      # None when satisfied, else the problem
    optional: bool = False               # optional inputs are reported as notes, not failures


def _file(path: Path, what: str, fix: str) -> Callable[[], str | None]:
    def check():
        return None if path.exists() else f"{what}: {path} is missing. {fix}"
    return check


def _sources_dir() -> Path:
    return Path(os.environ.get("SH_APKG_DIR") or config.APKG_DIR)


def _sources_check() -> str | None:
    d = _sources_dir()
    if not d.exists():
        return f"source directory {d} does not exist. Put your .apkg decks there or set SH_APKG_DIR / --sources."
    decks = [f for f in d.iterdir() if f.suffix == ".apkg"]
    if not decks:
        return f"no .apkg files in {d}. The pipeline ships no source content; bring your own (README: Bring your own source corpus)."
    matched = [f for f in decks if REGISTRY.role_for(f.name) is not None]
    if not matched:
        return (f"{len(decks)} .apkg files in {d} but none matches a deck profile in etl/deck_profiles/. "
                f"Add a profile whose filename_globs match (etl/stages/EXTENDING.md).")
    return None


def _llm_check() -> str | None:
    from etl.llm import LLMSettings, reachable
    s = LLMSettings.from_env()
    why = reachable(s)
    if why is None:
        return None
    return (f"LLM endpoint {s.endpoint} does not answer ({why}). Set SH_LLM_BASE_URL (and SH_LLM_API=openai|anthropic, "
            f"SH_LLM_MODEL, SH_LLM_API_KEY) to an OpenAI-compatible server.")


def _db_check(db_path: Path, tables: tuple[str, ...]) -> Callable[[], str | None]:
    def check():
        import sqlite3
        if not db_path.exists():
            return f"database {db_path} does not exist; run the earlier stages first."
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            have = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
            for t in tables:
                if t not in have:
                    return f"table {t} is missing from {db_path}; run the stage that produces it first."
                if conn.execute(f"select count(*) from {t}").fetchone()[0] == 0:
                    return f"table {t} in {db_path} is empty; run the stage that produces it first."
        finally:
            conn.close()
        return None
    return check


def requirements(db_path: Path, allow_missing_curated: bool = False, need_llm: bool = True) -> list[Requirement]:
    from etl.ontology import icd10, loinc, snomed
    from etl.stages import s06d_diagnosis_relations as s06d

    reqs: list[Requirement] = [
        Requirement(1, "source decks", _sources_check),
        Requirement(2, "stage-1 tables", _db_check(db_path, ("source_decks", "raw_cards"))),
        Requirement(3, "stage-2 tables", _db_check(db_path, ("raw_cards",))),
        Requirement(4, "stage-3 tables", _db_check(db_path, ("board_questions",))),
        Requirement(5, "board questions", _db_check(db_path, ("board_questions",))),
        Requirement(5, "ICD-10-CM table", _file(icd10.ICD10_FILE, "ICD-10-CM table",
                                                 "Build it with `python -m etl.stages.s05_ontology --download-icd10` (CMS FY2025 order file, public domain).")),
        Requirement(5, "SNOMED CT RF2 (concepts, descriptions, Extended Map)",
                    lambda: None if all(p.exists() for p in (snomed.CONCEPT_FILE, snomed.DESCRIPTION_FILE, snomed.EXTENDED_MAP_FILE))
                    else f"SNOMED CT US Edition RF2 files are missing under {snomed.SNOMED_DIR} (UMLS license). Without them stage 5 keeps the LLM's SNOMED ids.",
                    optional=True),
        Requirement(5, "LOINC table", _file(loinc.LOINC_FILE, "LOINC table",
                                             "Register at loinc.org and place the LOINC table at that path. Without it lab findings get no LOINC code."),
                    optional=True),
        Requirement(6, "stage-5 tables", _db_check(db_path, ("diagnoses", "clinical_findings"))),
        Requirement(7, "board questions", _db_check(db_path, ("board_questions",))),
        Requirement(8, "stage-6/7 tables", _db_check(db_path, ("question_diagnoses", "ehr_sections"))),
        Requirement(9, "stage-8 tables", _db_check(db_path, ("longitudinal_patients",))),
        Requirement(10, "stage-9 tables", _db_check(db_path, ("longitudinal_encounters", "encounter_ehr_sections"))),
        Requirement(11, "stage-6 tables", _db_check(db_path, ("diagnoses", "diagnosis_findings"))),
        Requirement(11, "clinician-graded finding sites", _file(s06d.FINDING_SITE_GRADES, "finding_site_review.csv",
                                                                "Stage 11 uses the clinician grading to decide which shared finding sites count; "
                                                                "pass --allow-missing-curated to fall back to the count guard (changes the labels)."),
                    optional=allow_missing_curated),
        Requirement(11, "clinician-curated diagnosis edges", _file(s06d.CURATED_EDGES, "curated_diagnosis_edges.csv",
                                                                    "Stage 11 adds these comorbidity edges; pass --allow-missing-curated to build without them (changes the labels)."),
                    optional=allow_missing_curated),
        Requirement(12, "stage-11 table", _db_check(db_path, ("diagnosis_relations",))),
    ]
    if need_llm:
        for st in sorted(LLM_STAGES):
            reqs.append(Requirement(st, "LLM endpoint", _llm_check))
    return reqs


def problems(stages: list[int], db_path: Path, allow_missing_curated: bool = False, need_llm: bool = True) -> tuple[list[str], list[str]]:
    """(failures, notes) for the requested stages. Each entry: 'stage N — <what>: <problem>'."""
    fails, notes = [], []
    seen: set[tuple[int, str]] = set()
    for r in requirements(db_path, allow_missing_curated, need_llm):
        if r.stage not in stages or (r.stage, r.what) in seen:
            continue
        seen.add((r.stage, r.what))
        why = r.check()
        if why:
            (notes if r.optional else fails).append(f"stage {r.stage} — {r.what}: {why}")
    return fails, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", type=int, action="append", help="stage to check (repeatable)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--allow-missing-curated", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="skip the endpoint reachability check")
    a = ap.parse_args(argv)
    stages = list(range(1, 13)) if a.all or not a.stage else a.stage
    fails, notes = problems(stages, Path(a.db), a.allow_missing_curated, need_llm=not a.no_llm)
    for n in notes:
        print(f"note: {n}")
    for f in fails:
        print(f"MISSING: {f}")
    print(f"preflight: {len(fails)} missing, {len(notes)} optional inputs absent, stages {stages}")
    return 2 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
