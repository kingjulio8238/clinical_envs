"""End-to-end smoke of the wired pipeline on synthetic questions and a mock LLM (Stage 6).

No source content, no model, no network beyond localhost: three synthetic board questions go through
stage 5 (5a extraction with a forced validation failure, 5b ICD-10 mapping) and stage 7 (segmentation)
via `python -m etl.main`, and the run leaves a manifest and a fully attributed `llm_call_log`.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ORDER_TXT = ROOT / "data" / "ontology" / "icd10cm_order_2025.txt"

VIGNETTES = [
    "A 23-year-old man presents with 18 hours of periumbilical pain that migrated to the right lower quadrant, "
    "anorexia and fever of 38.5°C. Examination shows guarding at McBurney's point. Leukocyte count is 14,000/mm³.",
    "A 61-year-old woman with type 2 diabetes presents with two days of right lower quadrant pain and nausea. "
    "Temperature is 38.2°C. There is rebound tenderness. Serum creatinine is 1.1 mg/dL.",
    "A 9-year-old boy is brought in with one day of abdominal pain that began around the umbilicus and now localizes "
    "to the right lower quadrant. He has vomited twice. Temperature is 38.4°C.",
]


def _mock_module():
    spec = importlib.util.spec_from_file_location("mock_llm", ROOT / "scripts" / "mock_llm_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed_db(path: Path) -> None:
    from etl.db import create_schema, get_connection
    conn = get_connection(path)
    create_schema(conn)
    conn.execute("INSERT INTO source_decks (deck_id, filename, deck_name, deck_type) VALUES (1, 'synthetic.apkg', 'synthetic', 'board_exam')")
    for i, v in enumerate(VIGNETTES, start=1):
        conn.execute("INSERT INTO raw_cards (raw_card_id, deck_id, anki_note_id, field_data, card_type, card_format, classification_method) "
                     "VALUES (?, 1, ?, '{}', 'board_exam', 'mcq_vignette', 'rule_based')", (i, 1000 + i))
        conn.execute("INSERT INTO board_questions (question_id, raw_card_id, question_format, vignette_text, question_stem, answer_choices, "
                     "correct_answer, correct_explanation, extraction_method) VALUES (?, ?, 'mcq_vignette', ?, 'Which is the most likely diagnosis?', "
                     "?, 'A', 'Migration of pain to the RLQ with fever suggests appendicitis.', 'rule_based')",
                     (i, i, v, json.dumps([{"letter": "A", "text": "Acute appendicitis"}, {"letter": "B", "text": "Mesenteric adenitis"}])))
    conn.commit()
    conn.close()


@pytest.fixture(scope="module")
def icd_csv():
    from etl.ontology.icd10 import ICD10_FILE
    if ICD10_FILE.exists():
        return ICD10_FILE
    if not ORDER_TXT.exists():
        pytest.skip("neither data/ontology/icd10cm_2025.csv nor the CMS order file is present")
    from etl.stages.s05_ontology import download_icd10
    return download_icd10(order_file=ORDER_TXT)


def _purge_stubbed_stage_modules():
    """eval/tests/test_current_visit.py installs a stub etl.stages.s05_ontology; this smoke needs the real one."""
    import sys
    real = sys.modules.get("etl.stages.s05_ontology")
    if real is not None and not hasattr(real, "extract_all"):
        for name in [m for m in sys.modules if m.startswith("etl.stages.") or m in ("etl.main",)]:
            del sys.modules[name]


def test_stage_5_and_7_run_end_to_end_on_the_mock(tmp_path, monkeypatch, icd_csv):
    _purge_stubbed_stage_modules()
    mock = _mock_module()
    server, state = mock.serve_in_thread(port=0, fail_first=1)          # the very first reply is structurally invalid
    port = server.server_address[1]
    monkeypatch.setenv("SH_LLM_API", "openai")
    monkeypatch.setenv("SH_LLM_BASE_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("SH_LLM_MODEL", "mock-model")
    monkeypatch.setenv("SH_LLM_SEED", "123")
    monkeypatch.setenv("SH_LLM_RETRY_DELAY", "0")
    monkeypatch.delenv("SH_LLM_GATEWAY_URL", raising=False)
    from etl import llm
    llm.set_client(None)                                                   # re-read the environment
    try:
        from etl import main as etl_main
        db = tmp_path / "synthetic.db"
        _seed_db(db)

        rc = etl_main.main(["--stage", "5", "--db", str(db), "--pilot", "3", "--workers", "2"])
        assert rc == 0
        conn = sqlite3.connect(db)
        rows = conn.execute("select stage, model, temperature, seed, attempts, endpoint, request_id, error, input_hash "
                            "from llm_call_log where stage='s05a_extract' order by call_id").fetchall()
        assert len(rows) == 3 and all(r[7] is None for r in rows), rows
        assert {r[1] for r in rows} == {"mock-model"} and {r[2] for r in rows} == {0.0} and {r[3] for r in rows} == {123}
        assert {r[5] for r in rows} == {f"http://127.0.0.1:{port}/v1/chat/completions"}
        assert sorted(r[4] for r in rows) == [1, 1, 2], "one extraction had to be retried after the forced invalid reply"
        assert all(r[6] and r[6].startswith("mock-") for r in rows), "the real request id is stored"
        assert len({r[8] for r in rows}) == 3
        assert state.requests == 4                                         # 3 questions + 1 retry
        assert conn.execute("select count(*) from diagnoses").fetchone()[0] == 2          # appendicitis + the distractor, merged across questions
        assert conn.execute("select count(*) from clinical_findings").fetchone()[0] == 3
        codes = {r[0] for r in conn.execute("select icd10_code from diagnoses")}
        assert "K35.80" in codes, codes                                   # 5b validated the suggested code against the CMS table

        # rerun: every call is a cache hit, no new request reaches the server
        before = state.requests
        assert etl_main.main(["--stage", "5", "--db", str(db), "--pilot", "3", "--workers", "2"]) == 0
        assert state.requests == before
        assert conn.execute("select count(*) from llm_call_log where stage='s05a_extract'").fetchone()[0] == 3

        # stage 7 through the same client
        assert etl_main.main(["--stage", "7", "--db", str(db), "--pilot", "3", "--workers", "2"]) == 0
        assert conn.execute("select count(*) from ehr_sections").fetchone()[0] >= 9
        s7 = conn.execute("select attempts, seed, endpoint from llm_call_log where stage like 's07%'").fetchall()
        assert s7 and all(r[1] == 123 for r in s7)

        manifest = json.loads((db.parent / "pipeline_manifest.json").read_text())
        stages = [r["stage"] for r in manifest["runs"]]
        assert stages == [5, 5, 7]
        llm_fp = manifest["runs"][0]["llm"]
        assert llm_fp["model"] == "mock-model" and llm_fp["seed"] == 123 and llm_fp["seed_applied"] is True and "api_key" not in llm_fp
        assert manifest["runs"][0]["inputs_sha256"]["icd10cm"]
        conn.close()
    finally:
        server.shutdown()
        llm.set_client(None)
