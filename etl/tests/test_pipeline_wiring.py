"""etl.main wires every stage; etl.preflight names what is missing; stage 11 fails loudly (Stage 6)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from etl import main as etl_main
from etl import preflight


def test_every_stage_1_to_12_resolves_to_a_callable():
    assert sorted(etl_main.STAGES) == list(range(1, 13))
    for n, (name, fn) in etl_main.STAGES.items():
        assert callable(fn) and name


def test_selected_stages_from_flags():
    a = etl_main.parse_args(["--from", "5", "--to", "7", "--skip-preflight"])
    assert etl_main.selected_stages(a) == [5, 6, 7]
    a = etl_main.parse_args(["--all"])
    assert etl_main.selected_stages(a) == list(range(1, 13))
    a = etl_main.parse_args(["--stage", "11", "--allow-missing-curated"])
    assert etl_main.selected_stages(a) == [11] and a.allow_missing_curated


def test_preflight_lists_missing_inputs_with_fixes(tmp_path, monkeypatch):
    monkeypatch.setenv("SH_APKG_DIR", str(tmp_path / "no_such_dir"))
    fails, notes = preflight.problems([1, 5, 11], tmp_path / "none.db", need_llm=False)
    text = "\n".join(fails)
    assert "stage 1 — source decks" in text and "SH_APKG_DIR" in text
    assert "stage 5 — board questions" in text and "does not exist" in text
    assert "finding_site_review.csv" in text and "--allow-missing-curated" in text
    # the curated files become notes, not failures, when the fallback is explicitly allowed
    fails2, notes2 = preflight.problems([11], tmp_path / "none.db", allow_missing_curated=True, need_llm=False)
    assert not any("finding_site_review" in f for f in fails2) and any("finding_site_review" in n for n in notes2)


def test_preflight_passes_when_inputs_exist(tmp_path, monkeypatch):
    src = tmp_path / "apkg"; src.mkdir()
    (src / "board_exam_deck_a.apkg").write_bytes(b"x")          # matches a shipped deck profile
    monkeypatch.setenv("SH_APKG_DIR", str(src))
    db = tmp_path / "b.db"
    conn = sqlite3.connect(db)
    conn.execute("create table board_questions (question_id integer)"); conn.execute("insert into board_questions values (1)"); conn.commit(); conn.close()
    fails, _ = preflight.problems([1, 7], db, need_llm=False)
    assert fails == []


def test_main_exits_2_with_the_missing_list(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SH_APKG_DIR", str(tmp_path / "empty"))
    rc = etl_main.main(["--stage", "1", "--db", str(tmp_path / "x.db"), "--dry-run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "MISSING: stage 1" in err and "preflight failed" in err


def test_llm_endpoint_unreachable_is_a_failure_for_llm_stages(tmp_path, monkeypatch):
    monkeypatch.setenv("SH_LLM_BASE_URL", "http://127.0.0.1:9")      # nothing listens on port 9
    monkeypatch.delenv("SH_LLM_GATEWAY_URL", raising=False)
    db = tmp_path / "b.db"
    conn = sqlite3.connect(db)
    conn.execute("create table board_questions (question_id integer)"); conn.execute("insert into board_questions values (1)"); conn.commit(); conn.close()
    fails, _ = preflight.problems([7], db, need_llm=True)
    assert any("LLM endpoint" in f and "SH_LLM_BASE_URL" in f for f in fails)


def test_stage_11_raises_without_curated_files(tmp_path):
    from etl.stages import s06d_diagnosis_relations as s06d
    with pytest.raises(s06d.CuratedInputMissing):
        s06d._load_finding_site_allowlist(tmp_path / "missing.csv")
    with pytest.raises(s06d.CuratedInputMissing):
        s06d._add_curated_edges({}, [], tmp_path / "missing.csv")
    assert s06d._load_finding_site_allowlist(tmp_path / "missing.csv", allow_missing=True) is None
    s06d._add_curated_edges({}, [], tmp_path / "missing.csv", allow_missing=True)      # no raise


def test_source_fingerprints_roundtrip(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("fp", Path(__file__).resolve().parents[2] / "scripts" / "source_fingerprints.py")
    fp = importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
    db = tmp_path / "s.db"
    conn = sqlite3.connect(db)
    conn.execute("create table board_questions (question_id integer, vignette_text text)")
    v = "A 45-year-old man presents with three days of right lower quadrant pain, fever, and anorexia. Examination shows guarding."
    conn.executemany("insert into board_questions values (?, ?)", [(1, v), (2, "Short one."), (3, None)])
    conn.commit(); conn.close()
    out = tmp_path / "fp.json"
    assert fp.write(db, out)["items"] == 2
    data = __import__("json").loads(out.read_text())
    assert "quadrant" not in out.read_text().lower() and "anorexia" not in out.read_text().lower()   # no text leaks
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join(__import__("json").dumps(x) for x in [
        {"id": "exact", "text": "<p>A 45-year-old man presents with three days of right lower quadrant pain, fever, and anorexia. Examination shows guarding.</p>"},
        {"id": "near", "text": "Case: a 45 year old man presents with three days of right lower quadrant pain fever and anorexia; examination shows guarding and rebound."},
        {"id": "other", "text": "The mitochondrion is the powerhouse of the cell and has its own genome."},
    ]))
    r = fp.check(corpus, out, threshold=0.5)
    assert [h["doc"] for h in r["exact_matches"]] == ["exact"]
    assert [h["doc"] for h in r["near_matches"]] == ["near"] and r["documents"] == 3
