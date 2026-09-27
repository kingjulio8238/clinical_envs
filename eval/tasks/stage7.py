"""Single-turn loaders, prompt formatters and parsers for the Stage-7 task families.

The chart is the one the environment serves: up to and including the index encounter, without
assessment/plan, with the instance's section overrides applied (error_detection, atypical_diagnosis) and
the index visit's result sections removed (test_selection). A single-turn model cannot order tests, so
for test_selection it names the tests it would order in `tests_ordered`; they are matched against the
encounter's documented findings by the scorer.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from eval import private_labels
from eval import stage7 as S7
from eval.agents.prompts import TASK_GOALS
from eval.tasks.diagnosis import _strip_markdown

log = logging.getLogger(__name__)

TASKS = S7.NEW_TASKS


@dataclass
class Stage7Input:
    gt_id: int
    patient_id: int
    encounter_id: int
    task: str
    ehr_text: str
    ehr_token_estimate: int
    ground_truth: dict
    metadata: dict = field(default_factory=dict)


def _chart(cur, patient_id: int, encounter_id: int, task: str, gt: dict) -> str:
    cur.execute("SELECT encounter_order, encounter_date, encounter_type FROM longitudinal_encounters WHERE encounter_id = %s", (encounter_id,))
    row = cur.fetchone()
    if row is None:
        return ""
    order, date, etype = row
    cur.execute("""
        SELECT le.encounter_id, le.encounter_date, le.encounter_type, le.chief_complaint, ees.id, ees.section_type, ees.section_text
        FROM encounter_ehr_sections ees JOIN longitudinal_encounters le ON ees.encounter_id = le.encounter_id
        WHERE le.patient_id = %s AND le.encounter_order <= %s AND ees.section_type::text NOT IN ('assessment', 'plan')
        ORDER BY le.encounter_order, ees.section_order
    """, (patient_id, order))
    hidden = set(gt.get("hidden_section_types") or []) if task == "test_selection" else set()
    overrides = gt.get("section_overrides") or {}
    parts, current = [], None
    for eid, enc_date, enc_type, cc, sid, st, text in cur.fetchall():
        st = getattr(st, "value", st)
        if eid == encounter_id and st in hidden:
            continue
        key = f"{enc_date}|{enc_type}"
        if key != current:
            current = key
            parts.append(f"\n{'=' * 60}\nENCOUNTER: {enc_date} ({enc_type}){' — ' + cc if cc else ''}\n{'=' * 60}")
        if str(sid) in overrides:
            text = S7.apply_override(text, overrides[str(sid)])
        parts.append(f"[{str(st).upper().replace('_', ' ')}]\n{text}")
    head = f"INDEX ENCOUNTER: {date} ({etype}), encounter_id {encounter_id}\n"
    return head + "\n\n".join(parts)


def load_inputs(conn, split: str = "public", granularity: str | None = None, pilot: int | None = None,
                task: str = "differential_diagnosis") -> list[Stage7Input]:
    cur = conn.cursor()
    sql = ("SELECT gt_id, patient_id, encounter_id, ground_truth FROM benchmark_ground_truth "
           "WHERE task::text = %s AND is_diagnostic AND split::text = %s ORDER BY gt_id")
    params: list = [task, split]
    if pilot:
        sql += " LIMIT %s"
        params.append(pilot)
    cur.execute(sql, params)
    rows = cur.fetchall()
    out = []
    overlay = private_labels.get()
    for gt_id, pid, eid, gt in rows:
        gt = gt if isinstance(gt, dict) else json.loads(gt)
        if gt.get(private_labels.REMOVED_FLAG):
            full = overlay.ground_truth(gt_id) if overlay else None
            if full is None:
                continue
            gt = full
        text = _chart(cur, pid, eid, task, gt)
        if not text:
            continue
        out.append(Stage7Input(gt_id=gt_id, patient_id=pid, encounter_id=eid, task=task, ehr_text=text,
                               ehr_token_estimate=len(text) // 4, ground_truth=gt))
    log.info("Loaded %d %s inputs (split=%s)", len(out), task, split)
    return out


def format_prompt(inp: Stage7Input, strategy: str = "zero_shot") -> tuple[str, str]:
    """(system, user). Strategies share one prompt: the environment's task goal and schema."""
    info = TASK_GOALS[inp.task]
    goal = info["goal"].format(encounter_id=inp.encounter_id)
    extra = ("\nYou cannot order tests in this setting: list, under \"tests_ordered\", the tests you would order "
             "(names or panels), then give the diagnosis they would establish.") if inp.task == "test_selection" else ""
    schema = info["schema"] if inp.task != "test_selection" else '{"icd10": "K85.90", "name": "Acute pancreatitis", "tests_ordered": ["lipase", "CT abdomen"]}'
    system = (f"You are a physician {info['description']}.\n{goal}{extra}\n"
              f"Respond with JSON only, in this shape:\n{schema}")
    return system, inp.ehr_text


def parse_output(raw_text: str) -> dict:
    text = _strip_markdown(raw_text or "")
    for candidate in (text, (re.search(r"\{.*\}", text, re.DOTALL) or re.search(r"\[.*\]", text, re.DOTALL) or [None])):
        c = candidate if isinstance(candidate, str) else (candidate.group() if candidate else None)
        if not c:
            continue
        try:
            data = json.loads(c, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return {**data, "_parse_tier": "A"}
        if isinstance(data, list):
            return {"differential": data, "_parse_tier": "A"}
    return {"_parse_tier": "C"}


def make_loader(task: str):
    def _load(conn, split: str = "public", granularity: str | None = None, pilot: int | None = None):
        return load_inputs(conn, split, granularity, pilot, task=task)
    return _load
