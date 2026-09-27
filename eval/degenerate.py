"""Degenerate (zero-model) policies and label oracles for the reward-hacking regression suite.

Every policy here produces a submission without clinical judgement: it reads nothing, or reads the
chart only in a mechanical way (copy a list, paste the text, rank by section type). Their rewards
are the *floors* that any reported score must be read against (eval/floors.py) and the *gates* that
eval/tests/test_reward_hacking.py enforces. The oracles are built from the labels exactly as the
simulator's `/env/oracle` endpoint builds them (epic_sim/app/services/env_service.py:
oracle_submission) and give the *ceilings*.

Everything runs in-process against the released SQLite database: no Postgres, no simulator, no LLM.
Scoring goes through the same function the paper, `/score` and `/env` use
(eval.scoring.compute_all_metrics), with the same per-instance context that
eval.score_one._attach_context adds.
"""

from __future__ import annotations

import asyncio
import collections
import importlib.util
import json
import random
import re
import sqlite3
from pathlib import Path
from typing import Callable

from eval import private_labels
from eval.scoring import compute_all_metrics

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "benchmark_v1.3.db"

TASKS = ("patient_diagnosis", "evidence_retrieval", "context_summarization",
         "specialty_involved", "specialty_absent", "imaging_indication")
"""Scoring units. The two specialty units are the `context_summarization` rows whose ground truth
carries `variant = specialty_conditioned`, split by `involvement`, because the scorer rewards them
with different metrics (conditioned_f1 vs abstention_accuracy)."""

PRIMARY_METRIC = {
    "patient_diagnosis": "weighted_problem_list_f1_neutral",
    "evidence_retrieval": "ndcg_10",
    "context_summarization": "clinical_f1",
    "specialty_involved": "conditioned_f1",
    "specialty_absent": "abstention_accuracy",
    "imaging_indication": "clinical_question_concept_f1",
}

ABSTAIN_PHRASE = "No significant distress."
"""One of the stock phrases eval.scoring._is_abstention treats as abstaining."""

INVENTED_SENTENCES = (
    "The patient has a history of metastatic pancreatic adenocarcinoma and was started on FOLFIRINOX. "
    "He was admitted for a pulmonary embolism and was treated with apixaban. "
    "His course was complicated by a left hip fracture after a fall at home."
)
"""Three sentences that appear in no chart; the hallucination metric should flag all of them."""

NEGATED_FINDINGS = {"summary": "Patient denies fever. No ketonuria.",
                    "must_include": ["Fever", "Ketonuria"]}


# ---------------------------------------------------------------------------
# Release database
# ---------------------------------------------------------------------------

class _CursorShim:
    """sqlite cursor that tolerates the absent `terminology_codes` table (chart_neutral_sets.py
    queries it; the release ships it empty)."""

    def __init__(self, conn: sqlite3.Connection):
        self.c = conn.cursor()

    def execute(self, sql, *args):
        try:
            self.c.execute(sql, *args)
        except sqlite3.OperationalError:
            self.c.execute("select 1 where 0")

    def fetchall(self):
        return self.c.fetchall()


class Instance(dict):
    """One benchmark_ground_truth row: gt_id, task, patient_id, encounter_id, split, gt (dict)."""


class ReleaseDB:
    """Read-only view of benchmark_v1.3.db with the per-patient material the policies need."""

    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self._cache: dict = {}

    def q(self, sql: str, *args):
        return self.conn.execute(sql, args).fetchall()

    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # -- instances -----------------------------------------------------------
    def instances(self, task: str, split: str) -> list[Instance]:
        """Instances of a scoring unit (see TASKS) in a split, ordered by gt_id."""
        def load():
            base_task = "context_summarization" if task.startswith("specialty") else task
            rows = self.q(
                "select b.gt_id, coalesce(b.patient_id, l.patient_id), b.encounter_id, b.split, b.ground_truth "
                "from benchmark_ground_truth b left join longitudinal_encounters l using(encounter_id) "
                "where b.task=? and b.split=? and b.is_diagnostic order by b.gt_id", base_task, split)
            out = []
            overlay = private_labels.get()
            for gt_id, pid, eid, sp, gt in rows:
                gt = json.loads(gt)
                if gt.get(private_labels.REMOVED_FLAG):
                    # private split: labels live in the operator's overlay; skip the unit if absent
                    full = overlay.ground_truth(gt_id) if overlay else None
                    if full is None:
                        continue
                    gt = full
                variant = gt.get("variant")
                if task == "context_summarization" and variant is not None:
                    continue
                if task == "specialty_involved" and not (variant == "specialty_conditioned" and gt.get("involvement") != "absent"):
                    continue
                if task == "specialty_absent" and not (variant == "specialty_conditioned" and gt.get("involvement") == "absent"):
                    continue
                out.append(Instance(gt_id=gt_id, task=task, patient_id=pid, encounter_id=eid, split=sp, gt=gt))
            return out
        return self._memo(("inst", task, split), load)

    # -- chart material --------------------------------------------------------
    def sections(self, patient_id: int) -> list[tuple[str, str, str]]:
        """(passage_id, section_type, text) for the patient's sections in chart order, excluding
        assessment and plan (the retrieval corpus and the single-turn input both exclude them)."""
        def load_all():
            d = collections.defaultdict(list)
            for pid, sid, st, txt in self.q(
                    "select l.patient_id, s.id, s.section_type, s.section_text from encounter_ehr_sections s "
                    "join longitudinal_encounters l using(encounter_id) "
                    "where s.section_type not in ('assessment','plan') "
                    "order by l.patient_id, l.encounter_order, s.section_order"):
                d[pid].append((f"ees_{sid}", st, txt or ""))
            return d
        return self._memo("sections", load_all).get(patient_id, [])

    def chart_text(self, patient_id: int, upto_encounter_id: int | None = None) -> str:
        """The chart as a policy may read it; for an index-encounter instance, up to that encounter."""
        secs = self.sections(patient_id)
        if upto_encounter_id is not None:
            allowed = self.encounters_upto(patient_id, upto_encounter_id)
            secs = [s for s in secs if self._section_encounter(s[0]) in allowed]
        return "\n".join(f"[{st.upper()}]\n{txt}" for _, st, txt in secs)

    def encounters_upto(self, patient_id: int, encounter_id: int) -> set[int]:
        orders = self._memo("enc_orders", lambda: {e: (p, o) for e, p, o in self.q(
            "select encounter_id, patient_id, encounter_order from longitudinal_encounters")})
        _, target = orders[encounter_id]
        return {e for e, (p, o) in orders.items() if p == patient_id and o <= target}

    def _section_encounter(self, passage_id: str) -> int:
        return self._memo("sec_enc", lambda: {f"ees_{i}": e for i, e in self.q(
            "select id, encounter_id from encounter_ehr_sections")})[passage_id]

    def section_type(self, passage_id: str) -> str:
        return self._memo("stype", lambda: {f"ees_{i}": t for i, t in self.q(
            "select id, section_type from encounter_ehr_sections")}).get(passage_id, "")

    def profile(self, patient_id: int) -> dict:
        return self._memo("profiles", lambda: {p: json.loads(pr) for p, pr in self.q(
            "select patient_id, profile from longitudinal_patients")}).get(patient_id, {})

    def chief_complaint(self, encounter_id: int) -> str:
        return self._memo("cc", lambda: dict(self.q(
            "select encounter_id, chief_complaint from longitudinal_encounters"))).get(encounter_id) or ""

    def imaging_order(self, gt_id: int) -> dict:
        return self._memo("orders", lambda: {g: {"modality": m, "body_region": b, "indication": i} for g, m, b, i in self.q(
            "select gt_id, modality, body_region, clinical_indication from imaging_orders")}).get(gt_id, {})

    # -- graph material ----------------------------------------------------------
    def name_to_icd(self) -> dict[str, str]:
        def load():
            out = {}
            for code, name in self.q("select icd10_code, display_name from diagnoses where icd10_code is not null"):
                out.setdefault(name.lower(), code)
            return out
        return self._memo("name2icd", load)

    def secondary_codes(self, patient_id: int) -> set[str]:
        """ICD codes of the graph's `secondary` diagnoses of the patient's source questions: the
        comorbidities the vignettes document but the reference omits."""
        def load():
            q2 = collections.defaultdict(set)
            for qid, code in self.q("select qd.question_id, d.icd10_code from question_diagnoses qd "
                                    "join diagnoses d using(diagnosis_id) where qd.role='secondary' and d.icd10_code is not null"):
                q2[qid].add(code)
            out = collections.defaultdict(set)
            for pid, sq in self.q("select patient_id, source_question_ids from longitudinal_encounters"):
                for qid in json.loads(sq):
                    out[pid] |= q2[qid]
            return out
        return self._memo("secondary", load).get(patient_id, set())

    def key_finding_names(self, patient_id: int) -> list[str]:
        """Distinct key-finding names of the patient's source questions, sorted. This is the pool the
        "structured" summarization prompt draws its 10 hints from (eval/tasks/summarization.py:
        _load_key_findings, which has no ORDER BY)."""
        def load():
            qf = collections.defaultdict(set)
            for qid, name in self.q("select qf.question_id, cf.display_name from question_findings qf "
                                    "join clinical_findings cf using(finding_id) where qf.relevance='key'"):
                qf[qid].add(name)
            out = collections.defaultdict(set)
            for pid, sq in self.q("select patient_id, source_question_ids from longitudinal_encounters"):
                for qid in json.loads(sq):
                    out[pid] |= qf[qid]
            return {p: sorted(v) for p, v in out.items()}
        return self._memo("keyfindings", load).get(patient_id, [])

    def patient_terms(self, patient_id: int) -> str:
        """The patient's annotated finding and diagnosis names (grounding set for summarization)."""
        def load():
            qf = collections.defaultdict(set)
            for qid, name in self.q("select qf.question_id, cf.display_name from question_findings qf join clinical_findings cf using(finding_id)"):
                qf[qid].add(name)
            for qid, name in self.q("select qd.question_id, d.display_name from question_diagnoses qd join diagnoses d using(diagnosis_id) "
                                    "where qd.role in ('correct','secondary')"):
                qf[qid].add(name)
            out = collections.defaultdict(set)
            for pid, sq in self.q("select patient_id, source_question_ids from longitudinal_encounters"):
                for qid in json.loads(sq):
                    out[pid] |= qf[qid]
            return {p: ". ".join(sorted(v)) for p, v in out.items()}
        return self._memo("terms", load).get(patient_id, "")

    def neutral_categories(self, patient_id: int) -> list[str]:
        """The chart-neutral set the scorer uses (scripts/chart_neutral_sets.py), as score_one attaches it."""
        def load():
            spec = importlib.util.spec_from_file_location("chart_neutral_sets", ROOT / "scripts" / "chart_neutral_sets.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sets, _, _ = mod.neutral_sets(_CursorShim(self.conn))
            return {int(p): sorted(v) for p, v in sets.items()}
        return self._memo("neutral", load).get(patient_id, [])

    def judgments(self, gt_id: int) -> dict[str, int]:
        return self._memo("judg", lambda: self._load_judgments()).get(gt_id, {})

    def _load_judgments(self):
        d = collections.defaultdict(dict)
        for g, p, r in self.q("select gt_id, passage_id, relevance_grade from relevance_judgments"):
            d[g][p] = r
        overlay = private_labels.get()
        if overlay is not None:
            for g, p, r in overlay.conn.execute("select gt_id, passage_id, relevance_grade from relevance_judgments"):
                d[g][p] = r
        return d

    def section_type_prior(self, fit_split: str = "train") -> dict[str, float]:
        """Mean relevance grade per section type, fitted on `fit_split` so that floors on other
        splits are not fitted on themselves."""
        def load():
            acc = collections.defaultdict(list)
            for pid, g in self.q("select r.passage_id, r.relevance_grade from relevance_judgments r "
                                 "join benchmark_ground_truth b using(gt_id) where b.split=?", fit_split):
                acc[self.section_type(pid)].append(g)
            return {t: sum(v) / len(v) for t, v in acc.items()}
        return self._memo(("prior", fit_split), load)

    def most_absent_specialties(self, fit_split: str = "train", n: int = 2) -> list[str]:
        """Specialties most often `absent` in `fit_split`: the labels a name-only policy abstains on."""
        def load():
            rows = self.q("select json_extract(ground_truth,'$.specialty'), avg(json_extract(ground_truth,'$.involvement')='absent') "
                          "from benchmark_ground_truth where task='context_summarization' and split=? "
                          "and json_extract(ground_truth,'$.variant')='specialty_conditioned' group by 1 order by 2 desc", fit_split)
            return [s for s, _ in rows[:n]]
        return self._memo(("absent", fit_split, n), load)

    def concept_extractor(self):
        """The paper's imaging concept matcher, built from the graph like ConceptExtractor.from_db."""
        def load():
            from eval.imaging_concepts import ConceptExtractor
            inv = []
            for did, sn, dn, sd in self.q("select diagnosis_id, snomed_id, display_name, snomed_desc from diagnoses"):
                cid = f"S{sn}" if sn else f"D{did}"
                inv += [(cid, dn), (cid, sd)]
            for fid, sn, dn, sd in self.q("select finding_id, snomed_id, display_name, snomed_desc from clinical_findings "
                                          "where finding_type <> 'demographic'"):
                cid = f"S{sn}" if sn else f"F{fid}"
                inv += [(cid, dn), (cid, sd)]
            return ConceptExtractor(inv)
        return self._memo("extractor", load)

    # -- the simulator's problem-list tool -----------------------------------------
    def problem_list_tool(self, patient_ids: list[int]) -> dict[int, list[dict]]:
        """What `view_problem_list` / `open_chart.active_problems` return, by calling the real
        service (epic_sim.app.services.epic_service.get_problem_list) over this SQLite file."""
        async def run():
            import warnings
            from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
            from epic_sim.app.services import epic_service
            engine = create_async_engine(f"sqlite+aiosqlite:///file:{self.path}?mode=ro&uri=true")
            out = {}
            try:
                async with async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)() as db:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # DISTINCT ON is PostgreSQL-only; SQLite renders DISTINCT
                        for pid in patient_ids:
                            rows = await epic_service.get_problem_list(db, pid)
                            out[pid] = [{"icd10": r.icd10_code or "", "name": r.display_name} for r in rows]
            finally:
                await engine.dispose()
            return out
        return asyncio.run(run())


# ---------------------------------------------------------------------------
# Scoring (mirrors eval.score_one)
# ---------------------------------------------------------------------------

def attach_context(db: ReleaseDB, inst: Instance) -> dict:
    """The ground truth plus the per-instance context score_one._attach_context adds."""
    gt = dict(inst["gt"])
    if inst["task"] == "patient_diagnosis":
        gt["_neutral_categories"] = sorted(set(db.neutral_categories(inst["patient_id"])) | set(gt.get("neutral_extra") or []))
    elif inst["task"] in ("context_summarization", "specialty_involved", "specialty_absent"):
        gt["_chart_text"] = db.chart_text(inst["patient_id"])
        gt["_patient_terms"] = db.patient_terms(inst["patient_id"])
    elif inst["task"] == "evidence_retrieval":
        gt["_judgments"] = db.judgments(inst["gt_id"])
    return gt


def score(db: ReleaseDB, task: str, predictions: list[dict], insts: list[Instance]) -> dict:
    """compute_all_metrics on a batch, with context attached. Returns the full metric dict."""
    base = "context_summarization" if task.startswith("specialty") else task
    kwargs = {}
    if task in ("imaging_indication", "context_summarization", "specialty_involved", "specialty_absent"):
        kwargs["concept_extractor"] = db.concept_extractor()
    return compute_all_metrics(base, predictions, [attach_context(db, i) for i in insts], **kwargs)


def primary(db: ReleaseDB, task: str, predictions: list[dict], insts: list[Instance]) -> float:
    return float(score(db, task, predictions, insts)[PRIMARY_METRIC[task]])


def per_item(db: ReleaseDB, task: str, predictions: list[dict], insts: list[Instance]) -> list[float]:
    """The single-item reward for each instance, i.e. what /score and /env return per episode."""
    return [primary(db, task, [p], [i]) for p, i in zip(predictions, insts)]


# ---------------------------------------------------------------------------
# Oracles (ceilings): built from the labels exactly like env_service.oracle_submission
# ---------------------------------------------------------------------------

def oracle(db: ReleaseDB, inst: Instance) -> dict:
    gt, task = inst["gt"], inst["task"]
    if task == "patient_diagnosis":
        return {
            "active_diagnoses": [{"icd10": d["icd10"], "name": d.get("display_name", ""), "acuity": d.get("acuity") or "acute"}
                                 for d in gt.get("active_diagnoses", []) if not d.get("excluded_nondiagnostic")],
            "chronic_conditions": [{"icd10": d["icd10"], "name": d.get("display_name", ""), "acuity": "chronic"}
                                   for d in gt.get("chronic_conditions", []) if not d.get("excluded_nondiagnostic")],
        }
    if task in ("specialty_involved", "specialty_absent"):
        tiers = gt.get("tiers", {})
        names = [f.get("display_name") for f in tiers.get("primary", []) + tiers.get("relevant", [])]
        if not names:
            return {"summary": "", "abstain": True}
        return {"summary": ". ".join(n for n in names if n) + ".", "abstain": False}
    if task == "context_summarization":
        names = [f.get("display_name") or f.get("name") for f in gt.get("must_include_findings", [])]
        return {"summary": ". ".join(n for n in names if n) + "."}
    if task == "evidence_retrieval":
        j = db.judgments(inst["gt_id"])
        ranked = sorted(j, key=lambda p: (-j[p], p))[:20]
        return {"rankings": [{"passage_id": p, "grade": 3} for p in ranked]}
    if task == "imaging_indication":
        from eval.scoring import reference_terms_text
        return {
            "clinical_question": reference_terms_text(gt) or gt.get("inferred_clinical_question", ""),
            "pre_read_summary": gt.get("pre_read_summary", ""),
            "must_include_findings": list(gt.get("must_include_findings", [])),
            "differential": [{"diagnosis": d.get("diagnosis", ""), "icd10": d.get("icd10", "")}
                             for d in gt.get("differential_context", [])],
        }
    raise ValueError(task)


# ---------------------------------------------------------------------------
# Degenerate policies. Signature: policy(db, inst) -> prediction dict.
# ---------------------------------------------------------------------------

def empty(db: ReleaseDB, inst: Instance) -> dict:
    """A malformed / missing submission."""
    return {}


# patient diagnosis -----------------------------------------------------------

def dx_profile_chronic(db: ReleaseDB, inst: Instance) -> dict:
    """List the profile's chronic conditions, mapped to ICD categories by the scorer's own
    chart-neutral mapper. Reads no encounter."""
    spec = importlib.util.spec_from_file_location("chart_neutral_sets", ROOT / "scripts" / "chart_neutral_sets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    db.neutral_categories(inst["patient_id"])  # ensures the mapper's ontology is loaded
    chron = []
    for s in db.profile(inst["patient_id"]).get("chronic_conditions") or []:
        cats, _ = mod.map_condition(str(s))
        if cats:
            chron.append({"icd10": sorted(cats)[0]})
    return {"active_diagnoses": [], "chronic_conditions": chron}


_PROBLEM_LINE = re.compile(r"^- (.+?) \(diagnosed \d{4}-\d\d-\d\d\)", re.M)


def dx_copy_problem_list(db: ReleaseDB, inst: Instance) -> dict:
    """Regex-copy the `<name> (diagnosed YYYY-MM-DD)` lines from the chart's problem lists and map
    each name to a code with the benchmark's own diagnosis table."""
    names = set(_PROBLEM_LINE.findall(db.chart_text(inst["patient_id"], inst.get("encounter_id"))))
    n2c = db.name_to_icd()
    return {"active_diagnoses": [{"icd10": n2c[n.lower()], "acuity": "acute"} for n in sorted(names) if n.lower() in n2c],
            "chronic_conditions": []}


def dx_copy_plus_chronic(db: ReleaseDB, inst: Instance) -> dict:
    a, b = dx_copy_problem_list(db, inst), dx_profile_chronic(db, inst)
    return {"active_diagnoses": a["active_diagnoses"], "chronic_conditions": b["chronic_conditions"]}


def dx_echo_problem_list_tool(db: ReleaseDB, inst: Instance) -> dict:
    """Submit whatever the simulator's `view_problem_list` tool returns for the patient."""
    tool = db._cache.setdefault("tool_pl", {})
    if inst["patient_id"] not in tool:
        tool.update(db.problem_list_tool([inst["patient_id"]]))
    return {"active_diagnoses": [{"icd10": r["icd10"]} for r in tool[inst["patient_id"]] if r["icd10"]],
            "chronic_conditions": []}


def dx_reference_categories(db: ReleaseDB, inst: Instance) -> dict:
    """The reference codes truncated to their 3-character category (probe: does a bare category
    earn full credit?)."""
    gt = inst["gt"]
    codes = [d["icd10"] for d in gt["active_diagnoses"] + gt["chronic_conditions"] if d.get("icd10")]
    return {"active_diagnoses": [{"icd10": c[:3]} for c in codes], "chronic_conditions": []}


def dx_reference_plus_secondary(db: ReleaseDB, inst: Instance) -> dict:
    """The oracle plus the comorbidities the source vignettes document (graph `secondary`
    diagnoses). Probe: is documented truth penalized?"""
    o = oracle(db, inst)
    have = {d["icd10"][:3] for d in o["active_diagnoses"] + o["chronic_conditions"] if d.get("icd10")}
    extra = sorted(c for c in db.secondary_codes(inst["patient_id"]) if c[:3] not in have)
    return {"active_diagnoses": o["active_diagnoses"] + [{"icd10": c} for c in extra],
            "chronic_conditions": o["chronic_conditions"]}


# evidence retrieval ---------------------------------------------------------------

def retr_single_hpi(db: ReleaseDB, inst: Instance) -> dict:
    """Submit exactly one passage: the first HPI section (any section if none)."""
    j = db.judgments(inst["gt_id"])
    hpi = [p for p in j if db.section_type(p) == "hpi"] or list(j)
    return {"rankings": [{"passage_id": hpi[0]}]} if hpi else {"rankings": []}


def retr_type_prior(db: ReleaseDB, inst: Instance) -> dict:
    """Rank every passage by the mean grade of its section *type* on the train split. Reads no text."""
    prior = db.section_type_prior("train")
    j = db.judgments(inst["gt_id"])
    ranked = sorted(j, key=lambda p: (-prior.get(db.section_type(p), 0.0), p))
    return {"rankings": [{"passage_id": p} for p in ranked]}


def retr_random(db: ReleaseDB, inst: Instance, seed: int = 0) -> dict:
    """A seeded random permutation of the passages."""
    ids = sorted(db.judgments(inst["gt_id"]))
    random.Random(seed * 1_000_003 + inst["gt_id"]).shuffle(ids)
    return {"rankings": [{"passage_id": p} for p in ids]}


# summarization (whole patient) -------------------------------------------------------

def summ_chart_dump(db: ReleaseDB, inst: Instance) -> dict:
    """Paste the whole chart (minus assessment/plan) as the summary."""
    return {"summary": db.chart_text(inst["patient_id"])}


def summ_echo_hints(db: ReleaseDB, inst: Instance, n: int = 10) -> dict:
    """Echo the key-finding names the "structured" prompt strategy injects (first `n` of the
    patient's key findings; the loader is unordered, so alphabetical is used here)."""
    return {"summary": "\n".join(f"- {name}" for name in db.key_finding_names(inst["patient_id"])[:n])}


# specialty-conditioned ------------------------------------------------------------------

def spec_chart_dump(db: ReleaseDB, inst: Instance) -> dict:
    return {"summary": db.chart_text(inst["patient_id"])}


def spec_chart_dump_plus_phrase(db: ReleaseDB, inst: Instance) -> dict:
    """The whole chart with a stock abstention phrase appended: content and abstention at once."""
    return {"summary": db.chart_text(inst["patient_id"]) + "\n" + ABSTAIN_PHRASE}


def spec_phrase_only(db: ReleaseDB, inst: Instance) -> dict:
    return {"summary": ABSTAIN_PHRASE}


def spec_abstain_by_name(db: ReleaseDB, inst: Instance) -> dict:
    """Abstain iff the specialty is one of the two most often absent on the train split; otherwise
    paste the chart. Reads the specialty name, not the chart's content."""
    if inst["gt"].get("specialty") in db.most_absent_specialties("train"):
        return {"summary": "", "abstain": True}
    return {"summary": db.chart_text(inst["patient_id"])}


def spec_abstain_always(db: ReleaseDB, inst: Instance) -> dict:
    """The explicit abstention, regardless of the item."""
    return {"summary": "", "abstain": True}


# imaging indication ----------------------------------------------------------------------

def img_restate_order(db: ReleaseDB, inst: Instance) -> dict:
    """Return the terse order indication as the inferred clinical question."""
    o = db.imaging_order(inst["gt_id"])
    return {"clinical_question": o.get("indication") or ""}


def img_chief_complaint(db: ReleaseDB, inst: Instance) -> dict:
    return {"clinical_question": db.chief_complaint(inst["encounter_id"])}


Policy = Callable[[ReleaseDB, Instance], dict]

POLICIES: dict[str, dict[str, Policy]] = {
    "patient_diagnosis": {
        "empty": empty,
        "profile_chronic": dx_profile_chronic,
        "copy_problem_list": dx_copy_problem_list,
        "copy_problem_list_plus_chronic": dx_copy_plus_chronic,
        "echo_problem_list_tool": dx_echo_problem_list_tool,
    },
    "evidence_retrieval": {
        "empty": empty,
        "random": retr_random,
        "single_hpi": retr_single_hpi,
        "section_type_prior": retr_type_prior,
    },
    "context_summarization": {
        "empty": empty,
        "chart_dump": summ_chart_dump,
        "echo_structured_hints": summ_echo_hints,
    },
    "specialty_involved": {
        "empty": empty,
        "abstain_always": spec_abstain_always,
        "phrase_only": spec_phrase_only,
        "chart_dump": spec_chart_dump,
        "chart_dump_plus_phrase": spec_chart_dump_plus_phrase,
        "abstain_by_name": spec_abstain_by_name,
    },
    "specialty_absent": {
        "empty": empty,
        "abstain_always": spec_abstain_always,
        "phrase_only": spec_phrase_only,
        "chart_dump": spec_chart_dump,
        "chart_dump_plus_phrase": spec_chart_dump_plus_phrase,
        "abstain_by_name": spec_abstain_by_name,
    },
    "imaging_indication": {
        "empty": empty,
        "restate_order": img_restate_order,
        "chief_complaint": img_chief_complaint,
    },
}
"""Policies that read no chart content or read it mechanically. Probes that *use the labels*
(dx_reference_categories, dx_reference_plus_secondary) are deliberately not floors."""


def run_policy(db: ReleaseDB, task: str, name: str, insts: list[Instance]) -> list[dict]:
    fn = POLICIES[task][name]
    if fn is dx_echo_problem_list_tool:  # one service session for the whole batch
        tool = db._cache.setdefault("tool_pl", {})
        missing = sorted({i["patient_id"] for i in insts} - set(tool))
        if missing:
            tool.update(db.problem_list_tool(missing))
    return [fn(db, i) for i in insts]
