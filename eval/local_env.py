"""In-process reset-and-step environment over the read-only SQLite release (ROADMAP Stage 5).

Same episode contract as the HTTP environment (`POST /env/reset|step|close`, `eval.env_client`): the
same brief (system prompt, patient intro, tool schemas), the same 9 EHR tools with the same observation
shapes, the same visibility rules (no assessment/plan, the chart's documented problem list, nothing after
the index encounter for point-in-time tasks), the same budget and forced-submit semantics, and the same
reward (`eval.scoring.compute_all_metrics` with the per-instance context the server attaches). No network,
no Postgres, no Redis: one process holds one `LocalEnv`, and processes scale linearly.

    from eval.local_env import LocalEnv
    env = LocalEnv()                              # benchmark_v1.3.db at the repo root
    ro = env.reset(task="patient_diagnosis", split="train", seed=0)
    obs, reward, done, info = env.step("view_encounters", {"patient_id": ro.patient_id})
    obs, reward, done, info = env.step(ro.submit_tool, {...})

Where the code is pure it is the server's code (`env_service.normalize_submission / render / _tool_schemas`,
`visibility.strip_outcome_sections / filter_future_encounters`, the prompt builders, the Epic pydantic
schemas); where the server queries Postgres, the same query is issued against SQLite. The one intended
difference: `search_chart` ranks with an FTS5 (porter, bm25) index instead of Postgres `ts_rank`, so the
candidate set is the same and the order of near-ties may differ. `eval/tests/test_local_env.py` checks the
parity, including against a live server when `SH_ENV_URL` is set.
"""

from __future__ import annotations

import json
import random
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval import degenerate as D
from eval import private_labels
from eval import stage7 as S7
from eval.agents.prompts import build_patient_intro, build_system_prompt
from eval.env_client import ResetObservation
from eval.score_one import _jsonable, _primary_for
from eval.scoring import compute_all_metrics
from epic_sim.app.config import settings
from epic_sim.app.routers.agent import TOOL_DEFINITIONS
from epic_sim.app.schemas.epic import (
    EncounterDetail, EncounterSummary, ImagingResult, LabResult, PathologyResult, PatientSummary, SectionEntry,
)
from epic_sim.app.services import visibility
from epic_sim.app.services.env_service import (
    SUBMIT_TOOLS, normalize_submission, oracle_submission as _server_oracle, render, submit_tool_for, _tool_schemas,
)

TOOL_DEFS = [t.model_dump() for t in TOOL_DEFINITIONS]


class EpisodeError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _dump(model) -> dict:
    return model.model_dump()


class LocalEnv:
    """One environment instance per process. Not thread-safe (sqlite3 connection)."""

    def __init__(self, db_path: str | Path = D.DEFAULT_DB, budget: int | None = None):
        self.db = D.ReleaseDB(db_path)
        self.conn: sqlite3.Connection = self.db.conn
        self.conn.row_factory = None
        self.default_budget = budget or settings.env_default_budget
        self.ep: dict | None = None
        self._fts: sqlite3.Connection | None = None
        self._enc_cache: dict[int, list[tuple]] = {}

    # ------------------------------------------------------------------ setup
    def warmup(self) -> None:
        """Load the per-process caches the scorer and search need (a few seconds), so timings are steady."""
        self.db.neutral_categories(0)
        self.db.concept_extractor()
        self.db.judgments(0)
        self.db.sections(0)
        self.db.patient_terms(0)
        self.db.key_finding_names(0)
        self.db.profile(0)
        self._fts_conn()
        try:
            from rouge_score import rouge_scorer  # noqa: F401 — 1.4 s import (nltk, scipy) on the first summary score
        except ImportError:
            pass

    def _q(self, sql: str, *args) -> list[tuple]:
        return self.conn.execute(sql, args).fetchall()

    # -------------------------------------------------------------- instances
    def instances(self, task: str | None = None, split: str | None = None, limit: int = 500, offset: int = 0) -> list[dict]:
        clauses, params = ["is_diagnostic"], []
        if task:
            clauses.append("task = ?"); params.append(task)
        if split:
            clauses.append("split = ?"); params.append(split)
        rows = self._q(f"select gt_id, task, granularity, split, patient_id, encounter_id, difficulty, "
                       f"coalesce(json_extract(ground_truth,'$.variant'),'unconditioned'), json_extract(ground_truth,'$.clinical_question'), "
                       f"json_extract(ground_truth,'$.specialty') from benchmark_ground_truth where {' and '.join(clauses)} "
                       f"order by gt_id limit ? offset ?", *params, limit, offset)
        out = []
        for r in rows:
            item = {"gt_id": r[0], "task": r[1], "granularity": r[2], "split": r[3], "patient_id": r[4], "encounter_id": r[5], "difficulty": r[6]}
            if r[1] == "context_summarization":
                item["variant"] = r[7]; item["clinical_question"] = r[8]
                if r[9]:
                    item["specialty"] = r[9]
            out.append(item)
        return out

    def _instance(self, gt_id: int) -> D.Instance:
        row = self._q("select b.gt_id, b.task, b.split, coalesce(b.patient_id, l.patient_id), b.encounter_id, b.ground_truth "
                      "from benchmark_ground_truth b left join longitudinal_encounters l using(encounter_id) where b.gt_id = ?", gt_id)
        if not row:
            raise EpisodeError(404, f"no benchmark instance with gt_id={gt_id}")
        gt_id, task, split, pid, eid, gt = row[0]
        gt = json.loads(gt)
        if gt.get(private_labels.REMOVED_FLAG):
            overlay = private_labels.get()
            full = overlay.ground_truth(gt_id) if overlay else None
            if full is None:
                raise EpisodeError(503, f"gt_id={gt_id} is in the private split and this process holds no label overlay ({private_labels.ENV_VAR})")
            gt = full
        return D.Instance(gt_id=gt_id, task=task, patient_id=pid, encounter_id=eid, split=split, gt=gt)

    def _context(self, inst: D.Instance) -> dict:
        """What env_service._load_context computes: prompt inputs, cutoff, chart problems. Never the labels."""
        task, gt, pid, eid = inst["task"], inst["gt"], inst["patient_id"], inst["encounter_id"]
        profile = self.db.profile(pid) or {}
        ctx: dict[str, Any] = {"task_kwargs": {}, "allowed_encounter_ids": None,
                               "chart_problems": [{"display_name": str(c if not isinstance(c, dict) else c.get("name") or c.get("condition") or ""),
                                                   "source": "chart_history"} for c in (profile.get("chronic_conditions") or []) if c]}
        if task == "context_summarization":
            ctx["task_kwargs"]["clinical_question"] = gt.get("clinical_question", "Summarize this patient's clinical course.")
        elif task == "evidence_retrieval":
            ids = [d["diagnosis_id"] for d in gt.get("query_diagnoses", [])]
            names = dict(self._q(f"select diagnosis_id, display_name from diagnoses where diagnosis_id in ({','.join('?' * len(ids))})", *ids)) if ids else {}
            ctx["task_kwargs"]["diagnosis_names"] = ", ".join(names.get(d, f"diagnosis_{d}") for d in ids)
        elif task == "imaging_indication":
            row = (self._q("select modality, body_region, clinical_indication from imaging_orders where encounter_id = ? limit 1", eid) or [(None, None, None)])[0]
            ctx["task_kwargs"].update({"modality": row[0] or "imaging study", "body_region": row[1] or "unspecified",
                                       "clinical_indication": row[2] or "clinical concern"})
        ctx["hidden_section_types"] = list(gt.get("hidden_section_types") or []) if task == "test_selection" else []
        ctx["section_overrides"] = dict(gt.get("section_overrides") or {}) if task in ("error_detection", "atypical_diagnosis") else {}
        ctx["orderable"] = S7.orderable_from_gt(gt) if task == "test_selection" else []
        if eid and task in visibility.POINT_IN_TIME_TASKS:
            ctx["allowed_encounter_ids"] = sorted(r[0] for r in self._q(
                "select encounter_id from longitudinal_encounters where patient_id = ? and encounter_order <= "
                "(select encounter_order from longitudinal_encounters where encounter_id = ?)", pid, eid))
        return ctx

    # ------------------------------------------------------------------ reset
    def reset(self, gt_id: int | None = None, task: str | None = None, split: str | None = None,
              seed: int | None = None, budget: int | None = None) -> ResetObservation:
        if gt_id is None:
            if not task:
                raise EpisodeError(422, "give gt_id, or task (with optional split/seed) to sample an instance")
            items = self.instances(task, split or "train", limit=5000)
            if not items:
                raise EpisodeError(404, f"no instances for task={task} split={split or 'train'}")
            gt_id = random.Random(seed).choice(items)["gt_id"]
        inst = self._instance(gt_id)
        if task and task != inst["task"]:
            raise EpisodeError(422, f"gt_id={gt_id} belongs to task '{inst['task']}', not '{task}'")
        ctx = self._context(inst)
        budget = budget or self.default_budget
        kwargs = dict(ctx["task_kwargs"])
        gt = inst["gt"]
        variant = gt.get("variant", "unconditioned") if inst["task"] == "context_summarization" else None
        self.ep = {
            "episode_id": str(uuid.uuid4()), "agent_token": secrets.token_urlsafe(24),
            "gt_id": gt_id, "task": inst["task"], "split": inst["split"], "variant": variant,
            "patient_id": inst["patient_id"], "encounter_id": inst["encounter_id"],
            "allowed_encounter_ids": ctx["allowed_encounter_ids"], "chart_problems": ctx["chart_problems"],
            "hidden_section_types": ctx["hidden_section_types"], "section_overrides": ctx["section_overrides"],
            "orderable": ctx["orderable"], "orders": [],
            "task_inputs": kwargs, "budget": budget, "steps": 0, "done": False, "submitted": False, "reward": None,
            "trace": [], "created_at": datetime.now(timezone.utc).isoformat(), "_inst": inst,
            "brief": {
                "instructions": build_system_prompt(task=inst["task"], arm="structured", budget=budget,
                                                    encounter_id=inst["encounter_id"], gt_id=gt_id, **kwargs),
                "intro": build_patient_intro(patient_id=inst["patient_id"], task=inst["task"], encounter_id=inst["encounter_id"], **kwargs),
                "tools": _tool_schemas(inst["task"], TOOL_DEFS),
            },
        }
        ep = self.ep
        return ResetObservation(
            episode_id=ep["episode_id"], agent_token=ep["agent_token"], gt_id=gt_id, task=ep["task"], split=ep["split"],
            variant=variant, patient_id=ep["patient_id"], encounter_id=ep["encounter_id"], budget=budget, remaining=budget,
            submit_tool=submit_tool_for(ep["task"]), instructions=ep["brief"]["instructions"], intro=ep["brief"]["intro"],
            task_inputs=kwargs, tools=ep["brief"]["tools"])

    @property
    def episode_id(self) -> str | None:
        return self.ep["episode_id"] if self.ep and not self.ep["done"] else None

    @property
    def submit_tool(self) -> str:
        return submit_tool_for(self.ep["task"])

    # ------------------------------------------------------------------- step
    def step(self, name: str, arguments: dict | None = None) -> tuple[Any, float, bool, dict]:
        """(observation, reward, done, info), as eval.env_client.SyntheticHospitalEnv.step returns them."""
        ep = self.ep
        if ep is None:
            raise EpisodeError(409, "call reset() first")
        if ep["done"]:
            raise EpisodeError(409, "episode is finished; call reset")
        arguments = arguments or {}
        task, submit_tool = ep["task"], submit_tool_for(ep["task"])
        remaining = ep["budget"] - ep["steps"]

        if name == submit_tool:
            prediction = normalize_submission(task, arguments)
            if task == "test_selection" and prediction is not None:
                prediction = {**prediction, "tests_ordered": list(ep["orders"])}      # the trace, never the claim
            result = self._score(ep["_inst"], prediction or {})
            ep.update({"done": True, "submitted": True, "reward": result["reward"], "steps": ep["steps"] + 1})
            ep["trace"].append({"step": ep["steps"], "tool_name": name, "submitted": True, "forced": remaining <= 0, "malformed": prediction is None})
            obs = {"status": "submitted", "malformed": prediction is None}
            verbose = {x.strip() for x in settings.verbose_score_splits.split(",") if x.strip()}
            info = {"forced": remaining <= 0, "malformed": prediction is None, "steps": ep["steps"],
                    "reward_metric": result["reward_metric"], "metrics": result["metrics"] if ep["split"] in verbose else {},
                    "remaining": max(remaining - 1, 0), "step": ep["steps"], "observation_text": render(obs)}
            return obs, float(result["reward"]), True, info

        if name in SUBMIT_TOOLS:
            obs = {"error": f"This task is scored through {submit_tool}; use that tool to finish the episode."}
        elif remaining <= 0:
            obs = {"error": f"Action budget of {ep['budget']} spent. Only {submit_tool} is accepted now; submit your best answer."}
        elif name == "order_test":
            if task != "test_selection":
                obs = {"error": "order_test is available only in test_selection episodes."}
            else:
                obs = S7.order_result(str(arguments.get("name", "")), ep["orderable"])
                ep["orders"].append({k: obs.get(k) for k in ("test", "matched")})
            ep["steps"] += 1
            remaining -= 1
        else:
            try:
                raw = self._dispatch(name, arguments)
            except KeyError as exc:
                raw = {"error": f"missing argument {exc}"}
            except EpisodeError as exc:
                raw = {"error": exc.detail}
            except Exception as exc:  # noqa: BLE001 — a bad tool call is an observation, not a crash
                raw = {"error": f"{type(exc).__name__}: {exc}"}
            obs = visibility.strip_outcome_sections(raw)
            if ep["allowed_encounter_ids"] is not None:
                obs = visibility.filter_future_encounters(obs, set(ep["allowed_encounter_ids"]))
            obs = S7.apply_episode_rules(name, arguments, obs, ep["section_overrides"], set(ep["hidden_section_types"]), ep["encounter_id"])
            ep["steps"] += 1
            remaining -= 1
        ep["trace"].append({"step": ep["steps"], "tool_name": name, "arguments": arguments,
                            "error": obs.get("error") if isinstance(obs, dict) else None})
        return obs, 0.0, False, {"steps": ep["steps"], "remaining": remaining, "step": ep["steps"], "observation_text": render(obs)}

    def state(self) -> dict:
        ep = self.ep
        view = {k: ep[k] for k in ("episode_id", "gt_id", "task", "split", "patient_id", "encounter_id", "budget", "steps",
                                   "done", "submitted", "reward", "trace", "created_at")}
        if ep["task"] == "test_selection":
            view["orders"] = list(ep["orders"])
        return view

    def close(self) -> dict | None:
        if self.ep is None:
            return None
        if not self.ep["done"]:
            self.ep["done"] = True
            self.ep["reward"] = 0.0
        return self.state()

    def oracle(self) -> dict:
        """A label-derived submission for the current episode (what GET /env/oracle returns)."""
        return D.oracle(self.db, self._unit_instance(self.ep["_inst"]))

    def oracle_orders(self) -> list[str]:
        """test_selection: the tests an oracle agent orders before submitting (empty for other tasks)."""
        return list(self.ep["_inst"]["gt"].get("discriminating", [])) if self.ep["task"] == "test_selection" else []

    # ----------------------------------------------------------------- reward
    @staticmethod
    def _unit_instance(inst: D.Instance) -> D.Instance:
        """degenerate.py splits summarization into scoring units; the scorer itself takes the base task."""
        if inst["task"] == "context_summarization" and inst["gt"].get("variant") == "specialty_conditioned":
            unit = "specialty_absent" if inst["gt"].get("involvement") == "absent" else "specialty_involved"
            return D.Instance({**inst, "task": unit})
        return inst

    def _score(self, inst: D.Instance, prediction: dict) -> dict:
        task = inst["task"]
        gt = D.attach_context(self.db, inst)
        kwargs = {"concept_extractor": self.db.concept_extractor()} if task in ("imaging_indication", "context_summarization") else {}
        metrics = _jsonable(compute_all_metrics(task, [prediction if isinstance(prediction, dict) else {}], [gt], **kwargs))
        metric = _primary_for(task, gt, metrics)
        reward = metrics.get(metric)
        reward = float(min(1.0, max(0.0, reward))) if isinstance(reward, (int, float)) else 0.0
        return {"reward": reward, "reward_metric": metric, "metrics": metrics}

    # ------------------------------------------------------------------ tools
    def _dispatch(self, tool: str, args: dict):
        ep = self.ep
        match tool:
            case "search_patients":
                return self._search_patients(args.get("name"), args.get("gender"), args.get("mrn"), args.get("limit", 10))
            case "open_chart":
                return self._open_chart(args["patient_id"])
            case "view_encounters":
                return [_dump(e) for e in self._encounters(args["patient_id"])]
            case "view_encounter_detail":
                d = self._encounter_detail(args["encounter_id"])
                return {"error": "Encounter not found"} if d is None else _dump(d)
            case "view_section":
                s = self._section(args["section_id"])
                return {"error": "Section not found or access denied"} if s is None else _dump(s)
            case "view_results":
                return self._results(args["patient_id"], args["result_type"])
            case "search_chart":
                return [_dump(s) for s in self._search_chart(args["patient_id"], args["query"])]
            case "view_problem_list":
                return list(ep["chart_problems"])
            case "view_medications":
                return [_dump(s) for s in self._medications(args["patient_id"])]
            case _:
                raise EpisodeError(400, f"Unknown tool: {tool}")

    def _encounter_rows(self, patient_id: int) -> list[tuple]:
        if patient_id not in self._enc_cache:
            self._enc_cache[patient_id] = self._q(
                "select encounter_id, encounter_date, encounter_type, chief_complaint, department, attending_name, encounter_order "
                "from longitudinal_encounters where patient_id = ? order by encounter_date, encounter_id", patient_id)
        return self._enc_cache[patient_id]

    def _encounters(self, patient_id: int) -> list[EncounterSummary]:
        return [EncounterSummary(encounter_id=r[0], date=r[1], type=r[2], chief_complaint=r[3], department=r[4], attending=r[5])
                for r in self._encounter_rows(patient_id)]

    def _open_chart(self, patient_id: int) -> dict:
        row = self._q("select profile, age, sex, race_ethnicity, insurance from longitudinal_patients where patient_id = ?", patient_id)
        if not row:
            return _dump(PatientSummary(patient_id=patient_id))
        profile = json.loads(row[0][0] or "{}") if isinstance(row[0][0], str) else (row[0][0] or {})
        encs = self._encounter_rows(patient_id)
        recent = [EncounterSummary(encounter_id=r[0], date=r[1], type=r[2], chief_complaint=r[3], department=r[4], attending=r[5])
                  for r in sorted(encs, key=lambda r: (r[1] or ""), reverse=True)[:5]]
        meds: list[str] = []
        allergies: list[str] = []
        if recent:
            for st, txt in self._q("select section_type, section_text from encounter_ehr_sections where encounter_id = ? "
                                   "and section_type in ('medications','allergies')", recent[0].encounter_id):
                lines = [ln.strip() for ln in (txt or "").split("\n") if ln.strip()]
                if st == "medications":
                    meds = lines
                else:
                    allergies = lines
        summary = PatientSummary(patient_id=patient_id, name=profile.get("name", f"Patient {patient_id}"), age=row[0][1], sex=row[0][2],
                                 race_ethnicity=row[0][3], insurance=row[0][4], pcp=None, recent_encounters=recent,
                                 medications=meds, allergies=allergies)
        out = _dump(summary)
        out["active_problems"] = list(self.ep["chart_problems"])     # env_service.replace_problem_list
        return out

    def _encounter_detail(self, encounter_id: int) -> EncounterDetail | None:
        row = self._q("select encounter_id, patient_id, encounter_date, encounter_type, chief_complaint, department, attending_name "
                      "from longitudinal_encounters where encounter_id = ?", encounter_id)
        if not row:
            return None
        e = row[0]
        sections = [SectionEntry(section_id=s[0], encounter_id=encounter_id, section_type=s[1], section_text=s[2] or "", section_order=s[3])
                    for s in self._q("select id, section_type, section_text, section_order from encounter_ehr_sections "
                                     "where encounter_id = ? order by section_order, id", encounter_id)
                    if not visibility.is_hidden(s[1])]
        return EncounterDetail(encounter_id=e[0], patient_id=e[1], date=e[2], type=e[3], chief_complaint=e[4], department=e[5],
                               attending=e[6], sections=sections)

    def _section(self, section_id: int) -> SectionEntry | None:
        row = self._q("select id, encounter_id, section_type, section_text, section_order from encounter_ehr_sections where id = ?", section_id)
        if not row or visibility.is_hidden(row[0][2]):
            return None
        s = row[0]
        return SectionEntry(section_id=s[0], encounter_id=s[1], section_type=s[2], section_text=s[3] or "", section_order=s[4])

    def _results(self, patient_id: int, result_type: str):
        if result_type == "labs":
            return [_dump(LabResult(encounter_id=r[0], encounter_date=r[1], section_text=r[2])) for r in self._sections_of_type(patient_id, "labs")]
        if result_type == "pathology":
            return [_dump(PathologyResult(encounter_id=r[0], encounter_date=r[1], section_text=r[2])) for r in self._sections_of_type(patient_id, "pathology")]
        if result_type == "imaging":
            rows = self._q("select ees.encounter_id, le.encounter_date, ees.section_text, "
                           "case when io.order_id is not null then 1 else 0 end from encounter_ehr_sections ees "
                           "join longitudinal_encounters le on le.encounter_id = ees.encounter_id "
                           "left join imaging_orders io on io.encounter_id = ees.encounter_id "
                           "where le.patient_id = ? and ees.section_type = 'imaging' order by le.encounter_date", patient_id)
            return [_dump(ImagingResult(encounter_id=r[0], encounter_date=r[1], section_text=r[2] or "", has_order=bool(r[3]))) for r in rows]
        return {"error": f"Unknown result_type: {result_type}"}

    def _sections_of_type(self, patient_id: int, section_type: str) -> list[tuple]:
        return self._q("select ees.encounter_id, le.encounter_date, coalesce(ees.section_text, '') from encounter_ehr_sections ees "
                       "join longitudinal_encounters le on le.encounter_id = ees.encounter_id "
                       "where le.patient_id = ? and ees.section_type = ? order by le.encounter_date", patient_id, section_type)

    def _medications(self, patient_id: int) -> list[SectionEntry]:
        rows = self._q("select ees.id, ees.encounter_id, ees.section_type, coalesce(ees.section_text, ''), ees.section_order "
                       "from encounter_ehr_sections ees join longitudinal_encounters le on le.encounter_id = ees.encounter_id "
                       "where le.patient_id = ? and ees.section_type = 'medications' order by le.encounter_date", patient_id)
        return [SectionEntry(section_id=r[0], encounter_id=r[1], section_type=r[2], section_text=r[3], section_order=r[4]) for r in rows]

    def _search_patients(self, name, gender, mrn, limit) -> list[dict]:
        if mrn:
            row = self._q("select patient_id, profile, age, sex from longitudinal_patients where patient_id = ?", int(mrn))
            if not row:
                return []
            p = json.loads(row[0][1] or "{}")
            return [{"patient_id": row[0][0], "name": p.get("name", f"Patient {row[0][0]}"), "age": row[0][2], "sex": row[0][3]}]
        clauses, params = [], []
        if name:
            clauses.append("lower(json_extract(profile,'$.name')) like ?"); params.append(f"%{name.lower()}%")
        if gender and gender[0].upper() in ("M", "F"):
            clauses.append("sex = ?"); params.append(gender[0].upper())
        rows = self._q(f"select patient_id, profile, age, sex from longitudinal_patients where {' and '.join(clauses) or '1'} "
                       f"order by patient_id limit ?", *params, limit)
        return [{"patient_id": r[0], "name": json.loads(r[1] or "{}").get("name", f"Patient {r[0]}"), "age": r[2], "sex": r[3]} for r in rows]

    # ----------------------------------------------------------------- search
    def _fts_conn(self) -> sqlite3.Connection:
        """FTS5 index over every section (porter stemmer, bm25): the SQLite analogue of the Postgres tsvector."""
        if self._fts is None:
            fts = sqlite3.connect(":memory:")
            fts.execute("create virtual table s using fts5(section_text, tokenize='porter unicode61')")
            fts.execute("create table meta(id integer primary key, patient_id integer, section_type text)")
            rows = self._q("select ees.id, le.patient_id, ees.section_type, coalesce(ees.section_text, '') from encounter_ehr_sections ees "
                           "join longitudinal_encounters le on le.encounter_id = ees.encounter_id")
            fts.executemany("insert into s(rowid, section_text) values (?, ?)", [(r[0], r[3]) for r in rows])
            fts.executemany("insert into meta(id, patient_id, section_type) values (?, ?, ?)", [(r[0], r[1], r[2]) for r in rows])
            fts.execute("create index meta_pid on meta(patient_id)")
            fts.commit()
            self._fts = fts
        return self._fts

    def _search_chart(self, patient_id: int, query: str, limit: int = 20) -> list[SectionEntry]:
        words = [w for w in re.findall(r"[A-Za-z0-9]+", query or "") if w]
        if not words:
            return []
        fts = self._fts_conn()
        hidden = tuple(visibility.hidden_sections())
        match = " ".join(f'"{w}"' for w in words)                       # plainto_tsquery: AND of the words
        sql = ("select s.rowid from s join meta m on m.id = s.rowid where s MATCH ? and m.patient_id = ? "
               + (f"and m.section_type not in ({','.join('?' * len(hidden))}) " if hidden else "") + "order by bm25(s) limit 50")
        ids = [r[0] for r in fts.execute(sql, (match, patient_id, *hidden)).fetchall()][:limit]
        if not ids:
            return []
        rows = {r[0]: r for r in self._q(f"select id, encounter_id, section_type, coalesce(section_text, ''), section_order "
                                         f"from encounter_ehr_sections where id in ({','.join('?' * len(ids))})", *ids)}
        return [SectionEntry(section_id=rows[i][0], encounter_id=rows[i][1], section_type=rows[i][2], section_text=rows[i][3], section_order=rows[i][4])
                for i in ids if i in rows]
