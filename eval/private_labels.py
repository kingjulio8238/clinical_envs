"""Labels of the private split live outside the released database.

`scripts/carve_private_split.py` moves the label fields of every private-split instance (and its
relevance judgments) into a separate SQLite file that is gitignored (`private/labels_v1.3.db`) and
leaves only the *inputs* in `benchmark_v1.3.db` (`{"_labels_removed": true, ...inputs}`). The scorer
overlays the labels back at scoring time when the file is present (`SH_PRIVATE_LABELS_DB`), so
`/score` and `/env` work for the operator who holds the file and nobody else can read the answers.

    from eval.private_labels import get
    labels = get()            # None when the overlay is absent
    labels.ground_truth(gt_id), labels.judgments(gt_id)
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "private" / "labels_v1.3.db"
ENV_VAR = "SH_PRIVATE_LABELS_DB"
PRIVATE_SPLIT = "private"
REMOVED_FLAG = "_labels_removed"

# Ground-truth keys that are labels, per task. Everything else in the JSON is an input the policy
# legitimately sees (retrieval query, clinical question, specialty name) and stays in the release.
LABEL_KEYS: dict[str, tuple[str, ...]] = {
    "patient_diagnosis": ("active_diagnoses", "chronic_conditions", "encounter_diagnosis_map", "neutral_extra"),
    "evidence_retrieval": ("grade_distribution", "num_passages"),      # judgments live in their own table
    "context_summarization": ("must_include_findings", "reference_summary", "key_encounters",
                              "tiers", "involvement", "tau_residual"),
    "imaging_indication": ("inferred_clinical_question", "pre_read_summary", "must_include_findings",
                           "differential_context", "relevant_clinical_data", "reference_terms"),
    # Stage 7 families
    "differential_diagnosis": ("correct", "distractors"),
    "test_selection": ("diagnosis", "orderable", "discriminating", "n_needed"),
    "error_detection": ("section_id", "section_type", "error_type", "original", "injected", "section_overrides"),
    "lab_triage": ("relevant", "background", "most_urgent", "results"),
    "atypical_diagnosis": ("active_diagnoses", "chronic_conditions", "encounter_diagnosis_map", "neutral_extra",
                           "masked_findings", "section_overrides"),
}


def strip_labels(task: str, gt: dict) -> dict:
    """The public form of a private instance's ground truth: inputs only, flagged."""
    keep = {k: v for k, v in gt.items() if k not in LABEL_KEYS.get(task, ())}
    keep[REMOVED_FLAG] = True
    return keep


class PrivateLabels:
    def __init__(self, path: Path):
        self.path = Path(path)
        # read-only and shared by the process; worker threads (eval.protocol_run) read it through a lock
        self.conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        self.lock = __import__("threading").Lock()

    def ground_truth(self, gt_id: int) -> dict | None:
        with self.lock:
            row = self.conn.execute("select ground_truth from benchmark_ground_truth where gt_id=?", (gt_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def judgments(self, gt_id: int) -> dict[str, int]:
        with self.lock:
            return dict(self.conn.execute("select passage_id, relevance_grade from relevance_judgments where gt_id=?", (gt_id,)).fetchall())

    def gt_ids(self) -> set[int]:
        with self.lock:
            return {r[0] for r in self.conn.execute("select gt_id from benchmark_ground_truth")}


def path() -> Path:
    return Path(os.environ.get(ENV_VAR) or DEFAULT_PATH)


_cache: dict[str, PrivateLabels] = {}


def get() -> PrivateLabels | None:
    """The overlay, or None when the operator does not hold the private labels. An absent file is
    re-checked on every call (cheap), so an overlay mounted after start-up is picked up."""
    p = path()
    key = str(p)
    if key not in _cache:
        if not p.exists():
            return None
        _cache[key] = PrivateLabels(p)
    return _cache[key]


class LabelsUnavailable(LookupError):
    """The instance is in the private split and this process holds no overlay."""
