"""Stage 7 task families in the eval_task enum and the task CHECK constraints.

Revision ID: e1f2a3b4c5d6
Revises: d0e1f2a3b4c5
"""
from typing import Sequence, Union
from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, None] = "d0e1f2a3b4c5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

NEW = ("differential_diagnosis", "test_selection", "error_detection", "lab_triage", "atypical_diagnosis")
ALL = ("diagnosis_accuracy", "context_summarization", "evidence_retrieval", "imaging_indication", "patient_diagnosis") + NEW


def upgrade() -> None:
    # enum values must be committed before a CHECK constraint refers to them
    with op.get_context().autocommit_block():
        for v in NEW:
            op.execute(f"ALTER TYPE eval_task ADD VALUE IF NOT EXISTS '{v}'")
    tasks = ", ".join(f"'{t}'" for t in ALL)
    for table in ("benchmark_ground_truth", "evaluation_runs"):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_task_check")
        op.execute(f"ALTER TABLE {table} ADD CONSTRAINT {table}_task_check CHECK (task IN ({tasks}))")


def downgrade() -> None:
    # PostgreSQL cannot drop enum values; restore the narrower CHECK constraints only
    tasks = ", ".join(f"'{t}'" for t in ALL if t not in NEW)
    for table in ("benchmark_ground_truth", "evaluation_runs"):
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_task_check")
        op.execute(f"ALTER TABLE {table} ADD CONSTRAINT {table}_task_check CHECK (task IN ({tasks}))")
