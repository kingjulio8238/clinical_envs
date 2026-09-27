"""ICD-10-CM provenance on diagnoses and the diagnosis_merges table (Stage 4).

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "c9d0e1f2a3b4"
down_revision: Union[str, None] = "b8c9d0e1f2a3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for col in ("icd10_status", "icd10_repaired_from", "icd10_repair_reason", "icd10_flag"):
        op.add_column("diagnoses", sa.Column(col, sa.Text(), nullable=True))
    op.add_column("diagnoses", sa.Column("merged_into", sa.Integer(), nullable=True))
    op.create_table(
        "diagnosis_merges",
        sa.Column("old_id", sa.Integer(), primary_key=True),
        sa.Column("new_id", sa.Integer(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("diagnosis_merges")
    for col in ("merged_into", "icd10_flag", "icd10_repair_reason", "icd10_repaired_from", "icd10_status"):
        op.drop_column("diagnoses", col)
