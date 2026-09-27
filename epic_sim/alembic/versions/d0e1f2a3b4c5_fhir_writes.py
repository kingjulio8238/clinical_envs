"""fhir_writes: resources created through the FHIR write API (Stage 4b).

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "d0e1f2a3b4c5"
down_revision: Union[str, None] = "c9d0e1f2a3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "fhir_writes",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("resource_type", sa.Text(), nullable=False),
        sa.Column("patient_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Text(), nullable=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("resource", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("idx_fhir_writes_patient_type", "fhir_writes", ["patient_id", "resource_type"])
    op.create_index("idx_fhir_writes_session", "fhir_writes", ["session_id"])


def downgrade() -> None:
    op.drop_index("idx_fhir_writes_session", table_name="fhir_writes")
    op.drop_index("idx_fhir_writes_patient_type", table_name="fhir_writes")
    op.drop_table("fhir_writes")
