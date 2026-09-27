"""Add the 'private' split label (Stage 2.7).

The private split is carved out of 'train' by scripts/carve_private_split.py; its labels live in an
operator-held overlay (eval/private_labels.py), not in the released database.

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
"""
from typing import Sequence, Union
from alembic import op

revision: str = "b8c9d0e1f2a3"
down_revision: Union[str, None] = "a7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE split_type ADD VALUE IF NOT EXISTS 'private'")


def downgrade() -> None:
    # PostgreSQL cannot drop an enum value; rows labelled 'private' would have to be relabelled first.
    pass
