"""Resources created through the FHIR write API (Stage 4b).

A runtime table: agents' orders, observations, medication requests and problem-list entries land here,
never in the benchmark tables. Rows are scoped to the agent session that wrote them (or, without a
session, to the user), so episodes never see each other's writes.
"""

from __future__ import annotations

import uuid

from sqlalchemy import JSON, Index, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from epic_sim.app.models.base import Base


class FhirWrite(Base):
    __tablename__ = "fhir_writes"

    id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: str(uuid.uuid4()))
    resource_type: Mapped[str] = mapped_column(Text, nullable=False)
    patient_id: Mapped[int] = mapped_column(Integer, nullable=False)
    session_id: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    resource: Mapped[dict] = mapped_column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    created_at: Mapped[str] = mapped_column(TIMESTAMP(timezone=True).with_variant(Text, "sqlite"), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_fhir_writes_patient_type", "patient_id", "resource_type"),
        Index("idx_fhir_writes_session", "session_id"),
    )
