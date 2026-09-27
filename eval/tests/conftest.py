"""Shared fixtures for the SQLite-backed eval suites.

The simulator's ORM models describe the PostgreSQL schema, which has a few columns the released
SQLite file does not carry (e.g. `encounter_ehr_sections.source_section_id`, `longitudinal_patients.pcp_name`).
To run the real service functions over the release file, `orm_sqlite_path` provides a temporary copy
with every missing ORM column added as NULL. Data is unchanged.
"""

from __future__ import annotations

import shutil
import sqlite3

import pytest

from eval import degenerate as D


@pytest.fixture(scope="session")
def orm_sqlite_path(tmp_path_factory):
    if not D.DEFAULT_DB.exists():
        pytest.skip(f"{D.DEFAULT_DB.name} not present")
    from epic_sim.app.models import Base  # imports every model

    path = tmp_path_factory.mktemp("release") / "benchmark_orm.db"
    shutil.copyfile(D.DEFAULT_DB, path)
    conn = sqlite3.connect(path)
    existing = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    for table in Base.metadata.tables.values():
        if table.name not in existing:
            continue
        have = {r[1] for r in conn.execute(f"pragma table_info({table.name})")}
        for col in table.columns:
            if col.name not in have:
                conn.execute(f'alter table {table.name} add column "{col.name}"')
    conn.commit()
    conn.close()
    return path


@pytest.fixture(scope="session")
def orm_sqlite_url(orm_sqlite_path):
    return f"sqlite+aiosqlite:///file:{orm_sqlite_path}?mode=ro&uri=true"
