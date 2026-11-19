"""
BARBARIK - Scenario database (SQLite)
=====================================

Persists simulated scenarios so the dashboard's History page can list, filter
and export them.  Uses only the stdlib ``sqlite3`` module.

Schema
------
scenarios (
    id                TEXT PRIMARY KEY,
    timestamp         TEXT,
    side_a_json       TEXT,     -- JSON-encoded Side A attributes
    side_b_json       TEXT,     -- JSON-encoded Side B attributes
    environment_json  TEXT,     -- JSON-encoded environment parameters
    predicted_outcome TEXT,     -- SIDE_A_ADVANTAGE / SIDE_B_ADVANTAGE / STALEMATE
    prob_a            REAL,
    prob_b            REAL,
    prob_draw         REAL
)

All queries handle the empty-database case gracefully (return ``[]`` / ``None``).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DB_PATH = Path("data/barbarik_history.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS scenarios (
    id                TEXT PRIMARY KEY,
    timestamp         TEXT NOT NULL,
    side_a_json       TEXT NOT NULL,
    side_b_json       TEXT NOT NULL,
    environment_json  TEXT NOT NULL,
    predicted_outcome TEXT NOT NULL,
    prob_a            REAL NOT NULL,
    prob_b            REAL NOT NULL,
    prob_draw         REAL NOT NULL
)
"""


def _connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: str | Path = DB_PATH) -> None:
    """Create the scenarios table if it does not exist. Idempotent."""
    with _connect(path) as conn:
        conn.execute(SCHEMA)


def save_scenario(
    scenario_id: str,
    timestamp: str,
    side_a: dict,
    side_b: dict,
    environment: dict,
    predicted_outcome: str,
    prob_a: float,
    prob_b: float,
    prob_draw: float,
    path: str | Path = DB_PATH,
) -> None:
    """Insert (or replace) a completed scenario in the database."""
    init_db(path)
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO scenarios
                (id, timestamp, side_a_json, side_b_json, environment_json,
                 predicted_outcome, prob_a, prob_b, prob_draw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scenario_id,
                timestamp,
                json.dumps(side_a),
                json.dumps(side_b),
                json.dumps(environment),
                predicted_outcome,
                float(prob_a),
                float(prob_b),
                float(prob_draw),
            ),
        )


def get_all_scenarios(path: str | Path = DB_PATH) -> list[dict]:
    """Return every scenario as a dict, newest first. Empty -> []."""
    init_db(path)
    with _connect(path) as conn:
        rows = conn.execute(
            "SELECT * FROM scenarios ORDER BY timestamp DESC, rowid DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_scenario(scenario_id: str, path: str | Path = DB_PATH) -> dict | None:
    """Return a single scenario by id, or None if not found / empty."""
    init_db(path)
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT * FROM scenarios WHERE id = ?", (scenario_id,)
        ).fetchone()
        return dict(row) if row is not None else None


def delete_scenario(scenario_id: str, path: str | Path = DB_PATH) -> None:
    """Delete a scenario by id. No-op if the id does not exist."""
    init_db(path)
    with _connect(path) as conn:
        conn.execute("DELETE FROM scenarios WHERE id = ?", (scenario_id,))


def _decode(row: dict) -> dict:
    """Decode the JSON columns of a stored row into python structures."""
    result = {
        "id": row["id"],
        "timestamp": row["timestamp"],
        "predicted_outcome": row["predicted_outcome"],
        "prob_a": row["prob_a"],
        "prob_b": row["prob_b"],
        "prob_draw": row["prob_draw"],
    }
    try:
        result["side_a"] = json.loads(row["side_a_json"])
    except (json.JSONDecodeError, TypeError):
        result["side_a"] = {}
    try:
        result["side_b"] = json.loads(row["side_b_json"])
    except (json.JSONDecodeError, TypeError):
        result["side_b"] = {}
    try:
        result["environment"] = json.loads(row["environment_json"])
    except (json.JSONDecodeError, TypeError):
        result["environment"] = {}
    return result