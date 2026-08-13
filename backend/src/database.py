import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).resolve().parent.parent / "careerpath.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL DEFAULT '',
            language_preference TEXT NOT NULL DEFAULT '',
            facts TEXT NOT NULL DEFAULT '{}',
            last_interaction TEXT NOT NULL DEFAULT ''
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS call_outcomes (
            call_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            outcome TEXT NOT NULL CHECK (outcome IN ('SUCCESS', 'FAILED')),
            started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL DEFAULT ''
        )
        """
    )

    conn.commit()
    conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start_call(call_id: str, user_id: str) -> None:
    """
    Create a call record as FAILED by default.

    A call only becomes SUCCESS when the agent explicitly confirms that
    the learner completed the daily learning exercise. Starting with
    FAILED also means an unexpected shutdown still leaves the call
    represented in the dashboard totals.
    """
    now = _utc_now()

    conn = get_connection()
    conn.execute(
        """
        INSERT INTO call_outcomes (
            call_id,
            user_id,
            outcome,
            started_at,
            ended_at
        )
        VALUES (?, ?, 'FAILED', ?, '')
        ON CONFLICT(call_id) DO UPDATE SET
            user_id = excluded.user_id,
            outcome = 'FAILED',
            started_at = excluded.started_at,
            ended_at = ''
        """,
        (call_id, user_id, now),
    )
    conn.commit()
    conn.close()


def finish_call(
    call_id: str,
    outcome: str,
) -> None:
    """
    Finalize a call outcome.

    Only SUCCESS and FAILED are accepted. The dashboard never needs
    caller-level information; it reads aggregate counts from this table.
    """
    normalized_outcome = outcome.strip().upper()

    if normalized_outcome not in {"SUCCESS", "FAILED"}:
        raise ValueError("Call outcome must be SUCCESS or FAILED.")

    conn = get_connection()
    conn.execute(
        """
        UPDATE call_outcomes
        SET outcome = ?, ended_at = ?
        WHERE call_id = ?
        """,
        (normalized_outcome, _utc_now(), call_id),
    )
    conn.commit()
    conn.close()


def get_call_stats() -> dict[str, int]:
    """Return only the aggregate call statistics needed by the dashboard."""
    conn = get_connection()

    row = conn.execute(
        """
        SELECT
            COUNT(*) AS total_calls,
            COALESCE(SUM(CASE WHEN outcome = 'SUCCESS' THEN 1 ELSE 0 END), 0)
                AS successful_calls,
            COALESCE(SUM(CASE WHEN outcome = 'FAILED' THEN 1 ELSE 0 END), 0)
                AS failed_calls
        FROM call_outcomes
        """
    ).fetchone()

    conn.close()

    return {
        "totalCalls": int(row["total_calls"] or 0),
        "successfulCalls": int(row["successful_calls"] or 0),
        "failedCalls": int(row["failed_calls"] or 0),
    }


def get_user(user_id: str) -> dict[str, Any] | None:
    """
    Return the saved memory for a caller.

    Returns None when this caller has never been saved.
    """
    conn = get_connection()

    row = conn.execute(
        """
        SELECT
            user_id,
            name,
            language_preference,
            facts,
            last_interaction
        FROM users
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()

    conn.close()

    if row is None:
        return None

    try:
        facts = json.loads(row["facts"] or "{}")
    except (json.JSONDecodeError, TypeError):
        facts = {}

    if not isinstance(facts, dict):
        facts = {}

    return {
        "user_id": row["user_id"],
        "name": row["name"],
        "language_preference": row["language_preference"],
        "facts": facts,
        "last_interaction": row["last_interaction"],
    }


def save_user_memory(
    user_id: str,
    name: str = "",
    language_preference: str = "",
    facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Insert or update caller memory.

    Existing facts are preserved unless a key is explicitly replaced.
    """
    existing = get_user(user_id)

    existing_facts: dict[str, Any] = {}

    if existing and isinstance(existing.get("facts"), dict):
        existing_facts.update(existing["facts"])

    if facts:
        existing_facts.update(facts)

    final_name = name.strip() if name.strip() else (
        existing.get("name", "") if existing else ""
    )

    final_language = (
        language_preference.strip()
        if language_preference.strip()
        else (
            existing.get("language_preference", "")
            if existing
            else ""
        )
    )

    if not final_name:
        raise ValueError("A name is required when creating caller memory.")

    now = _utc_now()

    conn = get_connection()

    conn.execute(
        """
        INSERT INTO users (
            user_id,
            name,
            language_preference,
            facts,
            last_interaction
        )
        VALUES (?, ?, ?, ?, ?)

        ON CONFLICT(user_id) DO UPDATE SET
            name = excluded.name,
            language_preference = excluded.language_preference,
            facts = excluded.facts,
            last_interaction = excluded.last_interaction
        """,
        (
            user_id,
            final_name,
            final_language,
            json.dumps(
                existing_facts,
                ensure_ascii=False,
            ),
            now,
        ),
    )

    conn.commit()
    conn.close()

    return {
        "user_id": user_id,
        "name": final_name,
        "language_preference": final_language,
        "facts": existing_facts,
        "last_interaction": now,
    }


init_db()
