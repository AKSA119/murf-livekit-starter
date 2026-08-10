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

    conn.commit()
    conn.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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