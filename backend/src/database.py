import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "careerpath.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            language_preference TEXT,
            current_level TEXT,
            interests TEXT,
            target_career TEXT,
            last_interaction TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_user(
    user_id: str,
    name: str,
    language_preference: str = "",
    current_level: str = "",
    interests: str = "",
    target_career: str = "",
):
    conn = get_connection()

    conn.execute("""
        INSERT INTO users (
            user_id,
            name,
            language_preference,
            current_level,
            interests,
            target_career,
            last_interaction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            name = excluded.name,
            language_preference = excluded.language_preference,
            current_level = excluded.current_level,
            interests = excluded.interests,
            target_career = excluded.target_career,
            last_interaction = excluded.last_interaction
    """, (
        user_id,
        name,
        language_preference,
        current_level,
        interests,
        target_career,
        datetime.now().isoformat(),
    ))

    conn.commit()
    conn.close()


def get_user(user_id: str):
    conn = get_connection()

    cursor = conn.execute("""
        SELECT
            user_id,
            name,
            language_preference,
            current_level,
            interests,
            target_career,
            last_interaction
        FROM users
        WHERE user_id = ?
    """, (user_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "user_id": row[0],
        "name": row[1],
        "language_preference": row[2],
        "facts": {
            "current_level": row[3],
            "interests": row[4],
            "target_career": row[5],
        },
        "last_interaction": row[6],
    }