"""
Persistent chat session storage using SQLite.
Sessions and messages survive container restarts via a mounted volume.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

import aiosqlite

from src.config import settings


async def init_db() -> None:
    """Create tables if they don't exist."""
    Path(settings.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                sources     TEXT,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        await db.commit()


async def create_session(title: str = "New Chat") -> dict:
    session_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        await db.execute(
            "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
            (session_id, title, created_at),
        )
        await db.commit()
    return {"id": session_id, "title": title, "created_at": created_at}


async def list_sessions() -> list[dict]:
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        async with db.execute(
            "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


async def get_session(session_id: str) -> dict | None:
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        async with db.execute(
            "SELECT id, title, created_at FROM sessions WHERE id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None

        async with db.execute(
            "SELECT role, content, sources FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ) as cursor:
            msg_rows = await cursor.fetchall()

    messages = [
        {
            "role": r[0],
            "content": r[1],
            "sources": json.loads(r[2]) if r[2] else [],
        }
        for r in msg_rows
    ]
    return {"id": row[0], "title": row[1], "created_at": row[2], "messages": messages}


async def append_messages(
    session_id: str,
    messages: list[dict],
    new_title: str | None = None,
) -> None:
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        for msg in messages:
            sources_json = json.dumps(msg.get("sources", []))
            await db.execute(
                "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, msg["role"], msg["content"], sources_json, now),
            )
        if new_title:
            await db.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (new_title, session_id),
            )
        await db.commit()


async def delete_session(session_id: str) -> None:
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()
