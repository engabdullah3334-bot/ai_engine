# -*- coding: utf-8 -*-
"""
ai_engine/chatbot/local_archive.py
==================================
Built-in persistent SQLite conversation store so **100% of conversations are saved**
automatically even when `ai_engine` runs completely standalone outside any project!

Implements the exact same chat-store protocol as `StorageEngine`, allowing `AIEngine`
to switch seamlessly between its built-in archive and a project's `StorageEngine`.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from ..settings import DEFAULT_CHAT_DB_PATH


class LocalConversationArchive:
    """Standalone WAL-mode SQLite archive for chat sessions and messages."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = (
            Path(db_path).resolve() if db_path else DEFAULT_CHAT_DB_PATH.resolve()
        )
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id       INTEGER,
                    mode          TEXT NOT NULL DEFAULT 'general',
                    title         TEXT NOT NULL,
                    model_name    TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id    INTEGER NOT NULL,
                    role          TEXT NOT NULL,
                    content       TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );
                """
            )
            conn.commit()

    def create_chat_session(
        self,
        mode: str = "general",
        title: str = "New Chat",
        model_name: str | None = None,
        file_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        clean_title = (title or "New Chat").strip()
        if len(clean_title) > 60:
            clean_title = clean_title[:57] + "..."
        now_iso = datetime.now().isoformat()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO chat_sessions (
                    file_id, mode, title, model_name, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    mode,
                    clean_title,
                    model_name,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def save_chat_message(
        self,
        session_id: int,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now_iso = datetime.now().isoformat()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO chat_messages (
                    session_id, role, content, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now_iso,
                ),
            )
            msg_id = int(cur.lastrowid)
            cur.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now_iso, session_id),
            )
            conn.commit()
            return msg_id

    def get_chat_sessions(
        self, mode: str | None = None, file_id: int | None = None
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if mode:
            clauses.append("mode = ?")
            params.append(mode)
        if file_id is not None:
            clauses.append("file_id = ?")
            params.append(file_id)

        sql = "SELECT * FROM chat_sessions"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC, id DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def get_chat_messages(self, session_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, created_at AS timestamp
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_chat_session(self, session_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cur.rowcount > 0

    def get_chat_stats(self) -> dict[str, int]:
        with self._connect() as conn:
            s_cnt = int(conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0])
            m_cnt = int(conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0])
            return {"total_sessions": s_cnt, "total_messages": m_cnt}
