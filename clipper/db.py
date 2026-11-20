"""SQLite state shared by the agents (WAL mode, so they can run as separate processes).

moments -> clips -> posts is the pipeline:
  * the scout inserts ``moments`` (chat spikes),
  * the editor turns a moment into a rendered ``clip``,
  * the publisher fans a clip out into one ``post`` per platform.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS moments (
    id                TEXT PRIMARY KEY,
    platform          TEXT NOT NULL,          -- twitch | youtube | other
    channel           TEXT NOT NULL,          -- twitch login / youtube channel name
    channel_display   TEXT,
    channel_id        TEXT,                   -- twitch user id / youtube channel id
    video_id          TEXT,                   -- youtube video id (live and replay share it)
    stream_title      TEXT,
    stream_started_at REAL,                   -- epoch seconds, when known
    onset_at          REAL,                   -- epoch seconds the chat burst began (live)
    onset_offset      REAL,                   -- seconds into the VOD the chat burst began
    source_url        TEXT,                   -- resolved VOD url
    peak_mpm          INTEGER,
    baseline_mpm      REAL,
    score             REAL,
    moment_type       TEXT,
    chat_sample       TEXT,                   -- JSON list of the most repeated messages
    viewer_count      INTEGER,
    status            TEXT NOT NULL DEFAULT 'new',  -- new|processing|edited|skipped|failed|expired
    errors            INTEGER NOT NULL DEFAULT 0,
    next_attempt_at   REAL NOT NULL DEFAULT 0,
    note              TEXT,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS moments_status ON moments(status, next_attempt_at);

CREATE TABLE IF NOT EXISTS clips (
    id           TEXT PRIMARY KEY,
    moment_id    TEXT NOT NULL REFERENCES moments(id),
    path         TEXT NOT NULL,
    duration     REAL,
    source_url   TEXT,
    transcript   TEXT,
    titles       TEXT,       -- JSON {quote, curiosity, reaction}
    hook         TEXT,
    description  TEXT,
    hashtags     TEXT,       -- JSON list
    format       TEXT,       -- JSON {layout, captions, length, hook}
    title_source TEXT,       -- claude | template
    status       TEXT NOT NULL DEFAULT 'ready',  -- ready|posting|posted|failed
    note         TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS clips_status ON clips(status, created_at);

CREATE TABLE IF NOT EXISTS posts (
    id             TEXT PRIMARY KEY,
    clip_id        TEXT NOT NULL REFERENCES clips(id),
    platform       TEXT NOT NULL,      -- tiktok | youtube | instagram
    account_id     TEXT NOT NULL,      -- Postiz integration id
    account_name   TEXT,
    postiz_post_id TEXT,
    title          TEXT,
    title_style    TEXT,
    scheduled_at   REAL NOT NULL,
    state          TEXT,               -- QUEUE | PUBLISHED | ERROR | DRAFT
    release_url    TEXT,
    views          INTEGER,
    likes          INTEGER,
    comments       INTEGER,
    shares         INTEGER,
    metrics_source TEXT,               -- postiz | sheet
    metrics_at     REAL,
    hit            INTEGER,            -- NULL until judged
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS posts_account ON posts(account_id, scheduled_at);

CREATE TABLE IF NOT EXISTS kv (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
"""


class DB:
    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), timeout=30, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    # -- generic helpers -------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, tuple(params))

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, tuple(params)).fetchall()

    def one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, tuple(params)).fetchone()

    def insert(self, table: str, row: dict[str, Any], ignore_existing: bool = False) -> bool:
        now = time.time()
        row = {"created_at": now, "updated_at": now, **row}
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        verb = "INSERT OR IGNORE" if ignore_existing else "INSERT"
        cur = self.execute(f"{verb} INTO {table} ({cols}) VALUES ({marks})", row.values())
        return cur.rowcount > 0

    def update(self, table: str, row_id: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*fields.values(), row_id])

    def claim(self, table: str, row_id: str, from_status: str, to_status: str) -> bool:
        """Atomically move a row between statuses; False if another worker won."""
        cur = self.execute(
            f"UPDATE {table} SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
            (to_status, time.time(), row_id, from_status),
        )
        return cur.rowcount == 1

    def requeue_stale(self, table: str, stuck_status: str, back_to: str, older_than_s: float) -> int:
        """Rows left mid-flight by a crashed worker go back in the queue."""
        cur = self.execute(
            f"UPDATE {table} SET status = ?, updated_at = ? WHERE status = ? AND updated_at < ?",
            (back_to, time.time(), stuck_status, time.time() - older_than_s),
        )
        return cur.rowcount

    # -- key/value (bandit state, priorities, cursors) --------------------
    def kv_get(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT v FROM kv WHERE k = ?", (key,))
        return json.loads(row["v"]) if row else default

    def kv_set(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            (key, json.dumps(value)),
        )

    def counts(self, table: str) -> dict[str, int]:
        return {r["status"]: r["n"] for r in self.query(f"SELECT status, COUNT(*) AS n FROM {table} GROUP BY status")}
