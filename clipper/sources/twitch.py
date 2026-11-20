"""Twitch: Helix API (top streams, VOD lookup) and anonymous IRC chat.

Helix needs an app (client-credentials) token from a Twitch developer app.
Reading chat over IRC needs no login at all: anonymous ``justinfanNNNN``
connections are Twitch's documented read-only access.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import ssl
import time
from datetime import datetime
from typing import Callable, Iterable

import httpx

log = logging.getLogger("scout.twitch")

HELIX = "https://api.twitch.tv/helix"
TOKEN_URL = "https://id.twitch.tv/oauth2/token"
IRC_HOST, IRC_PORT = "irc.chat.twitch.tv", 6697


def parse_ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def parse_duration(value: str) -> int:
    """Helix VOD durations look like '3h2m1s'."""
    total = 0
    for amount, unit in re.findall(r"(\d+)([hms])", value or ""):
        total += int(amount) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


class TwitchAPI:
    def __init__(self, client_id: str, client_secret: str, http: httpx.Client | None = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.http = http or httpx.Client(timeout=30)
        self._token: str | None = None
        self._token_exp = 0.0

    def _app_token(self) -> str:
        if self._token and time.time() < self._token_exp - 120:
            return self._token
        r = self.http.post(
            TOKEN_URL,
            data={"client_id": self.client_id, "client_secret": self.client_secret,
                  "grant_type": "client_credentials"},
        )
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]
        self._token_exp = time.time() + float(data.get("expires_in", 3600))
        return self._token

    def get(self, path: str, params: list[tuple[str, str]] | dict) -> dict:
        last: httpx.Response | None = None
        for attempt in range(5):
            r = self.http.get(
                HELIX + path, params=params,
                headers={"Client-Id": self.client_id, "Authorization": f"Bearer {self._app_token()}"},
            )
            last = r
            if r.status_code == 401:
                self._token = None
                continue
            if r.status_code == 429:
                reset = float(r.headers.get("Ratelimit-Reset", time.time() + 5))
                time.sleep(min(max(reset - time.time(), 1.0), 60.0))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        assert last is not None
        last.raise_for_status()
        return last.json()

    def game_ids(self, names: Iterable[str]) -> list[str]:
        names = [n for n in names if n]
        if not names:
            return []
        data = self.get("/games", [("name", n) for n in names])
        found = {g["name"].lower(): g["id"] for g in data.get("data", [])}
        missing = [n for n in names if n.lower() not in found]
        if missing:
            log.warning("Twitch category not found: %s", missing)
        return list(found.values())

    def top_streams(self, game_ids: list[str], limit: int = 100, languages: Iterable[str] = ()) -> list[dict]:
        """Live streams in these categories, most viewers first."""
        if not game_ids:
            return []
        streams: list[dict] = []
        cursor = None
        while len(streams) < limit:
            params = [("game_id", g) for g in game_ids] + [("first", "100"), ("type", "live")]
            params += [("language", lang) for lang in languages]
            if cursor:
                params.append(("after", cursor))
            data = self.get("/streams", params)
            for s in data.get("data", []):
                streams.append({
                    "user_id": s["user_id"],
                    "login": s["user_login"],
                    "name": s["user_name"],
                    "title": s.get("title", ""),
                    "viewers": int(s.get("viewer_count", 0)),
                    "started_at": parse_ts(s["started_at"]),
                    "language": s.get("language", ""),
                })
            cursor = data.get("pagination", {}).get("cursor")
            if not cursor or not data.get("data"):
                break
        streams.sort(key=lambda s: s["viewers"], reverse=True)
        return streams[:limit]

    def find_vod(self, user_id: str, at_epoch: float) -> dict | None:
        """The archive VOD covering ``at_epoch`` (works while the stream is still live)."""
        data = self.get("/videos", {"user_id": user_id, "type": "archive", "first": "20"})
        for v in data.get("data", []):
            created = parse_ts(v["created_at"])
            duration = parse_duration(v.get("duration", ""))
            # an in-progress VOD's duration lags behind real time, so allow slack
            if created - 60 <= at_epoch <= created + duration + 900:
                return {"id": v["id"], "url": v["url"], "created_at": created, "duration": duration}
        return None


# ---------------------------------------------------------------------------
# IRC
# ---------------------------------------------------------------------------

def parse_irc(line: str) -> tuple[dict[str, str], str | None, str, list[str]]:
    """Parse one IRCv3 line into (tags, prefix, command, params)."""
    tags: dict[str, str] = {}
    if line.startswith("@"):
        raw, _, line = line[1:].partition(" ")
        for item in raw.split(";"):
            key, _, value = item.partition("=")
            tags[key] = value
    prefix = None
    if line.startswith(":"):
        prefix, _, line = line[1:].partition(" ")
    if " :" in line:
        head, _, trailing = line.partition(" :")
        params = head.split() + [trailing]
    else:
        params = line.split()
    command = params.pop(0) if params else ""
    return tags, prefix, command, params


class _JoinLimiter:
    """Twitch allows 20 JOINs per 10 s; stay under it across connections."""

    def __init__(self, per_10s: int = 15):
        self.per_10s = per_10s
        self.sent: list[float] = []

    async def wait(self) -> None:
        while True:
            now = time.monotonic()
            self.sent = [t for t in self.sent if now - t < 10]
            if len(self.sent) < self.per_10s:
                self.sent.append(now)
                return
            await asyncio.sleep(10 - (now - self.sent[0]) + 0.05)


class _Connection:
    def __init__(self, idx: int, on_message: Callable[[str, float, str], None], limiter: _JoinLimiter):
        self.idx = idx
        self.on_message = on_message
        self.limiter = limiter
        self.wanted: set[str] = set()
        self.joined: set[str] = set()

    async def run(self, stop: asyncio.Event) -> None:
        backoff = 1.0
        while not stop.is_set():
            writer = None
            try:
                ctx = ssl.create_default_context()
                reader, writer = await asyncio.open_connection(IRC_HOST, IRC_PORT, ssl=ctx, limit=2**20)
                nick = f"justinfan{random.randint(10000, 99999)}"
                writer.write(f"CAP REQ :twitch.tv/tags twitch.tv/commands\r\nPASS SCHMOOPIIE\r\nNICK {nick}\r\n".encode())
                await writer.drain()
                self.joined = set()
                sync = asyncio.create_task(self._sync(writer, stop))
                try:
                    await self._read(reader, writer, stop)
                finally:
                    sync.cancel()
                backoff = 1.0
            except (OSError, asyncio.IncompleteReadError, ssl.SSLError) as e:
                log.warning("IRC connection %d dropped: %s", self.idx, e)
            finally:
                if writer is not None:
                    writer.close()
            if not stop.is_set():
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _read(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                raw = await asyncio.wait_for(reader.readline(), timeout=360)
            except asyncio.TimeoutError:
                log.warning("IRC connection %d idle for 6 min, reconnecting", self.idx)
                return
            if not raw:
                return
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("PING"):
                writer.write(("PONG" + line[4:] + "\r\n").encode())
                await writer.drain()
                continue
            tags, _, command, params = parse_irc(line)
            if command == "PRIVMSG" and len(params) >= 2:
                channel = params[0].lstrip("#").lower()
                # arrival time keeps every tracker on the local clock (no skew issues)
                self.on_message(channel, time.time(), params[1])
            elif command == "RECONNECT":
                log.info("IRC connection %d asked to reconnect", self.idx)
                return

    async def _sync(self, writer: asyncio.StreamWriter, stop: asyncio.Event) -> None:
        while not stop.is_set():
            for ch in sorted(self.joined - self.wanted):
                writer.write(f"PART #{ch}\r\n".encode())
                self.joined.discard(ch)
            for ch in sorted(self.wanted - self.joined):
                await self.limiter.wait()
                if ch not in self.wanted:
                    continue
                writer.write(f"JOIN #{ch}\r\n".encode())
                self.joined.add(ch)
            await writer.drain()
            await asyncio.sleep(2)


class TwitchChat:
    """Reads chat for a changing set of channels, sharded over IRC connections."""

    def __init__(self, on_message: Callable[[str, float, str], None], per_connection: int = 50):
        self.on_message = on_message
        self.per_connection = per_connection
        self.limiter = _JoinLimiter()
        self.connections: list[_Connection] = []
        self._tasks: list[asyncio.Task] = []
        self._stop: asyncio.Event | None = None

    def set_channels(self, channels: Iterable[str]) -> None:
        wanted = {c.lower() for c in channels}
        for conn in self.connections:
            conn.wanted &= wanted
        placed = set().union(*(c.wanted for c in self.connections)) if self.connections else set()
        for ch in sorted(wanted - placed):
            conn = next((c for c in self.connections if len(c.wanted) < self.per_connection), None)
            if conn is None:
                conn = _Connection(len(self.connections), self.on_message, self.limiter)
                self.connections.append(conn)
                if self._stop is not None:
                    self._tasks.append(asyncio.ensure_future(conn.run(self._stop)))
            conn.wanted.add(ch)

    @property
    def channels(self) -> set[str]:
        return set().union(*(c.wanted for c in self.connections)) if self.connections else set()

    async def run(self, stop: asyncio.Event) -> None:
        self._stop = stop
        self._tasks = [asyncio.ensure_future(c.run(stop)) for c in self.connections]
        await stop.wait()
        for t in self._tasks:
            t.cancel()
