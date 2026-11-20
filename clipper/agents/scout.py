"""Agent 1 - Scout.

Keeps the top live GTA 6 streams on Twitch and YouTube under watch, reads every
chat in real time, and queues a "moment" whenever chat explodes past the
messages-per-minute threshold.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from ..config import Config
from ..db import DB
from ..sources.twitch import TwitchAPI, TwitchChat
from ..sources.youtube import (
    YouTubeChatPool,
    YouTubeDiscovery,
    download_replay_chat,
    iter_replay_messages,
    watch_url,
)
from ..velocity import ChannelTrackers, Moment, VelocityTracker

log = logging.getLogger("scout")


def make_tracker(cfg: Config, grace: float) -> VelocityTracker:
    s = cfg.scout
    return VelocityTracker(
        threshold=s.spike_threshold_mpm,
        baseline_window=s.baseline_window_s,
        baseline_multiplier=s.baseline_multiplier,
        warmup=s.warmup_s,
        cooldown=s.cooldown_s,
        max_spike=s.max_spike_s,
        grace=grace,
    )


def moment_row(key: str, m: Moment, info: dict) -> dict:
    platform, ident = key.split(":", 1)
    common = {
        "onset_at": m.onset,
        "peak_mpm": m.peak_mpm,
        "baseline_mpm": m.baseline_mpm,
        "score": m.score,
        "moment_type": m.moment_type,
        "chat_sample": json.dumps(m.sample, ensure_ascii=False),
        "viewer_count": info.get("viewers"),
        "stream_title": info.get("title"),
        "stream_started_at": info.get("started_at"),
    }
    if platform == "twitch":
        return {"id": f"tw-{info['login']}-{int(m.onset)}", "platform": "twitch", "channel": info["login"],
                "channel_display": info.get("name"), "channel_id": info.get("user_id"), **common}
    return {"id": f"yt-{ident}-{int(m.onset)}", "platform": "youtube", "channel": info.get("channel") or ident,
            "channel_display": info.get("channel"), "channel_id": info.get("channel_id"), "video_id": ident,
            **common}


class Scout:
    def __init__(self, cfg: Config, db: DB):
        self.cfg = cfg
        self.db = db
        self.streams: dict[str, dict] = {}
        self.trackers = ChannelTrackers(
            lambda key: make_tracker(cfg, grace=8.0 if key.startswith("youtube:") else 2.0))
        sec = cfg.secrets
        self.twitch = TwitchAPI(sec.twitch_client_id, sec.twitch_client_secret) \
            if sec.twitch_client_id and sec.twitch_client_secret else None
        self.chat = TwitchChat(self._on_twitch)
        self.yt = YouTubeDiscovery(cfg.game.youtube_query, cfg.game.keywords, sec.youtube_api_key,
                                   cfg.editor.cookies_file)
        self.ytchat = YouTubeChatPool(cfg.data_path / "chat", self._on_youtube, cfg.editor.cookies_file,
                                      cfg.scout.chat_file_max_mb, is_busy=self.trackers.in_spike)
        self._game_ids: list[str] | None = None
        self.moments_found = 0

    # -- chat callbacks ---------------------------------------------------
    def _on_twitch(self, channel: str, ts: float, text: str) -> None:
        self._emit(self.trackers.add(f"twitch:{channel}", ts, text))

    def _on_youtube(self, key: str, ts: float, text: str) -> None:
        self._emit(self.trackers.add(key, ts, text))

    def _emit(self, found: list[tuple[str, Moment]]) -> None:
        for key, m in found:
            info = self.streams.get(key)
            if info is None:
                continue
            row = moment_row(key, m, info)
            if self.db.insert("moments", row, ignore_existing=True):
                self.moments_found += 1
                log.info("MOMENT %s | %s | %d msg/min (normal %.0f) | %s | chat: %s",
                         row["id"], key, m.peak_mpm, m.baseline_mpm, m.moment_type, "; ".join(m.sample[:3]))

    def _drop(self, key: str) -> None:
        self._emit(self.trackers.drop(key))
        self.streams.pop(key, None)

    # -- discovery --------------------------------------------------------
    def refresh_twitch(self) -> list[dict]:
        if self._game_ids is None:
            self._game_ids = self.twitch.game_ids(self.cfg.game.twitch_categories)
            log.info("Twitch categories %s -> ids %s", self.cfg.game.twitch_categories, self._game_ids)
        streams = self.twitch.top_streams(self._game_ids, self.cfg.scout.twitch_top_n, self.cfg.scout.languages)
        allowed = self.cfg.scout.streamers
        return [s for s in streams if allowed.allowed(s["login"], s["name"])]

    def refresh_youtube(self) -> list[dict]:
        streams = self.yt.top_live(self.cfg.scout.youtube_top_n)
        allowed = self.cfg.scout.streamers
        return [s for s in streams if allowed.allowed(s["channel"], s["channel_id"])]

    def _apply(self, platform: str, streams: list[dict], keyfn) -> None:
        wanted = {keyfn(s): s for s in streams}
        for key in [k for k in self.streams if k.startswith(platform + ":") and k not in wanted]:
            self._drop(key)
        self.streams.update(wanted)

    async def _twitch_loop(self, stop: asyncio.Event) -> None:
        if not self.twitch:
            log.warning("TWITCH_CLIENT_ID/TWITCH_CLIENT_SECRET not set: Twitch is not monitored")
            return
        while not stop.is_set():
            try:
                streams = await asyncio.to_thread(self.refresh_twitch)
                self._apply("twitch", streams, lambda s: f"twitch:{s['login']}")
                self.chat.set_channels(s["login"] for s in streams)
                log.info("Twitch: watching %d live GTA 6 streams (top: %s)", len(streams),
                         ", ".join(f"{s['login']} {s['viewers']:,}" for s in streams[:5]))
            except Exception:  # noqa: BLE001 - discovery must keep retrying
                log.exception("Twitch discovery failed")
            await _sleep(stop, self.cfg.scout.refresh_twitch_s)

    async def _youtube_loop(self, stop: asyncio.Event) -> None:
        if self.cfg.scout.youtube_top_n <= 0:
            return
        while not stop.is_set():
            try:
                streams = await asyncio.to_thread(self.refresh_youtube)
                self._apply("youtube", streams, lambda s: f"youtube:{s['video_id']}")
                readers = streams[: self.cfg.scout.youtube_chat_readers]
                await asyncio.to_thread(self.ytchat.set_videos,
                                        {s["video_id"]: f"youtube:{s['video_id']}" for s in readers})
                log.info("YouTube: watching %d live GTA 6 streams, reading %d chats (top: %s)",
                         len(streams), len(readers),
                         ", ".join(f"{s['channel']} {s['viewers']:,}" for s in streams[:5]))
            except Exception:  # noqa: BLE001
                log.exception("YouTube discovery failed")
            await _sleep(stop, self.cfg.scout.refresh_youtube_s)

    async def _tick_loop(self, stop: asyncio.Event) -> None:
        last_report = time.time()
        while not stop.is_set():
            try:
                await asyncio.to_thread(self.ytchat.poll)
                self._emit(self.trackers.tick(time.time()))
            except Exception:  # noqa: BLE001
                log.exception("tick failed")
            if time.time() - last_report > 60:
                last_report = time.time()
                rates = sorted(self.trackers.rates().items(), key=lambda kv: kv[1], reverse=True)[:5]
                log.info("chat rates (msg/min): %s | moments so far: %d",
                         ", ".join(f"{k} {v}" for k, v in rates) or "-", self.moments_found)
            await _sleep(stop, 1.0)

    async def run(self, stop: asyncio.Event) -> None:
        log.info("scout started: threshold %d msg/min, top %d Twitch + %d YouTube streams",
                 self.cfg.scout.spike_threshold_mpm, self.cfg.scout.twitch_top_n, self.cfg.scout.youtube_top_n)
        try:
            await asyncio.gather(self.chat.run(stop), self._twitch_loop(stop),
                                 self._youtube_loop(stop), self._tick_loop(stop))
        finally:
            self.ytchat.stop_all()


async def _sleep(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


def scan_youtube_replay(cfg: Config, db: DB, video: str) -> list[dict]:
    """Find chat spikes in a finished YouTube stream's replay and queue them.

    Replay chat carries exact VOD offsets, so these moments need no time mapping.
    """
    video_id = _youtube_id(video)
    path, info = download_replay_chat(video_id, cfg.data_path / "replays", cfg.editor.cookies_file)
    tracker = make_tracker(cfg, grace=0)
    found: list[Moment] = []
    for offset, text in iter_replay_messages(path):
        found += tracker.add(offset, text)
    found += tracker.flush()
    channel = info.get("channel") or info.get("uploader") or video_id
    rows = []
    for m in found:
        row = {
            "id": f"yt-{video_id}-off{int(m.onset)}", "platform": "youtube", "channel": channel,
            "channel_display": channel, "channel_id": info.get("channel_id"), "video_id": video_id,
            "stream_title": info.get("title"), "onset_offset": m.onset, "source_url": watch_url(video_id),
            "peak_mpm": m.peak_mpm, "baseline_mpm": m.baseline_mpm, "score": m.score,
            "moment_type": m.moment_type, "chat_sample": json.dumps(m.sample, ensure_ascii=False),
        }
        if db.insert("moments", row, ignore_existing=True):
            rows.append(row)
    Path(path).unlink(missing_ok=True)
    return rows


def _youtube_id(video: str) -> str:
    import re

    m = re.search(r"(?:v=|youtu\.be/|/live/|/shorts/)([A-Za-z0-9_-]{11})", video)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", video):
        return video
    raise ValueError(f"not a YouTube video URL or id: {video}")
