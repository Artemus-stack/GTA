"""YouTube: find the top GTA 6 live streams, and read live or replay chat through yt-dlp.

Chat comes from yt-dlp's own ``live_chat`` "subtitle" downloader, so YouTube
changes get fixed upstream in yt-dlp rather than here. yt-dlp writes one JSON
object per line:

* replay lines: ``{"replayChatItemAction": {"actions": [...], "videoOffsetTimeMsec": "123"}}``
* live lines:   ``{"replayChatItemAction": {"actions": [...]}, "videoOffsetTimeMsec": "...", "isLive": true}``

Each text message carries ``timestampUsec`` (epoch). Replay lines also carry
the exact offset into the VOD, which is what replay scans use.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote_plus

import httpx

log = logging.getLogger("scout.youtube")

API = "https://www.googleapis.com/youtube/v3"
# sp=CAMSAkAB = "sort by view count" + "live now" (base64 protobuf search filter)
LIVE_SEARCH_URL = "https://www.youtube.com/results?search_query={q}&sp=CAMSAkAB"


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _mentions(text: str, keywords: Iterable[str]) -> bool:
    low = (text or "").lower()
    return any(k.lower() in low for k in keywords)


class YouTubeDiscovery:
    def __init__(self, query: str, keywords: list[str], api_key: str = "", cookies_file: str | None = None,
                 http: httpx.Client | None = None):
        self.query = query
        self.keywords = keywords
        self.api_key = api_key
        self.cookies_file = cookies_file
        self.http = http or httpx.Client(timeout=30)

    def top_live(self, limit: int = 100) -> list[dict]:
        streams = self._via_api(limit) if self.api_key else self._via_ytdlp(limit)
        streams.sort(key=lambda s: s["viewers"], reverse=True)
        return streams[:limit]

    def _via_api(self, limit: int) -> list[dict]:
        """search.list costs 100 quota units per page; keep refresh_youtube_s generous."""
        ids: list[str] = []
        page = None
        while len(ids) < limit * 2:
            params = {"part": "id", "type": "video", "eventType": "live", "q": self.query,
                      "order": "viewCount", "maxResults": 50, "key": self.api_key}
            if page:
                params["pageToken"] = page
            r = self.http.get(f"{API}/search", params=params)
            r.raise_for_status()
            data = r.json()
            ids += [it["id"]["videoId"] for it in data.get("items", []) if it.get("id", {}).get("videoId")]
            page = data.get("nextPageToken")
            if not page:
                break
        out = []
        for i in range(0, len(ids), 50):
            r = self.http.get(f"{API}/videos", params={
                "part": "snippet,liveStreamingDetails", "id": ",".join(ids[i:i + 50]), "key": self.api_key})
            r.raise_for_status()
            for v in r.json().get("items", []):
                sn, live = v.get("snippet", {}), v.get("liveStreamingDetails", {})
                if not _mentions(sn.get("title", "") + " " + sn.get("description", ""), self.keywords):
                    continue
                started = live.get("actualStartTime")
                out.append({
                    "video_id": v["id"],
                    "channel": sn.get("channelTitle", ""),
                    "channel_id": sn.get("channelId", ""),
                    "title": sn.get("title", ""),
                    "viewers": int(live.get("concurrentViewers") or 0),
                    "started_at": _iso_to_epoch(started) if started else None,
                })
        return out

    def _via_ytdlp(self, limit: int) -> list[dict]:
        """No API key: read YouTube's own "live, by views" search page through yt-dlp."""
        import yt_dlp

        opts = {"extract_flat": "in_playlist", "playlistend": limit * 2, "quiet": True,
                "no_warnings": True, "skip_download": True}
        if self.cookies_file:
            opts["cookiefile"] = self.cookies_file
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(LIVE_SEARCH_URL.format(q=quote_plus(self.query)), download=False)
        out = []
        for e in (info or {}).get("entries") or []:
            if not e or e.get("live_status") != "is_live":
                continue
            if not _mentions(e.get("title", "") + " " + (e.get("description") or ""), self.keywords):
                continue
            out.append({
                "video_id": e["id"],
                "channel": e.get("channel") or e.get("uploader") or "",
                "channel_id": e.get("channel_id") or "",
                "title": e.get("title", ""),
                "viewers": int(e.get("concurrent_view_count") or 0),
                "started_at": None,
            })
        return out


def _iso_to_epoch(value: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


# ---------------------------------------------------------------------------
# chat parsing
# ---------------------------------------------------------------------------

def _runs_text(runs: list[dict]) -> str:
    parts = []
    for run in runs or []:
        if "text" in run:
            parts.append(run["text"])
        elif "emoji" in run:
            emoji = run["emoji"]
            eid = emoji.get("emojiId", "")
            # standard emoji use the character itself as id; custom ones use :shortcuts:
            parts.append(eid if len(eid) <= 4 else (emoji.get("shortcuts") or [""])[0])
    return "".join(parts)


def parse_chat_line(line: bytes | str) -> list[tuple[float | None, float | None, str]]:
    """One yt-dlp live_chat line -> [(epoch_ts, vod_offset_s, text), ...] for text messages."""
    try:
        obj = json.loads(line)
    except (ValueError, TypeError):
        return []
    action = obj.get("replayChatItemAction") or {}
    offset_ms = action.get("videoOffsetTimeMsec")
    if obj.get("isLive"):
        offset_ms = None   # live offsets are relative to when yt-dlp started, not the stream
    out = []
    for act in action.get("actions") or []:
        item = (act.get("addChatItemAction") or {}).get("item") or {}
        renderer = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
        if not renderer:
            continue
        text = _runs_text((renderer.get("message") or {}).get("runs", []))
        usec = renderer.get("timestampUsec")
        out.append((
            int(usec) / 1e6 if usec else None,
            int(offset_ms) / 1000.0 if offset_ms is not None else None,
            text,
        ))
    return out


def _chat_opts(base: Path, cookies_file: str | None) -> dict:
    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],
        "outtmpl": {"default": str(base)},
        "nopart": True,          # write straight to the final file so it can be tailed
        "overwrites": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    return opts


def download_replay_chat(video_id: str, workdir: Path, cookies_file: str | None = None) -> tuple[Path, dict]:
    """Download a finished stream's whole chat replay; returns (.live_chat.json path, video info)."""
    import yt_dlp

    workdir.mkdir(parents=True, exist_ok=True)
    base = workdir / video_id
    with yt_dlp.YoutubeDL(_chat_opts(base, cookies_file)) as ydl:
        info = ydl.extract_info(watch_url(video_id), download=False)
        if info.get("live_status") == "is_live":
            raise RuntimeError("stream is still live; the scout reads live chat, scan the replay once it ends")
        if "live_chat" not in (info.get("subtitles") or {}):
            raise RuntimeError("this video has no chat replay")
        ydl.process_ie_result(info, download=True)
    path = Path(f"{base}.live_chat.json")
    if not path.exists():
        raise RuntimeError("yt-dlp did not produce a chat replay file")
    return path, info


# ---------------------------------------------------------------------------
# live chat readers
# ---------------------------------------------------------------------------

class _Stopped(Exception):
    pass


class _ChatReader(threading.Thread):
    def __init__(self, video_id: str, path_base: Path, cookies_file: str | None):
        super().__init__(daemon=True, name=f"ytchat-{video_id}")
        self.video_id = video_id
        self.base = path_base
        self.path = Path(f"{path_base}.live_chat.json")
        self.cookies_file = cookies_file
        self.stop_event = threading.Event()
        self.started_at = time.time()
        self.offset = 0
        self.buffer = b""
        self.error: str | None = None

    def _hook(self, _status: dict) -> None:
        if self.stop_event.is_set():
            import yt_dlp

            raise yt_dlp.utils.DownloadCancelled("chat reader stopped")

    def run(self) -> None:
        import yt_dlp

        opts = _chat_opts(self.base, self.cookies_file)
        opts["progress_hooks"] = [self._hook]
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([watch_url(self.video_id)])
        except yt_dlp.utils.DownloadCancelled:
            pass
        except Exception as e:  # noqa: BLE001 - keep the pool alive whatever yt-dlp raises
            self.error = str(e)
            log.warning("YouTube chat %s stopped: %s", self.video_id, e)

    def read_new(self) -> list[bytes]:
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                chunk = f.read()
        except FileNotFoundError:
            return []
        self.offset += len(chunk)
        data = self.buffer + chunk
        lines = data.split(b"\n")
        self.buffer = lines.pop()       # keep a partial last line for next time
        return [ln for ln in lines if ln.strip()]


class YouTubeChatPool:
    """One yt-dlp chat reader per live video; tails their files into a callback."""

    def __init__(self, workdir: Path, on_message: Callable[[str, float, str], None],
                 cookies_file: str | None = None, max_file_mb: int = 64,
                 is_busy: Callable[[str], bool] = lambda key: False):
        self.workdir = workdir
        self.on_message = on_message
        self.cookies_file = cookies_file
        self.max_bytes = max_file_mb * 1024 * 1024
        self.is_busy = is_busy
        self.readers: dict[str, _ChatReader] = {}
        self.keys: dict[str, str] = {}
        self._lock = threading.RLock()
        workdir.mkdir(parents=True, exist_ok=True)

    def set_videos(self, videos: dict[str, str]) -> None:
        """videos: video_id -> tracker key. Starts new readers, stops dropped ones."""
        with self._lock:
            self._set_videos(videos)

    def _set_videos(self, videos: dict[str, str]) -> None:
        for vid in list(self.readers):
            if vid not in videos:
                self._stop(vid)
        for vid, key in videos.items():
            self.keys[vid] = key
            reader = self.readers.get(vid)
            if reader is None or (not reader.is_alive() and reader.error is not None):
                self._start(vid)

    def _start(self, vid: str) -> None:
        reader = _ChatReader(vid, self.workdir / vid, self.cookies_file)
        self.readers[vid] = reader
        reader.start()

    def _stop(self, vid: str, wait: float = 0.0) -> None:
        reader = self.readers.pop(vid, None)
        if reader:
            reader.stop_event.set()
            if wait:
                reader.join(wait)
            try:
                reader.path.unlink()
            except OSError:
                pass

    def poll(self) -> int:
        """Feed new chat lines to the callback; rotate files that grew too big."""
        with self._lock:
            return self._poll()

    def _poll(self) -> int:
        n = 0
        for vid, reader in list(self.readers.items()):
            key = self.keys.get(vid, vid)
            for line in reader.read_new():
                for ts, _offset, text in parse_chat_line(line):
                    self.on_message(key, ts if ts is not None else time.time(), text)
                    n += 1
            if not reader.is_alive() and reader.error and time.time() - reader.started_at > 300:
                log.info("restarting YouTube chat reader for %s", vid)
                self._start(vid)
            elif reader.offset > self.max_bytes and not self.is_busy(key):
                log.info("rotating chat file for %s (%.0f MB)", vid, reader.offset / 2**20)
                self._stop(vid, wait=15)
                self._start(vid)
        return n

    def stop_all(self) -> None:
        for vid in list(self.readers):
            self._stop(vid)

    @property
    def active(self) -> int:
        return sum(1 for r in self.readers.values() if r.is_alive())


def iter_replay_messages(path: Path) -> Iterable[tuple[float, str]]:
    """(vod_offset_s, text) for every message in a downloaded replay chat file."""
    with open(path, "rb") as f:
        for line in f:
            for _ts, offset, text in parse_chat_line(line):
                if offset is not None:
                    yield offset, text
