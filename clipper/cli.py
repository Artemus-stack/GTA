"""Command line entry point: ``clipper <command>``.

  run          all agents in one process (scout + editor + publisher + analyst)
  scout        agent 1 only     edit     agent 2 only
  publish      agent 3 only     analyze  the learning loop only
  scan URL     find chat spikes in a finished YouTube stream and queue them
  clip URL --at 1:23:45   clip a specific moment from any VOD (manual pick)
  accounts     list the Postiz accounts the publisher will rotate through
  sheet-init   create the Google Sheet tabs
  status       queue counts and the formats currently winning
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path

from .config import Config, ConfigError, load_config
from .db import DB
from .formats import FormatBandit
from .postiz import Postiz
from .sheets import open_sheet

log = logging.getLogger("clipper")


def parse_time(value: str) -> float:
    """'1:23:45', '83:45', '5025' or '1h23m45s' -> seconds."""
    value = value.strip()
    if re.fullmatch(r"(\d+h)?(\d+m)?(\d+(\.\d+)?s)?", value) and value:
        total = 0.0
        for amount, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", value):
            total += float(amount) * {"h": 3600, "m": 60, "s": 1}[unit]
        return total
    parts = [float(p) for p in value.split(":")]
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = DB(cfg.db_path)
        self.bandit = FormatBandit(self.db, cfg.analyst.arms)
        key = cfg.secrets.postiz_api_key
        self.postiz = Postiz(cfg.publisher.postiz_url, key, cfg.publisher.posts_per_hour_limit) if key else None
        self._sheet = None

    @property
    def sheet(self):
        if self._sheet is None:
            self._sheet = open_sheet(self.cfg.sheets.spreadsheet_id, self.cfg.sheets.credentials_file)
        return self._sheet

    def scout(self):
        from .agents.scout import Scout
        return Scout(self.cfg, self.db)

    def editor(self):
        from .agents.editor import Editor
        return Editor(self.cfg, self.db, self.bandit)

    def publisher(self):
        from .agents.publisher import Publisher
        return Publisher(self.cfg, self.db, self.bandit, self.postiz, self.sheet)

    def analyst(self):
        from .agents.analyst import Analyst
        return Analyst(self.cfg, self.db, self.bandit, self.postiz, self.sheet)


def _run_async_scout(app: App, stop_threads: threading.Event) -> None:
    async def main():
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):
                pass

        async def watch_threads():
            while not stop.is_set():
                if stop_threads.is_set():
                    stop.set()
                await asyncio.sleep(1)

        await asyncio.gather(app.scout().run(stop), watch_threads())

    try:
        asyncio.run(main())
    finally:
        stop_threads.set()


def _threads(targets: list) -> tuple[threading.Event, list[threading.Thread]]:
    stop = threading.Event()
    threads = [threading.Thread(target=t.run_forever, args=(stop,), name=type(t).__name__.lower(), daemon=True)
               for t in targets]
    for t in threads:
        t.start()
    return stop, threads


def _wait(stop: threading.Event, threads: list[threading.Thread]) -> None:
    try:
        while any(t.is_alive() for t in threads) and not stop.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    stop.set()
    for t in threads:
        t.join(timeout=30)


def cmd_run(app: App, args) -> int:
    stop, threads = _threads([app.editor(), app.publisher(), app.analyst()])
    _run_async_scout(app, stop)
    for t in threads:
        t.join(timeout=30)
    return 0


def cmd_scout(app: App, args) -> int:
    _run_async_scout(app, threading.Event())
    return 0


def _single(app: App, agent, once: bool) -> int:
    if once:
        if hasattr(agent, "run_once"):
            print(json.dumps(agent.run_once(), indent=2))
        else:
            while agent.process_next():
                pass
        return 0
    stop, threads = _threads([agent])
    _wait(stop, threads)
    return 0


def cmd_edit(app: App, args) -> int:
    return _single(app, app.editor(), args.once)


def cmd_publish(app: App, args) -> int:
    pub = app.publisher()
    if args.dry_run:
        for item in pub.plan():
            print(json.dumps(item, ensure_ascii=False, indent=2))
        return 0
    if app.postiz is None:
        print("POSTIZ_API_KEY is not set", file=sys.stderr)
        return 2
    return _single(app, pub, args.once)


def cmd_analyze(app: App, args) -> int:
    return _single(app, app.analyst(), args.once)


def cmd_scan(app: App, args) -> int:
    from .agents.scout import scan_youtube_replay

    rows = scan_youtube_replay(app.cfg, app.db, args.video)
    for r in rows:
        print(f"{r['id']}: {r['onset_offset']:.0f}s in, {r['peak_mpm']} msg/min ({r['moment_type']})")
    print(f"queued {len(rows)} moment(s)")
    return 0


def cmd_clip(app: App, args) -> int:
    from .media.download import probe

    at = parse_time(args.at)
    info = probe(args.url, app.cfg.editor.cookies_file)
    extractor = (info.get("extractor_key") or "").lower()
    platform = "twitch" if "twitch" in extractor else "youtube" if "youtube" in extractor else "other"
    channel = args.streamer or info.get("uploader_id") or info.get("channel") or info.get("uploader") or "unknown"
    moment_id = f"manual-{info.get('id', 'video')}-{int(at)}"
    app.db.insert("moments", {
        "id": moment_id, "platform": platform, "channel": channel.lower() if platform == "twitch" else channel,
        "channel_display": args.streamer or info.get("uploader") or channel, "video_id": info.get("id"),
        "stream_title": info.get("title"), "source_url": args.url,
        # --at is the moment itself; the editor expects the chat onset, which lags it
        "onset_offset": at + app.cfg.scout.reaction_lag_s,
        "peak_mpm": 0, "score": 1e9, "moment_type": args.type, "chat_sample": "[]",
    }, ignore_existing=True)
    print(f"queued {moment_id}")
    if args.queue_only:
        return 0
    editor = app.editor()
    if not editor.process_next():
        print("nothing to process", file=sys.stderr)
        return 1
    row = app.db.one("SELECT status, note FROM moments WHERE id = ?", (moment_id,))
    print(f"{moment_id}: {row['status']} {row['note'] or ''}")
    clip = app.db.one("SELECT * FROM clips WHERE moment_id = ?", (moment_id,))
    if clip:
        print(f"clip: {clip['path']}\ntitles: {clip['titles']}\nhook: {clip['hook']}\nformat: {clip['format']}")
    return 0 if clip else 1


def cmd_accounts(app: App, args) -> int:
    if app.postiz is None:
        print("POSTIZ_API_KEY is not set", file=sys.stderr)
        return 2
    for platform, accounts in app.publisher().accounts(refresh=True).items():
        names = ", ".join(f"{a['name']} ({a['identifier']}, id {a['id']})" for a in accounts) or "none connected"
        print(f"{platform:10s} {names}")
    return 0


def cmd_sheet_init(app: App, args) -> int:
    if not app.sheet.enabled:
        print("Set GSHEET_ID and GOOGLE_APPLICATION_CREDENTIALS (see README)", file=sys.stderr)
        return 2
    app.sheet.ensure_tabs()
    print("Google Sheet tabs ready: Posts, Formats, Streamers")
    return 0


def cmd_status(app: App, args) -> int:
    db = app.db
    print("moments:", db.counts("moments") or {})
    print("clips:  ", db.counts("clips") or {})
    posts = {r["platform"]: r["n"] for r in db.query("SELECT platform, COUNT(*) AS n FROM posts GROUP BY platform")}
    print("posts:  ", posts)
    print("\nwhat's winning (chance each option is best):")
    for row in app.bandit.report(app.cfg.publisher.platforms):
        if row[3]:
            print(f"  {row[0]:13s} {row[1]:12s} {row[2]:10s} posts={row[3]:<4} hits={row[4]:<4} "
                  f"best={row[7]:>4}  {row[8]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clipper", description="GTA 6 stream clipper: scout -> editor -> publisher")
    default_cfg = os.environ.get("CLIPPER_CONFIG") or ("config.yaml" if Path("config.yaml").is_file() else None)
    parser.add_argument("-c", "--config", default=default_cfg, help="YAML config (default: ./config.yaml)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="run every agent in one process")
    sub.add_parser("scout", help="agent 1: watch top streams, detect chat spikes")
    p = sub.add_parser("edit", help="agent 2: cut, caption and title queued moments")
    p.add_argument("--once", action="store_true", help="drain the queue and exit")
    p = sub.add_parser("publish", help="agent 3: post ready clips through Postiz")
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="show what would be posted")
    p = sub.add_parser("analyze", help="pull metrics, update the sheet and the format bandit")
    p.add_argument("--once", action="store_true")
    p = sub.add_parser("scan", help="queue chat spikes from a finished YouTube stream")
    p.add_argument("video", help="YouTube URL or video id")
    p = sub.add_parser("clip", help="clip a specific moment of a VOD")
    p.add_argument("url")
    p.add_argument("--at", required=True, help="when the moment happens, e.g. 1:23:45")
    p.add_argument("--streamer", help="streamer name to credit")
    p.add_argument("--type", default="general", choices=["funny", "hype", "shock", "fail", "general"])
    p.add_argument("--queue-only", action="store_true", help="queue it for the editor instead of cutting now")
    sub.add_parser("accounts", help="list connected Postiz accounts per platform")
    sub.add_parser("sheet-init", help="create the Google Sheet tabs")
    sub.add_parser("status", help="queue counts and winning formats")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "gspread"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    app = App(cfg)
    handler = {
        "run": cmd_run, "scout": cmd_scout, "edit": cmd_edit, "publish": cmd_publish, "analyze": cmd_analyze,
        "scan": cmd_scan, "clip": cmd_clip, "accounts": cmd_accounts, "sheet-init": cmd_sheet_init,
        "status": cmd_status,
    }[args.command]
    return handler(app, args)


if __name__ == "__main__":
    sys.exit(main())
