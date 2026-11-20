"""Agent 3 - Publisher.

Takes each rendered clip and schedules it on TikTok, YouTube Shorts and
Instagram Reels through Postiz, rotating through every connected account on
each platform. Every post is logged to SQLite and to the Google Sheet.

Rotation rules:
  * a clip goes to one account per platform (the same video on several accounts
    of one platform reads as spam to the platforms),
  * the account with the earliest free slot gets it, so posting spreads evenly,
  * each account keeps a minimum gap between posts and a daily cap.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

from ..config import Config
from ..db import DB
from ..formats import FormatBandit
from ..postiz import PROVIDERS, Postiz, PostizError, settings_for, to_html
from ..sheets import POSTS_HEADER, NullSheet, Sheet

log = logging.getLogger("publisher")


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def sheet_time(ts: float | None) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if ts else ""


def sheet_row(values: dict) -> list:
    return ["" if values.get(h) is None else values.get(h) for h in POSTS_HEADER]


def credit_line(m) -> str:
    if m["platform"] == "twitch":
        return f"🎥 {m['channel_display'] or m['channel']} live on twitch.tv/{m['channel']}"
    if m["platform"] == "youtube":
        return f"🎥 {m['channel_display'] or m['channel']} on YouTube"
    return f"🎥 {m['channel_display'] or m['channel']}"


def hashtag_list(clip, cfg: Config) -> list[str]:
    tags = []
    for t in json.loads(clip["hashtags"] or "[]") + cfg.publisher.hashtags:
        t = t.lstrip("#").lower()
        if t and t not in tags:
            tags.append(t)
    return tags[:8]


def caption(platform: str, title: str, clip, moment, cfg: Config) -> str:
    tags = hashtag_list(clip, cfg)
    tag_line = " ".join(f"#{t}" for t in tags)
    credit = credit_line(moment) if cfg.publisher.credit_streamer else ""
    if platform == "youtube":       # the title is a separate field; this is the description
        parts = [clip["description"], credit, tag_line + " #shorts"]
    elif platform == "tiktok":
        parts = [title, credit, tag_line]
    else:
        parts = [title, clip["description"], credit, tag_line]
    text = "\n\n".join(p for p in parts if p)
    return text.replace("<", "").replace(">", "")


class Publisher:
    def __init__(self, cfg: Config, db: DB, bandit: FormatBandit, postiz: Postiz | None,
                 sheet: Sheet | NullSheet | None = None):
        self.cfg = cfg
        self.db = db
        self.bandit = bandit
        self.postiz = postiz
        self.sheet = sheet or NullSheet()
        self._accounts: dict[str, list[dict]] | None = None
        self._accounts_at = 0.0

    # -- accounts -------------------------------------------------------------
    def accounts(self, refresh: bool = False) -> dict[str, list[dict]]:
        if self._accounts is not None and not refresh and time.time() - self._accounts_at < 600:
            return self._accounts
        pc = self.cfg.publisher
        include = {a.lower() for a in pc.include_accounts}
        exclude = {a.lower() for a in pc.exclude_accounts}
        grouped: dict[str, list[dict]] = {p: [] for p in pc.platforms}
        for acc in self.postiz.integrations():
            platform = PROVIDERS.get(acc.get("identifier", ""))
            if platform not in grouped or acc.get("disabled"):
                continue
            keys = {str(acc.get("id", "")).lower(), str(acc.get("name", "")).lower()}
            if (include and not keys & include) or keys & exclude:
                continue
            grouped[platform].append(acc)
        self._accounts, self._accounts_at = grouped, time.time()
        return grouped

    def next_slot(self, account_id: str, now: float) -> float:
        pc = self.cfg.publisher
        times = sorted(r["scheduled_at"] for r in self.db.query(
            "SELECT scheduled_at FROM posts WHERE account_id = ? AND scheduled_at > ?", (account_id, now - 86400)))
        slot = now + 120
        if times:
            slot = max(slot, times[-1] + pc.min_gap_minutes * 60)
        in_day = [t for t in times if t > slot - 86400]
        if len(in_day) >= pc.max_posts_per_day:
            slot = max(slot, in_day[-pc.max_posts_per_day] + 86400)
        return slot

    def pick_account(self, platform: str, now: float) -> tuple[dict, float] | None:
        options = []
        for acc in self.accounts().get(platform, []):
            slot = self.next_slot(acc["id"], now)
            last = self.db.one("SELECT MAX(scheduled_at) AS t FROM posts WHERE account_id = ?", (acc["id"],))["t"] or 0
            options.append((slot, last, acc))
        if not options:
            return None
        slot, _, acc = min(options, key=lambda o: (o[0], o[1]))
        if slot - now > self.cfg.publisher.max_schedule_ahead_h * 3600:
            return None
        return acc, slot

    # -- queue ----------------------------------------------------------------
    def process_next(self) -> bool:
        clip = self.db.one("SELECT * FROM clips WHERE status = 'ready' ORDER BY created_at LIMIT 1")
        if clip is None:
            return False
        if not self.db.claim("clips", clip["id"], "ready", "posting"):
            return True
        try:
            result = self.publish(clip)
            if result["posted"]:
                self.db.update("clips", clip["id"], status="posted")
            elif result["errors"]:
                self.db.update("clips", clip["id"], status="failed", note="Postiz refused every post (see log)")
            else:
                self.db.update("clips", clip["id"], status="skipped",
                               note="every account is booked more than max_schedule_ahead_h out")
        except Exception as e:  # noqa: BLE001
            self.db.update("clips", clip["id"], status="failed", note=f"{type(e).__name__}: {e}"[:500])
            log.exception("publishing %s failed", clip["id"])
        return True

    def run_forever(self, stop: threading.Event) -> None:
        if self.postiz is None:
            log.warning("POSTIZ_API_KEY not set: publisher is idle (clips stay in the queue)")
            return
        self.db.requeue_stale("clips", "posting", "ready", 3600)
        log.info("publisher started; accounts: %s",
                 {p: [a["name"] for a in accs] for p, accs in self.accounts(refresh=True).items()})
        while not stop.is_set():
            if not self.process_next():
                stop.wait(10)

    def publish(self, clip) -> dict[str, int]:
        moment = self.db.one("SELECT * FROM moments WHERE id = ?", (clip["moment_id"],))
        titles = json.loads(clip["titles"])
        fmt = json.loads(clip["format"])
        already = {r["platform"] for r in self.db.query("SELECT platform FROM posts WHERE clip_id = ?", (clip["id"],))}
        media = None
        posted, errors, rows = 0, 0, []
        now = time.time()
        for platform in self.cfg.publisher.platforms:
            if platform in already:
                continue
            choice = self.pick_account(platform, now)
            if choice is None:
                log.warning("no %s account free for %s", platform, clip["id"])
                continue
            account, slot = choice
            style = self.bandit.choose_title_style(platform)
            title = titles.get(style) or titles["quote"]
            if media is None:
                media = self.postiz.upload(clip["path"])
            try:
                result = self.postiz.create_post(
                    account["id"], to_html(caption(platform, title, clip, moment, self.cfg)), media,
                    settings_for(account["identifier"], title, self.cfg.publisher, hashtag_list(clip, self.cfg)),
                    iso(slot), self.cfg.publisher.post_type,
                )
            except PostizError as e:
                errors += 1
                log.error("Postiz refused the %s post for %s: %s", platform, clip["id"], e)
                continue
            postiz_id = next((r.get("postId") for r in result if r.get("integration") == account["id"]),
                             result[0].get("postId") if result else None)
            post_id = uuid.uuid4().hex[:12]
            self.db.insert("posts", {
                "id": post_id, "clip_id": clip["id"], "platform": platform, "account_id": account["id"],
                "account_name": account.get("name"), "postiz_post_id": postiz_id, "title": title,
                "title_style": style, "scheduled_at": slot,
                "state": "DRAFT" if self.cfg.publisher.post_type == "draft" else "QUEUE",
            })
            posted += 1
            rows.append(sheet_row({
                "Posted At (UTC)": sheet_time(now), "Post ID": post_id, "Clip ID": clip["id"],
                "Platform": platform, "Account": account.get("name"),
                "Streamer": moment["channel_display"] or moment["channel"], "Source": moment["platform"],
                "Moment Type": moment["moment_type"], "Peak Msgs/Min": moment["peak_mpm"], "Title": title,
                "Title Style": style, "Layout": fmt.get("layout"), "Captions": fmt.get("captions"),
                "Length (s)": round(clip["duration"] or 0), "Hook": fmt.get("hook"), "Status": "scheduled",
                "Source VOD": clip["source_url"], "Scheduled For (UTC)": sheet_time(slot),
            }))
            log.info("scheduled %s on %s/%s at %s: %r", clip["id"], platform, account.get("name"),
                     sheet_time(slot), title)
        try:
            self.sheet.append_posts(rows)
        except Exception:  # noqa: BLE001 - the sheet must never block posting
            log.exception("could not log %d posts to the Google Sheet", len(rows))
        return {"posted": posted, "errors": errors}

    def plan(self) -> list[dict]:
        """Dry run: what would be posted for the ready clips, without posting."""
        out = []
        now = time.time()
        for clip in self.db.query("SELECT * FROM clips WHERE status = 'ready' ORDER BY created_at"):
            moment = self.db.one("SELECT * FROM moments WHERE id = ?", (clip["moment_id"],))
            titles = json.loads(clip["titles"])
            for platform in self.cfg.publisher.platforms:
                style = self.bandit.choose_title_style(platform)
                choice = self.pick_account(platform, now) if self.postiz else None
                out.append({
                    "clip": clip["id"], "platform": platform, "title_style": style, "title": titles.get(style),
                    "account": choice[0].get("name") if choice else None,
                    "when": sheet_time(choice[1]) if choice else None,
                    "caption": caption(platform, titles.get(style, ""), clip, moment, self.cfg),
                })
        return out
