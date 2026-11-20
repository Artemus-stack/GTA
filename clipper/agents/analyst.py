"""The learning loop: what hit, and what to do more of.

Every interval:
  1. sync post states and published URLs from Postiz,
  2. pull per-post views/likes/comments/shares from Postiz analytics,
  3. pick up numbers typed into the sheet for posts the API has no stats for,
  4. judge posts older than ``judge_after_h``: a hit beats its platform's median
     viral score (or ``hit_score`` when set),
  5. retrain the format bandit, so the editor and publisher repeat what wins,
  6. re-weight streamers and moment types, so the editor clips those first,
  7. write metrics and the Formats/Streamers tabs back to the sheet.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
from collections import defaultdict

from ..config import Config
from ..db import DB
from ..formats import FormatBandit
from ..postiz import Postiz
from ..sheets import NullSheet, Sheet
from .publisher import iso, sheet_time

log = logging.getLogger("analyst")

STATUS = {"QUEUE": "scheduled", "PUBLISHED": "published", "ERROR": "error", "DRAFT": "draft"}
METRICS = ("views", "likes", "comments", "shares")


def _int(value) -> int | None:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except (TypeError, ValueError):
        return None


class Analyst:
    def __init__(self, cfg: Config, db: DB, bandit: FormatBandit, postiz: Postiz | None,
                 sheet: Sheet | NullSheet | None = None):
        self.cfg = cfg
        self.db = db
        self.bandit = bandit
        self.postiz = postiz
        self.sheet = sheet or NullSheet()
        self._sheet_rows: list[dict] = []

    def score(self, row) -> float:
        w = self.cfg.analyst.score_weights
        return sum(float(w.get(k, 0)) * float(row[k] or 0) for k in METRICS)

    def run_forever(self, stop: threading.Event) -> None:
        log.info("analyst started (every %ds)", self.cfg.analyst.interval_s)
        while not stop.is_set():
            try:
                summary = self.run_once()
                log.info("analyst pass: %s", summary)
            except Exception:  # noqa: BLE001
                log.exception("analyst pass failed")
            stop.wait(self.cfg.analyst.interval_s)

    def run_once(self) -> dict:
        summary = {"states": 0, "metrics": 0, "from_sheet": 0}
        if self.postiz:
            summary["states"] = self.sync_states()
            summary["metrics"] = self.pull_metrics()
        summary["from_sheet"] = self.merge_sheet()
        judged = self.judge()
        summary["judged"] = len(judged)
        summary["hits"] = sum(p["hit"] for p in judged)
        self.bandit.recompute(judged, self.cfg.analyst.half_life_days, time.time())
        self.update_priorities(judged)
        self.write_sheet(judged)
        return summary

    # -- 1-3: collect -----------------------------------------------------------
    def _recent_posts(self, days: int = 35):
        return self.db.query("SELECT * FROM posts WHERE postiz_post_id IS NOT NULL AND scheduled_at > ?",
                             (time.time() - days * 86400,))

    def sync_states(self) -> int:
        rows = self._recent_posts()
        if not rows:
            return 0
        start = min(r["scheduled_at"] for r in rows) - 86400
        end = max(r["scheduled_at"] for r in rows) + 86400
        remote = {p.get("id"): p for p in self.postiz.posts(iso(start), iso(end))}
        changed = 0
        for r in rows:
            p = remote.get(r["postiz_post_id"])
            if not p:
                continue
            state, url = p.get("state") or r["state"], p.get("releaseURL") or r["release_url"]
            if state != r["state"] or url != r["release_url"]:
                self.db.update("posts", r["id"], state=state, release_url=url)
                changed += 1
        return changed

    def pull_metrics(self) -> int:
        now, n = time.time(), 0
        for r in self._recent_posts(days=31):
            if r["state"] != "PUBLISHED":
                continue
            age_h = (now - r["scheduled_at"]) / 3600
            since_h = (now - (r["metrics_at"] or 0)) / 3600
            due = (age_h < 48 and since_h >= 1) or (age_h < 168 and since_h >= 6) or since_h >= 24
            if not due:
                continue
            try:
                m = self.postiz.post_analytics(r["postiz_post_id"])
            except Exception as e:  # noqa: BLE001 - one platform's API hiccup shouldn't stop the pass
                log.warning("analytics for post %s failed: %s", r["id"], e)
                continue
            if not m:
                continue
            self.db.update("posts", r["id"], **{k: m.get(k) for k in METRICS}, metrics_source="postiz",
                           metrics_at=now)
            n += 1
        return n

    def merge_sheet(self) -> int:
        """Numbers typed into the sheet count for posts with no API metrics."""
        try:
            self._sheet_rows = self.sheet.read_posts()
        except Exception:  # noqa: BLE001
            log.exception("could not read the Google Sheet")
            self._sheet_rows = []
        n = 0
        for row in self._sheet_rows:
            post = self.db.one("SELECT * FROM posts WHERE id = ?", (row.get("Post ID", ""),))
            if post is None or post["metrics_source"] == "postiz":
                continue
            typed = {k: _int(row.get(k.title())) for k in METRICS}
            if typed["views"] is None:
                continue
            if any(typed[k] != post[k] for k in METRICS if typed[k] is not None):
                self.db.update("posts", post["id"], **{k: v for k, v in typed.items() if v is not None},
                               metrics_source="sheet", metrics_at=time.time())
                n += 1
        return n

    # -- 4: judge ----------------------------------------------------------------
    def judge(self) -> list[dict]:
        now = time.time()
        rows = self.db.query(
            "SELECT p.*, c.format, m.channel, m.channel_display, m.moment_type, m.platform AS source "
            "FROM posts p JOIN clips c ON c.id = p.clip_id JOIN moments m ON m.id = c.moment_id "
            "WHERE p.views IS NOT NULL")
        ready = [r for r in rows if now - r["scheduled_at"] >= self.cfg.analyst.judge_after_h * 3600]
        scores = {r["id"]: self.score(r) for r in ready}
        threshold: dict[str, float] = {}
        if self.cfg.analyst.hit_score > 0:
            threshold = defaultdict(lambda: self.cfg.analyst.hit_score)
        else:
            by_platform = defaultdict(list)
            for r in ready:
                by_platform[r["platform"]].append(scores[r["id"]])
            threshold = {p: statistics.median(v) for p, v in by_platform.items()}
        judged = []
        for r in ready:
            hit = int(scores[r["id"]] >= threshold[r["platform"]] and scores[r["id"]] > 0)
            if r["hit"] != hit:
                self.db.update("posts", r["id"], hit=hit)
            judged.append({
                "id": r["id"], "platform": r["platform"], "hit": hit, "created_at": r["scheduled_at"],
                "format": json.loads(r["format"] or "{}"), "title_style": r["title_style"],
                "score": scores[r["id"]], "streamer": r["channel"], "source": r["source"],
                "moment_type": r["moment_type"],
            })
        return judged

    # -- 6: priorities -----------------------------------------------------------
    def update_priorities(self, judged: list[dict]) -> dict:
        if not judged:
            return {}
        overall = max(sum(p["hit"] for p in judged) / len(judged), 0.05)
        prio: dict[str, dict[str, float]] = {}
        for field, key in (("streamer", "streamer"), ("type", "moment_type")):
            groups: dict[str, list[int]] = defaultdict(list)
            for p in judged:
                groups[p[key]].append(p["hit"])
            prio[field] = {
                name: round(min(2.0, max(0.5, ((sum(h) + 4 * overall) / (len(h) + 4)) / overall)), 3)
                for name, h in groups.items()
            }
        self.db.kv_set("priority", prio)
        return prio

    # -- 7: sheet ------------------------------------------------------------------
    def write_sheet(self, judged: list[dict]) -> None:
        if not self.sheet.enabled:
            return
        try:
            self._write_posts()
            self.sheet.replace("Formats", self.bandit.report(self.cfg.publisher.platforms))
            self.sheet.replace("Streamers", self._streamer_rows(judged))
        except Exception:  # noqa: BLE001
            log.exception("could not update the Google Sheet")

    def _write_posts(self) -> None:
        updates: dict[int, dict] = {}
        for row in self._sheet_rows:
            post = self.db.one("SELECT * FROM posts WHERE id = ?", (row.get("Post ID", ""),))
            if post is None:
                continue
            want = {
                "Status": STATUS.get(post["state"] or "", post["state"] or ""),
                "Post URL": post["release_url"] or "",
            }
            if post["views"] is not None:
                want.update({
                    "Views": post["views"], "Likes": post["likes"] or 0, "Comments": post["comments"] or 0,
                    "Shares": post["shares"] or 0, "Viral Score": round(self.score(post)),
                    "Metrics Updated": sheet_time(post["metrics_at"]),
                })
            if post["hit"] is not None:
                want["Hit"] = "YES" if post["hit"] else "no"
            diff = {k: v for k, v in want.items() if str(row.get(k, "")) != str(v)}
            if diff:
                updates[row["_row"]] = diff
        self.sheet.update_posts(updates)

    def _streamer_rows(self, judged: list[dict]) -> list[list]:
        prio = (self.db.kv_get("priority", {}) or {}).get("streamer", {})
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for p in judged:
            groups[(p["streamer"], p["source"])].append(p)
        rows = []
        for (name, source), posts in sorted(groups.items(), key=lambda kv: -sum(p["hit"] for p in kv[1])):
            hits = sum(p["hit"] for p in posts)
            rows.append([name, source, len(posts), hits, f"{hits / len(posts):.0%}",
                         round(sum(p["score"] for p in posts) / len(posts)), prio.get(name, 1.0)])
        return rows
