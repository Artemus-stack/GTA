"""Agent 2 - Editor.

For each queued moment (highest priority first): find the VOD and the offset,
pull only the surrounding window with yt-dlp, let ClipsAI transcribe it and pick
clean clip edges around the chat spike, cut it, render a 9:16 version with
burned-in captions, and write three title variants.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path

from ..config import Config
from ..db import DB
from ..formats import FormatBandit
from ..media.captions import build_ass
from ..media.clipsai_adapter import (
    ClipsAI,
    choose_boundaries,
    sentences_of,
    speaker_available,
    words_between,
)
from ..media.download import NotReady, Skip, download_section, probe
from ..media.render import POSITIONS, detect_facecam, facecam_from_fraction, render
from ..media.render import probe as probe_file
from ..sources.twitch import TwitchAPI
from ..sources.youtube import watch_url
from ..titles import ClipContext, TitleWriter

log = logging.getLogger("editor")


def vod_link(url: str, platform: str, offset: float) -> str:
    t = max(0, int(offset))
    if platform == "twitch":
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{url}?t={h}h{m}m{s}s"
    if platform == "youtube":
        return f"{url}&t={t}s" if "?" in url else f"{url}?t={t}s"
    return url


def credit_text(m) -> str:
    if m["platform"] == "twitch":
        return f"twitch.tv/{m['channel']}"
    if m["platform"] == "youtube":
        return f"YouTube: {m['channel_display'] or m['channel']}"
    return m["channel_display"] or m["channel"]


class Editor:
    def __init__(self, cfg: Config, db: DB, bandit: FormatBandit, titles: TitleWriter | None = None,
                 twitch: TwitchAPI | None = None, clipsai: ClipsAI | None = None):
        self.cfg = cfg
        self.db = db
        self.bandit = bandit
        self.titles = titles or TitleWriter(cfg.titles)
        sec = cfg.secrets
        if twitch is None and sec.twitch_client_id and sec.twitch_client_secret:
            twitch = TwitchAPI(sec.twitch_client_id, sec.twitch_client_secret)
        self.twitch = twitch
        self._clipsai = clipsai
        self._speaker_problem = speaker_available() if sec.hf_token else "HF_TOKEN is not set"

    @property
    def clipsai(self) -> ClipsAI:
        if self._clipsai is None:
            e = self.cfg.editor
            self._clipsai = ClipsAI(e.whisper_model, e.device, e.language)
        return self._clipsai

    # -- queue ------------------------------------------------------------
    def next_moment(self):
        now = time.time()
        rows = self.db.query("SELECT * FROM moments WHERE status = 'new' AND next_attempt_at <= ?", (now,))
        prio = self.db.kv_get("priority", {}) or {}
        best, best_score = None, -1.0
        for m in rows:
            seen_at = m["onset_at"] or m["created_at"]
            age_h = (now - seen_at) / 3600
            if m["onset_at"] and age_h > self.cfg.editor.max_moment_age_h:
                self.db.update("moments", m["id"], status="expired", note="too old to post")
                continue
            boost = prio.get("streamer", {}).get(m["channel"], 1.0) * prio.get("type", {}).get(m["moment_type"], 1.0)
            score = (m["score"] or 0.0) * boost / (1.0 + age_h / 6.0)    # fresher moments first
            if score > best_score:
                best, best_score = m, score
        return best

    def process_next(self) -> bool:
        m = self.next_moment()
        if m is None:
            return False
        if not self.db.claim("moments", m["id"], "new", "processing"):
            return True
        t0 = time.time()
        try:
            clip_id = self.process(m)
            self.db.update("moments", m["id"], status="edited", note=clip_id)
            log.info("clip %s ready in %.0fs", clip_id, time.time() - t0)
        except NotReady as e:
            self.db.update("moments", m["id"], status="new", note=f"waiting: {e}",
                           next_attempt_at=time.time() + self.cfg.editor.retry_delay_s)
            log.info("moment %s not ready yet: %s", m["id"], e)
        except Skip as e:
            self.db.update("moments", m["id"], status="skipped", note=str(e)[:500])
            log.info("moment %s skipped: %s", m["id"], e)
        except Exception as e:  # noqa: BLE001 - one bad moment must not stop the agent
            errors = m["errors"] + 1
            done = errors >= self.cfg.editor.max_errors
            self.db.update("moments", m["id"], status="failed" if done else "new", errors=errors,
                           note=f"{type(e).__name__}: {e}"[:500], next_attempt_at=time.time() + 300)
            log.exception("moment %s failed (%d/%d)", m["id"], errors, self.cfg.editor.max_errors)
        return True

    def run_forever(self, stop: threading.Event) -> None:
        self.db.requeue_stale("moments", "processing", "new", 3600)
        log.info("editor started")
        while not stop.is_set():
            if not self.process_next():
                stop.wait(5)

    # -- one moment ---------------------------------------------------------
    def resolve(self, m) -> tuple[str, float]:
        """VOD url and the offset (seconds into it) where the chat burst began."""
        if m["source_url"] and m["onset_offset"] is not None:
            return m["source_url"], float(m["onset_offset"])
        now = time.time()
        if m["platform"] == "twitch":
            if self.twitch is None:
                raise Skip("Twitch API credentials are needed to find the VOD")
            if now - m["onset_at"] < self.cfg.editor.post_roll_s + 120:
                raise NotReady("waiting for the VOD to catch up")
            vod = self.twitch.find_vod(m["channel_id"], m["onset_at"])
            if vod is None:
                if now - m["onset_at"] > 3 * 3600:
                    raise Skip("no VOD for this stream (VODs disabled or subscriber-only)")
                raise NotReady("VOD not listed yet")
            url, offset = vod["url"], m["onset_at"] - vod["created_at"]
        elif m["platform"] == "youtube":
            url = watch_url(m["video_id"])
            info = probe(url, self.cfg.editor.cookies_file)
            if info.get("live_status") in ("is_live", "is_upcoming"):
                raise NotReady("stream still live; YouTube VODs can be cut once it ends")
            start = info.get("release_timestamp") or m["stream_started_at"]
            if not start:
                raise Skip("can't tell when the stream started")
            offset = m["onset_at"] - float(start)
        else:
            raise Skip(f"no way to resolve {m['platform']} moments")
        if offset < 0:
            raise Skip(f"moment is before the VOD starts (offset {offset:.0f}s)")
        self.db.update("moments", m["id"], source_url=url, onset_offset=offset)
        return url, offset

    def layouts_available(self) -> list[str]:
        layouts = ["blur", "crop", "stack"]
        if self._speaker_problem is None:
            layouts.append("speaker")
        return layouts

    def process(self, m) -> str:
        e = self.cfg.editor
        url, onset = self.resolve(m)
        fmt = self.bandit.choose_format(self.layouts_available())
        target = float(fmt["length"])
        lag = self.cfg.scout.reaction_lag_s
        win_start = max(0.0, onset - e.pre_roll_s)
        win_end = onset + e.post_roll_s
        work = self.cfg.data_path / "work" / m["id"]
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        try:
            log.info("moment %s: downloading %.0f-%.0fs of %s", m["id"], win_start, win_end, url)
            window = download_section(url, win_start, win_end, work, max_height=e.max_height,
                                      precise=e.precise_cuts, cookies_file=e.cookies_file)
            window_len = probe_file(window).duration
            event_t = min(max(0.0, onset - lag - win_start), window_len)

            transcription = self.clipsai.transcribe(window)
            candidates = self.clipsai.find_clips(transcription, target) if transcription else []
            start, end, how = choose_boundaries(event_t, target, window_len, candidates,
                                                sentences_of(transcription))
            log.info("moment %s: clip %.1f-%.1fs of the window (%s, target %.0fs, %d ClipsAI segments)",
                     m["id"], start, end, how, target, len(candidates))
            # credit the length option closest to what was actually cut
            fmt["length"] = min(self.bandit.arms["length"], key=lambda o: abs(float(o) - (end - start)))
            cut = self.clipsai.trim(window, start, end, work / "cut.mp4")
            words = words_between(transcription, start, end)
            transcript = " ".join(w.text for w in words)

            chat_sample = json.loads(m["chat_sample"] or "[]")
            pack, title_source = self.titles.write(ClipContext(
                streamer=m["channel_display"] or m["channel"], platform=m["platform"],
                stream_title=m["stream_title"] or "", moment_type=m["moment_type"] or "general",
                peak_mpm=m["peak_mpm"] or 0, chat_sample=chat_sample, transcript=transcript,
            ))

            layout, facecam, src = self._layout(fmt["layout"], cut, m, work)
            fmt["layout"] = layout              # credit the layout actually used
            ass = work / "captions.ass"
            ass.write_text(build_ass(
                words, fmt["captions"], end - start, POSITIONS[layout],
                hook=pack.hook if fmt["hook"] == "banner" else None,
                credit=credit_text(m) if e.credit_watermark else None,
                font=e.caption_font,
            ), encoding="utf-8")

            clip_id = f"c-{m['id']}"
            final = self.cfg.data_path / "clips" / clip_id / "clip.mp4"
            render(src, final, layout, ass, facecam=facecam, fonts_dir=e.fonts_dir)
            self.db.insert("clips", {
                "id": clip_id, "moment_id": m["id"], "path": str(final), "duration": round(end - start, 2),
                "source_url": vod_link(url, m["platform"], win_start + start),
                "transcript": transcript, "titles": json.dumps(
                    {"quote": pack.quote, "curiosity": pack.curiosity, "reaction": pack.reaction},
                    ensure_ascii=False),
                "hook": pack.hook, "description": pack.description,
                "hashtags": json.dumps(pack.hashtags), "format": json.dumps(fmt), "title_source": title_source,
            }, ignore_existing=True)
            return clip_id
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def _layout(self, choice: str, cut: Path, m, work: Path):
        """Returns (layout actually used, facecam box, source video for the renderer)."""
        if choice == "stack":
            p = probe_file(cut)
            frac = self.cfg.editor.facecams.get(m["channel"]) or self.cfg.editor.facecams.get(m["channel_display"] or "")
            box = facecam_from_fraction(frac, p) if frac else detect_facecam(cut, p)
            if box:
                return "stack", box, cut
            log.info("no webcam found for %s; using the blur layout", m["channel"])
            return "blur", None, cut
        if choice == "speaker":
            try:
                return "speaker", None, self.clipsai.speaker_crop(cut, work / "speaker.mp4", self.cfg.secrets.hf_token)
            except Exception as e:  # noqa: BLE001 - optional layout
                log.warning("ClipsAI speaker crop failed (%s); using the blur layout", e)
                return "blur", None, cut
        return choice, None, cut
