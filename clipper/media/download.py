"""yt-dlp: the raw material for every clip.

Rather than pulling a whole multi-hour VOD, only the window around the moment
is downloaded (``download_ranges``).
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("editor.download")


class NotReady(Exception):
    """The source exists but can't be cut yet (e.g. the YouTube stream is still live)."""


class Skip(Exception):
    """The moment can never be clipped (no VOD, subscriber-only, removed...)."""


_SKIP_HINTS = ("subscriber", "private video", "video unavailable", "removed", "members-only",
               "has been deleted", "does not exist", "sign in to confirm your age")


def _opts(cookies_file: str | None, **extra) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "retries": 5, "fragment_retries": 10}
    if cookies_file:
        opts["cookiefile"] = cookies_file
    opts.update(extra)
    return opts


def probe(url: str, cookies_file: str | None = None) -> dict:
    """Metadata only (live_status, release_timestamp, duration, uploader...)."""
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL(_opts(cookies_file, skip_download=True)) as ydl:
            return ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        _raise_classified(e)
        raise


def download_section(url: str, start: float, end: float, out_dir: Path, *, max_height: int = 1080,
                     precise: bool = True, cookies_file: str | None = None) -> Path:
    """Download [start, end] seconds of ``url`` as an mp4 in ``out_dir``.

    ``precise`` re-encodes at the cut points (yt-dlp's force_keyframes_at_cuts)
    so the file starts exactly at ``start``. A stream copy would snap to the
    previous keyframe and shift every timestamp inside the window by up to a
    few seconds.
    """
    import yt_dlp
    from yt_dlp.utils import download_range_func

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("window.*"):
        old.unlink()
    opts = _opts(
        cookies_file,
        format=f"bv*[height<={max_height}]+ba/b[height<={max_height}]/bv*+ba/b",
        download_ranges=download_range_func(None, [(max(0.0, start), end)]),
        force_keyframes_at_cuts=precise,
        outtmpl={"default": str(out_dir / "window.%(ext)s")},
        merge_output_format="mp4",
        overwrites=True,
    )
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        _raise_classified(e)
        raise
    paths = [Path(d["filepath"]) for d in (info or {}).get("requested_downloads") or [] if d.get("filepath")]
    paths = [p for p in paths if p.exists()] or sorted(out_dir.glob("window.*"))
    if not paths:
        raise RuntimeError(f"yt-dlp finished but no file appeared in {out_dir}")
    return paths[0]


def _raise_classified(err: Exception) -> None:
    msg = str(err).lower()
    if "is live" in msg or "live event will begin" in msg or "premieres in" in msg:
        raise NotReady(str(err)) from err
    if any(h in msg for h in _SKIP_HINTS):
        raise Skip(str(err)) from err
