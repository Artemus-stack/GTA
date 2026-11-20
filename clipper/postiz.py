"""Postiz public API client (https://github.com/gitroomhq/postiz-app).

Covers what the publisher and analyst use:
  GET  /public/v1/integrations             connected accounts ("channels")
  POST /public/v1/upload                   upload the clip (multipart field "file")
  POST /public/v1/posts                    schedule a post (rate-limited, 90/hour by default)
  GET  /public/v1/posts                    states and published URLs
  GET  /public/v1/analytics/post/{id}      per-post views/likes/comments/shares

Auth is the raw API key in the Authorization header (Settings -> Public API).
Base URL: https://api.postiz.com for Postiz Cloud. Self-hosted with Postiz's
docker-compose it is http://localhost:4007/api (or https://your-domain/api).
"""

from __future__ import annotations

import html
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import httpx

from .config import PublisherConfig

log = logging.getLogger("publisher.postiz")

# Postiz provider identifier -> our platform name
PROVIDERS = {
    "tiktok": "tiktok",
    "tiktok-business": "tiktok",
    "youtube": "youtube",
    "instagram": "instagram",
    "instagram-standalone": "instagram",
}


class PostizError(RuntimeError):
    pass


class Postiz:
    def __init__(self, base_url: str, api_key: str, posts_per_hour: int = 80, http: httpx.Client | None = None):
        self.base = base_url.rstrip("/") + "/public/v1"
        self.headers = {"Authorization": api_key}
        self.http = http or httpx.Client(timeout=httpx.Timeout(300, connect=20))
        self.posts_per_hour = posts_per_hour
        self._post_times: deque[float] = deque()

    def _request(self, method: str, path: str, **kwargs) -> Any:
        last_error = ""
        for attempt in range(5):
            try:
                r = self.http.request(method, self.base + path, headers=self.headers, **kwargs)
            except httpx.TransportError as e:
                last_error = str(e)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After") or 60 * (attempt + 1))
                log.warning("Postiz rate limit hit on %s; waiting %.0fs", path, wait)
                time.sleep(min(wait, 900))
                continue
            if r.status_code >= 500:
                last_error = f"{r.status_code}: {r.text[:300]}"
                time.sleep(2 ** attempt * 2)
                continue
            if r.status_code >= 400:
                raise PostizError(f"{method} {path} -> {r.status_code}: {r.text[:600]}")
            return r.json() if r.content else None
        raise PostizError(f"{method} {path} failed after retries: {last_error}")

    def _throttle_posts(self) -> None:
        """POST /posts is the only throttled route (API_LIMIT, 90/hour by default)."""
        now = time.time()
        while self._post_times and now - self._post_times[0] > 3600:
            self._post_times.popleft()
        if len(self._post_times) >= self.posts_per_hour:
            wait = 3600 - (now - self._post_times[0]) + 1
            log.info("pausing %.0fs to stay under the Postiz posts/hour limit", wait)
            time.sleep(wait)
        self._post_times.append(time.time())

    # -- endpoints ------------------------------------------------------------
    def integrations(self) -> list[dict]:
        return self._request("GET", "/integrations") or []

    def upload(self, path: Path) -> dict:
        data = Path(path).read_bytes()
        media = self._request("POST", "/upload", files={"file": (Path(path).name, data, "video/mp4")})
        if not media or not media.get("id") or not media.get("path"):
            raise PostizError(f"unexpected upload response: {media}")
        return media

    def create_post(self, integration_id: str, content: str, media: dict, settings: dict, date_iso: str,
                    post_type: str = "schedule") -> list[dict]:
        self._throttle_posts()
        body = {
            "type": post_type,
            "date": date_iso,
            "shortLink": False,
            "tags": [],
            "posts": [{
                "integration": {"id": integration_id},
                "value": [{"content": content, "image": [{"id": media["id"], "path": media["path"]}]}],
                "settings": settings,
            }],
        }
        return self._request("POST", "/posts", json=body) or []

    def posts(self, start_iso: str, end_iso: str) -> list[dict]:
        data = self._request("GET", "/posts", params={"startDate": start_iso, "endDate": end_iso}) or {}
        return data.get("posts", []) if isinstance(data, dict) else data

    def post_analytics(self, post_id: str, days: int = 7) -> dict[str, int] | None:
        """{'views': n, 'likes': n, 'comments': n, 'shares': n} (whatever the platform reports)."""
        data = self._request("GET", f"/analytics/post/{post_id}", params={"date": str(days)})
        if isinstance(data, dict) and data.get("missing"):
            return None
        metrics: dict[str, int] = {}
        for item in data or []:
            key = str(item.get("label", "")).strip().lower()
            if key not in ("views", "likes", "comments", "shares"):
                continue
            totals = []
            for point in item.get("data") or []:
                try:
                    totals.append(float(point.get("total") or 0))
                except (TypeError, ValueError):
                    pass
            if totals:
                metrics[key] = int(max(totals))
        return metrics


def to_html(text: str) -> str:
    """Postiz stores post bodies as HTML paragraphs (that is what its editor produces)."""
    return "".join(f"<p>{html.escape(line)}</p>" for line in text.split("\n"))


def settings_for(identifier: str, title: str, cfg: PublisherConfig, tags: list[str]) -> dict:
    platform = PROVIDERS.get(identifier)
    if platform == "tiktok":
        return {
            "title": title[:90],
            "privacy_level": cfg.tiktok_privacy,
            "duet": True,
            "stitch": True,
            "comment": True,
            "autoAddMusic": "no",
            "brand_content_toggle": False,
            "brand_organic_toggle": False,
            "content_posting_method": cfg.tiktok_method,
        }
    if platform == "youtube":
        return {
            "title": title[:100],
            "type": cfg.youtube_privacy,
            "selfDeclaredMadeForKids": "no",
            "tags": [{"value": t, "label": t} for t in tags[:12]],
        }
    if platform == "instagram":
        return {"post_type": "post"}    # a single video posts as a Reel
    raise ValueError(f"unsupported Postiz provider: {identifier}")
