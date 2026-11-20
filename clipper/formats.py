"""Repeat the format that wins: a Thompson-sampling bandit over clip formats.

Each format choice is an independent Beta-Bernoulli arm:
  * clip format (shared across platforms): layout, captions, length, hook
  * title style, separately per platform (what works on TikTok may not on Shorts)

A post is a "hit" or a "miss" (the analyst decides, using views and
engagement), and recent posts weigh more than old ones (half-life), so the
pipeline follows trends as they shift. Thompson sampling keeps exploring
options that are still uncertain while mostly exploiting the winners.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Iterable

from .config import TITLE_STYLES
from .db import DB

DEFAULT_ARMS: dict[str, list[str]] = {
    "layout": ["blur", "stack", "crop"],
    "captions": ["pop", "karaoke", "clean"],
    "length": ["20", "30", "45", "58"],
    "hook": ["banner", "none"],
}
STATE_KEY = "bandit"


class FormatBandit:
    def __init__(self, db: DB, arms: dict[str, list[str]] | None = None, rng: random.Random | None = None):
        self.db = db
        self.arms = {**DEFAULT_ARMS, **{k: [str(o) for o in v] for k, v in (arms or {}).items()}}
        self.rng = rng or random.Random()

    def _state(self) -> dict:
        return self.db.kv_get(STATE_KEY, {}) or {}

    def _ab(self, state: dict, scope: str, dim: str, option: str) -> tuple[float, float]:
        entry = state.get(scope, {}).get(dim, {}).get(option)
        return (entry["a"], entry["b"]) if entry else (1.0, 1.0)

    def _draw(self, state: dict, scope: str, dim: str, options: list[str]) -> str:
        draws = {o: self.rng.betavariate(*self._ab(state, scope, dim, o)) for o in options}
        return max(draws, key=draws.get)

    def choose_format(self, layouts_available: Iterable[str] | None = None) -> dict[str, str]:
        state = self._state()
        allowed = set(layouts_available) if layouts_available is not None else None
        fmt = {}
        for dim, options in self.arms.items():
            opts = [o for o in options if dim != "layout" or allowed is None or o in allowed] or options
            fmt[dim] = self._draw(state, "format", dim, opts)
        return fmt

    def choose_title_style(self, platform: str) -> str:
        return self._draw(self._state(), f"title:{platform}", "style", list(TITLE_STYLES))

    def recompute(self, judged: list[dict], half_life_days: float, now: float) -> dict:
        """Rebuild the posteriors from scratch from every judged post.

        judged: dicts with platform, hit (0/1), created_at, format (dict), title_style, score.
        """
        state: dict = {}
        for p in judged:
            weight = 0.5 ** (max(0.0, now - p["created_at"]) / 86400.0 / max(half_life_days, 0.1))
            entries = [("format", dim, str(val)) for dim, val in (p.get("format") or {}).items() if dim in self.arms]
            if p.get("title_style"):
                entries.append((f"title:{p['platform']}", "style", p["title_style"]))
            for scope, dim, option in entries:
                e = state.setdefault(scope, {}).setdefault(dim, {}).setdefault(
                    option, {"a": 1.0, "b": 1.0, "posts": 0, "hits": 0, "score_sum": 0.0})
                e["a" if p["hit"] else "b"] += weight
                e["posts"] += 1
                e["hits"] += int(bool(p["hit"]))
                e["score_sum"] += float(p.get("score") or 0.0)
        self.db.kv_set(STATE_KEY, state)
        return state

    def win_probabilities(self, scope: str, dim: str, options: list[str], samples: int = 4000) -> dict[str, float]:
        state = self._state()
        wins: Counter = Counter()
        for _ in range(samples):
            draws = {o: self.rng.betavariate(*self._ab(state, scope, dim, o)) for o in options}
            wins[max(draws, key=draws.get)] += 1
        return {o: wins[o] / samples for o in options}

    def report(self, platforms: Iterable[str]) -> list[list]:
        """Rows for the 'Formats' sheet tab."""
        state = self._state()
        groups = [("format", dim, opts) for dim, opts in self.arms.items()]
        groups += [(f"title:{p}", "style", list(TITLE_STYLES)) for p in platforms]
        rows = []
        for scope, dim, options in groups:
            probs = self.win_probabilities(scope, dim, options, samples=2000)
            leader = max(probs, key=probs.get)
            for o in options:
                e = state.get(scope, {}).get(dim, {}).get(o, {})
                posts, hits = e.get("posts", 0), e.get("hits", 0)
                if posts == 0:
                    status = "untested"
                elif o == leader and probs[o] >= 0.6:
                    status = "WINNING"
                elif probs[o] < 0.05 and posts >= 5:
                    status = "losing"
                else:
                    status = "testing"
                rows.append([
                    "all platforms" if scope == "format" else scope.split(":", 1)[1],
                    "title style" if scope.startswith("title:") else dim,
                    o, posts, hits,
                    f"{hits / posts:.0%}" if posts else "",
                    round(e.get("score_sum", 0.0) / posts) if posts else "",
                    f"{probs[o]:.0%}",
                    status,
                ])
        return rows
