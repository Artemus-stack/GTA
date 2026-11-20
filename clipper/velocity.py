"""Chat velocity: rolling messages-per-minute per channel, and spike ("moment") detection.

The tracker does not care about the time base. Feed it ``(t, text)`` with ``t``
in seconds: epoch seconds for live chat, or seconds into the VOD when scanning a
replay. A moment's ``onset``/``peak``/``end`` come back in the same time base.

A moment fires when the rolling 60 s message count reaches ``threshold``
(500 msg/min by default) *and*, unless ``baseline_multiplier`` is 0, when that
is at least ``baseline_multiplier`` times the channel's normal rate. Without
the second check a huge channel, whose chat never drops below 500 msg/min,
would produce a "moment" every couple of minutes.
"""

from __future__ import annotations

import math
import re
import statistics
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable

CATEGORIES: dict[str, set[str]] = {
    "funny": {"KEKW", "LUL", "LULW", "OMEGALUL", "LMAO", "LMFAO", "LOL", "ICANT", "KEKL", "PEPELAUGH",
              "XD", "HAHA", "HAHAHA", "DEAD", "😂", "🤣", "💀", "☠"},
    "hype": {"POG", "POGGERS", "POGCHAMP", "POGU", "W", "WW", "LETSGO", "HYPE", "GOAT", "CLUTCH", "EZ",
             "GG", "INSANE", "🔥", "🐐", "💪"},
    "shock": {"WTF", "NOWAY", "OMG", "WHAT", "HOLY", "MONKAS", "MONKAW", "D:", "BRO", "?", "??", "???",
              "😱", "😳", "😮", "🤯"},
    "fail": {"L", "LL", "RIP", "F", "SADGE", "UNLUCKY", "NO", "NOO", "💔", "😭"},
}
CLIP_TOKENS = {"CLIP", "CLIPIT", "CLIPPED", "CLIPTHAT"}
PHRASES = {"LETS GO": "hype", "LET'S GO": "hype", "NO WAY": "shock", "OH MY GOD": "shock",
           "CLIP IT": "clip", "CLIP THAT": "clip"}

_TOKEN_RE = re.compile(r"[A-Z0-9']+|D:|\?+|[^\sA-Z0-9]")
_SQUASH_RE = re.compile(r"(.)\1{2,}")


def _tokens(upper_text: str) -> list[str]:
    return [_SQUASH_RE.sub(r"\1", t) for t in _TOKEN_RE.findall(upper_text)]


def classify(texts: list[str]) -> tuple[str, Counter, int]:
    """Label a burst of chat as funny/hype/shock/fail/general; count clip requests."""
    counts: Counter = Counter()
    clip_requests = 0
    for text in texts:
        up = text.upper()
        seen: set[str] = {cat for phrase, cat in PHRASES.items() if phrase in up}
        for tok in _tokens(up):
            if tok in CLIP_TOKENS:
                seen.add("clip")
            if "HAHA" in tok or tok.startswith(("LMAO", "LMFAO")):
                seen.add("funny")
            for cat, words in CATEGORIES.items():
                if tok in words:
                    seen.add(cat)
        clip_requests += "clip" in seen
        counts.update(seen - {"clip"})
    if not counts:
        return "general", counts, clip_requests
    cat, n = counts.most_common(1)[0]
    if n < max(3, 0.05 * len(texts)):
        return "general", counts, clip_requests
    return cat, counts, clip_requests


def top_messages(texts: list[str], n: int = 8) -> list[str]:
    norm: Counter = Counter()
    first: dict[str, str] = {}
    for t in texts:
        key = _SQUASH_RE.sub(r"\1\1", " ".join(t.split()).lower())[:80]
        if key:
            norm[key] += 1
            first.setdefault(key, " ".join(t.split())[:80])
    return [f"{first[k]} (x{c})" if c > 1 else first[k] for k, c in norm.most_common(n)]


@dataclass
class Moment:
    onset: float           # when the chat burst began
    peak: float            # end of the densest 60 s window
    end: float             # when the burst died down
    peak_mpm: int          # messages in that densest 60 s window
    baseline_mpm: float    # the channel's normal rate before the burst
    moment_type: str
    score: float
    clip_requests: int = 0
    sample: list[str] = field(default_factory=list)


class VelocityTracker:
    def __init__(
        self,
        threshold: int = 500,
        window: int = 60,
        baseline_window: int = 900,
        baseline_multiplier: float = 2.0,
        warmup: int = 180,
        cooldown: int = 120,
        max_spike: int = 180,
        release_ratio: float = 0.75,
        grace: float = 2.0,
        sample_cap: int = 40,
    ):
        self.threshold = threshold
        self.window = window
        self.baseline_window = baseline_window
        self.baseline_multiplier = baseline_multiplier
        self.warmup = warmup
        self.cooldown = cooldown
        self.max_spike = max_spike
        self.release_ratio = release_ratio
        self.grace = grace
        self.sample_cap = sample_cap

        self._win: deque[int] = deque()
        self._win_sum = 0
        self._recent_texts: deque[tuple[int, list[str]]] = deque()
        self._pending: Counter = Counter()
        self._pending_texts: dict[int, list[str]] = {}
        self._next: int | None = None   # oldest second not yet closed
        self._first: int | None = None
        self._samples: deque[int] = deque(maxlen=max(6, baseline_window // 10))
        self._active: dict | None = None
        self._cooldown_until = -math.inf
        self.total = 0

    @property
    def rate(self) -> int:
        """Messages in the last closed 60 s window."""
        return self._win_sum

    @property
    def baseline(self) -> float | None:
        return statistics.median(self._samples) if len(self._samples) >= 3 else None

    @property
    def in_spike(self) -> bool:
        return self._active is not None

    def add(self, t: float, text: str = "") -> list[Moment]:
        sec = math.floor(t)
        if self._next is None:
            self._next = self._first = sec
        if sec < self._next:      # arrived after its second was closed
            sec = self._next
        self._pending[sec] += 1
        texts = self._pending_texts.setdefault(sec, [])
        if text and len(texts) < self.sample_cap:
            texts.append(text)
        return self.tick(t)

    def tick(self, now: float) -> list[Moment]:
        """Close every second older than ``now - grace``."""
        if self._next is None:
            return []
        limit = math.floor(now - self.grace)
        out = []
        while self._next < limit:
            m = self._close(self._next)
            if m:
                out.append(m)
            self._next += 1
        return out

    def flush(self) -> list[Moment]:
        """End of data (replay scans): close the tail and any open spike."""
        if self._next is None:
            return []
        out = self.tick(self._next + self.window + self.grace + 2)
        if self._active:
            out.append(self._finish(self._next - 1))
        return out

    # ------------------------------------------------------------------
    def _close(self, sec: int) -> Moment | None:
        count = self._pending.pop(sec, 0)
        texts = self._pending_texts.pop(sec, None) or []
        self.total += count
        self._win.append(count)
        self._win_sum += count
        if len(self._win) > self.window:
            self._win_sum -= self._win.popleft()
        self._recent_texts.append((sec, texts))
        horizon = sec - (self.window + self.max_spike)
        while self._recent_texts and self._recent_texts[0][0] <= horizon:
            self._recent_texts.popleft()

        rolling = self._win_sum
        elapsed = sec - self._first + 1
        if self._active is None:
            if elapsed >= self.window and sec >= self._cooldown_until and (sec - self._first) % 10 == 0:
                self._samples.append(rolling)
            if sec < self._cooldown_until or rolling < self.threshold:
                return None
            baseline = self.baseline
            if self.baseline_multiplier > 0:
                if elapsed < self.warmup or baseline is None:
                    return None
                if rolling < self.baseline_multiplier * max(baseline, 1.0):
                    return None
            self._active = {
                "onset": self._find_onset(sec, baseline),
                "start": sec,
                "peak": rolling,
                "peak_sec": sec,
                "baseline": baseline if baseline is not None else 0.0,
            }
            return None

        active = self._active
        if rolling > active["peak"]:
            active["peak"], active["peak_sec"] = rolling, sec
        floor = max(self.threshold, self.baseline_multiplier * active["baseline"]) * self.release_ratio
        if rolling < floor or sec - active["start"] >= self.max_spike:
            return self._finish(sec)
        return None

    def _find_onset(self, sec: int, baseline: float | None) -> float:
        """Most likely start of the burst: a Poisson change point.

        Up to the densest 3 s of the window, pick the split where "normal rate
        before, higher rate after" explains the per-second counts best. Unlike a
        fixed level, this isn't thrown off by a few noisy seconds.
        """
        counts = list(self._win)
        n = len(counts)
        if n < 3:
            return float(sec - n + 1)
        dens = [sum(counts[max(0, i - 2): i + 1]) for i in range(n)]
        end = max(range(n), key=dens.__getitem__) + 1
        prefix = [0]
        for c in counts[:end]:
            prefix.append(prefix[-1] + c)
        best_s, best_ll = 0, -math.inf
        for s in range(1, end):
            sum_b, sum_a = prefix[s], prefix[end] - prefix[s]
            rate_b, rate_a = sum_b / s, sum_a / (end - s)
            if rate_a <= rate_b:
                continue
            # Poisson log-likelihood up to terms that don't depend on s
            ll = (sum_b * math.log(rate_b) if sum_b else 0.0) + sum_a * math.log(rate_a)
            if ll > best_ll:
                best_ll, best_s = ll, s
        return float(sec - (n - 1) + best_s)

    def _finish(self, sec: int) -> Moment:
        active, self._active = self._active, None
        self._cooldown_until = sec + self.cooldown
        onset = active["onset"]
        texts = [t for s, ts in self._recent_texts if onset <= s <= onset + self.window for t in ts]
        mtype, _, clip_requests = classify(texts)
        baseline = active["baseline"]
        ratio = active["peak"] / max(baseline or self.threshold / 2, 1.0)
        clip_share = clip_requests / len(texts) if texts else 0.0
        score = active["peak"] * (1 + 0.1 * min(ratio, 20.0)) * (1 + clip_share)
        return Moment(
            onset=onset,
            peak=float(active["peak_sec"]),
            end=float(sec),
            peak_mpm=int(active["peak"]),
            baseline_mpm=round(baseline, 1),
            moment_type=mtype,
            score=round(score, 1),
            clip_requests=clip_requests,
            sample=top_messages(texts),
        )


class ChannelTrackers:
    """Thread-safe map of channel key -> VelocityTracker."""

    def __init__(self, factory: Callable[[str], VelocityTracker]):
        self._factory = factory
        self._trackers: dict[str, VelocityTracker] = {}
        self._lock = threading.Lock()

    def add(self, key: str, t: float, text: str) -> list[tuple[str, Moment]]:
        with self._lock:
            tr = self._trackers.get(key)
            if tr is None:
                tr = self._trackers[key] = self._factory(key)
            return [(key, m) for m in tr.add(t, text)]

    def tick(self, now: float) -> list[tuple[str, Moment]]:
        with self._lock:
            return [(key, m) for key, tr in self._trackers.items() for m in tr.tick(now)]

    def drop(self, key: str) -> list[tuple[str, Moment]]:
        with self._lock:
            tr = self._trackers.pop(key, None)
            return [(key, m) for m in tr.flush()] if tr else []

    def rates(self) -> dict[str, int]:
        with self._lock:
            return {k: tr.rate for k, tr in self._trackers.items()}

    def in_spike(self, key: str) -> bool:
        with self._lock:
            tr = self._trackers.get(key)
            return bool(tr and tr.in_spike)
