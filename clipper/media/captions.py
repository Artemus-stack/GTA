"""Burned-in captions as an ASS subtitle file (rendered by ffmpeg/libass).

Three caption styles for the bandit to test:
  * pop     - 1-3 big uppercase words at a time, the spoken word turns yellow
  * karaoke - a short line that fills green word by word as it is spoken
  * clean   - sentence-case subtitles on a dark box

The same file carries the hook banner and a small streamer credit, so a single
ffmpeg pass renders all text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

WIDTH, HEIGHT = 1080, 1920
YELLOW = "&H0000E6FF&"   # ASS colours are &HBBGGRR&
WHITE = "&H00FFFFFF&"

# libass can't draw colour emoji with normal fonts, and braces/backslashes are ASS syntax
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]")


class WordLike(Protocol):
    text: str
    start: float
    end: float


@dataclass
class Positions:
    captions_y: int
    hook_y: int


def clean_text(text: str, upper: bool = False) -> str:
    t = _EMOJI_RE.sub("", text).replace("{", "(").replace("}", ")").replace("\\", "/")
    t = " ".join(t.split())
    return t.upper() if upper else t


def ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def chunk_words(words: Sequence[WordLike], max_words: int, max_chars: int, max_gap: float = 0.6) -> list[list]:
    chunks: list[list] = []
    cur: list = []
    for w in words:
        if not clean_text(w.text):
            continue
        length = sum(len(x.text) + 1 for x in cur) + len(w.text)
        gap = w.start - cur[-1].end if cur else 0.0
        if cur and (len(cur) >= max_words or length > max_chars or gap > max_gap or cur[-1].text[-1:] in ".?!"):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def _chunk_end(chunks: list[list], i: int, duration: float) -> float:
    """Hold a chunk until the next one starts if the gap is short (avoids flicker)."""
    end = max(chunks[i][-1].end, chunks[i][0].start + 0.45)
    if i + 1 < len(chunks):
        nxt = chunks[i + 1][0].start
        if nxt - end < 0.35:
            end = nxt
        end = min(end, nxt)
    return min(end, duration)


def _header(font: str) -> str:
    styles = [
        # Name, Fontsize, Primary, Secondary, Outline, Back, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL/R/V
        ("Pop", 92, WHITE, WHITE, "&H00000000&", "&H64000000&", 1, 8, 3, 5, 70, 70, 0),
        ("Karaoke", 80, "&H0014FF39&", WHITE, "&H00000000&", "&H64000000&", 1, 7, 2, 5, 70, 70, 0),
        ("Clean", 60, WHITE, WHITE, "&H00000000&", "&H8C000000&", 3, 12, 0, 5, 90, 90, 0),
        ("Hook", 68, WHITE, WHITE, "&H00000000&", "&H26000000&", 3, 18, 0, 8, 90, 90, 0),
        ("Credit", 34, "&H33FFFFFF&", "&H33FFFFFF&", "&H66000000&", "&H00000000&", 1, 2, 0, 9, 30, 30, 0),
    ]
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {WIDTH}", f"PlayResY: {HEIGHT}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
    ]
    for name, size, prim, sec, outl, back, border, outline, shadow, align, ml, mr, mv in styles:
        lines.append(f"Style: {name},{font},{size},{prim},{sec},{outl},{back},-1,0,0,0,100,100,0,0,"
                     f"{border},{outline},{shadow},{align},{ml},{mr},{mv},1")
    lines += ["", "[Events]",
              "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    return "\n".join(lines)


def _dialogue(start: float, end: float, style: str, text: str, layer: int = 0) -> str:
    return f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{text}"


def _pop(words, y: int, duration: float) -> list[str]:
    events = []
    chunks = chunk_words(words, max_words=3, max_chars=16)
    for ci, chunk in enumerate(chunks):
        end = _chunk_end(chunks, ci, duration)
        texts = [clean_text(w.text, upper=True) for w in chunk]
        for i, w in enumerate(chunk):
            s = chunk[0].start if i == 0 else w.start
            e = chunk[i + 1].start if i + 1 < len(chunk) else end
            if e <= s:
                continue
            parts = [f"{{\\c{YELLOW}}}{t}{{\\c{WHITE}}}" if j == i else t for j, t in enumerate(texts)]
            pop_in = "\\fscx80\\fscy80\\t(0,90,\\fscx100\\fscy100)" if i == 0 else ""
            events.append(_dialogue(s, e, "Pop", f"{{\\an5\\pos({WIDTH // 2},{y}){pop_in}}}" + " ".join(parts), 1))
    return events


def _karaoke(words, y: int, duration: float) -> list[str]:
    events = []
    chunks = chunk_words(words, max_words=5, max_chars=24)
    for ci, chunk in enumerate(chunks):
        start, end = chunk[0].start, _chunk_end(chunks, ci, duration)
        if end <= start:
            continue
        text, cursor = "", start
        for w in chunk:
            if w.start > cursor:
                text += f"{{\\k{int(round((w.start - cursor) * 100))}}}"
            text += f"{{\\kf{max(1, int(round((w.end - w.start) * 100)))}}}{clean_text(w.text, upper=True)} "
            cursor = max(cursor, w.end)
        events.append(_dialogue(start, end, "Karaoke", f"{{\\an5\\pos({WIDTH // 2},{y})}}{text.strip()}", 1))
    return events


def _clean(words, y: int, duration: float) -> list[str]:
    events = []
    chunks = chunk_words(words, max_words=8, max_chars=34, max_gap=0.8)
    for ci, chunk in enumerate(chunks):
        start, end = chunk[0].start, _chunk_end(chunks, ci, duration)
        if end <= start:
            continue
        text = clean_text(" ".join(w.text for w in chunk))
        if text:
            text = text[0].upper() + text[1:]
        events.append(_dialogue(start, end, "Clean", f"{{\\an5\\pos({WIDTH // 2},{y})}}{text}", 1))
    return events


def build_ass(words: Sequence[WordLike], style: str, duration: float, positions: Positions, *,
              hook: str | None = None, credit: str | None = None, font: str = "DejaVu Sans") -> str:
    events: list[str] = []
    if hook:
        hook_text = clean_text(hook, upper=True)
        if hook_text:
            events.append(_dialogue(0, duration, "Hook", f"{{\\an8\\pos({WIDTH // 2},{positions.hook_y})}}{hook_text}", 2))
    if credit:
        events.append(_dialogue(0, duration, "Credit", f"{{\\an9\\pos({WIDTH - 30},36)}}{clean_text(credit)}", 2))
    render = {"pop": _pop, "karaoke": _karaoke, "clean": _clean}.get(style, _pop)
    events += render(words, positions.captions_y, duration)
    return _header(font) + "\n" + "\n".join(events) + "\n"
