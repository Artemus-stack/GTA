"""ClipsAI glue: transcribe the downloaded window, pick clip boundaries around the
chat spike, and cut the clip.

ClipsAI's own "find clips" is TextTiling over the transcript (topic shifts). It
was built for podcasts, so on its own it doesn't know which part of a gaming
stream is viral. Chat velocity supplies *where* the moment is; ClipsAI supplies
*clean edges*: a segment that starts and ends on natural transcript boundaries
instead of mid-sentence. ClipsAI's MediaEditor does the cut, and its speaker
tracking (``resize``) powers the optional "speaker" layout.

ClipsAI 0.2.1 dates from January 2024; ``_patch_clipsai`` papers over the three
places current dependencies broke it.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("editor.clipsai")

_patched = False


def _ensure_nltk() -> None:
    # ClipsAI downloads "punkt" at import; NLTK >= 3.9 also needs "punkt_tab" for sent_tokenize.
    import nltk

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            if not nltk.download(pkg, quiet=True):
                log.warning("NLTK could not download %r; behind a proxy set NLTK_ALLOW_PROXIED_URLOPEN=1 "
                            "or run: python -m nltk.downloader punkt punkt_tab", pkg)


def _patch_clipsai() -> None:
    global _patched
    if _patched:
        return
    _ensure_nltk()
    import clipsai.clip.clipfinder as clipfinder
    from clipsai.clip.text_embedder import TextEmbedder

    # 1. ClipFinder builds a new TextEmbedder (a 1.4 GB sentence-transformer) for
    #    every call. Keep one.
    cache: dict[str, TextEmbedder] = {}

    def cached_embedder() -> TextEmbedder:
        if "model" not in cache:
            cache["model"] = TextEmbedder()
        return cache["model"]

    clipfinder.TextEmbedder = cached_embedder

    # 2. pyannote.audio 4 renamed from_pretrained(use_auth_token=) to token= (speaker layout only).
    try:
        import clipsai.diarize.pyannote as diarize
        from pyannote.audio import Pipeline

        if "use_auth_token" not in inspect.signature(Pipeline.from_pretrained).parameters:
            class _PipelineCompat:
                @staticmethod
                def from_pretrained(checkpoint, use_auth_token=None, **kwargs):
                    if use_auth_token and "token" not in kwargs:
                        kwargs["token"] = use_auth_token
                    return Pipeline.from_pretrained(checkpoint, **kwargs)

            diarize.Pipeline = _PipelineCompat
    except Exception as e:  # noqa: BLE001 - only the optional speaker layout depends on this
        log.debug("pyannote compat patch skipped: %s", e)
    # 3. mediapipe >= 0.10.30 dropped mp.solutions, which ClipsAI's face tracker
    #    uses. That can't be patched; speaker_available() reports it instead.
    _patched = True


def speaker_available() -> str | None:
    """None if the ClipsAI speaker layout can run, otherwise the reason it can't."""
    try:
        import mediapipe as mp
    except ImportError:
        return "mediapipe is not installed"
    if not hasattr(mp, "solutions"):
        return "ClipsAI's speaker tracking needs a mediapipe with the legacy 'solutions' API (mediapipe<=0.10.21)"
    return None


@dataclass
class TranscriptWord:
    text: str
    start: float
    end: float


class ClipsAI:
    """Loads the Whisper model once and reuses it for every clip."""

    def __init__(self, model_size: str = "small", device: str | None = None, language: str | None = "en"):
        _patch_clipsai()
        from clipsai import Transcriber

        self.device = device
        self.language = language
        log.info("loading WhisperX %s through ClipsAI (first run downloads the model)", model_size)
        self.transcriber = Transcriber(model_size=model_size, device=device)

    def transcribe(self, path: Path):
        from clipsai.transcribe.exceptions import NoSpeechError

        try:
            return self.transcriber.transcribe(audio_file_path=str(path), iso6391_lang_code=self.language)
        except NoSpeechError:
            log.info("no speech in %s; the clip gets no captions", path.name)
            return None

    def find_clips(self, transcription, target: float) -> list[tuple[float, float]]:
        from clipsai import ClipFinder

        finder = ClipFinder(device=self.device, min_clip_duration=max(5, int(target * 0.6)),
                            max_clip_duration=int(target * 1.3) + 1)
        try:
            clips = finder.find_clips(transcription=transcription)
        except Exception as e:  # noqa: BLE001 - too few sentences etc.; fall back to snapping
            log.info("ClipFinder found nothing usable (%s)", e)
            return []
        return [(float(c.start_time), float(c.end_time)) for c in clips]

    def trim(self, src: Path, start: float, end: float, dst: Path) -> Path:
        from clipsai import AudioVideoFile, MediaEditor

        media = AudioVideoFile(str(src))
        end = min(end, media.get_duration() - 0.05)
        out = MediaEditor().trim(
            media_file=media, start_time=start, end_time=end, trimmed_media_file_path=str(dst),
            video_codec="libx264", audio_codec="aac", crf="18", preset="veryfast",
        )
        if out is None:
            raise RuntimeError("ClipsAI trim failed")
        return dst

    def speaker_crop(self, src: Path, dst: Path, hf_token: str) -> Path:
        """ClipsAI's speaker-tracking 9:16 crop (diarization + face detection)."""
        from clipsai import MediaEditor, VideoFile, resize

        crops = resize(video_file_path=str(src), pyannote_auth_token=hf_token, aspect_ratio=(9, 16),
                       device=self.device)
        out = MediaEditor().resize_video(
            original_video_file=VideoFile(str(src)), resized_video_file_path=str(dst),
            width=crops.crop_width, height=crops.crop_height,
            segments=[s.to_dict() for s in crops.segments],
        )
        if out is None:
            raise RuntimeError("ClipsAI resize_video failed")
        return dst


def sentences_of(transcription) -> list[tuple[float, float]]:
    if transcription is None:
        return []
    out = []
    for s in transcription.get_sentence_info():
        if s.get("start_time") is not None and s.get("end_time") is not None:
            out.append((float(s["start_time"]), float(s["end_time"])))
    return out


def words_between(transcription, start: float, end: float) -> list[TranscriptWord]:
    """Words inside [start, end], re-timed so the clip starts at 0."""
    if transcription is None:
        return []
    words = []
    for w in transcription.get_word_info():
        ws, we = w.get("start_time"), w.get("end_time")
        text = (w.get("word") or "").strip()
        if ws is None or we is None or not text or we <= start or ws >= end:
            continue
        words.append(TranscriptWord(text, max(0.0, ws - start), min(end, max(we, ws + 0.05)) - start))
    for i, w in enumerate(words):
        if w.end <= w.start:
            w.end = w.start + 0.2
        if i + 1 < len(words) and w.end > words[i + 1].start:
            w.end = max(w.start + 0.05, words[i + 1].start)
    return words


def choose_boundaries(event_t: float, target: float, window_len: float,
                      candidates: list[tuple[float, float]],
                      sentences: list[tuple[float, float]]) -> tuple[float, float, str]:
    """Pick [start, end] for the clip inside the downloaded window.

    ``event_t`` is the estimated time of the moment itself (chat onset minus the
    reaction lag). Clips put the moment about a third of the way in: enough
    setup to understand it, then the payoff and the reaction.

    1. A ClipsAI segment that contains the moment, with setup before and
       reaction after, close to the target length.
    2. Otherwise a target-length window snapped to transcript sentence edges.
    3. Otherwise (no speech) the same window, unsnapped.
    """
    lead = min(0.35 * target, 12.0)
    best = None
    for s, e in candidates:
        dur = e - s
        if not (0.6 * target <= dur <= 1.3 * target):
            continue
        if not (s <= event_t - 2.0 and e >= event_t + 4.0):
            continue
        pos = (event_t - s) / dur
        cost = abs(dur - target) / target + abs(pos - lead / target)
        if best is None or cost < best[0]:
            best = (cost, s, e)
    if best is not None and best[0] < 0.8:
        start, end = _clamp(best[1], best[2], window_len)
        return start, end, "clipsai"

    start, end = event_t - lead, event_t - lead + target
    how = "fixed"
    if sentences:
        starts = [s for s, _ in sentences if start - 4.0 <= s <= start + 1.5]
        ends = [e for _, e in sentences if end - 3.0 <= e <= end + 4.0 and e > event_t + 3.0]
        if starts:
            start = min(starts, key=lambda s: abs(s - start) + (0.5 if s > start else 0.0))
            how = "snapped"
        if ends:
            end = min(ends, key=lambda e: abs(e - end))
            how = "snapped"
    start, end = _clamp(start, end, window_len)
    return start, end, how


def _clamp(start: float, end: float, window_len: float) -> tuple[float, float]:
    length = end - start
    start = max(0.0, start)
    end = min(window_len, max(end, start + min(length, window_len)))
    if end - start < min(length, window_len):     # pinned against the end of the window
        start = max(0.0, end - length)
    return round(start, 2), round(end, 2)
