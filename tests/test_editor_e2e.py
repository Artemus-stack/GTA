"""End-to-end editor run: yt-dlp -> ClipsAI -> ffmpeg, against a local "VOD".

Uses the real yt-dlp section download, the real ClipsAI code (Transcription,
sentence splitting, TextTiling ClipFinder, MediaEditor.trim) and the real
renderer. Only the model weights are stubbed: WhisperX's ASR/alignment models
and the sentence-embedding model, which would otherwise be downloaded from
Hugging Face.
"""

import functools
import http.server
import json
import re
import subprocess
import threading
from pathlib import Path

import pytest

pytest.importorskip("clipsai")
pytest.importorskip("whisperx")
torch = pytest.importorskip("torch")

from clipper.agents.editor import Editor                     # noqa: E402
from clipper.formats import FormatBandit                     # noqa: E402
from clipper.media.download import NotReady, Skip, download_section  # noqa: E402
from clipper.media.render import probe                       # noqa: E402
from clipper.titles import TitlePack, TitleWriter            # noqa: E402

TOPIC_A = [f"We are driving the stolen car down the highway number {i} right now." for i in range(10)]
TOPIC_B = [f"Oh no the police helicopter is chasing us over the bridge wave {i} now." for i in range(11)]
TOPIC_C = [f"Okay we lost them so let us go buy some new clothes at the store {i}." for i in range(7)]
# Sentences are 4.2 s apart, so topic B spans ~42-88 s of the downloaded window,
# around the moment at 67 s: topic changes ClipsAI's TextTiling can find.
SENTENCES = TOPIC_A + TOPIC_B + TOPIC_C


@pytest.fixture(scope="module")
def vod_server(tmp_path_factory):
    root = tmp_path_factory.mktemp("vod")
    # 180 s served as HLS, the way Twitch delivers VODs; a beep exactly at
    # 100.0-100.5 s lets the test check cut precision
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
        "-f", "lavfi", "-i", "aevalsrc='if(between(t,100,100.5),0.8*sin(2*PI*1000*t),0)':s=16000:d=180",
        "-t", "180", "-c:v", "libx264", "-preset", "ultrafast", "-g", "60", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-f", "hls", "-hls_time", "4", "-hls_playlist_type", "vod",
        "-hls_segment_filename", str(root / "seg%03d.ts"), str(root / "vod.m3u8"),
    ], check=True)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def handle(self):
            try:
                super().handle()
            except ConnectionError:      # ffmpeg hangs up mid-segment when it has enough
                pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/vod.m3u8"
    server.shutdown()


def beep_at(path: Path) -> float:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "silencedetect=noise=-30dB:d=0.2",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    return float(re.search(r"silence_end: ([\d.]+)", out).group(1))


def test_section_download_is_frame_accurate(vod_server, tmp_path):
    window = download_section(vod_server, 90, 110, tmp_path)
    assert abs(probe(window).duration - 20) < 0.25
    assert abs(beep_at(window) - 10.0) < 0.15      # the beep at 100 s lands 10 s into the window


class FakeWhisperModel:
    def transcribe(self, path, language=None, batch_size=16):
        segs = [{"start": i * 4.2, "end": i * 4.2 + 3.6, "text": " " + s} for i, s in enumerate(SENTENCES)]
        return {"segments": segs, "language": language or "en"}


def fake_align(segments, model, metadata, audio, device, return_char_alignments=False, **kw):
    out = []
    for seg in segments:
        text = seg["text"] if seg["text"].startswith(" ") else " " + seg["text"]
        letters = [i for i, c in enumerate(text) if c != " "]
        step = (seg["end"] - seg["start"]) / max(len(letters), 1)
        chars, k = [], 0
        for c in text:
            if c == " ":
                chars.append({"char": c})          # whisperx leaves spaces untimed
            else:
                t = seg["start"] + k * step
                chars.append({"char": c, "start": round(t, 3), "end": round(t + step, 3), "score": 0.9})
                k += 1
        out.append({**seg, "words": [], "chars": chars})
    return {"segments": out, "word_segments": []}


def fake_embed(self, sentences):
    gen = torch.Generator().manual_seed(0)
    rows = []
    for s in sentences:
        base = torch.zeros(32)
        base[0 if "driving" in s else 1 if "helicopter" in s else 2] = 1.0
        rows.append(base + 0.05 * torch.randn(32, generator=gen))
    return torch.stack(rows)


@pytest.fixture
def stub_models(monkeypatch):
    import whisperx
    from clipsai.clip import text_embedder

    monkeypatch.setattr(whisperx, "load_model", lambda *a, **k: FakeWhisperModel())
    monkeypatch.setattr(whisperx, "load_align_model", lambda *a, **k: (None, {}))
    monkeypatch.setattr(whisperx, "align", fake_align)
    monkeypatch.setattr(text_embedder.TextEmbedder, "__init__", lambda self: None)
    monkeypatch.setattr(text_embedder.TextEmbedder, "embed_sentences", fake_embed)


@pytest.fixture
def boundaries(monkeypatch):
    import clipper.agents.editor as editor_mod

    seen = {}
    real = editor_mod.choose_boundaries

    def spy(event_t, target, window_len, candidates, sentences):
        out = real(event_t, target, window_len, candidates, sentences)
        seen.update(event_t=event_t, start=out[0], end=out[1], how=out[2], window=window_len,
                    candidates=candidates)
        return out

    monkeypatch.setattr(editor_mod, "choose_boundaries", spy)
    return seen


class FakeClaude:
    class beta:
        class messages:
            @staticmethod
            def parse(**kwargs):
                class R:
                    stop_reason = "end_turn"
                    parsed_output = TitlePack(
                        quote="\"Oh no the police helicopter\" 😱", curiosity="Tester did NOT see this coming...",
                        reaction="Chat lost it when the chopper showed up", hook="WAIT FOR THE CHOPPER",
                        description="GTA 6 chaos.", hashtags=["gta6", "police"])
                return R()


def queue_manual_moment(db, url, onset_offset, mid="manual-vod-1"):
    db.insert("moments", {
        "id": mid, "platform": "other", "channel": "tester", "channel_display": "Tester",
        "source_url": url, "onset_offset": onset_offset, "peak_mpm": 1800, "score": 1e9,
        "moment_type": "shock", "chat_sample": json.dumps(["WTF (x200)", "NO WAY (x150)"]),
    })
    return mid


@pytest.mark.parametrize("layout,captions", [("blur", "pop"), ("stack", "karaoke")])
def test_editor_turns_a_moment_into_a_titled_vertical_clip(cfg, db, vod_server, stub_models, boundaries,
                                                           layout, captions):
    cfg.analyst.arms = {"layout": [layout], "captions": [captions], "length": ["45"], "hook": ["banner"]}
    cfg.editor.whisper_model = "tiny"
    cfg.editor.facecams = {"tester": [0.70, 0.02, 0.28, 0.30]}
    cfg.titles.enabled = True
    mid = queue_manual_moment(db, vod_server, onset_offset=100.0)
    editor = Editor(cfg, db, FormatBandit(db, cfg.analyst.arms), titles=TitleWriter(cfg.titles, client=FakeClaude()))

    assert editor.process_next() is True
    moment = db.one("SELECT * FROM moments WHERE id = ?", (mid,))
    assert moment["status"] == "edited", moment["note"]
    clip = db.one("SELECT * FROM clips WHERE moment_id = ?", (mid,))
    out = probe(Path(clip["path"]))
    assert (out.width, out.height) == (1080, 1920) and out.has_audio
    assert 0.6 * 45 <= clip["duration"] <= 1.5 * 45
    assert abs(out.duration - clip["duration"]) < 0.5
    fmt = json.loads(clip["format"])
    assert fmt == {"layout": layout, "captions": captions, "length": "45", "hook": "banner"}
    titles = json.loads(clip["titles"])
    assert set(titles) == {"quote", "curiosity", "reaction"} and clip["title_source"] == "claude"
    assert clip["hook"] == "WAIT FOR THE CHOPPER"
    assert "helicopter" in clip["transcript"]
    # ClipsAI's topic segment around the moment supplied the edges
    assert boundaries["how"] == "clipsai" and boundaries["candidates"]
    assert boundaries["start"] + 2 <= boundaries["event_t"] <= boundaries["end"] - 4
    assert clip["source_url"].startswith(vod_server)
    assert not (Path(cfg.data_dir) / "work" / mid).exists()          # scratch files cleaned up


def test_clip_window_contains_the_moment(cfg, db, vod_server, stub_models, boundaries):
    cfg.analyst.arms = {"layout": ["crop"], "captions": ["clean"], "length": ["30"], "hook": ["none"]}
    cfg.editor.whisper_model = "tiny"
    seen = boundaries
    queue_manual_moment(db, vod_server, onset_offset=100.0, mid="manual-vod-2")
    Editor(cfg, db, FormatBandit(db, cfg.analyst.arms)).process_next()
    assert db.one("SELECT status FROM moments WHERE id = 'manual-vod-2'")["status"] == "edited"
    assert abs(seen["window"] - 120) < 0.5                  # 75 s before + 45 s after the onset
    assert abs(seen["event_t"] - 67) < 0.01                 # (100 - 8) - 25
    assert seen["start"] + 2 <= seen["event_t"] <= seen["end"] - 4
    assert seen["how"] == "snapped"       # topic B (46 s) is too long for 30 s: snapped to sentences
    assert abs((seen["end"] - seen["start"]) - 30) < 4


def test_not_ready_and_skip_paths(cfg, db, monkeypatch):
    import clipper.agents.editor as editor_mod

    monkeypatch.setattr(editor_mod, "probe", lambda url, cookies=None: {"live_status": "is_live"})
    db.insert("moments", {"id": "yt-abc-1", "platform": "youtube", "channel": "x", "video_id": "abcdefghijk",
                          "onset_at": 1e9, "score": 1.0, "chat_sample": "[]"})
    db.insert("moments", {"id": "tw-x-1", "platform": "twitch", "channel": "x", "channel_id": "1",
                          "onset_at": 1e9, "score": 0.5, "chat_sample": "[]"})
    cfg.editor.max_moment_age_h = 1e9
    editor = Editor(cfg, db, FormatBandit(db), clipsai=object())
    editor.process_next()
    editor.process_next()
    yt = db.one("SELECT * FROM moments WHERE id = 'yt-abc-1'")
    assert yt["status"] == "new" and "still live" in yt["note"] and yt["next_attempt_at"] > 1e9
    tw = db.one("SELECT * FROM moments WHERE id = 'tw-x-1'")
    assert tw["status"] == "skipped" and "Twitch API credentials" in tw["note"]
