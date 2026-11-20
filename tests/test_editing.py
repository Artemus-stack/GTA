import re

from clipper.media.captions import Positions, ass_time, build_ass, chunk_words, clean_text
from clipper.media.clipsai_adapter import TranscriptWord as W
from clipper.media.clipsai_adapter import choose_boundaries
from clipper.titles import ClipContext, TitlePack, TitleWriter, polish, template_pack
from clipper.config import TitlesConfig


# -- clip boundaries --------------------------------------------------------------

def test_prefers_clipsai_segment_containing_the_moment():
    event = 60.0
    candidates = [(0.0, 120.0), (10.0, 40.0), (50.0, 82.0), (55.0, 61.0), (45.0, 95.0)]
    start, end, how = choose_boundaries(event, 30, 120, candidates, [])
    assert how == "clipsai"
    assert (start, end) == (50.0, 82.0)       # (45, 95) is 50 s: too long for a 30 s target


def test_rejects_segment_that_cuts_off_the_reaction():
    # (40, 62) ends 2 s after the moment, too soon for the reaction
    start, end, how = choose_boundaries(60.0, 30, 120, [(40.0, 62.0)], [])
    assert how == "fixed"
    assert start < 60 < end and abs((end - start) - 30) < 0.01


def test_snaps_to_sentence_edges():
    sentences = [(40.0, 47.5), (48.3, 55.0), (55.2, 66.0), (66.4, 79.1), (79.5, 90.0)]
    start, end, how = choose_boundaries(60.0, 30, 120, [], sentences)
    assert how == "snapped"
    assert start == 48.3 and end == 79.1


def test_clamps_to_window_edges():
    start, end, _ = choose_boundaries(3.0, 30, 120, [], [])
    assert start == 0.0 and end == 30.0
    start, end, _ = choose_boundaries(118.0, 30, 120, [], [])
    assert end == 120.0 and start == 90.0
    start, end, _ = choose_boundaries(10.0, 45, 20, [], [])     # window shorter than the target
    assert (start, end) == (0.0, 20.0)


# -- captions ---------------------------------------------------------------------

WORDS = [W("bro", 0.2, 0.5), W("did", 0.5, 0.7), W("you", 0.7, 0.9), W("see", 0.9, 1.3), W("that?!", 1.3, 1.9),
         W("no", 2.8, 3.0), W("way", 3.0, 3.4), W("{evil}", 3.5, 3.8), W("😂", 3.8, 4.0)]
POS = Positions(captions_y=1400, hook_y=300)


def test_ass_time():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(65.432) == "0:01:05.43"
    assert ass_time(3725.5) == "1:02:05.50"


def test_clean_text_strips_emoji_and_ass_syntax():
    assert clean_text("He did NOT 💀 {\\b1}", upper=True) == "HE DID NOT (/B1)"


def test_chunks_break_on_gaps_and_punctuation():
    chunks = chunk_words(WORDS, max_words=3, max_chars=16)
    texts = [[w.text for w in c] for c in chunks]
    assert texts[0] == ["bro", "did", "you"]
    assert texts[1] == ["see", "that?!"]           # ends on punctuation
    assert texts[2][0] == "no"                     # 0.9 s pause starts a new chunk
    assert all("😂" not in t for c in texts for t in c)


def _events(ass: str) -> list[str]:
    return [l for l in ass.splitlines() if l.startswith("Dialogue:")]


def test_pop_style_highlights_each_word_in_turn():
    ass = build_ass(WORDS, "pop", 5.0, POS, hook="He did NOT expect that 💀", credit="twitch.tv/x")
    events = _events(ass)
    hook = [e for e in events if ",Hook," in e]
    assert hook and "HE DID NOT EXPECT THAT" in hook[0] and "💀" not in hook[0]
    assert any(",Credit," in e and "twitch.tv/x" in e for e in events)
    pops = [e for e in events if ",Pop," in e]
    assert len(pops) == 8                            # one event per spoken word (emoji-only word dropped)
    assert "{\\c&H0000E6FF&}BRO{\\c&H00FFFFFF&} DID YOU" in pops[0]
    assert "BRO {\\c&H0000E6FF&}DID" in pops[1]
    assert all("{evil}" not in e for e in events)
    starts = [e.split(",")[1] for e in pops]
    assert starts == sorted(starts)


def test_karaoke_timings_cover_the_words():
    ass = build_ass(WORDS[:5], "karaoke", 5.0, POS)
    line = _events(ass)[0]
    durations = [int(x) for x in re.findall(r"\\kf?(\d+)", line)]
    assert abs(sum(durations) - (1.9 - 0.2) * 100) <= 2


def test_clean_style_is_sentence_case():
    ass = build_ass(WORDS[:5], "clean", 5.0, POS)
    assert "Bro did you see that?!" in _events(ass)[0]


# -- titles -------------------------------------------------------------------------

CTX = ClipContext(streamer="Somestreamer", platform="twitch", stream_title="GTA 6 day 1", moment_type="funny",
                  peak_mpm=2400, chat_sample=["KEKW (x300)"],
                  transcript="okay okay. Bro did you just see that car fly?! I can't believe it.")


def test_template_titles_use_transcript_and_streamer():
    pack = template_pack(CTX)
    assert pack.quote.startswith('"Bro did you just see that car fly?!"')
    assert "Somestreamer" in pack.curiosity and "Somestreamer" in pack.reaction
    assert pack.hook == "CHAT LOST IT"
    assert all(len(t) <= 90 for t in (pack.quote, pack.curiosity, pack.reaction))


def test_polish_cleans_model_output():
    pack = polish(TitlePack(quote="#gta6 He <really> did that", curiosity="x" * 200, reaction="ok",
                            hook="WAIT 💀 FOR #IT", description="d", hashtags=["#GTA6", "gta6", "Big Plays!"]))
    assert pack.quote == "He really did that"
    assert len(pack.curiosity) == 90
    assert pack.hook == "WAIT FOR IT"
    assert pack.hashtags == ["gta6", "bigplays"]


class _FakeResponse:
    def __init__(self, parsed, stop_reason="end_turn"):
        self.parsed_output, self.stop_reason = parsed, stop_reason


class _FakeClient:
    def __init__(self, response):
        self.calls = []
        outer = self

        class Messages:
            def parse(self, **kwargs):
                outer.calls.append(kwargs)
                return response

        class Beta:
            messages = Messages()

        self.beta = Beta()


def test_title_writer_calls_claude_with_structured_output():
    parsed = TitlePack(quote="q title", curiosity="c title", reaction="r title", hook="NO WAY",
                       description="desc", hashtags=["gta6"])
    client = _FakeClient(_FakeResponse(parsed))
    pack, source = TitleWriter(TitlesConfig(enabled=True), client=client).write(CTX)
    assert source == "claude" and pack.quote == "q title"
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_format"] is TitlePack
    assert call["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in call["betas"]
    assert call["output_config"] == {"effort": "medium"}
    assert "<transcript>" in call["messages"][0]["content"]


def test_title_writer_falls_back_on_refusal():
    client = _FakeClient(_FakeResponse(None, stop_reason="refusal"))
    pack, source = TitleWriter(TitlesConfig(enabled=True), client=client).write(CTX)
    assert source == "template" and pack.hook == "CHAT LOST IT"
