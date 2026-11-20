import json
import random

import clipper.agents.scout as scout_mod
from clipper.agents.scout import Scout, scan_youtube_replay


def feed_chat(callback, start, seconds, base_mpm, burst_at, burst_rate, burst_len, rng, text="KEKW"):
    t = start
    while t < start + seconds:
        in_burst = burst_at <= t < burst_at + burst_len
        t += rng.expovariate(burst_rate if in_burst else base_mpm / 60)
        callback(t, text if in_burst else "chatting")
    return t


def test_twitch_chat_burst_becomes_a_moment_row(cfg, db):
    scout = Scout(cfg, db)
    info = {"user_id": "42", "login": "somestreamer", "name": "SomeStreamer", "title": "GTA 6 day one",
            "viewers": 25000, "started_at": 1_763_600_000.0}
    scout.streams["twitch:somestreamer"] = info
    rng = random.Random(3)
    t0 = 1_763_640_000.0
    end = feed_chat(lambda t, x: scout._on_twitch("somestreamer", t, x), t0, 900, 300, t0 + 500, 40, 30, rng)
    scout._emit(scout.trackers.tick(end + 120))
    rows = db.query("SELECT * FROM moments")
    assert len(rows) == 1
    m = rows[0]
    assert m["platform"] == "twitch" and m["channel"] == "somestreamer" and m["channel_id"] == "42"
    assert abs(m["onset_at"] - (t0 + 500)) <= 4
    assert m["peak_mpm"] >= 500 and m["moment_type"] == "funny"
    assert m["viewer_count"] == 25000 and m["stream_started_at"] == info["started_at"]
    assert json.loads(m["chat_sample"])[0].startswith("KEKW")
    assert m["status"] == "new"


def test_stream_leaving_the_top_list_mid_burst_still_records(cfg, db):
    scout = Scout(cfg, db)
    scout.streams["twitch:a"] = {"user_id": "1", "login": "a", "name": "A", "title": "", "viewers": 1}
    rng = random.Random(4)
    t0 = 1_000_000.0
    feed_chat(lambda t, x: scout._on_twitch("a", t, x), t0, 520, 200, t0 + 500, 40, 30, rng)
    scout._apply("twitch", [], lambda s: f"twitch:{s['login']}")      # dropped from the top 100
    assert "twitch:a" not in scout.streams
    assert len(db.query("SELECT id FROM moments")) == 1


def test_streamer_filters(cfg, db):
    cfg.scout.streamers.blocklist = ["NoClips"]

    class FakeTwitch:
        def game_ids(self, names):
            return ["999"]

        def top_streams(self, ids, n, langs):
            return [{"login": "noclips", "name": "NoClips"}, {"login": "ok", "name": "Ok"}]

    scout = Scout(cfg, db)
    scout.twitch = FakeTwitch()
    assert [s["login"] for s in scout.refresh_twitch()] == ["ok"]
    cfg.scout.streamers.allowlist = ["friendly"]
    assert scout.refresh_twitch() == []


def test_scan_youtube_replay_queues_exact_offsets(cfg, db, monkeypatch, tmp_path):
    rng = random.Random(5)
    lines, t = [], 0.0
    while t < 1500:
        burst = 900 <= t < 940
        t += rng.expovariate(25 if burst else 3)
        action = {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {
            "message": {"runs": [{"text": "NO WAY" if burst else "gg"}]}, "timestampUsec": "1"}}}}
        lines.append(json.dumps({"replayChatItemAction": {"actions": [action],
                                                          "videoOffsetTimeMsec": str(int(t * 1000))}}))
    chat = tmp_path / "abcdefghijk.live_chat.json"
    chat.write_text("\n".join(lines))
    monkeypatch.setattr(scout_mod, "download_replay_chat",
                        lambda vid, workdir, cookies=None: (chat, {"channel": "Big YouTuber", "channel_id": "UC1",
                                                                   "title": "GTA 6 first look"}))
    rows = scan_youtube_replay(cfg, db, "https://www.youtube.com/watch?v=abcdefghijk")
    assert len(rows) == 1
    r = rows[0]
    assert abs(r["onset_offset"] - 900) <= 3
    assert r["moment_type"] == "shock" and r["channel"] == "Big YouTuber"
    assert r["source_url"] == "https://www.youtube.com/watch?v=abcdefghijk"
    assert db.one("SELECT status FROM moments WHERE id = ?", (r["id"],))["status"] == "new"
    assert not chat.exists()
