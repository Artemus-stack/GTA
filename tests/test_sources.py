import json

import httpx
import pytest

from clipper.sources.twitch import TwitchAPI, TwitchChat, parse_duration, parse_irc
from clipper.sources.youtube import YouTubeChatPool, iter_replay_messages, parse_chat_line


# -- Twitch ---------------------------------------------------------------------

def test_parse_irc_privmsg_with_tags():
    line = ("@badge-info=;color=#FF0000;display-name=Viewer;tmi-sent-ts=1763650000123 "
            ":viewer!viewer@viewer.tmi.twitch.tv PRIVMSG #somestreamer :KEKW he really did that :D")
    tags, prefix, command, params = parse_irc(line)
    assert command == "PRIVMSG"
    assert params == ["#somestreamer", "KEKW he really did that :D"]
    assert tags["display-name"] == "Viewer"
    assert prefix.startswith("viewer!")


def test_parse_irc_ping():
    assert parse_irc("PING :tmi.twitch.tv")[2:] == ("PING", ["tmi.twitch.tv"])


def test_parse_duration():
    assert parse_duration("3h2m1s") == 10921
    assert parse_duration("45m") == 2700
    assert parse_duration("") == 0


def _helix_transport(requests):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "id.twitch.tv":
            return httpx.Response(200, json={"access_token": "app-token", "expires_in": 5000})
        path = request.url.path
        if path.endswith("/games"):
            return httpx.Response(200, json={"data": [{"id": "999", "name": "Grand Theft Auto VI"}]})
        if path.endswith("/streams"):
            after = request.url.params.get("after")
            if after is None:
                data = [{"user_id": str(i), "user_login": f"s{i}", "user_name": f"S{i}", "title": "gta 6!",
                         "viewer_count": 1000 - i, "started_at": "2026-11-20T10:00:00Z", "language": "en"}
                        for i in range(100)]
                return httpx.Response(200, json={"data": data, "pagination": {"cursor": "page2"}})
            data = [{"user_id": "x", "user_login": "late", "user_name": "Late", "title": "", "viewer_count": 5000,
                     "started_at": "2026-11-20T10:00:00Z", "language": "en"}]
            return httpx.Response(200, json={"data": data, "pagination": {}})
        if path.endswith("/videos"):
            return httpx.Response(200, json={"data": [
                {"id": "v2", "url": "https://www.twitch.tv/videos/v2", "created_at": "2026-11-20T10:00:00Z",
                 "duration": "2h0m0s"},
                {"id": "v1", "url": "https://www.twitch.tv/videos/v1", "created_at": "2026-11-19T10:00:00Z",
                 "duration": "5h0m0s"},
            ]})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_helix_top_streams_pages_and_sorts():
    requests = []
    api = TwitchAPI("cid", "secret", http=httpx.Client(transport=_helix_transport(requests)))
    assert api.game_ids(["Grand Theft Auto VI"]) == ["999"]
    streams = api.top_streams(["999"], limit=100, languages=["en"])
    assert len(streams) == 100
    assert sum(r.url.path.endswith("/streams") for r in requests) == 1   # one page is enough for 100
    more = api.top_streams(["999"], limit=150)
    assert len(more) == 101 and more[0]["login"] == "late"               # pages until exhausted, sorted
    stream_req = next(r for r in requests if r.url.path.endswith("/streams"))
    assert stream_req.headers["Client-Id"] == "cid"
    assert stream_req.headers["Authorization"] == "Bearer app-token"
    assert stream_req.url.params.get_list("language") == ["en"]


def test_find_vod_covers_live_moment():
    api = TwitchAPI("cid", "secret", http=httpx.Client(transport=_helix_transport([])))
    from clipper.sources.twitch import parse_ts
    vod = api.find_vod("1", parse_ts("2026-11-20T11:30:00Z"))
    assert vod["id"] == "v2" and vod["url"].endswith("/v2")
    assert parse_ts("2026-11-20T11:30:00Z") - vod["created_at"] == 5400
    assert api.find_vod("1", parse_ts("2026-11-18T00:00:00Z")) is None


def test_twitch_chat_shards_channels():
    chat = TwitchChat(lambda *a: None, per_connection=2)
    chat.set_channels(["a", "B", "c", "d", "e"])
    assert chat.channels == {"a", "b", "c", "d", "e"}
    assert len(chat.connections) == 3
    chat.set_channels(["a", "e", "f"])
    assert chat.channels == {"a", "e", "f"}
    assert all(len(c.wanted) <= 2 for c in chat.connections)


# -- YouTube chat ------------------------------------------------------------------

def _text_action(text, usec=1763650000000000, emoji=None):
    runs = [{"text": text}] + ([{"emoji": emoji}] if emoji else [])
    return {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {
        "message": {"runs": runs}, "timestampUsec": str(usec), "authorName": {"simpleText": "x"}}}}}


def test_parse_replay_line():
    line = json.dumps({"replayChatItemAction": {
        "actions": [_text_action("NO WAY", emoji={"emojiId": "😱", "shortcuts": [":scream:"]})],
        "videoOffsetTimeMsec": "123456"}})
    assert parse_chat_line(line) == [(1763650000.0, 123.456, "NO WAY😱")]


def test_parse_live_line_drops_relative_offset():
    line = json.dumps({"replayChatItemAction": {"actions": [_text_action("W")]},
                       "videoOffsetTimeMsec": "5000", "isLive": True})
    assert parse_chat_line(line) == [(1763650000.0, None, "W")]


def test_parse_ignores_non_text_and_garbage():
    assert parse_chat_line(json.dumps({"replayChatItemAction": {"actions": [{"addBannerToLiveChatCommand": {}}]}})) == []
    assert parse_chat_line("not json") == []


def test_iter_replay_messages(tmp_path):
    p = tmp_path / "v.live_chat.json"
    p.write_text("\n".join(json.dumps({"replayChatItemAction": {
        "actions": [_text_action(f"m{i}")], "videoOffsetTimeMsec": str(i * 1000)}}) for i in range(3)))
    assert list(iter_replay_messages(p)) == [(0.0, "m0"), (1.0, "m1"), (2.0, "m2")]


def test_chat_pool_tails_partial_lines(tmp_path):
    got = []
    pool = YouTubeChatPool(tmp_path, lambda key, ts, text: got.append((key, text)))

    class FakeReader:
        def __init__(self, path):
            self.path, self.offset, self.buffer, self.error, self.started_at = path, 0, b"", None, 0

        def is_alive(self):
            return True

    from clipper.sources.youtube import _ChatReader
    reader = FakeReader(tmp_path / "vid.live_chat.json")
    reader.read_new = _ChatReader.read_new.__get__(reader)
    pool.readers["vid"], pool.keys["vid"] = reader, "youtube:vid"
    line = json.dumps({"replayChatItemAction": {"actions": [_text_action("first")]}, "isLive": True})
    with open(reader.path, "wb") as f:
        f.write(line.encode() + b"\n" + line[:20].encode())     # second line only half written
    pool.poll()
    assert got == [("youtube:vid", "first")]
    with open(reader.path, "ab") as f:
        f.write(line[20:].encode() + b"\n")
    pool.poll()
    assert got == [("youtube:vid", "first"), ("youtube:vid", "first")]


@pytest.mark.parametrize("url,vid", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ?t=10", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
])
def test_youtube_id(url, vid):
    from clipper.agents.scout import _youtube_id
    assert _youtube_id(url) == vid
