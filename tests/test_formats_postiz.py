import json
import random
import time

import httpx
import pytest

from clipper.config import PublisherConfig
from clipper.formats import FormatBandit
from clipper.postiz import Postiz, PostizError, settings_for, to_html


# -- bandit -------------------------------------------------------------------------

def _judged(n, platform="tiktok", winner="stack", now=None):
    now = now or time.time()
    posts = []
    rng = random.Random(7)
    for i in range(n):
        layout = rng.choice(["blur", "stack", "crop"])
        hit = int(rng.random() < (0.8 if layout == winner else 0.2))
        posts.append({"platform": platform, "hit": hit, "created_at": now - i * 3600, "score": 1000 * hit,
                      "format": {"layout": layout, "captions": "pop", "length": "30", "hook": "banner"},
                      "title_style": "curiosity" if hit else "quote"})
    return posts


def test_bandit_learns_the_winning_layout(db):
    bandit = FormatBandit(db, rng=random.Random(1))
    bandit.recompute(_judged(120), half_life_days=10, now=time.time())
    picks = [bandit.choose_format()["layout"] for _ in range(300)]
    assert picks.count("stack") > 240
    probs = bandit.win_probabilities("format", "layout", ["blur", "stack", "crop"])
    assert probs["stack"] > 0.95


def test_bandit_explores_when_nothing_is_known(db):
    bandit = FormatBandit(db, rng=random.Random(2))
    picks = [bandit.choose_format()["captions"] for _ in range(300)]
    assert all(picks.count(c) > 50 for c in ("pop", "karaoke", "clean"))


def test_title_style_is_learned_per_platform(db):
    bandit = FormatBandit(db, rng=random.Random(3))
    bandit.recompute(_judged(80, platform="youtube"), half_life_days=10, now=time.time())
    yt = [bandit.choose_title_style("youtube") for _ in range(200)]
    tt = [bandit.choose_title_style("tiktok") for _ in range(200)]
    assert yt.count("curiosity") > 170
    assert tt.count("curiosity") < 120          # no TikTok data yet: still exploring


def test_old_results_fade(db):
    now = time.time()
    bandit = FormatBandit(db)
    old = [{"platform": "tiktok", "hit": 1, "created_at": now - 90 * 86400, "score": 1,
            "format": {"layout": "crop"}, "title_style": "quote"}] * 30
    state = bandit.recompute(old, half_life_days=10, now=now)
    assert state["format"]["layout"]["crop"]["a"] < 1.1


def test_layouts_can_be_restricted(db):
    bandit = FormatBandit(db)
    assert {bandit.choose_format(["blur"])["layout"] for _ in range(50)} == {"blur"}


def test_report_flags_the_winner(db):
    bandit = FormatBandit(db, rng=random.Random(4))
    bandit.recompute(_judged(120), half_life_days=10, now=time.time())
    rows = bandit.report(["tiktok"])
    stack = next(r for r in rows if r[1] == "layout" and r[2] == "stack")
    assert stack[8] == "WINNING"
    assert any(r[0] == "tiktok" and r[1] == "title style" for r in rows)


# -- Postiz client ---------------------------------------------------------------------

class Recorder:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = (request.method, request.url.path)
        resp = self.responses[key]
        return resp.pop(0) if isinstance(resp, list) else resp


def _client(responses):
    rec = Recorder(responses)
    return Postiz("https://postiz.example.com/api", "secret-key", http=httpx.Client(transport=httpx.MockTransport(rec))), rec


def test_upload_and_create_post_shapes(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 100)
    pz, rec = _client({
        ("POST", "/api/public/v1/upload"): httpx.Response(200, json={"id": "m1", "path": "https://cdn/x.mp4"}),
        ("POST", "/api/public/v1/posts"): httpx.Response(200, json=[{"postId": "p1", "integration": "int1"}]),
    })
    media = pz.upload(video)
    settings = settings_for("tiktok", "A title", PublisherConfig(), ["gta6"])
    out = pz.create_post("int1", to_html("line one\nline <two>"), media, settings, "2026-11-20T12:00:00.000Z")
    assert out == [{"postId": "p1", "integration": "int1"}]
    upload, post = rec.requests
    assert upload.headers["Authorization"] == "secret-key"
    assert b'name="file"; filename="clip.mp4"' in upload.content
    body = json.loads(post.content)
    assert body["type"] == "schedule" and body["date"] == "2026-11-20T12:00:00.000Z"
    assert body["shortLink"] is False and body["tags"] == []
    p = body["posts"][0]
    assert p["integration"] == {"id": "int1"}
    assert p["value"][0]["content"] == "<p>line one</p><p>line &lt;two&gt;</p>"
    assert p["value"][0]["image"] == [{"id": "m1", "path": "https://cdn/x.mp4"}]
    assert p["settings"]["privacy_level"] == "PUBLIC_TO_EVERYONE"


def test_provider_settings_match_postiz_dtos():
    cfg = PublisherConfig()
    tiktok = settings_for("tiktok", "t" * 120, cfg, [])
    assert len(tiktok["title"]) == 90
    assert set(tiktok) >= {"privacy_level", "duet", "stitch", "comment", "autoAddMusic", "brand_content_toggle",
                           "brand_organic_toggle", "content_posting_method"}
    yt = settings_for("youtube", "y" * 150, cfg, ["gta6", "gaming"])
    assert len(yt["title"]) == 100 and yt["type"] == "public"
    assert yt["tags"] == [{"value": "gta6", "label": "gta6"}, {"value": "gaming", "label": "gaming"}]
    assert settings_for("instagram-standalone", "x", cfg, []) == {"post_type": "post"}
    with pytest.raises(ValueError):
        settings_for("linkedin", "x", cfg, [])


def test_post_analytics_parsing():
    pz, _ = _client({
        ("GET", "/api/public/v1/analytics/post/p1"): httpx.Response(200, json=[
            {"label": "Views", "data": [{"total": "1200", "date": "2026-11-20"}, {"total": "1500", "date": "x"}],
             "percentageChange": 0},
            {"label": "Likes", "data": [{"total": "90", "date": "2026-11-20"}], "percentageChange": 0},
            {"label": "Saves", "data": [{"total": "4", "date": "2026-11-20"}], "percentageChange": 0},
        ]),
        ("GET", "/api/public/v1/analytics/post/p2"): httpx.Response(200, json={"missing": True}),
    })
    assert pz.post_analytics("p1") == {"views": 1500, "likes": 90}
    assert pz.post_analytics("p2") is None


def test_retries_rate_limit_then_raises_on_client_error(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    pz, rec = _client({
        ("GET", "/api/public/v1/integrations"): [httpx.Response(429, headers={"Retry-After": "1"}),
                                                 httpx.Response(200, json=[{"id": "a"}])],
        ("POST", "/api/public/v1/posts"): httpx.Response(400, json={"msg": "Please fix your settings"}),
    })
    assert pz.integrations() == [{"id": "a"}]
    with pytest.raises(PostizError, match="fix your settings"):
        pz.create_post("a", "<p>x</p>", {"id": "m", "path": "p"}, {}, "2026-11-20T00:00:00.000Z")


def test_post_throttle_waits_when_hourly_budget_is_spent(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    pz, _ = _client({("POST", "/api/public/v1/posts"): httpx.Response(200, json=[])})
    pz.posts_per_hour = 2
    for _ in range(3):
        pz.create_post("a", "<p>x</p>", {"id": "m", "path": "p"}, {}, "2026-11-20T00:00:00.000Z")
    assert len(slept) == 1 and slept[0] > 3500
