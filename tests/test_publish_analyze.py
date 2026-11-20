import json
import random
import time

from clipper.agents.analyst import Analyst
from clipper.agents.publisher import Publisher, caption
from clipper.formats import FormatBandit
from clipper.sheets import POSTS_HEADER


class FakePostiz:
    def __init__(self):
        self.created, self.uploads = [], []
        self.analytics: dict[str, dict] = {}
        self.remote_posts: list[dict] = []
        self._n = 0

    def integrations(self):
        return [
            {"id": "tt1", "name": "Clips A", "identifier": "tiktok", "disabled": False},
            {"id": "tt2", "name": "Clips B", "identifier": "tiktok", "disabled": False},
            {"id": "tt3", "name": "Old", "identifier": "tiktok", "disabled": True},
            {"id": "yt1", "name": "Shorts", "identifier": "youtube", "disabled": False},
            {"id": "ig1", "name": "Reels", "identifier": "instagram-standalone", "disabled": False},
            {"id": "li1", "name": "LinkedIn", "identifier": "linkedin", "disabled": False},
        ]

    def upload(self, path):
        self.uploads.append(path)
        return {"id": f"m{len(self.uploads)}", "path": f"https://cdn/{len(self.uploads)}.mp4"}

    def create_post(self, integration_id, content, media, settings, date_iso, post_type="schedule"):
        self._n += 1
        self.created.append({"integration": integration_id, "content": content, "media": media,
                             "settings": settings, "date": date_iso, "type": post_type})
        return [{"postId": f"p{self._n}", "integration": integration_id}]

    def posts(self, start, end):
        return self.remote_posts

    def post_analytics(self, post_id, days=7):
        return self.analytics.get(post_id)


class FakeSheet:
    enabled = True

    def __init__(self):
        self.appended, self.rows, self.updates, self.replaced = [], [], {}, {}

    def append_posts(self, rows):
        self.appended += rows

    def read_posts(self):
        return self.rows

    def update_posts(self, updates):
        self.updates.update(updates)

    def replace(self, name, rows):
        self.replaced[name] = rows


def add_clip(db, n, streamer="somestreamer", moment_type="funny", layout="blur"):
    mid = f"tw-{streamer}-{n}"
    db.insert("moments", {"id": mid, "platform": "twitch", "channel": streamer, "channel_display": streamer.title(),
                          "channel_id": "1", "onset_at": time.time() - 600, "peak_mpm": 900, "score": 100.0,
                          "moment_type": moment_type, "chat_sample": "[]", "status": "edited"})
    cid = f"c-{mid}"
    db.insert("clips", {
        "id": cid, "moment_id": mid, "path": f"/clips/{cid}.mp4", "duration": 30.0,
        "source_url": "https://www.twitch.tv/videos/1?t=0h10m0s",
        "titles": json.dumps({"quote": f"quote {n}", "curiosity": f"curious {n}", "reaction": f"react {n}"}),
        "hook": "WAIT FOR IT", "description": "desc", "hashtags": json.dumps(["gta6", "funny"]),
        "format": json.dumps({"layout": layout, "captions": "pop", "length": "30", "hook": "banner"}),
    })
    return cid


def test_rotation_one_account_per_platform_and_sheet_log(cfg, db):
    pz, sheet = FakePostiz(), FakeSheet()
    pub = Publisher(cfg, db, FormatBandit(db, rng=random.Random(1)), pz, sheet)
    clips = [add_clip(db, i) for i in range(3)]
    while pub.process_next():
        pass
    assert {r["status"] for r in db.query("SELECT status FROM clips")} == {"posted"}
    assert len(pz.uploads) == 3                                   # one upload per clip, reused on every platform
    posts = db.query("SELECT * FROM posts ORDER BY scheduled_at")
    assert len(posts) == 9
    for cid in clips:
        per_clip = [p for p in posts if p["clip_id"] == cid]
        assert sorted(p["platform"] for p in per_clip) == ["instagram", "tiktok", "youtube"]
    tiktok_accounts = [p["account_id"] for p in db.query(
        "SELECT account_id FROM posts WHERE platform = 'tiktok' ORDER BY created_at")]
    assert tiktok_accounts[:2] in (["tt1", "tt2"], ["tt2", "tt1"])        # rotates, never the disabled one
    assert "tt3" not in tiktok_accounts
    yt = [p["scheduled_at"] for p in posts if p["platform"] == "youtube"]
    assert all(b - a >= cfg.publisher.min_gap_minutes * 60 - 1 for a, b in zip(yt, yt[1:]))
    assert len(sheet.appended) == 9 and all(len(r) == len(POSTS_HEADER) for r in sheet.appended)
    row = dict(zip(POSTS_HEADER, sheet.appended[0]))
    assert row["Streamer"] == "Somestreamer" and row["Layout"] == "blur" and row["Status"] == "scheduled"
    ig = next(c for c in pz.created if c["integration"] == "ig1")
    assert ig["settings"] == {"post_type": "post"} and "#gta6" in ig["content"]


def test_skips_clip_when_every_account_is_booked(cfg, db):
    cfg.publisher.platforms = ["youtube"]
    cfg.publisher.min_gap_minutes = 600            # 10 h between posts on the only YouTube account
    cfg.publisher.max_schedule_ahead_h = 12
    pub = Publisher(cfg, db, FormatBandit(db), FakePostiz(), FakeSheet())
    for i in range(3):
        add_clip(db, i)
    while pub.process_next():
        pass
    assert [r["status"] for r in db.query("SELECT status FROM clips ORDER BY created_at")] == \
        ["posted", "posted", "skipped"]


def test_daily_cap(cfg, db):
    cfg.publisher.platforms = ["youtube"]
    cfg.publisher.min_gap_minutes = 1
    cfg.publisher.max_posts_per_day = 2
    cfg.publisher.max_schedule_ahead_h = 48
    pub = Publisher(cfg, db, FormatBandit(db), FakePostiz(), FakeSheet())
    for i in range(3):
        add_clip(db, i)
    while pub.process_next():
        pass
    times = [r["scheduled_at"] for r in db.query("SELECT scheduled_at FROM posts ORDER BY scheduled_at")]
    assert len(times) == 3 and times[2] - times[0] >= 86400


def test_captions_per_platform(cfg, db):
    add_clip(db, 1)
    clip = db.one("SELECT * FROM clips")
    moment = db.one("SELECT * FROM moments")
    yt = caption("youtube", "T", clip, moment, cfg)
    assert yt.startswith("desc") and "#shorts" in yt and "twitch.tv/somestreamer" in yt
    tt = caption("tiktok", "My title", clip, moment, cfg)
    assert tt.startswith("My title") and "#gta6 #funny #gtavi" in tt


def test_analyst_closes_the_loop(cfg, db):
    pz, sheet = FakePostiz(), FakeSheet()
    bandit = FormatBandit(db, rng=random.Random(5))
    pub = Publisher(cfg, db, bandit, pz, sheet)
    for i in range(6):
        add_clip(db, i, streamer="winner" if i % 2 == 0 else "loser", layout="stack" if i % 2 == 0 else "crop")
    while pub.process_next():
        pass
    # pretend everything went out two days ago
    db.execute("UPDATE posts SET scheduled_at = scheduled_at - 2 * 86400")
    posts = db.query("SELECT p.*, m.channel FROM posts p JOIN clips c ON c.id = p.clip_id "
                     "JOIN moments m ON m.id = c.moment_id")
    pz.remote_posts = [{"id": p["postiz_post_id"], "state": "PUBLISHED",
                        "releaseURL": f"https://example.com/{p['postiz_post_id']}"} for p in posts]
    manual = None
    for p in posts:
        views = 50000 if p["channel"] == "winner" else 800
        if p["platform"] == "instagram" and manual is None:
            manual = p                       # no API stats for this one: typed into the sheet instead
            continue
        pz.analytics[p["postiz_post_id"]] = {"views": views, "likes": views // 20, "comments": 5, "shares": 2}
    sheet.rows = [{"Post ID": p["id"], "Views": "", "_row": i + 2} for i, p in enumerate(posts)]
    manual_row = next(r for r in sheet.rows if r["Post ID"] == manual["id"])
    manual_row.update({"Views": "1,200", "Likes": "40", "Comments": "1", "Shares": "0"})

    summary = Analyst(cfg, db, bandit, pz, sheet).run_once()
    assert summary["states"] == len(posts) and summary["from_sheet"] == 1
    assert summary["judged"] == len(posts)
    got = db.one("SELECT * FROM posts WHERE id = ?", (manual["id"],))
    assert got["views"] == 1200 and got["metrics_source"] == "sheet"
    hits = db.query("SELECT p.hit, m.channel FROM posts p JOIN clips c ON c.id = p.clip_id "
                    "JOIN moments m ON m.id = c.moment_id")
    assert all(h["hit"] == 1 for h in hits if h["channel"] == "winner")
    assert all(h["hit"] == 0 for h in hits if h["channel"] == "loser")
    prio = db.kv_get("priority")
    assert prio["streamer"]["winner"] > 1 > prio["streamer"]["loser"]
    assert [bandit.choose_format(["stack", "crop"])["layout"] for _ in range(100)].count("stack") > 85
    row_update = next(iter(sheet.updates.values()))
    assert row_update["Status"] == "published" and row_update["Hit"] in ("YES", "no")
    assert "Formats" in sheet.replaced and sheet.replaced["Streamers"][0][0] == "winner"
