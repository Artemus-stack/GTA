import random

import pytest

from clipper.velocity import ChannelTrackers, VelocityTracker, classify, top_messages

from conftest import simulate_chat


@pytest.mark.parametrize("base_mpm,burst_rate,duration", [
    (200, 12, 40),     # 720 msg/min burst on a mid channel
    (200, 30, 40),     # 1800 msg/min
    (60, 10, 60),      # small channel that just reaches 600 msg/min
    (1500, 90, 40),    # huge channel: 1500 normal, 5400 in the burst
])
def test_detects_burst_and_finds_onset(base_mpm, burst_rate, duration):
    errors = []
    for seed in range(15):
        rng = random.Random(seed)
        start = 400 + rng.random()
        tr = VelocityTracker(threshold=500, baseline_multiplier=2.0, grace=0)
        moments = simulate_chat(tr, 900, base_mpm, [(start, duration, burst_rate, "KEKW")], rng)
        assert len(moments) == 1
        assert moments[0].peak_mpm >= 500
        errors.append(moments[0].onset - int(start))
    assert max(abs(e) for e in errors) <= 5, errors


def test_big_channel_idling_above_threshold_is_not_a_moment(rng):
    tr = VelocityTracker(threshold=500, baseline_multiplier=2.0, grace=0)
    assert simulate_chat(tr, 3600, 1500, [], rng) == []


def test_absolute_threshold_only_when_multiplier_disabled(rng):
    tr = VelocityTracker(threshold=500, baseline_multiplier=0, grace=0)
    moments = simulate_chat(tr, 600, 300, [(200, 40, 15, "W")], rng)
    assert len(moments) == 1


def test_below_threshold_never_fires(rng):
    tr = VelocityTracker(threshold=500, baseline_multiplier=2.0, grace=0)
    # 450 msg/min "burst" on a 60/min channel: 9x the baseline but under 500
    assert simulate_chat(tr, 900, 60, [(400, 60, 7.5, "LUL")], rng) == []


def test_warmup_blocks_detection_right_after_joining(rng):
    tr = VelocityTracker(threshold=500, baseline_multiplier=2.0, warmup=180, grace=0)
    assert simulate_chat(tr, 150, 200, [(70, 40, 30, "KEKW")], rng) == []


def test_cooldown_merges_back_to_back_bursts(rng):
    tr = VelocityTracker(threshold=500, baseline_multiplier=2.0, cooldown=120, grace=0)
    moments = simulate_chat(tr, 1200, 200, [(400, 30, 30, "KEKW"), (470, 30, 30, "KEKW"), (900, 30, 30, "W")], rng)
    assert len(moments) == 2


def test_moment_type_and_sample(rng):
    tr = VelocityTracker(threshold=500, baseline_multiplier=2.0, grace=0)
    m = simulate_chat(tr, 900, 200, [(400, 40, 30, "KEKW KEKW")], rng)[0]
    assert m.moment_type == "funny"
    assert m.sample[0].startswith("KEKW KEKW")


def test_classify_categories():
    assert classify(["LMAOOOO", "KEKW", "😂😂"] * 5)[0] == "funny"
    assert classify(["W", "LETS GOOO", "POGGERS"] * 5)[0] == "hype"
    assert classify(["WTF", "NO WAY", "???"] * 5)[0] == "shock"
    assert classify(["L", "RIP", "F"] * 5)[0] == "fail"
    assert classify(["hello", "what's up"] * 5)[0] == "general"
    _, _, clips = classify(["CLIP IT", "clip that", "clip", "nice"])
    assert clips == 3


def test_top_messages_groups_repeats():
    out = top_messages(["KEKW", "KEKW", "kekw", "lol", "LUL LUL"])
    assert out[0] == "KEKW (x3)"


def test_late_messages_count_in_open_second():
    tr = VelocityTracker(threshold=5, baseline_multiplier=0, grace=2)
    tr.add(100.0, "a")
    tr.tick(110)                 # seconds < 108 are closed
    tr.add(101.0, "late")        # belongs to a closed second: counted in the oldest open one
    tr.tick(200)
    assert tr.total == 2


def test_channel_trackers_route_by_key():
    trackers = ChannelTrackers(lambda key: VelocityTracker(threshold=3, baseline_multiplier=0, grace=0, cooldown=0))
    found = []
    for i in range(10):
        found += trackers.add("twitch:a", 1000 + i * 0.1, "W")
    found += trackers.tick(1200)
    assert [k for k, _ in found] == ["twitch:a"]
    assert trackers.rates()["twitch:a"] == 0
