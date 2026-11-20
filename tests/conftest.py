import random
import subprocess
from pathlib import Path

import pytest

from clipper.config import Config
from clipper.db import DB


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config(data_dir=str(tmp_path / "data"))
    c.titles.enabled = False
    return c


@pytest.fixture
def db(cfg) -> DB:
    return DB(cfg.db_path)


@pytest.fixture
def rng() -> random.Random:
    return random.Random(1234)


def make_video(path: Path, seconds: float = 8, size: str = "1280x720", fps: int = 30) -> Path:
    """A synthetic test video with a tone, so it has both streams."""
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={fps}",
        "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=16000",
        "-t", str(seconds), "-c:v", "libx264", "-g", str(fps * 2), "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(path),
    ], check=True)
    return path


def simulate_chat(tracker, seconds: float, base_mpm: float, bursts: list[tuple[float, float, float, str]],
                  rng: random.Random, base_text: str = "hello"):
    """Poisson chat: bursts are (start, duration, msgs_per_second, text). Returns moments."""
    moments, t = [], 0.0
    while t < seconds:
        rate, text = base_mpm / 60.0, base_text
        for start, dur, burst_rate, burst_text in bursts:
            if start <= t < start + dur:
                rate, text = burst_rate, burst_text
        t += rng.expovariate(rate)
        moments += tracker.add(t, text)
    return moments + tracker.flush()
