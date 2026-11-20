"""Configuration loading.

Non-secret settings live in a YAML file (see ``config.example.yaml``). Secrets
come from environment variables, or from a local ``.env`` file, so they never
end up in git.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml

PLATFORMS = ("tiktok", "youtube", "instagram")
LAYOUTS = ("blur", "stack", "crop", "speaker")
CAPTION_STYLES = ("pop", "karaoke", "clean")
TITLE_STYLES = ("quote", "curiosity", "reaction")


class ConfigError(ValueError):
    pass


@dataclass
class StreamerFilter:
    allowlist: list[str] = field(default_factory=list)
    blocklist: list[str] = field(default_factory=list)

    def allowed(self, *names: str | None) -> bool:
        keys = {n.lower() for n in names if n}
        if keys & {b.lower() for b in self.blocklist}:
            return False
        if self.allowlist:
            return bool(keys & {a.lower() for a in self.allowlist})
        return True


@dataclass
class GameConfig:
    # Twitch category names, resolved to category IDs at runtime.
    twitch_categories: list[str] = field(default_factory=lambda: ["Grand Theft Auto VI"])
    youtube_query: str = "GTA 6"
    # A YouTube live stream must mention one of these in its title/description.
    keywords: list[str] = field(
        default_factory=lambda: ["gta 6", "gta vi", "gta6", "gtavi", "grand theft auto vi", "grand theft auto 6"]
    )


@dataclass
class ScoutConfig:
    twitch_top_n: int = 100
    youtube_top_n: int = 100
    youtube_chat_readers: int = 100
    languages: list[str] = field(default_factory=list)
    refresh_twitch_s: int = 300
    refresh_youtube_s: int = 3600
    spike_threshold_mpm: int = 500
    # Also require chat to be this many times the channel's normal rate.
    # Big channels sit above 500 msg/min all the time; 0 disables the check.
    baseline_multiplier: float = 2.0
    baseline_window_s: int = 900
    warmup_s: int = 180
    cooldown_s: int = 120
    max_spike_s: int = 180
    # Chat reacts this many seconds after the moment (stream delay + typing).
    reaction_lag_s: float = 8.0
    chat_file_max_mb: int = 64
    streamers: StreamerFilter = field(default_factory=StreamerFilter)


@dataclass
class EditorConfig:
    pre_roll_s: float = 75.0
    post_roll_s: float = 45.0
    max_height: int = 1080
    precise_cuts: bool = True
    whisper_model: str = "small"
    device: str | None = None
    language: str | None = "en"
    caption_font: str = "DejaVu Sans"
    fonts_dir: str | None = None
    credit_watermark: bool = True
    # streamer login/channel -> [x, y, w, h] of the webcam, as fractions of the frame
    facecams: dict[str, list[float]] = field(default_factory=dict)
    retry_delay_s: int = 600
    max_errors: int = 3
    max_moment_age_h: float = 30.0
    cookies_file: str | None = None


@dataclass
class TitlesConfig:
    enabled: bool = True
    model: str = "claude-opus-5-5"
    effort: str = "medium"


@dataclass
class PublisherConfig:
    postiz_url: str = "https://api.postiz.com"
    platforms: list[str] = field(default_factory=lambda: list(PLATFORMS))
    include_accounts: list[str] = field(default_factory=list)
    exclude_accounts: list[str] = field(default_factory=list)
    min_gap_minutes: int = 90
    max_posts_per_day: int = 6
    max_schedule_ahead_h: float = 12.0
    # "schedule" posts automatically; "draft" parks posts in Postiz for review.
    post_type: str = "schedule"
    tiktok_privacy: str = "PUBLIC_TO_EVERYONE"
    tiktok_method: str = "DIRECT_POST"
    youtube_privacy: str = "public"
    hashtags: list[str] = field(default_factory=lambda: ["gta6", "gtavi", "gta", "gaming"])
    credit_streamer: bool = True
    posts_per_hour_limit: int = 80


@dataclass
class AnalystConfig:
    interval_s: int = 3600
    judge_after_h: float = 24.0
    # 0 = a post is a hit when it beats the median of its platform.
    hit_score: float = 0.0
    half_life_days: float = 10.0
    score_weights: dict[str, float] = field(
        default_factory=lambda: {"views": 1.0, "likes": 1.0, "comments": 3.0, "shares": 10.0}
    )
    # Override the format options the bandit chooses between.
    arms: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class SheetsConfig:
    spreadsheet_id: str = ""
    credentials_file: str = "service-account.json"


@dataclass
class Secrets:
    twitch_client_id: str = ""
    twitch_client_secret: str = ""
    youtube_api_key: str = ""
    postiz_api_key: str = ""
    hf_token: str = ""

    @classmethod
    def from_env(cls) -> "Secrets":
        env = os.environ.get
        return cls(
            twitch_client_id=env("TWITCH_CLIENT_ID", ""),
            twitch_client_secret=env("TWITCH_CLIENT_SECRET", ""),
            youtube_api_key=env("YOUTUBE_API_KEY", ""),
            postiz_api_key=env("POSTIZ_API_KEY", ""),
            hf_token=env("HF_TOKEN", ""),
        )


@dataclass
class Config:
    data_dir: str = "data"
    game: GameConfig = field(default_factory=GameConfig)
    scout: ScoutConfig = field(default_factory=ScoutConfig)
    editor: EditorConfig = field(default_factory=EditorConfig)
    titles: TitlesConfig = field(default_factory=TitlesConfig)
    publisher: PublisherConfig = field(default_factory=PublisherConfig)
    analyst: AnalystConfig = field(default_factory=AnalystConfig)
    sheets: SheetsConfig = field(default_factory=SheetsConfig)
    secrets: Secrets = field(default_factory=Secrets)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def db_path(self) -> Path:
        return self.data_path / "clipper.db"


def _build(cls: type, data: Any, where: str):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints[f.name]
        if dataclasses.is_dataclass(hint):
            value = _build(hint, value, f"{where}.{f.name}")
        kwargs[f.name] = value
    return cls(**kwargs)


def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines; never overrides real env vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _validate(cfg: Config) -> None:
    unknown = [p for p in cfg.publisher.platforms if p not in PLATFORMS]
    if unknown:
        raise ConfigError(f"publisher.platforms: unknown {unknown}; choose from {list(PLATFORMS)}")
    if cfg.publisher.post_type not in ("schedule", "draft"):
        raise ConfigError("publisher.post_type must be 'schedule' or 'draft'")
    if cfg.scout.spike_threshold_mpm <= 0:
        raise ConfigError("scout.spike_threshold_mpm must be positive")
    for dim, options in cfg.analyst.arms.items():
        allowed = {"layout": LAYOUTS, "captions": CAPTION_STYLES, "hook": ("banner", "none")}.get(dim)
        if dim not in ("layout", "captions", "length", "hook"):
            raise ConfigError(f"analyst.arms: unknown dimension {dim!r}")
        if not options:
            raise ConfigError(f"analyst.arms.{dim}: needs at least one option")
        if allowed and any(o not in allowed for o in options):
            raise ConfigError(f"analyst.arms.{dim}: options must be among {list(allowed)}")
        if dim == "length" and any(not str(o).isdigit() for o in options):
            raise ConfigError("analyst.arms.length: options are target seconds, e.g. ['20', '30']")


def load_config(path: str | os.PathLike | None = None) -> Config:
    load_dotenv(Path(".env"))
    data: dict = {}
    if path:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.pop("secrets", None)  # secrets only come from the environment
    cfg = _build(Config, data, "config")
    env = os.environ.get
    if env("POSTIZ_URL"):
        cfg.publisher.postiz_url = env("POSTIZ_URL")
    if env("GSHEET_ID"):
        cfg.sheets.spreadsheet_id = env("GSHEET_ID")
    if env("GOOGLE_APPLICATION_CREDENTIALS"):
        cfg.sheets.credentials_file = env("GOOGLE_APPLICATION_CREDENTIALS")
    cfg.secrets = Secrets.from_env()
    _validate(cfg)
    return cfg
