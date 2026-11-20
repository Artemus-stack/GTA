# GTA 6 stream clipper

Three agents that turn GTA 6 live streams into vertical shorts, post them, and
learn which formats go viral.

```
 Twitch top 100 ─┐                       ┌──────────── Agent 2: Editor ────────────┐
 YouTube top 100 ┤   Agent 1: Scout      │ yt-dlp  pulls only the window around    │
                 ├─▶ reads every chat ──▶│         the spike from the VOD           │
                 │   500+ msg/min =      │ ClipsAI transcribes, picks clean clip    │
                 │   a "moment"          │         edges around the moment, cuts it │
                 │                       │ ffmpeg  9:16 layout + burned captions    │
                 │                       │ Claude  3 title variants + hook          │
                 │                       └────────────────────┬─────────────────────┘
                 │                                            ▼
 Google Sheet ◀──┴── Analyst (hourly) ◀── Postiz ◀── Agent 3: Publisher
 "what hits"        metrics → hits →       stats     TikTok / Shorts / Reels,
                    bandit: repeat the               rotating every account
                    format that wins
```

| Repo | Where it's used |
|---|---|
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | Downloads the exact window around each moment from Twitch/YouTube VODs (`download_ranges`, frame-accurate cuts). Also reads YouTube live and replay chat (its `live_chat` downloader) and finds YouTube live streams when there's no API key. `clipper/media/download.py`, `clipper/sources/youtube.py` |
| [ClipsAI](https://github.com/ClipsAI/clipsai) | WhisperX transcription with word timings (→ captions, → titles), TextTiling `ClipFinder` to pick clip edges on natural boundaries around the spike, `MediaEditor.trim` to cut, and `resize` speaker tracking for the optional "speaker" layout. `clipper/media/clipsai_adapter.py` |
| [Postiz](https://github.com/gitroomhq/postiz-app) | Public API: lists connected accounts, uploads the clip, schedules the post on each platform, reports post states/URLs and per-post views/likes/comments/shares. `clipper/postiz.py` |

## The agents

**Agent 1, Scout** (`clipper scout`): every 5 minutes it pulls the top 100 live
streams in Twitch's *Grand Theft Auto VI* category (Helix API) and, every hour,
the top 100 live GTA 6 streams on YouTube. It reads all of those chats in real
time: Twitch through anonymous IRC (Twitch's official read-only access) and
YouTube through yt-dlp. When a channel's chat crosses **500 messages per minute**
it records a moment, with the second the burst started, the peak rate, what
chat was spamming, and a label (funny / hype / shock / fail).

Huge channels sit above 500 msg/min all the time, so a moment also has to be
**2x the channel's normal rate** (`baseline_multiplier`; set it to 0 to use the
500 threshold alone). The burst start is found with a change-point estimate on
per-second counts; on simulated chat it lands within a few seconds of the true
start, for channels from 60 to 1,500 msg/min.

**Agent 2, Editor** (`clipper edit`): takes the best queued moment (by chat
intensity, freshness, and which streamers and moment types have been winning),
finds the VOD (works while a Twitch stream is still live; YouTube VODs once
the stream ends), downloads 75 s before and 45 s after the spike, and:

1. ClipsAI transcribes the window and proposes topic segments.
2. The clip is placed so the moment (chat onset minus ~8 s reaction lag) lands
   about a third of the way in: setup, payoff, reaction. A ClipsAI segment is
   used when one fits; otherwise the edges snap to sentence boundaries.
3. ClipsAI cuts it. ffmpeg renders 1080x1920 with the chosen layout (`blur`,
   `stack` = webcam over gameplay, `crop`, or ClipsAI `speaker`) and captions
   (`pop`, `karaoke`, `clean`) burned in, plus an optional hook banner and a
   small streamer credit.
4. Claude writes three title variants (quote, curiosity, reaction), the
   on-video hook, a caption and hashtags. Template titles stand in when no key
   is set.

**Agent 3, Publisher** (`clipper publish`): uploads each clip to Postiz once,
then schedules it on TikTok, YouTube Shorts and Instagram Reels. Accounts
rotate: each clip goes to **one account per platform** (the account with the
earliest free slot), each account keeps a 90-minute gap and a 6-per-day cap,
and the title variant per platform comes from the bandit. Every post becomes a
row in the Google Sheet.

**Analyst** (`clipper analyze`): hourly it syncs post states and URLs from
Postiz, pulls per-post stats, and also reads any numbers you typed into the
sheet. Once a post is 24 h old it is judged: a **hit** beats its platform's
median *viral score* (views + likes + 3x comments + 10x shares, configurable).
Hits and misses retrain a Thompson-sampling bandit over layout, caption style,
length, hook, and title style (per platform), weighting recent results more.
The editor and publisher sample from it, so winning formats get used more while
others still get occasional tests. Streamers and moment types that hit get
clipped first.

## Setup

You need Python 3.10+ (developed on 3.13; the Docker image uses 3.12), ffmpeg,
and `libmagic` (`apt install ffmpeg libmagic1` / `brew install ffmpeg libmagic`).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -c constraints.txt -e ".[clip,dev]"     # pulls torch; the editor needs it
python -m nltk.downloader punkt punkt_tab
cp config.example.yaml config.yaml
cp .env.example .env
```

Then fill in `.env`:

1. **Twitch**: create an app at https://dev.twitch.tv/console/apps → `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET`.
2. **Claude**: `ANTHROPIC_API_KEY` for the titles (model `claude-opus-5-5`, set in `config.yaml`).
3. **Postiz**: use [Postiz Cloud](https://postiz.com) or self-host it with its
   docker-compose. Connect every TikTok, YouTube and Instagram account you want
   in rotation, then Settings → Public API → `POSTIZ_API_KEY`. Self-hosted, set
   `POSTIZ_URL=http://localhost:4007/api`. Check with `clipper accounts`.
4. **Google Sheet**: in Google Cloud, enable the Google Sheets API, create a
   service account, download its JSON key as `service-account.json`, create a
   sheet and share it with the service account's email as Editor. Put the
   sheet id (the long part of its URL) in `GSHEET_ID`, then run `clipper sheet-init`.
5. Optional: `YOUTUBE_API_KEY` (better YouTube discovery), `HF_TOKEN` (speaker layout).

### First run

```bash
# 1. Clip one moment by hand to see the output (any Twitch/YouTube VOD):
clipper clip "https://www.twitch.tv/videos/123456789" --at 1:23:45 --streamer SomeStreamer
#    -> data/clips/<id>/clip.mp4, titles printed

# 2. Or find spikes in a finished YouTube stream's chat replay:
clipper scan "https://www.youtube.com/watch?v=VIDEO_ID" && clipper edit --once

# 3. See what would be posted, without posting:
clipper publish --dry-run

# 4. Everything, continuously:
clipper run
```

Set `publisher.post_type: draft` at first to approve each post in Postiz
before it goes out. `clipper status` shows the queues and what's winning.

### The sheet

- **Posts**: one row per post: platform, account, streamer, title + style,
  layout, captions, length, hook, then status, URL, views, likes, comments,
  shares, viral score and hit as they come in. For posts Postiz can't fetch
  stats for, type the numbers into Views/Likes/Comments/Shares and the analyst
  uses them.
- **Formats**: every option with posts, hits, hit rate, and the chance it's the
  best option. Status reads WINNING / testing / losing.
- **Streamers**: hit rate per streamer and their current priority boost.

## Production

```bash
mkdir -p secrets && cp service-account.json secrets/
docker compose up -d --build            # scout, editor, publisher, analyst as separate services
docker compose logs -f editor
```

- The editor is the heavy part (transcription and rendering). CPU works with
  `whisper_model: small` or smaller; with an NVIDIA GPU, build with
  `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128` and set
  `whisper_model: large-v2` for better captions.
- Reading 100 YouTube chats means 100 yt-dlp threads and steady traffic to
  YouTube. Lower `youtube_chat_readers` if that's too much; smaller streams
  rarely reach 500 msg/min anyway.
- YouTube often challenges datacenter IPs ("confirm you're not a bot"). Export
  cookies from a logged-in browser to `secrets/cookies.txt` and set
  `editor.cookies_file: /app/secrets/cookies.txt`.
- Models download on first use (Whisper, alignment and ClipsAI's 1.4 GB
  sentence model) into `data/models`.

## Before you scale it up

- **Rights**: streamers own their streams. Many actively want clippers (some
  pay for clips), others file takedowns. Put the ones who welcome it in
  `scout.streamers.allowlist`. Every post credits the streamer in the caption
  and on the video.
- **Platform rules**: TikTok, YouTube and Instagram demote or demonetize
  reposted footage that isn't transformed. The captions, hooks, titles and
  layouts help. Posting the same video to several accounts on one platform
  reads as spam, so the publisher never does that.
- **Platform posting limits**: a self-hosted Postiz uses your own developer
  apps. TikTok only allows public posts from an app that passed TikTok's
  review (until then use `tiktok_privacy: SELF_ONLY`), and YouTube's upload API
  has a daily quota per Google Cloud project. Postiz Cloud uses its own
  approved apps.

## ClipsAI compatibility notes

ClipsAI 0.2.1 (January 2024) needs help with current dependencies; this repo handles it:

- NLTK 3.9+ needs `punkt_tab`, not just the `punkt` ClipsAI downloads; it's fetched automatically.
- pyannote.audio 4 renamed `use_auth_token` to `token`; a shim maps it (speaker layout only).
- `ClipFinder` reloads its 1.4 GB sentence model for every clip; it's cached once per process.
- ClipsAI's speaker tracking uses `mediapipe.solutions`, which newer mediapipe
  releases removed. The `speaker` layout is only offered when `HF_TOKEN` is set
  and a compatible mediapipe (≤0.10.21, Python ≤3.12) is installed; otherwise
  that layout falls back to `blur`.

## Tests

```bash
pytest
```

The suite covers spike detection on simulated chat, Twitch/YouTube parsing,
clip-boundary logic, caption files, the bandit, the Postiz request shapes
(checked against Postiz's DTOs), account rotation, and the analyst loop. The
end-to-end test serves a 3-minute HLS "VOD" locally and runs the real yt-dlp
download (checking cuts are frame-accurate), the real ClipsAI code and ffmpeg
render. Only the model weights are stubbed.
