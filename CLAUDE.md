# AutoClipper — project spec for Claude Code

## What we're building
A tool that watches Twitch and Kick streams (starting with GTA 6 content), finds the most
clip-worthy moments, cuts them into vertical 9:16 short-form clips with captions and a hook,
and puts them in a review queue. After I approve them, it posts them and tracks their views.
I earn per 1,000 views through clipping campaigns (Content Rewards / Whop) and direct streamer deals,
so the goal is **many good clips, fast, with no spoilers**.

## About me / how to work with me
- I'm a software engineering student, not advanced. Explain *why* for each design choice and
  any non-obvious code. Keep the code readable over clever.
- Build **one stage at a time**. Every stage ends with a test on a real VOD and my sign-off
  before we move on. Don't scaffold future stages early.
- Machine: **Windows laptop, RTX 4060 (8 GB VRAM)**. All commands and paths must work on Windows
  (PowerShell). Use `pathlib`, never hard-coded `/` paths.
- Python 3.11+. Keep dependencies few. ffmpeg / yt-dlp are external tools called via `subprocess`.

## Hard rules (never break these)
1. **Nothing is posted without my approval.** The tool fills a review queue and I approve it. Auto-posting
   can only be a per-streamer opt-in flag added later, default OFF.
2. **No spoilers.** Never output clips of cutscenes, story reveals, mission endings, or the game ending.
   Rockstar removes "isolated cinematic" videos, cutscene compilations, ending videos and pre-release/leaked
   footage. Anything the AI flags as possibly story/cutscene goes to the queue marked ⚠ and is never auto-posted.
3. Only clip streamers on an allow-list in config (streamers who allow or pay for clipping).
4. No view botting, no posting identical clips across many accounts, no faking country/residency.
5. Secrets (API keys, tokens) live in `.env`, never in code or git. Add `.env` and `out/` to `.gitignore`.

## Project layout
```
autoclipper/
  CLAUDE.md
  config.toml            # all tunable numbers live here, not in code
  .env                   # TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, ANTHROPIC_API_KEY, ...
  streamers.toml         # allow-list + per-streamer layout profiles
  clipper/
    config.py            # loads config.toml with defaults
    fetch.py             # yt-dlp / TwitchDownloaderCLI wrappers
    kick_chat.py         # live Kick chat recorder
    signals.py           # chat / audio / viewer-clip signals -> per-second z-scores
    scoring.py           # combine signals, pick peaks, rough clip bounds
    transcribe.py        # faster-whisper on candidate windows only
    judge.py             # AI model: rate, hook, title, spoiler flag
    edit.py              # 9:16 layout, captions, hook, loudness (stage 2)
    queue_app.py         # review web app (stage 3)
    live.py              # live mode (stage 4)
    publish.py / stats.py  # posting + view tracking (stage 5)
  run.py                 # CLI entry point
  tests/
  out/<streamer>/<vod_id>/   # all outputs per VOD
```

## Stages

### Stage 1 — Detector (VOD mode)
Input: a Twitch or Kick VOD URL. Output: `candidates.json` + rough horizontal preview clips.

Pipeline:
1. **Download audio only** with yt-dlp (`-f audio_only/bestaudio/worst`). Never download the full video
   (a 6 h 1080p VOD is ~20 GB).
2. **Get chat**
   - Twitch: `TwitchDownloaderCLI chatdownload --id <vod_id> -o chat.json` (gives `comments[].content_offset_seconds`,
     `commenter._id`, `message.body`).
   - Kick: there is no official chat replay for past VODs, so chat must be recorded **live** by `kick_chat.py`
     while the stream runs, saved as `[{"t": seconds_since_stream_start, "user": "...", "text": "..."}]`.
     First check whether Kick's official developer API (chat webhooks/events) works for this; otherwise use
     the public chat websocket. Verify current Kick docs before building. If no chat exists, run audio-only.
3. **Viewer clips (Twitch)**: Helix `GET /clips?broadcaster_id=...&started_at=...&ended_at=...` gives each clip's
   `vod_offset`, `duration`, `view_count`. Many viewers clipping the same moment = strongest signal.
4. **Build per-second signals** (array index = second of the VOD):
   - `chat`: messages per second, summed over a 5 s window
   - `unique`: unique chatters in that 5 s window
   - `hype`: messages that are mostly hype emotes/words (KEKW, OMEGALUL, LUL, POG, "clip it", W, 💀, 😂, ?????, ...;
     list in config)
   - `audio`: loudness in dB per second (stream audio through ffmpeg at 8 kHz mono, 1 s chunks; never load whole file)
   - `viewer_clips`: Gaussian bump (σ ≈ 8 s) centred at `vod_offset + 0.6 × clip_duration`, weighted by
     `1 + log10(1 + views)`
5. **Normalise each signal with a robust z-score**: `(x − rolling_median) / (1.4826 × rolling_MAD + floor)` over a
   centred 300 s window. This makes small and huge streamers comparable and handles a stream getting busier over time.
   The floor stops dead chat from producing huge z-scores.
6. **Correct chat lag**: shift all chat signals ~6 s earlier (chat reacts after the moment). Configurable.
7. **Combine**: `score = Σ weight × clip(z, 0, 8)` + bonus `0.5 × min(z_audio, z_chat)` when both spike together,
   then smooth over 3 s. Starting weights: chat 1.0, unique 0.8, hype 1.2, audio 0.7, viewer_clips 1.5.
8. **Pick peaks**: take the highest second, block ±45 s around it, repeat until the score is below threshold (2.5)
   or we have `top_n` (15–20).
9. **Rough bounds**: start = peak − 12 s (extend back to include build-up while score > 35% of peak, max 20 s);
   end = when score falls below 35% of peak (max 45 s after peak) + 3 s.
10. **Transcribe only candidate windows** (±10 s around rough bounds) with faster-whisper, word timestamps on.
    Never transcribe the whole VOD.
11. **Snap to speech**: move start to the beginning of the sentence running at start (max 8 s earlier) and end to the
    end of the sentence running at end (max 6 s later). Never start after `peak − 3`.
12. **Clamp length** to 15–60 s (target ~30). Too long: keep the peak, ~45% before it. Too short: grow 60% backwards,
    40% forwards.
13. **Download only those sections** with `yt-dlp --download-sections "*START-END"` (+2 s padding) and save
    preview clips.

Acceptance test: on 2–3 real GTA 6 VODs I watch the top 15 previews. At least ~8 should be genuinely good moments,
and none should start or end mid-sentence. Include a `--debug` flag that saves a PNG plot of all signals + picked peaks,
so we can see *why* it picked something.

### Stage 1b — AI judge
For each candidate send to the AI model (Anthropic API): transcript, streamer name, signal breakdown, duration.
Ask for strict JSON:
`{"score": 1-10, "hook": "<= 8 words on-screen text", "title": "...", "spoiler_risk": "none|low|high",
  "reason": "...", "suggested_trim": {"start_offset": s, "end_offset": s}}`
- Only the top ~20 candidates per VOD are judged, so cost stays at cents per stream.
- `spoiler_risk` high → clip is still saved but marked ⚠ and never auto-posted. Also flag long stretches with no
  streamer speech + no chat (often cutscenes).
- Final ranking = signal score × AI score. Validate the JSON; retry once on bad output.

### Stage 2 — Editor
- `streamers.toml` stores each streamer's layout once: facecam box (x, y, w, h) and gameplay crop box. Add a
  helper that saves a frame so I can measure the boxes. Support "no facecam" (gameplay only, with blurred background fill).
- ffmpeg `filter_complex`: facecam scaled to the top ~35%, gameplay below, output 1080×1920, 30 or 60 fps,
  H.264 `-preset veryfast -crf 20`, AAC audio.
- Captions: word-level timings from Whisper → `.ass` subtitles, 2–4 words at a time, bold, outlined, centred
  between the facecam and gameplay. Keep clear of the platform UI zones (bottom ~20%, right edge).
- Hook text overlay for the first ~2 s.
- Audio: `loudnorm` to −14 LUFS. Option to duck in-game music (the radio gets copyright claims).
- Small streamer credit ("twitch.tv/name") — campaigns usually require it.

Acceptance: 10 finished clips look right on my phone.

### Stage 3 — Review queue
- Small FastAPI (or Flask) app + one simple HTML page: video player, hook/title (editable), AI reason, ⚠ flag,
  buttons Approve / Reject / Trim ±2 s.
- SQLite database for clips and their status.
- Usable from my phone on the same Wi-Fi; later through Tailscale when I'm away.

### Stage 4 — Live mode
- Watch the allow-listed streamers. When one goes live, record chat (Twitch IRC/EventSub, Kick) and run the
  scoring on a sliding window.
- When a peak is confirmed (~30–60 s after it, once the reaction has played out), grab that section from the live
  stream and send it through the same pipeline. Goal: clip in my queue within ~5 minutes of the moment.
- Optional: create an official Twitch clip via Helix `POST /clips` as the source.

### Stage 5 — Publishing + learning loop
- Upload through official APIs only:
  - YouTube Data API (check the current upload quota; it limits uploads per day per project).
  - TikTok Content Posting API (needs app review; until then I post manually from the queue,
    so the tool should export a ready-to-post file plus caption).
  - Instagram Graph API (needs an Instagram professional account).
- Log each clip's views at 24 h and 72 h. Weekly job: compare which signals predicted views and suggest new weights
  (show me before changing config).
- Track campaign submissions and earnings per clip.

## Config (config.toml) — starting values
```toml
[general]
top_n = 15
[chat]
lag_seconds = 6
smooth_seconds = 5
baseline_seconds = 300
[audio]
sample_rate = 8000
baseline_seconds = 300
[weights]
chat = 1.0
unique = 0.8
hype = 1.2
audio = 0.7
viewer_clips = 1.5
[peaks]
threshold = 2.5
min_gap_seconds = 45
[clip]
min_seconds = 15
max_seconds = 60
target_seconds = 30
pre_roll = 12
post_roll = 3
decay_ratio = 0.35
[whisper]
model = "large-v3"
device = "cuda"
compute_type = "int8_float16"
```

## Windows setup notes
- Python 3.11+, `pip install numpy faster-whisper yt-dlp requests python-dotenv matplotlib anthropic`
- ffmpeg: `winget install Gyan.FFmpeg`. TwitchDownloaderCLI: download the Windows release and put it on PATH.
- faster-whisper on GPU needs CUDA 12 cuBLAS + cuDNN 9. Easiest: `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`
  and make sure their DLL folders are found. If GPU setup fights us, fall back to `device="cpu"`, model `small`,
  and fix GPU later. Don't let this block stage 1.
- Twitch API: register an app at dev.twitch.tv for client ID/secret (app access token is enough for clips).

## Testing
- Unit tests with synthetic data: fake chat with known spikes + a generated audio file with loud bursts. The detector
  must find the planted moments within ±3 s, with chat lag handled.
- `run.py <url> --debug` must run end-to-end on a real VOD before a stage is called done.
