# AutoClipper

Finds the most clip-worthy moments in Twitch and Kick VODs. The full plan is in [CLAUDE.md](CLAUDE.md).
**Built so far: Stage 1 (the detector), Stage 1b (the AI judge) and Stage 2 (the editor).** It turns a VOD URL into
`candidates.json` plus short horizontal preview clips, each rated by an AI model with a hook, a title
and a spoiler check, and then into finished vertical 9:16 clips with captions. It doesn't post or upload anything.

## How it works (Stage 1)

```
VOD URL ─► audio only (yt-dlp) ─► loudness per second ─┐
        ├► chat replay (Twitch) / live recording (Kick) ─► chat, unique chatters, hype ─┤
        └► viewer clips (Twitch API) ─────────────────► clip bumps ─────────────────────┤
                                                                                        ▼
            robust z-score each signal ─► weighted sum ─► peaks ─► rough start/end
            ─► Whisper on those windows only ─► snap to sentence edges ─► clamp 15–60 s
            ─► AI judge: score 1-10, hook, title, spoiler risk (+ local cutscene check)
            ─► rank by signal score × AI score ─► download only those sections as previews
```

| File | What it does |
|---|---|
| `run.py` | The command you run. Ties all the steps together. |
| `clipper/config.py` | Loads `config.toml` (all tunable numbers) and `streamers.toml` (allow-list). |
| `clipper/fetch.py` | yt-dlp, ffprobe and TwitchDownloaderCLI wrappers, Twitch API viewer clips. |
| `clipper/kick_chat.py` | Records Kick chat live (Kick has no chat replay). |
| `clipper/signals.py` | Chat / audio / viewer-clip signals and the robust z-score. |
| `clipper/scoring.py` | Combines signals, picks peaks, rough bounds, speech snapping, length clamp. |
| `clipper/transcribe.py` | faster-whisper on the candidate windows only, with CPU fallback. |
| `clipper/judge.py` | Stage 1b: AI rating, hook, title, spoiler flag, cutscene check. |
| `clipper/plot.py` | The `--debug` plot. |

## Setup on Windows (PowerShell)

```powershell
cd $HOME\OneDrive\Desktop\Clipping
py -m venv .venv               # Python 3.10 or newer
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

winget install Gyan.FFmpeg          # then open a NEW terminal so ffmpeg is on PATH
```

1. **TwitchDownloaderCLI** (for Twitch chat): download the Windows CLI zip from
   <https://github.com/lay295/TwitchDownloader/releases>, unzip `TwitchDownloaderCLI.exe` into a folder
   (e.g. `C:\Tools`) and add that folder to PATH. Check with `TwitchDownloaderCLI --help`.
2. **Twitch API keys** (optional, adds the viewer-clips signal): register an app at
   <https://dev.twitch.tv/console/apps> (OAuth redirect `http://localhost`, category "Other"), then
   `copy .env.example .env` and fill in `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET`.
3. **Anthropic API key** (optional, turns on the AI judge): create a key at <https://platform.claude.com>
   and put it in `.env` as `ANTHROPIC_API_KEY=...`. A judged VOD costs roughly $0.30–0.80 with the default
   model (`claude-opus-5-5`, effort `low`, top 20 clips). For cents per VOD, set `model = "claude-haiku-5-5"`
   in the `[judge]` section of `config.toml`.
4. **GPU Whisper** (optional, much faster): `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`.
   If the GPU setup fails, the tool prints a warning and uses the `small` model on CPU instead.
   To force CPU, set `device = "cpu"` in `config.toml`.
5. **Allow-list**: add each streamer you're allowed to clip to `streamers.toml`. The tool refuses anyone else.

## Running it

```powershell
# Twitch
python run.py https://www.twitch.tv/videos/2245678901 --debug

# Kick: record chat while the stream is live (leave it running; it stops when the stream ends)
python -m clipper.kick_chat somechannel
# ...then after the stream, point run.py at the recorded chat
python run.py https://kick.com/somechannel/videos/<id> --chat out\somechannel\kick_live_20261117_1800\chat.json --debug
```

Useful flags: `--top-n 20`, `--no-transcribe` (skip Whisper), `--no-judge` (skip the AI), `--no-previews`, `--chat-offset 12`
(if Kick chat and VOD are out of sync by a few seconds).

Everything lands in `out\<streamer>\<vod_id>\`. Re-running the same URL reuses the audio and chat already
downloaded, so after changing `config.toml` only the scoring is redone.

### Reading the output

- `candidates.json`: best first. Each has `start`/`end`/`peak` in seconds, `breakdown` (each signal's z-score at
  the peak, i.e. *why* it was picked), the transcript, a Twitch link to that moment, and its preview file.
- Each candidate also has `ai` (score, hook, title, spoiler_risk, reason, suggested_trim), `final_score`
  (signal score × AI score, what the ranking uses) and `warning` / `warning_reasons`.
- `previews\01_1h23m45s.mp4`: rank and peak time. **`WARN_` in the name** means a possible spoiler or
  cutscene: the AI rated its spoiler risk `low` or `high`, or the clip has 12+ seconds with no speech and
  quiet chat. Those clips are kept so you can check them, but must never be posted without a look.
- `debug_signals.png`: every signal over time, red lines at the peaks, orange spans for the clips, the blue
  dashed line is the threshold. If a clearly good moment is missing, look at which signal didn't rise.

## Tests

```powershell
python -m pytest
```

The detector test fakes a 1-hour stream with three planted moments (loud audio, then a chat explosion 6 s later)
and checks that all three are found within ±3 s.

## Stage 1 acceptance test (your part)

Run it on 2–3 real GTA 6 VODs and watch the top 15 previews of each. Pass = at least ~8 genuinely good moments
per VOD, and none start or end mid-sentence.

## Stage 2: finished vertical clips

```powershell
# 1. Once per streamer: save a frame and measure where the facecam and gameplay are
python -m clipper.edit frame out\ishowspeed\2928036561 --at 21m41s
#    -> out\...\layout\frame_grid.png   (the frame with a pixel grid: read the boxes off it)
#    -> out\...\layout\preview_9x16.png (how the vertical clip will look with the current boxes)

# 2. Make the finished clips (the best 10 that have no warning)
python -m clipper.edit render out\ishowspeed\2928036561
#    --only 1,3,7  just those ranks   --include-warnings  also flagged clips (check them first!)
#    --layout crop / no_facecam  for clips from another scene (e.g. fullscreen camera or fullscreen game)
```

Layout boxes go in `streamers.toml`, in pixels of the saved 1920x1080 frame, as `[x, y, width, height]`
from the top-left corner:

```toml
[streamers.ishowspeed]
platform = "twitch"
layout = "split"                 # "split" = facecam top 35%, gameplay below
frame = [1920, 1080]             # size of the frame the boxes were measured on
facecam = [0, 398, 457, 256]
gameplay = [460, 110, 760, 880]
```

Other layouts: `layout = "no_facecam"` (gameplay in the middle with a blurred copy filling the rest) and
`layout = "crop"` (one box fills the whole screen; good for IRL / Just Chatting). After changing boxes, run
`frame` again and look at `preview_9x16.png`. Streamers often switch scenes (gameplay + facecam, fullscreen camera,
fullscreen game), so check each finished clip and re-render the odd ones, e.g.
`python -m clipper.edit render out\ishowspeed\2928036561 --only 5 --layout crop`.

What `render` does per clip:
1. Downloads **only that section** of the VOD at up to 1080p (`out\...\sources\`).
2. Builds the 9:16 picture with one ffmpeg `filter_complex`, 1080x1920, 30 fps, H.264 (`veryfast`, CRF 20), AAC.
3. Captions: Whisper word timings (saved by `run.py`; re-transcribed for older runs), 2-4 words at a time,
   bold with an outline, on the line between facecam and gameplay. Hook text (from the AI judge) in a box for
   the first ~2 s. A small `twitch.tv/<name>` credit bottom-left. Nothing goes in the bottom 20% or at the
   right edge, where TikTok/Shorts/Reels put their buttons. The `.ass` subtitle files are in `final\work\`.
4. Audio: loudness normalised to -14 LUFS (two passes: measure, then one exact gain change). `duck_music_db` in `config.toml` (off by default) lowers the volume
   when nobody is talking. Stream audio is one mixed track, so this **only makes music quieter between
   sentences; it doesn't remove it and doesn't make a clip safe from copyright claims.**

Clips flagged with a warning (possible spoiler/cutscene) are skipped unless you pass `--include-warnings`.
Finished clips land in `out\<streamer>\<vod_id>\final\`. All the numbers (font, sizes, positions, quality) are
in the `[edit]` section of `config.toml`.
