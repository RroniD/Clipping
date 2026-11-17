# AutoClipper

Finds the most clip-worthy moments in Twitch and Kick VODs. The full plan is in [CLAUDE.md](CLAUDE.md).
**Built so far: Stage 1, the detector.** It turns a VOD URL into `candidates.json` plus short
horizontal preview clips. It doesn't edit, post or upload anything.

## How it works (Stage 1)

```
VOD URL ─► audio only (yt-dlp) ─► loudness per second ─┐
        ├► chat replay (Twitch) / live recording (Kick) ─► chat, unique chatters, hype ─┤
        └► viewer clips (Twitch API) ─────────────────► clip bumps ─────────────────────┤
                                                                                        ▼
            robust z-score each signal ─► weighted sum ─► peaks ─► rough start/end
            ─► Whisper on those windows only ─► snap to sentence edges ─► clamp 15–60 s
            ─► download only those sections as previews
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
| `clipper/plot.py` | The `--debug` plot. |

## Setup on Windows (PowerShell)

```powershell
cd $HOME\OneDrive\Desktop\Clipping
py -3.11 -m venv .venv
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
3. **GPU Whisper** (optional, much faster): `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`.
   If the GPU setup fails, the tool prints a warning and uses the `small` model on CPU instead.
   To force CPU, set `device = "cpu"` in `config.toml`.
4. **Allow-list**: add each streamer you're allowed to clip to `streamers.toml`. The tool refuses anyone else.

## Running it

```powershell
# Twitch
python run.py https://www.twitch.tv/videos/2245678901 --debug

# Kick: record chat while the stream is live (leave it running; it stops when the stream ends)
python -m clipper.kick_chat somechannel
# ...then after the stream, point run.py at the recorded chat
python run.py https://kick.com/somechannel/videos/<id> --chat out\somechannel\kick_live_20261117_1800\chat.json --debug
```

Useful flags: `--top-n 20`, `--no-transcribe` (skip Whisper), `--no-previews`, `--chat-offset 12`
(if Kick chat and VOD are out of sync by a few seconds).

Everything lands in `out\<streamer>\<vod_id>\`. Re-running the same URL reuses the audio and chat already
downloaded, so after changing `config.toml` only the scoring is redone.

### Reading the output

- `candidates.json`: best first. Each has `start`/`end`/`peak` in seconds, `breakdown` (each signal's z-score at
  the peak, i.e. *why* it was picked), the transcript, a Twitch link to that moment, and its preview file.
- `previews\01_1h23m45s.mp4`: rank and peak time.
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
