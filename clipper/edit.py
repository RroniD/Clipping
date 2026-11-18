"""Stage 2: turn picked moments into finished vertical 9:16 clips.

For each candidate in candidates.json:
  1. download ONLY that section of the VOD at up to 1080p (never the full VOD)
  2. build the 9:16 picture with ffmpeg: facecam on top + gameplay below ("split"),
     gameplay over a blurred copy of itself ("no_facecam"), or one box filling the
     whole screen ("crop", good for IRL / Just Chatting)
  3. burn in word-by-word captions, the hook text (first ~2 s) and a small streamer credit,
     all from one .ass subtitle file
  4. loudness-normalise the audio to -14 LUFS (what TikTok / Shorts / Reels play at)

Usage (PowerShell, from the project folder):
    python -m clipper.edit frame out\\ishowspeed\\2928036561          # save a frame to measure the layout
    python -m clipper.edit render out\\ishowspeed\\2928036561         # make the finished clips

Why one .ass file for captions, hook AND credit?
  ffmpeg's `drawtext` needs a font file path, and Windows paths inside ffmpeg filter
  strings need awkward escaping. The `subtitles` filter (libass) finds fonts by name and
  handles positioning, outlines and boxes for us. One file = one filter = one place to look.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clipper import fetch
from clipper.config import PROJECT_ROOT, load_campaigns, load_config, load_streamers

OUT_W, OUT_H = 1080, 1920   # 9:16 output size


# --------------------------------------------------------------------------- layout

@dataclass
class Layout:
    """Where things are in the source picture, as fractions (0..1) of its width/height.

    Fractions, not pixels: you measure boxes on a 1920x1080 frame, but a VOD section might
    only exist in 720p. Fractions work at any resolution.
    """
    mode: str                                  # "split" | "no_facecam" | "crop"
    gameplay: tuple[float, float, float, float]
    facecam: tuple[float, float, float, float] | None = None


def _box_to_fractions(box, frame_w: float, frame_h: float) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in box)
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame_w + 1 or y + h > frame_h + 1:
        raise ValueError(f"layout box {list(box)} does not fit inside a {frame_w:.0f}x{frame_h:.0f} frame")
    return (x / frame_w, y / frame_h, w / frame_w, h / frame_h)


def layout_for(streamer_cfg: dict) -> Layout:
    """Read a streamer's layout from streamers.toml (all keys optional).

        layout   = "split"            # "split" (facecam top) | "no_facecam" | "crop"
        frame    = [1920, 1080]       # size of the frame you measured the boxes on
        facecam  = [x, y, w, h]       # pixels, from the top-left corner
        gameplay = [x, y, w, h]       # the main picture (gameplay, or the camera for IRL)

    No layout given -> "split" if a facecam box exists, otherwise "no_facecam".
    """
    frame_w, frame_h = (float(v) for v in streamer_cfg.get("frame", [1920, 1080]))
    facecam = streamer_cfg.get("facecam")
    mode = streamer_cfg.get("layout") or ("split" if facecam else "no_facecam")
    if mode not in ("split", "no_facecam", "crop"):
        raise ValueError(f'unknown layout "{mode}" (use "split", "no_facecam" or "crop")')
    if mode == "split" and not facecam:
        raise ValueError('layout = "split" needs a facecam = [x, y, w, h] box')

    if streamer_cfg.get("gameplay"):
        gameplay = _box_to_fractions(streamer_cfg["gameplay"], frame_w, frame_h)
    elif mode == "crop":
        # A centred 9:16 slice of a 16:9 frame: width = (9/16 * h) / w of the frame.
        w = (9 / 16) * (frame_h / frame_w)
        gameplay = ((1 - w) / 2, 0.0, w, 1.0)
    else:
        gameplay = (0.0, 0.0, 1.0, 1.0)
    face = _box_to_fractions(facecam, frame_w, frame_h) if (facecam and mode == "split") else None
    return Layout(mode=mode, gameplay=gameplay, facecam=face)


def _crop(box: tuple[float, float, float, float]) -> str:
    """ffmpeg crop that works at any input size: iw/ih are the input width/height."""
    x, y, w, h = box
    return f"crop=iw*{w:.5f}:ih*{h:.5f}:iw*{x:.5f}:ih*{y:.5f}"


def _fill(w: int, h: int) -> str:
    """Scale up until the picture covers w x h, then cut off what sticks out (no black bars)."""
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"


def facecam_height(cfg: dict) -> int:
    return int(round(OUT_H * cfg["edit"]["facecam_share"] / 2)) * 2   # even number for H.264


def video_filter(layout: Layout, cfg: dict, inp: str = "[0:v]") -> str:
    """The video part of ffmpeg's filter_complex. Input is `inp` ([0:v]), output is [base]."""
    fps = cfg["edit"]["fps"]
    if layout.mode == "split":
        top = facecam_height(cfg)
        return (f"{inp}split=2[fc][gp];"
                f"[fc]{_crop(layout.facecam)},{_fill(OUT_W, top)},setsar=1[top];"
                f"[gp]{_crop(layout.gameplay)},{_fill(OUT_W, OUT_H - top)},setsar=1[bot];"
                f"[top][bot]vstack=inputs=2,fps={fps}[base]")
    if layout.mode == "crop":
        return f"{inp}{_crop(layout.gameplay)},{_fill(OUT_W, OUT_H)},setsar=1,fps={fps}[base]"
    # no_facecam: the gameplay in the middle; above and below is a blurred, darker, zoomed-in
    # copy of the same picture. The blur is done on a small copy (270x480) and scaled back up
    # because blurring at full size is slow and looks the same.
    # no_facecam_zoom (1.3): the game is scaled 30% wider than the screen and its sides are
    # trimmed, so it's big enough to see on a phone (the action is almost always in the middle).
    blur = cfg["edit"]["blur_strength"]
    zw = int(round(OUT_W * float(cfg["edit"]["no_facecam_zoom"]) / 2) * 2)
    return (f"{inp}{_crop(layout.gameplay)},split=2[a][b];"
            f"[a]{_fill(270, 480)},boxblur={blur}:2,scale={OUT_W}:{OUT_H},eq=brightness=-0.12[bg];"
            f"[b]scale={zw}:{OUT_H}:force_original_aspect_ratio=decrease,crop='min(iw,{OUT_W})':ih[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps}[base]")


def text_positions(layout: Layout, cfg: dict) -> dict[str, int]:
    """Where text goes (output pixels), always inside the SAFE ZONE.

    TikTok / Shorts / Reels draw their own buttons, captions and progress bar over the video:
    roughly the top 15%, the bottom 35% and a strip on the right. Anything important must sit
    in the box safe_left..safe_right x safe_top..safe_bottom (default 120-888 x 288-1248).
    split: captions sit on the line between facecam and gameplay (eyes are already there).
    """
    e = cfg["edit"]
    if layout.mode == "split":
        caption_y = facecam_height(cfg)
    elif layout.mode == "no_facecam":
        caption_y = int(OUT_H * e["caption_y_no_facecam"])
    else:
        caption_y = int(OUT_H * e["caption_y_crop"])
    caption_y = min(max(caption_y, e["safe_top"] + 60), e["safe_bottom"] - 60)
    hook_y = min(max(int(OUT_H * e["hook_y"]), e["safe_top"] + 40), e["safe_bottom"] - 40)
    return {"caption": caption_y, "hook": hook_y, "x": (e["safe_left"] + e["safe_right"]) // 2,
            "disclosure": hook_y + 90, "credit_x": e["safe_left"], "credit_y": e["safe_bottom"]}


# --------------------------------------------------------------------------- captions (.ass)

def chunk_words(words: list[dict], max_words: int = 3, max_chars: int = 16,
                max_gap: float = 0.6) -> list[dict]:
    """Group words into short caption chunks of 2-4 words: {"start", "end", "text"}.

    A new chunk starts when the current one is full (words or characters), after a pause,
    or after a word that ends a sentence. Short chunks are easier to read on a phone.
    `words` are {"start", "end", "text"} with times in seconds from the clip start.
    """
    # Whisper sometimes splits one word in two: "y" + "'all", "four" + "-night".
    # Glue those pieces back onto the previous word so captions don't read "Y 'ALL".
    merged: list[dict] = []
    for w in words:
        text = w["text"].strip()
        if not text:
            continue
        if merged and text[0] in "'’-" and len(text) > 1:
            merged[-1] = {**merged[-1], "end": w["end"], "text": merged[-1]["text"] + text}
        else:
            merged.append({**w, "text": text})

    chunks: list[list[dict]] = []
    for w in merged:
        text = w["text"]
        cur = chunks[-1] if chunks else None
        new_chunk = (
            cur is None
            or len(cur) >= max_words
            or len(" ".join(x["text"] for x in cur)) + 1 + len(text) > max_chars
            or w["start"] - cur[-1]["end"] > max_gap
            or cur[-1]["text"].rstrip().endswith((".", "!", "?"))
        )
        if new_chunk:
            chunks.append([])
        chunks[-1].append({**w, "text": text})

    result = []
    for i, c in enumerate(chunks):
        start = c[0]["start"]
        end = c[-1]["end"] + 0.3                       # linger a moment after the last word
        if i + 1 < len(chunks):
            end = min(end, chunks[i + 1][0]["start"])  # but never overlap the next chunk
        end = max(end, start + 0.25)                   # and never flash by too fast to read
        result.append({"start": start, "end": end, "text": " ".join(x["text"] for x in c), "words": c})
    return result


def _ass_time(t: float) -> str:
    """ASS time format is h:mm:ss.cc (hundredths)."""
    cs = int(round(max(0.0, t) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _ass_text(text: str) -> str:
    """Make text safe for ASS: braces start override tags and backslashes start commands."""
    return re.sub(r"\s+", " ", text.replace("\\", "/").replace("{", "(").replace("}", ")")).strip()


def _caption_text(text: str, upper: bool) -> str:
    # Commas/full stops look messy in 3-word captions; keep ? and ! because they carry tone.
    text = re.sub(r"[.,;:]+(?=\s|$)", "", text)
    return text.upper() if upper else text


def build_ass(chunks: list[dict], duration: float, hook: str, credit: str,
              positions: dict[str, int], cfg: dict, banner: str = "", disclosure: str = "",
              disclosure_seconds: float = 0.0, punch_words: list[str] | tuple = ()) -> str:
    """Write the whole subtitle file: captions, hook, context banner, disclosure and credit.

    Colours in ASS are &HAABBGGRR (alpha first, then blue-green-red, alpha 00 = solid).
    Alignment 5 = centred on the \\pos point; 1 = bottom-left; 8 = top-centre.
    BorderStyle 3 = solid box behind the text (used for the hook and the banner).

    Captions: the chunk (2-3 words) stays on screen and the word being SPOKEN turns yellow,
    the style most viral clips use (it pulls the eye along and works on mute). Each chunk
    appears `caption_lead` (0.1 s) early, because text that appears exactly with the sound
    feels late. A chunk "pops" in (110% -> 100% size) to catch the eye.
    The hook (first ~2 s) and then the banner (context line) share the top spot.
    """
    e = cfg["edit"]
    font = e["font"]
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{e['caption_size']},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,7,3,5,80,80,0,1
Style: Hook,{font},{e['hook_size']},&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,22,0,5,110,110,0,1
Style: Credit,{font},{e['credit_size']},&H30FFFFFF,&H30FFFFFF,&H60000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,1,40,40,0,1
Style: Banner,{font},{e['banner_size']},&H00FFFFFF,&H00FFFFFF,&HB4000000,&HB4000000,-1,0,0,0,100,100,0,0,3,14,0,5,{e['safe_left']},{OUT_W - e['safe_right']},0,1
Style: Disclosure,Arial,30,&H00FFFFFF,&H00FFFFFF,&H90000000,&H90000000,-1,0,0,0,100,100,0,0,3,8,0,8,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    cx, cy, hy = positions["x"], positions["caption"], positions["hook"]
    lead, hl, upper = float(e["caption_lead"]), e["caption_highlight"], e["caption_uppercase"]
    punch = {re.sub(r"[^\w']", "", p).lower() for p in punch_words if p}

    def styled(t: str, raw: str, active: bool) -> str:
        # Punch words (the AI's 1-2 key words) are always red and grow when spoken;
        # the word being spoken is yellow.
        if re.sub(r"[^\w']", "", raw).lower() in punch:
            grow = "\\fscx118\\fscy118" if active else ""
            return f"{{\\c{e['punch_color']}{grow}}}{t}{{\\c&HFFFFFF&\\fscx100\\fscy100}}"
        return f"{{\\c{hl}}}{t}{{\\c&HFFFFFF&}}" if active else t

    for c in chunks:
        words = c.get("words") or [{"start": c["start"], "end": c["end"], "text": c["text"]}]
        texts = [_ass_text(_caption_text(w["text"], upper)) for w in words]
        for i, w in enumerate(words):
            # One event per spoken word: same chunk, with word i highlighted.
            a = (c["start"] if i == 0 else w["start"]) - lead
            b = (words[i + 1]["start"] if i + 1 < len(words) else c["end"]) - lead
            if b - a < 0.05:
                continue
            shown = " ".join(styled(t, words[k]["text"], k == i) for k, t in enumerate(texts))
            pop = "\\fscx110\\fscy110\\t(0,80,\\fscx100\\fscy100)" if i == 0 else ""
            lines.append(f"Dialogue: 1,{_ass_time(a)},{_ass_time(b)},Caption,,0,0,0,,"
                         f"{{\\pos({cx},{cy}){pop}}}{shown}")
    hook_end = min(e["hook_seconds"], duration) if hook else 0.0
    if hook:
        # \fad(0,200): no fade in (it must be readable on the very first frame), 0.2 s fade out.
        lines.append(f"Dialogue: 2,{_ass_time(0)},{_ass_time(hook_end)},Hook,,0,0,0,,"
                     f"{{\\pos({cx},{hy})\\fad(0,200)}}{_ass_text(hook)}")
    if banner and e["banner"]:
        # The context line: tells a stranger who and what this is, for the rest of the clip.
        lines.append(f"Dialogue: 2,{_ass_time(hook_end)},{_ass_time(duration)},Banner,,0,0,0,,"
                     f"{{\\pos({cx},{hy})\\fad(150,0)}}{_ass_text(banner)}")
    if disclosure and disclosure_seconds > 0:
        # Campaign disclosure ("#ad", "Sponsored by ...") burned into the opening seconds.
        lines.append(f"Dialogue: 3,{_ass_time(0)},{_ass_time(min(disclosure_seconds, duration))},Disclosure,,0,0,0,,"
                     f"{{\\an8\\pos({cx},{positions['disclosure']})}}{_ass_text(disclosure)}")
    if credit:
        # Bottom-left corner of the safe zone.
        lines.append(f"Dialogue: 0,{_ass_time(0)},{_ass_time(duration)},Credit,,0,0,0,,"
                     f"{{\\an1\\pos({positions['credit_x']},{positions['credit_y']})}}{_ass_text(credit)}")
    return head + "\n".join(lines) + "\n"


def credit_text(streamer: str, platform: str) -> str:
    return f"{'kick.com' if platform == 'kick' else 'twitch.tv'}/{streamer}"


# --------------------------------------------------------------------------- audio

def duck_filter(words: list[dict], duration: float, duck_db: float, min_gap: float = 1.0) -> str:
    """Lower the volume wherever the streamer isn't talking (gaps of min_gap+ seconds).

    Honest limit: stream audio is ONE mixed track, so this can't remove in-game music or the
    radio. It only makes it quieter between sentences; under speech the music is still there.
    It does not make a clip safe from copyright claims. Returns "" when ducking is off.
    """
    if duck_db <= 0 or not words:
        return ""
    gaps, prev_end = [], 0.0
    for w in sorted(words, key=lambda w: w["start"]):
        if w["start"] - prev_end >= min_gap:
            gaps.append((prev_end + 0.15, w["start"] - 0.15))
        prev_end = max(prev_end, w["end"])
    if duration - prev_end >= min_gap:
        gaps.append((prev_end + 0.15, duration))
    if not gaps:
        return ""
    enable = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in gaps)
    return f"volume=-{duck_db}dB:enable='{enable}',"


def _loudnorm(cfg: dict, measured: dict | None = None) -> str:
    target = f"loudnorm=I={cfg['edit']['loudness_lufs']}:TP=-1.5:LRA=11"
    if not measured:
        return target
    # Second pass: with the real measurements, loudnorm can apply one exact gain change
    # ("linear") instead of guessing as it goes, so the clip lands on the target.
    return (f"{target}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true")


def measure_loudness(source: Path, offset: float, duration: float, duck: str, cfg: dict) -> dict | None:
    """First loudnorm pass: measure the clip's audio (fast, audio only).

    Why two passes? One-pass loudnorm adjusts on the fly without knowing what's coming, so
    short clips end up 1-2 LU off target (we measured -13.7 to -16 LUFS on real clips).
    """
    ffmpeg = fetch.require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    result = fetch.run([ffmpeg, "-hide_banner", "-nostats", "-ss", f"{offset:.3f}", "-i", str(source),
                        "-t", f"{duration:.3f}", "-vn",
                        "-af", f"{duck}{_loudnorm(cfg)}:print_format=json", "-f", "null", "-"])
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr)
    if not match:
        return None
    data = json.loads(match.group(0))
    # Silent clips measure as "-inf", which loudnorm can't use: fall back to one pass.
    if any("inf" in str(v) for v in data.values()):
        return None
    return data


def audio_filter(words: list[dict], duration: float, cfg: dict, measured: dict | None = None) -> str:
    """[0:a] -> [aout]: optional ducking, then loudness normalisation to -14 LUFS.

    loudnorm upsamples internally, so we resample back to 48 kHz for AAC.
    """
    duck = duck_filter(words, duration, cfg["edit"]["duck_music_db"])
    return f"[0:a]{duck}{_loudnorm(cfg, measured)},aresample=48000[aout]"


# --------------------------------------------------------------------------- rendering

# --------------------------------------------------------------------------- pacing: dead air, teaser, zoom

def clip_levels(source: Path, offset: float, duration: float, step: float = 0.1) -> np.ndarray:
    """Loudness (dB) of the clip's audio every `step` seconds (decoded at 8 kHz, it's quick)."""
    ffmpeg = fetch.require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    result = subprocess.run([ffmpeg, "-v", "error", "-ss", f"{offset:.3f}", "-t", f"{duration:.3f}",
                             "-i", str(source), "-vn", "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                            capture_output=True)
    samples = np.frombuffer(result.stdout[: len(result.stdout) // 2 * 2], dtype=np.int16).astype(np.float64)
    n = int(8000 * step)
    if len(samples) < n:
        return np.zeros(0)
    blocks = samples[: len(samples) // n * n].reshape(-1, n)
    rms = np.sqrt((blocks ** 2).mean(axis=1)) / 32768.0
    return 20 * np.log10(np.maximum(rms, 10 ** (-90 / 20)))


def plan_segments(words: list[dict], duration: float, peak: float, levels: np.ndarray, cfg: dict,
                  step: float = 0.1) -> list[tuple[float, float]]:
    """Which parts of the clip to KEEP, as (start, end) in clip seconds.

    Dead air = a pause in speech longer than `dead_air_gap` where the game is also quiet
    (a few dB below the clip's typical level). Those get cut down to a short breath.
    By default only at the EDGES (before the first word, after the last): a real Speed clip lost
    the play that led to his win when pauses inside the clip were cut. `dead_air_inside = true`
    also cuts inside pauses. When in doubt, nothing is cut.
    Why "game also quiet"? A silent streamer during a car chase is not dead air: the action
    is the point. And we never cut within 3 s before / 4 s after the peak (the moment itself).
    Faster pacing keeps people watching (retention is what the platforms reward).
    """
    e = cfg["edit"]
    if not e["dead_air"] or not words or len(levels) == 0:
        return [(0.0, duration)]
    typical = float(np.median(levels))
    keep_air = float(e["dead_air_keep"])
    cuts = []
    edges = [(-1.0, 0.0)] + [(w["start"], w["end"]) for w in sorted(words, key=lambda w: w["start"])] \
        + [(duration, duration + 1)]
    last = len(edges) - 2
    for i, ((_, prev_end), (next_start, _)) in enumerate(zip(edges, edges[1:])):
        if not e["dead_air_inside"] and 0 < i < last:
            continue   # only the quiet bit before the first word and after the last one
        # At the clip's own start/end there's nothing to keep air around: trim to the very edge.
        a = 0.0 if i == 0 else prev_end + keep_air
        b = duration if i == last else min(duration, next_start - keep_air)
        if b - a < e["dead_air_gap"] - 2 * keep_air:
            continue
        if a < peak + 4 and b > peak - 3:
            continue                                      # never touch the moment itself
        quiet = levels[int(a / step): max(int(a / step) + 1, int(b / step))]
        if len(quiet) and float(np.mean(quiet)) <= typical - e["dead_air_quiet_db"]:
            cuts.append((a, b))
    keep, t = [], 0.0
    for a, b in cuts:
        if a > t:
            keep.append((t, a))
        t = b
    if t < duration:
        keep.append((t, duration))
    return [(round(a, 2), round(b, 2)) for a, b in keep if b - a > 0.05] or [(0.0, duration)]


def remap(t: float, segments: list[tuple[float, float]], lead_in: float = 0.0) -> float | None:
    """Clip time -> time in the edited video (after cutting dead air and adding a teaser).
    None if that moment was cut out."""
    out = lead_in
    for a, b in segments:
        if a <= t <= b:
            return out + (t - a)
        out += b - a
    return None


def remap_words(words: list[dict], segments: list[tuple[float, float]], lead_in: float = 0.0) -> list[dict]:
    result = []
    for w in words:
        s, e = remap(w["start"], segments, lead_in), remap(w["end"], segments, lead_in)
        if s is not None and e is not None and e > s:
            result.append({**w, "start": s, "end": e})
    return result


# --------------------------------------------------------------------------- rendering

def render_clip(source: Path, offset: float, duration: float, layout: Layout, ass_path: Path,
                words: list[dict], out_path: Path, cfg: dict,
                segments: list[tuple[float, float]] | None = None, zoom_at: float | None = None,
                audio_file: Path | None = None, safe_overlay: bool = False) -> Path:
    """Render one finished clip with a single ffmpeg call.

    segments: the parts of [0, duration] to keep, in order (dead air cut out; a teaser is just
      an extra segment at the front). Each is cut with trim/atrim and they're joined with concat.
    zoom_at: time in the EDITED video for a short punch-in zoom (the peak reaction).
    audio_file: replacement audio for the clip (music removed), same timeline as the clip.

    The subtitles filter is given the .ass file by its bare name and ffmpeg runs with
    cwd = that folder. Why: inside a filter string, "C:\\..." needs the colon and every
    backslash escaped, which is easy to get wrong. A bare file name needs no escaping.
    """
    ffmpeg = fetch.require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    e = cfg["edit"]
    segments = segments or [(0.0, duration)]
    a_in = "[1:a]" if audio_file else "[0:a]"
    measured = measure_loudness(source.resolve(), offset, duration,
                                duck_filter(words, duration, e["duck_music_db"]), cfg)
    parts = []
    if segments == [(0.0, duration)]:
        v_label, a_label = "[0:v]", a_in
    else:
        # split/asplit: one copy of the picture and sound per piece we keep.
        n = len(segments)
        parts.append("[0:v]split=" + str(n) + "".join(f"[sv{i}]" for i in range(n)))
        parts.append(f"{a_in}asplit=" + str(n) + "".join(f"[sa{i}]" for i in range(n)))
        joins = ""
        for i, (a, b) in enumerate(segments):
            parts.append(f"[sv{i}]trim=start={a:.3f}:end={b:.3f},setpts=PTS-STARTPTS[v{i}]")
            parts.append(f"[sa{i}]atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS[a{i}]")
            joins += f"[v{i}][a{i}]"
        parts.append(f"{joins}concat=n={len(segments)}:v=1:a=1[vcat][acat]")
        v_label, a_label = "[vcat]", "[acat]"
    parts.append(video_filter(layout, cfg, v_label))
    last = "[base]"
    if zoom_at is not None and e["zoom"]:
        # Punch-in: for a moment the whole picture is scaled up ~12% (centre crop), a classic
        # editing trick that makes the reaction land. eval=frame re-evaluates the size per frame.
        z, a, b = float(e["zoom_factor"]), max(0.0, zoom_at - 0.15), zoom_at + float(e["zoom_seconds"])
        zw, zh = int(round(OUT_W * z / 2) * 2), int(round(OUT_H * z / 2) * 2)
        parts.append(f"{last}scale=w='if(between(t,{a:.2f},{b:.2f}),{zw},{OUT_W})'"
                     f":h='if(between(t,{a:.2f},{b:.2f}),{zh},{OUT_H})':eval=frame,crop={OUT_W}:{OUT_H}[zoomed]")
        last = "[zoomed]"
    if safe_overlay:
        # Debug: green box = the safe zone where important things must be.
        parts.append(f"{last}drawbox=x={e['safe_left']}:y={e['safe_top']}:w={e['safe_right'] - e['safe_left']}"
                     f":h={e['safe_bottom'] - e['safe_top']}:color=lime@0.7:t=4[safe]")
        last = "[safe]"
    parts.append(f"{last}subtitles={ass_path.name}[vout]")
    duck = duck_filter(words, duration, e["duck_music_db"]) if segments == [(0.0, duration)] else ""
    parts.append(f"{a_label}{duck}{_loudnorm(cfg, measured)},aresample=48000[aout]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # -ss and -t BEFORE -i: ffmpeg only reads that stretch of the source (fast, and the
    # clip's timestamps start at 0, so trims and subtitles use clip time).
    cmd = [ffmpeg, "-v", "error", "-y", "-ss", f"{offset:.3f}", "-t", f"{duration:.3f}",
           "-i", str(source.resolve())]
    if audio_file:
        cmd += ["-i", str(audio_file.resolve())]
    cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", e["preset"], "-crf", str(e["crf"]), "-pix_fmt", "yuv420p",
            # 0 = use every CPU core. Live mode sets fewer so a game/stream on the laptop stays smooth.
            "-threads", str(e.get("threads", 0)),
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out_path.resolve())]
    fetch.run(cmd, cwd=str(ass_path.parent))
    return out_path


def save_cover(video: Path, at: float, out: Path) -> Path | None:
    """cover.jpg: the frame at the peak, for the platform's thumbnail picker."""
    ffmpeg = fetch.require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    result = subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{max(0.0, at):.2f}", "-i", str(video),
                             "-frames:v", "1", "-q:v", "2", str(out)], capture_output=True)
    return out if result.returncode == 0 and out.exists() else None


# --------------------------------------------------------------------------- music removal (optional)

def demucs_python() -> Path | None:
    """Demucs lives in its OWN Python environment (.venv-demucs), because it brings PyTorch
    with its own copies of NVIDIA libraries, which could break faster-whisper's GPU setup in
    the main environment. We just run that environment's python as a separate program."""
    for rel in ("Scripts/python.exe", "bin/python"):
        p = PROJECT_ROOT / ".venv-demucs" / rel
        if p.exists():
            return p
    return None


def remove_music(source: Path, offset: float, duration: float, work: Path, cfg: dict) -> tuple[Path | None, str]:
    """Optional: split the clip's audio into voice + everything else with Demucs and mix the
    'everything else' (music AND game sounds) far down. Returns (audio file or None, warning).

    Off unless audio.music_removal = "demucs" AND demucs is installed. Why optional: Demucs
    needs PyTorch (~2.5 GB) and a few seconds of GPU per clip. Even with it, quiet music can
    survive; the warning says when music was loud before removal (copyright-claim risk).
    """
    if cfg["audio"]["music_removal"] != "demucs":
        return None, ""
    python = demucs_python()
    if python is None:
        return None, ""
    ffmpeg = fetch.require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    work.mkdir(parents=True, exist_ok=True)
    wav = work / f"{source.stem}_{offset:.0f}.wav"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{offset:.3f}", "-t", f"{duration:.3f}",
                    "-i", str(source), "-vn", "-ac", "2", "-ar", "44100", str(wav)], check=True)
    sep = subprocess.run([str(python), "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
                          "-o", str(work / "demucs"), str(wav)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    stems = work / "demucs" / "htdemucs" / wav.stem
    voice, rest = stems / "vocals.wav", stems / "no_vocals.wav"
    if sep.returncode != 0 or not voice.exists():
        return None, "music removal failed: check for copyrighted music before posting"
    v_db = float(np.mean(clip_levels(voice, 0, duration, 1.0)))
    r_db = float(np.mean(clip_levels(rest, 0, duration, 1.0)))
    if r_db <= v_db - cfg["audio"]["music_loud_db"]:
        # Background is quiet next to the voice: keep the original sound. In GTA the
        # "background" is also crashes, sirens and explosions, which are part of the moment.
        return None, ""
    mixed = work / f"{wav.stem}_clean.wav"
    down = float(cfg["audio"]["music_down_db"])
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(voice), "-i", str(rest), "-filter_complex",
                    f"[1:a]volume=-{down}dB[r];[0:a][r]amix=inputs=2:normalize=0[a]", "-map", "[a]", str(mixed)],
                   check=True)
    return mixed, f"loud background music was turned down {down:.0f} dB (check the sound before posting)"


def _words_for(d: dict, clip_start: float, clip_end: float) -> list[dict] | None:
    """Words saved by run.py (VOD seconds) -> clip seconds. None if this run didn't save any."""
    if "words" not in d:
        return None
    return [{"start": s - clip_start, "end": e - clip_start, "text": t}
            for s, e, t in d["words"] if e > clip_start and s < clip_end]


def render_vod(out_dir: Path, cfg: dict, top_n: int | None = None, only: list[int] | None = None,
               include_warnings: bool = False, captions: bool = True,
               layout_override: str | None = None, campaign: str | None = None,
               safe_overlay: bool = False) -> list[Path]:
    """Make finished clips for one VOD folder (the one run.py wrote candidates.json into).

    layout_override: many streamers switch scenes (gameplay with facecam -> fullscreen camera),
    so one layout doesn't fit every clip. Re-render those clips with e.g. --only 5 --layout crop.
    """
    data = json.loads((out_dir / "candidates.json").read_text(encoding="utf-8"))
    streamer, platform, url = data["streamer"], data["platform"], data["url"]
    streamers = load_streamers()
    if streamer not in streamers:   # hard rule 3, checked again here
        raise SystemExit(f"Refusing to edit: {streamer} is not in streamers.toml.")
    e = cfg["edit"]
    print(f"Layout for {streamer}: " + (layout_override or "chosen per clip (see layout_reason)"
                                         if cfg["layout"]["auto"] else
                                         streamer_layout(streamers[streamer], layout_override).mode))

    picked = []
    for d in data["candidates"]:
        if only and d["rank"] not in only:
            continue
        if not only and not d.get("kept", True):
            continue   # the AI judge (or the explore picks) didn't keep this one
        if d.get("warning") and not include_warnings:
            # Hard rule 2: possible spoiler/cutscene clips are never turned into finished clips
            # unless you ask for it after looking at the preview.
            print(f"  skipping #{d['rank']} {d['peak_time']}: warning ({', '.join(d.get('warning_reasons', []))})")
            continue
        picked.append(d)
    if not only:
        picked = picked[: top_n or e["top_n"]]

    renderer = ClipRenderer(cfg)
    finals = []
    for i, d in enumerate(picked, 1):
        print(f"  [{i}/{len(picked)}] {clip_name(d)} ({float(d['end']) - float(d['start']):.0f} s)")
        finals.append(renderer.render(out_dir, data, d, layout_mode=layout_override, captions=captions,
                                      campaign=campaign, safe_overlay=safe_overlay))
    return finals


def choose_layouts(out_dir: Path, cfg: dict) -> list[tuple[int, str | None, str, str]]:
    """Choose the layout of every kept clip of an already-processed VOD (no re-run needed) and
    save it in candidates.json. Returns [(rank, old layout, new layout, reason)]."""
    from clipper.layoutcheck import choose_layout
    path = out_dir / "candidates.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    streamer_cfg = load_streamers().get(data["streamer"], {})
    facecam = layout_for(streamer_cfg).facecam if streamer_cfg.get("facecam") else None
    changes = []
    for d in data["candidates"]:
        if not d.get("kept", True):
            continue
        old = d.get("layout")
        video = out_dir / d["analysis"] if d.get("analysis") else None
        choose_layout(d, video, d.get("window_start", d["start"]), facecam, cfg)
        changes.append((d["rank"], old, d["layout"], d["layout_reason"]))
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return changes


def clip_name(d: dict) -> str:
    """File name (no extension) of a candidate's finished clip, e.g. 03_WARN_1h02m03s."""
    return f"{d['rank']:02d}_{'WARN_' if d.get('warning') else ''}{d['peak_time']}"


def streamer_layout(streamer_cfg: dict, mode: str | None = None) -> Layout:
    """The streamer's layout, or another layout mode for a clip from a different scene."""
    streamer_cfg = dict(streamer_cfg)
    if mode:
        streamer_cfg["layout"] = mode
        if mode != "split":
            # The gameplay box was measured for the split scene; other scenes use the full frame
            # (or a centred 9:16 slice for "crop").
            streamer_cfg.pop("gameplay", None)
    return layout_for(streamer_cfg)


class ClipRenderer:
    """Renders single clips. Used by `render` above and by the review queue (Stage 3) when
    you trim a clip, change its hook or switch its layout.

    It's a class only so the Whisper model is loaded once and reused for every clip.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.transcriber = None

    def _transcribe(self, source: Path, offset: float, duration: float) -> list[dict]:
        if self.transcriber is None:
            from clipper.transcribe import Transcriber
            self.transcriber = Transcriber(self.cfg)
        raw = self.transcriber.transcribe_window(source, offset, offset + duration)
        return [{"start": w.start - offset, "end": w.end - offset, "text": w.text} for w in raw]

    def render(self, out_dir: Path, data: dict, d: dict, start: float | None = None,
               end: float | None = None, layout_mode: str | None = None, hook: str | None = None,
               out_name: str | None = None, captions: bool = True,
               source: Path | None = None, source_start: float = 0.0, banner: str | None = None,
               campaign: str | None = None, safe_overlay: bool = False) -> Path:
        """Render candidate `d` of candidates.json `data` and return the finished .mp4.

        start/end (VOD seconds), layout_mode, hook and banner (context line) default to the
        candidate's own values. campaign: a profile name from campaigns.toml (its disclosure,
        credit and length limit are applied).
        source/source_start: a video that's already on disk and starts at VOD second
        `source_start` (live mode cuts it from its own recording), instead of downloading.
        A small .json next to the .ass file records what was used, so the review queue knows.
        """
        cfg, e = self.cfg, self.cfg["edit"]
        streamer, platform, url = data["streamer"], data["platform"], data["url"]
        streamers = load_streamers()
        if streamer not in streamers:   # hard rule 3, checked again here
            raise SystemExit(f"Refusing to edit: {streamer} is not in streamers.toml.")
        # Layout: one you chose (queue / --layout) wins; else the per-clip choice from run.py
        # (clipper.layoutcheck + the AI); else the streamer's default.
        auto = d.get("layout") if cfg["layout"]["auto"] else None
        layout = streamer_layout(streamers[streamer], layout_mode or auto)
        if layout_mode and layout_mode != auto:
            layout_reason = "picked by hand"
        elif auto:
            layout_reason = d.get("layout_reason", "")
        else:
            layout_reason = "streamer's default"
        profile = load_campaigns().get(campaign) if campaign else None
        if campaign and profile is None:
            raise ValueError(f"no campaign called {campaign!r} in campaigns.toml")
        start = float(d["start"] if start is None else start)
        end = float(d["end"] if end is None else end)
        if profile and profile.get("max_seconds") and end - start > profile["max_seconds"]:
            end = start + float(profile["max_seconds"])   # the campaign's length limit wins
        duration = end - start
        name = out_name or clip_name(d)

        # 1. Download just this section at up to 1080p. A little padding on both sides
        #    so the cut in step 3 lands on exact frames.
        if source is None:
            pad = cfg["download"]["padding"]
            source_start = max(0.0, start - pad)
            source = out_dir / "sources" / f"{start:.0f}-{end:.0f}.mp4"
            fetch.download_section(url, source_start, end + pad, source, e["source_format"])
        offset = start - source_start

        # 2. Word timings: from run.py if it saved them, else transcribe this clip now.
        #    run.py only saved words inside the original clip, so a clip trimmed LONGER needs
        #    a fresh transcription to get captions for the extra seconds.
        words: list[dict] | None = []
        if captions:
            lo, hi = d.get("words_range") or (float(d["start"]), float(d["end"]))
            inside = start >= float(lo) - 0.5 and end <= float(hi) + 0.5   # snapping can nudge 0.1-0.3 s
            words = _words_for(d, start, end) if inside else None
            if words is None:
                words = self._transcribe(source, offset, duration)

        # 3. Pacing: cut dead air, optional teaser of the peak first, zoom on the peak.
        peak = float(d.get("peak", start + duration / 2)) - start
        levels = clip_levels(source, offset, duration) if e["dead_air"] else np.zeros(0)
        segments = plan_segments(words, duration, peak, levels, cfg)
        lead_in = 0.0
        if e["teaser_seconds"] > 0 and 0 <= peak < duration:
            t0 = max(0.0, peak - 0.3)
            teaser = (round(t0, 2), round(min(duration, t0 + float(e["teaser_seconds"])), 2))
            lead_in = teaser[1] - teaser[0]
            segments = [teaser] + segments
            main = segments[1:]
        else:
            main = segments
        edited = sum(b - a for a, b in segments)
        words_edit = remap_words(words, main, lead_in)
        zoom_at = remap(peak, main, lead_in)

        # 4. Subtitle file: captions, hook, context banner, campaign disclosure, credit.
        ai = d.get("ai") or {}
        if hook is None:
            hook = (ai.get("hook") or "").strip() if e["hook"] else ""
        if banner is None:
            banner = (ai.get("context_line") or "").strip()
        credit = (profile or {}).get("credit") or (credit_text(streamer, platform) if e["credit"] else "")
        disclosure = (profile or {}).get("disclosure", "")
        chunks = chunk_words(words_edit, e["caption_max_words"], e["caption_max_chars"], e["caption_gap"])
        work = out_dir / "final" / "work"
        work.mkdir(parents=True, exist_ok=True)
        ass_path = work / f"{name}.ass"
        ass_path.write_text(build_ass(chunks, edited, hook, credit, text_positions(layout, cfg), cfg,
                                      banner=banner, disclosure=disclosure,
                                      disclosure_seconds=float((profile or {}).get("disclosure_seconds", 3)),
                                      punch_words=ai.get("punch_words", [])),
                            encoding="utf-8")

        # 5. Optional music removal, then one ffmpeg render, then the cover image.
        audio_file, music_warning = remove_music(source, offset, duration, work, cfg)
        final = render_clip(source, offset, duration, layout, ass_path, words, out_dir / "final" / f"{name}.mp4",
                            cfg, segments=segments, zoom_at=zoom_at, audio_file=audio_file,
                            safe_overlay=safe_overlay)
        if e["cover"]:
            save_cover(final, zoom_at if zoom_at is not None else min(1.0, edited / 2), final.with_suffix(".jpg"))
        (work / f"{name}.json").write_text(json.dumps(
            {"rank": d["rank"], "start": start, "end": end, "layout": layout.mode,
             "layout_reason": layout_reason, "hook": hook,
             "banner": banner, "campaign": campaign, "duration": round(edited, 2),
             "cuts": len(segments) - 1 - (1 if lead_in else 0),
             "warnings": [music_warning] if music_warning else []}),
            encoding="utf-8")
        return final


# --------------------------------------------------------------------------- layout helper

def parse_time(text: str) -> float:
    """'1h02m03s', '21m41s', '1301' or '0:21:41' -> seconds."""
    if re.fullmatch(r"[\d.]+", text):
        return float(text)
    if ":" in text:
        total = 0.0
        for part in text.split(":"):
            total = total * 60 + float(part)
        return total
    parts = dict((u, float(n)) for n, u in re.findall(r"([\d.]+)([hms])", text))
    if not parts:
        raise ValueError(f"can't read time {text!r}")
    return parts.get("h", 0) * 3600 + parts.get("m", 0) * 60 + parts.get("s", 0)


def save_frame(out_dir: Path, at: float | None, cfg: dict) -> list[Path]:
    """Save a 1920x1080 frame + a copy with a pixel grid, so you can measure layout boxes.

    If streamers.toml already has boxes for this streamer, they're drawn on the grid copy
    (red = facecam, green = gameplay) and a 9:16 preview of the layout is saved too.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    data = json.loads((out_dir / "candidates.json").read_text(encoding="utf-8"))
    if at is None:
        at = float(data["candidates"][0]["peak"]) if data["candidates"] else 60.0
    folder = out_dir / "layout"
    src = fetch.download_section(data["url"], at, at + 3, folder / f"frame_{at:.0f}.mp4",
                                 cfg["edit"]["source_format"])
    ffmpeg = fetch.require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    frame = folder / "frame.png"
    fetch.run([ffmpeg, "-v", "error", "-y", "-ss", "1", "-i", str(src), "-frames:v", "1",
               "-vf", "scale=1920:1080", str(frame)])

    # Grid copy: big figure, tick labels every 100 px, light lines every 50 px.
    img = plt.imread(frame)
    fig, ax = plt.subplots(figsize=(19.2, 11.6), dpi=100)
    ax.imshow(img)
    ax.set_xticks(range(0, 1921, 100))
    ax.set_yticks(range(0, 1081, 100))
    ax.set_xticks(range(0, 1921, 50), minor=True)
    ax.set_yticks(range(0, 1081, 50), minor=True)
    ax.grid(which="major", color="yellow", alpha=0.6, linewidth=0.8)
    ax.grid(which="minor", color="yellow", alpha=0.25, linewidth=0.5)
    ax.tick_params(labeltop=True, labelright=True, labelsize=8)
    ax.set_title("Measure boxes as [x, y, width, height] in these pixels (0,0 = top-left)")
    streamer_cfg = load_streamers().get(data["streamer"], {})
    for key, colour in (("facecam", "red"), ("gameplay", "lime")):
        if streamer_cfg.get(key):
            x, y, w, h = streamer_cfg[key]
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=colour, linewidth=3))
            ax.text(x + 6, y + 30, key, color=colour, fontsize=14, weight="bold")
    grid = folder / "frame_grid.png"
    fig.savefig(grid, bbox_inches="tight")
    plt.close(fig)
    saved = [frame, grid]

    # 9:16 preview of the current layout (same filter as the real render, on the still frame).
    if streamer_cfg:
        layout = layout_for(streamer_cfg)
        preview = folder / "preview_9x16.png"
        graph = video_filter(layout, cfg).replace(f",fps={cfg['edit']['fps']}", "")
        fetch.run([ffmpeg, "-v", "error", "-y", "-i", str(frame), "-filter_complex", graph,
                   "-map", "[base]", "-frames:v", "1", str(preview)])
        saved.append(preview)
    return saved


# --------------------------------------------------------------------------- CLI

def main() -> None:
    p = argparse.ArgumentParser(description="AutoClipper Stage 2: vertical 9:16 clips.")
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("frame", help="save a frame (+ pixel grid) to measure the layout boxes")
    f.add_argument("out_dir", type=Path, help=r"the VOD folder, e.g. out\ishowspeed\2928036561")
    f.add_argument("--at", help="time in the VOD, e.g. 1h02m03s (default: the top clip's peak)")
    lp = sub.add_parser("layouts", help="(re)choose the layout of each kept clip from its frames")
    lp.add_argument("out_dir", type=Path, help=r"the VOD folder, e.g. out\ishowspeed\2928036561")
    r = sub.add_parser("render", help="make the finished clips")
    r.add_argument("out_dir", type=Path, help=r"the VOD folder, e.g. out\ishowspeed\2928036561")
    r.add_argument("--top-n", type=int, help="override edit.top_n from config.toml")
    r.add_argument("--only", help="only these ranks, e.g. 1,3,7")
    r.add_argument("--include-warnings", action="store_true",
                   help="also edit clips flagged as possible spoiler/cutscene (check them first!)")
    r.add_argument("--no-captions", action="store_true", help="skip captions (and Whisper)")
    r.add_argument("--layout", choices=["split", "no_facecam", "crop"],
                   help="use this layout instead of the streamer's (for clips from another scene)")
    r.add_argument("--campaign", help="apply this campaign profile from campaigns.toml")
    r.add_argument("--safe-overlay", action="store_true",
                   help="debug: draw the safe zone (green box) on the clips")
    p.add_argument("--config", type=Path, help="use a different config.toml")
    args = p.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    if not (out_dir / "candidates.json").exists():
        raise SystemExit(f"No candidates.json in {out_dir}. Run run.py on the VOD first.")

    if args.command == "frame":
        for path in save_frame(out_dir, parse_time(args.at) if args.at else None, cfg):
            print(f"Saved {path}")
        return
    if args.command == "layouts":
        for rank, old, new, reason in choose_layouts(out_dir, cfg):
            print(f"  #{rank:2d}: {old or '-'} -> {new}   ({reason})")
        return
    only = [int(x) for x in args.only.split(",")] if args.only else None
    finals = render_vod(out_dir, cfg, args.top_n, only, args.include_warnings, not args.no_captions,
                        args.layout, args.campaign, args.safe_overlay)
    print(f"{len(finals)} finished clip(s) in {out_dir / 'final'}")
    if finals:
        from dotenv import load_dotenv
        from clipper import notify
        load_dotenv(PROJECT_ROOT / ".env")
        notify.send(f"{len(finals)} clips ready", f"{out_dir.parent.name}: open the review queue to approve them.",
                    click=notify.queue_url(cfg))


if __name__ == "__main__":
    try:
        main()
    except fetch.ToolMissing as exc:
        raise SystemExit(f"Missing tool: {exc}")
