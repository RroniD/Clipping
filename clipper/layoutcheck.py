"""Pick the right 9:16 layout for each clip by looking at its frames.

Why: streamers switch scenes. Speed's stream is sometimes "screen share + webcam box"
(layout "split" works), sometimes fullscreen Fortnite where the webcam box spot shows his
chat overlay instead of his face (split then puts CHAT on top of the clip), and sometimes a
fullscreen camera with no game ("crop"). One fixed layout per streamer is wrong for some clips.

Two opinions are combined:
1. This local check (free, no extra installs): a few small frames, looking inside the
   streamer's facecam box for a FACE (skin-coloured pixels in a steady box) versus chat
   text or game scenery. Also: a big face in the middle of the whole frame = fullscreen camera.
2. The AI judge's `layout` answer (it already sees 15 frames of the clip).
If they agree, that's the layout. If they disagree, the SAFE choice wins: "no_facecam"
(the whole picture over a blurred copy), which never puts chat or an empty box on top.
A layout you pick by hand in the review queue always wins over both.

Skin detection (YCrCb colour space): skin of every tone has a narrow range of "redness"
(Cr) and "blueness" (Cb), largely independent of brightness (Y). It's a classic, fast,
dependency-free trick; it can be fooled by orange/brown game scenery, which is why the box
must also be steady (an overlay webcam is; scenery behind it isn't) and the AI's opinion is used.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from clipper.fetch import require_tool

W, H = 320, 180
LAYOUTS = ("split", "no_facecam", "crop")


def rgb_frames(video: Path, start: float, end: float, n: int = 6) -> np.ndarray:
    """n small RGB frames spread over [start, end] (seconds into the file): (n, 180, 320, 3)."""
    ffmpeg = require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    dur = max(0.5, end - start)
    result = subprocess.run([ffmpeg, "-v", "error", "-ss", f"{max(0.0, start):.2f}", "-t", f"{dur:.2f}",
                             "-i", str(video), "-vf", f"fps={n / dur:.4f},scale={W}:{H}", "-pix_fmt", "rgb24",
                             "-f", "rawvideo", "-"], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg couldn't read frames from {Path(video).name}")
    size = W * H * 3
    count = len(result.stdout) // size
    return np.frombuffer(result.stdout[: count * size], dtype=np.uint8).reshape(count, H, W, 3)


def skin_fraction(rgb: np.ndarray) -> float:
    """Share of pixels whose colour looks like skin (YCrCb ranges)."""
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
    mask = (y > 40) & (cr >= 135) & (cr <= 175) & (cb >= 85) & (cb <= 130)
    return float(mask.mean()) if mask.size else 0.0


def _crop(frames: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = box
    return frames[:, int(y * H): int((y + h) * H), int(x * W): int((x + w) * W)]


def region_stats(frames: np.ndarray, box: tuple[float, float, float, float]) -> dict:
    """Skin share, movement between frames, and fine detail inside a box (fractions of the frame)."""
    region = _crop(frames, box)
    if region.size == 0 or len(region) == 0:
        return {"skin": 0.0, "motion": 0.0, "detail": 0.0}
    gray = region.mean(axis=3)
    motion = float(np.abs(np.diff(gray, axis=0)).mean()) if len(gray) > 1 else 0.0
    detail = float(np.mean([np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean() for g in gray]))
    return {"skin": round(float(np.mean([skin_fraction(f) for f in region])), 3),
            "motion": round(motion, 2), "detail": round(detail, 2)}


CENTER = (0.25, 0.10, 0.50, 0.80)   # middle of the frame: where a fullscreen camera shows the face


def local_layout(frames: np.ndarray, facecam: tuple[float, float, float, float] | None,
                 cfg: dict) -> tuple[str, str, dict]:
    """(layout, reason, numbers) from the frames alone.

    Measured on 15 real Speed clips: a webcam box is STABLE (it changes ~7 brightness levels
    between frames: same room, same light), while game scenery showing through the box spot
    changes 27-40. So "a face in the box" = enough skin-coloured pixels AND a stable box.
    That also rejects orange/sandy game scenery, which can look skin-coloured but moves.
    - Face in the facecam box (skin >= face_skin, motion <= box_max_motion)  -> split
    - No face in the box, but a big still-ish face fills the middle            -> crop (fullscreen camera)
    - Otherwise                                                               -> no_facecam (safe)
    """
    lc = cfg["layout"]
    center = region_stats(frames, CENTER)
    stats = {"center": center}
    if facecam is not None:
        box = region_stats(frames, facecam)
        stats["facecam"] = box
        if box["skin"] >= lc["face_skin"] and box["motion"] <= lc["box_max_motion"]:
            return "split", f"face in the facecam box (skin {box['skin']:.0%}, steady)", stats
        if box["detail"] >= lc["text_detail"] and box["skin"] < 0.05:
            reason = "facecam box shows text/chat or the game, not a face"
        else:
            reason = f"no face in the facecam box (skin {box['skin']:.0%}, movement {box['motion']:.0f})"
    else:
        reason = "no facecam box set for this streamer"
    if center["skin"] >= lc["fullscreen_face_skin"] and center["motion"] <= lc["fullscreen_max_motion"]:
        return "crop", f"{reason}; a face fills the middle of the frame (fullscreen camera)", stats
    return "no_facecam", reason, stats


def find_facecam(frames: np.ndarray, cfg: dict) -> tuple[tuple[float, float, float, float], float] | None:
    """Look for the webcam box anywhere along the edges of the frame: (box, skin share) or None.

    Streamers move their webcam overlay between scenes and streams, so the box measured once in
    streamers.toml is often in the wrong place. Webcam overlays are almost always 16:9-ish
    rectangles touching a side or corner. We try a grid of those and keep the one that looks most
    like a webcam: lots of skin-coloured pixels in a steady picture (the same test as above).
    """
    lc = cfg["layout"]
    best = None
    # "Steady" relative to this clip: an energetic streamer moves a lot, but the game in the middle
    # of the screen still moves more. So the box may move up to 80% as much as the game does.
    game_motion = region_stats(frames, CENTER)["motion"]
    max_motion = max(lc["box_max_motion"], 0.8 * game_motion)
    for w in (0.18, 0.22, 0.26, 0.30):
        h = w * (W / H) * (9 / 16)   # height of a 16:9 box w wide, as a share of the frame height
        for x in (0.0, 1 - w):   # left or right edge: a centred "face" is usually a game character
            for y in np.arange(0.0, 1 - h + 1e-6, 0.05):
                box = (x, float(y), w, h)
                s = region_stats(frames, box)
                if s["skin"] >= lc["face_skin"] and s["motion"] <= max_motion:
                    score = s["skin"] - 0.005 * s["motion"]
                    if best is None or score > best[1]:
                        best = (box, score)
    if best is None:
        return None
    return (tuple(round(v, 3) for v in best[0]), round(best[1], 3))


def is_face_box(frames: np.ndarray, box, cfg: dict) -> bool:
    """Does this box look like a webcam with a face in this clip (same rule as find_facecam)?"""
    lc = cfg["layout"]
    s = region_stats(frames, box)
    max_motion = max(lc["box_max_motion"], 0.8 * region_stats(frames, CENTER)["motion"])
    return s["skin"] >= lc["face_skin"] and s["motion"] <= max_motion


def vote_facecam(results: list[dict], video_of, cfg: dict) -> int:
    """Streamers keep their webcam in one place most of a stream. Take the box found most often
    across this VOD's clips and re-check every clip with it; clips where it holds a face get
    "split" with that box, the others stay as they are. Returns how many clips use it.
    video_of(d) -> (video path, video start) for a clip, or None."""
    found = [tuple(d["facecam_box"]) for d in results if d.get("facecam_box")]
    if len(found) < 2:
        return 0
    from collections import Counter
    # Boxes within 0.05 of each other count as the same spot.
    key = lambda b: (round(b[0] / 0.05), round(b[1] / 0.05))   # noqa: E731
    winner_key, _ = Counter(key(b) for b in found).most_common(1)[0]
    box = max((b for b in found if key(b) == winner_key), key=lambda b: b[2])
    used = 0
    for d in results:
        src = video_of(d)
        if d.get("layout") is None or src is None:
            continue
        if d["layout"] == "split" and not d.get("facecam_box"):
            continue   # the measured box (streamers.toml) already shows the face: keep it
        try:
            frames = rgb_frames(src[0], d["start"] - src[1], d["end"] - src[1])
        except Exception:
            continue
        if len(frames) and is_face_box(frames, box, cfg):
            d["facecam_box"], d["layout"] = list(box), "split"
            d["layout_reason"] = "webcam at the streamer's usual spot in this stream"
            used += 1
        elif d.get("facecam_box") and key(tuple(d["facecam_box"])) != winner_key:
            d.pop("facecam_box", None)        # an odd one-off spot: don't trust it
            d["layout"], d["layout_reason"] = "no_facecam", "webcam not at its usual spot: safe layout"
    return used


def reaction_score(video: Path, video_start: float, window_start: float, window_end: float, peak: float,
                   facecam, cfg: dict, fps: int = 4) -> tuple[bool, float]:
    """How big the streamer's visible reaction is around the peak: (face_in_box, score).

    Movement in the facecam box (frame-to-frame difference at 4 frames a second) in the 5 s
    before to 8 s after the peak, divided by the box's typical movement over the whole window.
    A streamer jumping up, screaming or leaning into the camera scores 2-5; sitting still ~1.
    Only meaningful when the box really shows a face (otherwise it's game scenery moving).
    """
    if facecam is None:
        return False, 0.0
    n = max(2, int((window_end - window_start) * fps))
    frames = rgb_frames(video, window_start - video_start, window_end - video_start, n=n)
    if len(frames) < 3:
        return False, 0.0
    layout, _, _ = local_layout(frames, facecam, cfg)
    if layout != "split":
        return False, 0.0
    gray = _crop(frames, facecam).mean(axis=3)
    motion = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))           # one value per frame step
    times = window_start + (np.arange(len(motion)) + 1) * (window_end - window_start) / len(frames)
    near = motion[(times >= peak - 5) & (times <= peak + 8)]
    if len(near) == 0:
        return True, 0.0
    return True, round(float(near.max() / (np.median(motion) + 1.0)), 2)


def choose_layout(d: dict, video: Path | None, video_start: float, facecam, cfg: dict) -> None:
    """Set d["layout"] and d["layout_reason"] for one candidate.
    video: a copy of the moment (run.py's 360p analysis clip, or live mode's own recording)
    that starts at VOD second `video_start`."""
    local = None
    found = None
    if video is not None:
        try:
            frames = rgb_frames(video, d["start"] - video_start, d["end"] - video_start)
            if len(frames):
                layout, reason, _ = local_layout(frames, facecam, cfg)
                local = (layout, reason)
                if layout != "split" and cfg["layout"]["find_facecam"]:
                    # The measured box shows no face: maybe the webcam moved. Look for it.
                    found = find_facecam(frames, cfg)
                    if found:
                        local = ("split", f"webcam found at a different spot ({reason})")
        except Exception:
            local = None   # couldn't read frames: fall back to the AI / the default
    d["layout"], d["layout_reason"] = combine(local, (d.get("ai") or {}).get("layout"),
                                              facecam is not None or found is not None)
    if found and d["layout"] == "split":
        d["facecam_box"] = list(found[0])     # fractions of the frame; the editor uses this box
    else:
        d.pop("facecam_box", None)


def combine(local: tuple[str, str] | None, ai: str | None, has_facecam: bool) -> tuple[str, str]:
    """Merge the local check and the AI's suggestion into (layout, reason).
    Disagreement -> no_facecam: it shows the whole picture and never puts chat or an empty
    box on top, so it's never badly wrong."""
    if ai == "split" and not has_facecam:
        ai = "no_facecam"            # no facecam box measured: split isn't possible
    if local and ai:
        if local[0] == ai:
            return ai, f"{local[1]}; AI agrees"
        return "no_facecam", f"local check said {local[0]} ({local[1]}), AI said {ai}: using the safe layout"
    if local:
        return local
    if ai:
        return ai, "AI's choice (frames couldn't be checked locally)"
    return ("split" if has_facecam else "no_facecam"), "streamer's default"
