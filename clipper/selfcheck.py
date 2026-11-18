"""Self-check: the AI looks at the FINISHED vertical clip, like a viewer would.

Why: every step before this works on the horizontal stream. Problems only visible in the
final 9:16 video (the face cut in half by the crop, captions sitting on top of the action,
an unreadable hook, a boring first frame) are caught here. It costs about $0.0005 per clip
with Haiku (4 small frames).

What it may change: only the LAYOUT (re-render once with the suggested one), and only if you
didn't pick the layout by hand. It never changes the cut - the setup always stays in.
Everything else is reported as a note in the review queue.
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from clipper.fetch import require_tool

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "face_ok": {"type": "boolean", "description": "faces are not cut off by the crop"},
        "captions_ok": {"type": "boolean", "description": "captions don't cover the important action"},
        "hook_readable": {"type": "boolean"},
        "first_frame_score": {"type": "integer", "description": "1-10: would the first frame stop a scroller"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "fix_layout": {"type": "string", "enum": ["none", "split", "no_facecam", "crop"]},
    },
    "required": ["face_ok", "captions_ok", "hook_readable", "first_frame_score", "problems", "fix_layout"],
    "additionalProperties": False,
}

CHECK_PROMPT = """You check a finished vertical (9:16) short-form clip before a human reviews it. You \
get 4 frames (start, ~1/3, the key moment, near the end) and its texts. Judge only what you SEE:
- face_ok: false if a person's face is cut in half by the crop, or the top area meant for the \
streamer's face shows something else (chat, an empty corner, game menus).
- captions_ok: false if the captions sit on top of the important action.
- hook_readable: the text in the first frame is readable and not covering the action.
- first_frame_score: 1-10, would this first frame make someone stop scrolling.
- problems: short plain-English notes for the human (empty if fine).
- fix_layout: the layout that would fix a face/crop problem: "split" (face on top, game below), \
"no_facecam" (the whole game picture with a blurred background), "crop" (a camera filling the \
screen), or "none" if the layout is fine. Be conservative: "none" unless clearly wrong."""


def final_frames(video: Path, duration: float, peak_at: float | None, width: int = 360) -> list[tuple[float, bytes]]:
    ffmpeg = require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    times = [0.3, duration / 3, peak_at if peak_at is not None else duration / 2, max(0.5, duration - 1.5)]
    frames = []
    for t in times:
        r = subprocess.run([ffmpeg, "-v", "error", "-ss", f"{max(0.0, t):.2f}", "-i", str(video), "-frames:v", "1",
                            "-vf", f"scale={width}:-2", "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
                           capture_output=True)
        if r.returncode == 0 and r.stdout:
            frames.append((t, r.stdout))
    return frames


def check(judge, video: Path, duration: float, peak_at: float | None, layout: str, hook: str,
          banner: str) -> dict | None:
    """Ask the (cheap) judge model to look at the finished clip. None if it couldn't."""
    frames = final_frames(video, duration, peak_at)
    if not frames:
        return None
    content = [{"type": "text", "text": f"Layout used: {layout}. Hook (first 2 s): {hook or '(none)'}. "
                                        f"Banner: {banner or '(none)'}. Clip length {duration:.0f} s."}]
    for t, jpeg in frames:
        content.append({"type": "text", "text": f"Frame at {t:.1f} s:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": base64.standard_b64encode(jpeg).decode("ascii")}})
    try:
        data = judge._ask(CHECK_PROMPT, content, CHECK_SCHEMA, max_tokens=4000)
    except Exception:
        return None
    fix = data.get("fix_layout") if data.get("fix_layout") in ("split", "no_facecam", "crop") else "none"
    return {"face_ok": bool(data.get("face_ok")), "captions_ok": bool(data.get("captions_ok")),
            "hook_readable": bool(data.get("hook_readable")),
            "first_frame_score": int(data.get("first_frame_score") or 0),
            "problems": [str(p)[:160] for p in (data.get("problems") or [])][:5],
            "fix_layout": fix if fix != layout else "none"}
