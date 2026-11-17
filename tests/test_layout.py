"""Automatic layout choice: a face in the facecam box -> split; chat/game there -> no_facecam;
fullscreen camera -> crop; disagreement between the local check and the AI -> no_facecam."""

import shutil
import subprocess

import numpy as np
import pytest

from clipper import layoutcheck
from clipper.config import load_config

CFG = load_config()
BOX = (0.0, 0.37, 0.24, 0.24)          # Speed's facecam box (fractions of the frame)
SKIN = (170, 110, 80)                   # an RGB skin tone (passes the YCrCb skin test)


def frames(n=6, box_fill=None, box_moving=False, center_fill=None, center_moving=False, seed=0):
    """Fake stream frames: random 'game' everywhere, with optional skin-coloured regions."""
    rng = np.random.default_rng(seed)
    f = rng.integers(0, 255, size=(n, 180, 320, 3)).astype(np.uint8)
    f[..., 2] = 255                                         # bluish game: not skin
    f[..., 0] //= 3
    if center_fill is not None:
        y0, y1, x0, x1 = int(0.10 * 180), int(0.90 * 180), int(0.25 * 320), int(0.75 * 320)
        f[:, y0:y1, x0:x1] = center_fill
        if center_moving:
            f[:, y0:y1, x0:x1] = (f[:, y0:y1, x0:x1] + rng.integers(0, 60, size=(n, 1, 1, 1))).clip(0, 255)
    if box_fill is not None:
        y0, y1, x0, x1 = int(BOX[1] * 180), int((BOX[1] + BOX[3]) * 180), 0, int(BOX[2] * 320)
        base = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
        base[:] = box_fill
        for i in range(n):
            region = base.astype(int) + rng.integers(-6, 6, size=base.shape)   # webcam noise: steady
            if box_moving:
                region += rng.integers(-90, 90, size=(1, 1, 3))                  # scenery: changes a lot
            f[i, y0:y1, x0:x1] = region.clip(0, 255)
    return f


def test_skin_detector():
    assert layoutcheck.skin_fraction(np.full((10, 10, 3), SKIN, dtype=np.uint8)) == 1.0
    assert layoutcheck.skin_fraction(np.full((10, 10, 3), (30, 80, 220), dtype=np.uint8)) == 0.0


def test_face_in_a_steady_box_is_split():
    layout, reason, stats = layoutcheck.local_layout(frames(box_fill=SKIN), BOX, CFG)
    assert layout == "split" and "face" in reason


def test_skin_coloured_but_moving_box_is_not_a_face():
    """Orange/sandy game scenery behind the box spot: skin-coloured, but it moves."""
    layout, _, _ = layoutcheck.local_layout(frames(box_fill=SKIN, box_moving=True), BOX, CFG)
    assert layout == "no_facecam"


def test_no_face_in_box_is_no_facecam():
    layout, reason, _ = layoutcheck.local_layout(frames(), BOX, CFG)
    assert layout == "no_facecam" and ("not a face" in reason or "no face" in reason)


def test_fullscreen_camera_is_crop():
    layout, reason, _ = layoutcheck.local_layout(frames(center_fill=SKIN), BOX, CFG)
    assert layout == "crop" and "fullscreen camera" in reason


def test_combine_rules():
    assert layoutcheck.combine(("split", "face"), "split", True) == ("split", "face; AI agrees")
    layout, reason = layoutcheck.combine(("split", "face"), "no_facecam", True)
    assert layout == "no_facecam" and "safe" in reason          # disagree -> safe layout
    assert layoutcheck.combine(None, "split", False)[0] == "no_facecam"   # no box measured: no split
    assert layoutcheck.combine(("crop", "camera"), None, True) == ("crop", "camera")
    assert layoutcheck.combine(None, None, True)[0] == "split"   # nothing known: streamer default


def test_choose_layout_without_a_video_uses_the_ai():
    d = {"start": 0, "end": 10, "ai": {"layout": "crop"}}
    layoutcheck.choose_layout(d, None, 0, BOX, CFG)
    assert d["layout"] == "crop" and "AI" in d["layout_reason"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_rgb_frames_from_a_video(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30",
                    "-t", "6", "-c:v", "libx264", "-preset", "ultrafast", str(video)], check=True)
    f = layoutcheck.rgb_frames(video, 1, 5, n=4)
    assert f.shape[1:] == (180, 320, 3) and 3 <= len(f) <= 5
