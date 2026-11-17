import json
import shutil
import subprocess

import numpy as np
import pytest

from clipper import edit
from clipper.config import load_config


def words(*items):
    """words((0.0, 0.3, "hello"), ...) -> the dicts edit.py uses."""
    return [{"start": s, "end": e, "text": t} for s, e, t in items]


def test_chunks_respect_word_limit_pauses_and_sentences():
    w = words((0.0, 0.2, "one"), (0.25, 0.4, "two"), (0.45, 0.6, "three"), (0.65, 0.8, "four"),
              (2.0, 2.2, "after"), (2.25, 2.4, "pause."), (2.45, 2.6, "New"))
    chunks = edit.chunk_words(w, max_words=3, max_chars=40, max_gap=0.6)
    assert [c["text"] for c in chunks] == ["one two three", "four", "after pause.", "New"]
    for a, b in zip(chunks, chunks[1:]):
        assert a["end"] <= b["start"]          # never two captions on screen at once
    assert all(c["end"] - c["start"] >= 0.25 for c in chunks)


def test_chunks_respect_character_limit():
    w = words((0, 0.1, "extraordinarily"), (0.1, 0.2, "long"), (0.2, 0.3, "x"))
    assert [c["text"] for c in edit.chunk_words(w, max_words=4, max_chars=16)] == \
        ["extraordinarily", "long x"]


def test_ass_time_and_escaping():
    assert edit._ass_time(0) == "0:00:00.00"
    assert edit._ass_time(61.234) == "0:01:01.23"
    assert edit._ass_text("a {tag} b\\N  c") == "a (tag) b/N c"


def test_build_ass_has_captions_hook_banner_disclosure_and_credit():
    cfg = load_config()
    pos = edit.text_positions(edit.layout_for({"facecam": [0, 0, 400, 300]}), cfg)
    chunks = edit.chunk_words(words((0.5, 0.8, "no"), (0.8, 1.0, "way,"), (1.0, 1.3, "bro.")))
    ass = edit.build_ass(chunks, 20.0, "HE DID WHAT", "twitch.tv/someone", pos, cfg,
                         banner="Speed robs the wrong guy", disclosure="#ad", disclosure_seconds=3)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    dialogue = [line for line in ass.splitlines() if line.startswith("Dialogue: 1,")]
    assert len(dialogue) == 3                                   # one event per spoken word
    # Uppercased, comma/full stop removed, the spoken word highlighted in yellow.
    assert "{\\c&H00FFFF&}NO{\\c&HFFFFFF&} WAY BRO" in dialogue[0]
    assert "NO {\\c&H00FFFF&}WAY{\\c&HFFFFFF&} BRO" in dialogue[1]
    assert dialogue[0].startswith("Dialogue: 1,0:00:00.40")    # shown 0.1 s before the word
    assert ",Hook,," in ass and "HE DID WHAT" in ass
    assert ",Banner,," in ass and "Speed robs the wrong guy" in ass
    assert ",Disclosure,," in ass and "#ad" in ass
    # Text stays inside the safe zone: centred between x 120 and 888, credit at its bottom-left.
    assert "\\pos(504," in dialogue[0]
    assert "\\an1\\pos(120,1248)}twitch.tv/someone" in ass


def test_safe_zone_positions():
    cfg = load_config()
    e = cfg["edit"]
    for layout in ({"facecam": [0, 0, 400, 300]}, {"layout": "no_facecam"}, {"layout": "crop"}):
        p = edit.text_positions(edit.layout_for(layout), cfg)
        assert e["safe_top"] < p["caption"] < e["safe_bottom"]
        assert e["safe_top"] < p["hook"] < e["safe_bottom"]


def test_dead_air_is_cut_only_when_quiet_and_away_from_the_peak():
    cfg = load_config()
    # Words at 0-2 s, then a 4 s pause, then words at 6-8 s, pause 8-15, words 15-16.
    w = words((0.0, 2.0, "hello"), (6.0, 8.0, "there"), (15.0, 16.0, "end"))
    levels = np.full(170, -20.0)          # 0.1-s levels, typical -20 dB
    levels[22:58] = -40.0                 # 2.2-5.8 s: quiet  -> dead air, cut
    levels[82:148] = -20.0                # 8.2-14.8 s: loud (game action) -> kept
    segs = edit.plan_segments(w, 17.0, peak=16.0, levels=levels, cfg=cfg)
    assert segs == [(0.0, 2.15), (5.85, 17.0)]
    # Near the peak nothing is cut, even if quiet.
    levels[:] = -40.0
    levels[0:20] = -10.0
    segs = edit.plan_segments(w, 17.0, peak=4.0, levels=levels, cfg=cfg)
    assert segs[0] == (0.0, 8.15) or len(segs) == 1 or segs[0][1] >= 7.0
    assert edit.remap(6.0, [(0.0, 2.15), (5.85, 17.0)]) == pytest.approx(2.30)
    assert edit.remap(3.0, [(0.0, 2.15), (5.85, 17.0)]) is None


def test_layout_from_streamers_toml():
    split = edit.layout_for({"facecam": [0, 398, 457, 256], "gameplay": [380, 70, 1200, 960]})
    assert split.mode == "split"
    assert split.facecam == pytest.approx((0, 398 / 1080, 457 / 1920, 256 / 1080))
    # Boxes measured on a 1280x720 frame give the same fractions.
    small = edit.layout_for({"frame": [1280, 720], "facecam": [0, 0, 640, 360]})
    assert small.facecam == pytest.approx((0, 0, 0.5, 0.5))
    assert edit.layout_for({}).mode == "no_facecam"
    crop = edit.layout_for({"layout": "crop"})
    assert crop.gameplay[2] == pytest.approx(9 / 16 * 1080 / 1920)   # centred 9:16 slice
    with pytest.raises(ValueError):
        edit.layout_for({"layout": "split"})                         # split needs a facecam
    with pytest.raises(ValueError):
        edit.layout_for({"facecam": [1800, 0, 400, 300]})            # sticks out of the frame


def test_caption_sits_between_facecam_and_gameplay():
    cfg = load_config()
    split = edit.layout_for({"facecam": [0, 0, 400, 300]})
    assert edit.text_positions(split, cfg)["caption"] == edit.facecam_height(cfg) == 672


def test_duck_only_between_speech():
    w = words((0.0, 1.0, "hi"), (3.0, 4.0, "there"))
    f = edit.duck_filter(w, 10.0, 12)
    assert "between(t,1.15,2.85)" in f and "between(t,4.15,10.00)" in f
    assert edit.duck_filter(w, 10.0, 0) == ""      # off by default
    assert edit.duck_filter([], 10.0, 12) == ""    # no speech: don't duck the whole clip


def test_parse_time():
    assert edit.parse_time("1h02m03s") == 3723
    assert edit.parse_time("21m41s") == 1301
    assert edit.parse_time("0:21:41") == 1301
    assert edit.parse_time("1301") == 1301


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="needs ffmpeg")
@pytest.mark.parametrize("layout", [
    {"facecam": [0, 398, 457, 256], "gameplay": [380, 70, 1200, 960]},
    {"layout": "no_facecam"},
    {"layout": "crop"},
])
def test_render_real_ffmpeg(tmp_path, layout):
    """End-to-end ffmpeg render on a generated test video: checks the filter graph, the
    subtitles filter (libass + the relative .ass path trick) and the output format."""
    cfg = load_config()
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "4",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src)],
                   check=True)
    w = words((0.2, 0.6, "hello"), (0.7, 1.2, "world"))
    ass = tmp_path / "work" / "clip.ass"
    ass.parent.mkdir()
    lay = edit.layout_for(layout)
    ass.write_text(edit.build_ass(edit.chunk_words(w), 3.0, "HOOK", "twitch.tv/x",
                                  edit.text_positions(lay, cfg), cfg), encoding="utf-8")
    out = edit.render_clip(src, 0.5, 3.0, lay, ass, w, tmp_path / "out.mp4", cfg)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
                            "-show_entries", "format=duration", "-of", "json", str(out)],
                           capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in info["streams"])
    assert abs(float(info["format"]["duration"]) - 3.0) < 0.3


def test_two_pass_loudnorm_uses_measurements():
    cfg = load_config()
    measured = {"input_i": "-20.1", "input_tp": "-3.0", "input_lra": "5.0",
                "input_thresh": "-30.2", "target_offset": "0.4"}
    f = edit.audio_filter([], 10.0, cfg, measured)
    assert "loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=-20.1" in f and "linear=true" in f
    assert edit.audio_filter([], 10.0, cfg).startswith("[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,")


def test_split_words_are_glued_back():
    w = words((0.0, 0.2, "love"), (0.25, 0.4, "y"), (0.4, 0.6, "'all"), (1.0, 1.2, "four"), (1.2, 1.4, "-night"))
    assert " ".join(c["text"] for c in edit.chunk_words(w, max_words=4, max_chars=40)) == "love y'all four-night"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="needs ffmpeg")
def test_render_with_cuts_zoom_and_safe_overlay(tmp_path):
    """Dead-air cut (segments), punch-in zoom and the safe-zone overlay in one real render."""
    cfg = load_config()
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "5",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src)], check=True)
    w = words((0.2, 0.6, "hello"), (2.6, 3.0, "world"))
    segs = [(0.0, 1.0), (2.0, 4.0)]                      # 1-2 s cut out -> 3 s long
    lay = edit.layout_for({"facecam": [0, 398, 457, 256]})
    ass = tmp_path / "work" / "clip.ass"
    ass.parent.mkdir()
    ass.write_text(edit.build_ass(edit.chunk_words(edit.remap_words(w, segs)), 3.0, "HOOK", "twitch.tv/x",
                                  edit.text_positions(lay, cfg), cfg, banner="Context line"), encoding="utf-8")
    out = edit.render_clip(src, 0.5, 4.0, lay, ass, w, tmp_path / "out.mp4", cfg,
                           segments=segs, zoom_at=1.2, safe_overlay=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
                            "-show_entries", "format=duration", "-of", "json", str(out)],
                           capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert abs(float(info["format"]["duration"]) - 3.0) < 0.3
    assert edit.save_cover(out, 1.2, tmp_path / "cover.jpg").exists()
