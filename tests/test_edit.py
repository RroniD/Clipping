import json
import shutil
import subprocess

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


def test_build_ass_has_captions_hook_and_credit():
    cfg = load_config()
    chunks = [{"start": 0.5, "end": 1.0, "text": "no way, bro."}]
    ass = edit.build_ass(chunks, 20.0, "HE DID WHAT", "twitch.tv/someone",
                         {"caption": 672, "hook": 326}, cfg)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "\\pos(540,672)}NO WAY BRO" in ass          # uppercased, comma/full stop removed
    assert ",Hook,," in ass and "HE DID WHAT" in ass
    assert "twitch.tv/someone" in ass
    # Credit sits above the bottom 20% UI zone.
    assert f"\\pos(40,{int(1920 * 0.8) - 10})" in ass


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
