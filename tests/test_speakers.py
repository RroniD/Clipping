"""Speaker colours: sentence grouping, main voice = most talk time, caption colour."""

from clipper import edit, speakers
from clipper.config import load_config


def w(s, e, t):
    return {"start": s, "end": e, "text": t}


def test_sentence_spans_split_on_pauses_and_punctuation():
    words = [w(0, 0.4, "yo"), w(0.4, 0.8, "bro."), w(0.9, 1.2, "what"), w(3.0, 3.4, "run")]
    assert speakers.sentence_spans(words) == [(0, 1), (2, 2), (3, 3)]


def test_main_speaker_is_the_one_who_talks_most():
    words = [w(0, 5, "a"), w(5, 6, "b"), w(6, 12, "c")]
    spans = [(0, 0), (1, 1), (2, 2)]
    assert speakers.main_speaker_labels(spans, [3, 7, 3], words) == [0, 1, 0]
    assert speakers.main_speaker_labels(spans, [-1, -1, -1], words) == [0, 0, 0]   # can't tell -> all main


def test_other_speaker_words_get_their_own_colour():
    cfg = load_config()
    pos = edit.text_positions(edit.layout_for({"facecam": [0, 0, 400, 300]}), cfg)
    words = [{**w(0.5, 0.9, "no"), "speaker": 0}, {**w(0.9, 1.3, "way"), "speaker": 1}]
    ass = edit.build_ass(edit.chunk_words(words), 5.0, "", "", pos, cfg)
    first = [l for l in ass.splitlines() if l.startswith("Dialogue: 1,")][0]
    assert cfg["edit"]["other_speaker_color"] + "}WAY" in first     # not spoken yet, but other voice: cyan