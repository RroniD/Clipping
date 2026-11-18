"""Self-check of finished clips: asks the cheap model, only the layout may be fixed."""

import json
from types import SimpleNamespace

from clipper import selfcheck


class FakeJudge:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def _ask(self, system, content, schema, max_tokens=None, model=None):
        self.calls.append(content)
        return self.answer


def test_check_reports_and_suggests_a_layout(monkeypatch):
    monkeypatch.setattr(selfcheck, "final_frames", lambda *a, **k: [(0.3, b"\xff\xd8a"), (5, b"\xff\xd8b")])
    judge = FakeJudge({"face_ok": False, "captions_ok": True, "hook_readable": True, "first_frame_score": 6,
                       "problems": ["the top shows chat, not the streamer"], "fix_layout": "no_facecam"})
    r = selfcheck.check(judge, "v.mp4", 30, 12, "split", "He's about to...", "Speed vs cops")
    assert r["fix_layout"] == "no_facecam" and r["problems"] == ["the top shows chat, not the streamer"]
    assert sum(1 for c in judge.calls[0] if c["type"] == "image") == 2
    # Suggesting the layout it already has = nothing to fix.
    judge.answer = {**judge.answer, "fix_layout": "split"}
    assert selfcheck.check(judge, "v.mp4", 30, 12, "split", "", "")["fix_layout"] == "none"


def test_no_frames_no_check(monkeypatch):
    monkeypatch.setattr(selfcheck, "final_frames", lambda *a, **k: [])
    assert selfcheck.check(FakeJudge({}), "v.mp4", 30, None, "split", "", "") is None