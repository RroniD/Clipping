"""Stage 1b tests. A fake API client stands in for Anthropic, so no key or money is needed."""

import json
from types import SimpleNamespace

import httpx2
import numpy as np
import pytest

import anthropic
from clipper.config import load_config
from clipper.judge import (Judge, JudgeOutputError, build_prompt, chat_sample, judge_all,
                           quiet_stretch, validate)
from clipper.signals import ChatMessage
from clipper.transcribe import Word

CFG = load_config(None)
GOOD = {"score": 8, "hook": "He did NOT see that coming", "title": "Worst getaway ever",
        "spoiler_risk": "none", "reason": "Funny crash.", "suggested_trim": {"start_offset": 2, "end_offset": -1}}


def reply(data=None, text=None, stop_reason="end_turn"):
    text = json.dumps(data) if data is not None else text
    content = [SimpleNamespace(type="thinking", thinking="")]
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeClient:
    """Returns the queued replies in order and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        create = lambda **kw: (self.calls.append(kw), self._next())[1]  # noqa: E731
        self.messages = SimpleNamespace(create=create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))

    def _next(self):
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_validate_accepts_and_cleans():
    out = validate({**GOOD, "score": 7.6, "suggested_trim": {"start_offset": 30, "end_offset": -2}})
    assert out["score"] == 8
    assert out["suggested_trim"] == {"start_offset": 10.0, "end_offset": -2.0}   # clamped to +/-10


@pytest.mark.parametrize("bad", [
    {**GOOD, "score": 11},
    {**GOOD, "score": True},
    {**GOOD, "spoiler_risk": "maybe"},
    {**GOOD, "hook": "one two three four five six seven eight nine"},
    {**GOOD, "suggested_trim": {"start_offset": "x", "end_offset": 0}},
    {k: v for k, v in GOOD.items() if k != "title"},
])
def test_validate_rejects(bad):
    with pytest.raises(JudgeOutputError):
        validate(bad)


def test_request_shape_uses_schema_and_fallbacks():
    client = FakeClient([reply(GOOD)])
    Judge(CFG, client=client).judge("prompt")
    call = client.calls[0]
    assert call["model"] == CFG["judge"]["model"]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["effort"] == "low"
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]


def test_retries_once_then_succeeds():
    client = FakeClient([reply(text="not json"), reply(GOOD)])
    assert Judge(CFG, client=client).judge("p")["score"] == 8
    assert len(client.calls) == 2


def test_refusal_is_retried_then_gives_up():
    client = FakeClient([reply(stop_reason="refusal"), reply({**GOOD, "score": 0})])
    with pytest.raises(JudgeOutputError):
        Judge(CFG, client=client).judge("p")
    assert len(client.calls) == 2


def test_quiet_stretch_needs_no_speech_and_quiet_chat():
    chat_z = np.zeros(100)
    chat_z[40:45] = 3.0                       # chat busy for 5 s in the middle
    words = [Word(10, 12, "hey"), Word(60, 61, "ok")]
    # Quiet runs inside [10, 70): 12..40 = 28 s, then 45..60 = 15 s, then 61..70 = 9 s
    assert quiet_stretch(10, 70, words, chat_z, 0.0) == 28
    assert quiet_stretch(10, 70, None, chat_z, 0.0) is None     # no transcript: can't tell
    assert quiet_stretch(10, 70, words, None, 0.0) is None      # no chat: can't tell


def test_chat_sample_spreads_and_includes_lag():
    msgs = [ChatMessage(t, "u", f"m{t}") for t in range(0, 100)]
    sample = chat_sample(msgs, 10, 20, n=5)
    assert len(sample) == 5 and sample[0] == "m10" and sample[-1] == "m28"


def test_prompt_contains_the_essentials():
    d = {"duration": 30.0, "peak": 120, "start": 108, "transcript": "no way",
         "breakdown": {"chat": 4.2, "audio": 1.0}, "quiet_stretch_seconds": 3}
    text = build_prompt(d, "somestreamer", "GTA 6 day 1", ["KEKW"])
    for piece in ["somestreamer", "GTA 6 day 1", "30 s", "chat +4.2", "no way", "- KEKW", "3 s"]:
        assert piece in text


def _results():
    return [{"peak_score": 10.0, "start": 0.0, "end": 30.0, "peak": 12, "duration": 30.0, "peak_time": "a",
             "breakdown": {}, "transcript": ""},
            {"peak_score": 8.0, "start": 100.0, "end": 130.0, "peak": 112, "duration": 30.0, "peak_time": "b",
             "breakdown": {}, "transcript": ""},
            {"peak_score": 6.0, "start": 200.0, "end": 230.0, "peak": 212, "duration": 30.0, "peak_time": "c",
             "breakdown": {}, "transcript": ""}]


def test_judge_all_ranks_and_flags():
    client = FakeClient([reply({**GOOD, "score": 2}),
                         reply({**GOOD, "score": 9, "spoiler_risk": "high"}),
                         reply({**GOOD, "score": 9})])
    results = _results()
    chat_z = np.ones(300)                                  # chat busy everywhere: no cutscene flags
    judge_all(results, [[], [], []], [], chat_z, CFG, "s", "t", Judge(CFG, client=client), log=lambda *_: None)
    assert [d["final_score"] for d in results] == [20.0, 72.0, 54.0]
    assert [d["warning"] for d in results] == [False, True, False]
    assert results[1]["warning_reasons"] == ["AI spoiler risk: high"]


def test_judge_all_without_judge_keeps_signal_order_and_flags_cutscenes():
    results = _results()
    chat_z = np.zeros(300)                                 # quiet chat everywhere
    words = [None, [Word(100, 130, "talking the whole time")], [Word(200, 201, "hi")]]
    judge_all(results, words, [], chat_z, CFG, "s", "t", None)
    assert [d["final_score"] for d in results] == [50.0, 40.0, 30.0]   # x neutral score 5
    assert [d["warning"] for d in results] == [False, False, True]     # 29 s quiet in clip c
    assert results[0]["quiet_stretch_seconds"] is None


def test_bad_key_stops_judging_instead_of_retrying_every_clip():
    response = httpx2.Response(401, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    auth_error = anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)
    client = FakeClient([auth_error, reply(GOOD), reply(GOOD)])
    results = _results()
    judge_all(results, [None] * 3, [], None, CFG, "s", "t", Judge(CFG, client=client), log=lambda *_: None)
    assert len(client.calls) == 1
    assert all(d["ai"] is None for d in results)
