"""Stage 1b tests. A fake API client stands in for Anthropic, so no key or money is needed."""

import json
from types import SimpleNamespace

import httpx2
import numpy as np
import pytest

import anthropic
from clipper.config import load_config
from clipper.judge import (Judge, JudgeOutputError, ai_quality, apply_cut, build_content, build_prompt,
                           chat_sample, final_score, frame_times, judge_all, quiet_stretch,
                           timestamped_transcript, validate, warning_reasons)
from clipper.signals import ChatMessage
from clipper.transcribe import Word

CFG = load_config(None)
GOOD = {"category": "funny", "standalone": 8, "hook_strength": 7, "payoff": 9, "score": 8,
        "visual_payoff": 7, "facecam_reaction": "big", "is_cutscene": False, "menu_or_map": False,
        "layout": "split", "hook_options": ["He did NOT see that coming", "5 stars in 10 seconds", "Chat warned him"],
        "title_options": ["Speed trusted his crew... mistake", "Worst getaway ever"], "punch_words": ["WASTED"],
        "caption": "he really thought that would work", "story_beat": False, "translations": [],
        "setup_start": 5.0, "cut_start": 5.0, "cut_end": 30.0, "hook": "He did NOT see that coming",
        "context_line": "Speed trusts his crew with the money", "title": "Speed trusted his crew... mistake",
        "hashtags": ["#ishowspeed", "gta", "funny"], "spoiler_risk": "none", "reason": "Funny crash."}


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
    out = validate({**GOOD, "score": 7.6, "cut_end": 99, "category": "weird",
                    "facecam_reaction": "huge"}, window_len=40)
    assert out["score"] == 8
    assert out["cut_end"] == 40                         # clamped into the window
    assert out["category"] == "other" and out["facecam_reaction"] == "none"
    assert out["hashtags"] == ["ishowspeed", "gta", "funny"]   # '#' stripped


@pytest.mark.parametrize("bad", [
    {**GOOD, "score": 11},
    {**GOOD, "standalone": True},
    {**GOOD, "visual_payoff": 0},
    {**GOOD, "spoiler_risk": "maybe"},
    {**GOOD, "hook": "   "},
    {**GOOD, "cut_start": 20, "cut_end": 21},           # too short
    {**GOOD, "cut_start": "x"},
    {k: v for k, v in GOOD.items() if k != "title"},
])
def test_validate_rejects(bad):
    with pytest.raises(JudgeOutputError):
        validate(bad, window_len=40)


def test_haiku_request_has_no_fallbacks():
    """Haiku has no server-side fallback: the plain endpoint, no beta, no fallbacks field."""
    client = FakeClient([reply(GOOD)])
    Judge(CFG, client=client).judge([{"type": "text", "text": "p"}], 40)
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert "fallbacks" not in call and "betas" not in call
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["effort"] == "low"
    assert "temperature" not in call and call["max_tokens"] >= 4000


def test_bigger_models_use_fallbacks():
    cfg = load_config(None)
    cfg["judge"]["model"] = "claude-opus-5-5"
    client = FakeClient([reply(GOOD)])
    Judge(cfg, client=client).judge("p", 40)
    assert client.calls[0]["fallbacks"] == "default"


def test_retries_once_then_succeeds():
    client = FakeClient([reply(text="not json"), reply(GOOD)])
    assert Judge(CFG, client=client).judge("p", 40)["score"] == 8
    assert len(client.calls) == 2


def test_refusal_gives_up_and_warns():
    client = FakeClient([reply(stop_reason="refusal"), reply(stop_reason="refusal")])
    results = _results()[:1]
    judge_all(results, [None], [], None, CFG, "s", "t", Judge(CFG, client=client), log=lambda *_: None)
    assert results[0]["ai"] is None
    assert "AI declined to rate this clip" in results[0]["warning_reasons"]


def test_content_has_timestamps_frames_and_viewer_clips():
    d = {"window_start": 100.0, "window_end": 140.0, "peak": 120, "start": 105, "end": 135,
         "breakdown": {"chat": 4.2}, "chat_mood": {"funny": 70, "hype": 30},
         "framecheck_reasons": ["possible cutscene: cinematic black bars for 5 s"]}
    words = [Word(103.0, 103.4, "no"), Word(103.5, 104.0, "way."), Word(110.0, 110.5, "bro")]
    content = build_content(d, "speed", "GTA day 1", words, [ChatMessage(125, "u", "KEKW")],
                            [{"title": "he crashed lol", "views": 900, "vod_offset": 110, "duration": 30}],
                            [(120.0, b"\xff\xd8jpeg")])
    text = content[0]["text"]
    for piece in ["speed", "GTA day 1", "[3.0-4.0] no way.", "[10.0-10.5] bro", "[25] KEKW",
                  '"he crashed lol" (900 views)', "funny 70%", "black bars", "chat +4.2", "peak is at 20 s"]:
        assert piece in text, piece
    assert content[1]["text"] == "Frame at 20.0 s:"
    assert content[2]["type"] == "image" and content[2]["source"]["media_type"] == "image/jpeg"


def test_build_prompt_text_only():
    d = {"duration": 30.0, "peak": 120, "start": 108, "end": 138, "transcript": "no way",
         "breakdown": {"chat": 4.2, "audio": 1.0}, "quiet_stretch_seconds": 3}
    text = build_prompt(d, "somestreamer", "GTA 6 day 1", ["KEKW"])
    for piece in ["somestreamer", "GTA 6 day 1", "chat +4.2", "no way", "KEKW", "3 s"]:
        assert piece in text


def test_frame_times_every_two_seconds_with_peak():
    times = frame_times(100, 150, 120, 15)
    assert len(times) <= 15 and 120 in times and 122.5 in times
    assert all(100 <= t <= 150 for t in times)
    assert len(frame_times(100, 110, 104, 15)) <= 8     # short window: about one every 2 s


def test_apply_cut_snaps_to_words_and_keeps_the_peak():
    words = [Word(104.8, 105.6, "so"), Word(129.7, 130.4, "dead")]
    ai = {"setup_start": 5.0, "cut_start": 5.0, "cut_end": 30.0}           # window starts at 100
    assert apply_cut(100, ai, words, 112, CFG["clip"], 1000) == (104.8, 130.65)
    # A cut that starts after the peak is pulled back to peak - peak_guard.
    start, end = apply_cut(100, {"setup_start": 15.0, "cut_start": 15.0, "cut_end": 40.0}, [], 112,
                           CFG["clip"], 1000)
    assert start == 109 and end == 140
    # Too long, payoff early -> the END is shortened to max_seconds; the start stays.
    start, end = apply_cut(0, {"setup_start": 0.0, "cut_start": 0.0, "cut_end": 90.0}, [], 10, CFG["clip"], 1000)
    assert start == 0 and end == CFG["clip"]["max_seconds"]


def test_build_up_is_kept_when_the_payoff_comes_15_s_after_the_action():
    """Regression test (Speed's first win): the play starts at 100 s, the win (peak) is at 115 s.
    The AI's cut must not start on the win screen."""
    clip = CFG["clip"]
    words = [Word(100.2, 100.8, "go"), Word(101.0, 101.6, "push"), Word(115.5, 116.0, "WE"), Word(116.0, 116.6, "WON")]
    # 1. The AI says the setup starts at 100 but (wrongly) wants to cut at the win: setup wins.
    ai = {"setup_start": 10.0, "cut_start": 23.0, "cut_end": 40.0}       # window starts at 90
    start, end = apply_cut(90, ai, words, 115, clip, 1000, detector_start=101, max_start_delay=3)
    assert start <= 100.2 and end >= 115 + 1.5
    # 2. The AI misses the setup entirely (setup_start = cut_start = the win): the start may be at
    #    most max_start_delay later than the detector's start, so the play stays in.
    ai = {"setup_start": 24.0, "cut_start": 24.0, "cut_end": 40.0}
    start, _ = apply_cut(90, ai, words, 115, clip, 1000, detector_start=100, max_start_delay=3)
    assert start <= 103 and 115 - start >= 12
    # 3. Moving EARLIER is always allowed.
    ai = {"setup_start": 2.0, "cut_start": 2.0, "cut_end": 40.0}
    start, _ = apply_cut(90, ai, words, 115, clip, 1000, detector_start=101, max_start_delay=3)
    assert start == 92


def test_quality_uses_subscores_reaction_and_rank():
    good = validate(GOOD, 40)
    base = ai_quality(good, None, 0, CFG)
    assert base == pytest.approx((2 * 8 + 8 + 7 + 9 + 7) / 6 + 0.5)
    assert ai_quality({**good, "menu_or_map": True}, None, 0, CFG) < base
    assert ai_quality(good, 0, 10, CFG) > ai_quality(good, 9, 10, CFG)   # ranked first beats last
    assert ai_quality(None, None, 0, CFG) == CFG["judge"]["neutral_score"]


def test_final_score_weights_ai_over_signal():
    # A great clip with a modest spike beats a weak clip with a huge spike.
    assert final_score(4.0, 9.0, CFG) > final_score(16.0, 4.0, CFG)


def test_warnings_launch_mode_and_cutscene():
    low = {**validate(GOOD, 40), "spoiler_risk": "low"}
    assert "AI spoiler risk: low" in warning_reasons(low, None, CFG)   # launch_mode is on
    cfg = load_config(None)
    cfg["launch"]["launch_mode"] = False
    assert warning_reasons(low, None, cfg) == []
    cut = {**validate(GOOD, 40), "is_cutscene": True}
    assert "AI thinks the frames show a cutscene" in warning_reasons(cut, None, CFG)
    assert warning_reasons(None, None, CFG, ["pre-release GTA 6 footage"]) == ["pre-release GTA 6 footage"]


def test_quiet_stretch_needs_no_speech_and_quiet_chat():
    chat_z = np.zeros(100)
    chat_z[40:45] = 3.0
    words = [Word(10, 12, "hey"), Word(60, 61, "ok")]
    assert quiet_stretch(10, 70, words, chat_z, 0.0) == 28
    assert quiet_stretch(10, 70, None, chat_z, 0.0) is None
    assert quiet_stretch(10, 70, words, None, 0.0) is None


def test_chat_sample_spreads_and_includes_lag():
    msgs = [ChatMessage(t, "u", f"m{t}") for t in range(0, 100)]
    sample = chat_sample(msgs, 10, 20, n=5)
    assert len(sample) == 5 and sample[0].text == "m10" and sample[-1].text == "m28"


def test_timestamped_transcript_groups_sentences():
    words = [Word(1, 1.5, "hi."), Word(2, 2.5, "so"), Word(4, 4.5, "what")]
    assert timestamped_transcript(words, 0, 10) == "#1 [1.0-1.5] hi.\n#2 [2.0-2.5] so\n#3 [4.0-4.5] what"


def _results():
    def r(peak_score, start, peak):
        return {"peak_score": peak_score, "start": start, "end": start + 30.0, "peak": peak, "duration": 30.0,
                "peak_time": str(start), "breakdown": {}, "transcript": "",
                "window_start": start - 5, "window_end": start + 35}
    return [r(10.0, 10.0, 22), r(8.0, 100.0, 112), r(6.0, 200.0, 212)]


def test_judge_all_cuts_ranks_and_flags():
    client = FakeClient([reply({**GOOD, "score": 2, "standalone": 2, "payoff": 2}),
                         reply({**GOOD, "score": 9, "spoiler_risk": "high"}),
                         reply({**GOOD, "score": 9}),
                         reply({"order": [2, 1, 0]})])      # the comparative ranking
    results = _results()
    chat_z = np.ones(400)
    judge_all(results, [[], [], []], [], chat_z, CFG, "s", "t", Judge(CFG, client=client), log=lambda *_: None)
    assert len(client.calls) == 4
    # The AI's cut (5..30 s into the window) was applied.
    assert (results[1]["start"], results[1]["end"]) == (100.0, 125.0)
    # Clip 0 (weak) ends up last despite the strongest signal.
    assert results[0]["final_score"] < results[2]["final_score"]
    assert [d["warning"] for d in results] == [False, True, False]


def test_judge_all_without_judge_keeps_signal_order_and_flags_cutscenes():
    results = _results()
    chat_z = np.zeros(400)
    words = [None, [Word(100, 130, "talking the whole time")], [Word(200, 201, "hi")]]
    judge_all(results, words, [], chat_z, CFG, "s", "t", None)
    scores = [d["final_score"] for d in results]
    assert scores == sorted(scores, reverse=True)
    assert [d["warning"] for d in results] == [False, False, True]
    assert results[0]["quiet_stretch_seconds"] is None


def test_bad_key_stops_judging_instead_of_retrying_every_clip():
    response = httpx2.Response(401, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    auth_error = anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)
    client = FakeClient([auth_error, reply(GOOD), reply(GOOD)])
    results = _results()
    judge_all(results, [None] * 3, [], None, CFG, "s", "t", Judge(CFG, client=client), log=lambda *_: None)
    assert len(client.calls) == 1
    assert all(d["ai"] is None for d in results)


def test_long_hook_is_shortened_not_rejected():
    out = validate({**GOOD, "hook": "one two three four five six seven eight nine ten eleven twelve"}, 40)
    assert out["hook"] == "one two three four five six seven eight nine ten"


def test_packaging_fields_and_story_beat():
    out = validate({**GOOD, "hook": "A brand new hook", "punch_words": ["WASTED!", "run", "extra"]}, 40)
    assert out["hook"] == "A brand new hook" and out["hook_options"][0] == "A brand new hook"
    assert len(out["hook_options"]) == 3 and out["punch_words"] == ["wasted", "run"]
    assert out["caption"] == "he really thought that would work"
    story = {**validate(GOOD, 40), "story_beat": True}
    assert any("story" in r for r in warning_reasons(story, None, CFG))


def test_local_story_words_warn():
    from clipper.framecheck import story_reason
    assert "mission passed" in story_reason("yo mission passed let's go", [], CFG)
    assert story_reason("", ["SPOILERS", "spoilers!!", "SPOILER ALERT"], CFG)
    assert story_reason("we crashed the car lol", ["KEKW"], CFG) is None


def test_translations_are_matched_to_line_times():
    words = [Word(101.0, 101.5, "no"), Word(101.5, 102.0, "puede."), Word(105.0, 105.5, "mira")]
    data = {**GOOD, "translations": ["no way", "look"]}
    client = FakeClient([reply(data)])
    d = {"peak_score": 5.0, "start": 100.0, "end": 130.0, "peak": 110, "duration": 30.0, "peak_time": "x",
         "breakdown": {}, "window_start": 95.0, "window_end": 135.0, "language": "es"}
    judge_all([d], [words], [], None, CFG, "s", "t", Judge(CFG, client=client), log=lambda *_: None)
    assert d["translated_segments"] == [[101.0, 102.0, "no puede.", "no way"], [105.0, 105.5, "mira", "look"]]
    sent = client.calls[0]["messages"][0]["content"][0]["text"]
    assert "NOT English" in sent and "#1 [6.0-7.0] no puede." in sent


def test_polish_pass_rewrites_hooks_and_only_tightens_spoilers():
    first = {**GOOD, "spoiler_risk": "low"}
    better = {"hook_options": ["He's about to...", "5 stars incoming", "Chat warned him"], "hook": "He's about to...",
              "title_options": ["Speed vs the cops", "Speed's worst getaway"], "title": "Speed vs the cops",
              "context_line": "Speed tries to escape 5 stars", "punch_words": ["run"],
              "caption": "he did not think this through", "hashtags": ["ishowspeed", "gta6"],
              "spoiler_risk": "none"}                     # Sonnet says "none": must NOT clear Haiku's "low"
    client = FakeClient([reply(first), reply(better)])
    judge = Judge(CFG, client=client)
    d = _results()[0]
    judge_all([d], [[]], [], np.ones(400), CFG, "s", "t", judge, log=lambda *_: None)
    d["kept"] = True
    from clipper.judge import polish_kept
    frames = [(t, b"\xff\xd8x") for t in (5, 10, 15, 20, 25, 30)]
    assert polish_kept([d], {id(d): ([], frames)}, CFG, "speed", judge, log=lambda *_: None) == 1
    call = client.calls[1]
    assert call["model"] == "claude-sonnet-5-5" and call["fallbacks"] == "default"
    assert call["output_config"]["effort"] == "low" and "temperature" not in call
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sum(1 for c in call["messages"][0]["content"] if c["type"] == "image") <= 4
    assert d["ai"]["hook"] == "He's about to..." and d["first_hook"] == "He did NOT see that coming"
    assert d["ai"]["spoiler_risk"] == "low" and d["warning"]      # stricter verdict kept
    assert "claude-sonnet-5-5" in judge.usage or judge.usage == {}
