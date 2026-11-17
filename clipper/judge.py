"""Stage 1b: an AI model looks at each candidate and rates it, writes a hook and title,
and flags possible spoilers. Plus a cheap local check for likely cutscenes.

Why an AI judge on top of the signals?
  The signals know *that* chat went crazy, not *why*. A spike can be a great moment, a
  donation sound, a raid, or (worst for us) a story cutscene. The model reads what was
  said and decides whether it works as a short clip, and whether it might be a spoiler.

How we keep it reliable and cheap
  - Only the top `max_candidates` (~20) per VOD are judged, one short request each.
  - The API is told to answer in a fixed JSON shape ("structured outputs"), so the answer
    is always parseable. We still check the values (score 1-10, etc.) ourselves and ask
    once more if something is off. If it fails twice, the clip is kept, marked unjudged.
  - Hard rule 2 (no spoilers): anything rated spoiler_risk "low" or "high", or that looks
    like a cutscene, gets a warning flag. Flagged clips are still saved so you can look,
    but nothing in this project ever posts automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

SPOILER_LEVELS = ("none", "low", "high")

# JSON Schema the API must follow. "additionalProperties: false" + "required" means the
# answer has exactly these fields, no more, no fewer.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "1 (useless) to 10 (viral)"},
        "hook": {"type": "string", "description": "on-screen text for the first 2 s, max 8 words"},
        "title": {"type": "string"},
        "spoiler_risk": {"type": "string", "enum": list(SPOILER_LEVELS)},
        "reason": {"type": "string"},
        "suggested_trim": {
            "type": "object",
            "properties": {"start_offset": {"type": "number"}, "end_offset": {"type": "number"}},
            "required": ["start_offset", "end_offset"],
            "additionalProperties": False,
        },
    },
    "required": ["score", "hook", "title", "spoiler_risk", "reason", "suggested_trim"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You review candidate short-form clips (TikTok / YouTube Shorts / Reels, 15-60 s) \
that were automatically cut from a livestream VOD, mostly GTA 6 streams. A detector picked each \
moment because chat, audio or viewer clips spiked. You get the streamer's speech transcript, a few \
chat messages from that moment, and the signal strengths. Decide how well it works as a standalone \
short clip for viewers who have never seen this stream.

score (1-10): 8-10 = clear, funny, shocking or skillful moment that makes sense on its own and \
would hold attention. 5-7 = decent but needs context or is slow. 1-4 = nothing happens, \
it's chat reacting to something off-screen (a donation, a raid, an ad), or it makes no sense alone.

hook: at most 8 words of on-screen text for the first 2 seconds that makes people keep watching. \
No clickbait lies, no spoilers.

title: a short post title (under 70 characters).

spoiler_risk: Rockstar takes down cutscenes, story reveals, mission endings and ending videos, \
and we never post story spoilers.
- "high": the clip likely shows or talks about a story cutscene, a plot reveal, a character's fate, \
a mission's ending or the game's ending.
- "low": it might touch the story (a story mission is mentioned, chat reacts to plot, long stretch \
with no streamer talking) but probably isn't a reveal.
- "none": normal gameplay, chaos, banter, reactions, free roam, online modes.
When unsure between two levels, pick the higher one.

reason: one or two sentences on why you gave this score and spoiler level.

suggested_trim: seconds to move the clip's start and end. start_offset > 0 starts later \
(cut a slow intro), < 0 starts earlier. end_offset < 0 ends earlier (cut dead air), > 0 ends later. \
Use 0 and 0 when the cut is fine. Keep each within -10..10."""


class JudgeOutputError(ValueError):
    """The model's answer was missing, refused, or had values outside the allowed ranges."""


@dataclass
class JudgeConfig:
    model: str
    effort: str
    max_candidates: int
    max_tokens: int
    chat_sample: int
    fallbacks: bool

    @classmethod
    def from_cfg(cls, cfg: dict) -> "JudgeConfig":
        j = cfg["judge"]
        return cls(model=j["model"], effort=j["effort"], max_candidates=int(j["max_candidates"]),
                   max_tokens=int(j["max_tokens"]), chat_sample=int(j["chat_sample"]),
                   fallbacks=bool(j["fallbacks"]))


# --------------------------------------------------------------------------- validation

def validate(data: dict) -> dict:
    """Check the model's answer and return a cleaned copy, or raise JudgeOutputError."""
    if not isinstance(data, dict):
        raise JudgeOutputError("answer is not a JSON object")
    missing = [k for k in JUDGE_SCHEMA["required"] if k not in data]
    if missing:
        raise JudgeOutputError(f"missing fields: {missing}")

    score = data["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= score <= 10:
        raise JudgeOutputError(f"score must be 1-10, got {score!r}")
    if data["spoiler_risk"] not in SPOILER_LEVELS:
        raise JudgeOutputError(f"spoiler_risk must be one of {SPOILER_LEVELS}, got {data['spoiler_risk']!r}")
    hook = str(data["hook"]).strip()
    if not hook or len(hook.split()) > 8:
        raise JudgeOutputError(f"hook must be 1-8 words, got {hook!r}")
    trim = data["suggested_trim"]
    if not isinstance(trim, dict) or not all(isinstance(trim.get(k), (int, float))
                                             for k in ("start_offset", "end_offset")):
        raise JudgeOutputError(f"bad suggested_trim: {trim!r}")

    clamp = lambda v: round(max(-10.0, min(10.0, float(v))), 1)  # noqa: E731
    return {"score": int(round(score)), "hook": hook, "title": str(data["title"]).strip(),
            "spoiler_risk": data["spoiler_risk"], "reason": str(data["reason"]).strip(),
            "suggested_trim": {"start_offset": clamp(trim["start_offset"]),
                               "end_offset": clamp(trim["end_offset"])}}


# --------------------------------------------------------------------------- the prompt

def build_prompt(candidate: dict, streamer: str, vod_title: str, chat_lines: list[str]) -> str:
    """Everything the model needs about one candidate, as plain text."""
    breakdown = ", ".join(f"{k} {v:+.1f}" for k, v in candidate.get("breakdown", {}).items())
    chat = "\n".join(f"- {line}" for line in chat_lines) or "(no chat available)"
    quiet = candidate.get("quiet_stretch_seconds")
    quiet_line = (f"Longest stretch with no streamer speech and quiet chat: {quiet} s\n"
                  if quiet is not None else "")
    return (
        f"Streamer: {streamer}\n"
        f"Stream title: {vod_title or '(unknown)'}\n"
        f"Clip length: {candidate['duration']:.0f} s (peak {candidate['peak'] - candidate['start']:.0f} s in)\n"
        f"Signal strengths at the peak (z-scores, higher = more unusual): {breakdown}\n"
        f"{quiet_line}"
        f"\nStreamer transcript of the clip:\n{candidate.get('transcript') or '(no transcript)'}\n"
        f"\nChat during the clip (sample):\n{chat}\n"
    )


def chat_sample(messages, start: float, end: float, n: int) -> list[str]:
    """Up to n chat messages from the clip window, spread evenly so we see the whole reaction."""
    inside = [m for m in messages if start <= m.t <= end + 8 and m.text.strip()]  # +8 s: chat lags
    if len(inside) > n:
        idx = np.linspace(0, len(inside) - 1, n).round().astype(int)
        inside = [inside[i] for i in idx]
    return [m.text.strip()[:120] for m in inside]


# --------------------------------------------------------------------------- the API call

class Judge:
    def __init__(self, cfg: dict, client=None):
        self.cfg = JudgeConfig.from_cfg(cfg)
        if client is None:
            import anthropic  # only needed when judging is on
            # Reads ANTHROPIC_API_KEY from the environment (.env is loaded by run.py).
            client = anthropic.Anthropic()
        self.client = client

    def _ask(self, prompt: str) -> dict:
        request = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            # effort "low": this is a quick rating task, more thinking costs more without
            # much gain. "format" forces the answer to match JUDGE_SCHEMA.
            output_config={"effort": self.cfg.effort,
                           "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        )
        if self.cfg.fallbacks:
            # If the model declines (safety filter false positive on, say, GTA violence),
            # the API retries the same request on a fallback model inside the same call.
            request.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            response = self.client.beta.messages.create(**request)
        else:
            response = self.client.messages.create(**request)

        if response.stop_reason == "refusal":
            raise JudgeOutputError("the model declined to rate this clip")
        if response.stop_reason == "max_tokens":
            raise JudgeOutputError("answer was cut off (raise judge.max_tokens)")
        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), None)
        if text is None:
            raise JudgeOutputError("no text in the answer")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise JudgeOutputError(f"answer is not valid JSON: {exc}") from exc

    def judge(self, prompt: str) -> dict:
        """Ask, validate, and retry once on a bad answer. Raises JudgeOutputError after two tries.

        API errors like a wrong key are NOT retried here; they bubble up so run.py can
        stop judging instead of failing the same way 15 times.
        """
        last_error = None
        for _attempt in range(2):
            try:
                return validate(self._ask(prompt))
            except JudgeOutputError as exc:
                last_error = exc
        raise JudgeOutputError(f"gave up after 2 tries: {last_error}")


# --------------------------------------------------------------------------- local cutscene check

def quiet_stretch(start: float, end: float, words, chat_z: np.ndarray | None,
                  chat_quiet_z: float) -> int | None:
    """Longest run of seconds in [start, end) with no streamer speech AND quiet chat.

    Cutscenes are where the streamer stops talking to watch and chat goes calm (or turns
    into "SPOILERS" spam, which the AI catches instead). Returns None when we can't tell,
    i.e. we have no transcript or no chat for this VOD.
    """
    if words is None or chat_z is None:
        return None
    s, e = int(start), int(np.ceil(end))
    speaking = np.zeros(e - s, dtype=bool)
    for w in words:
        a, b = max(s, int(w.start)), min(e, int(np.ceil(w.end)))
        if b > a:
            speaking[a - s: b - s] = True
    chat_quiet = chat_z[s:e] <= chat_quiet_z
    quiet = ~speaking[: len(chat_quiet)] & chat_quiet
    longest = run = 0
    for q in quiet:
        run = run + 1 if q else 0
        longest = max(longest, run)
    return int(longest)


def warning_reasons(ai: dict | None, quiet_seconds: int | None, cfg: dict) -> list[str]:
    """Why a clip gets the warning flag (empty list = no warning)."""
    j = cfg["judge"]
    reasons = []
    if ai and ai["spoiler_risk"] in j["warn_on"]:
        reasons.append(f"AI spoiler risk: {ai['spoiler_risk']}")
    if quiet_seconds is not None and quiet_seconds >= j["cutscene_quiet_seconds"]:
        reasons.append(f"possible cutscene: {quiet_seconds} s with no speech and quiet chat")
    return reasons


# --------------------------------------------------------------------------- whole step

def judge_all(results: list[dict], words_per_candidate: list, messages: list, chat_z,
              cfg: dict, streamer: str, vod_title: str, judge: Judge | None, log=print) -> None:
    """Add AI ratings, the cutscene check, warnings and final_score to every candidate dict.

    `judge=None` means judging is off: candidates get the neutral score, so the signal
    order is kept, but the local cutscene check still runs.
    Final ranking = signal score x AI score (CLAUDE.md Stage 1b).
    """
    j = cfg["judge"]
    for d, words in zip(results, words_per_candidate):
        d["quiet_stretch_seconds"] = quiet_stretch(d["start"], d["end"], words, chat_z, j["chat_quiet_z"])
        d["ai"] = None

    if judge is not None:
        import anthropic
        # These mean "your key/model setting is wrong": every other clip would fail the
        # same way, so stop judging instead of trying 20 times.
        fatal = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
                 anthropic.BadRequestError)
        to_judge = results[: j["max_candidates"]]
        for i, d in enumerate(to_judge, 1):
            prompt = build_prompt(d, streamer, vod_title,
                                  chat_sample(messages, d["start"], d["end"], judge.cfg.chat_sample))
            try:
                d["ai"] = judge.judge(prompt)
                log(f"    [{i}/{len(to_judge)}] {d['peak_time']}: {d['ai']['score']}/10, "
                    f"spoiler {d['ai']['spoiler_risk']}: {d['ai']['hook']}")
            except JudgeOutputError as exc:
                d["ai_error"] = str(exc)
                log(f"    [{i}/{len(to_judge)}] {d['peak_time']}: not judged ({exc})")
            except fatal as exc:
                log(f"    AI judge stopped: {exc}. Check ANTHROPIC_API_KEY in .env and judge.model "
                    "in config.toml. The remaining clips keep their signal order.")
                d["ai_error"] = str(exc)
                break
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
                # The SDK already retried these (rate limits, server errors, network) twice.
                d["ai_error"] = str(exc)
                log(f"    [{i}/{len(to_judge)}] {d['peak_time']}: API error, skipped ({exc})")

    for d in results:
        ai_score = d["ai"]["score"] if d["ai"] else j["neutral_score"]
        d["final_score"] = round(d["peak_score"] * ai_score, 3)
        d["warning_reasons"] = warning_reasons(d["ai"], d["quiet_stretch_seconds"], cfg)
        d["warning"] = bool(d["warning_reasons"])
