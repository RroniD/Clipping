"""Stage 1b: an AI model watches each candidate (transcript + chat + still frames), picks the
exact cut, rates it, writes a hook, title and hashtags, and flags possible spoilers. Then it
ranks all candidates of a stream against each other. Plus a cheap local check for cutscenes.

Why an AI judge on top of the signals?
  The signals know *that* chat went crazy, not *why*. A spike can be a great moment, a
  donation sound, a raid, or (worst for us) a story cutscene. The model decides whether the
  moment works as a short clip for a stranger, and where exactly the clip should start and end.

What the model gets for each moment (more context = better judgement)
  - the transcript WITH timestamps, so it can choose cut points on sentence edges
  - chat messages with timestamps (chat reacts ~5 s after the moment)
  - the titles viewers gave their own Twitch clips of this moment (they literally say what
    happened: "he drove off the pier")
  - 6 still frames from the clip, so it can SEE what happens (Claude reads images)
  - the detector's signal strengths

What we know works on TikTok / Shorts / Reels (research, Nov 2026), built into the prompt
  - people decide in the first 1-3 s: start on the action, not on the setup
  - streamer moments do best at ~18-40 s; end right after the payoff, no outro
  - the "stranger test": it must land for someone who never saw the stream
  - models compare better than they rate in isolation, so after rating each clip we ask for
    one ranking of all of them, and blend both into the final score

How we keep it reliable and cheap
  - Structured outputs: the API must answer in our JSON shape. We still check the values and
    ask once more if something is off; after two failures the clip is kept, marked unjudged.
  - Default model is Haiku (about a cent per VOD). Change judge.model in config.toml.
  - Hard rule 2 (no spoilers): spoiler_risk "low"/"high", or a long quiet stretch (cutscene),
    gives a warning flag. Flagged clips are kept so you can look, never posted automatically.
"""

from __future__ import annotations

import base64
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPOILER_LEVELS = ("none", "low", "high")
CATEGORIES = ("funny", "fail", "skill", "rage", "shock", "wholesome", "story", "hot_take", "banter", "other")
REACTIONS = ("none", "mild", "big")
# $ per million tokens (input, output), list prices Nov 2026. Haiku's price is for prompts up
# to 100K tokens, which ours always are.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}

# JSON Schema the API must follow. "additionalProperties: false" + "required" means the
# answer has exactly these fields, no more, no fewer.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": list(CATEGORIES)},
        "standalone": {"type": "integer", "description": "1-10: makes sense to a stranger with no context"},
        "hook_strength": {"type": "integer", "description": "1-10: how hard the first 2 s of YOUR cut grab"},
        "payoff": {"type": "integer", "description": "1-10: how strong the punchline / reaction / result is"},
        "score": {"type": "integer", "description": "1-10 overall chance to go viral"},
        "setup_start": {"type": "number", "description": "seconds from the window start: when the action "
                                                          "that leads to the payoff begins"},
        "cut_start": {"type": "number", "description": "seconds from the window start"},
        "cut_end": {"type": "number", "description": "seconds from the window start"},
        "visual_payoff": {"type": "integer", "description": "1-10: how much the frames show it (vs. only talk)"},
        "facecam_reaction": {"type": "string", "enum": list(REACTIONS)},
        "is_cutscene": {"type": "boolean"},
        "layout": {"type": "string", "enum": ["split", "no_facecam", "crop"]},
        "menu_or_map": {"type": "boolean"},
        "hook_options": {"type": "array", "items": {"type": "string"}, "description": "3 different hooks"},
        "hook": {"type": "string", "description": "the best of hook_options"},
        "title_options": {"type": "array", "items": {"type": "string"}, "description": "2 different titles"},
        "punch_words": {"type": "array", "items": {"type": "string"},
                        "description": "1-2 words from the transcript to highlight in the captions"},
        "caption": {"type": "string", "description": "short post caption for TikTok (one line, no hashtags)"},
        "story_beat": {"type": "boolean", "description": "a story mission ending / major story moment"},
        "context_line": {"type": "string", "description": "max 10 words: who/what, shown as a banner"},
        "title": {"type": "string"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
        "spoiler_risk": {"type": "string", "enum": list(SPOILER_LEVELS)},
        "reason": {"type": "string"},
    },
    "required": ["category", "standalone", "hook_strength", "payoff", "score", "setup_start", "cut_start", "cut_end",
                 "visual_payoff", "facecam_reaction", "is_cutscene", "menu_or_map", "layout", "hook_options",
                 "hook", "title_options", "punch_words", "caption", "story_beat", "context_line",
                 "title", "hashtags", "spoiler_risk", "reason"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are the best short-form clip editor in the world, working for a clipper \
who gets paid per 1,000 views on TikTok, YouTube Shorts and Instagram Reels. A detector found a \
moment in a livestream (mostly GTA streams) because chat, audio or viewer clips spiked. You get a \
time WINDOW around it: the streamer's transcript with timestamps, chat with timestamps, titles of \
clips viewers made, still frames, and the signal strengths. Times are seconds from the window start.

How these platforms decide: viewers decide in the first 1-3 seconds whether to swipe away, and \
reach depends on how much of the clip people watch and rewatch. But a clip that shows only a \
reaction, without what caused it, gets swiped too. So:
- A STRANGER MUST SEE WHAT HAPPENED, NOT JUST THE REACTION TO IT. Find the payoff (the win, the \
crash, the punchline, the reaction), then find setup_start: the moment the action that LEADS to it \
begins - the final fight, the last kill, the chase, the question that gets the funny answer, the \
line that sets up the joke. Look at the frames and the transcript before the peak to find it.
- START AT OR BEFORE setup_start (1-2 s before it is ideal). Starting on the result screen or the \
celebration is WRONG: "he won" means nothing without seeing how. Only skip true dead time (silence, \
menus, waiting) that comes before the setup.
- END RIGHT AFTER THE PAYOFF and its reaction, at most ~2 s later. No outro.
- Length 25-45 s is typical; if the setup needs it, up to 60 s. Setup beats shortness.
- Cut on sentence edges: don't start or end in the middle of the streamer's sentence.
- The STRANGER TEST: would someone who never saw this stream understand it and care?

Scores (be strict; most moments are 3-6; 9-10 only if you'd bet it gets 100k+ views):
- standalone: makes sense with zero context (inside jokes, donations, raids, chat drama, \
off-screen things score low).
- hook_strength: how hard the first 2 seconds of YOUR cut grab attention.
- payoff: how strong the punchline / reaction / result is.
- score: overall chance this clip goes viral.

setup_start / cut_start / cut_end: in seconds from the window start, inside the window. \
cut_start must be at or before setup_start.

From the frames:
- visual_payoff (1-10): how much you can SEE it happen (a crash, a kill, a face), vs. only hear it.
- facecam_reaction: none / mild / big - how strong the streamer's visible reaction is.
- is_cutscene: true if frames look like a story cutscene (cinematic bars, no HUD, scripted camera).
- menu_or_map: true if it's mostly a menu, map, loading screen, or a browser/desktop with nothing \
happening.
- layout (how to make the vertical video): "split" if the streamer's WEBCAM is visible as a small box \
over the game/screen during most of the clip (we put the face on top, the game below); "crop" if \
the whole frame is the streamer's camera with no game (IRL / just chatting); "no_facecam" if it's \
fullscreen gameplay without a visible webcam, or the webcam spot shows chat or something else.

hook_options: 3 DIFFERENT hooks (at most 8 words each) of on-screen text for the first 2 seconds, \
using different angles: a question ("Why did he do THAT?"), the stakes ("5 stars and no car left"), \
a reaction ("Chat told him not to..."). Then hook = the one most likely to stop the scroll. No \
generic hooks like "insane moment", no lies, no spoilers.

punch_words: 1-2 single words that are SAID in the transcript and carry the moment ("WASTED", \
"run", "bro"). They get a different colour in the captions.

caption: a short, natural post caption for TikTok (under 100 characters, no hashtags) that makes \
people want to comment, e.g. "he really thought that would work".

title_options: 2 different titles; title = the better one.

context_line: at most 10 words that tell a stranger who and what this is, shown as a banner for \
the rest of the clip ("Speed tries to rob a bank with his own crew"). This added context is what \
makes the clip count as original, so make it specific and true.

title: a post title under 70 characters that tells a tiny story (setup + twist, e.g. "Speed \
trusted his crew... big mistake") and includes the streamer's name (helps YouTube search).

hashtags: 5-8 hashtags without '#': the streamer's name, the game, the moment type, plus 1-2 \
broad ones (e.g. gaming, fyp).

GTA streams: the clips that go viral and are SAFE to post are free-roam chaos - police chases and \
"5 stars", crashes and physics fails, weird NPC interactions, heists going wrong, the streamer's \
reaction to something absurd. Favour those. Story content is the opposite: never reward it.
story_beat: true if the clip shows or discusses a story mission ending, a main-story cutscene, a \
character's fate or a plot twist (even if the streamer is talking over it).

spoiler_risk: Rockstar takes down cutscenes, story reveals, mission endings and ending videos, \
and we never post story spoilers.
- "high": the clip likely shows or talks about a story cutscene, a plot reveal, a character's fate, \
a mission's ending or the game's ending.
- "low": it might touch the story (a story mission is mentioned, chat reacts to plot, a long \
stretch with no streamer talking, frames look cinematic) but probably isn't a reveal.
- "none": normal gameplay, chaos, banter, reactions, free roam, online modes.
When unsure between two levels, pick the higher one.

reason: one or two sentences: what happens, and why this score."""

RANK_SCHEMA = {
    "type": "object",
    "properties": {"order": {"type": "array", "items": {"type": "integer"}}},
    "required": ["order"],
    "additionalProperties": False,
}

RANK_PROMPT = """You rank candidate short-form clips from the same livestream for a clipper who is \
paid per 1,000 views on TikTok / Shorts / Reels. Put the clip most likely to get the most views \
first. What matters most: a stranger understands it instantly, the first 2 seconds grab, the payoff \
is strong, and it's the kind of moment people share or rewatch. Return every id exactly once."""


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
    frames: int
    frame_width: int

    @classmethod
    def from_cfg(cls, cfg: dict) -> "JudgeConfig":
        j = cfg["judge"]
        return cls(model=j["model"], effort=j["effort"], max_candidates=int(j["max_candidates"]),
                   max_tokens=int(j["max_tokens"]), chat_sample=int(j["chat_sample"]),
                   fallbacks=bool(j["fallbacks"]), frames=int(j["frames"]), frame_width=int(j["frame_width"]))


# --------------------------------------------------------------------------- validation

def _score(data: dict, key: str) -> int:
    v = data[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 1 <= v <= 10:
        raise JudgeOutputError(f"{key} must be 1-10, got {v!r}")
    return int(round(v))


def validate(data: dict, window_len: float | None = None) -> dict:
    """Check the model's answer and return a cleaned copy, or raise JudgeOutputError.

    A cut outside the window is clamped into it rather than rejected: the rating is still
    useful, and apply_cut() makes the cut sensible afterwards.
    """
    if not isinstance(data, dict):
        raise JudgeOutputError("answer is not a JSON object")
    missing = [k for k in JUDGE_SCHEMA["required"] if k not in data]
    if missing:
        raise JudgeOutputError(f"missing fields: {missing}")
    scores = {k: _score(data, k) for k in ("standalone", "hook_strength", "payoff", "score", "visual_payoff")}
    reaction = data["facecam_reaction"] if data["facecam_reaction"] in REACTIONS else "none"
    context = " ".join(str(data["context_line"]).split()[:12])
    if data["spoiler_risk"] not in SPOILER_LEVELS:
        raise JudgeOutputError(f"spoiler_risk must be one of {SPOILER_LEVELS}, got {data['spoiler_risk']!r}")
    category = data["category"] if data["category"] in CATEGORIES else "other"
    hook = str(data["hook"]).strip()
    if not hook:
        raise JudgeOutputError("empty hook")
    if len(hook.split()) > 10:
        # A slightly long hook is still a good hook: keep the first 10 words instead of throwing
        # away the whole judgement (that lost 3 of 40 moments in a real run).
        hook = " ".join(hook.split()[:10])
    try:
        cut_start, cut_end = float(data["cut_start"]), float(data["cut_end"])
        setup_start = float(data["setup_start"])
    except (TypeError, ValueError):
        raise JudgeOutputError(f"bad cut: {data['cut_start']!r}-{data['cut_end']!r}") from None
    if window_len is not None:
        cut_start = min(max(0.0, cut_start), window_len)
        cut_end = min(max(0.0, cut_end), window_len)
        setup_start = min(max(0.0, setup_start), window_len)
    if cut_end - cut_start < 3:
        raise JudgeOutputError(f"cut too short or backwards: {cut_start}-{cut_end}")
    def strings(key, n, max_words):
        items = data.get(key) if isinstance(data.get(key), list) else []
        out = [" ".join(str(x).split()[:max_words]) for x in items if str(x).strip()]
        return out[:n]
    hook_options = strings("hook_options", 3, 10)
    if hook not in hook_options:
        hook_options = [hook] + hook_options[:2]
    title_options = strings("title_options", 2, 20)
    title = str(data["title"]).strip()[:100]
    if title and title not in title_options:
        title_options = [title] + title_options[:1]
    punch = [w.strip(".,!?\"'").lower() for w in strings("punch_words", 2, 1)]
    tags = data["hashtags"] if isinstance(data["hashtags"], list) else []
    tags = [str(t).strip().lstrip("#").replace(" ", "") for t in tags if str(t).strip()][:8]
    return {**scores, "category": category, "hook": hook, "title": title,
            "hook_options": hook_options, "title_options": title_options, "punch_words": [p for p in punch if p],
            "caption": " ".join(str(data.get("caption", "")).split())[:150],
            "story_beat": bool(data.get("story_beat")),
            "context_line": context, "facecam_reaction": reaction,
            "is_cutscene": bool(data["is_cutscene"]), "menu_or_map": bool(data["menu_or_map"]),
            "layout": data["layout"] if data.get("layout") in ("split", "no_facecam", "crop") else None,
            "hashtags": tags, "spoiler_risk": data["spoiler_risk"], "reason": str(data["reason"]).strip(),
            "cut_start": round(cut_start, 2), "cut_end": round(cut_end, 2), "setup_start": round(setup_start, 2)}


# --------------------------------------------------------------------------- what the model sees

def chat_sample(messages, start: float, end: float, n: int) -> list:
    """Up to n chat messages from the window, spread evenly so we see the whole reaction.
    Chat lags the moment, so we look 8 s past the end."""
    inside = [m for m in messages if start <= m.t <= end + 8 and m.text.strip()]
    if len(inside) > n:
        idx = np.linspace(0, len(inside) - 1, n).round().astype(int)
        inside = [inside[i] for i in idx]
    return inside


def timestamped_transcript(words, window_start: float, window_end: float, gap: float = 0.8) -> str:
    """'[3.2-7.9] so I went into the bank' lines, one per sentence, times from the window start."""
    if not words:
        return "(no transcript)"
    lines, cur = [], []

    def flush():
        if cur:
            lines.append(f"[{cur[0].start - window_start:.1f}-{cur[-1].end - window_start:.1f}] "
                         + " ".join(w.text for w in cur))
            cur.clear()

    for w in words:
        if w.end < window_start or w.start > window_end:
            continue
        if cur and w.start - cur[-1].end > gap:
            flush()
        cur.append(w)
        if w.text.endswith((".", "!", "?")):
            flush()
    flush()
    return "\n".join(lines) or "(no speech in this window)"


def build_content(d: dict, streamer: str, vod_title: str, words, chat, viewer_clips,
                  frames: list[tuple[float, bytes]]) -> list[dict]:
    """The user message for one moment: text first, then the frames, each with its time."""
    ws, we = d["window_start"], d["window_end"]
    breakdown = ", ".join(f"{k} {v:+.1f}" for k, v in d.get("breakdown", {}).items())
    quiet = d.get("quiet_stretch_seconds")
    chat_lines = "\n".join(f"[{m.t - ws:.0f}] {m.text.strip()[:120]}" for m in chat) or "(no chat available)"
    clips = "\n".join(f"- \"{c['title']}\" ({c.get('views', 0)} views)" for c in viewer_clips) or "(none)"
    mix = d.get("chat_mood")
    mood_line = ("Chat mood during the moment: " + ", ".join(f"{k} {v}%" for k, v in mix.items()) + "\n"
                 if mix else "")
    checks = d.get("framecheck_reasons")
    check_line = f"Local frame check found: {'; '.join(checks)}\n" if checks else ""
    text = (
        f"Streamer: {streamer}\n"
        f"Stream title: {vod_title or '(unknown)'}\n"
        f"Window: {we - ws:.0f} s long. The detector's peak is at {d['peak'] - ws:.0f} s; "
        f"its rough cut was {d['start'] - ws:.1f}-{d['end'] - ws:.1f} s.\n"
        f"Signal strengths at the peak (z-scores, higher = more unusual): {breakdown}\n"
        + (f"Longest stretch with no streamer speech and quiet chat: {quiet} s\n" if quiet is not None else "")
        + mood_line + check_line
        + f"\nTranscript (timestamps in seconds from the window start):\n"
          f"{timestamped_transcript(words, ws, we)}\n"
          f"\nChat (seconds from the window start; chat reacts a few seconds late):\n{chat_lines}\n"
          f"\nClips viewers made of this moment:\n{clips}\n"
    )
    content = [{"type": "text", "text": text}]
    for t, jpeg in frames:
        content.append({"type": "text", "text": f"Frame at {t - ws:.1f} s:"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": base64.standard_b64encode(jpeg).decode("ascii")}})
    return content


def build_prompt(d: dict, streamer: str, vod_title: str, chat_lines: list[str]) -> str:
    """Text-only summary (kept for callers/tests that don't have frames or words)."""
    content = build_content({**d, "window_start": d.get("window_start", d["start"]),
                             "window_end": d.get("window_end", d["end"])}, streamer, vod_title, None,
                            [_Msg(d["start"], line) for line in chat_lines], [], [])
    return content[0]["text"].replace("(no transcript)", d.get("transcript") or "(no transcript)")


@dataclass
class _Msg:
    t: float
    text: str


def frame_times(window_start: float, window_end: float, peak: float, n: int, every: float = 2.0) -> list[float]:
    """Where to grab frames: about one every `every` seconds over the window (at most n), always
    including the peak and the reaction just after it."""
    if n <= 0:
        return []
    count = int(min(n - 2, max(1, (window_end - window_start) // every)))
    spread = list(np.linspace(window_start + 0.5, window_end - 0.5, max(1, count)))
    times = sorted(set(round(float(t), 1) for t in spread + [peak, peak + 2.5]))
    return [t for t in times if window_start <= t <= window_end][:n]


def extract_frames(video: Path, video_start: float, times: list[float], width: int) -> list[tuple[float, bytes]]:
    """JPEG stills from `video` (which starts at VOD second `video_start`) at the given VOD times.
    Small frames (768 px wide) keep the cost low: about 450 tokens each."""
    from clipper.fetch import require_tool
    ffmpeg = require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    frames = []
    for t in times:
        result = subprocess.run([ffmpeg, "-v", "error", "-ss", f"{max(0.0, t - video_start):.2f}", "-i", str(video),
                                 "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "5", "-f", "image2pipe",
                                 "-vcodec", "mjpeg", "-"], capture_output=True)
        if result.returncode == 0 and result.stdout:
            frames.append((t, result.stdout))
    return frames


# --------------------------------------------------------------------------- the API calls

class Judge:
    def __init__(self, cfg: dict, client=None):
        self.cfg = JudgeConfig.from_cfg(cfg)
        if client is None:
            import anthropic  # only needed when judging is on
            # Reads ANTHROPIC_API_KEY from the environment (.env is loaded by run.py / live.py).
            # max_retries=5: home internet blips happen (one real run lost every request to a
            # "connection error"); the SDK waits a little longer before each retry.
            client = anthropic.Anthropic(max_retries=5)
        self.client = client
        self.tokens_in = self.tokens_out = 0

    def cost(self) -> float | None:
        """Dollars spent so far (list prices, Nov 2026; None for a model not in the table)."""
        per_million = PRICES.get(self.cfg.model)
        if per_million is None:
            return None
        return self.tokens_in / 1e6 * per_million[0] + self.tokens_out / 1e6 * per_million[1]

    def _ask(self, system: str, content, schema: dict, max_tokens: int | None = None) -> dict:
        request = dict(
            model=self.cfg.model,
            max_tokens=max_tokens or self.cfg.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            # effort "low": a quick judgement; more thinking costs more without much gain.
            # "format" forces the answer to match our schema.
            output_config={"effort": self.cfg.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        # Server-side fallback (retry on another model if this one declines, e.g. a safety-filter
        # false positive on GTA violence) exists for the bigger models, not for Haiku.
        if self.cfg.fallbacks and "haiku" not in self.cfg.model:
            request.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            response = self.client.beta.messages.create(**request)
        else:
            response = self.client.messages.create(**request)
        usage = getattr(response, "usage", None)
        if usage is not None:   # add up what this run costs (printed at the end)
            self.tokens_in += int(getattr(usage, "input_tokens", 0) or 0)
            self.tokens_out += int(getattr(usage, "output_tokens", 0) or 0)

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

    def judge(self, content, window_len: float | None = None) -> dict:
        """Ask, validate, and retry once on a bad answer. Raises JudgeOutputError after two tries.

        API errors like a wrong key are NOT retried here; they bubble up so the caller can
        stop judging instead of failing the same way 20 times.
        """
        last_error = None
        for _attempt in range(2):
            try:
                return validate(self._ask(SYSTEM_PROMPT, content, JUDGE_SCHEMA), window_len)
            except JudgeOutputError as exc:
                last_error = exc
        raise JudgeOutputError(f"gave up after 2 tries: {last_error}")

    def rank(self, summaries: list[dict]) -> list[int]:
        """One call that orders all judged clips best-first. Returns their ids."""
        ids = [s["id"] for s in summaries]
        text = "\n\n".join(
            f"id {s['id']} | {s['category']} | standalone {s['standalone']}, hook {s['hook_strength']}, "
            f"payoff {s['payoff']}, score {s['score']} | {s['duration']:.0f} s\n"
            f"title: {s['title']}\nwhat happens: {s['reason']}\nspeech: {s['transcript'][:300]}"
            for s in summaries)
        data = self._ask(RANK_PROMPT, text, RANK_SCHEMA, max_tokens=2000)
        order = [i for i in data.get("order", []) if i in ids]
        order = list(dict.fromkeys(order))                 # drop repeats, keep first
        return order + [i for i in ids if i not in order]  # anything forgotten goes last


# --------------------------------------------------------------------------- applying the cut

def apply_cut(window_start: float, ai: dict, words, peak: float, clip_cfg: dict,
              vod_length: float, detector_start: float | None = None,
              max_start_delay: float = 3.0) -> tuple[float, float]:
    """The AI's cut in VOD seconds, with the BUILD-UP PROTECTED, on word edges, in the allowed length.

    Lesson from a real clip (Speed's first win): "start on the action" made the cut start on the
    win screen, without the play that won it. So:
    - The start is never later than the AI's own setup_start (where the action leading to the
      payoff begins). The AI may move the start EARLIER freely, but LATER than the detector's
      start by at most `max_start_delay` s (enough to skip a little dead time, not the setup).
    - The moment itself is always in: start <= peak - peak_guard, end >= peak + 1.5.
    - A start inside a word moves to that word's start; an end inside a word to its end (+0.25 s).
    - Too long: the END is shortened first (never before the payoff's reaction), because setup
      beats shortness. Only beyond `max_seconds_setup` is the start moved.
    """
    start = window_start + min(ai["cut_start"], ai.get("setup_start", ai["cut_start"]))
    end = window_start + ai["cut_end"]
    if detector_start is not None:
        start = min(start, detector_start + max_start_delay)
    start = min(start, peak - clip_cfg["peak_guard"])
    end = max(end, peak + 1.5)
    for w in words or []:
        if w.start < start < w.end:
            start = w.start
        if w.start < end < w.end + 0.25:
            end = w.end + 0.25
    start, end = fit_length(start, end, peak, clip_cfg, words)
    return round(max(0.0, start), 2), round(min(vod_length, end), 2)


def fit_length(start: float, end: float, peak: float, clip_cfg: dict, words=None) -> tuple[float, float]:
    """Force a cut into the allowed length WITHOUT cutting the setup.

    Too long: shorten the end, but keep the payoff and ~4 s of reaction after the peak; the clip may
    then run up to max_seconds_setup (60 s) because the setup is worth it. Only if it's still longer
    does the start move. Too short: grow mostly at the end (60%), a little before (40%).
    """
    lo, hi = clip_cfg["min_seconds"], clip_cfg["max_seconds"]
    hard = clip_cfg["max_seconds_setup"]
    if end - start > hi:
        end = max(start + hi, min(end, peak + clip_cfg["reaction_keep"]))
        inside = [w for w in words or [] if start < w.end <= end]
        if inside and end - inside[-1].end < 3:       # finish on a word, not mid-word
            end = inside[-1].end + 0.25
        if end - start > hard:
            start = end - hard
    if end - start < lo:
        missing = lo - (end - start)
        end += 0.6 * missing
        start -= 0.4 * missing
    return start, end


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


def warning_reasons(ai: dict | None, quiet_seconds: int | None, cfg: dict,
                    extra: list[str] | None = None) -> list[str]:
    """Why a clip gets the warning flag (empty list = no warning).

    extra: reasons found elsewhere (launch gate, local frame check, an AI refusal).
    In launch_mode (the weeks around a big release) "low" spoiler risk warns too.
    """
    j = cfg["judge"]
    warn_on = set(j["warn_on"]) | ({"low", "high"} if cfg["launch"]["launch_mode"] else set())
    reasons = list(extra or [])
    if ai and ai["spoiler_risk"] in warn_on:
        reasons.append(f"AI spoiler risk: {ai['spoiler_risk']}")
    if ai and ai.get("is_cutscene"):
        reasons.append("AI thinks the frames show a cutscene")
    if ai and ai.get("story_beat"):
        reasons.append("AI: looks like a story mission ending / story moment (spoiler)")
    if quiet_seconds is not None and quiet_seconds >= j["cutscene_quiet_seconds"]:
        reasons.append(f"possible cutscene: {quiet_seconds} s with no speech and quiet chat")
    return reasons


# --------------------------------------------------------------------------- scoring

def ai_quality(ai: dict | None, rank_pos: int | None, n_ranked: int, cfg: dict) -> float:
    """One 1-10 number for how good the AI thinks a clip is.

    The overall score counts double; the three sub-scores keep it honest (a "9" that a
    stranger wouldn't understand gets pulled down). The comparative ranking is blended in,
    because models are better at "A beats B" than at absolute numbers.
    """
    j = cfg["judge"]
    if not ai:
        return float(j["neutral_score"])
    core = (2 * ai["score"] + ai["standalone"] + ai["hook_strength"] + ai["payoff"] + ai["visual_payoff"]) / 6
    if ai.get("menu_or_map"):
        core -= 2          # a menu or a map on screen: people swipe away
    if ai.get("facecam_reaction") == "big":
        core += 0.5        # a big visible reaction is what makes stream clips work
    core = min(10.0, max(1.0, core))
    if rank_pos is None or n_ranked < 3:
        return core
    rank_score = 10 - 9 * rank_pos / (n_ranked - 1)       # 1st = 10, last = 1
    w = float(j["rank_weight"])
    return (1 - w) * core + w * rank_score


def final_score(peak_score: float, quality: float, cfg: dict) -> float:
    """How the final ranking combines the detector and the AI.

    signal^0.5 x AI^2 (configurable): once a moment passed the detector, how GOOD it is
    matters far more than how loud chat was. (CLAUDE.md started with signal x AI; the
    exponents make the AI's opinion count more. Set both to 1 to get the original.)
    """
    j = cfg["judge"]
    return round(max(peak_score, 0.0) ** float(j["signal_power"]) * quality ** float(j["ai_power"]), 3)


# --------------------------------------------------------------------------- whole step

def judge_all(results: list[dict], words_per_candidate: list, messages: list, chat_z,
              cfg: dict, streamer: str, vod_title: str, judge: Judge | None, log=print,
              frames_per_candidate: list | None = None, viewer_clips: list | None = None,
              vod_length: float | None = None) -> None:
    """Judge every candidate dict in place: AI rating + cut, cutscene check, warnings, final_score.

    Each dict needs start/end/peak/peak_score/duration/peak_time, and window_start/window_end
    (the stretch the AI may cut from; defaults to start/end). `judge=None` means judging is off:
    candidates get the neutral score, so the signal order is kept, but the cutscene check runs.
    """
    j = cfg["judge"]
    frames_per_candidate = frames_per_candidate or [[] for _ in results]
    vod_length = vod_length or max((d["end"] for d in results), default=0) + 3600
    for d, words in zip(results, words_per_candidate):
        d.setdefault("window_start", d["start"])
        d.setdefault("window_end", d["end"])
        d["ai"] = None

    if judge is not None:
        import anthropic
        # These mean "your key/model setting is wrong": every other clip would fail the
        # same way, so stop judging instead of trying 20 times.
        fatal = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
                 anthropic.BadRequestError)
        to_judge = list(zip(results, words_per_candidate, frames_per_candidate))[: j["max_candidates"]]
        for i, (d, words, frames) in enumerate(to_judge, 1):
            ws, we = d["window_start"], d["window_end"]
            clips = [c for c in viewer_clips or []
                     if c["vod_offset"] < we and c["vod_offset"] + c["duration"] > ws][:8]
            content = build_content(d, streamer, vod_title, words,
                                    chat_sample(messages, ws, we, judge.cfg.chat_sample), clips, frames)
            try:
                d["ai"] = judge.judge(content, we - ws)
                d["pre_judge_start"] = d["start"]
                d["start"], d["end"] = apply_cut(ws, d["ai"], words, d["peak"], cfg["clip"], vod_length,
                                                 detector_start=d["start"],
                                                 max_start_delay=float(j["max_start_delay"]))
                d["duration"] = round(d["end"] - d["start"], 2)
                log(f"    [{i}/{len(to_judge)}] {d['peak_time']}: {d['ai']['score']}/10 {d['ai']['category']}, "
                    f"{d['duration']:.0f} s, spoiler {d['ai']['spoiler_risk']}: {d['ai']['hook']}")
            except JudgeOutputError as exc:
                d["ai_error"] = str(exc)
                if "declined" in str(exc):
                    # A refusal on a GTA clip usually means violence/sensitive content the model
                    # didn't want to rate. We can't know it's safe, so a human must look.
                    d.setdefault("extra_warnings", []).append("AI declined to rate this clip")
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

    # The cut may have changed, so the cutscene check runs on the final cut.
    from clipper.framecheck import story_reason
    for d, words in zip(results, words_per_candidate):
        d["quiet_stretch_seconds"] = quiet_stretch(d["start"], d["end"], words, chat_z, j["chat_quiet_z"])
        said = " ".join(w.text for w in words or [] if w.end > d["start"] and w.start < d["end"])
        chat_lines = [m.text for m in messages if d["start"] <= m.t <= d["end"] + 10]
        story = story_reason(said, chat_lines, cfg)
        if story:
            d.setdefault("extra_warnings", []).append(story)

    # One comparative ranking of everything that was judged.
    judged = [(k, d) for k, d in enumerate(results) if d["ai"]]
    positions: dict[int, int] = {}
    if judge is not None and len(judged) >= 3 and j["rank_weight"] > 0:
        try:
            order = judge.rank([{"id": k, **d["ai"], "duration": d["end"] - d["start"],
                                 "transcript": d.get("transcript", "")} for k, d in judged])
            positions = {k: pos for pos, k in enumerate(order)}
            log(f"    ranked {len(order)} clips against each other")
        except Exception as exc:   # ranking is a bonus; the individual scores still stand
            log(f"    comparative ranking skipped ({exc})")

    for k, d in enumerate(results):
        quality = ai_quality(d["ai"], positions.get(k), len(positions), cfg)
        d["ai_quality"] = round(quality, 2)
        d["final_score"] = final_score(d["peak_score"], quality, cfg)
        d["warning_reasons"] = warning_reasons(d["ai"], d["quiet_stretch_seconds"], cfg,
                                               d.get("framecheck_reasons", []) + d.get("extra_warnings", []))
        d["warning"] = bool(d["warning_reasons"])
