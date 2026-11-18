"""clipper.learn: weight suggestions from queue decisions (never changes the config)."""

import random

from clipper import learn
from clipper.config import load_config


def fake_decisions(n=60, seed=1):
    """You approve clips with a high 'hype' z-score; 'audio' says nothing."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        hype, audio, chat = rng.uniform(0, 8), rng.uniform(0, 8), rng.uniform(0, 4)
        out.append({"approved": hype > 4, "breakdown": {"hype": hype, "audio": audio, "chat": chat},
                    "explore": i % 7 == 0, "ai": {"score": 8 if hype > 4 else 4, "standalone": 6,
                                                   "hook_strength": 6, "payoff": 6, "visual_payoff": 6}})
    return out


def test_suggestion_moves_towards_what_you_approve():
    cfg = load_config()
    s = learn.suggest_weights(fake_decisions(), cfg)
    assert s["hype"] > cfg["weights"]["hype"]          # hype predicts approvals -> up
    assert s["audio"] <= cfg["weights"]["audio"]       # audio doesn't -> not up
    for n, w in s.items():                              # never more than +/-50% per run
        assert 0.5 * cfg["weights"][n] - 0.01 <= w <= 1.5 * cfg["weights"][n] + 0.01


def test_needs_enough_decisions():
    assert learn.suggest_weights(fake_decisions(10), load_config()) is None


def test_report_text():
    text = learn.report(fake_decisions(), load_config())
    assert "Approval rate: top picks" in text and "Suggested [weights]" in text and "Nothing was changed" in text
    assert "score" in text