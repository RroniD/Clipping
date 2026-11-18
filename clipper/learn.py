"""Learn from your decisions: which signals actually predict the clips you approve?

Usage (PowerShell, from the project folder):
    python -m clipper.learn

It reads the decision log (out\\queue.db, written by the review queue) and prints:
  - how often you approve the top picks vs. the "explore" picks (is the ranking right?)
  - which AI ratings go with approvals
  - SUGGESTED new [weights] for config.toml
It NEVER changes config.toml itself: you look at the suggestion and decide (CLAUDE.md:
"show me before changing config"). Once Stage 5 logs real view counts, the same idea will
use views instead of approve/reject, which is what actually pays.

How the suggestion works (ridge regression, explained):
  Each decided clip is one example: x = the detector's signal strengths at its peak (the
  z-scores: chat, unique, hype, audio, viewer_clips, spam), y = 1 if you approved it, 0 if not.
  We fit y ~ x with a penalty that keeps the answer CLOSE TO YOUR CURRENT WEIGHTS ("ridge
  towards the current weights"), so with few decisions it barely moves, and with many it can
  move more. Then each weight is nudged at most +/-50% per run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from clipper.config import PROJECT_ROOT, load_config

DECISIONS = ("approved", "rejected")


def load_decisions(db_path: Path) -> list[dict]:
    """The LAST approve/reject per clip (you may change your mind), with its snapshot."""
    from clipper.queue_app import connect
    db = connect(db_path)
    rows = db.execute("SELECT clip_id, action, data FROM events WHERE action IN (?, ?) ORDER BY ts",
                      DECISIONS).fetchall()
    db.close()
    last: dict[int, dict] = {}
    for r in rows:
        last[r["clip_id"]] = {"approved": r["action"] == "approved", **json.loads(r["data"])}
    return list(last.values())


def suggest_weights(decisions: list[dict], cfg: dict, strength: float = 5.0) -> dict | None:
    """Ridge regression towards the current weights. Returns {signal: suggested weight}."""
    weights = cfg["weights"]
    names = [n for n in weights if any(n in d["breakdown"] for d in decisions)]
    if len(decisions) < 20 or not names:
        return None
    cap = float(cfg["scoring"]["z_cap"])
    x = np.array([[min(max(d["breakdown"].get(n, 0.0), 0.0), cap) for n in names] for d in decisions])
    y = np.array([1.0 if d["approved"] else 0.0 for d in decisions])
    # Scale the current weights to the 0..1 target, so the prior and the data speak the same units.
    w0 = np.array([float(weights[n]) for n in names])
    scale = (y.mean() / max((x @ w0).mean(), 1e-9))
    prior = w0 * scale
    lam = strength * len(decisions) / 20
    # Closed form: minimise ||y - Xw||^2 + lam ||w - prior||^2
    w = np.linalg.solve(x.T @ x + lam * np.eye(len(names)), x.T @ y + lam * prior)
    ratio = np.clip(w / np.where(prior == 0, 1e-9, prior), 0.5, 1.5)   # at most +/-50% per run
    return {n: round(float(w0[i] * ratio[i]), 2) for i, n in enumerate(names)}


def report(decisions: list[dict], cfg: dict) -> str:
    lines = [f"{len(decisions)} decided clips "
             f"({sum(d['approved'] for d in decisions)} approved, {sum(not d['approved'] for d in decisions)} rejected)"]
    if not decisions:
        return lines[0] + "\nApprove or reject clips in the review queue first (python -m clipper.queue_app)."

    def rate(group):
        return f"{100 * sum(d['approved'] for d in group) / len(group):.0f}% of {len(group)}" if group else "-"
    top = [d for d in decisions if not d.get("explore")]
    explore = [d for d in decisions if d.get("explore")]
    lines.append(f"Approval rate: top picks {rate(top)}, explore picks {rate(explore)}")
    if len(explore) >= 5 and top:
        gap = sum(d["approved"] for d in top) / len(top) - sum(d["approved"] for d in explore) / len(explore)
        lines.append("  -> the ranking works (top picks win)" if gap > 0.1 else
                     "  -> explore picks do about as well: the ranking isn't adding much yet; keep deciding, "
                     "and consider more candidates (general.top_n)")

    judged = [d for d in decisions if d.get("ai")]
    if len(judged) >= 10:
        lines.append("AI ratings, approved vs rejected (average):")
        for key in ("score", "standalone", "hook_strength", "payoff", "visual_payoff"):
            a = [d["ai"][key] for d in judged if d["approved"] and key in d["ai"]]
            r = [d["ai"][key] for d in judged if not d["approved"] and key in d["ai"]]
            if a and r:
                lines.append(f"  {key:14s} {np.mean(a):4.1f} vs {np.mean(r):4.1f}")

    suggestion = suggest_weights(decisions, cfg)
    if suggestion is None:
        lines.append(f"Weight suggestions need at least 20 decided clips (you have {len(decisions)}).")
    else:
        lines.append("Suggested [weights] for config.toml (current -> suggested). Nothing was changed:")
        for n, w in suggestion.items():
            cur = cfg["weights"][n]
            arrow = "  (same)" if abs(w - cur) < 0.05 else ""
            lines.append(f"  {n:13s} = {w:<5}   # was {cur}{arrow}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Suggest better signal weights from your queue decisions.")
    p.add_argument("--db", type=Path, default=PROJECT_ROOT / "out" / "queue.db")
    p.add_argument("--config", type=Path, help="use a different config.toml")
    args = p.parse_args()
    if not args.db.exists():
        raise SystemExit(f"No review-queue database at {args.db} yet.")
    print(report(load_decisions(args.db), load_config(args.config)))


if __name__ == "__main__":
    main()
