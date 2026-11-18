"""Phone notifications when a clip is ready, through ntfy.sh (free, no account).

How it works: ntfy.sh is a free push service. Anyone who knows a "topic" name can send to it
and receive from it, so we use a long random topic (like a private channel name) stored in
.env as NTFY_TOPIC, never in git. You install the ntfy app on your phone and subscribe to that
topic once; from then on every finished clip buzzes your phone with its hook and AI score, and
tapping the notification opens the review queue.

Setup (once):  python -m clipper.notify --setup   (creates the topic and sends a test message)
"""

from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path

import requests

from clipper.config import PROJECT_ROOT

SERVER = "https://ntfy.sh"


def topic() -> str | None:
    return os.getenv("NTFY_TOPIC") or None


def ensure_topic(env_path: Path = PROJECT_ROOT / ".env") -> str:
    """Return NTFY_TOPIC, creating a random one in .env if there isn't one yet."""
    existing = topic()
    if not existing and env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("NTFY_TOPIC=") and line.split("=", 1)[1].strip():
                existing = line.split("=", 1)[1].strip()
    if existing:
        os.environ["NTFY_TOPIC"] = existing
        return existing
    new = "autoclipper-" + secrets.token_hex(10)   # 20 random hex characters: unguessable
    with env_path.open("a", encoding="utf-8") as f:
        f.write(f"\n# Phone notifications (ntfy app): subscribe to this topic\nNTFY_TOPIC={new}\n")
    os.environ["NTFY_TOPIC"] = new
    return new


def send(title: str, message: str, click: str | None = None, tags: list[str] | None = None,
         priority: int = 3) -> bool:
    """Push a notification. Returns False (and never raises) if it couldn't be sent:
    a missing notification must never stop clipping."""
    t = topic()
    if not t:
        return False
    body = {"topic": t, "title": title[:120], "message": message[:1000], "priority": priority,
            "tags": tags or ["movie_camera"]}
    if click:
        body["click"] = click
    try:
        # JSON publishing (not headers), so emoji and accents in hooks work.
        r = requests.post(SERVER + "/", json=body, timeout=10)
        return r.ok
    except requests.RequestException:
        return False


def clip_ready(streamer: str, d: dict, queue_url: str | None) -> bool:
    """The notification for one finished clip."""
    ai = d.get("ai") or {}
    score = f"AI {ai['score']}/10 · " if ai.get("score") else ""
    warn = "⚠ CHECK FOR SPOILERS · " if d.get("warning") else ""
    title = f"New clip: {streamer}"
    message = f"{warn}{score}{d.get('peak_time', '')}\n{ai.get('hook') or ai.get('title') or ''}"
    return send(title, message, click=queue_url, tags=["warning"] if d.get("warning") else ["movie_camera"],
                priority=4 if (ai.get("score") or 0) >= 7 else 3)


def queue_url(cfg: dict) -> str:
    from clipper.queue_app import lan_ip
    return f"http://{lan_ip()}:{cfg['queue']['port']}"


def main() -> None:
    p = argparse.ArgumentParser(description="Phone notifications for finished clips (ntfy.sh).")
    p.add_argument("--setup", action="store_true", help="create the private topic (if needed) and send a test")
    args = p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    t = ensure_topic() if args.setup else topic()
    if not t:
        raise SystemExit("No NTFY_TOPIC yet. Run: python -m clipper.notify --setup")
    print("1. Install the 'ntfy' app on your phone (App Store / Google Play).")
    print(f"2. In the app tap +, and subscribe to this topic:  {t}")
    print("   (keep it private: anyone with the name can see your notifications)")
    ok = send("AutoClipper is connected", "You'll get a message here every time a clip is ready.")
    print("3. Test message sent - check your phone." if ok else "Couldn't reach ntfy.sh (internet?).")


if __name__ == "__main__":
    main()
