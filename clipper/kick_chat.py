"""Record a Kick channel's chat live, so we have chat for its VOD later.

Why this exists
  Kick has no chat replay for past VODs. If we want chat signals for a Kick VOD, we have
  to record the chat while the stream is live.

Why the public websocket and not Kick's official API (checked Nov 2026)
  Kick's official developer API does have a `chat.message.sent` event, but it is only
  delivered as a *webhook*: Kick POSTs each message to a public HTTPS URL that you
  register on your Kick developer app, and subscribing needs an OAuth token with the
  `events:subscribe` scope. A laptop has no public HTTPS URL without running a tunnel
  (ngrok / Cloudflare Tunnel), so for Stage 1 we use the same public, read-only chat
  websocket that the kick.com website itself uses (a Pusher server, no login needed).
  If this ever breaks, the official webhook route is the fallback (Stage 4 may want it anyway).

Usage (PowerShell), start it when the stream goes live and leave it running:
    python -m clipper.kick_chat <channel>
    python -m clipper.kick_chat <channel> --chatroom-id 123456 --stream-start "2026-11-17 18:00:00"

It stops by itself when the stream ends, or press Ctrl+C. Output:
    out/<channel>/kick_live_<date>/chat.jsonl   (written as messages arrive, crash-safe)
    out/<channel>/kick_live_<date>/chat.json    ([{"t", "user", "text", "ts"}], used by run.py --chat)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from clipper.config import PROJECT_ROOT

# Public key of Kick's Pusher app (the same one the kick.com web page connects with).
PUSHER_URL = ("wss://ws-us2.pusher.com/app/32cbd69e4b950bf97679"
              "?protocol=7&client=js&version=8.4.0&flash=false")
CHANNEL_API = "https://kick.com/api/v2/channels/{slug}"
# kick.com sits behind Cloudflare, which blocks requests that don't look like a browser.
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/130.0 Safari/537.36",
                   "Accept": "application/json"}


def parse_time(text: str | None) -> datetime | None:
    """Parse Kick's timestamps, which come as ISO ('2026-11-17T18:00:01+00:00') or
    plain ('2026-11-17 18:00:00', which is UTC). Returns an aware UTC datetime."""
    if not text:
        return None
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def lookup_channel(slug: str) -> dict:
    """Return {chatroom_id, channel_id, stream_start} from kick.com's channel API."""
    r = requests.get(CHANNEL_API.format(slug=slug), headers=BROWSER_HEADERS, timeout=30)
    if r.status_code == 403:
        raise RuntimeError(
            "kick.com blocked the lookup (Cloudflare). Open "
            f"{CHANNEL_API.format(slug=slug)} in your browser, copy chatroom.id and "
            "livestream.created_at, and pass them with --chatroom-id and --stream-start.")
    r.raise_for_status()
    data = r.json()
    live = data.get("livestream") or {}
    return {"chatroom_id": data["chatroom"]["id"],
            "channel_id": data.get("id"),
            "stream_start": parse_time(live.get("start_time") or live.get("created_at"))}


def message_from_event(data: dict, stream_start: datetime | None, recorder_start: datetime) -> dict:
    """Turn one Kick ChatMessageEvent payload into our chat format."""
    sent = parse_time(data.get("created_at")) or datetime.now(timezone.utc)
    zero = stream_start or recorder_start
    return {"t": round((sent - zero).total_seconds(), 2),
            "user": (data.get("sender") or {}).get("username", ""),
            "text": data.get("content", ""),
            "ts": sent.isoformat()}


def record(slug: str, out_dir: Path, chatroom_id: int | None = None,
           channel_id: int | None = None, stream_start: datetime | None = None) -> Path:
    """Connect to the chat websocket and append messages to chat.jsonl until the stream ends."""
    import websocket  # imported here so the rest of the project works without websocket-client

    if chatroom_id is None:
        info = lookup_channel(slug)
        chatroom_id, channel_id = info["chatroom_id"], info["channel_id"]
        stream_start = stream_start or info["stream_start"]
    recorder_start = datetime.now(timezone.utc)
    if stream_start is None:
        print("WARNING: stream start time unknown (is the channel live?). Times will be "
              "measured from when this recorder started, so they may not line up with the VOD.")

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "chat.jsonl"
    meta = {"channel": slug, "chatroom_id": chatroom_id, "channel_id": channel_id,
            "stream_start": stream_start.isoformat() if stream_start else None,
            "recorder_start": recorder_start.isoformat()}
    (out_dir / "chat_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    subscriptions = [f"chatrooms.{chatroom_id}.v2"]
    if channel_id:
        subscriptions.append(f"channel.{channel_id}")  # carries the "stream ended" event

    count, stream_over, backoff = 0, False, 1
    print(f"Recording chat of '{slug}' (chatroom {chatroom_id}) -> {jsonl_path}. Ctrl+C to stop.")
    with jsonl_path.open("a", encoding="utf-8") as out:
        while not stream_over:
            try:
                ws = websocket.create_connection(PUSHER_URL, timeout=60)
                for channel in subscriptions:
                    ws.send(json.dumps({"event": "pusher:subscribe",
                                        "data": {"auth": "", "channel": channel}}))
                backoff = 1
                while True:
                    raw = ws.recv()
                    if not raw:
                        break
                    msg = json.loads(raw)
                    event = msg.get("event", "")
                    if event == "pusher:ping":  # Pusher drops us if we don't answer pings
                        ws.send(json.dumps({"event": "pusher:pong", "data": {}}))
                    elif event == "App\\Events\\ChatMessageEvent":
                        # Pusher double-encodes: "data" is a JSON string inside the JSON.
                        data = json.loads(msg["data"]) if isinstance(msg["data"], str) else msg["data"]
                        out.write(json.dumps(message_from_event(data, stream_start, recorder_start),
                                             ensure_ascii=False) + "\n")
                        out.flush()
                        count += 1
                        if count % 500 == 0:
                            print(f"  {count} messages")
                    elif event == "App\\Events\\StopStreamBroadcast":
                        print("Stream ended.")
                        stream_over = True
                        break
                ws.close()
            except KeyboardInterrupt:
                print("Stopped by you.")
                break
            except Exception as exc:  # network blip: reconnect, waiting a bit longer each time
                print(f"  connection problem ({exc}); reconnecting in {backoff}s")
                try:
                    time.sleep(backoff)
                except KeyboardInterrupt:
                    break
                backoff = min(backoff * 2, 60)

    return finalize(out_dir)


def finalize(out_dir: Path) -> Path:
    """Convert chat.jsonl into the chat.json list that run.py --chat reads."""
    jsonl_path, json_path = out_dir / "chat.jsonl", out_dir / "chat.json"
    messages = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                messages.append(json.loads(line))
    messages.sort(key=lambda m: m["t"])
    json_path.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(messages)} messages to {json_path}")
    return json_path


def main() -> None:
    p = argparse.ArgumentParser(description="Record a Kick channel's live chat.")
    p.add_argument("channel", help="Kick channel name as in kick.com/<channel>")
    p.add_argument("--chatroom-id", type=int, help="skip the kick.com lookup (if Cloudflare blocks it)")
    p.add_argument("--channel-id", type=int, help="lets the recorder notice when the stream ends")
    p.add_argument("--stream-start", help='stream start in UTC, e.g. "2026-11-17 18:00:00"')
    p.add_argument("--out", type=Path, help="output folder")
    args = p.parse_args()

    slug = args.channel.lower()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out_dir = args.out or PROJECT_ROOT / "out" / slug / f"kick_live_{stamp}"
    record(slug, out_dir, args.chatroom_id, args.channel_id, parse_time(args.stream_start))


if __name__ == "__main__":
    main()
