"""Edit streamers.toml and campaigns.toml from the review queue's page, so you never have to.

Small, careful text edits (not a full rewrite): your own comments and the layout boxes you
measured stay exactly as they are. Names are checked so nothing weird can get into the file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from clipper.config import PROJECT_ROOT

NAME = re.compile(r"^[a-z0-9_]{2,25}$")   # Twitch / Kick login names


def _toml_str(value: str) -> str:
    """A TOML string. JSON's escaping (\\" \\\\ \\n \\uXXXX) is valid TOML basic-string escaping."""
    return json.dumps(str(value), ensure_ascii=False)


def add_streamer(name: str, platform: str, path: Path | None = None) -> None:
    path = path or PROJECT_ROOT / "streamers.toml"
    if not NAME.match(name):
        raise ValueError("Use the channel name exactly as in the link (letters, numbers, _).")
    if platform not in ("twitch", "kick"):
        raise ValueError("Platform must be twitch or kick.")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if re.search(rf"^\[streamers\.{name}\]", text, re.M):
        raise ValueError(f"{name} is already on your list.")
    block = (f"\n[streamers.{name}]\nplatform = {_toml_str(platform)}\n"
             f"note = \"added from the review queue (you said they allow clipping)\"\n")
    path.write_text(text.rstrip("\n") + "\n" + block, encoding="utf-8")


def remove_streamer(name: str, path: Path | None = None) -> None:
    path = path or PROJECT_ROOT / "streamers.toml"
    if not NAME.match(name) or not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    # The streamer's block: from its [streamers.name] line up to the next [section] or the end.
    text = re.sub(rf"\n?^\[streamers\.{name}\]\n(?:(?!^\[).*\n?)*", "\n", text, flags=re.M)
    path.write_text(text.rstrip("\n") + "\n", encoding="utf-8")


def add_campaign(fields: dict, path: Path | None = None) -> str:
    """Write one [campaigns.<key>] block from the page's form. Returns the key."""
    path = path or PROJECT_ROOT / "campaigns.toml"
    name = str(fields.get("name", "")).strip()
    if not name:
        raise ValueError("Give the campaign a name.")
    key = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")[:30] or "campaign"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if re.search(rf"^\[campaigns\.{key}\]", text, re.M):
        raise ValueError("A campaign with that name already exists.")
    tags = [t.strip() for t in re.split(r"[,\s]+", str(fields.get("required_tags", ""))) if t.strip()]
    platforms = [p for p in ("tiktok", "youtube", "instagram") if fields.get(p)] or ["tiktok"]
    streamer = str(fields.get("streamer", "")).strip().lower().lstrip("@")

    def number(k, default):
        try:
            return float(fields.get(k) or default)
        except ValueError:
            raise ValueError(f"{k.replace('_', ' ')} must be a number.") from None

    lines = [f"\n[campaigns.{key}]", f"name = {_toml_str(name)}"]
    if streamer:
        if not NAME.match(streamer):
            raise ValueError("Streamer: use the channel name exactly as in the link.")
        lines.append(f"streamer = {_toml_str(streamer)}")
    lines += [f"disclosure = {_toml_str(fields.get('disclosure', ''))}",
              "disclosure_seconds = 3",
              f"credit = {_toml_str(fields.get('credit', ''))}",
              f"required_tags = [{', '.join(_toml_str(t) for t in tags)}]",
              f"platforms = [{', '.join(_toml_str(p) for p in platforms)}]",
              f"max_seconds = {number('max_seconds', 60):g}",
              f"rate_per_1000 = {number('rate_per_1000', 0):g}",
              f"rules_url = {_toml_str(fields.get('rules_url', ''))}"]
    path.write_text(text.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return key
