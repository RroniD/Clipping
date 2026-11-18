"""Stage 3: the review queue. A small web app you open on your phone to approve or reject clips.

Usage (PowerShell, from the project folder):
    python -m clipper.queue_app

It imports every finished clip in out\\<streamer>\\<vod_id>\\final\\ into a SQLite database
(out\\queue.db), then serves one page where you can watch each clip, edit its hook and title,
trim it by 2 s at either end or switch its layout (both re-render it), and Approve / Reject it.

Hard rules (CLAUDE.md):
- Approve ONLY marks the clip approved in the database. Nothing is posted anywhere. Posting is Stage 5.
- Clips with a warning (possible spoiler / cutscene) stay marked ⚠, and approving one needs an
  extra confirmation. Nothing is ever approved automatically.

Why Flask? It's one extra package (FastAPI needs two: fastapi + uvicorn), it can stream video
files with HTTP Range requests (phones need those to play and seek video), and it's the
simplest web framework to read.

Why SQLite? It's built into Python (no server to install), it's one file, and it keeps your
decisions safe when you re-import: importing only ADDS clips, it never resets a status.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import secrets
import socket
import sqlite3
import threading
import time
from pathlib import Path

from clipper.config import PROJECT_ROOT, load_campaigns, load_config

OUT_ROOT = PROJECT_ROOT / "out"
STATUSES = ("pending", "approved", "rejected")
LAYOUTS = ("split", "no_facecam", "crop")

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    id              INTEGER PRIMARY KEY,
    streamer        TEXT NOT NULL,
    vod_id          TEXT NOT NULL,
    rank            INTEGER NOT NULL,
    peak_time       TEXT,
    video           TEXT NOT NULL,          -- path of the finished .mp4, relative to out/
    status          TEXT NOT NULL DEFAULT 'pending',
    warning         INTEGER NOT NULL DEFAULT 0,
    warning_reasons TEXT NOT NULL DEFAULT '[]',   -- JSON list
    ai_score        REAL,
    ai_reason       TEXT NOT NULL DEFAULT '',
    hook            TEXT NOT NULL DEFAULT '',
    title           TEXT NOT NULL DEFAULT '',
    transcript      TEXT NOT NULL DEFAULT '',
    final_score     REAL,
    start           REAL NOT NULL,          -- VOD seconds of the current cut
    "end"           REAL NOT NULL,
    layout          TEXT NOT NULL DEFAULT 'split',
    rendering       INTEGER NOT NULL DEFAULT 0,
    render_error    TEXT NOT NULL DEFAULT '',
    version         INTEGER NOT NULL DEFAULT 0,   -- bumped on every re-render (cache-busting)
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    decided_at      REAL
);
CREATE INDEX IF NOT EXISTS clips_by_vod ON clips (streamer, vod_id, rank);

-- Every decision you make, with what the tool knew at that moment (signals + AI ratings).
-- python -m clipper.learn reads this to see which signals predict the clips you approve.
CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY,
    clip_id  INTEGER NOT NULL,
    ts       REAL NOT NULL,
    action   TEXT NOT NULL,           -- approved / rejected / pending / render / title / campaign / post
    data     TEXT NOT NULL DEFAULT '{}'
);

-- Where each clip was posted. A post URL can belong to one campaign only: submitting the
-- same post to two campaigns gets clips rejected (and can get you banned from both).
CREATE TABLE IF NOT EXISTS posts (
    id        INTEGER PRIMARY KEY,
    clip_id   INTEGER NOT NULL,
    campaign  TEXT NOT NULL DEFAULT '',
    platform  TEXT NOT NULL DEFAULT '',
    url       TEXT NOT NULL UNIQUE,
    posted_at REAL NOT NULL
);
"""

# Columns added after Stage 3 shipped. connect() adds them to an existing queue.db, so your
# old decisions are kept (SQLite can add columns in place).
EXTRA_COLUMNS = {
    "category": "TEXT NOT NULL DEFAULT ''",
    "hashtags": "TEXT NOT NULL DEFAULT '[]'",
    "context_line": "TEXT NOT NULL DEFAULT ''",
    "campaign": "TEXT NOT NULL DEFAULT ''",
    "explore": "INTEGER NOT NULL DEFAULT 0",
    "peak_score": "REAL",
    "ai_json": "TEXT NOT NULL DEFAULT '{}'",
    "breakdown_json": "TEXT NOT NULL DEFAULT '{}'",
    "layout_reason": "TEXT NOT NULL DEFAULT ''",
}


# --------------------------------------------------------------------------- database

def connect(db_path: Path) -> sqlite3.Connection:
    """Open the database (creating it if needed). Rows come back as dict-like objects."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: the web server and the render worker are different threads.
    # Every write goes through `with db:` blocks under one lock, so that is safe here.
    db = sqlite3.connect(str(db_path), check_same_thread=False)
    db.row_factory = sqlite3.Row
    old = db.execute("SELECT sql FROM sqlite_master WHERE name='clips'").fetchone()
    if old and "UNIQUE (streamer, vod_id, rank)" in old["sql"]:
        # Older queue.db: it allowed only one clip per rank number, so a re-processed VOD couldn't
        # add a new moment with a rank that's already used. SQLite can't drop a rule in place, so
        # the table is copied into one without it (all rows, ids and decisions kept).
        db.executescript("ALTER TABLE clips RENAME TO clips_old;")
        db.executescript(SCHEMA)
        cols = [r["name"] for r in db.execute("PRAGMA table_info(clips_old)")]
        have_new = {r["name"] for r in db.execute("PRAGMA table_info(clips)")}
        for name, kind in EXTRA_COLUMNS.items():
            if name not in have_new:
                db.execute(f"ALTER TABLE clips ADD COLUMN {name} {kind}")
        quoted = ", ".join(f'"{c}"' for c in cols)
        db.executescript(f"INSERT INTO clips ({quoted}) SELECT {quoted} FROM clips_old; DROP TABLE clips_old;")
    db.executescript(SCHEMA)
    have = {r["name"] for r in db.execute("PRAGMA table_info(clips)")}
    for name, kind in EXTRA_COLUMNS.items():
        if name not in have:
            db.execute(f"ALTER TABLE clips ADD COLUMN {name} {kind}")
    db.commit()
    return db


def log_event(db: sqlite3.Connection, clip_id: int, action: str, extra: dict | None = None) -> None:
    """Record a decision together with everything the tool knew about the clip."""
    row = db.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
    snapshot = {"breakdown": json.loads(row["breakdown_json"]), "ai": json.loads(row["ai_json"]),
                "peak_score": row["peak_score"], "final_score": row["final_score"],
                "explore": bool(row["explore"]), "rank": row["rank"], "duration": row["end"] - row["start"],
                "warning": bool(row["warning"]), **(extra or {})}
    db.execute("INSERT INTO events (clip_id, ts, action, data) VALUES (?,?,?,?)",
               (clip_id, time.time(), action, json.dumps(snapshot)))


def _rank_of(path: Path) -> int | None:
    m = re.match(r"(\d+)_", path.name)
    return int(m.group(1)) if m else None


def _clip_fields(d: dict, used: dict, video: str) -> dict:
    """The queue columns that come from candidates.json + the render's sidecar file."""
    ai = d.get("ai") or {}
    reasons = list(d.get("warning_reasons", [])) + list(used.get("warnings", []))
    return {"peak_time": d.get("peak_time"), "video": video, "warning": int(bool(reasons)),
            "warning_reasons": json.dumps(reasons), "ai_score": ai.get("score"), "ai_reason": ai.get("reason", ""),
            "hook": used.get("hook", ai.get("hook", "")), "title": ai.get("title", ""),
            "transcript": d.get("transcript", ""), "final_score": d.get("final_score", d.get("peak_score")),
            "start": used.get("start", d["start"]), "end": used.get("end", d["end"]),
            "layout": used.get("layout", d.get("layout", "split")), "category": ai.get("category", ""),
            "hashtags": json.dumps(ai.get("hashtags", [])), "context_line": used.get("banner", ai.get("context_line", "")),
            "campaign": used.get("campaign") or "", "explore": int(bool(d.get("explore"))),
            "peak_score": d.get("peak_score"), "ai_json": json.dumps(ai),
            "breakdown_json": json.dumps(d.get("breakdown", {})),
            "layout_reason": used.get("layout_reason", d.get("layout_reason", ""))}


def import_clips(db: sqlite3.Connection, out_root: Path = OUT_ROOT) -> int:
    """Add every finished clip under out_root to the database. Returns how many were new.

    A clip is identified by its FILE and its MOMENT (peak time), not by its rank number,
    because re-processing a VOD reshuffles the ranks. For each finished .mp4:
      1. a row already shows this file                              -> nothing to do
      2. a row has the same moment but its file is gone (re-render) -> follow it to the new file
      3. a row has the same rank, a different moment, and you never
         touched it (still pending, no decisions or edits)          -> reuse that row for the new moment
      4. otherwise                                                  -> a NEW row
    Your approvals, rejections and edits are never moved to a different moment.
    """
    added = 0
    now = time.time()
    for cand_path in sorted(out_root.glob("*/*/candidates.json")):
        vod_dir = cand_path.parent
        final_dir = vod_dir / "final"
        if not final_dir.is_dir():
            continue
        data = json.loads(cand_path.read_text(encoding="utf-8"))
        by_rank = {d["rank"]: d for d in data["candidates"]}
        key = (data["streamer"], data["vod_id"])
        for mp4 in sorted(final_dir.glob("*.mp4")):
            rank = _rank_of(mp4)
            d = by_rank.get(rank)
            if d is None:
                continue
            # What the clip was actually rendered with (written by clipper.edit). Older renders
            # don't have this file, so fall back to the candidate's own values.
            sidecar = final_dir / "work" / f"{mp4.stem}.json"
            used = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
            video = mp4.relative_to(out_root).as_posix()
            fields = _clip_fields(d, used, video)
            with db:
                if db.execute("SELECT 1 FROM clips WHERE video=?", (video,)).fetchone():
                    continue                                                          # 1.
                same = db.execute("SELECT id, video FROM clips WHERE streamer=? AND vod_id=? AND rank=? "
                                  "AND peak_time=?", (*key, rank, d.get("peak_time"))).fetchall()
                gone = [r for r in same if not (out_root / r["video"]).exists()]
                if gone:                                                              # 2.
                    db.execute("UPDATE clips SET video=?, updated_at=? WHERE id=?", (video, now, gone[0]["id"]))
                    continue
                if same:
                    continue   # this moment is already in the queue with a working file
                reusable = [r for r in db.execute(
                    "SELECT id, video FROM clips WHERE streamer=? AND vod_id=? AND rank=? AND status='pending' "
                    "AND version=0 AND id NOT IN (SELECT clip_id FROM events)", (*key, rank)).fetchall()
                    if not (out_root / r["video"]).exists()]
                if reusable:                                                          # 3.
                    sets = ", ".join(f'"{k}"=?' for k in fields)
                    db.execute(f"UPDATE clips SET {sets}, render_error='', updated_at=? WHERE id=?",
                               (*fields.values(), now, reusable[0]["id"]))
                    continue
                cols = ["streamer", "vod_id", "rank", *fields, "created_at", "updated_at"]  # 4.
                db.execute(f"INSERT INTO clips ({', '.join(chr(34) + c + chr(34) for c in cols)}) "
                           f"VALUES ({', '.join('?' * len(cols))})", (*key, rank, *fields.values(), now, now))
                added += 1
    return added


def clip_dict(row: sqlite3.Row, posts: list | None = None, submit_minutes: float = 30) -> dict:
    d = dict(row)
    d["warning"] = bool(d["warning"])
    d["warning_reasons"] = json.loads(d["warning_reasons"])
    d["rendering"] = bool(d["rendering"])
    d["explore"] = bool(d.get("explore"))
    d["duration"] = round(d["end"] - d["start"], 2)
    d["hashtags"] = json.loads(d.get("hashtags") or "[]")
    d["ai"] = json.loads(d.pop("ai_json", None) or "{}")
    d.pop("breakdown_json", None)
    # Campaigns usually reward clips posted fast; the page counts down from approval.
    d["submit_deadline"] = (d["decided_at"] + 60 * submit_minutes
                            if d["status"] == "approved" and d["decided_at"] else None)
    d["posts"] = [dict(p) for p in posts or []]
    return d


# --------------------------------------------------------------------------- re-rendering

class RenderWorker:
    """Re-renders clips one at a time in a background thread.

    Why a thread + queue? A render takes 10-60 s. The web page must not freeze meanwhile, and
    two renders at once would fight over the GPU (Whisper) and the CPU (x264). One worker,
    one job at a time, keeps it simple and predictable.
    """

    def __init__(self, db: sqlite3.Connection, lock: threading.Lock, cfg: dict, out_root: Path):
        self.db, self.lock, self.cfg, self.out_root = db, lock, cfg, out_root
        self.jobs: queue.Queue = queue.Queue()
        self.renderer = None
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, clip_id: int, start: float, end: float, layout: str, hook: str,
               banner: str, campaign: str) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE clips SET rendering=1, render_error='' WHERE id=?", (clip_id,))
        self.jobs.put((clip_id, start, end, layout, hook, banner, campaign))

    def _loop(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                self._render(*job)
            except Exception as exc:   # show the error on the page instead of killing the worker
                with self.lock, self.db:
                    self.db.execute("UPDATE clips SET rendering=0, render_error=? WHERE id=?",
                                    (str(exc)[-500:], job[0]))

    def _render(self, clip_id: int, start: float, end: float, layout: str, hook: str,
                banner: str = "", campaign: str = "") -> None:
        from clipper.edit import ClipRenderer, clip_name   # heavy imports only when needed
        if self.renderer is None:
            self.renderer = ClipRenderer(self.cfg)
        with self.lock:
            row = self.db.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
        vod_dir = self.out_root / row["streamer"] / row["vod_id"]
        data = json.loads((vod_dir / "candidates.json").read_text(encoding="utf-8"))
        d = next(c for c in data["candidates"] if c["rank"] == row["rank"])
        version = row["version"] + 1
        # A NEW file name for every version: on Windows a file can't be replaced while the
        # phone is still streaming it, and a new name also stops the browser showing a cached copy.
        final = self.renderer.render(vod_dir, data, d, start=start, end=end, layout_mode=layout,
                                     hook=hook, banner=banner, campaign=campaign or None,
                                     out_name=f"{clip_name(d)}_v{version}")
        old = self.out_root / row["video"]
        with self.lock, self.db:
            self.db.execute(
                """UPDATE clips SET video=?, start=?, "end"=?, layout=?, hook=?, context_line=?, campaign=?,
                   layout_reason=?, version=?, rendering=0, render_error='', updated_at=? WHERE id=?""",
                (final.relative_to(self.out_root).as_posix(), start, end, layout, hook, banner, campaign,
                 "picked by hand" if layout != row["layout"] else row["layout_reason"],
                 version, time.time(), clip_id))
        for path in (old, old.with_suffix(".jpg")):
            if path != final:
                try:
                    path.unlink()   # best effort: it may still be open on the phone
                except OSError:
                    pass


# --------------------------------------------------------------------------- web app

def create_app(db_path: Path | None = None, out_root: Path = OUT_ROOT, cfg: dict | None = None,
               start_worker: bool = True):
    from flask import Flask, abort, jsonify, request, send_file

    cfg = cfg or load_config()
    q = cfg["queue"]
    db = connect(db_path or out_root / "queue.db")
    lock = threading.Lock()   # one writer at a time (web requests + render worker)
    with lock:
        added = import_clips(db, out_root)
    print(f"Review queue: {added} new clip(s) imported")
    worker = RenderWorker(db, lock, cfg, out_root) if start_worker else None

    app = Flask(__name__)

    def get_clip(clip_id: int) -> sqlite3.Row:
        with lock:
            row = db.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
        if row is None:
            abort(404)
        return row

    @app.get("/")
    def page():
        return PAGE.replace("__TRIM_STEP__", str(q["trim_step"]))

    def full(clip_id: int) -> dict:
        row = get_clip(clip_id)
        with lock:
            posts = db.execute("SELECT * FROM posts WHERE clip_id=? ORDER BY posted_at", (clip_id,)).fetchall()
        return clip_dict(row, posts, q["submit_minutes"])

    @app.get("/api/clips")
    def list_clips():
        status = request.args.get("status", "pending")
        with lock:
            # Re-import every call: live mode and `clipper.edit render` add clips while the queue runs.
            import_clips(db, out_root)
            rows = db.execute("SELECT * FROM clips WHERE status=? ORDER BY final_score DESC, rank",
                              (status,)).fetchall()
            counts = dict(db.execute("SELECT status, COUNT(*) FROM clips GROUP BY status").fetchall())
            posts = db.execute("SELECT * FROM posts").fetchall()
        by_clip: dict[int, list] = {}
        for p in posts:
            by_clip.setdefault(p["clip_id"], []).append(p)
        return jsonify({"clips": [clip_dict(r, by_clip.get(r["id"]), q["submit_minutes"]) for r in rows],
                        "counts": {s: counts.get(s, 0) for s in STATUSES}})

    @app.get("/api/campaigns")
    def campaigns():
        """Campaign profiles from campaigns.toml (what the picker offers)."""
        return jsonify({name: {**c, "id": name} for name, c in load_campaigns().items()})

    @app.post("/api/clips/<int:clip_id>/text")
    def save_text(clip_id: int):
        """Save the title (used when posting later). The hook is burned into the video,
        so changing it goes through /render instead."""
        get_clip(clip_id)
        title = str(request.json.get("title", "")).strip()[:200]
        with lock, db:
            db.execute("UPDATE clips SET title=?, updated_at=? WHERE id=?", (title, time.time(), clip_id))
            log_event(db, clip_id, "title", {"title": title})
        return jsonify(full(clip_id))

    @app.post("/api/clips/<int:clip_id>/posts")
    def add_post(clip_id: int):
        """Log where you posted an approved clip (and for which campaign)."""
        row = get_clip(clip_id)
        body = request.json or {}
        url = str(body.get("url", "")).strip()
        if row["status"] != "approved":
            return jsonify({"error": "Approve the clip before logging a post."}), 409
        if not re.match(r"^https?://\S+$", url):
            return jsonify({"error": "Paste the full link of the post (https://...)."}), 400
        with lock:
            taken = db.execute("SELECT clip_id, campaign FROM posts WHERE url=?", (url,)).fetchone()
            if taken:
                return jsonify({"error": f"This post is already logged (clip {taken['clip_id']}, "
                                         f"campaign '{taken['campaign'] or 'none'}'). One post = one campaign."}), 409
            with db:
                db.execute("INSERT INTO posts (clip_id, campaign, platform, url, posted_at) VALUES (?,?,?,?,?)",
                           (clip_id, row["campaign"], str(body.get("platform", ""))[:20], url,
                            float(body.get("posted_at") or time.time())))
                log_event(db, clip_id, "post", {"platform": body.get("platform", ""), "campaign": row["campaign"]})
        return jsonify(full(clip_id))

    @app.post("/api/clips/<int:clip_id>/status")
    def set_status(clip_id: int):
        row = get_clip(clip_id)
        status = request.json.get("status")
        if status not in STATUSES:
            abort(400, "unknown status")
        if status == "approved" and row["warning"] and not request.json.get("confirm_warning"):
            # Hard rule 2: a possible spoiler/cutscene needs an explicit "yes, I checked it".
            return jsonify({"error": "This clip has a warning. Confirm you watched it first."}), 409
        if status == "approved" and row["rendering"]:
            return jsonify({"error": "Wait until the re-render has finished."}), 409
        with lock, db:
            db.execute("UPDATE clips SET status=?, decided_at=?, updated_at=? WHERE id=?",
                       (status, time.time() if status != "pending" else None, time.time(), clip_id))
            log_event(db, clip_id, status)
        return jsonify(full(clip_id))

    @app.post("/api/clips/<int:clip_id>/render")
    def rerender(clip_id: int):
        """Re-render with new start/end (trim), layout, hook, context line and/or campaign."""
        row = get_clip(clip_id)
        if worker is None:
            abort(503, "rendering is off")
        body = request.json or {}
        start = max(0.0, float(body.get("start", row["start"])))
        end = float(body.get("end", row["end"]))
        layout = body.get("layout", row["layout"])
        hook = str(body.get("hook", row["hook"])).strip()[:80]
        banner = " ".join(str(body.get("context_line", row["context_line"])).split())[:90]
        campaign = str(body.get("campaign", row["campaign"]) or "")
        if layout not in LAYOUTS:
            abort(400, "unknown layout")
        if campaign and campaign not in load_campaigns():
            return jsonify({"error": f"No campaign '{campaign}' in campaigns.toml."}), 400
        if not q["min_seconds"] <= end - start <= q["max_seconds"]:
            return jsonify({"error": f"A clip must be {q['min_seconds']}-{q['max_seconds']} s long."}), 400
        if row["rendering"]:
            return jsonify({"error": "Already re-rendering this clip."}), 409
        with lock, db:
            log_event(db, clip_id, "render", {
                "start_delta": round(start - row["start"], 2), "end_delta": round(end - row["end"], 2),
                "layout": layout, "hook_changed": hook != row["hook"],
                "context_changed": banner != row["context_line"], "campaign": campaign})
        worker.submit(clip_id, start, end, layout, hook, banner, campaign)
        return jsonify(full(clip_id))

    @app.get("/video/<int:clip_id>")
    def video(clip_id: int):
        path = (out_root / get_clip(clip_id)["video"]).resolve()
        if not path.is_relative_to(out_root.resolve()) or not path.exists():
            abort(404)
        # conditional=True turns on HTTP Range support, which phones need to play/seek video.
        return send_file(path, mimetype="video/mp4", conditional=True,
                         as_attachment=request.args.get("download") == "1")

    return app


def lan_ip() -> str:
    """This laptop's address on the Wi-Fi. Connecting a UDP socket sends nothing; it only
    makes Windows pick the network card it would use to reach the internet."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main() -> None:
    p = argparse.ArgumentParser(description="AutoClipper Stage 3: review queue on your phone.")
    p.add_argument("--port", type=int, help="override queue.port from config.toml")
    p.add_argument("--host", help="override queue.host (127.0.0.1 = this laptop only)")
    p.add_argument("--config", type=Path, help="use a different config.toml")
    args = p.parse_args()
    cfg = load_config(args.config)
    host = args.host or cfg["queue"]["host"]
    port = args.port or cfg["queue"]["port"]
    app = create_app(cfg=cfg)
    print(f"\nOn this laptop:        http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        print(f"On your phone (same Wi-Fi): http://{lan_ip()}:{port}")
        print("If the phone can't connect: when Windows asks, allow Python on PRIVATE networks, and make\n"
              "sure your Wi-Fi is set to 'Private' in Windows Settings > Network & internet > Wi-Fi.")
    print("Anyone on the same Wi-Fi can open this page: don't run it on public / school Wi-Fi.")
    print("Stop with Ctrl+C.\n")
    # threaded=True: a phone loading a video and the page polling at the same time is fine.
    app.run(host=host, port=port, threaded=True, debug=False)


# --------------------------------------------------------------------------- the page

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clip queue</title>
<style>
  :root { --bg:#111; --card:#1d1d1f; --text:#eee; --muted:#999; --ok:#2e9d4f; --bad:#c0392b;
          --warn:#e6a700; --btn:#333; }
  * { box-sizing:border-box; }
  body { margin:0; font:16px/1.4 system-ui, sans-serif; background:var(--bg); color:var(--text); }
  header { position:sticky; top:0; z-index:2; background:#000; display:flex; gap:6px; padding:10px; }
  header button { flex:1; }
  main { max-width:560px; margin:0 auto; padding:10px; }
  .card { background:var(--card); border-radius:12px; padding:12px; margin-bottom:18px; }
  .card.warn { border:3px solid var(--warn); }
  video { width:100%; max-height:70vh; background:#000; border-radius:8px; display:block; }
  .meta { color:var(--muted); font-size:14px; margin:8px 0; }
  .warnbox { background:var(--warn); color:#000; font-weight:700; padding:8px; border-radius:8px; margin:8px 0; }
  label { display:block; font-size:13px; color:var(--muted); margin-top:8px; }
  input, select { width:100%; padding:10px; font-size:16px; border-radius:8px; border:1px solid #444;
                  background:#000; color:var(--text); }
  button { padding:12px 10px; font-size:16px; border:0; border-radius:8px; background:var(--btn);
           color:var(--text); cursor:pointer; }
  button.on { background:#555; font-weight:700; }
  button:disabled { opacity:.4; }
  .row { display:flex; gap:6px; margin-top:8px; align-items:center; }
  .row > * { flex:1; }
  .row span { flex:1.4; text-align:center; font-size:14px; }
  .approve { background:var(--ok); } .reject { background:var(--bad); }
  .status { font-size:14px; margin-top:8px; color:var(--warn); min-height:1em; }
  .empty { text-align:center; color:var(--muted); margin-top:40px; }
  a { color:#8ab4f8; }
</style></head>
<body>
<header>
  <button data-tab="pending">Pending</button>
  <button data-tab="approved">Approved</button>
  <button data-tab="rejected">Rejected</button>
</header>
<main id="list"></main>
<script>
const STEP = __TRIM_STEP__;
let CAMPAIGNS = {};
let tab = "pending";
let clips = [];
// Unsaved edits per clip (trim / layout / hook), kept here until you press "Re-render".
const draft = {};

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} :
    {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }
function fmt(s) { s = Math.round(s); return Math.floor(s/3600) + ":" + String(Math.floor(s%3600/60)).padStart(2,"0") + ":" + String(s%60).padStart(2,"0"); }

async function load() {
  const data = await api("/api/clips?status=" + tab);
  clips = data.clips;
  document.querySelectorAll("header button").forEach(b => {
    b.classList.toggle("on", b.dataset.tab === tab);
    b.textContent = b.dataset.tab[0].toUpperCase() + b.dataset.tab.slice(1) + " (" + data.counts[b.dataset.tab] + ")";
  });
  render();
}

function render() {
  const list = document.getElementById("list");
  // Keep cards that are playing: only rebuild the ones whose data changed.
  if (!clips.length) { list.innerHTML = '<p class="empty">Nothing here.</p>'; return; }
  const keep = new Set(clips.map(c => "c" + c.id));
  [...list.children].forEach(el => { if (!keep.has(el.id)) el.remove(); });
  clips.forEach((c, i) => {
    const sig = c.version + c.status + c.rendering + c.render_error + c.title + c.posts.length;
    let el = document.getElementById("c" + c.id);
    if (el && el.dataset.sig === sig) { list.appendChild(el); return; }
    const fresh = document.createElement("div");
    fresh.id = "c" + c.id; fresh.dataset.sig = sig;
    fresh.className = "card" + (c.warning ? " warn" : "");
    fresh.innerHTML = card(c);
    if (el) el.replaceWith(fresh); else list.appendChild(fresh);
    list.appendChild(fresh);
  });
}

function card(c) {
  const d = draft[c.id] || {};
  const start = d.start ?? c.start, end = d.end ?? c.end;
  const changed = ["start", "end", "layout", "hook", "context_line", "campaign"].some(k => d[k] !== undefined);
  const ds = start - c.start, de = end - c.end;
  const ai = c.ai || {};
  const camp = CAMPAIGNS[d.campaign ?? c.campaign];
  const sub = ai.standalone ? ` · makes sense ${ai.standalone} · hook ${ai.hook_strength} · payoff ${ai.payoff}` +
    ` · visual ${ai.visual_payoff}${ai.facecam_reaction === "big" ? " · big reaction" : ""}` : "";
  const left = c.submit_deadline ? Math.round((c.submit_deadline * 1000 - Date.now()) / 60000) : null;
  return `
    <div class="meta">#${c.rank} · ${esc(c.streamer)} · peak ${esc(c.peak_time)} · ${(end - start).toFixed(0)} s
      ${c.ai_score != null ? " · AI " + c.ai_score + "/10" : ""}${ai.category ? " · " + esc(ai.category) : ""}
      ${c.explore ? ' · <b title="A lower-ranked pick, included so the tool can learn if its ranking is right">explore</b>' : ""}</div>
    ${sub ? `<div class="meta">${sub}</div>` : ""}
    ${c.warning ? `<div class="warnbox">⚠ Possible spoiler / cutscene: ${esc(c.warning_reasons.join(", "))}.
       Watch it fully before approving.</div>` : ""}
    ${left !== null ? `<div class="warnbox" style="background:${left > 0 ? "#2e9d4f" : "#c0392b"};color:#fff">
       ${left > 0 ? `Post + submit within ${left} min` : `Submit window passed ${-left} min ago`}</div>` : ""}
    <video controls playsinline preload="metadata" src="/video/${c.id}?v=${c.version}#t=0.1"></video>
    ${c.ai_reason ? `<div class="meta">AI: ${esc(c.ai_reason)}</div>` : ""}
    ${c.transcript ? `<div class="meta">“${esc(c.transcript.slice(0, 220))}${c.transcript.length > 220 ? "…" : ""}”</div>` : ""}
    <label>Title (for posting later)</label>
    <input value="${esc(c.title)}" onchange="saveTitle(${c.id}, this.value)">
    ${(ai.title_options || []).length > 1 ? `<div class="row">${ai.title_options.map((t, k) =>
       `<button onclick="useTitle(${c.id}, ${k})">${esc(t)}</button>`).join("")}</div>` : ""}
    <div class="row"><button onclick="copyText(postCaption(${c.id}))">📋 Copy caption for TikTok</button></div>
    <div class="meta">${esc(postCaption(c.id))}</div>
    <label>Hook (big text for the first 2 s)</label>
    <input value="${esc(d.hook ?? c.hook)}" oninput="edit(${c.id}, 'hook', this.value, false)">
    ${(ai.hook_options || []).length > 1 ? `<div class="row">${ai.hook_options.map((h, k) =>
       `<button class="${h === (d.hook ?? c.hook) ? "on" : ""}" onclick="useHook(${c.id}, ${k})">${esc(h)}</button>`).join("")}</div>
       <div class="meta">Tap another hook, then "Re-render".</div>` : ""}
    <label>Context line (banner for the rest of the clip; makes it count as original)</label>
    <input value="${esc(d.context_line ?? c.context_line)}" oninput="edit(${c.id}, 'context_line', this.value, false)">
    <label>Campaign</label>
    <select onchange="edit(${c.id}, 'campaign', this.value)">
      <option value="">(none)</option>
      ${Object.values(CAMPAIGNS).filter(k => !k.streamer || k.streamer === c.streamer).map(k =>
        `<option value="${esc(k.id)}" ${k.id === (d.campaign ?? c.campaign) ? "selected" : ""}>${esc(k.name || k.id)}</option>`).join("")}
    </select>
    ${camp ? `<div class="meta">Tags: ${esc((camp.required_tags || []).join(" ")) || "-"} · Platforms: ${esc((camp.platforms || []).join(", ")) || "-"}
       ${camp.banned && camp.banned.length ? "<br>Not allowed: " + esc(camp.banned.join(", ")) : ""}
       ${camp.rules_url ? `<br><a href="${esc(camp.rules_url)}" target="_blank">rules</a>` : ""}</div>` : ""}
    <label>Start ${ds ? (ds > 0 ? "+" : "") + ds + " s" : ""} (${fmt(start)})</label>
    <div class="row"><button onclick="trim(${c.id}, 'start', -STEP)">− ${STEP} s</button>
      <button onclick="trim(${c.id}, 'start', STEP)">+ ${STEP} s</button></div>
    <label>End ${de ? (de > 0 ? "+" : "") + de + " s" : ""} (${fmt(end)})</label>
    <div class="row"><button onclick="trim(${c.id}, 'end', -STEP)">− ${STEP} s</button>
      <button onclick="trim(${c.id}, 'end', STEP)">+ ${STEP} s</button></div>
    <label>Layout</label>
    <select onchange="edit(${c.id}, 'layout', this.value)">
      ${["split", "no_facecam", "crop"].map(l => `<option ${l === (d.layout ?? c.layout) ? "selected" : ""}
        value="${l}">${{split: "Facecam on top + gameplay", no_facecam: "Gameplay, blurred background",
        crop: "Full-screen crop (camera / IRL)"}[l]}</option>`).join("")}
    </select>
    ${c.layout_reason ? `<div class="meta">Layout: ${esc(c.layout_reason)}</div>` : ""}
    <div class="row">
      <button ${changed && !c.rendering ? "" : "disabled"} onclick="rerender(${c.id})">Re-render</button>
      ${changed ? `<button onclick="resetDraft(${c.id})">Undo changes</button>` : ""}
    </div>
    <div class="row">
      ${c.status !== "approved" ? `<button class="approve" ${c.rendering ? "disabled" : ""} onclick="decide(${c.id}, 'approved')">Approve</button>` : ""}
      ${c.status !== "rejected" ? `<button class="reject" onclick="decide(${c.id}, 'rejected')">Reject</button>` : ""}
      ${c.status !== "pending" ? `<button onclick="decide(${c.id}, 'pending')">Back to pending</button>` : ""}
    </div>
    <div class="status" id="s${c.id}">${c.rendering ? "Re-rendering… (this takes up to a minute)" : esc(c.render_error && "Re-render failed: " + c.render_error)}</div>
    ${c.status === "approved" ? `
      <label>Posted it? Log the link (one post = one campaign)</label>
      <div class="row"><select id="pf${c.id}"><option>tiktok</option><option>youtube</option><option>instagram</option><option>other</option></select>
        <input id="pu${c.id}" placeholder="https://..."></div>
      <div class="row"><button onclick="logPost(${c.id})">Save post link</button></div>
      ${c.posts.map(p => `<div class="meta">${esc(p.platform)}: <a href="${esc(p.url)}" target="_blank">${esc(p.url)}</a>
        ${p.campaign ? "(" + esc(p.campaign) + ")" : ""} · ${new Date(p.posted_at * 1000).toLocaleString()}</div>`).join("")}` : ""}
    <div class="meta"><a href="/video/${c.id}?download=1">Download</a></div>`;
}

async function logPost(id) {
  const url = document.getElementById("pu" + id).value.trim();
  const platform = document.getElementById("pf" + id).value;
  try { await api(`/api/clips/${id}/posts`, {url, platform}); redrawAfterLoad(id); }
  catch (e) { say(id, e.message); }
}
async function redrawAfterLoad(id) { await load(); redraw(id); }
function useHook(id, k) { const c = clips.find(x => x.id === id); edit(id, "hook", c.ai.hook_options[k]); }
async function useTitle(id, k) {
  const c = clips.find(x => x.id === id);
  await saveTitle(id, c.ai.title_options[k]); await load(); redraw(id);
}
function postCaption(id) {
  // What to paste under the TikTok video: the AI's caption line, the campaign's required tags /
  // handle, and 3-5 hashtags (TikTok search reads them).
  const c = clips.find(x => x.id === id) || {};
  const camp = CAMPAIGNS[c.campaign] || {};
  const tags = (c.hashtags || []).slice(0, 5).map(t => "#" + t);
  const required = (camp.required_tags || []);
  const extra = camp.credit ? [camp.credit] : [];
  return [(c.ai && c.ai.caption) || c.title || "", ...required, ...extra, ...tags.filter(t => !required.includes(t))]
    .filter(Boolean).join(" ").trim();
}
function copyText(text) {
  navigator.clipboard && navigator.clipboard.writeText(text).then(() => alert("Copied"), () => prompt("Copy:", text));
}

function redraw(id) {
  const c = clips.find(x => x.id === id), el = document.getElementById("c" + id);
  if (c && el) { el.dataset.sig = ""; el.innerHTML = card(c); }
}
function edit(id, key, value, redrawCard = true) {
  draft[id] = {...draft[id], [key]: value};
  if (redrawCard) redraw(id); else updateButtons(id);
}
function updateButtons(id) {
  // Typing in the hook box: enable "Re-render" without rebuilding the card (keeps the keyboard open).
  const el = document.getElementById("c" + id);
  const b = [...el.querySelectorAll("button")].find(x => x.textContent === "Re-render");
  const c = clips.find(x => x.id === id);
  if (b && !c.rendering) b.disabled = false;
}
function trim(id, key, delta) {
  const c = clips.find(x => x.id === id), d = draft[id] || {};
  edit(id, key, Math.max(0, (d[key] ?? c[key]) + delta));
}
function resetDraft(id) { delete draft[id]; redraw(id); }
function say(id, text) { const s = document.getElementById("s" + id); if (s) s.textContent = text; }

async function saveTitle(id, title) {
  try { await api(`/api/clips/${id}/text`, {title}); say(id, "Title saved."); }
  catch (e) { say(id, e.message); }
}
async function rerender(id) {
  try { await api(`/api/clips/${id}/render`, draft[id] || {}); delete draft[id]; await load(); }
  catch (e) { say(id, e.message); }
}
async function decide(id, status) {
  const c = clips.find(x => x.id === id);
  let confirm_warning = false;
  if (status === "approved" && c.warning) {
    confirm_warning = confirm("⚠ This clip may show a cutscene or story spoiler.\n\n" +
      "Approve it only if you watched it fully and it's not a spoiler. Approve?");
    if (!confirm_warning) return;
  }
  try { await api(`/api/clips/${id}/status`, {status, confirm_warning}); await load(); }
  catch (e) { say(id, e.message); }
}

document.querySelectorAll("header button").forEach(b =>
  b.onclick = () => { tab = b.dataset.tab; document.getElementById("list").innerHTML = ""; load(); });
// While something is re-rendering, check every 3 s so the new video appears by itself.
setInterval(() => { if (clips.some(c => c.rendering)) load(); }, 3000);
// Keep the submit countdowns current.
setInterval(() => { if (clips.some(c => c.submit_deadline)) clips.filter(c => c.submit_deadline).forEach(c => redraw(c.id)); }, 60000);
api("/api/campaigns").then(c => { CAMPAIGNS = c; }).catch(() => {}).finally(load);
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
