import json
import time

import pytest

from clipper import queue_app
from clipper.config import load_config


def make_vod(out_root, streamer="somebody", vod_id="123", warning_rank=2):
    """A fake VOD folder: candidates.json + two finished clips (rank 1 normal, rank 2 warned)."""
    vod = out_root / streamer / vod_id
    (vod / "final" / "work").mkdir(parents=True)
    cands = []
    for rank in (1, 2):
        warn = rank == warning_rank
        cands.append({"rank": rank, "peak_time": f"0h0{rank}m00s", "start": 100.0 * rank,
                      "end": 100.0 * rank + 30, "peak": 100.0 * rank + 12, "peak_score": 5.0,
                      "transcript": "no way bro", "warning": warn,
                      "warning_reasons": ["AI spoiler risk: high"] if warn else [],
                      "ai": {"score": 7, "hook": "HE DID WHAT", "title": "Big moment", "reason": "funny"}})
        name = f"{rank:02d}_{'WARN_' if warn else ''}0h0{rank}m00s"
        (vod / "final" / f"{name}.mp4").write_bytes(b"\x00" * 5000)
    # Rank 1 was re-rendered with another layout: the sidecar says so.
    (vod / "final" / "work" / "01_0h01m00s.json").write_text(
        json.dumps({"rank": 1, "start": 98.0, "end": 130.0, "layout": "crop", "hook": "EDITED"}))
    (vod / "candidates.json").write_text(json.dumps(
        {"url": "https://www.twitch.tv/videos/123", "platform": "twitch", "vod_id": vod_id,
         "streamer": streamer, "candidates": cands}))
    return vod


@pytest.fixture
def client(tmp_path):
    out_root = tmp_path / "out"
    make_vod(out_root)
    app = queue_app.create_app(db_path=tmp_path / "q.db", out_root=out_root, cfg=load_config(),
                               start_worker=False)
    return app.test_client()


def test_import_reads_candidates_and_sidecar(client):
    clips = client.get("/api/clips?status=pending").get_json()["clips"]
    assert [c["rank"] for c in clips] == [1, 2]
    one, two = clips
    assert (one["layout"], one["hook"], one["start"], one["end"]) == ("crop", "EDITED", 98.0, 130.0)
    assert (two["layout"], two["hook"], two["warning"]) == ("split", "HE DID WHAT", True)
    assert one["title"] == "Big moment" and one["ai_reason"] == "funny"


def test_reimport_keeps_decisions(tmp_path):
    out_root = tmp_path / "out"
    make_vod(out_root)
    db = queue_app.connect(tmp_path / "q.db")
    assert queue_app.import_clips(db, out_root) == 2
    db.execute("UPDATE clips SET status='approved', title='mine' WHERE rank=1")
    assert queue_app.import_clips(db, out_root) == 0
    row = db.execute("SELECT status, title FROM clips WHERE rank=1").fetchone()
    assert (row["status"], row["title"]) == ("approved", "mine")


def test_warned_clip_needs_confirmation(client):
    clips = client.get("/api/clips").get_json()["clips"]
    warned = next(c for c in clips if c["warning"])
    r = client.post(f"/api/clips/{warned['id']}/status", json={"status": "approved"})
    assert r.status_code == 409
    r = client.post(f"/api/clips/{warned['id']}/status", json={"status": "approved", "confirm_warning": True})
    assert r.status_code == 200 and r.get_json()["status"] == "approved"
    counts = client.get("/api/clips").get_json()["counts"]
    assert counts == {"pending": 1, "approved": 1, "rejected": 0}


def test_reject_and_back_to_pending(client):
    cid = client.get("/api/clips").get_json()["clips"][0]["id"]
    assert client.post(f"/api/clips/{cid}/status", json={"status": "rejected"}).get_json()["status"] == "rejected"
    assert client.post(f"/api/clips/{cid}/status", json={"status": "pending"}).get_json()["status"] == "pending"
    assert client.post(f"/api/clips/{cid}/status", json={"status": "posted"}).status_code == 400


def test_title_saved(client):
    cid = client.get("/api/clips").get_json()["clips"][0]["id"]
    assert client.post(f"/api/clips/{cid}/text", json={"title": " New title "}).get_json()["title"] == "New title"


def test_video_supports_range_requests(client):
    cid = client.get("/api/clips").get_json()["clips"][0]["id"]
    r = client.get(f"/video/{cid}", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.data) == 100     # phones need this to play / seek
    assert client.get("/video/9999").status_code == 404


def test_page_loads(client):
    html = client.get("/").get_data(as_text=True)
    assert "Approve" in html and "Re-render" in html and "__TRIM_STEP__" not in html


def test_rerender_validates_and_queues(tmp_path, monkeypatch):
    out_root = tmp_path / "out"
    make_vod(out_root)
    jobs = []
    # Don't really render: record the job and pretend a new version was made.
    def fake_render(self, clip_id, start, end, layout, hook):
        jobs.append((clip_id, start, end, layout, hook))
        with self.lock, self.db:
            self.db.execute('UPDATE clips SET start=?, "end"=?, layout=?, hook=?, version=version+1, '
                            "rendering=0 WHERE id=?", (start, end, layout, hook, clip_id))
    monkeypatch.setattr(queue_app.RenderWorker, "_render", fake_render)
    client = queue_app.create_app(db_path=tmp_path / "q.db", out_root=out_root, cfg=load_config()).test_client()
    c = client.get("/api/clips").get_json()["clips"][1]

    assert client.post(f"/api/clips/{c['id']}/render", json={"start": 0, "end": 2}).status_code == 400  # too short
    assert client.post(f"/api/clips/{c['id']}/render", json={"layout": "weird"}).status_code == 400
    r = client.post(f"/api/clips/{c['id']}/render",
                    json={"start": c["start"] - 2, "end": c["end"] + 2, "layout": "no_facecam", "hook": "NEW"})
    assert r.status_code == 200
    for _ in range(50):
        if jobs:
            break
        time.sleep(0.05)
    assert jobs == [(c["id"], c["start"] - 2, c["end"] + 2, "no_facecam", "NEW")]
    for _ in range(50):
        after = next(x for x in client.get("/api/clips").get_json()["clips"] if x["id"] == c["id"])
        if not after["rendering"]:
            break
        time.sleep(0.05)
    assert (after["layout"], after["hook"], after["version"]) == ("no_facecam", "NEW", 1)
