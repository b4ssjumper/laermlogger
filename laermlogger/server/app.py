"""FastAPI-Dashboard (Prozess 2): reiner Leser + Steuerung.

Die Messung läuft im Mess-Daemon (Prozess 1). Kommunikation über zwei Dateien:
- data/status.json  (Daemon schreibt Live-Status)  -> /api/state, /ws, /api/levels
- data/control.json (Dashboard schreibt Kommandos) <- /api/session/start|stop

Dadurch ist das Dashboard frei neustartbar, ohne die Messung zu unterbrechen.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from ..config import Config

log = logging.getLogger(__name__)

app = FastAPI(title="Lärmlogger")
STATIC_DIR = Path(__file__).parent / "static"

_cfg = Config.load()
_DATA = Path(_cfg.db_dir)
_STATUS = _DATA / "status.json"
_CONTROL = _DATA / "control.json"
_STATUS_STALE_S = 5.0


def get_config() -> Config:
    return _cfg


def _read_status() -> dict:
    """status.json lesen; veraltet/fehlt -> Daemon läuft nicht / keine Messung."""
    try:
        st = json.loads(_STATUS.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"running": False, "daemon": False}
    if time.time() - st.get("updated_at", 0) > _STATUS_STALE_S:
        return {"running": False, "daemon": False}
    st["daemon"] = True
    return st


def _write_control(command: str, **fields) -> None:
    """Kommando an den Mess-Daemon schreiben (seq hochzählen)."""
    seq = 0
    try:
        seq = int(json.loads(_CONTROL.read_text()).get("seq", 0))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        pass
    payload = {"seq": seq + 1, "command": command, **fields}
    _DATA.mkdir(parents=True, exist_ok=True)
    tmp = _CONTROL.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(_CONTROL)


def _active_db() -> Path | None:
    st = _read_status()
    p = st.get("active_db")
    return Path(p) if p and Path(p).exists() else None


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        (STATIC_DIR / "index.html").read_text(),
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.post("/api/session/start")
async def start_session(payload: dict | None = None):
    st = _read_status()
    if not st.get("daemon"):
        raise HTTPException(503, "Mess-Daemon läuft nicht (laermlogger measure)")
    if st.get("running"):
        raise HTTPException(409, "Es läuft bereits eine Messung")
    payload = payload or {}
    _write_control(
        "start",
        location=payload.get("location", ""),
        operator=payload.get("operator", ""),
        notes=payload.get("notes", ""),
        daily_rollover=bool(payload.get("daily_rollover", False)),
        threshold_db=payload.get("threshold_db"),
    )
    return {"ok": True}


@app.post("/api/session/stop")
async def stop_session():
    _write_control("stop")
    return {"ok": True}


@app.get("/api/state")
async def state():
    st = _read_status()
    snap = st.get("snapshot", {"running": False})
    snap["running"] = st.get("running", False)
    snap["daemon"] = st.get("daemon", False)
    snap["session_name"] = st.get("session_name")
    snap["events"] = st.get("events", [])
    return snap


@app.get("/api/levels")
async def levels(seconds: float = 120.0):
    from ..report.protocol import session_levels

    db = _active_db()
    if db is None:
        return []
    return await asyncio.to_thread(session_levels, db, 600, seconds)


def _sessions_overlapping(cutoff: float) -> list[Path]:
    """Session-Dateien, deren Zeitraum das Fenster [cutoff, jetzt] berührt."""
    import sqlite3

    out = []
    for f in Path(_cfg.db_dir).glob("session_*.sqlite"):
        try:
            c = sqlite3.connect(f)
            row = c.execute("SELECT started_at, ended_at FROM session "
                            "WHERE id = 1").fetchone()
            c.close()
        except Exception:
            continue
        if not row or not row[0]:
            continue
        end = row[1] or time.time()   # laufende Messung: bis jetzt
        if end >= cutoff:
            out.append(f)
    return sorted(out)


@app.get("/api/timeline")
async def timeline(seconds: float = 120.0):
    """Durchgehender Pegelverlauf über ALLE Messungen im gewählten Fenster."""
    from ..report.protocol import session_levels

    cutoff = time.time() - seconds
    paths = _sessions_overlapping(cutoff)
    if not paths:
        return []

    def build():
        per = max(900 // len(paths), 200)
        pts: list[dict] = []
        for p in paths:
            pts.extend(session_levels(p, per, seconds))
        pts.sort(key=lambda x: x["ts"])
        return pts

    return await asyncio.to_thread(build)


@app.get("/api/events")
async def events(limit: int = 20):
    return _read_status().get("events", [])[:limit]


@app.get("/api/sessions")
async def list_sessions():
    import sqlite3

    out = []
    for f in sorted(Path(_cfg.db_dir).glob("*.sqlite"), reverse=True):
        # Nur echte Messungen: müssen eine gültige session-Zeile haben.
        # (schließt z.B. custom_labels.sqlite = Trainings-Labels aus)
        info = {"name": f.stem, "started": None, "ended": None,
                "location": "", "n_samples": 0, "n_events": 0}
        try:
            c = sqlite3.connect(f)
            row = c.execute("SELECT started_at, ended_at, location FROM session "
                            "WHERE id = 1").fetchone()
            if not row:
                c.close()
                continue
            info["started"], info["ended"], info["location"] = row
            info["n_samples"] = c.execute("SELECT COUNT(*) FROM spl_samples").fetchone()[0]
            info["n_events"] = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            c.close()
        except Exception:
            continue   # keine Session-Struktur -> überspringen
        out.append(info)
    return out


@app.get("/api/audio/{session_name}/{filename}")
async def audio_clip(session_name: str, filename: str):
    """MP3-Ereignisclip einer Session ausliefern."""
    # Pfad-Traversal verhindern
    if "/" in filename or ".." in filename or not filename.endswith(".mp3"):
        raise HTTPException(400, "ungültiger Dateiname")
    path = Path(_cfg.db_dir) / session_name / filename
    if not path.exists():
        raise HTTPException(404, "Clip nicht gefunden")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)


@app.get("/api/session/{session_name}/summary")
async def session_summary_ep(session_name: str):
    """Kennwerte einer abgeschlossenen Messung (ohne PDF)."""
    from ..report.protocol import session_summary

    db_path = Path(_cfg.db_dir) / f"{session_name}.sqlite"
    if not db_path.exists():
        raise HTTPException(404, "Session nicht gefunden")
    return await asyncio.to_thread(session_summary, db_path, _cfg)


@app.get("/api/session/{session_name}/levels")
async def session_levels_ep(session_name: str, buckets: int = 800):
    """Aggregierter Pegelverlauf einer abgeschlossenen Messung."""
    from ..report.protocol import session_levels

    db_path = Path(_cfg.db_dir) / f"{session_name}.sqlite"
    if not db_path.exists():
        raise HTTPException(404, "Session nicht gefunden")
    return await asyncio.to_thread(session_levels, db_path, buckets)


def _sessions_in_range(from_date: str | None, to_date: str | None) -> list[Path]:
    """session_*.sqlite, deren Messzeitraum den Filter [from, to] BERÜHRT
    (Überlappung — eine Messung über Mitternacht zählt für beide Tage)."""
    import sqlite3
    from datetime import date, datetime, time as dtime

    ts_from = (datetime.combine(date.fromisoformat(from_date), dtime.min).timestamp()
               if from_date else 0.0)
    ts_to = (datetime.combine(date.fromisoformat(to_date), dtime.max).timestamp()
             if to_date else 9e18)
    out = []
    for f in Path(_cfg.db_dir).glob("session_*.sqlite"):
        try:
            c = sqlite3.connect(f)
            row = c.execute("SELECT started_at, ended_at FROM session "
                            "WHERE id = 1").fetchone()
            c.close()
        except Exception:
            continue
        if not row or not row[0]:
            continue
        start, end = row[0], (row[1] or time.time())
        if end >= ts_from and start <= ts_to:   # Überlappung
            out.append(f)
    return sorted(out)


@app.get("/api/combine/summary")
async def combine_summary(from_: str = Query(None, alias="from"), to: str = None):
    from ..report.protocol import combined_summary

    paths = _sessions_in_range(from_, to)
    if not paths:
        raise HTTPException(404, "keine Messungen im Zeitraum")
    return await asyncio.to_thread(combined_summary, paths, _cfg)


@app.get("/api/combine/levels")
async def combine_levels(from_: str = Query(None, alias="from"), to: str = None, buckets: int = 900):
    from ..report.protocol import combined_levels

    paths = _sessions_in_range(from_, to)
    if not paths:
        return []
    return await asyncio.to_thread(combined_levels, paths, buckets)


@app.get("/api/combine/events")
async def combine_events(from_: str = Query(None, alias="from"), to: str = None,
                         offset: int = 0, limit: int = 25, include_done: bool = False):
    """Ereignis-Clips im Zeitraum (nach Pegel absteigend), zum Labeln.
    Erledigte Clips (im Archiv) werden standardmäßig ausgeblendet.
    offset/limit für Lazy Loading; gibt {clips, total, remaining} zurück."""
    from .. import custom_sounds
    from ..report.protocol import _combined_audio_events

    paths = _sessions_in_range(from_, to)
    events = await asyncio.to_thread(_combined_audio_events, paths)
    label_cache: dict = {}
    done_cache: dict = {}
    filtered = []
    for e in events:
        s = e["session"]
        if s not in label_cache:
            label_cache[s] = custom_sounds.labels_for_session(_cfg, s)
            done_cache[s] = custom_sounds.done_for_session(_cfg, s)
        is_done = e["mp3"] in done_cache[s]
        if is_done and not include_done:
            continue
        filtered.append({"ts": e["start"].timestamp(), "peak_db": e["peak_db"],
                         "category": e["category"], "mp3": e["mp3"],
                         "custom_label": e.get("custom_label", ""),
                         "session": e["session"],
                         "user_label": label_cache[s].get(e["mp3"], ""),
                         "done": is_done})
    page = filtered[offset:offset + limit]
    return {"clips": page, "total": len(filtered),
            "remaining": max(0, len(filtered) - offset - len(page))}


@app.post("/api/done")
async def set_done(payload: dict):
    """Clip als erledigt markieren (Archiv) oder zurückholen."""
    from .. import custom_sounds

    session, mp3 = payload.get("session"), payload.get("mp3")
    if not session or not mp3:
        raise HTTPException(400, "session und mp3 nötig")
    custom_sounds.mark_done(_cfg, session, mp3, bool(payload.get("done", True)))
    return {"ok": True}


@app.get("/api/combine/report")
async def combine_report(from_: str = Query(None, alias="from"), to: str = None):
    from ..report.protocol import build_combined_report

    paths = _sessions_in_range(from_, to)
    if not paths:
        raise HTTPException(404, "keine Messungen im Zeitraum")
    name = f"kombi_{from_ or 'start'}_{to or 'ende'}.pdf"
    out = Path(_cfg.db_dir) / name
    await asyncio.to_thread(build_combined_report, paths, _cfg, out)
    return FileResponse(out, media_type="application/pdf", filename=name)


@app.get("/api/session/{session_name}/events")
async def session_events(session_name: str):
    """Alle Ereignisse einer (auch abgeschlossenen) Session aus SQLite."""
    import sqlite3

    db_path = Path(_cfg.db_dir) / f"{session_name}.sqlite"
    if not db_path.exists():
        raise HTTPException(404, "Session nicht gefunden")
    c = sqlite3.connect(db_path)
    try:
        rows = c.execute("SELECT ts, peak_db, category, mp3_path, custom_label "
                         "FROM events ORDER BY ts DESC").fetchall()
    except sqlite3.OperationalError:
        rows = [(r[0], r[1], r[2], r[3], "") for r in c.execute(
            "SELECT ts, peak_db, category, mp3_path FROM events ORDER BY ts DESC")]
    c.close()
    return [{"ts": r[0], "peak_db": r[1], "category": r[2], "mp3": r[3],
             "custom_label": r[4]} for r in rows]


@app.delete("/api/session/{session_name}")
async def delete_session(session_name: str):
    """Session komplett löschen: SQLite, PDF/CSV/JSON und Audio-Clips."""
    import shutil

    if "/" in session_name or ".." in session_name:
        raise HTTPException(400, "ungültiger Name")
    st = _read_status()
    if st.get("running") and st.get("session_name") == session_name:
        raise HTTPException(409, "Laufende Messung kann nicht gelöscht werden")
    base = Path(_cfg.db_dir)
    db_path = base / f"{session_name}.sqlite"
    if not db_path.exists():
        raise HTTPException(404, "Session nicht gefunden")
    removed = []
    for suffix in (".sqlite", ".pdf", ".csv", ".json"):
        p = base / f"{session_name}{suffix}"
        if p.exists():
            p.unlink()
            removed.append(p.name)
    clip_dir = base / session_name
    if clip_dir.is_dir():
        shutil.rmtree(clip_dir)
        removed.append(f"{session_name}/ (Clips)")
    log.info("Session %s gelöscht: %s", session_name, ", ".join(removed))
    return {"deleted": session_name, "files": removed}


@app.get("/api/export/{session_name}")
async def export(session_name: str, fmt: str = "csv"):
    """CSV- oder JSON-Rohexport einer Session."""
    from ..report.protocol import export_csv, export_json

    db_path = Path(_cfg.db_dir) / f"{session_name}.sqlite"
    if not db_path.exists():
        raise HTTPException(404, f"Session {session_name} nicht gefunden")
    if fmt == "json":
        path = await asyncio.to_thread(export_json, db_path, _cfg)
        return FileResponse(path, media_type="application/json", filename=path.name)
    path = await asyncio.to_thread(export_csv, db_path)
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/report/{session_name}")
async def report(session_name: str):
    """PDF-Protokoll für eine abgeschlossene Session erzeugen und liefern."""
    from ..report.protocol import build_report

    db_path = Path(_cfg.db_dir) / f"{session_name}.sqlite"
    if not db_path.exists():
        raise HTTPException(404, f"Session {session_name} nicht gefunden")
    pdf_path = await asyncio.to_thread(build_report, db_path, _cfg)
    return FileResponse(pdf_path, media_type="application/pdf",
                        filename=pdf_path.name)


_training_classifier = None


def _get_classifier():
    """Classifier fürs Trainieren — einmalig im Dashboard-Prozess laden.
    Der Mess-Daemon lädt das neu trainierte Modell selbstständig per Datei-mtime."""
    global _training_classifier
    if _training_classifier is None:
        from ..classifier import YamnetClassifier
        _training_classifier = YamnetClassifier(_cfg.classifier)
    return _training_classifier


@app.post("/api/label")
async def set_label(payload: dict):
    """Einem Clip ein eigenes Label zuweisen (leeres Label = entfernen)."""
    from .. import custom_sounds

    session = payload.get("session")
    mp3 = payload.get("mp3")
    label = payload.get("label", "")
    if not session or not mp3:
        raise HTTPException(400, "session und mp3 nötig")
    custom_sounds.label_clip(_cfg, session, mp3, label)
    return {"ok": True, "summary": custom_sounds.label_summary(_cfg)}


@app.get("/api/labels")
async def get_labels():
    """Übersicht aller vergebenen Labels + Modellstatus."""
    from .. import custom_sounds

    return {"summary": custom_sounds.label_summary(_cfg),
            "model": custom_sounds.model_status(_cfg)}


@app.get("/api/session/{session_name}/labels")
async def session_labels(session_name: str):
    from .. import custom_sounds

    return custom_sounds.labels_for_session(_cfg, session_name)


@app.post("/api/train")
async def train_model():
    """Eigenes Sound-Modell aus den gelabelten Clips bauen."""
    from .. import custom_sounds

    classifier = await asyncio.to_thread(_get_classifier)
    result = await asyncio.to_thread(custom_sounds.train, _cfg, classifier)
    # Der Mess-Daemon lädt das neue Modell automatisch (Datei-mtime)
    return result


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Pusht Live-State (aus status.json) jede Sekunde."""
    await ws.accept()
    try:
        while True:
            st = _read_status()
            snap = st.get("snapshot", {"running": False})
            snap["running"] = st.get("running", False)
            snap["daemon"] = st.get("daemon", False)
            snap["session_name"] = st.get("session_name")
            snap["events"] = st.get("events", [])
            await ws.send_json(snap)
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, ConnectionError):
        pass


def run(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    uvicorn.run(app, host=host or _cfg.dashboard_host,
                port=port or _cfg.dashboard_port, log_level="info")
