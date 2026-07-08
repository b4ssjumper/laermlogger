"""FastAPI-Server: Live-Dashboard + Session-Steuerung + Export-Endpunkte."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

from ..aggregator import SessionAggregator
from ..config import Config

log = logging.getLogger(__name__)

app = FastAPI(title="Lärmlogger")
STATIC_DIR = Path(__file__).parent / "static"

_cfg = Config.load()
_session: SessionAggregator | None = None


def get_config() -> Config:
    return _cfg


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(
        (STATIC_DIR / "index.html").read_text(),
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.post("/api/session/start")
async def start_session(payload: dict | None = None):
    global _session
    if _session and _session.state.running:
        raise HTTPException(409, "Es läuft bereits eine Session")
    payload = payload or {}
    _session = SessionAggregator(
        _cfg,
        location=payload.get("location", ""),
        operator=payload.get("operator", ""),
        notes=payload.get("notes", ""),
    )
    await asyncio.to_thread(_session.start)
    return {"session": _session.session_name}


@app.post("/api/session/stop")
async def stop_session():
    global _session
    if not _session or not _session.state.running:
        raise HTTPException(409, "Keine laufende Session")
    await asyncio.to_thread(_session.stop)
    return {"session": _session.session_name, "db": str(_session.db_path)}


@app.get("/api/state")
async def state():
    if _session is None:
        return {"running": False}
    return _session.snapshot()


@app.get("/api/levels")
async def levels(seconds: float = 120.0):
    if _session is None:
        return []
    return _session.recent_levels(seconds)


@app.get("/api/events")
async def events(limit: int = 20):
    if _session is None:
        return []
    return _session.recent_events(limit)


@app.get("/api/sessions")
async def list_sessions():
    import sqlite3

    out = []
    for f in sorted(Path(_cfg.db_dir).glob("*.sqlite"), reverse=True):
        info = {"name": f.stem, "started": None, "ended": None,
                "n_samples": 0, "n_events": 0}
        try:
            c = sqlite3.connect(f)
            row = c.execute("SELECT started_at, ended_at, location FROM session "
                            "WHERE id = 1").fetchone()
            if row:
                info["started"], info["ended"], info["location"] = row
            info["n_samples"] = c.execute("SELECT COUNT(*) FROM spl_samples").fetchone()[0]
            info["n_events"] = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            c.close()
        except Exception:
            pass
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
    if _session is not None and _session.state.running \
            and _session.session_name == session_name:
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
    """Classifier für Labeln/Trainieren — reuse der laufenden Session oder einmalig laden."""
    global _training_classifier
    if _session is not None and getattr(_session, "_classifier", None) is not None:
        return _session._classifier
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
    # laufende Session soll das neue Modell sofort nutzen
    if _session is not None and _session.state.running:
        await asyncio.to_thread(_session.reload_custom_model)
    return result


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Pusht Live-State + aktuelle Pegel jede Sekunde."""
    await ws.accept()
    try:
        while True:
            if _session is not None:
                snap = _session.snapshot()
                snap["levels"] = _session.recent_levels(120.0)
                snap["events"] = _session.recent_events(15)
                snap["session_name"] = _session.session_name
            else:
                snap = {"running": False, "levels": [], "events": []}
            await ws.send_json(snap)
            await asyncio.sleep(1.0)
    except (WebSocketDisconnect, ConnectionError):
        pass


def run(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    uvicorn.run(app, host=host or _cfg.dashboard_host,
                port=port or _cfg.dashboard_port, log_level="info")
