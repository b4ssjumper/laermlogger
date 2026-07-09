"""Mess-Daemon (Prozess 1) — entkoppelt von der Weboberfläche.

Führt die Messung (SessionAggregator) und kommuniziert mit dem Dashboard über
zwei Dateien in data/:
- control.json : Kommandos vom Dashboard (start/stop + Metadaten, mit seq)
- status.json  : Live-Snapshot, jede Sekunde geschrieben

So kann das Dashboard beliebig neu gestartet/aktualisiert werden, ohne die
laufende Messung zu unterbrechen.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from .aggregator import SessionAggregator
from .config import Config

log = logging.getLogger(__name__)

STATUS_STALE_S = 5.0    # ab wann das Dashboard einen Status als veraltet wertet
WATCHDOG_TIMEOUT = 30.0  # hängt die Hauptschleife länger -> Prozess neu starten lassen


class MeasureDaemon:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data_dir = Path(cfg.db_dir)
        self.control_path = self.data_dir / "control.json"
        self.status_path = self.data_dir / "status.json"
        self.model_path = Path(cfg.classifier.model_path).parent / "custom_model.npz"
        self.agg: SessionAggregator | None = None
        self.meta: dict = {}          # aktive Kampagne: location/operator/notes/rollover/threshold
        self.last_seq = -1
        self.session_day = None        # Kalendertag der aktiven Session (für Rollover)
        self.model_mtime = self._model_mtime()
        self._last_tick = time.time()  # Heartbeat für den Watchdog

    # -- Datei-Helfer ----------------------------------------------------
    def _read_control(self) -> dict | None:
        try:
            return json.loads(self.control_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _write_status(self) -> None:
        running = bool(self.agg and self.agg.state.running)
        snap = self.agg.snapshot() if self.agg else {"running": False}
        status = {
            "updated_at": time.time(),
            "running": running,
            "session_name": self.agg.session_name if self.agg else None,
            "active_db": str(self.agg.db_path) if self.agg else None,
            "daily_rollover": bool(self.meta.get("daily_rollover")),
            "threshold_db": self.meta.get("threshold_db", self.cfg.events.threshold_db),
            "snapshot": snap,
            "events": self.agg.recent_events(15) if running else [],
        }
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, default=str))
        os.replace(tmp, self.status_path)   # atomar

    def _model_mtime(self) -> float:
        try:
            return self.model_path.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    # -- Session-Lebenszyklus -------------------------------------------
    def _start_session(self) -> None:
        # Schwelle dieser Messung anwenden (aus dem Start-Kommando)
        if self.meta.get("threshold_db") is not None:
            self.cfg.events.threshold_db = float(self.meta["threshold_db"])
        self.agg = SessionAggregator(
            self.cfg,
            location=self.meta.get("location", ""),
            operator=self.meta.get("operator", ""),
            notes=self.meta.get("notes", ""),
        )
        self.agg.start()
        self.session_day = datetime.now().date()
        log.info("Session gestartet: %s (rollover=%s, schwelle=%.0f dB)",
                 self.agg.session_name, self.meta.get("daily_rollover"),
                 self.cfg.events.threshold_db)

    def _stop_session(self) -> None:
        if self.agg and self.agg.state.running:
            self.agg.stop()

    # -- Schleifen-Schritte ---------------------------------------------
    def _poll_control(self) -> None:
        cmd = self._read_control()
        if not cmd or int(cmd.get("seq", 0)) <= self.last_seq:
            return
        self.last_seq = int(cmd["seq"])
        action = cmd.get("command")
        if action == "start":
            self._stop_session()
            self.meta = {k: cmd.get(k) for k in
                         ("location", "operator", "notes", "daily_rollover", "threshold_db")}
            self._start_session()
        elif action == "stop":
            self._stop_session()
            log.info("Session gestoppt (Kommando)")

    def _maybe_rollover(self) -> None:
        if not (self.agg and self.agg.state.running and self.meta.get("daily_rollover")):
            return
        today = datetime.now().date()
        if self.session_day and today != self.session_day:
            log.info("Tageswechsel — Session wird rolliert")
            self._stop_session()
            self._start_session()

    def _maybe_reload_model(self) -> None:
        m = self._model_mtime()
        if m != self.model_mtime:
            self.model_mtime = m
            if self.agg and self.agg.state.running:
                self.agg.reload_custom_model()
                log.info("Eigenes Sound-Modell neu geladen (Datei geändert)")

    def _watchdog(self) -> None:
        """Beendet den Prozess, wenn die Hauptschleife hängt — systemd startet neu."""
        while True:
            time.sleep(10.0)
            if time.time() - self._last_tick > WATCHDOG_TIMEOUT:
                log.error("Hauptschleife hängt >%.0fs — beende Prozess zum Neustart",
                          WATCHDOG_TIMEOUT)
                os._exit(1)   # harter Exit -> systemd Restart=always fängt es ab

    def run(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        log.info("Mess-Daemon gestartet — wartet auf Kommandos (%s)", self.control_path)
        self._last_tick = time.time()
        threading.Thread(target=self._watchdog, name="watchdog", daemon=True).start()
        self._write_status()
        try:
            while True:
                self._last_tick = time.time()
                # jeder Schritt einzeln abgesichert -> ein Fehler killt die Schleife nie
                for step in (self._poll_control, self._maybe_rollover,
                             self._maybe_reload_model, self._write_status):
                    try:
                        step()
                    except Exception as exc:
                        log.error("Daemon-Schritt %s fehlgeschlagen: %s",
                                  step.__name__, exc)
                time.sleep(1.0)
        except KeyboardInterrupt:
            log.info("Daemon beendet — stoppe Session")
            self._stop_session()
            self._write_status()


def run_daemon(cfg: Config | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    MeasureDaemon(cfg or Config.load()).run()
