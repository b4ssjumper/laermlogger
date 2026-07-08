"""Session-Aggregator: führt SL322-Pegel und Audio-Klassifikation zusammen.

- startet/stoppt die Reader (Seriell-Thread, Audio-Stream, Classifier-Thread)
- persistiert alles in eine SQLite-Datei pro Session
- hält einen Live-State für Dashboard/WebSocket
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time as time_mod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from collections import deque

from . import custom_sounds
from .audio_capture import AudioCapture
from .audio_events import encode_mp3, ffmpeg_available
from .calibration import LevelCalibration, audio_dbfs
from .classifier import YamnetClassifier
from .config import Config
from .metrics import laeq, percentile_levels
from .serial_reader import Sl322Reader, SplSample

log = logging.getLogger(__name__)

AUDIO_EST_RANGE = "Audio-Schätzung"

SCHEMA = """
CREATE TABLE IF NOT EXISTS session (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    started_at REAL NOT NULL,
    ended_at REAL,
    device TEXT DEFAULT 'PeakTech 8005',
    location TEXT DEFAULT '',
    operator TEXT DEFAULT '',
    notes TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS spl_samples (
    ts REAL NOT NULL,
    db REAL NOT NULL,
    weighting TEXT NOT NULL,
    time_const TEXT NOT NULL,
    range_db TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spl_ts ON spl_samples(ts);
CREATE TABLE IF NOT EXISTS classifications (
    ts REAL NOT NULL,
    category TEXT NOT NULL,
    top_classes TEXT NOT NULL,       -- "Name:Score|Name:Score|..."
    impulsive INTEGER NOT NULL,
    tonal INTEGER NOT NULL,
    tonal_freq_hz REAL,
    audio_dbfs REAL
);
CREATE INDEX IF NOT EXISTS idx_cls_ts ON classifications(ts);
CREATE TABLE IF NOT EXISTS events (
    ts REAL NOT NULL,             -- Zeitpunkt des Peaks
    peak_db REAL NOT NULL,
    category TEXT DEFAULT '',     -- erkannte Lärmquelle zum Peak (YAMNet)
    top_classes TEXT DEFAULT '',
    mp3_path TEXT DEFAULT '',     -- relativ zum Session-Ordner
    duration_s REAL DEFAULT 0,
    custom_label TEXT DEFAULT ''  -- Vorhersage des eigenen Modells
);
CREATE INDEX IF NOT EXISTS idx_ev_ts ON events(ts);
"""


@dataclass
class LiveState:
    """Momentaufnahme für das Dashboard."""

    running: bool = False
    started_at: float | None = None
    current_db: float | None = None
    current_category: str = "—"
    current_top: list = field(default_factory=list)
    laeq_db: float | None = None
    lafmax_db: float | None = None
    lafmin_db: float | None = None
    percentiles: dict = field(default_factory=dict)
    n_samples: int = 0
    impulse_events: int = 0
    tonal_events: int = 0
    serial_ok: bool = False
    audio_ok: bool = False
    level_source: str = "—"       # "SL322" | "Audio (kalibriert)" | "—"
    calibration: dict = field(default_factory=dict)
    n_events: int = 0             # aufgezeichnete Audio-Ereignisse

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "started_at": self.started_at,
            "current_db": self.current_db,
            "current_category": self.current_category,
            "current_top": self.current_top,
            "laeq_db": self.laeq_db,
            "lafmax_db": self.lafmax_db,
            "lafmin_db": self.lafmin_db,
            "percentiles": self.percentiles,
            "n_samples": self.n_samples,
            "impulse_events": self.impulse_events,
            "tonal_events": self.tonal_events,
            "serial_ok": self.serial_ok,
            "audio_ok": self.audio_ok,
            "level_source": self.level_source,
            "calibration": self.calibration,
            "n_events": self.n_events,
        }


class SessionAggregator:
    def __init__(self, cfg: Config, session_name: str | None = None,
                 location: str = "", operator: str = "", notes: str = ""):
        self.cfg = cfg
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"session_{stamp}"
        self.db_path = Path(cfg.db_dir) / f"{self.session_name}.sqlite"
        self.meta = {"location": location, "operator": operator, "notes": notes}

        self.state = LiveState()
        self._state_lock = threading.Lock()
        self._spl_queue: queue.Queue = queue.Queue()
        self._serial_reader: Sl322Reader | None = None
        self._audio: AudioCapture | None = None
        self._classifier: YamnetClassifier | None = None
        self._calib = LevelCalibration()
        self._workers: list[threading.Thread] = []
        self._stop = threading.Event()
        self._conn: sqlite3.Connection | None = None
        self._db_lock = threading.Lock()
        # RAM-sichere Live-Statistik (wochentauglich): LAeq/Lmax/Lmin inkrementell,
        # Perzentile nur auf einem begrenzten jüngeren Fenster. Der FINALE Report
        # rechnet exakt aus SQLite über den gesamten Zeitraum.
        self._energy_sum = 0.0            # Summe 10^(db/10)
        self._n_total = 0
        self._lmax = float("-inf")
        self._lmin = float("inf")
        self._recent = deque(maxlen=200_000)  # ~2,8 h @20 Hz für Live-Perzentile
        self._last_spl: SplSample | None = None
        self._last_real_serial_ts = 0.0   # nur echte SL322-Samples (kein Fallback)
        # Audio-Ereignisse (Peak-Clips)
        self.session_dir = Path(cfg.db_dir) / self.session_name
        self._event_queue: queue.Queue = queue.Queue()
        self._last_event_ts = 0.0
        self._n_events = 0
        self._recent_events: deque = deque(maxlen=50)
        self._events_lock = threading.Lock()
        self._custom_model = None   # eigenes Sound-Modell (falls trainiert)

    # ------------------------------------------------------------------
    def start(self) -> None:
        Path(self.cfg.db_dir).mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.execute(
            "INSERT OR REPLACE INTO session (id, started_at, location, operator, notes) "
            "VALUES (1, ?, ?, ?, ?)",
            (time_mod.time(), self.meta["location"], self.meta["operator"],
             self.meta["notes"]),
        )
        self._conn.commit()

        self._stop.clear()
        self.state.running = True
        self.state.started_at = time_mod.time()

        # Seriell
        self._serial_reader = Sl322Reader(self.cfg.serial, self._spl_queue)
        self._serial_reader.start()

        # Audio + Classifier
        try:
            self._audio = AudioCapture(self.cfg.audio)
            self._audio.start()
            self.state.audio_ok = True
        except Exception as exc:
            log.error("Audio-Start fehlgeschlagen: %s — Messung läuft ohne Audio", exc)
            self._audio = None

        if self._audio is not None:
            try:
                self._classifier = YamnetClassifier(self.cfg.classifier)
            except Exception as exc:
                log.error("Classifier-Start fehlgeschlagen: %s", exc)
                self._classifier = None
        self.reload_custom_model()

        t1 = threading.Thread(target=self._consume_spl, name="spl-consumer", daemon=True)
        t1.start()
        self._workers.append(t1)
        if self._audio and self._classifier:
            t2 = threading.Thread(target=self._classify_loop, name="classifier", daemon=True)
            t2.start()
            self._workers.append(t2)
        # Event-Worker (MP3-Clips), nur wenn Audio + ffmpeg vorhanden
        if self._audio and self.cfg.events.enabled and ffmpeg_available():
            self.session_dir.mkdir(parents=True, exist_ok=True)
            t3 = threading.Thread(target=self._event_worker, name="event-worker", daemon=True)
            t3.start()
            self._workers.append(t3)
        elif self.cfg.events.enabled and not ffmpeg_available():
            log.warning("ffmpeg fehlt — keine Audio-Ereignisclips")
        log.info("Session %s gestartet -> %s", self.session_name, self.db_path)

    def stop(self) -> None:
        self._stop.set()
        if self._serial_reader:
            self._serial_reader.stop()
        for t in self._workers:
            t.join(timeout=3.0)
        if self._audio:
            self._audio.stop()
        if self._conn:
            with self._db_lock:
                self._conn.execute("UPDATE session SET ended_at = ? WHERE id = 1",
                                   (time_mod.time(),))
                self._conn.commit()
                self._conn.close()
            self._conn = None
        self.state.running = False
        log.info("Session %s beendet (%d Pegel-Samples)", self.session_name,
                 self.state.n_samples)

    # ------------------------------------------------------------------
    def _consume_spl(self) -> None:
        """SplSamples aus der Queue -> SQLite + Live-Statistik."""
        batch: list[SplSample] = []
        last_flush = time_mod.monotonic()
        while not self._stop.is_set() or not self._spl_queue.empty():
            try:
                sample: SplSample = self._spl_queue.get(timeout=0.5)
            except queue.Empty:
                sample = None
            if sample is not None:
                batch.append(sample)
                self._last_spl = sample
                # inkrementelle Statistik (O(1), wochentauglich)
                self._energy_sum += 10 ** (sample.db / 10)
                self._n_total += 1
                self._lmax = max(self._lmax, sample.db)
                self._lmin = min(self._lmin, sample.db)
                self._recent.append(sample.db)
                if sample.range_db != AUDIO_EST_RANGE:
                    self._last_real_serial_ts = sample.timestamp
                self._maybe_trigger_event(sample)
            now = time_mod.monotonic()
            if batch and (len(batch) >= 40 or now - last_flush > 2.0):
                with self._db_lock:
                    self._conn.executemany(
                        "INSERT INTO spl_samples VALUES (?, ?, ?, ?, ?)",
                        [(s.timestamp, s.db, s.weighting, s.time_const, s.range_db)
                         for s in batch],
                    )
                    self._conn.commit()
                batch.clear()
                last_flush = now
                self._update_stats()

    def _update_stats(self) -> None:
        now = time_mod.time()
        recent = np.asarray(self._recent) if self._recent else np.empty(0)
        with self._state_lock:
            self.state.serial_ok = (now - self._last_real_serial_ts) < 3.0
            if self.state.serial_ok:
                self.state.level_source = "SL322"
            elif self._last_spl is not None and (now - self._last_spl.timestamp) < 3.0:
                self.state.level_source = "Audio (kalibriert)"
            else:
                self.state.level_source = "—"
            self.state.n_samples = self._n_total
            if self._n_total:
                self.state.current_db = float(self._last_spl.db)
                # LAeq inkrementell über die GESAMTE Session (kein RAM-Wachstum)
                self.state.laeq_db = float(10 * np.log10(self._energy_sum / self._n_total))
                self.state.lafmax_db = self._lmax
                self.state.lafmin_db = self._lmin
                if len(recent) >= 100:
                    self.state.percentiles = {
                        k: round(v, 1) for k, v in percentile_levels(recent).items()
                    }
            self.state.calibration = self._calib.stats()
            self.state.n_events = self._n_events

    def reload_custom_model(self) -> None:
        """Eigenes Sound-Modell (neu) laden — nach dem Trainieren aufgerufen."""
        try:
            self._custom_model = custom_sounds.CustomModel.load(self.cfg)
            if self._custom_model:
                log.info("Eigenes Sound-Modell aktiv: %s", self._custom_model.labels)
        except Exception as exc:
            log.warning("Custom-Modell laden fehlgeschlagen: %s", exc)
            self._custom_model = None

    # ---- Audio-Ereignisse (Peak-Clips) --------------------------------
    def _maybe_trigger_event(self, sample: SplSample) -> None:
        """Bei Pegelspitze über Schwelle (mit Cooldown) ein Ereignis anstoßen."""
        ec = self.cfg.events
        if not (ec.enabled and self._audio and ffmpeg_available()):
            return
        if sample.db < ec.threshold_db or self._n_events >= ec.max_events:
            return
        if sample.timestamp - self._last_event_ts < ec.cooldown_seconds:
            return
        self._last_event_ts = sample.timestamp
        self._event_queue.put((sample.timestamp, sample.db))

    def _event_worker(self) -> None:
        """Wartet je Ereignis post_seconds ab, schneidet Audio-Clip als MP3."""
        ec = self.cfg.events
        rate = self.cfg.audio.target_rate
        clip_len = int((ec.pre_seconds + ec.post_seconds) * rate)
        while not self._stop.is_set() or not self._event_queue.empty():
            try:
                peak_ts, peak_db = self._event_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            # abwarten, bis genügend Nachlauf im Ringpuffer liegt
            wait_until = peak_ts + ec.post_seconds + 0.3
            while time_mod.time() < wait_until and not self._stop.is_set():
                time_mod.sleep(0.2)
            wave = self._audio.ring.latest(clip_len)
            if wave is None:
                continue
            dt = datetime.fromtimestamp(peak_ts)
            fname = f"{dt:%Y%m%d_%H%M%S}_{peak_db:.0f}dB.mp3"
            mp3_path = self.session_dir / fname
            if not encode_mp3(wave, rate, mp3_path, ec.mp3_quality):
                continue
            category, top = self._category_at(peak_ts)
            # eigenes Modell (falls trainiert) auf den Clip anwenden
            custom_label = ""
            if self._custom_model and self._classifier:
                try:
                    feat = self._classifier.embed(wave)
                    lbl, _sim = self._custom_model.predict(feat)
                    custom_label = lbl or ""
                except Exception as exc:
                    log.debug("Custom-Predict fehlgeschlagen: %s", exc)
            with self._db_lock:
                self._conn.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (peak_ts, peak_db, category, top, fname,
                     ec.pre_seconds + ec.post_seconds, custom_label),
                )
                self._conn.commit()
            self._n_events += 1
            with self._events_lock:
                self._recent_events.appendleft({
                    "ts": peak_ts, "peak_db": round(peak_db, 1),
                    "category": category, "mp3": fname, "custom_label": custom_label,
                })
            log.info("Ereignis %.1f dB um %s -> %s", peak_db, dt.strftime("%H:%M:%S"), fname)

    def _category_at(self, ts: float) -> tuple[str, str]:
        """Nächstgelegene Klassifikation (±3 s) zu einem Zeitpunkt holen."""
        if self._conn is None:
            return "", ""
        with self._db_lock:
            row = self._conn.execute(
                "SELECT category, top_classes FROM classifications "
                "WHERE ts BETWEEN ? AND ? ORDER BY ABS(ts - ?) LIMIT 1",
                (ts - 3, ts + 3, ts),
            ).fetchone()
        return (row[0], row[1]) if row else ("", "")

    def _classify_loop(self) -> None:
        """Alle ~1 s ein Audio-Fenster klassifizieren und persistieren."""
        window_s = self.cfg.classifier.window_seconds
        n = int(window_s * self.cfg.audio.target_rate)
        while not self._stop.is_set():
            time_mod.sleep(window_s)
            wave = self._audio.ring.latest(n)
            if wave is None:
                continue
            ts = time_mod.time()
            try:
                cls = self._classifier.classify(wave, ts)
            except Exception as exc:
                log.error("Klassifikation fehlgeschlagen: %s", exc)
                continue

            dbfs = audio_dbfs(wave)
            # Kalibrier-Paar bilden, wenn zeitnah ein echter serieller Pegel vorliegt
            if (self._last_spl and abs(ts - self._last_spl.timestamp) < 1.0
                    and self._last_spl.range_db != AUDIO_EST_RANGE):
                self._calib.add_pair(dbfs, self._last_spl.db, self._last_spl.range_db)

            # Audio-Fallback: seriell tot + Kalibrier-Offset vorhanden ->
            # SPL aus 125-ms-RMS-Blöcken schätzen (Fast-äquivalent, 8 Hz)
            offset = self.cfg.audio.fallback_offset_db
            if offset is not None and ts - self._last_real_serial_ts > 3.0:
                rate = self.cfg.audio.target_rate
                block = rate // 8  # 125 ms
                n_blocks = len(wave) // block
                for k in range(n_blocks):
                    seg = wave[k * block : (k + 1) * block]
                    est = audio_dbfs(seg) + offset
                    if 20.0 <= est <= 140.0:
                        self._spl_queue.put(SplSample(
                            timestamp=ts - (n_blocks - k) * 0.125,
                            db=round(est, 1), weighting="A", time_const="F",
                            range_db=AUDIO_EST_RANGE,
                        ))

            top_str = "|".join(f"{name}:{score:.3f}" for name, score in cls.top_classes)
            with self._db_lock:
                self._conn.execute(
                    "INSERT INTO classifications VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ts, cls.category, top_str, int(cls.impulsive), int(cls.tonal),
                     cls.tonal_freq_hz, dbfs),
                )
                self._conn.commit()

            with self._state_lock:
                self.state.audio_ok = True
                self.state.current_category = cls.category
                self.state.current_top = [
                    {"name": nm, "score": round(sc, 3)} for nm, sc in cls.top_classes[:3]
                ]
                if cls.impulsive:
                    self.state.impulse_events += 1
                if cls.tonal:
                    self.state.tonal_events += 1

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        with self._state_lock:
            return self.state.to_dict()

    def recent_levels(self, seconds: float = 120.0) -> list[dict]:
        """Letzte Pegel für den Dashboard-Graphen (aus SQLite, dezimiert)."""
        if self._conn is None:
            return []
        cutoff = time_mod.time() - seconds
        with self._db_lock:
            rows = self._conn.execute(
                "SELECT ts, db FROM spl_samples WHERE ts >= ? ORDER BY ts", (cutoff,)
            ).fetchall()
        step = max(1, len(rows) // 600)   # max ~600 Punkte
        return [{"ts": r[0], "db": r[1]} for r in rows[::step]]

    def recent_events(self, limit: int = 20) -> list[dict]:
        """Letzte Audio-Ereignisse für das Dashboard."""
        with self._events_lock:
            return list(self._recent_events)[:limit]
