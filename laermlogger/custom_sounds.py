"""Eigene Geräusche antrainieren (Nearest-Centroid auf YAMNet-Fingerabdrücken).

Ablauf:
1. Nutzer labelt gespeicherte Ereignis-Clips ("Wärmepumpe", "Verkehr", ...).
2. train() bildet aus den Fingerabdrücken (embed) je Label einen Durchschnitts-
   vektor (Centroid).
3. CustomModel.predict() ordnet neue Clips per Kosinus-Ähnlichkeit zu.

Alles dependency-frei (numpy + ffmpeg), kein zusätzliches ML-Framework.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

FFMPEG = shutil.which("ffmpeg")


def _labels_db(cfg) -> Path:
    return Path(cfg.db_dir) / "custom_labels.sqlite"


def _model_path(cfg) -> Path:
    return Path(cfg.classifier.model_path).parent / "custom_model.npz"


def _conn(cfg) -> sqlite3.Connection:
    c = sqlite3.connect(_labels_db(cfg))
    c.execute("CREATE TABLE IF NOT EXISTS labels "
              "(session TEXT, mp3 TEXT, label TEXT, done INTEGER DEFAULT 0, "
              "PRIMARY KEY(session, mp3))")
    # Migration für bestehende DBs ohne done-Spalte
    try:
        c.execute("ALTER TABLE labels ADD COLUMN done INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    c.commit()
    return c


def mark_done(cfg, session: str, mp3: str, done: bool = True) -> None:
    """Einen Clip als erledigt markieren (wandert ins Archiv) oder zurückholen."""
    c = _conn(cfg)
    c.execute("INSERT INTO labels (session, mp3, done) VALUES (?, ?, ?) "
              "ON CONFLICT(session, mp3) DO UPDATE SET done=excluded.done",
              (session, mp3, 1 if done else 0))
    c.commit()
    c.close()


def done_for_session(cfg, session: str) -> set:
    c = _conn(cfg)
    rows = c.execute("SELECT mp3 FROM labels WHERE session=? AND done=1",
                     (session,)).fetchall()
    c.close()
    return {r[0] for r in rows}


# -- Labeln ------------------------------------------------------------
def label_clip(cfg, session: str, mp3: str, label: str) -> None:
    c = _conn(cfg)
    label = label.strip()
    # Spalten explizit benennen (Tabelle hat auch done) und done-Flag erhalten
    c.execute("INSERT INTO labels (session, mp3, label) VALUES (?, ?, ?) "
              "ON CONFLICT(session, mp3) DO UPDATE SET label=excluded.label",
              (session, mp3, label))
    c.commit()
    c.close()


def labels_for_session(cfg, session: str) -> dict[str, str]:
    c = _conn(cfg)
    rows = c.execute("SELECT mp3, label FROM labels WHERE session=?", (session,)).fetchall()
    c.close()
    return {mp3: lab for mp3, lab in rows}


def all_labels(cfg) -> list[tuple[str, str, str]]:
    c = _conn(cfg)
    rows = c.execute("SELECT session, mp3, label FROM labels").fetchall()
    c.close()
    return rows


def label_summary(cfg) -> dict[str, int]:
    c = _conn(cfg)
    rows = c.execute("SELECT label, COUNT(*) FROM labels GROUP BY label "
                     "ORDER BY 2 DESC").fetchall()
    c.close()
    return {lab: n for lab, n in rows}


# -- Feature-Extraktion ------------------------------------------------
def decode_mp3_16k(path: Path) -> np.ndarray:
    """MP3 -> 16-kHz-mono-float via ffmpeg."""
    if FFMPEG is None:
        raise RuntimeError("ffmpeg fehlt")
    out = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1"],
        capture_output=True, check=True, timeout=30).stdout
    return np.frombuffer(out, dtype="<f4").copy()


def clip_feature(cfg, classifier, session: str, mp3: str) -> np.ndarray | None:
    path = Path(cfg.db_dir) / session / mp3
    if not path.exists():
        return None
    try:
        wave = decode_mp3_16k(path)
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        log.warning("Clip %s nicht dekodierbar: %s", mp3, exc)
        return None
    return classifier.embed(wave)


# -- Training ----------------------------------------------------------
def train(cfg, classifier) -> dict:
    """Aus allen gelabelten Clips ein Centroid-Modell bauen und speichern."""
    groups: dict[str, list] = {}
    skipped = 0
    for session, mp3, label in all_labels(cfg):
        feat = clip_feature(cfg, classifier, session, mp3)
        if feat is None:
            skipped += 1
            continue
        groups.setdefault(label, []).append(feat)
    if not groups:
        return {"trained": False, "reason": "keine (gültigen) gelabelten Clips",
                "labels": {}, "skipped": skipped}

    labels_list = sorted(groups)
    centroids = []
    for lab in labels_list:
        c = np.mean(groups[lab], axis=0)
        norm = np.linalg.norm(c)
        centroids.append(c / norm if norm > 0 else c)
    np.savez(_model_path(cfg),
             labels=np.array(labels_list, dtype=object),
             centroids=np.array(centroids, dtype="float32"))
    counts = {lab: len(groups[lab]) for lab in labels_list}
    log.info("Custom-Modell trainiert: %s (%d Clips übersprungen)", counts, skipped)
    return {"trained": True, "labels": counts, "skipped": skipped}


class CustomModel:
    def __init__(self, labels: list[str], centroids: np.ndarray, threshold: float = 0.55):
        self.labels = labels
        self.centroids = centroids
        self.threshold = threshold

    @classmethod
    def load(cls, cfg) -> "CustomModel | None":
        p = _model_path(cfg)
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=True)
        return cls(list(d["labels"]), d["centroids"])

    def predict(self, feature: np.ndarray) -> tuple[str | None, float]:
        """Kosinus-Ähnlichkeit (feature & centroids sind L2-normiert)."""
        sims = self.centroids @ feature
        i = int(np.argmax(sims))
        return (self.labels[i], float(sims[i])) if sims[i] >= self.threshold \
            else (None, float(sims[i]))


def model_status(cfg) -> dict:
    p = _model_path(cfg)
    if not p.exists():
        return {"trained": False, "labels": []}
    d = np.load(p, allow_pickle=True)
    return {"trained": True, "labels": list(d["labels"])}
