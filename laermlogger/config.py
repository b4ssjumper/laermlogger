"""Zentrale Konfiguration des Lärmloggers.

Hardware-Defaults stammen vom Scan am 2026-07-07:
- Serieller Adapter: Silicon Labs CP210x -> /dev/ttyUSB0 (SL322 RS-232-Kabel)
- Audiointerface:    TI PCM2902 -> ALSA card "CODEC" (SL322 AC-Output via Cinch)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_FILE = PROJECT_ROOT / "config.json"


@dataclass
class SerialConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 9600          # PeakTech 8005 / CEM DT-885x: 9600 8N1
    timeout: float = 1.0          # Sekunden Lese-Timeout


@dataclass
class AudioConfig:
    device: str = "CODEC"         # ALSA-Kartenname des PCM2902 (sounddevice-Substring-Match)
    capture_rate: int = 48000     # native PCM2902-Rate
    target_rate: int = 16000      # YAMNet-Eingangsrate (Dezimation x3)
    channels: int = 1
    blocksize: int = 4800         # 100 ms Blöcke bei 48 kHz
    ring_seconds: float = 20.0    # Ringpuffer-Länge (>= pre+post der Event-Clips)
    # Kalibrier-Offset für den Audio-Fallback (SPL ≈ dBFS + Offset), gesetzt
    # via `laermlogger calibrate`. None = Fallback aus. Gilt nur für den beim
    # Kalibrieren eingestellten Messbereich des SL322!
    fallback_offset_db: float | None = None


@dataclass
class ClassifierConfig:
    model_path: str = str(MODELS_DIR / "yamnet.tflite")
    class_map_path: str = str(MODELS_DIR / "yamnet_class_map.csv")
    source_map_path: str = str(MODELS_DIR / "quellen_mapping.yaml")
    window_seconds: float = 0.96  # YAMNet-Fenster
    top_k: int = 5
    min_score: float = 0.2        # Mindest-Score für Mapping-Kandidaten
    # Erst ab dieser Konfidenz wird eine konkrete Lärmquelle angezeigt,
    # sonst "Sonstiges" — verhindert scheinpräzise Fehlvermutungen bei
    # schwachem/mehrdeutigem Signal.
    min_confidence: float = 0.35


@dataclass
class EventConfig:
    """Peak-getriggerte Audio-Ereignisse (MP3-Clips zum Anhören)."""

    enabled: bool = True
    threshold_db: float = 60.0     # Pegel-Schwelle für ein Ereignis
    pre_seconds: float = 6.0       # Audio vor dem Peak
    post_seconds: float = 6.0      # Audio nach dem Peak
    cooldown_seconds: float = 15.0 # Mindestabstand zwischen Ereignissen
    mp3_quality: int = 5           # ffmpeg -q:a (0=beste .. 9=kleinste)
    max_events: int = 5000         # Obergrenze pro Session (Plattenschutz)


@dataclass
class RatingConfig:
    """TA Lärm / DIN 45645-1 — Beurteilungszeiten und Richtwerte."""

    day_start: str = "06:00"      # Tag: 06–22 Uhr
    night_start: str = "22:00"    # Nacht: 22–06 Uhr (lauteste volle Stunde)
    # Ruhezeiten-Zuschlag (TA Lärm Nr. 6.5): werktags 06–07 und 20–22, So/Feiertag zusätzlich
    quiet_hours: tuple = (("06:00", "07:00"), ("20:00", "22:00"))
    quiet_bonus_db: float = 6.0
    # Immissionsrichtwerte (TA Lärm Nr. 6.1) — Default: allgemeines Wohngebiet
    limit_day_db: float = 55.0
    limit_night_db: float = 40.0
    # Zuschläge, wenn Detektion anschlägt
    impulse_surcharge_db: float = 3.0   # KI (3 oder 6 dB je nach Auffälligkeit)
    tonal_surcharge_db: float = 3.0     # KT (3 oder 6 dB)
    # Methodik-/Klasse-2-Hinweis im PDF (fachlich korrekt; schützt vor Überinterpretation).
    # Auf False, um ihn wegzulassen — dann aber bewusst nicht als amtliche Messung ausgeben.
    include_methodology_note: bool = True

    def day_start_time(self) -> time:
        return time.fromisoformat(self.day_start)

    def night_start_time(self) -> time:
        return time.fromisoformat(self.night_start)


@dataclass
class Config:
    serial: SerialConfig = field(default_factory=SerialConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    events: EventConfig = field(default_factory=EventConfig)
    rating: RatingConfig = field(default_factory=RatingConfig)
    db_dir: str = str(DATA_DIR)
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000
    daily_rollover: bool = False   # UI-Standard: Tageswechsel um Mitternacht
    label_set: list = field(default_factory=list)  # feste Labels für die Klassifizierung
    clip_filter_db: float = 0.0    # Anzeige-Filter: nur Clips ab diesem Pegel labeln (0 = alle)

    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> "Config":
        """config.json über den Defaults mergen (nur bekannte Schlüssel)."""
        cfg = cls()
        if path.exists():
            raw = json.loads(path.read_text())
            for section, values in raw.items():
                target = getattr(cfg, section, None)
                if target is None:
                    continue
                if hasattr(target, "__dataclass_fields__") and isinstance(values, dict):
                    for k, v in values.items():
                        if k in target.__dataclass_fields__:
                            setattr(target, k, v)
                else:
                    setattr(cfg, section, values)
        return cfg

    def save(self, path: Path = CONFIG_FILE) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))
