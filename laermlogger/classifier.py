"""Geräuschquellen-Klassifizierung mit YAMNet (TFLite / LiteRT).

Zusätzlich zwei DSP-Detektoren, die in die TA-Lärm-Zuschläge einfließen:
- Impulshaltigkeit (Crest-/Transienten-Analyse)  -> KI
- Tonhaltigkeit (hervortretender Spektralpeak)   -> KT
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .config import ClassifierConfig

log = logging.getLogger(__name__)

YAMNET_INPUT_SAMPLES = 15600  # 0.975 s @ 16 kHz
FALLBACK_CATEGORY = "Sonstiges"


@dataclass
class Classification:
    timestamp: float
    category: str                 # gemappte Lärmquelle (z.B. "Verkehr")
    top_classes: list[tuple[str, float]] = field(default_factory=list)
    impulsive: bool = False
    tonal: bool = False
    tonal_freq_hz: float | None = None
    scores: "np.ndarray | None" = None   # voller 521-dim Score-Vektor (Fingerabdruck)
    custom_label: str | None = None      # Vorhersage des eigenen Modells (falls vorhanden)


class YamnetClassifier:
    def __init__(self, cfg: ClassifierConfig):
        from ai_edge_litert.interpreter import Interpreter

        self.cfg = cfg
        self.interpreter = Interpreter(model_path=cfg.model_path)
        self.interpreter.allocate_tensors()
        self._input = self.interpreter.get_input_details()[0]
        self._output = self.interpreter.get_output_details()[0]

        with open(cfg.class_map_path, newline="") as f:
            self.class_names = [row["display_name"] for row in csv.DictReader(f)]

        self.source_map = self._load_source_map(Path(cfg.source_map_path))
        log.info("YAMNet geladen: %d Klassen, %d gemappt",
                 len(self.class_names), len(self.source_map))

    @staticmethod
    def _load_source_map(path: Path) -> dict[str, str]:
        """quellen_mapping.yaml -> {display_name: Kategorie}."""
        raw = yaml.safe_load(path.read_text())
        mapping: dict[str, str] = {}
        for category, names in raw.items():
            for name in names or []:
                mapping[str(name)] = category
        return mapping

    def _map_category(self, class_name: str) -> str:
        return self.source_map.get(class_name, FALLBACK_CATEGORY)

    def score_window(self, waveform_16k: np.ndarray) -> np.ndarray:
        """Roher 521-Score-Vektor für EIN 0,975-s-Fenster (16 kHz, -1..1)."""
        x = np.asarray(waveform_16k, dtype=np.float32)
        if len(x) < YAMNET_INPUT_SAMPLES:
            x = np.pad(x, (0, YAMNET_INPUT_SAMPLES - len(x)))
        x = x[:YAMNET_INPUT_SAMPLES]
        peak = float(np.max(np.abs(x)))
        if peak > 1e-4:
            x = x * (0.5 / peak)
        self.interpreter.set_tensor(self._input["index"], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self._output["index"])[0].astype("float32")

    def embed(self, waveform_16k: np.ndarray) -> np.ndarray:
        """L2-normalisierter Mittel-Score-Vektor über einen ganzen Clip.

        Dient als „Klang-Fingerabdruck" fürs eigene Sound-Training.
        """
        x = np.asarray(waveform_16k, dtype=np.float32)
        step = YAMNET_INPUT_SAMPLES
        if len(x) <= step:
            vec = self.score_window(x)
        else:
            n = max(1, len(x) // step)
            vecs = [self.score_window(x[i * step:(i + 1) * step]) for i in range(n)]
            vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def classify(self, waveform_16k: np.ndarray, timestamp: float) -> Classification:
        """Ein 0,975-s-Fenster (16 kHz float32, -1..1) klassifizieren."""
        x = np.asarray(waveform_16k, dtype=np.float32)
        if len(x) < YAMNET_INPUT_SAMPLES:
            x = np.pad(x, (0, YAMNET_INPUT_SAMPLES - len(x)))
        x = x[:YAMNET_INPUT_SAMPLES]

        # AC-Ausgang des SL322 kann leise sein -> auf einheitlichen Pegel
        # normalisieren (YAMNet ist pegel-, aber nicht form-invariant trainiert)
        peak = float(np.max(np.abs(x)))
        if peak > 1e-4:
            x = x * (0.5 / peak)

        self.interpreter.set_tensor(self._input["index"], x)
        self.interpreter.invoke()
        scores = self.interpreter.get_tensor(self._output["index"])[0]

        top_idx = np.argsort(scores)[::-1][: self.cfg.top_k]
        top = [(self.class_names[i], float(scores[i])) for i in top_idx]

        # Kategorie: bestbewertete gemappte Klasse oberhalb der Schwelle;
        # "Stille/Hintergrund" nur wenn nichts anderes anschlägt
        category = FALLBACK_CATEGORY
        for name, score in top:
            if score < self.cfg.min_score:
                break
            cat = self._map_category(name)
            if cat != "Stille/Hintergrund":
                category = cat
                break
            category = cat

        impulsive = detect_impulsiveness(waveform_16k)
        tonal, tonal_freq = detect_tonality(waveform_16k)
        return Classification(
            timestamp=timestamp, category=category, top_classes=top,
            impulsive=impulsive, tonal=tonal, tonal_freq_hz=tonal_freq,
            scores=scores.astype("float32"),
        )


def detect_impulsiveness(waveform: np.ndarray, rate: int = 16000,
                         crest_threshold_db: float = 15.0,
                         rise_threshold_db: float = 10.0) -> bool:
    """Impulshaltigkeit: hoher Crest-Faktor UND schneller Pegelanstieg.

    35-ms-Kurzzeit-RMS (Taktmaximalpegel-Idee aus DIN 45645); ein Fenster gilt
    als impulshaltig, wenn das lauteste Kurzzeitfenster deutlich über dem
    Gesamt-RMS liegt und der Anstieg zwischen Nachbarfenstern schnell ist.
    """
    x = np.asarray(waveform, dtype=np.float64)
    if len(x) < rate // 10 or np.max(np.abs(x)) < 1e-5:
        return False
    win = int(0.035 * rate)
    n = len(x) // win
    seg_rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1))
    seg_rms = np.maximum(seg_rms, 1e-10)
    total_rms = max(np.sqrt(np.mean(x**2)), 1e-10)
    crest_db = 20 * np.log10(np.max(seg_rms) / total_rms)
    rises_db = 20 * np.diff(np.log10(seg_rms))
    return bool(crest_db >= crest_threshold_db - 10  # Crest ggü. RMS
                and np.max(rises_db, initial=0.0) >= rise_threshold_db)


def detect_tonality(waveform: np.ndarray, rate: int = 16000,
                    prominence_db: float = 10.0,
                    fmin: float = 50.0, fmax: float = 4000.0
                    ) -> tuple[bool, float | None]:
    """Tonhaltigkeit: schmalbandiger Peak, der das Nachbarspektrum überragt.

    Vereinfachtes Verfahren angelehnt an DIN 45681: Leistungsspektrum (Welch),
    Peak muss den Median seiner spektralen Umgebung (±10 %) deutlich überragen.
    """
    x = np.asarray(waveform, dtype=np.float64)
    if len(x) < rate // 4 or np.max(np.abs(x)) < 1e-5:
        return False, None
    nfft = 4096
    n = len(x) // nfft
    if n == 0:
        x = np.pad(x, (0, nfft - len(x)))
        n = 1
    segs = x[: n * nfft].reshape(n, nfft) * np.hanning(nfft)
    psd = np.mean(np.abs(np.fft.rfft(segs, axis=1)) ** 2, axis=0)
    freqs = np.fft.rfftfreq(nfft, 1 / rate)
    band = (freqs >= fmin) & (freqs <= fmax)
    psd_db = 10 * np.log10(np.maximum(psd, 1e-20))

    band_idx = np.where(band)[0]
    peak_i = band_idx[np.argmax(psd_db[band_idx])]
    f_peak = freqs[peak_i]
    lo = np.searchsorted(freqs, f_peak * 0.9)
    hi = np.searchsorted(freqs, f_peak * 1.1)
    # Umgebung ohne den Peak selbst (±2 Bins)
    env = np.concatenate([psd_db[lo:max(peak_i - 2, lo)],
                          psd_db[min(peak_i + 3, hi):hi]])
    if len(env) < 4:
        return False, None
    prominence = psd_db[peak_i] - np.median(env)
    if prominence >= prominence_db:
        return True, float(f_peak)
    return False, None
