"""Abgleich Audio-dBFS <-> kalibrierte SL322-Pegel.

Der AC-Ausgang des SL322 ist bereichsabhängig, aber innerhalb eines Bereichs
linear: SPL [dB] ≈ dBFS + Offset. Der Offset wird laufend per Median über
gepaarte Messwerte geschätzt — robust gegen Ausreißer und automatisch neu,
wenn sich der Messbereich ändert.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class LevelCalibration:
    def __init__(self, window: int = 300):
        self._pairs: dict[str, deque] = {}
        self._window = window

    def add_pair(self, audio_dbfs: float, spl_db: float, range_key: str) -> None:
        if not np.isfinite(audio_dbfs) or not np.isfinite(spl_db):
            return
        self._pairs.setdefault(range_key, deque(maxlen=self._window)).append(
            spl_db - audio_dbfs
        )

    def offset(self, range_key: str) -> float | None:
        pairs = self._pairs.get(range_key)
        if not pairs or len(pairs) < 20:
            return None
        return float(np.median(pairs))

    def estimate_spl(self, audio_dbfs: float, range_key: str) -> float | None:
        off = self.offset(range_key)
        return audio_dbfs + off if off is not None else None

    def stats(self) -> dict:
        return {
            key: {
                "n": len(d),
                "offset_db": float(np.median(d)) if len(d) >= 20 else None,
                "spread_db": float(np.percentile(d, 75) - np.percentile(d, 25))
                if len(d) >= 20 else None,
            }
            for key, d in self._pairs.items()
        }


def audio_dbfs(waveform: np.ndarray) -> float:
    """RMS-Pegel eines Audio-Fensters in dBFS."""
    rms = float(np.sqrt(np.mean(np.asarray(waveform, dtype=np.float64) ** 2)))
    return 20 * np.log10(rms) if rms > 1e-10 else -120.0
