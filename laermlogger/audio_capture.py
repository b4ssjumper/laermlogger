"""Audio-Erfassung vom USB-Interface (PCM2902) am AC-Ausgang des SL322.

Aufnahme mit 48 kHz mono (native Codec-Rate), Dezimation x3 auf 16 kHz für
YAMNet. Ein Ringpuffer hält die letzten Sekunden; der Classifier zieht sich
daraus fensterweise Daten.
"""

from __future__ import annotations

import logging
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from .config import AudioConfig

log = logging.getLogger(__name__)


def find_input_device(name_substring: str) -> int:
    """Index des Eingabegeräts finden, dessen Name den Substring enthält."""
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and name_substring.lower() in dev["name"].lower():
            return idx
    raise RuntimeError(
        f"Kein Eingabegerät mit Namen '*{name_substring}*' gefunden. "
        f"Verfügbar: {[d['name'] for d in sd.query_devices() if d['max_input_channels'] > 0]}"
    )


class DecimatingRingBuffer:
    """Threadsicherer Ringpuffer, der 48-kHz-Blöcke dezimiert (x3 -> 16 kHz) ablegt.

    Vor der Dezimation läuft ein einfacher FIR-Tiefpass (Anti-Aliasing).
    """

    def __init__(self, cfg: AudioConfig):
        assert cfg.capture_rate % cfg.target_rate == 0
        self.factor = cfg.capture_rate // cfg.target_rate
        self.rate = cfg.target_rate
        self.size = int(cfg.ring_seconds * cfg.target_rate)
        self._buf = np.zeros(self.size, dtype=np.float32)
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()
        # 31-Tap-FIR-Tiefpass, Grenze ~7 kHz bei 48 kHz (0.146 * fs)
        n = np.arange(31) - 15
        h = np.sinc(2 * 7000 / cfg.capture_rate * n) * np.hamming(31)
        self._fir = (h / h.sum()).astype(np.float32)
        self._carry = np.zeros(len(self._fir) - 1, dtype=np.float32)

    def push(self, block48k: np.ndarray) -> None:
        x = np.concatenate([self._carry, block48k])
        self._carry = x[-(len(self._fir) - 1):].copy()
        filtered = np.convolve(x, self._fir, mode="valid")
        decimated = filtered[:: self.factor].astype(np.float32)
        with self._lock:
            n = len(decimated)
            pos = self._write_pos
            end = pos + n
            if end <= self.size:
                self._buf[pos:end] = decimated
            else:
                k = self.size - pos
                self._buf[pos:] = decimated[:k]
                self._buf[: n - k] = decimated[k:]
            self._write_pos = end % self.size
            self._total_written += n

    def latest(self, n_samples: int) -> np.ndarray | None:
        """Die letzten n Samples (16 kHz) chronologisch, oder None wenn zu wenig da."""
        with self._lock:
            if self._total_written < n_samples or n_samples > self.size:
                return None
            end = self._write_pos
            start = (end - n_samples) % self.size
            if start < end:
                return self._buf[start:end].copy()
            return np.concatenate([self._buf[start:], self._buf[:end]])

    @property
    def total_written(self) -> int:
        with self._lock:
            return self._total_written


class AudioCapture:
    """Kontinuierliche Aufnahme in den DecimatingRingBuffer."""

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.ring = DecimatingRingBuffer(cfg)
        self._stream: sd.InputStream | None = None
        self.clipped_blocks = 0
        self.peak = 0.0

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.warning("Audio-Status: %s", status)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        peak = float(np.max(np.abs(mono)))
        self.peak = max(self.peak, peak)
        if peak >= 0.99:
            self.clipped_blocks += 1
        self.ring.push(mono.astype(np.float32))

    def start(self) -> None:
        device = find_input_device(self.cfg.device)
        self._stream = sd.InputStream(
            device=device,
            samplerate=self.cfg.capture_rate,
            channels=self.cfg.channels,
            blocksize=self.cfg.blocksize,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        log.info("Audio-Aufnahme gestartet: Gerät #%d, %d Hz -> %d Hz",
                 device, self.cfg.capture_rate, self.cfg.target_rate)

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def record_test_wav(self, path: Path, seconds: float = 5.0) -> dict:
        """Testaufnahme für Pegel-/Clipping-Kontrolle; gibt Statistik zurück."""
        import time as time_mod

        self.peak = 0.0
        self.clipped_blocks = 0
        started = self._stream is not None
        if not started:
            self.start()
        try:
            time_mod.sleep(seconds + 0.5)
            n = int(seconds * self.cfg.target_rate)
            data = self.ring.latest(n)
        finally:
            if not started:
                self.stop()
        if data is None:
            raise RuntimeError("zu wenig Audiodaten aufgenommen")
        pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.cfg.target_rate)
            wf.writeframes(pcm.tobytes())
        rms = float(np.sqrt(np.mean(data**2)))
        return {
            "path": str(path),
            "seconds": seconds,
            "peak": self.peak,
            "rms_dbfs": 20 * np.log10(rms) if rms > 0 else -120.0,
            "clipped_blocks": self.clipped_blocks,
        }
