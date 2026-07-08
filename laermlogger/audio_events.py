"""MP3-Kodierung der Ereignis-Clips via ffmpeg (16 kHz mono float -> MP3)."""

from __future__ import annotations

import logging
import shutil
import subprocess

import numpy as np

log = logging.getLogger(__name__)

FFMPEG = shutil.which("ffmpeg")


def ffmpeg_available() -> bool:
    return FFMPEG is not None


def encode_mp3(waveform: np.ndarray, rate: int, out_path, quality: int = 5) -> bool:
    """16-kHz-mono-float (-1..1) als MP3 schreiben. True bei Erfolg.

    Das Signal wird auf -1 dBFS Spitze normalisiert, damit leise AC-Ausgangs-
    Clips gut hörbar sind.
    """
    if FFMPEG is None:
        log.warning("ffmpeg fehlt — kein MP3 möglich")
        return False
    x = np.asarray(waveform, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 1e-5:
        x = x * (0.89 / peak)          # ~ -1 dBFS
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0",
        "-codec:a", "libmp3lame", "-q:a", str(quality), str(out_path),
    ]
    try:
        subprocess.run(cmd, input=pcm, check=True, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        log.error("MP3-Kodierung fehlgeschlagen: %s", exc)
        return False
