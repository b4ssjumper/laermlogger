"""Serieller Leser für das Schallpegelmessgerät.

Gerät: PeakTech 8005 (baugleich CEM DT-8852 / Voltcraft SL-451).
Protokoll: sigrok-Treiber `cem-dt-885x` (verifiziert am realen Gerät 2026-07-08).

- 9600 Baud, 8N1. Das Gerät streamt kontinuierlich mit ~20 Hz, ohne Poll-Kommando.
- Paketformat: Header 0xA5, dann Token-Byte, dann 0..3 Payload-Bytes.
  Wichtige Tokens:
    0x0D MEASUREMENT   2 Byte BCD: Pegel = HH.H  (z.B. 0x05 0x74 -> 57.4 dB)
    0x02/0x03          Zeitbewertung FAST / SLOW
    0x1B/0x1C          Frequenzbewertung dBA / dBC  (je 1 Payload-Byte)
    0x06 TIME          3 Byte (Uhrzeit, ungenutzt)
    0x30/0x40/0x4B/0x4C  Messbereich 30-80 / 30-130 / 50-100 / 80-130
    0x0A/0x1A          Aufnahme an / aus
    0x0F/0x1F          Batterie schwach / ok
    0x07/0x08/0x11     Bereich über- / unter- / im Bereich
  0xFF = Hold-Modus (Anzeige eingefroren) -> State-Reset.
- Einzelbyte-Kommandos ans Gerät (ohne Ack):
    0x55 Aufnahme toggeln, 0x99 dBA/dBC, 0x77 Fast/Slow, 0x88 Bereich,
    0x11 Max/Min, 0x33 Power off, 0xAC internen Speicher senden.
"""

from __future__ import annotations

import logging
import threading
import time as time_mod
from dataclasses import dataclass
from queue import Queue

import serial

from .config import SerialConfig

log = logging.getLogger(__name__)

# Token -> Anzahl Payload-Bytes (sigrok token_payloads[])
TOKEN_PAYLOAD = {
    0x02: 0, 0x03: 0, 0x04: 0, 0x05: 0, 0x06: 3, 0x07: 0, 0x08: 0, 0x09: 0,
    0x0A: 0, 0x0B: 1, 0x0C: 0, 0x0D: 2, 0x0E: 0, 0x0F: 0, 0x11: 0, 0x19: 0,
    0x1A: 0, 0x1B: 1, 0x1C: 1, 0x1F: 0, 0x30: 0, 0x40: 0, 0x4B: 0, 0x4C: 0,
}

TOKEN_MEASUREMENT = 0x0D
RANGE_TOKENS = {0x30: "30-80", 0x40: "30-130 (auto)", 0x4B: "50-100", 0x4C: "80-130"}

CMD_TOGGLE_RECORD = 0x55
CMD_TOGGLE_WEIGHT_FREQ = 0x99
CMD_TOGGLE_WEIGHT_TIME = 0x77
CMD_TOGGLE_RANGE = 0x88
CMD_TOGGLE_MAXMIN = 0x11
CMD_POWER_OFF = 0x33
CMD_TRANSFER_MEMORY = 0xAC

HEADER_LIVE = 0xA5
HEADER_LOG = 0xBB
HOLD_BYTE = 0xFF


@dataclass
class SplSample:
    """Ein dekodierter Pegelwert des Messgeräts."""

    timestamp: float          # time.time()
    db: float                 # Schalldruckpegel in dB
    weighting: str = "A"      # "A" oder "C"
    time_const: str = "F"     # "F"ast oder "S"low
    range_db: str = "30-130 (auto)"


def _bcd_measurement(data: bytes) -> float:
    """2-Byte-BCD des MEASUREMENT-Tokens -> Pegel (z.B. b'\\x05\\x74' -> 57.4)."""
    return ((data[0] >> 4) * 100 + (data[0] & 0x0F) * 10
            + (data[1] >> 4) + (data[1] & 0x0F) / 10.0)


class CemDecoder:
    """Zustandsautomat für den 0xA5-Live-Strom (ein Byte pro Aufruf).

    Hält die zuletzt gesehenen Zustands-Flags (Bewertung, Zeitkonstante,
    Bereich) und stempelt sie auf jedes MEASUREMENT-Sample.
    """

    def __init__(self):
        self._state = "INIT"
        self._token = 0
        self._payload = bytearray()
        self.weighting = "A"
        self.time_const = "F"
        self.range_db = "30-130 (auto)"
        self.recording = False
        self.battery_low = False
        self.samples_decoded = 0

    def feed(self, data: bytes) -> list[SplSample]:
        out = []
        for c in data:
            sample = self._byte(c)
            if sample is not None:
                out.append(sample)
        return out

    def _apply_flag(self, token: int) -> None:
        if token == 0x02:
            self.time_const = "F"
        elif token == 0x03:
            self.time_const = "S"
        elif token == 0x1B:
            self.weighting = "A"
        elif token == 0x1C:
            self.weighting = "C"
        elif token in RANGE_TOKENS:
            self.range_db = RANGE_TOKENS[token]
        elif token == 0x0A:
            self.recording = True
        elif token == 0x1A:
            self.recording = False
        elif token == 0x0F:
            self.battery_low = True
        elif token == 0x1F:
            self.battery_low = False

    def _byte(self, c: int) -> SplSample | None:
        if c == HOLD_BYTE:
            # Hold-Modus: Gerät friert ein und beginnt danach von vorn
            self._state = "INIT"
            return None

        if self._state == "INIT":
            if c == HEADER_LIVE:
                self._state = "TOKEN"
        elif self._state == "TOKEN":
            self._token = c
            self._payload.clear()
            need = TOKEN_PAYLOAD.get(c, -1)
            if need > 0 or need == -1:
                self._state = "DATA"
            else:
                self._apply_flag(c)
                self._state = "INIT"
        elif self._state == "DATA":
            need = TOKEN_PAYLOAD.get(self._token, -1)
            if need == -1:
                # Unbekanntes Token: bei neuem Header abbrechen
                if c in (HEADER_LIVE, HEADER_LOG):
                    self._state = "TOKEN" if c == HEADER_LIVE else "INIT"
                return None
            self._payload.append(c)
            if len(self._payload) == need:
                sample = self._finish_token()
                self._state = "INIT"
                return sample
        return None

    def _finish_token(self) -> SplSample | None:
        if self._token == TOKEN_MEASUREMENT:
            db = _bcd_measurement(bytes(self._payload))
            if 20.0 <= db <= 140.0:
                self.samples_decoded += 1
                return SplSample(
                    timestamp=time_mod.time(), db=db,
                    weighting=self.weighting, time_const=self.time_const,
                    range_db=self.range_db,
                )
        else:
            self._apply_flag(self._token)  # z.B. 0x1B/0x1C mit Payload
        return None


class Sl322Reader(threading.Thread):
    """Liest den Live-Strom des Messgeräts in einem Thread in eine Queue."""

    # Kein Datenempfang länger als diese Zeit -> Verbindung als tot betrachten
    # (das Gerät streamt sonst durchgehend mit ~20 Hz)
    STALL_TIMEOUT = 4.0
    RECONNECT_DELAY = 2.0

    def __init__(self, cfg: SerialConfig, out_queue: Queue):
        super().__init__(name="slm-reader", daemon=True)
        self.cfg = cfg
        self.out_queue = out_queue
        self._stop_event = threading.Event()
        self._ser: serial.Serial | None = None
        self._decoder = CemDecoder()
        self.reconnects = 0
        self.connected = False

    @property
    def samples_decoded(self) -> int:
        return self._decoder.samples_decoded

    def send_command(self, cmd: int) -> None:
        if self._ser and self._ser.is_open:
            try:
                self._ser.write(bytes([cmd]))
                self._ser.flush()
            except (serial.SerialException, OSError):
                pass

    def stop(self) -> None:
        self._stop_event.set()

    def _find_port(self) -> str | None:
        """Konfigurierten Port bevorzugen; sonst irgendeinen USB-Seriell-Port suchen.

        Nach einer USB-Neuanmeldung kann sich der Name ändern (ttyUSB0 -> ttyUSB1).
        """
        import glob

        if Path(self.cfg.port).exists():
            return self.cfg.port
        candidates = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        return candidates[0] if candidates else None

    def run(self) -> None:
        """Verbindungs-Loop: bei Verlust automatisch neu verbinden, endlos."""
        log.info("SLM-Reader gestartet (Ziel %s, %d Baud)",
                 self.cfg.port, self.cfg.baudrate)
        while not self._stop_event.is_set():
            port = self._find_port()
            if port is None:
                self._sleep(self.RECONNECT_DELAY)
                continue
            try:
                self._ser = serial.Serial(port, self.cfg.baudrate,
                                          timeout=self.cfg.timeout)
            except (serial.SerialException, OSError) as exc:
                log.warning("Port %s nicht öffenbar (%s) — neuer Versuch in %.0fs",
                            port, exc, self.RECONNECT_DELAY)
                self._sleep(self.RECONNECT_DELAY)
                continue

            self.connected = True
            self._decoder = CemDecoder()   # frischer Zustand nach (Wieder-)Verbindung
            log.info("Verbunden auf %s", port)
            try:
                self._read_loop()
            except (serial.SerialException, OSError) as exc:
                log.warning("Verbindung verloren (%s) — verbinde neu…", exc)
            finally:
                self.connected = False
                try:
                    if self._ser:
                        self._ser.close()
                except (serial.SerialException, OSError):
                    pass
                self._ser = None
            if not self._stop_event.is_set():
                self.reconnects += 1
                self._sleep(self.RECONNECT_DELAY)
        log.info("SLM-Reader beendet (%d Samples, %d Reconnects)",
                 self.samples_decoded, self.reconnects)

    def _read_loop(self) -> None:
        last_data = time_mod.monotonic()
        while not self._stop_event.is_set():
            data = self._ser.read(256)
            if data:
                last_data = time_mod.monotonic()
                for sample in self._decoder.feed(data):
                    self.out_queue.put(sample)
            elif time_mod.monotonic() - last_data > self.STALL_TIMEOUT:
                # Gerät streamt normal durchgehend -> längere Stille = Verbindung tot
                raise serial.SerialException("keine Daten (Stall-Timeout)")

    def _sleep(self, seconds: float) -> None:
        self._stop_event.wait(seconds)


def dump_raw(port: str, baudrate: int = 9600, seconds: float = 10.0) -> bytes:
    """Rohbytes mitschneiden — zur Protokoll-Verifikation."""
    with serial.Serial(port, baudrate, timeout=1.0) as ser:
        deadline = time_mod.monotonic() + seconds
        chunks = []
        while time_mod.monotonic() < deadline:
            chunks.append(ser.read(256))
        return b"".join(chunks)


def read_live(port: str, baudrate: int = 9600, seconds: float = 5.0) -> list[SplSample]:
    """Kurzer Live-Mitschnitt, dekodiert — für `laermlogger dump`/Verifikation."""
    decoder = CemDecoder()
    samples = []
    with serial.Serial(port, baudrate, timeout=0.2) as ser:
        deadline = time_mod.monotonic() + seconds
        while time_mod.monotonic() < deadline:
            samples.extend(decoder.feed(ser.read(256)))
    return samples
