"""Kommandozeile: laermlogger <befehl>

  scan-hardware   Seriellen Port + Audiointerface erkennen und testen
  dump            Rohbytes vom SL322 mitschneiden (Protokoll-Verifikation)
  record          Messung ohne Dashboard starten (Ctrl+C beendet)
  measure         Mess-Daemon (entkoppelt) — vom Dashboard gesteuert
  calibrate       Audio-Pfad gegen die SL322-Anzeige kalibrieren (Fallback)
  dashboard       Live-Dashboard-Server starten
  report          PDF-Protokoll aus einer Session erzeugen
  export          CSV + JSON aus einer Session erzeugen
  test-audio      Kurze Testaufnahme mit Pegel-/Clipping-Statistik
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import Config

log = logging.getLogger("laermlogger")


def cmd_scan_hardware(cfg: Config, args) -> int:
    import sounddevice as sd
    from serial.tools import list_ports

    print("=== Serielle Ports ===")
    ports = list(list_ports.comports())
    if not ports:
        print("  keine gefunden")
    for p in ports:
        marker = " <- konfiguriert" if p.device == cfg.serial.port else ""
        print(f"  {p.device}: {p.description}{marker}")

    print("\n=== Audio-Eingabegeräte ===")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            marker = " <- konfiguriert" if cfg.audio.device.lower() in dev["name"].lower() else ""
            print(f"  #{i}: {dev['name']} ({dev['max_input_channels']} ch, "
                  f"{dev['default_samplerate']:.0f} Hz){marker}")

    print(f"\nKonfiguration: seriell={cfg.serial.port}, audio=*{cfg.audio.device}*, "
          f"{cfg.audio.capture_rate} Hz -> {cfg.audio.target_rate} Hz")
    return 0


def cmd_dump(cfg: Config, args) -> int:
    from .serial_reader import dump_raw

    print(f"Schneide {args.seconds:.0f} s Rohbytes von {cfg.serial.port} mit …")
    data = dump_raw(cfg.serial.port, cfg.serial.baudrate, args.seconds)
    print(f"{len(data)} Bytes empfangen")
    if not data:
        print("Keine Daten. Prüfen: Gerät eingeschaltet? Kabel im RS-232-Port "
              "(nicht 'Output')? Sende-Modus des SL322 aktiv?")
        return 1
    for off in range(0, min(len(data), 512), 16):
        chunk = data[off : off + 16]
        print(f"  {off:04x}: {chunk.hex(' ')}")
    if args.out:
        Path(args.out).write_bytes(data)
        print(f"Voller Dump -> {args.out}")
    # Schnelle Heuristik
    n_hdr = data.count(0x7F)
    print(f"\nHeuristik: {n_hdr} x 0x7F-Header in {len(data)} Bytes "
          f"({'sieht nach PCE-322A-Frames aus' if n_hdr > 10 else 'unerwartetes Format?'})")
    return 0


def cmd_record(cfg: Config, args) -> int:
    from .aggregator import SessionAggregator

    agg = SessionAggregator(cfg, location=args.location or "",
                            operator=args.operator or "", notes=args.notes or "")
    agg.start()
    print(f"Messung läuft -> {agg.db_path}  (Ctrl+C beendet)")
    try:
        while True:
            time.sleep(5)
            s = agg.snapshot()
            db = f"{s['current_db']:.1f}" if s["current_db"] is not None else "--.-"
            leq = f"{s['laeq_db']:.1f}" if s["laeq_db"] is not None else "--.-"
            print(f"  {time.strftime('%H:%M:%S')}  LAF={db} dB  LAeq={leq} dB  "
                  f"Quelle={s['current_category']}  n={s['n_samples']}"
                  f"{'' if s['serial_ok'] else '  [SERIELL FEHLT]'}")
            if args.duration and time.time() - s["started_at"] >= args.duration:
                break
    except KeyboardInterrupt:
        print("\nBeende …")
    agg.stop()
    print(f"Session gespeichert: {agg.db_path}")
    return 0


def cmd_measure(cfg: Config, args) -> int:
    from .daemon import MeasureDaemon

    MeasureDaemon(cfg).run()
    return 0


def cmd_calibrate(cfg: Config, args) -> int:
    """Offset Audio-dBFS -> SPL bestimmen: Nutzer liest die Geräteanzeige ab.

    Wichtig: Messbereich am SL322 vorher FEST einstellen (nicht Auto) und
    für spätere Messungen beibehalten — der AC-Ausgangspegel hängt davon ab.
    """
    import statistics

    import numpy as np

    from .audio_capture import AudioCapture
    from .calibration import audio_dbfs

    cap = AudioCapture(cfg.audio)
    cap.start()
    print("Kalibrierung: Ich messe je 2 s das Audiosignal — gib danach den auf dem\n"
          "SL322 angezeigten dB-Wert ein (leer = fertig). Am besten bei 3-4\n"
          "verschiedenen Lautstärken (leise/normal/laut, z.B. Radio).\n")
    pairs = []
    try:
        while True:
            time.sleep(2.2)
            wave = cap.ring.latest(int(2.0 * cfg.audio.target_rate))
            if wave is None:
                print("  (noch zu wenig Audio, warte …)")
                continue
            dbfs = audio_dbfs(wave)
            fast = [audio_dbfs(seg) for seg in np.array_split(wave, 16)]
            print(f"  Audio: {dbfs:6.1f} dBFS  (Spanne {min(fast):.1f} … {max(fast):.1f})")
            try:
                raw = input("  Anzeige SL322 [dB, leer=fertig]: ").strip().replace(",", ".")
            except EOFError:
                break
            if not raw:
                break
            try:
                shown = float(raw)
            except ValueError:
                print("  ungültig, übersprungen")
                continue
            pairs.append(shown - dbfs)
            print(f"  -> Offset-Kandidat: {pairs[-1]:.1f} dB ({len(pairs)} Paare)\n")
    finally:
        cap.stop()

    if not pairs:
        print("Keine Kalibrier-Paare erfasst — Offset unverändert.")
        return 1
    offset = statistics.median(pairs)
    spread = max(pairs) - min(pairs) if len(pairs) > 1 else 0.0
    cfg.audio.fallback_offset_db = round(offset, 1)
    cfg.save()
    print(f"\nOffset gespeichert: {offset:+.1f} dB (Streuung {spread:.1f} dB, "
          f"{len(pairs)} Paare) -> config.json")
    if spread > 3.0:
        print("WARNUNG: Streuung > 3 dB — Messbereich am Gerät fixiert? Nochmal "
              "mit stabileren Pegeln kalibrieren.")
    print("Der Audio-Fallback ist damit aktiv, sobald keine seriellen Daten kommen.")
    return 0


def cmd_dashboard(cfg: Config, args) -> int:
    from .server.app import run

    run(host=args.host, port=args.port)
    return 0


def cmd_report(cfg: Config, args) -> int:
    from .report.protocol import build_report

    db_path = _resolve_session(cfg, args.session)
    pdf = build_report(db_path, cfg)
    print(f"Protokoll: {pdf}")
    return 0


def cmd_export(cfg: Config, args) -> int:
    from .report.protocol import export_csv, export_json

    db_path = _resolve_session(cfg, args.session)
    print(f"CSV:  {export_csv(db_path)}")
    print(f"JSON: {export_json(db_path, cfg)}")
    return 0


def cmd_test_audio(cfg: Config, args) -> int:
    from .audio_capture import AudioCapture

    cap = AudioCapture(cfg.audio)
    out = Path(cfg.db_dir) / "test_capture.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    stats = cap.record_test_wav(out, seconds=args.seconds)
    print(f"Aufnahme: {stats['path']}")
    print(f"  Peak: {stats['peak']:.4f}  RMS: {stats['rms_dbfs']:.1f} dBFS  "
          f"geclippte Blöcke: {stats['clipped_blocks']}")
    if stats["peak"] < 0.001:
        print("  WARNUNG: praktisch kein Signal — AC-Ausgang/Kabel prüfen")
    elif stats["clipped_blocks"]:
        print("  WARNUNG: Clipping — Eingangspegel am Interface reduzieren")
    return 0


def _resolve_session(cfg: Config, name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    p = Path(cfg.db_dir) / f"{name}.sqlite"
    if p.exists():
        return p
    candidates = sorted(Path(cfg.db_dir).glob("*.sqlite"))
    if name == "latest" and candidates:
        return candidates[-1]
    sys.exit(f"Session '{name}' nicht gefunden. Vorhanden: "
             f"{[c.stem for c in candidates] or 'keine'}")


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="laermlogger",
                                     description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("scan-hardware", help="Hardware erkennen")

    p = sub.add_parser("dump", help="SL322-Rohbytes mitschneiden")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--out", help="Dump-Datei")

    p = sub.add_parser("record", help="Messung starten (ohne Dashboard)")
    p.add_argument("--location")
    p.add_argument("--operator")
    p.add_argument("--notes")
    p.add_argument("--duration", type=float, help="Sekunden, sonst bis Ctrl+C")

    sub.add_parser("measure", help="Mess-Daemon (entkoppelt, vom Dashboard gesteuert)")

    sub.add_parser("calibrate", help="Audio-Pfad gegen SL322-Anzeige kalibrieren")

    p = sub.add_parser("dashboard", help="Live-Dashboard starten")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)

    p = sub.add_parser("report", help="PDF-Protokoll erzeugen")
    p.add_argument("session", help="Session-Name, Pfad oder 'latest'")

    p = sub.add_parser("export", help="CSV/JSON exportieren")
    p.add_argument("session", help="Session-Name, Pfad oder 'latest'")

    p = sub.add_parser("test-audio", help="Audio-Testaufnahme")
    p.add_argument("--seconds", type=float, default=5.0)

    args = parser.parse_args()
    cfg = Config.load()
    handler = {
        "scan-hardware": cmd_scan_hardware,
        "dump": cmd_dump,
        "record": cmd_record,
        "measure": cmd_measure,
        "calibrate": cmd_calibrate,
        "dashboard": cmd_dashboard,
        "report": cmd_report,
        "export": cmd_export,
        "test-audio": cmd_test_audio,
    }[args.cmd]
    return handler(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
