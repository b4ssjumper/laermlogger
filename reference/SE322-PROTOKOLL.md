# SL322 (alte Generation, TestLink SE-322) — Protokoll-Erkenntnisse

Stand 2026-07-07, Quelle: Binary-Analyse von `SE322.exe` V4.03
(Dostmann-Download, Kopie in `SL322_V403.zip`; Methodentabelle + DFM-Ressourcen
+ Disassembly mit capstone/pefile).

## Serielle Parameter (aus DFM der TComPort-Komponente)

- **9600 Baud, 8N1**
- **FlowControl.DTR = dcDisable** (DTR AUS!)
- **FlowControl.RTS = rcEnable** (RTS AN)
- Gerät **streamt von selbst** — die Software sendet kein Poll-Kommando:
  `MainTimerTimer` (0x413228) zählt nur empfangene Frames und zeigt
  "No Connection", wenn keine ankommen.

## Wichtige Handler-Adressen (Methodentabelle SE322.exe)

- `MainTimerTimer`  0x413228 — Verbindungs-Überwachung
- `Comm1RxChar`     0x41a120 — Frame-Parser (Format von hier extrahierbar)
- `LoadData1Click`  0x4193d4 — Logger-Download
- `EraseMemory1Click` 0x419cec
- `Timer1Timer`     0x41afe4

## Diagnose-Historie (Kabel/Gerät)

Trotz korrekter Konfiguration (auch via libusb, alle DTR/RTS-Kombinationen,
alle Baudraten, Brute-Force 1-Byte-Kommandos): Gerät sendet nie etwas.
Kernel-Treiber-Problem des CP2102-Clones (`failed set request 0x12 -110`)
kommt erschwerend dazu, wurde aber via libusb umgangen — Stille bleibt.
Windows 11 + Originalsoftware (SE322 und PCE-V3.4.2): ebenfalls No Connection.

**Arbeitshypothese:** Die alte SL322-Generation nutzt am Klinkenstecker echte
**RS-232-Pegel** (Original-Zubehör war ein RS232-Kabel „SE-300" für
PC-COM-Ports). Das vorhandene USB-Kabel ist ein TTL-Pegel-CP2102 —
Pegel-/Polaritäts-Mismatch erklärt alle Symptome (nichts empfangbar,
Kommandos kommen nie an). Ersatz bestellt: DSD-TECH FT232RL **mit echten
RS-232-Pegeln** und 3,5-mm-Klinke.

## Wenn das neue Kabel da ist

1. `laermlogger dump --seconds 10` (config: 9600 Baud!) — Gerät muss von
   selbst streamen, sonst DTR/RTS-Zustände prüfen (DTR aus, RTS an).
2. Frame-Format aus Dump ableiten; falls unklar: `Comm1RxChar` @ 0x41a120
   disassemblieren (Parser-Logik = Frame-Definition).
3. `serial_reader.py` von PCE-322A-Frames auf SE322-Frames umstellen.
