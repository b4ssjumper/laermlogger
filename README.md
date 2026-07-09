# Lärmlogger

Lärmmessprotokoll-System für den Raspberry Pi 5 mit **PeakTech 8005**
Schallpegelmessgerät (baugleich CEM DT-8852 / Voltcraft SL-451): kalibrierte
Pegel über den USB-Livestrom, optionale Geräuschquellen-Erkennung per YAMNet aus
dem Analogausgang, Kennwerte nach **TA Lärm / DIN 45645-1**, Live-Dashboard und
PDF-Protokoll.

## Verkabelung

```
PeakTech 8005
├── USB-Buchse ──(USB-Kabel)──> Pi USB  => /dev/ttyUSB0
│      liefert kalibrierte dB-Werte (CEM-DT-885x-Protokoll, 9600 8N1,
│      0xA5-Pakete, ~20 Hz — Gerät streamt von selbst, kein Kommando nötig)
└── AC/DC-OUTPUT-Buchse ──(Klinke-auf-Cinch)──> USB-Audiointerface ──> Pi USB
       liefert das Analogsignal für die Klassifizierung   => ALSA card "CODEC"  (optional)
```

Der Audio-Pfad (Klassifizierung) ist optional — ohne Interface läuft die
Messung nur über den digitalen USB-Stream.

**Fallback ohne serielle Verbindung:** `laermlogger calibrate` bestimmt den
Offset zwischen Audio-dBFS und der Geräteanzeige; danach schätzt das System
die Pegel aus dem Audiosignal (Messbereich am Gerät fixieren!). Das Protokoll
kennzeichnet solche Messungen.

Wichtig am Gerät:
- Frequenzbewertung **dBA**, Zeitbewertung **Fast** einstellen.
- Messbereich möglichst fest wählen (nicht Auto), damit der Audio-Pegel stabil bleibt.

## Installation

```bash
sudo apt install libportaudio2 libpango-1.0-0 libpangocairo-1.0-0 libcairo2 libgdk-pixbuf-2.0-0
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Für die Audio-Ereignisclips wird **ffmpeg** benötigt (`sudo apt install ffmpeg`).

YAMNet-Modell (nicht im Git, weil groß) einmalig laden:

```bash
curl -sL "https://www.kaggle.com/api/v1/models/google/yamnet/tfLite/classification-tflite/1/download" | tar xz -C /tmp
cp /tmp/1.tflite models/yamnet.tflite
curl -sL -o models/yamnet_class_map.csv "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
```

Das Lärmquellen-Mapping ist in `models/quellen_mapping.yaml` anpassbar.

## Funktionen

- **Live-Dashboard** mit Pegelverlauf, laufendem LAeq/Perzentilen und erkannter Lärmquelle.
- **Audio-Ereignisse:** Bei jeder Pegelspitze (Schwelle `events.threshold_db`) wird ein
  **MP3-Clip** (Sekunden davor + danach) gespeichert und ist im Dashboard direkt anhörbar —
  so hörst du, *was* den Peak verursacht hat.
- **PDF-Protokoll** nach TA Lärm / DIN 45645-1 mit Kennwerten, Beurteilungspegel Tag/Nacht,
  **Lärmkarte (Stunde×Tag-Heatmap)**, **Tagesverläufen**, Quellen-Zeitanteilen und Clip-Liste.
- **Session-Liste** im Dashboard mit direkten PDF-/CSV-Download-Links.

## Bedienung

```bash
.venv/bin/laermlogger scan-hardware        # Geräte prüfen
.venv/bin/laermlogger dump                 # SL322-Protokoll verifizieren (Gerät einschalten!)
.venv/bin/laermlogger test-audio           # Audiopegel/Clipping prüfen
.venv/bin/laermlogger dashboard            # Live-Dashboard auf http://<pi>:8000
.venv/bin/laermlogger record --location "Schlafzimmer, Fenster gekippt"   # ohne Dashboard
.venv/bin/laermlogger report latest        # PDF-Protokoll
.venv/bin/laermlogger export latest        # CSV + JSON
```

Sessions liegen als SQLite-Dateien in `data/`.

## Architektur: zwei Prozesse (entkoppelt)

- **Mess-Daemon** (`laermlogger measure`) — führt die Messung, schreibt SQLite + Clips
  und `data/status.json`. Stabil, wird selten neu gestartet.
- **Dashboard** (`laermlogger dashboard`) — Weboberfläche, liest `status.json` und
  steuert über `data/control.json`. **Frei neustartbar, ohne die Messung zu unterbrechen.**

So kann man am Dashboard weiterentwickeln/aktualisieren, während eine Langzeitmessung läuft.

## Workflow

1. **Sammeln:** Messung im Dashboard starten (Schwelle + optional „Tageswechsel um
   Mitternacht" wählen). Läuft im Hintergrund, tage-/wochenlang.
2. **Auswerten:** Im Dashboard einen **Zeitraum** wählen → Kennwerte + Verlauf ansehen,
   die **Clips anhören & benennen** (labeln), bei Bedarf **trainieren**.
3. **Protokoll:** Im Zeitraum **PDF erzeugen** — nutzt deine Labels als Quellen.

## Dauerbetrieb / Langzeitmessung (z.B. eine Woche)

```bash
sudo cp systemd/laermlogger-measure.service systemd/laermlogger-dashboard.service /etc/systemd/system/
sudo systemctl enable --now laermlogger-measure laermlogger-dashboard
```

Beide starten automatisch mit dem Pi und laufen ohne offene SSH-/Browser-Sitzung weiter.
Dashboard aktualisieren ohne die Messung zu stören: `sudo systemctl restart laermlogger-dashboard`.

**Wichtig am Messgerät für eine Wochenmessung:**
- **Auto-Power-Off deaktivieren:** Gerät aus, FAST/SLOW-Taste halten und dabei einschalten
  (das Uhr-Symbol darf nicht erscheinen). Sonst schaltet sich das Gerät nach 30 min ab.
- **Netzteil anschließen** — Batterie hält keine Woche.
- **Messbereich fest** einstellen (z.B. 50–100), nicht Auto.

Die Software selbst ist wochentauglich: LAeq/Lmin/Lmax werden fortlaufend berechnet
(kein RAM-Wachstum), Perzentile live auf einem gleitenden Fenster; das finale Protokoll
rechnet exakt über den gesamten Zeitraum aus der SQLite-Datenbank.

## Konfiguration

`config.json` im Projektverzeichnis überschreibt Defaults aus
`laermlogger/config.py` — u.a. Richtwerte (`rating.limit_day_db` /
`limit_night_db`, Default: allgemeines Wohngebiet 55/40 dB), Ruhezeiten,
Zuschläge, serieller Port, Audio-Device.

## Grenzen

- SL322 ist Klasse 2; für behördlich verwertbare Messungen ist Klasse 1 nötig.
- LAeq wird aus ~20-Hz-Fast-Stützstellen energetisch gemittelt (Näherung).
- Impuls-/Tonzuschläge werden automatisch detektiert und ersetzen keine
  gutachterliche Bewertung.
