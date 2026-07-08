"""Lärmmessprotokoll-Erzeugung aus einer Session-SQLite.

build_report(db_path, cfg) -> PDF-Pfad
export_csv / export_json   -> Roh-Export
"""

from __future__ import annotations

import base64
import csv
import io
import json
import logging
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import Config
from ..metrics import SessionMetrics, evaluate_session

log = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "template.html"


def _load_session(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    try:
        meta_row = conn.execute(
            "SELECT started_at, ended_at, device, location, operator, notes "
            "FROM session WHERE id = 1"
        ).fetchone()
        spl = conn.execute(
            "SELECT ts, db, weighting, time_const, range_db FROM spl_samples ORDER BY ts"
        ).fetchall()
        cls = conn.execute(
            "SELECT ts, category, top_classes, impulsive, tonal, tonal_freq_hz, audio_dbfs "
            "FROM classifications ORDER BY ts"
        ).fetchall()
    finally:
        conn.close()
    if not meta_row:
        raise ValueError(f"{db_path}: keine Session-Metadaten")
    return {
        "meta": {
            "started_at": meta_row[0], "ended_at": meta_row[1],
            "device": meta_row[2], "location": meta_row[3],
            "operator": meta_row[4], "notes": meta_row[5],
        },
        "spl": spl,
        "classifications": cls,
    }


def _level_chart_png(timestamps: np.ndarray, levels: np.ndarray,
                     metrics: SessionMetrics) -> str:
    """Pegel-Zeitverlauf als base64-PNG (für Einbettung ins PDF)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dts = [datetime.fromtimestamp(t) for t in timestamps]
    fig, ax = plt.subplots(figsize=(9.5, 3.2), dpi=140)
    ax.plot(dts, levels, lw=0.4, color="#1f77b4", alpha=0.7, label="LAF (Messgerät)")
    if metrics.timeline_minutes:
        mt = [m.start for m in metrics.timeline_minutes]
        mv = [m.laeq_db for m in metrics.timeline_minutes]
        ax.plot(mt, mv, lw=1.6, color="#d62728", label="LAeq (1 min)")
    ax.axhline(metrics.overall.laeq_db, color="#2ca02c", lw=1.2, ls="--",
               label=f"LAeq gesamt = {metrics.overall.laeq_db:.1f} dB")
    ax.set_ylabel("Schalldruckpegel dB(A)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _fig_to_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _hourly_laeq_grid(timestamps: np.ndarray, levels: np.ndarray):
    """LAeq je (Kalendertag, Stunde) — Grundlage für Heatmap & Tagesstreifen."""
    dts = np.array([datetime.fromtimestamp(t) for t in timestamps])
    days = sorted({d.date() for d in dts})
    day_idx = {d: i for i, d in enumerate(days)}
    energy = np.full((len(days), 24), np.nan)
    esum = np.zeros((len(days), 24))
    cnt = np.zeros((len(days), 24))
    for d, lv in zip(dts, levels):
        r = day_idx[d.date()]
        esum[r, d.hour] += 10 ** (lv / 10)
        cnt[r, d.hour] += 1
    mask = cnt > 0
    energy[mask] = 10 * np.log10(esum[mask] / cnt[mask])
    return days, energy


def _heatmap_png(timestamps: np.ndarray, levels: np.ndarray,
                 limit_night: float) -> str:
    """Heatmap Stunde (x) × Tag (y), Farbe = LAeq der Stunde."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days, grid = _hourly_laeq_grid(timestamps, levels)
    h = max(1.6, 0.5 * len(days) + 1.0)
    fig, ax = plt.subplots(figsize=(9.6, h), dpi=140)
    vmin = max(25, np.nanmin(grid) - 2) if np.isfinite(grid).any() else 30
    vmax = np.nanmax(grid) + 2 if np.isfinite(grid).any() else 70
    im = ax.imshow(grid, aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_yticks(range(len(days)))
    ax.set_yticklabels([d.strftime("%a %d.%m.") for d in days])
    ax.set_xlabel("Uhrzeit (Stunde)")
    # Nachtbereich markieren (0-6 und 22-24)
    for hx in list(range(0, 6)) + list(range(22, 24)):
        ax.axvline(hx - 0.5, color="white", lw=0.3, alpha=0.25)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("LAeq je Stunde  dB(A)")
    ax.set_title("Lärmkarte: Wann ist es laut? (Stunde × Tag)", fontsize=10, pad=8)
    return _fig_to_b64(fig)


def _daily_strips_png(timestamps: np.ndarray, levels: np.ndarray,
                      limit_day: float, limit_night: float) -> str:
    """Ein 24h-Streifen pro Tag mit Pegelverlauf, Nacht-Schattierung, Richtwerten."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dts = np.array([datetime.fromtimestamp(t) for t in timestamps])
    days = sorted({d.date() for d in dts})
    n = len(days)
    fig, axes = plt.subplots(n, 1, figsize=(9.6, 1.5 * n + 0.5), dpi=140,
                             squeeze=False, sharex=True)
    for i, day in enumerate(days):
        ax = axes[i][0]
        m = np.array([d.date() == day for d in dts])
        hours = np.array([d.hour + d.minute / 60 + d.second / 3600
                          for d in dts[m]])
        lv = levels[m]
        order = np.argsort(hours)
        ax.axvspan(0, 6, color="#334", alpha=0.15)
        ax.axvspan(22, 24, color="#334", alpha=0.15)
        ax.plot(hours[order], lv[order], lw=0.5, color="#1f77b4")
        ax.axhline(limit_day, color="#2ca02c", lw=0.8, ls="--")
        ax.axhline(limit_night, color="#d62728", lw=0.8, ls=":")
        ax.set_ylabel(day.strftime("%a\n%d.%m."), fontsize=8, rotation=0,
                      ha="right", va="center")
        ax.set_xlim(0, 24); ax.set_ylim(25, max(75, lv.max() + 5))
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(labelsize=7)
    axes[-1][0].set_xticks(range(0, 25, 2))
    axes[-1][0].set_xlabel("Uhrzeit")
    axes[0][0].set_title(
        f"Tagesverläufe  (grün ⋯ Tagesrichtwert {limit_day:.0f} dB, "
        f"rot ⋯ Nachtrichtwert {limit_night:.0f} dB, grau = Nacht)",
        fontsize=9, pad=6)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _detect_surcharges(cls_rows: list, min_share: float = 0.02,
                       min_windows: int = 10) -> tuple[bool, bool]:
    """Impuls-/Tonhaltigkeit der Session bestimmen.

    Ein Zuschlag gilt erst, wenn ein nennenswerter Anteil der Fenster
    (>= min_share oder >= min_windows) auffällig war — nicht bei einem
    einzelnen Ausreißer-Fenster.
    """
    n = len(cls_rows)
    if n == 0:
        return False, False
    n_imp = sum(1 for r in cls_rows if r[3])
    n_ton = sum(1 for r in cls_rows if r[4])
    impulse = n_imp >= min_windows and n_imp / n >= min_share
    tonal = n_ton >= min_windows and n_ton / n >= min_share
    return impulse, tonal


def _category_shares(cls_rows: list) -> list[dict]:
    """Zeitanteile der Lärmquellen aus den Klassifikationsfenstern."""
    counts = Counter(row[1] for row in cls_rows)
    total = sum(counts.values()) or 1
    return [
        {"category": cat, "share": n / total, "windows": n}
        for cat, n in counts.most_common()
    ]


def _exceedance_events(timestamps: np.ndarray, levels: np.ndarray,
                       threshold_db: float, min_gap_s: float = 5.0) -> list[dict]:
    """Zusammenhängende Überschreitungen des Schwellenwerts finden."""
    events = []
    above = levels > threshold_db
    start_i = None
    last_above_ts = None
    for i, (ts, up) in enumerate(zip(timestamps, above)):
        if up:
            if start_i is None:
                start_i = i
            last_above_ts = ts
        elif start_i is not None and ts - last_above_ts > min_gap_s:
            seg = levels[start_i:i]
            events.append({
                "start": datetime.fromtimestamp(timestamps[start_i]),
                "duration_s": timestamps[i - 1] - timestamps[start_i],
                "max_db": float(seg.max()),
            })
            start_i = None
    if start_i is not None:
        seg = levels[start_i:]
        events.append({
            "start": datetime.fromtimestamp(timestamps[start_i]),
            "duration_s": timestamps[-1] - timestamps[start_i],
            "max_db": float(seg.max()),
        })
    return events


def session_levels(db_path: Path, buckets: int = 800) -> list[dict]:
    """Pegelverlauf einer (abgeschlossenen) Session, in ~buckets Zeit-Buckets
    aggregiert (MAX je Bucket). Für die interaktive Review-Ansicht."""
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT MIN(ts), MAX(ts) FROM spl_samples").fetchone()
        if not row or row[0] is None:
            return []
        t0, t1 = row
        bucket = max((t1 - t0) / buckets, 1e-6)
        rows = conn.execute(
            "SELECT AVG(ts), MAX(db) FROM spl_samples "
            "GROUP BY CAST((ts - ?) / ? AS INT) ORDER BY 1", (t0, bucket)
        ).fetchall()
    finally:
        conn.close()
    return [{"ts": r[0], "db": r[1]} for r in rows]


def session_summary(db_path: Path, cfg: Config) -> dict:
    """Kennwerte einer Session (LAeq, Perzentile, Tag/Nacht) als dict — ohne PDF."""
    data = _load_session(db_path)
    spl = data["spl"]
    meta = data["meta"]
    if not spl:
        return {"meta": meta, "n_samples": 0}
    ts = np.array([r[0] for r in spl])
    lv = np.array([r[1] for r in spl])
    imp, ton = _detect_surcharges(data["classifications"])
    m = evaluate_session(ts, lv, cfg.rating, imp, ton)
    events = _load_audio_events(db_path)
    return {
        "meta": meta,
        "started": meta["started_at"], "ended": meta["ended_at"],
        "duration_min": (ts[-1] - ts[0]) / 60,
        "n_samples": len(lv), "n_events": len(events),
        "laeq_db": round(m.overall.laeq_db, 1),
        "lafmax_db": round(m.overall.lafmax_db, 1),
        "lafmin_db": round(m.overall.lafmin_db, 1),
        "percentiles": {k: round(v, 1) for k, v in m.overall.percentiles.items()},
        "day": _rating_dict(m.day), "night": _rating_dict(m.night),
        "sources": _event_source_shares(db_path, cfg, events),
    }


def _rating_dict(r) -> dict | None:
    if r is None:
        return None
    return {"laeq_db": round(r.laeq_db, 1), "rating_level_db": round(r.rating_level_db, 1),
            "limit_db": r.limit_db, "exceeds_limit": r.exceeds_limit, "note": r.note}


def _load_audio_events(db_path: Path) -> list[dict]:
    """Aufgezeichnete Audio-Ereignisse (MP3-Clips) einer Session laden."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT ts, peak_db, category, mp3_path, custom_label "
            "FROM events ORDER BY peak_db DESC"
        ).fetchall()
    except sqlite3.OperationalError:
        try:
            rows = [(r[0], r[1], r[2], r[3], "") for r in conn.execute(
                "SELECT ts, peak_db, category, mp3_path FROM events "
                "ORDER BY peak_db DESC")]
        except sqlite3.OperationalError:
            return []
    finally:
        conn.close()
    return [{"start": datetime.fromtimestamp(r[0]), "peak_db": r[1],
             "category": r[2], "mp3": r[3], "custom_label": r[4]} for r in rows]


def _event_source_shares(db_path: Path, cfg: Config, events: list[dict]) -> list[dict]:
    """Lärmquellen-Verteilung aus den Ereignis-Clips mit der jeweils besten
    verfügbaren Bezeichnung: Nutzer-Label > trainiertes Modell > YAMNet-Kategorie."""
    from .. import custom_sounds

    if not events:
        return []
    user_labels = custom_sounds.labels_for_session(cfg, db_path.stem)
    counts: Counter = Counter()
    for e in events:
        best = (user_labels.get(e["mp3"]) or e.get("custom_label")
                or e.get("category") or "Sonstiges")
        counts[best] += 1
    total = sum(counts.values()) or 1
    return [{"category": c, "share": n / total, "windows": n}
            for c, n in counts.most_common()]


def build_report(db_path: Path, cfg: Config, out_path: Path | None = None) -> Path:
    """PDF-Lärmmessprotokoll für eine Session erzeugen."""
    from jinja2 import Template
    from weasyprint import HTML

    data = _load_session(db_path)
    spl = data["spl"]
    if not spl:
        raise ValueError("Session enthält keine Pegel-Samples")

    timestamps = np.array([r[0] for r in spl])
    levels = np.array([r[1] for r in spl])
    weightings = {r[2] for r in spl}
    ranges = {r[4] for r in spl}
    audio_fallback_used = "Audio-Schätzung" in ranges

    impulse_detected, tonal_detected = _detect_surcharges(data["classifications"])

    metrics = evaluate_session(timestamps, levels, cfg.rating,
                               impulse_detected, tonal_detected)

    # Ereignisliste: Überschreitungen des jeweils gültigen Richtwerts
    day_limit = cfg.rating.limit_day_db
    events = _exceedance_events(timestamps, levels, day_limit)

    # Aufgezeichnete Audio-Clips laden
    audio_events = _load_audio_events(db_path)
    # Lärmquellen bevorzugt aus den (ggf. gelabelten) Ereignis-Clips,
    # sonst aus der Sekunden-Klassifikation
    event_shares = _event_source_shares(db_path, cfg, audio_events)
    shares = event_shares if event_shares else _category_shares(data["classifications"])
    shares_from_events = bool(event_shares)

    # Große Visualisierungen nur bei ausreichend langem Zeitraum (> 1 h)
    span_s = float(timestamps[-1] - timestamps[0])
    heatmap_b64 = daily_strips_b64 = None
    if span_s > 3600:
        heatmap_b64 = _heatmap_png(timestamps, levels, cfg.rating.limit_night_db)
        daily_strips_b64 = _daily_strips_png(
            timestamps, levels, cfg.rating.limit_day_db, cfg.rating.limit_night_db)

    tmpl = Template(TEMPLATE_PATH.read_text())
    html = tmpl.render(
        meta=data["meta"],
        started=datetime.fromtimestamp(data["meta"]["started_at"]),
        ended=datetime.fromtimestamp(data["meta"]["ended_at"])
        if data["meta"]["ended_at"] else None,
        duration_min=(timestamps[-1] - timestamps[0]) / 60,
        weightings=", ".join(sorted(weightings)),
        ranges=", ".join(sorted(ranges)),
        m=metrics,
        cfg=cfg.rating,
        chart_b64=_level_chart_png(timestamps, levels, metrics),
        heatmap_b64=heatmap_b64,
        daily_strips_b64=daily_strips_b64,
        shares=shares,
        shares_from_events=shares_from_events,
        impulse_detected=impulse_detected,
        tonal_detected=tonal_detected,
        audio_fallback_used=audio_fallback_used,
        include_methodology_note=cfg.rating.include_methodology_note,
        events=events[:50],
        n_events_total=len(events),
        audio_events=audio_events,
        created=datetime.now(),
        version="0.1.0",
    )

    out_path = out_path or db_path.with_suffix(".pdf")
    HTML(string=html).write_pdf(out_path)
    log.info("Protokoll erzeugt: %s", out_path)
    return out_path


def export_csv(db_path: Path, out_path: Path | None = None) -> Path:
    data = _load_session(db_path)
    out_path = out_path or db_path.with_suffix(".csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "iso_time", "db", "weighting", "time_const", "range"])
        for ts, db, wg, tc, rg in data["spl"]:
            w.writerow([ts, datetime.fromtimestamp(ts).isoformat(), db, wg, tc, rg])
    return out_path


def export_json(db_path: Path, cfg: Config, out_path: Path | None = None) -> Path:
    data = _load_session(db_path)
    timestamps = np.array([r[0] for r in data["spl"]])
    levels = np.array([r[1] for r in data["spl"]])
    payload = {"meta": data["meta"], "n_samples": len(levels)}
    if len(levels):
        impulse, tonal = _detect_surcharges(data["classifications"])
        m = evaluate_session(timestamps, levels, cfg.rating, impulse, tonal)
        payload["metrics"] = {
            "laeq_db": m.overall.laeq_db,
            "lafmax_db": m.overall.lafmax_db,
            "lafmin_db": m.overall.lafmin_db,
            "percentiles": m.overall.percentiles,
            "day": vars(m.day) if m.day else None,
            "night": vars(m.night) if m.night else None,
        }
        payload["sources"] = _category_shares(data["classifications"])
    out_path = out_path or db_path.with_suffix(".json")
    out_path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    return out_path
