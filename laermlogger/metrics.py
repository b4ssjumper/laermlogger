"""Akustische Kennwerte nach TA Lärm / DIN 45645-1.

Eingang ist die Zeitreihe der A-bewerteten Fast-Pegel aus dem SL322
(~20 Hz Stützstellen). Alle energetischen Mittelungen arbeiten auf
10^(L/10)-Basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

import numpy as np

from .config import RatingConfig

PERCENTILES = (1, 5, 10, 50, 90, 95)


def laeq(levels_db: np.ndarray) -> float:
    """Energieäquivalenter Dauerschallpegel."""
    if len(levels_db) == 0:
        return float("nan")
    return float(10 * np.log10(np.mean(10 ** (np.asarray(levels_db) / 10))))


def percentile_levels(levels_db: np.ndarray) -> dict[str, float]:
    """Überschreitungspegel L1..L95 (L1 = in 1 % der Zeit überschritten)."""
    if len(levels_db) == 0:
        return {f"L{p}": float("nan") for p in PERCENTILES}
    return {
        f"L{p}": float(np.percentile(levels_db, 100 - p)) for p in PERCENTILES
    }


@dataclass
class IntervalMetrics:
    """Kennwerte eines Auswerte-Intervalls."""

    start: datetime
    end: datetime
    n_samples: int
    laeq_db: float
    lafmax_db: float
    lafmin_db: float
    percentiles: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_series(cls, timestamps: np.ndarray, levels: np.ndarray,
                    start: datetime, end: datetime) -> "IntervalMetrics":
        return cls(
            start=start, end=end, n_samples=len(levels),
            laeq_db=laeq(levels),
            lafmax_db=float(np.max(levels)) if len(levels) else float("nan"),
            lafmin_db=float(np.min(levels)) if len(levels) else float("nan"),
            percentiles=percentile_levels(levels),
        )


@dataclass
class RatingResult:
    """Beurteilungspegel-Ergebnis für Tag oder Nacht."""

    period: str                    # "Tag" / "Nacht"
    laeq_db: float
    impulse_surcharge_db: float    # KI
    tonal_surcharge_db: float      # KT
    quiet_time_share: float        # Anteil der Samples in Ruhezeiten (nur Tag)
    quiet_surcharge_effective_db: float
    rating_level_db: float         # Lr
    limit_db: float
    exceeds_limit: bool
    note: str = ""


def _in_quiet_hours(dt: datetime, cfg: RatingConfig) -> bool:
    t = dt.time()
    for start_s, end_s in cfg.quiet_hours:
        if time.fromisoformat(start_s) <= t < time.fromisoformat(end_s):
            return True
    return False


def _is_day(dt: datetime, cfg: RatingConfig) -> bool:
    return cfg.day_start_time() <= dt.time() < cfg.night_start_time()


def split_day_night(timestamps: np.ndarray, levels: np.ndarray,
                    cfg: RatingConfig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Zeitreihe in Tag- (06–22) und Nacht-Anteile (22–06) trennen."""
    dts = [datetime.fromtimestamp(ts) for ts in timestamps]
    day_mask = np.array([_is_day(dt, cfg) for dt in dts], dtype=bool)
    return {
        "Tag": (timestamps[day_mask], levels[day_mask]),
        "Nacht": (timestamps[~day_mask], levels[~day_mask]),
    }


def loudest_hour_laeq(timestamps: np.ndarray, levels: np.ndarray) -> tuple[float, datetime | None]:
    """LAeq der lautesten vollen Stunde (TA Lärm: nachts maßgeblich)."""
    if len(levels) == 0:
        return float("nan"), None
    hours: dict[datetime, list[int]] = {}
    for i, ts in enumerate(timestamps):
        h = datetime.fromtimestamp(ts).replace(minute=0, second=0, microsecond=0)
        hours.setdefault(h, []).append(i)
    best_level, best_hour = -np.inf, None
    for hour, idx in hours.items():
        l = laeq(levels[np.asarray(idx)])
        if l > best_level:
            best_level, best_hour = l, hour
    return float(best_level), best_hour


def rate_period(period: str, timestamps: np.ndarray, levels: np.ndarray,
                cfg: RatingConfig, impulse_detected: bool,
                tonal_detected: bool) -> RatingResult:
    """Beurteilungspegel Lr für Tag oder Nacht bilden.

    Vereinfachte, aber norm-orientierte Umsetzung:
    - Nacht: LAeq der lautesten vollen Stunde ist maßgeblich.
    - Tag: LAeq über die Messzeit; Ruhezeiten-Zuschlag anteilig
      (energetisch über die betroffenen Samples).
    - KI/KT als pauschale Zuschläge, wenn die Audio-Analyse Impuls-
      bzw. Tonhaltigkeit erkannt hat.
    """
    note_parts = []
    if period == "Nacht":
        base, hour = loudest_hour_laeq(timestamps, levels)
        if hour is not None:
            note_parts.append(f"lauteste Stunde ab {hour:%H:%M}")
        quiet_share = 0.0
        quiet_eff = 0.0
        limit = cfg.limit_night_db
    else:
        dts = [datetime.fromtimestamp(ts) for ts in timestamps]
        quiet_mask = np.array([_in_quiet_hours(dt, cfg) for dt in dts], dtype=bool) \
            if len(levels) else np.zeros(0, dtype=bool)
        quiet_share = float(np.mean(quiet_mask)) if len(levels) else 0.0
        if quiet_share > 0:
            # Ruhezeiten-Samples energetisch um den Zuschlag anheben
            boosted = np.array(levels, dtype=float)
            boosted[quiet_mask] += cfg.quiet_bonus_db
            base = laeq(boosted)
            note_parts.append(f"Ruhezeitenanteil {quiet_share:.0%} (+{cfg.quiet_bonus_db:g} dB)")
        else:
            base = laeq(levels)
        quiet_eff = base - laeq(levels) if quiet_share > 0 and len(levels) else 0.0
        limit = cfg.limit_day_db

    ki = cfg.impulse_surcharge_db if impulse_detected else 0.0
    kt = cfg.tonal_surcharge_db if tonal_detected else 0.0
    if impulse_detected:
        note_parts.append(f"Impulszuschlag KI={ki:g} dB")
    if tonal_detected:
        note_parts.append(f"Tonzuschlag KT={kt:g} dB")

    lr = base + ki + kt
    return RatingResult(
        period=period, laeq_db=base,
        impulse_surcharge_db=ki, tonal_surcharge_db=kt,
        quiet_time_share=quiet_share,
        quiet_surcharge_effective_db=quiet_eff,
        rating_level_db=lr, limit_db=limit,
        exceeds_limit=bool(lr > limit) if not np.isnan(lr) else False,
        note="; ".join(note_parts),
    )


@dataclass
class SessionMetrics:
    """Gesamtauswertung einer Messsession."""

    overall: IntervalMetrics
    day: RatingResult | None
    night: RatingResult | None
    timeline_minutes: list[IntervalMetrics] = field(default_factory=list)


def evaluate_session(timestamps: np.ndarray, levels: np.ndarray,
                     cfg: RatingConfig, impulse_detected: bool = False,
                     tonal_detected: bool = False,
                     timeline_step_s: int = 60) -> SessionMetrics:
    """Komplette Session auswerten: Gesamt, Tag/Nacht-Beurteilung, Minuten-Zeitreihe."""
    timestamps = np.asarray(timestamps, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if len(timestamps) == 0:
        raise ValueError("keine Pegel-Samples in der Session")

    start = datetime.fromtimestamp(float(timestamps[0]))
    end = datetime.fromtimestamp(float(timestamps[-1]))
    overall = IntervalMetrics.from_series(timestamps, levels, start, end)

    parts = split_day_night(timestamps, levels, cfg)
    day = night = None
    d_ts, d_lv = parts["Tag"]
    if len(d_lv):
        day = rate_period("Tag", d_ts, d_lv, cfg, impulse_detected, tonal_detected)
    n_ts, n_lv = parts["Nacht"]
    if len(n_lv):
        night = rate_period("Nacht", n_ts, n_lv, cfg, impulse_detected, tonal_detected)

    # Minuten-Zeitreihe für Diagramm/Protokoll
    timeline = []
    t0 = float(timestamps[0])
    t_end = float(timestamps[-1])
    t = t0
    while t < t_end:
        mask = (timestamps >= t) & (timestamps < t + timeline_step_s)
        if mask.any():
            timeline.append(IntervalMetrics.from_series(
                timestamps[mask], levels[mask],
                datetime.fromtimestamp(t),
                datetime.fromtimestamp(min(t + timeline_step_s, t_end)),
            ))
        t += timeline_step_s

    return SessionMetrics(overall=overall, day=day, night=night,
                          timeline_minutes=timeline)
