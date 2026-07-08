"""Tests für metrics.py gegen analytisch bekannte Signale."""

from datetime import datetime

import numpy as np
import pytest

from laermlogger.config import RatingConfig
from laermlogger.metrics import (
    evaluate_session,
    laeq,
    loudest_hour_laeq,
    percentile_levels,
    rate_period,
    split_day_night,
)


def ts_at(day: str, clock: str) -> float:
    return datetime.fromisoformat(f"{day}T{clock}").timestamp()


class TestLaeq:
    def test_constant_level(self):
        assert laeq(np.full(100, 60.0)) == pytest.approx(60.0)

    def test_energetic_mean_dominated_by_loud(self):
        # 50 % bei 60 dB, 50 % bei 90 dB -> LAeq = 90 + 10*log10(0.5) ≈ 86.99
        levels = np.array([60.0] * 500 + [90.0] * 500)
        expected = 10 * np.log10(0.5 * 10**6 + 0.5 * 10**9)
        assert laeq(levels) == pytest.approx(expected)
        assert laeq(levels) == pytest.approx(86.99, abs=0.01)

    def test_empty(self):
        assert np.isnan(laeq(np.array([])))


class TestPercentiles:
    def test_uniform_ramp(self):
        levels = np.linspace(40, 90, 10001)  # gleichverteilt 40..90
        p = percentile_levels(levels)
        assert p["L50"] == pytest.approx(65.0, abs=0.1)
        assert p["L95"] == pytest.approx(42.5, abs=0.1)   # in 95 % der Zeit überschritten
        assert p["L1"] == pytest.approx(89.5, abs=0.1)
        assert p["L90"] > p["L95"]
        assert p["L1"] > p["L50"]


class TestDayNightSplit:
    def test_split(self):
        cfg = RatingConfig()
        ts = np.array([
            ts_at("2026-07-06", "12:00"),   # Tag
            ts_at("2026-07-06", "23:00"),   # Nacht
            ts_at("2026-07-07", "03:00"),   # Nacht
            ts_at("2026-07-07", "06:00"),   # Tag (Grenze inklusiv)
            ts_at("2026-07-07", "21:59"),   # Tag
            ts_at("2026-07-07", "22:00"),   # Nacht (Grenze)
        ])
        levels = np.arange(6, dtype=float)
        parts = split_day_night(ts, levels, cfg)
        assert list(parts["Tag"][1]) == [0.0, 3.0, 4.0]
        assert list(parts["Nacht"][1]) == [1.0, 2.0, 5.0]


class TestLoudestHour:
    def test_picks_loudest(self):
        # Stunde 22: 50 dB, Stunde 23: 70 dB
        ts, lv = [], []
        for minute in range(60):
            ts.append(ts_at("2026-07-06", f"22:{minute:02d}"))
            lv.append(50.0)
            ts.append(ts_at("2026-07-06", f"23:{minute:02d}"))
            lv.append(70.0)
        level, hour = loudest_hour_laeq(np.array(ts), np.array(lv))
        assert level == pytest.approx(70.0)
        assert hour.hour == 23


class TestRating:
    def test_night_uses_loudest_hour(self):
        cfg = RatingConfig()
        ts, lv = [], []
        for minute in range(60):
            ts.append(ts_at("2026-07-06", f"23:{minute:02d}"))
            lv.append(35.0)
            ts.append(ts_at("2026-07-07", f"02:{minute:02d}"))
            lv.append(45.0)
        r = rate_period("Nacht", np.array(ts), np.array(lv), cfg,
                        impulse_detected=False, tonal_detected=False)
        assert r.laeq_db == pytest.approx(45.0)
        assert r.rating_level_db == pytest.approx(45.0)
        assert r.exceeds_limit  # 45 > 40 (allg. Wohngebiet nachts)

    def test_surcharges_added(self):
        cfg = RatingConfig()
        ts = np.array([ts_at("2026-07-07", "12:00") + i for i in range(100)])
        lv = np.full(100, 50.0)
        r = rate_period("Tag", ts, lv, cfg, impulse_detected=True, tonal_detected=True)
        assert r.rating_level_db == pytest.approx(50.0 + 3.0 + 3.0)
        # 50 + 3 + 3 = 56 > 55 -> erst die Zuschläge führen zur Überschreitung
        assert r.exceeds_limit

    def test_quiet_hours_boost(self):
        cfg = RatingConfig()
        # gleiche Pegel, komplett in Ruhezeit 06–07 -> Lr = LAeq + 6 dB
        ts = np.array([ts_at("2026-07-07", "06:30") + i for i in range(100)])
        lv = np.full(100, 50.0)
        r = rate_period("Tag", ts, lv, cfg, False, False)
        assert r.quiet_time_share == pytest.approx(1.0)
        assert r.rating_level_db == pytest.approx(56.0)

    def test_no_quiet_hours_no_boost(self):
        cfg = RatingConfig()
        ts = np.array([ts_at("2026-07-07", "12:00") + i for i in range(100)])
        r = rate_period("Tag", ts, np.full(100, 50.0), cfg, False, False)
        assert r.quiet_time_share == 0.0
        assert r.rating_level_db == pytest.approx(50.0)


class TestEvaluateSession:
    def test_full_session(self):
        cfg = RatingConfig()
        ts, lv = [], []
        base = ts_at("2026-07-07", "21:00")
        for i in range(7200):  # 21:00–23:00, 1 Hz
            ts.append(base + i)
            lv.append(55.0 if i < 3600 else 42.0)
        m = evaluate_session(np.array(ts), np.array(lv), cfg)
        assert m.day is not None and m.night is not None
        # Tag (21:00–22:00): 55 dB in Ruhezeit 20–22 -> +6
        assert m.day.laeq_db == pytest.approx(61.0, abs=0.01)
        assert m.night.laeq_db == pytest.approx(42.0, abs=0.01)
        assert len(m.timeline_minutes) == pytest.approx(120, abs=1)
        assert m.overall.lafmax_db == 55.0
        assert m.overall.lafmin_db == 42.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            evaluate_session(np.array([]), np.array([]), RatingConfig())
