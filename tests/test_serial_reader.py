"""Tests für den CEM-DT-885x-Decoder (PeakTech 8005), gegen synthetische Bytes."""

import pytest

from laermlogger.serial_reader import CemDecoder, _bcd_measurement


class TestBcd:
    def test_basic(self):
        assert _bcd_measurement(b"\x05\x74") == pytest.approx(57.4)

    def test_hundreds(self):
        assert _bcd_measurement(b"\x13\x00") == pytest.approx(130.0)

    def test_low(self):
        assert _bcd_measurement(b"\x03\x54") == pytest.approx(35.4)


def a5(token, *payload):
    return bytes([0xA5, token, *payload])


class TestDecoder:
    def test_single_measurement(self):
        d = CemDecoder()
        out = d.feed(a5(0x0D, 0x05, 0x74))
        assert len(out) == 1
        assert out[0].db == pytest.approx(57.4)
        assert out[0].weighting == "A"
        assert out[0].time_const == "F"

    def test_flags_applied_to_following_measurement(self):
        d = CemDecoder()
        stream = a5(0x03) + a5(0x1C, 0x00) + a5(0x4B) + a5(0x0D, 0x06, 0x33)
        out = d.feed(stream)
        assert len(out) == 1
        assert out[0].time_const == "S"     # 0x03 SLOW
        assert out[0].weighting == "C"      # 0x1C dBC
        assert out[0].range_db == "50-100"  # 0x4B

    def test_realistic_cycle(self):
        # Echter Gerätezyklus (aus dem Live-Mitschnitt); Flag-Tokens kommen
        # nach dem Messwert, daher zwei Zyklen füttern -> 2. Sample trägt Flags
        d = CemDecoder()
        cycle = (a5(0x08) + a5(0x0D, 0x03, 0x54) + a5(0x0C) + a5(0x1B, 0x00)
                 + a5(0x06, 0x01, 0x31, 0x03) + a5(0x4B) + a5(0x0E))
        out = d.feed(cycle + cycle)
        assert len(out) == 2
        assert out[0].db == pytest.approx(35.4)
        assert out[1].weighting == "A"
        assert out[1].range_db == "50-100"

    def test_multiple_measurements(self):
        d = CemDecoder()
        out = d.feed(a5(0x0D, 0x05, 0x00) + a5(0x0D, 0x05, 0x55))
        assert [s.db for s in out] == pytest.approx([50.0, 55.5])

    def test_byte_split_across_feeds(self):
        d = CemDecoder()
        assert d.feed(b"\xa5\x0d\x05") == []   # Paket unvollständig
        out = d.feed(b"\x74")                  # letztes Byte
        assert len(out) == 1 and out[0].db == pytest.approx(57.4)

    def test_hold_byte_resets(self):
        d = CemDecoder()
        # 0xFF mitten im Paket -> Reset, kein Sample
        assert d.feed(b"\xa5\x0d\xff") == []
        out = d.feed(a5(0x0D, 0x05, 0x00))
        assert len(out) == 1 and out[0].db == pytest.approx(50.0)

    def test_time_token_payload_skipped(self):
        # TOKEN_TIME (0x06) hat 3 Payload-Bytes, darf keinen Messwert erzeugen
        d = CemDecoder()
        out = d.feed(a5(0x06, 0x12, 0x34, 0x56) + a5(0x0D, 0x04, 0x00))
        assert [s.db for s in out] == pytest.approx([40.0])

    def test_implausible_value_dropped(self):
        d = CemDecoder()
        assert d.feed(a5(0x0D, 0x99, 0x99)) == []   # 999.9 > 140

    def test_recording_and_battery_flags(self):
        d = CemDecoder()
        d.feed(a5(0x0A))   # RECORDING_ON
        assert d.recording is True
        d.feed(a5(0x0F))   # BATTERY_LOW
        assert d.battery_low is True
        d.feed(a5(0x1A) + a5(0x1F))
        assert d.recording is False and d.battery_low is False
