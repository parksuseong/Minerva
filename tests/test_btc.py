from datetime import datetime

import pytest

from minerva.strategy import btc
from minerva.strategy.indicators import TMState


def _row(hour_ms, close, step=4 * 3600 * 1000):
    return [
        hour_ms,
        str(close),
        str(close + 2),
        str(close - 2),
        str(close),
        "1",
        hour_ms + step - 1,
    ]


def _rows(n, start=1_700_000_000_000, step=4 * 3600 * 1000, wave=True):
    closes = [100 + (i % 20 if wave else i) for i in range(n)]
    return [_row(start + i * step, c, step) for i, c in enumerate(closes)]


def test_parse_drops_candle_still_open():
    rows = _rows(3)
    now_ms = rows[-1][0] + 1000
    bars, opened = btc.parse(rows, now_ms)
    assert len(bars) == 2 and len(opened) == 2
    assert opened[0].utcoffset().total_seconds() == 9 * 3600


@pytest.mark.parametrize(
    "cci,rsi,want",
    [(10, 60, "상방"), (-10, 40, "하방"), (10, 40, "중립"), (-10, 60, "중립"), (None, 40, "중립")],
)
def test_direction_needs_cci_and_rsi_to_agree(cci, rsi, want):
    s = TMState(datetime(2026, 1, 1).date(), 100, 50, cci, rsi, 0, 0)
    assert btc.direction(s) == want


def test_last_signal_finds_latest_of_each_side():
    d = datetime(2026, 1, 1)
    day = d.date()
    states = [
        TMState(day, 1, 50, 0, 50, 2, 0),
        TMState(day, 2, 50, 0, 50, 0, 3),
        TMState(day, 3, 50, 0, 50, 0, 0),
    ]
    opened = [d.replace(hour=h) for h in (1, 5, 9)]
    assert btc.last_signal(states, opened, "long")["at"] == "2026-01-01 01:00"
    assert btc.last_signal(states, opened, "short")["level"] == 3


def test_summary_reports_both_sides():
    def frame(name, direction, lo, sh):
        return {"name": name, "direction": direction, "long_level": lo, "short_level": sh}

    got = btc.summary([frame("일봉", "상방", 1, 0), frame("주봉", "하방", 0, 2)])
    assert got == [
        "상방 1개(일봉), 하방 1개(주봉)예요.",
        "롱 신호: 일봉 1단.",
        "숏 신호: 주봉 2단.",
    ]


def test_build_uses_every_frame_and_only_closed_candles(monkeypatch):
    rows = _rows(100)
    monkeypatch.setattr(btc, "fetch", lambda iv: rows)
    monkeypatch.setattr(btc, "fetch_price", lambda: 105.0)
    now = datetime.fromtimestamp(rows[-1][0] / 1000 + 60, btc.KST)
    data = btc.build(now)
    assert data["plan"]["price"] == 105.0 and data["plan"]["t6d_label"] in (
        "롱 우위",
        "숏 우위",
        "중립",
    )
    assert [f["frame"] for f in data["frames"]] == ["15m", "4h", "1d", "1w", "1M"]
    assert data["frames"][0]["close"] == float(rows[-2][4])


def test_build_refuses_short_history(monkeypatch):
    monkeypatch.setattr(btc, "fetch", lambda iv: _rows(5))
    with pytest.raises(ValueError):
        btc.build(datetime(2030, 1, 1, tzinfo=btc.KST))


def _gate_frames(w15, w4, w1d, close=100.0):
    return [
        {"frame": "15m", "direction": w15, "close": close},
        {"frame": "4h", "direction": w4, "close": close},
        {"frame": "1d", "direction": w1d, "close": close},
        {"frame": "1w", "direction": "중립", "close": close},
    ]


def test_entry_gate_default_one_percent():
    g = btc.entry_gate(_gate_frames("하방", "하방", "상방"))
    assert (g["big"], g["pct"], g["long_max"], g["short_min"]) == (False, 1.0, 101.0, 99.0)


def test_entry_gate_widens_when_all_frames_agree():
    g = btc.entry_gate(_gate_frames("하방", "하방", "하방"))
    assert (g["big"], g["big_way"], g["pct"], g["long_max"], g["short_min"]) == (
        True,
        "하방",
        3.0,
        103.0,
        97.0,
    )


def test_entry_gate_neutral_is_not_big():
    assert btc.entry_gate(_gate_frames("중립", "중립", "중립"))["big"] is False
