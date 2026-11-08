import math
from datetime import date, datetime, timedelta

import pytest

from minerva.strategy import btc_plan
from minerva.strategy.indicators import Bar


def _series(n=400):
    c = [100 + 20 * math.sin(i / 9) + 8 * math.sin(i / 2.3) + i * 0.05 for i in range(n)]
    h = [x + 2 + abs(math.sin(i)) for i, x in enumerate(c)]
    lo = [x - 2 - abs(math.cos(i)) for i, x in enumerate(c)]
    return h, lo, c


# 백테스트 원본(pine_ports.s6_trend_dev, numpy)이 같은 입력에서 낸 신호
T6D_EXPECTED = {
    23: -1, 47: 1, 50: -1, 55: 1, 80: -1, 104: 1, 109: -1, 112: 1, 137: -1,
    161: 1, 195: -1, 218: 1, 252: -1, 276: 1, 309: -1, 333: 1, 367: -1, 391: 1,
}  # fmt: skip


def test_trend_dev_matches_numpy_original():
    sig = btc_plan.trend_dev(*_series())
    assert {i: v for i, v in enumerate(sig) if v} == T6D_EXPECTED


def test_t6d_state_is_last_signal_direction():
    h, lo, c = _series()
    assert btc_plan.t6d_state(h, lo, c) == (1, 391)
    assert btc_plan.t6d_state(h[:380], lo[:380], c[:380]) == (-1, 367)
    assert btc_plan.t6d_state(h[:20], lo[:20], c[:20]) == (0, None)


def test_atr_is_wilder_seeded_with_mean():
    h, lo, c = [10.0] * 20, [8.0] * 20, [9.0] * 20
    out = btc_plan.atr(h, lo, c, 14)
    assert math.isnan(out[12]) and out[13] == pytest.approx(2.0) and out[-1] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "ways,want",
    [
        (("하방", "하방", "하방"), True),
        (("하방", "하방", "중립"), False),
        (("상방", "상방", "상방"), False),
        (("하방", "상방", "하방"), False),
    ],
)
def test_big_move_needs_daily_4h_15m_all_down(ways, want):
    frames = [{"frame": k, "direction": w} for k, w in zip(("1d", "4h", "15m"), ways)]
    frames.append({"frame": "1w", "direction": "상방"})
    assert btc_plan.big_move(frames) is want


def test_last_pivot_needs_two_bars_each_side():
    lows = [5, 4, 3, 4, 5, 2, 3]
    assert btc_plan.last_pivot(lows, low=True) == 2  # 2는 오른쪽 봉이 하나뿐이라 미확정
    highs = [1, 2, 5, 2, 1, 9, 8]
    assert btc_plan.last_pivot(highs, low=False) == 2


def test_size_long_risks_one_percent_with_fees():
    r = btc_plan.size_position("long", 100_000, 98_000, 10_000, 1.0)
    per = 2000 + 0.0005 * 198_000
    assert r["qty"] == math.floor(100 / per * 1000) / 1000 == 0.047
    assert r["loss_at_stop"] == pytest.approx(0.047 * per, abs=0.01)
    assert r["loss_pct"] <= 1.0
    assert r["notional"] == pytest.approx(4700)
    assert r["leverage"] == pytest.approx(0.47)
    assert r["r_levels"] == {"1R": 102_000, "2R": 104_000, "3R": 106_000}
    assert r["breakeven_stop"] == pytest.approx(100_100.1, abs=0.1)
    assert not r["capped"] and r["warnings"] == []


def test_size_short_mirrors_levels():
    r = btc_plan.size_position("short", 100_000, 101_000, 10_000, 0.5)
    assert r["r_levels"]["2R"] == 98_000
    assert r["breakeven_stop"] < 100_000
    assert r["loss_pct"] <= 0.5


def test_size_caps_at_five_times_and_reports_reduced_risk():
    # 손절 0.2%면 2% 위험은 10배 가까이 필요 → 5배로 줄인다
    r = btc_plan.size_position("long", 100_000, 99_800, 1_000, 2.0)
    assert r["capped"] and r["leverage_wanted"] > 5
    assert r["qty"] == 0.05 and r["leverage"] == pytest.approx(5.0)
    assert r["loss_pct"] < 2.0
    assert "5배로 줄였어요" in r["warnings"][0]


def test_size_cap_uses_available_margin():
    r = btc_plan.size_position("long", 100_000, 99_800, 1_000, 2.0, available=500)
    assert r["qty"] == 0.025


def test_size_warns_below_exchange_minimum():
    r = btc_plan.size_position("long", 100_000, 90_000, 500, 0.5)
    assert r["qty"] == 0.0 and "0.001" in r["warnings"][0]


@pytest.mark.parametrize(
    "side,stop", [("long", 101_000), ("short", 99_000), ("long", 100_000), ("up", 99_000)]
)
def test_size_rejects_stop_on_wrong_side(side, stop):
    with pytest.raises(ValueError):
        btc_plan.size_position(side, 100_000, stop, 1_000)


def _bars(h, lo, c):
    start = date(2025, 1, 1)
    return [
        Bar(start + timedelta(days=i), x, a, b, x, 1.0) for i, (a, b, x) in enumerate(zip(h, lo, c))
    ]


def test_today_builds_plan():
    h, lo, c = _series()
    daily = _bars(h, lo, c)
    at = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(len(daily))]
    frames = [
        {"frame": "15m", "direction": "하방"},
        {"frame": "4h", "direction": "하방"},
        {"frame": "1d", "direction": "하방"},
    ]
    intraday = {"15m": daily, "30m": daily[:300]}
    p = btc_plan.today(frames, daily, at, intraday, {"1h": "상방", "4h": "하방"}, 120.0)
    assert p["big_move"] and p["t6d"] == 1 and p["t6d_label"] == "롱 우위"
    assert p["t6d_since"] == at[391].strftime("%Y-%m-%d")
    assert p["ways"] == {"1h": "상방", "4h": "하방"}
    assert p["align"] == {"long": "불일치", "short": "불일치"}
    assert set(p["tf"]) == {"15m", "30m"}
    f = p["tf"]["15m"]
    assert f["stops"]["long"]["atr1"] == pytest.approx(120 - f["atr"], abs=0.11)
    assert f["stops"]["long"]["atr15"] == pytest.approx(120 - 1.5 * f["atr"], abs=0.11)
    assert f["stops"]["short"]["atr15"] > f["stops"]["short"]["atr1"] > 120
    assert f["prev_low"] == round(lo[-1], 1) and f["prev_high"] == round(h[-1], 1)


def test_frame_plan_stops_use_that_frame():
    # 직전 봉 저가 99, 고가 101, 스윙 저점(좌우 2봉) 97, 스윙 고점 104
    lows = [100] * 15 + [98, 97, 98, 99, 99]
    highs = [102] * 15 + [103, 104, 103, 102, 101]
    bars = [Bar(date(2025, 1, 1), 100, hi, lo, 100, 1) for hi, lo in zip(highs, lows)]
    f = btc_plan.frame_plan(bars, 100.0)
    assert f["stops"]["long"]["swing"] == 97 and f["stops"]["long"]["prev"] == 99
    assert f["stops"]["short"]["swing"] == 104 and f["stops"]["short"]["prev"] == 101
    assert f["prev_low"] == 99 and f["prev_high"] == 101
    # 현재가가 이미 직전 저가 아래면 롱 손절 후보에서 뺀다
    assert btc_plan.frame_plan(bars, 98.5)["stops"]["long"]["prev"] is None


@pytest.mark.parametrize(
    "side,ways,want",
    [
        ("long", {"1h": "상방", "4h": "상방"}, "일치"),
        ("long", {"1h": "상방", "4h": "중립"}, "일부"),
        ("long", {"1h": "하방", "4h": "상방"}, "불일치"),
        ("short", {"1h": "하방", "4h": "하방"}, "일치"),
        ("short", {"1h": "중립", "4h": "상방"}, "불일치"),
    ],
)
def test_align_with_1h_and_4h(side, ways, want):
    assert btc_plan.align(side, ways) == want


def test_size_warns_when_fees_eat_a_narrow_stop():
    # 15분봉 손절 0.3%: 왕복 수수료 약 0.1%가 손절 폭의 33%
    r = btc_plan.size_position("long", 100_000, 99_700, 10_000, 1.0)
    assert r["fee_share_pct"] == pytest.approx(99.85 / 399.85 * 100, abs=0.1)
    assert any("손절 폭이 좁아 수수료가 손실의 25%" in w for w in r["warnings"])
    wide = btc_plan.size_position("long", 100_000, 99_000, 10_000, 1.0)
    assert not any("수수료" in w for w in wide["warnings"])
