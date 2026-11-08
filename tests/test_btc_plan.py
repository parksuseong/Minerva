import math
from datetime import date, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from minerva.main import app
from minerva.strategy import btc_plan
from minerva.strategy.indicators import Bar

client = TestClient(app)


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


def test_size_endpoint():
    q = {"side": "long", "entry": 100000, "stop": 98000, "equity": 10000, "risk_pct": 1}
    got = client.get("/api/btc/size", params=q)
    assert got.status_code == 200 and got.json()["qty"] == 0.047
    bad = client.get("/api/btc/size", params={**q, "stop": 101000})
    assert bad.status_code == 400


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
    p = btc_plan.today(frames, daily, at, daily, 120.0)
    assert p["big_move"] and p["t6d"] == 1 and p["t6d_label"] == "롱 우위"
    assert p["t6d_since"] == at[391].strftime("%Y-%m-%d")
    assert p["stops"]["long"]["atr"] == pytest.approx(120 - p["atr_d"], abs=0.11)
    assert p["stops"]["long"]["swing"] < 120 < p["stops"]["short"]["atr"]


def test_brief_text_says_direction_is_coin_flip():
    data = {
        "generated_at": "2026-11-09T08:30:00+09:00",
        "plan": {
            "price": 80000.0,
            "big_move": True,
            "t6d_label": "숏 우위",
            "t6d_since": "2026-10-23",
            "atr_d": 2285.5,
            "atr_d_pct": 2.86,
        },
    }
    text = btc_plan.brief_text(data)
    assert "큰 움직임 가능성 높음 (방향은 반반)" in text
    assert "숏 우위" in text and "2.86%" in text and "80,000.0" in text
