from datetime import date, timedelta

from minerva.strategy import btc_outlook as bo
from minerva.strategy.indicators import Bar


def _bars(closes):
    d0 = date(2020, 1, 1)
    return [
        Bar(d0 + timedelta(days=i), c, c * 1.01, c * 0.99, c, 1.0) for i, c in enumerate(closes)
    ]


def _raw(trend):
    up = [100 + i for i in range(260)]
    down = [400 - i for i in range(260)]
    return {k: _bars(up if trend[k] == "up" else down) for k, _ in bo.FRAMES}


def test_all_up_is_strong_uptrend():
    raw = _raw({"4h": "up", "1d": "up", "1w": "up", "1M": "up"})
    out = bo.build(raw, 360.0)
    s = out["summary"]
    assert [f["direction"] for f in out["frames"]] == ["상방"] * 4
    assert s["tone"] == "up" and "강한 상승" in s["verdict"]
    assert all(c["price"] > 360 for c in s["resist"])
    assert all(c["price"] < 360 for c in s["support"])


def test_big_up_mid_down_is_pullback():
    raw = _raw({"4h": "down", "1d": "down", "1w": "up", "1M": "up"})
    s = bo.build(raw, 150.0)["summary"]
    assert s["big"].startswith("상방") and s["mid"].startswith("하방")
    assert "조정" in s["verdict"]


def test_levels_merge_same_price_and_sort():
    f = {
        "name": "일봉",
        "ma": {"20": {"value": 110.0}},
        "swing": {"high": 120.0, "low": 90.0},
    }
    g = dict(f, name="4시간봉", ma={}, swing={"high": 105.0, "low": 90.0})
    up, dn = bo.levels({"1d": f, "4h": g}, 100.0)
    assert [c["price"] for c in up] == [105.0, 110.0, 120.0]
    assert dn[0]["price"] == 90.0 and "4시간봉" in dn[0]["what"] and "일봉" in dn[0]["what"]


def test_short_history_skips_summary():
    raw = {k: _bars([100 + i for i in range(10)]) for k, _ in bo.FRAMES}
    out = bo.build(raw, 100.0)
    assert out["frames"] == [] and "summary" not in out
