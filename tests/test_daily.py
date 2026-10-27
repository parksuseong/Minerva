from datetime import date, datetime, timedelta, timezone

from minerva.strategy.daily import completed, fear_status, telegram_summary
from minerva.strategy.indicators import Bar


def bars_until(last: date, n: int = 3):
    return [
        Bar(day=last - timedelta(days=n - 1 - i), open=1, high=1, low=1, close=1) for i in range(n)
    ]


def test_us_bar_dropped_while_market_open():
    now = datetime(2026, 10, 27, 16, 20, tzinfo=timezone.utc)  # 뉴욕 12:20
    assert completed(bars_until(date(2026, 10, 27)), "US", now)[-1].day == date(2026, 10, 26)
    later = datetime(2026, 10, 27, 21, 0, tzinfo=timezone.utc)  # 뉴욕 17:00
    assert completed(bars_until(date(2026, 10, 27)), "US", later)[-1].day == date(2026, 10, 27)


def test_kr_bar_kept_after_close():
    now = datetime(2026, 10, 27, 7, 0, tzinfo=timezone.utc)  # 서울 16:00
    assert completed(bars_until(date(2026, 10, 27)), "KR", now)[-1].day == date(2026, 10, 27)
    early = datetime(2026, 10, 27, 1, 0, tzinfo=timezone.utc)  # 서울 10:00
    assert completed(bars_until(date(2026, 10, 27)), "KR", early)[-1].day == date(2026, 10, 26)


def test_crypto_drops_today_utc():
    now = datetime(2026, 10, 27, 3, 0, tzinfo=timezone.utc)
    assert completed(bars_until(date(2026, 10, 27)), "CRYPTO", now)[-1].day == date(2026, 10, 26)


def test_fear_signal_detected_on_entry_day():
    days = [date(2026, 1, 1) + timedelta(days=i) for i in range(3)]
    vix = [Bar(d, 1, 1, 1, c) for d, c in zip(days, [10.0, 10.0, 11.0])]
    vix3m = [Bar(d, 1, 1, 1, 10.0) for d in days]
    out = fear_status(vix, vix3m)
    assert out["last_signal"] == "2026-01-03" and out["state"].startswith("신호 발생")


def _data(**kw):
    base = {
        "generated_at": "2026-10-28T09:10",
        "vix": {"close": 18.4, "regime": "R2", "size_limit": "계획의 1/2", "ma50": 16.0},
        "fear": {"state": "없음", "premium": -0.08},
        "assets": [],
        "actions": [],
    }
    base.update(kw)
    return base


def test_telegram_without_actions_says_nothing_to_do():
    text = telegram_summary(_data())
    assert "오늘 할 일: 없음" in text and "R2" in text


def test_actions_from_entries_and_exits():
    from minerva.strategy.daily import make_actions, telegram_summary

    ver = {"long": {"label": "검증됨"}, "short": {"label": "미검증"}}
    asset = {
        "symbol": "SPY",
        "name": "SPY",
        "market": "US",
        "date": "2026-10-27",
        "verify": ver,
        "tm": {"state": "진입 신호", "ref_price": 100, "stop": 90},
        "short": {
            "state": "청산 신호",
            "reason": "익절 조건",
            "pnl": 0.03,
            "exit": "다음 거래일 시가 청산",
        },
    }
    inv = {
        "symbol": "QID",
        "name": "QID",
        "market": "US",
        "date": "2026-10-27",
        "kind": "inv",
        "verify": {"short": {"label": "검증됨"}},
        "tm": {"state": "진입 신호", "stop_px": 107},
    }
    acts = make_actions([asset], [inv])
    assert [a["action"] for a in acts] == ["BUY", "LONG_ENTRY", "SHORT_EXIT"]
    assert acts[0]["stop"] == 107 and acts[1]["stop"] == 90 and not acts[2]["verified"]
    text = telegram_summary(_data(actions=acts))
    assert "🎯 오늘 할 일 3건" in text and "롱 진입 SPY [✅]" in text


def test_inverse_follows_underlying_short(monkeypatch):
    from minerva.strategy import daily

    bars = bars_until(date(2026, 10, 20), 5)
    monkeypatch.setattr(daily, "fetch_yahoo", lambda s, r: bars)
    monkeypatch.setattr(daily, "short_status", lambda b: {"state": "진입 신호", "stop_px": 1})
    monkeypatch.setattr(daily, "regime_is_fear", lambda vix: False)
    monkeypatch.setattr(daily, "verify", lambda s, b, c: {"short": {"label": "검증됨"}})
    inv = daily.Asset("QID", "QID", "US", "inv", "QQQ", -2)
    out = daily.product_status(inv, bars, {"QQQ": "나스닥100"})
    assert out["tm"]["state"] == "진입 신호" and out["underlying"] == "나스닥100"
    assert all(abs(p.mult) <= 2 and p.kind == "inv" for p in daily.PRODUCTS)


def test_verdict_labels():
    from minerva.strategy.daily import _verdict

    def res(n, i, o):
        return {
            "all": {"trades": n, "edge": (i + o) / 2},
            "in_sample": {"edge": i},
            "out_of_sample": {"edge": o},
        }

    assert _verdict(res(40, 0.01, 0.002))["label"] == "검증됨"
    assert _verdict(res(40, -0.01, -0.002))["label"] == "부적합"
    assert _verdict(res(40, 0.01, -0.002))["label"] == "미검증"
    assert _verdict(res(10, 0.01, 0.02))["label"] == "미검증"


def test_wait_levels_give_prices_and_dates():
    from minerva.strategy.daily import _month_end, _week_end, wait_levels

    assert _week_end(date(2026, 10, 27)) == date(2026, 10, 30)
    assert _week_end(date(2026, 10, 30)) == date(2026, 11, 6)
    assert _month_end(date(2026, 12, 3)) == date(2026, 12, 31)
    bars = [
        Bar(date(2025, 1, 1) + timedelta(days=i), 100 + i, 101 + i, 99 + i, 100 + i)
        for i in range(400)
    ]
    lv = wait_levels(bars, "long", 16.0)
    assert "16.00" in lv["VIX 평온(R0·R1)"] and "확정" in lv["주봉 20주선 위"]
    assert "아래로" in wait_levels(bars, "short")["50일선 아래"]


def test_base_asset_gets_short_side_and_fear_blocks_it(monkeypatch):
    from minerva.strategy import daily

    monkeypatch.setattr(daily, "short_status", lambda bars: {"state": "진입 신호", "stop": "x"})
    monkeypatch.setattr(daily, "regime_is_fear", lambda vix: False)
    assert daily.base_short([], None)["state"] == "진입 신호"
    monkeypatch.setattr(daily, "regime_is_fear", lambda vix: True)
    assert daily.base_short([], None)["missing"] == ["백워데이션 중 신규 숏 금지"]
