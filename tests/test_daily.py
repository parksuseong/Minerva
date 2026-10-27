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


def test_summary_mentions_unverified_and_no_signal():
    data = {
        "generated_at": "2026-10-28T09:10",
        "vix": {"close": 18.4, "regime": "R2", "size_limit": "계획의 1/2", "leverage": "금지"},
        "fear": {"state": "없음", "premium": -0.08},
        "assets": [{"name": "코스피", "verified": False, "tm": {"state": "대기"}}],
    }
    text = telegram_summary(data)
    assert "전 종목 대기" in text and "R2" in text
    data["products"] = [
        {
            "name": "SOXL",
            "verified": False,
            "tm": {"state": "보유 중", "pnl": None, "stop": "기초 1", "days_left": 3},
        }
    ]
    text = telegram_summary(data)
    assert "SOXL" in text and "TM 되돌림 숏" in text


def test_inverse_uses_own_chart_and_lev_follows_base(monkeypatch):
    from minerva.strategy import daily

    bars = bars_until(date(2026, 10, 20), 5)
    monkeypatch.setattr(daily, "fetch_yahoo", lambda s, r: bars)
    inv = daily.Asset("SQQQ", "SQQQ", "US", "inv", "QQQ", -3)
    base = {"name": "QQQ", "close": 90, "ma50": 100, "tm": {"state": "진입 신호", "stop": 81}}
    out = daily.product_status(inv, base, "R0")
    assert out["tm"]["state"] == "대기" and "숏" in out["tm"]["note"]
    lev = daily.Asset("TQQQ", "TQQQ", "US", "lev", "QQQ", 3)
    assert daily.product_status(lev, base, "R0")["tm"]["state"] == "진입 신호"
    blocked = daily.product_status(lev, base, "R2")["tm"]
    assert blocked["state"] == "대기" and blocked["missing"][0].startswith("레짐 R2")


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
    assert "BTC-USD" in daily.SHORT_VERIFIED and "TSLA" not in daily.SHORT_VERIFIED


def test_telegram_lists_base_short_signal():
    data = {
        "generated_at": "2026-10-28T09:10",
        "vix": {"close": 18.4, "regime": "R2", "size_limit": "계획의 1/2", "leverage": "금지"},
        "fear": {"state": "없음", "premium": -0.08},
        "assets": [
            {
                "name": "비트코인",
                "short_verified": True,
                "tm": {"state": "대기"},
                "short": {"state": "진입 신호", "stop": "기초자산 85,000 돌파 시 손절"},
            }
        ],
    }
    text = daily_telegram(data)
    assert "🔻 비트코인: 숏 진입 신호" in text and "전 종목 대기" not in text


def daily_telegram(data):
    from minerva.strategy.daily import telegram_summary

    return telegram_summary(data)
