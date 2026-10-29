from datetime import date, datetime, timedelta, timezone

from minerva.strategy import intraday as it
from minerva.strategy.indicators import Bar

KST = timezone(timedelta(hours=9))


def _ts(y, m, d, h):
    return int(datetime(y, m, d, h, tzinfo=KST).timestamp())


def test_parse_60m_drops_filler_and_unfinished_days():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [_ts(2026, 1, 2, 9), _ts(2026, 1, 2, 10), _ts(2026, 1, 5, 9)],
                    "indicators": {
                        "quote": [
                            {
                                "open": [1, 2, 3],
                                "high": [1, 2, 3],
                                "low": [1, 2, 3],
                                "close": [1, 2, 3],
                                "volume": [0, 10, 10],
                            }
                        ]
                    },
                }
            ]
        }
    }
    bars = it.parse_60m(payload, before=date(2026, 1, 5))
    assert [(b.day, b.open) for b in bars] == [(date(2026, 1, 2), 2)]


def test_prev_day_mask_uses_previous_close_only():
    daily = [Bar(date(2026, 1, d), 1, 1, 1, 1) for d in (2, 5, 6)]
    mask = [True, False, True]
    bars = [Bar(date(2026, 1, d), 1, 1, 1, 1) for d in (2, 5, 5, 6, 7)]
    # 1/2 봉은 전날 정보가 없고, 1/5 봉은 1/2 값, 1/6 봉은 1/5 값, 1/7 봉은 1/6 값
    assert it.prev_day_mask(bars, daily, mask) == [False, True, True, False, True]
