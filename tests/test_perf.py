import json
from datetime import datetime

from minerva import kis, perf

SELL = {
    "key": "2026-10-28:000660:sell",
    "side": "sell",
    "result": "sent",
    "code": "000660",
    "qty": 3,
    "name": "SK하이닉스",
    "kind": "long",
    "action": "LONG_EXIT",
    "entry_avg": 100000,
    "opened": "2026-10-20",
    "at": "2026-10-29T08:50:00",
}
FILLS = [
    {"time": "2026-10-29 09:00", "code": "000660", "side": "현금매도", "qty": 3, "price": 110000}
]


def test_closed_trade_net_of_costs():
    row = perf.closed_trade(SELL, FILLS)
    assert row["gross_ret"] == 0.1
    assert row["net_ret"] == round(0.1 - perf.cost_rate("000660"), 5)
    assert row["pnl"] == round(3 * 100000 * row["net_ret"])
    assert perf.cost_rate("114800") < perf.cost_rate("000660")  # ETF엔 거래세가 없다


def test_record_closed_dedupes_and_skips_tests(tmp_path):
    test_sell = {**SELL, "key": "selftest:x", "kind": "test"}
    acc = {"trades": FILLS}
    assert len(perf.record_closed(tmp_path, [SELL, test_sell], acc, "2026-10-29")) == 1
    assert perf.record_closed(tmp_path, [SELL], acc, "2026-10-29") == []


def test_bot_pnl_and_weekly_report(tmp_path):
    perf.record_closed(tmp_path, [SELL], {"trades": FILLS}, "2026-10-29")
    owned = {"005380": {"name": "현대차"}}
    acc = {
        "equity": 1_000_000,
        "positions": [{"code": "005380", "pnl": -5000}, {"code": "035420", "pnl": 9999}],
    }
    realized = json.loads((tmp_path / "closed.jsonl").read_text())["pnl"]
    assert perf.bot_pnl(tmp_path, owned, acc) == realized - 5000
    daily = {
        "assets": [{"symbol": "000660.KS", "verify": {"long": {"win_rate": 0.6, "net_edge": 0.01}}}]
    }
    text = perf.weekly_report(
        tmp_path, daily, acc, owned, datetime(2026, 10, 30, 17, tzinfo=kis.KST)
    )
    assert "청산 1건" in text and "승률 100%" in text and "백테스트 기대: 승률 60%" in text
    assert "30건부터" in text


def test_closed_trade_ignores_user_sells_same_day():
    fills = FILLS + [
        {
            "time": "2026-10-29 10:00",
            "code": "000660",
            "side": "현금매도",
            "qty": 7,
            "price": 110000,
        }
    ]
    row = perf.closed_trade({**SELL, "qty": 3}, fills)
    assert row["qty"] == 3
