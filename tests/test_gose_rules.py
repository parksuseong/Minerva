"""gose_rules 자체 점검 — 고정 합성 데이터, 네트워크 없음."""

import math
from datetime import date, timedelta

from minerva import gose_rules as R


def make_bars(closes, start=date(2020, 1, 6), spread=0.01):
    """평일만 이어지는 일봉. 시가=전일 종가, 고저는 종가±spread."""
    out, d, prev = [], start, closes[0]
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append(
            (d.isoformat(), prev, max(prev, c) * (1 + spread), min(prev, c) * (1 - spread), c)
        )
        prev = c
        d += timedelta(days=1)
    return out


def wave(n=400, period=40, amp=0.15, drift=0.0):
    return [
        100 * (1 + amp * math.sin(2 * math.pi * i / period)) * (1 + drift) ** i for i in range(n)
    ]


def test_cross_definitions():
    s = [10.0, 20.0, 25.0, 19.0]
    assert R.cross_up(s, 1, 20.0)  # 10 < 20 <= 20
    assert not R.cross_up(s, 2, 20.0)
    assert R.cross_down(s, 3, 20.0)  # 25 > 20 >= 19
    assert R.cross_down([25.0, 20.0], 1, 20.0)  # 25 > 20 >= 20


def test_rsi_bounds_and_flat():
    r = R.rsi([1.0] * 30)
    assert r[13] is None and r[14] == 50.0
    up = R.rsi([float(i) for i in range(1, 40)])
    assert up[-1] == 100.0


def test_stoch_range():
    b = make_bars(wave())
    k = R.stoch_fast_k([x[2] for x in b], [x[3] for x in b], [x[4] for x in b])
    assert all(v is None or 0.0 <= v <= 100.0 for v in k)


def test_resample_weekly_closed_flag():
    b = make_bars(wave(n=12))  # 2020-01-06(월)부터 12 평일 → 마지막 2020-01-21(화)
    wk, last_idx, closed = R.resample(b, "W")
    assert len(wk) == 3 and last_idx[0] == 4 and not closed
    assert wk[0][0] == "2020-01-10" and wk[0][4] == b[4][4]


def test_deterministic_and_no_lookahead():
    closes = wave(n=500, drift=0.0005)
    b = make_bars(closes)
    for sid in ("1A", "1B", "4", "5", "8", "9", "14", "6"):
        full = R.simulate(sid, b, stop_pct=0.10)
        again = R.simulate(sid, b, stop_pct=0.10)
        assert [t["ret"] for t in full["trades"]] == [t["ret"] for t in again["trades"]]
        # 뒤쪽 데이터를 잘라도 앞쪽에서 끝난 거래는 같아야 한다
        cut = R.simulate(sid, b[:350], stop_pct=0.10)
        early = [
            (t["entry_date"], t["exit_date"], round(t["ret"], 12))
            for t in full["trades"]
            if t["exit_i"] < 340
        ]
        early_cut = [
            (t["entry_date"], t["exit_date"], round(t["ret"], 12))
            for t in cut["trades"]
            if t["exit_i"] < 340
        ]
        assert early == early_cut, sid


def test_entry_fills_next_open_and_fee():
    b = make_bars(wave(n=300))
    res = R.simulate("1A", b)
    assert res["trades"], "합성 파동에서 1단 거래가 있어야 함"
    t = res["trades"][0]
    sig = R.signals("1A", b)
    buy = [s for s in sig if s[1] == "BUY"][0]
    assert buy[0] < t["entry_date"]  # 신호일 다음 날 체결
    i = t["entry_i"]
    assert t["fills"][0][1] == b[i][1]  # 시가 체결
    gross = t["exit_px"] / t["fills"][0][1]
    assert abs((1 + t["ret"]) - gross * (1 - R.FEE) / (1 + R.FEE)) < 1e-12


def test_stop_caps_loss():
    # 단조 하락: 1단 진입 후 10% 손절이 걸려야 한다
    closes = wave(n=120, period=30, amp=0.2) + [70 * 0.97**k for k in range(60)]
    b = make_bars(closes)
    stopped = 0
    for sid in ("1A", "8"):
        for t in R.simulate(sid, b, stop_pct=0.10)["trades"]:
            if t["reason_out"].startswith("손절"):
                stopped += 1
                assert -0.105 < t["ret"] < 0  # 합성 데이터는 갭이 작아 손실이 10% + 수수료 안쪽
    assert stopped > 0


def test_short_return_sign():
    b = make_bars(wave(n=400, amp=0.3, period=60))
    res = R.simulate("14", b)
    for t in res["trades"]:
        px_in = t["fills"][0][1]
        if t["exit_px"] < px_in * 0.99:
            assert t["ret"] > 0


def test_time_exit():
    b = make_bars(
        [100 + 0.01 * i for i in range(30)] + [90.0] + [90 + 0.001 * i for i in range(200)]
    )
    for t in R.simulate("9", b)["trades"]:
        assert t["exit_i"] - t["entry_i"] <= R.STRATEGIES["9"]["max_hold"] + 1


def test_strategy_table():
    for _sid, s in R.STRATEGIES.items():
        assert s["side"] in ("long", "short") and s["timeframe"] in ("D", "W", "M")
        assert s["rule_ids"] and s["max_hold"] > 0
