"""gose_rules_v2 자체 점검 — 고정 합성 데이터, 네트워크 없음."""

import math
from datetime import date, timedelta

from minerva import gose_rules as R
from minerva import gose_rules_v2 as V


def make_bars(closes, start=date(2015, 1, 5), spread=0.01):
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


def wave(n=600, period=40, amp=0.15, drift=0.0005, period2=170, amp2=0.10):
    return [
        100
        * (1 + amp * math.sin(2 * math.pi * i / period))
        * (1 + amp2 * math.sin(2 * math.pi * i / period2))
        * (1 + drift) ** i
        for i in range(n)
    ]


def test_same_bar_cross_and_falling():
    ind = {"stoch": [10.0, 25.0], "cci": [-150.0, -90.0], "rsi": [28.0, 31.0]}
    assert V.all_cross_up(ind, 1)
    ind["rsi"] = [28.0, 29.0]
    assert not V.all_cross_up(ind, 1)  # RSI가 30을 못 넘으면 같은 봉 돌파 아님
    ind = {"stoch": [50, 85, 90, 70, 60], "cci": [0, 50, 60, 40, 30], "rsi": [50, 55, 60, 58, 57]}
    assert V.e19_exit(ind, 4)  # 셋 다 하락, 스토캐가 t-3..t-1에 80 이상
    ind["rsi"][4] = 59
    assert not V.e19_exit(ind, 4)  # RSI가 오르면 아님
    ind = {"stoch": [50, 60, 70, 75, 60], "cci": [0, 50, 60, 40, 30], "rsi": [50, 55, 60, 58, 57]}
    assert not V.e19_exit(ind, 4)  # 스토캐가 80을 안 찍음


def test_deterministic_and_no_lookahead():
    b = make_bars(wave())
    vol = make_bars([20 * (1 + 0.5 * math.sin(2 * math.pi * i / 33)) ** 2 + 5 for i in range(600)])
    for sid in V.STRATEGIES:
        full = V.simulate(sid, b, vol=vol, stop_pct=0.10)
        again = V.simulate(sid, b, vol=vol, stop_pct=0.10)
        assert [t["ret"] for t in full["trades"]] == [t["ret"] for t in again["trades"]]
        cut = V.simulate(sid, b[:450], vol=vol[:450], stop_pct=0.10)
        early = [
            (t["entry_date"], t["exit_date"], round(t["ret"], 12))
            for t in full["trades"]
            if t["exit_i"] < 440
        ]
        early_cut = [
            (t["entry_date"], t["exit_date"], round(t["ret"], 12))
            for t in cut["trades"]
            if t["exit_i"] < 440
        ]
        assert early == early_cut, sid


def test_fill_next_open_and_fee():
    b = make_bars(wave())
    vol = make_bars([20 * (1 + 0.5 * math.sin(2 * math.pi * i / 33)) ** 2 + 5 for i in range(600)])
    found = 0
    for sid in V.STRATEGIES:
        res = V.simulate(sid, b, vol=vol)
        for t in res["trades"]:
            found += 1
            i = t["entry_i"]
            assert t["signal_date"] == b[i - 1][0]  # 신호 다음 봉 시가
            assert t["fills"][0][1] == b[i][1]
            gross = t["exit_px"] / t["fills"][0][1]
            assert abs((1 + t["ret"]) - gross * (1 - V.FEE) / (1 + V.FEE)) < 1e-12
            assert t["exit_i"] - t["entry_i"] <= V.STRATEGIES[sid]["max_hold"] + 1
    assert found > 0


def test_vol_triggers_a_definition():
    # 손으로 만든 지표 대신 실제 함수 결과가 정의를 지키는지 확인
    vol = make_bars([20 * (1 + 0.5 * math.sin(2 * math.pi * i / 33)) ** 2 + 5 for i in range(600)])

    ind = R.indicators(vol)
    hits = V.vol_triggers_a(vol)
    assert hits
    for t, _d in hits:
        assert ind["stoch"][t] < 80 and ind["cci"][t] < 100 and ind["rsi"][t] < 70
        assert ind["stoch"][t - 1] >= 80 or ind["cci"][t - 1] >= 100 or ind["rsi"][t - 1] >= 70
        assert max(ind["rsi"][t - 20 : t]) >= 70
    for t, _d in V.vol_triggers_b(vol):
        assert V.all_falling(ind, t)


def test_stop_caps_loss():
    closes = wave(n=300) + [100 * 0.97**k for k in range(80)]
    b = make_bars(closes)
    for sid in ("1v2", "3v2-loose"):
        for t in V.simulate(sid, b, stop_pct=0.10)["trades"]:
            if t["reason_out"].startswith("손절"):
                assert -0.105 < t["ret"] < 0
