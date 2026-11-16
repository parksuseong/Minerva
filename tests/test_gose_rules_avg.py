"""고서 "SCR 3단 + 물타기" 규칙(gose_rules_avg): 지표 손계산, 신호 창, 체결·평단·익절."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pytest

from minerva import gose_rules_avg as A

# ---------------------------------------------------------------- 지표


def test_stoch_k_hand_case_and_flat():
    h = [10, 12, 11, 13, 14]
    lo = [8, 9, 7, 10, 11]
    c = [9, 11, 10, 12, 13]
    k = A.stoch_k(h, lo, c)
    assert k[:4] == [None] * 4
    assert k[4] == pytest.approx((13 - 7) / (14 - 7) * 100)
    assert A.stoch_k([5] * 5, [5] * 5, [5] * 5)[4] == 50.0


def test_cci_hand_case_and_flat():
    # n=3: TP = 1,2,3 → 평균 2, 평균절대편차 2/3 → (3−2)/(0.015×2/3) = 100
    h = lo = c = [1.0, 2.0, 3.0]
    assert A.cci(h, lo, c, n=3)[2] == pytest.approx(100.0)
    assert A.cci([2.0] * 3, [2.0] * 3, [2.0] * 3, n=3)[2] == 0.0
    # 20봉 기본: TP가 직선이면 마지막 값 = (19−9.5)/(0.015×5) = 126.67
    tp = [float(i) for i in range(20)]
    assert A.cci(tp, tp, tp)[19] == pytest.approx(9.5 / (0.015 * 5.0))
    assert A.cci(tp, tp, tp)[18] is None


def test_rsi_sma_hand_case_and_edges():
    # n=3: 변화 +1, −1, +2 → ΣU 3, ΣD 1 → 100 − 100/4 = 75
    assert A.rsi_sma([10, 11, 10, 12], n=3)[3] == pytest.approx(75.0)
    assert A.rsi_sma([10, 11, 12, 13], n=3)[3] == 100.0  # ΣD=0, ΣU>0
    assert A.rsi_sma([10, 10, 10, 10], n=3)[3] == 50.0  # 둘 다 0
    r = A.rsi_sma(list(range(30)))
    assert r[13] is None and r[14] == 100.0


# ---------------------------------------------------------------- 신호


def _ind(s, c, r):
    return {"S": list(s), "C": list(c), "R": list(r)}


def test_signal_needs_cross_above_all_and_window():
    # 0..2 과매도, 3봉에 스토캐 20 위로(셋 다 선 위)
    s = [50, 10, 15, 30, 40]
    c = [0, -150, -120, -50, 0]
    r = [50, 25, 28, 35, 40]
    assert A.signals(_ind(s, c, r)) == [3]
    # RSI가 창에서 33 넘게만 있었으면 신호 아님(34)
    r2 = [50, 34, 34, 35, 40]
    assert A.signals(_ind(s, c, r2)) == []
    # RSI 33 정확히(≤33)는 됨, 단 지금 30 위여야 함
    r3 = [50, 33, 33, 35, 40]
    assert A.signals(_ind(s, c, r3)) == [3]
    # 셋 중 하나라도 선 아래면 아님
    c4 = [0, -150, -120, -101, 0]
    assert 3 not in A.signals(_ind(s, c4, r))


def test_signal_window_is_ten_bars():
    # 과매도 봉이 t−11이면 창 [t−10, t−1] 밖
    n = 14
    s = [50.0] * n
    c = [0.0] * n
    r = [50.0] * n
    # 빈 값(None) 봉으로 회복 돌파를 감춰 과매도 봉 하나만 남긴다
    s[1], c[1], r[1] = 10, -150, 25  # t=12 기준 t−11
    s[2] = c[2] = r[2] = None
    s[11] = 15  # 12봉에서 스토캐 돌파(스토캐 과매도는 11봉이 채움)
    assert A.signals(_ind(s, c, r)) == []
    s2, c2, r2 = list(s), list(c), list(r)
    s2[1], c2[1], r2[1] = None, None, None
    s2[2], c2[2], r2[2] = 10, -150, 25  # t−10: 창 안
    s2[3] = c2[3] = r2[3] = None
    assert A.signals(_ind(s2, c2, r2)) == [12]


def test_signal_window_starts_after_previous_sig():
    # 1봉 과매도 → 3봉 SIG. 5봉에 스토캐가 다시 20 아래→6봉 돌파하지만
    # CCI·RSI 과매도는 직전 SIG(3) 전에만 있었다 → 6봉은 SIG 아님
    s = [50, 10, 15, 30, 40, 18, 30, 40]
    c = [0, -150, -120, -50, 0, 0, 0, 0]
    r = [50, 25, 28, 35, 40, 40, 40, 40]
    assert A.signals(_ind(s, c, r)) == [3]
    # 같은 자리에 CCI·RSI 과매도가 직전 SIG 뒤(5봉)에도 있으면 SIG(값은 지금 선 위)
    c2 = [0, -150, -120, -50, 0, -110, -50, 0]
    r2 = [50, 25, 28, 35, 40, 29, 35, 40]
    assert A.signals(_ind(s, c2, r2)) == [3, 6]


def test_signal_none_values_skip():
    s = [None, None, 10, 30]
    c = [None, -150, -150, -50]
    r = [None, 25, 25, 35]
    assert A.signals(_ind(s, c, r)) == [3]  # 2봉 앞(None)은 건너뛰지만 창 안 값은 봄


# ---------------------------------------------------------------- 체결


def _bars(rows, d0=date(2025, 1, 6)):
    out, d = [], d0
    for o, h, lo, c in rows:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append((d.isoformat(), float(o), float(h), float(lo), float(c)))
        d += timedelta(days=1)
    return out


DUMMY = {"S": [], "C": [], "R": []}


def test_entry_next_open_unit_is_third_and_tp_at_high():
    rows = [(100, 101, 99, 100)] * 3 + [
        (100, 105, 99, 104),
        (104, 111, 103, 110),
        (110, 111, 109, 110),
    ]
    b = _bars(rows)
    res = A.simulate(b, capital=9000.0, ind=DUMMY, sigs=[2], fee=0.0)
    (cy,) = res["cycles"]
    assert cy["entry_date"] == b[3][0] and cy["unit"] == pytest.approx(3000.0)
    # 평단 100 → 익절가 110, 4봉 고가 111 ≥ 110 → 110에 체결(시가 104 < 110)
    assert cy["exit_date"] == b[4][0] and cy["exit_px"] == pytest.approx(110.0)
    assert res["equity"] == pytest.approx(9000 + 3000 * 0.10)
    assert res["open"] is None


def test_tp_gap_open_fills_at_open_but_not_on_cycle_start_bar():
    rows = [(100, 101, 99, 100)] * 3 + [(100, 101, 99, 100), (120, 125, 119, 121)]
    b = _bars(rows)
    res = A.simulate(b, capital=3000.0, ind=DUMMY, sigs=[2], fee=0.0)
    assert res["cycles"][0]["exit_px"] == pytest.approx(120.0)  # 시가 갭 → 시가
    # 시작한 봉에는 시가 체결이 없다: 시가가 높아도 평단도 그 시가라 목표 = 시가×1.1
    rows2 = [(100, 101, 99, 100)] * 3 + [(100, 109.9, 99, 100)]
    res2 = A.simulate(_bars(rows2), capital=3000.0, ind=DUMMY, sigs=[2], fee=0.0)
    assert res2["cycles"] == [] and res2["open"]["tp_px"] == pytest.approx(110.0)


def test_adds_capped_at_two_and_avg_cost_includes_fees():
    fee = 0.0005
    rows = [(100, 101, 99, 100)] * 2 + [
        (100, 100, 90, 90),
        (90, 90, 80, 80),
        (80, 80, 70, 70),
        (70, 70, 60, 60),
        (60, 60, 60, 60),
    ]
    b = _bars(rows)
    # 신호 1,2,3,4 → 2봉 진입, 3·4봉 추가, 5봉 신호는 한도
    res = A.simulate(b, capital=3000.0, ind=DUMMY, sigs=[1, 2, 3, 4], fee=fee)
    o = res["open"]
    assert o["adds"] == 2 and o["adds_left"] == 0
    assert [f[2] for f in o["fills"]] == [100.0, 90.0, 80.0]
    sh = sum(1000 / (p * (1 + fee)) for p in (100, 90, 80))
    assert o["shares"] == pytest.approx(sh)
    assert o["invested"] == pytest.approx(3000.0) and res["cash"] == pytest.approx(0.0)
    assert o["avg"] == pytest.approx(3000 / sh)
    assert o["tp_px"] == pytest.approx(3000 / sh * 1.1)
    assert o["mae"] == pytest.approx(60 / (3000 / sh) - 1)
    assert res["next"] is None  # 마지막 봉(6)은 신호 아님
    res2 = A.simulate(b, capital=3000.0, ind=DUMMY, sigs=[1, 2, 3, 4, 6], fee=fee)
    assert res2["next"]["kind"] == "capped"


def test_add_recomputes_tp_then_exit_and_no_reentry_on_exit_bar():
    rows = [(100, 101, 99, 100)] * 2 + [(100, 100, 80, 80), (80, 95, 80, 94), (94, 95, 93, 94)]
    b = _bars(rows)
    # 2봉 진입 100, 3봉 추가 80 → 평단 88.9, 익절 97.8 (고가 95라 미체결)
    res = A.simulate(b, capital=3000.0, ind=DUMMY, sigs=[1, 2], fee=0.0)
    o = res["open"]
    avg = 2000 / (10 + 12.5)
    assert o["avg"] == pytest.approx(avg) and o["adds"] == 1
    assert o["tp_px"] == pytest.approx(avg * 1.1)
    # 신호 3 → 4봉 시가 추가(3번째 단위) 뒤 익절 판정
    rows2 = [(100, 101, 99, 100)] * 2 + [
        (100, 100, 80, 80),
        (80, 81, 79, 80),
        (80, 99, 80, 98),
        (98, 99, 97, 98),
    ]
    b2 = _bars(rows2)
    res2 = A.simulate(b2, capital=3000.0, ind=DUMMY, sigs=[1, 2, 3], fee=0.0)
    (cy,) = res2["cycles"]
    avg2 = 3000 / (10 + 12.5 + 12.5)
    assert cy["exit_date"] == b2[4][0] and cy["exit_px"] == pytest.approx(avg2 * 1.1)
    assert cy["adds"] == 2
    # 익절 봉(4)에 신호가 있으면 진입은 그 다음 봉(5) 시가
    res3 = A.simulate(b2, capital=3000.0, ind=DUMMY, sigs=[1, 2, 3, 4], fee=0.0)
    assert res3["open"]["entry_date"] == b2[5][0]


def test_add_day_before_tp_order_open_gap():
    # 추가하는 날 시가가 옛 익절가 위여도 추가가 먼저, 새 평단으로 판정
    rows = [(100, 101, 99, 100)] * 2 + [(100, 101, 99, 100), (112, 113, 111, 112)]
    b = _bars(rows)
    res = A.simulate(b, capital=3000.0, ind=DUMMY, sigs=[1, 2], fee=0.0)
    o = res["open"]
    assert o is not None and o["adds"] == 1  # 새 평단 105.66 → 익절 116.2, 미체결
    assert o["avg"] == pytest.approx(2000 / (10 + 1000 / 112))


def test_start_ignores_earlier_signals():
    rows = [(100, 101, 99, 100)] * 6
    b = _bars(rows)
    res = A.simulate(b, start=b[3][0], capital=3000.0, ind=DUMMY, sigs=[2, 4], fee=0.0)
    # 2봉 신호(시작 전) → 3봉 진입 안 함, 4봉 신호 → 5봉 진입
    assert res["open"]["entry_date"] == b[5][0]
    assert res["curve"][0][0] == b[3][0]
    nothing = A.simulate(b, start="2099-01-01", capital=1.0, ind=DUMMY, sigs=[2])
    assert nothing["curve"] == [] and nothing["equity"] == 1.0 and nothing["open"] is None


def test_next_action_entry_and_add():
    rows = [(100, 101, 99, 100)] * 4
    b = _bars(rows)
    r = A.simulate(b, capital=900.0, ind=DUMMY, sigs=[3], fee=0.0)
    assert r["next"]["kind"] == "entry" and r["next"]["amount"] == pytest.approx(300.0)
    r = A.simulate(b, capital=900.0, ind=DUMMY, sigs=[1, 3], fee=0.0)
    assert r["next"]["kind"] == "add" and r["next"]["amount"] == pytest.approx(300.0)


# ---------------------------------------------------------------- 실제 비슷한 자료


def _walk(n=1500, seed=3):
    rng = random.Random(seed)
    out, d, c = [], date(2015, 1, 5), 50.0
    for _ in range(n):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        o = c * (1 + rng.gauss(0, 0.01))
        c = o * (1 + rng.gauss(0.0004, 0.025))
        h = max(o, c) * (1 + abs(rng.gauss(0, 0.01)))
        lo = min(o, c) * (1 - abs(rng.gauss(0, 0.01)))
        out.append((d.isoformat(), o, h, lo, c))
        d += timedelta(days=1)
    return out


def test_deterministic_and_all_closed_cycles_win():
    b = _walk()
    r1 = A.simulate(b, capital=10000.0)
    r2 = A.simulate(b, capital=10000.0)
    assert r1 == r2
    assert r1["cycles"] and all(c["ret"] > 0.09 for c in r1["cycles"])
    assert all(c["adds"] <= 2 for c in r1["cycles"])
    m = A.metrics(r1)
    assert m["win"] == 1.0 and m["mdd"] <= 0 and m["n"] == len(r1["cycles"])
    assert not math.isnan(m["cagr"])


def test_no_lookahead_by_truncation():
    b = _walk()
    full_sigs = A.signals(A.indicators(b))
    full = A.simulate(b, capital=10000.0)
    for k in (300, 777, 1200):
        part = b[:k]
        assert A.signals(A.indicators(part)) == [s for s in full_sigs if s < k]
        r = A.simulate(part, capital=10000.0)
        assert r["curve"] == full["curve"][:k]
        assert [c["exit_date"] for c in r["cycles"]] == [
            c["exit_date"] for c in full["cycles"] if c["exit_date"] <= part[-1][0]
        ]
