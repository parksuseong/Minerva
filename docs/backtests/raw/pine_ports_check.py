"""pine_ports·pine_eval 손 계산 점검(합성 봉, 네트워크 없음).

실행: PYTHONPATH=<pyml>:. python3 pine_ports_check.py
"""

from __future__ import annotations

import math

import numpy as np
import pine_eval as E
import pine_ports as P

M15 = 900_000


def bars(o, h, lo, c, v=None, t0=1_767_225_600_000) -> dict:
    n = len(c)
    return {
        "t": np.arange(n, dtype=np.int64) * M15 + t0,
        "o": np.array(o, float),
        "h": np.array(h, float),
        "l": np.array(lo, float),
        "c": np.array(c, float),
        "v": np.ones(n) if v is None else np.array(v, float),
        "ft": np.array([t0 + 10**12], dtype=np.int64),
        "fr": np.array([0.0]),
    }


def check_indicators() -> None:
    x = np.array([1, 2, 3, 4, 5, 6], float)
    assert np.allclose(P.sma(x, 3)[2:], [2, 3, 4, 5])
    e = P.ema(x, 3)
    assert math.isnan(e[1]) and e[2] == 2.0 and abs(e[3] - (0.5 * 4 + 0.5 * 2)) < 1e-12
    r = P.rma(x, 3)
    assert r[2] == 2.0 and abs(r[3] - (4 / 3 + 2 * 2 / 3)) < 1e-12
    assert abs(P.stdev(np.array([1, 2, 3, 4], float), 4)[3] - math.sqrt(1.25)) < 1e-12
    # 피벗: 봉 4에서 x[2]가 왼쪽 2·오른쪽 2보다 낮으면 확정
    pv = P.pivots(np.array([5, 4, 1, 3, 6, 7], float), 2, 2, low=True)
    assert pv[4] == 1 and np.isnan(pv[3]) and np.isnan(pv[5])
    # rsi: 계속 오르면 100
    assert P.rsi(np.arange(30, dtype=float), 14)[-1] == 100.0
    # 교차
    a = np.array([1, 2, 3, 1], float)
    bb = np.array([2, 2, 2, 2], float)
    assert P.cross_over(a, bb).tolist() == [False, False, True, False]
    assert P.cross_under(a, bb).tolist() == [False, False, False, True]


def check_s8() -> None:
    c = np.r_[np.full(25, 100.0) + np.tile([0.1, -0.1], 13)[:25], 90.0, 95.0, 99.0]
    s = P.s8_extreme(c)
    sd = (c - P.sma(c, 20)) / P.stdev(c, 20)
    i = int(np.flatnonzero(s == 1)[0])
    assert sd[i - 1] < -2 < sd[i], (i, sd[i - 1 : i + 1])


def check_s10() -> None:
    c = np.r_[np.linspace(100, 80, 120), np.linspace(80, 120, 120)]
    s = P.s10_ma(c)["sig"]
    f, sl = P.ema(c, 24), P.ema(c, 65)
    i = int(np.flatnonzero(s == 1)[0])
    assert f[i] >= sl[i] and f[i - 1] < sl[i - 1]
    assert (s == -1).sum() == 0


def check_s1() -> None:
    # 30봉 횡보 → 큰 양봉(구역 = 직전 봉) → 되돌림 양봉이 구역에 닿으면 롱
    n = 30
    o = [100.0] * n
    c = [100.2 if k % 2 else 99.8 for k in range(n)]
    h = [max(a, b) + 0.1 for a, b in zip(o, c)]
    lo = [min(a, b) - 0.1 for a, b in zip(o, c)]
    v = [1.0] * n
    # 충격 봉(30): 직전 봉 29가 구역
    o += [100.0, 103.0, 101.5]
    c += [103.0, 102.0, 101.8]
    h += [103.1, 103.1, 102.0]
    lo += [99.9, 101.9, 100.0]
    v += [5.0, 1.0, 1.0]
    b = bars(o, h, lo, c, v)
    r = P.s1_ob_retest(b)
    z_top, z_bot = h[29], lo[29]
    assert r["sig"][32] == 1, r["sig"][28:]
    assert abs(r["native"]["sl"][32] - (z_bot - 0.2 * P.atr(b["h"], b["l"], b["c"], 14)[32])) < 1e-9
    assert z_top >= lo[32]


def check_s7() -> None:
    # 13:00 세션 봉, 5봉 뒤 상자, 그다음 봉 고가 돌파 → 롱 체결(상자 고가)
    t0 = 1_767_225_600_000 + 5 * 3_600_000  # 05:00 UTC (ATR30 예열)
    n = 200
    o = np.full(n, 100.0)
    c = np.full(n, 100.0)
    h = np.full(n, 100.5)
    lo = np.full(n, 99.5)
    k = 32 + 5 + 1  # 13:00 = 32번째 봉, 상자 = 37번째 봉, 체결 = 38번째
    h[k] = 102.0
    lo[k] = 99.9
    b = bars(o, h, lo, c, t0=t0)
    r = P.s7_session(b)
    assert r["sig_all"][k] == 1 and r["epx"][k] == 100.5, np.flatnonzero(r["sig_all"])
    assert abs(r["native"]["tp"][k] - (100.5 + 600)) < 1e-9


def check_sim() -> None:
    # 롱 신호 봉 0 → 봉1 시가 100 진입, 봉2에서 목표(1×ATR) 도달
    n = 40
    o = np.full(n, 100.0)
    c = np.full(n, 100.0)
    h = np.full(n, 100.5)
    lo = np.full(n, 99.5)
    b = bars(o, h, lo, c)
    B = E.Bars(b)
    at = B.atr[20]
    h[22] = 100 + at + 0.01
    b = bars(o, h, lo, c)
    B = E.Bars(b)
    sig = np.zeros(n, np.int8)
    sig[20] = 1
    S = {"sig": sig, "native": None}
    tr = E.simulate(B, S, ("T1", "S0"), 0, n)
    assert len(tr) == 1 and tr[0]["e"] == 21 and tr[0]["x"] == 22 and tr[0]["why"] == "tp"
    px = 100 + at
    exp = (px / 100 - 1) - E.FEE * px / 100 - E.FEE
    assert abs(tr[0]["r"] - exp) < 1e-12
    # 같은 봉 손절·목표 동시 → 손절
    lo2 = lo.copy()
    lo2[22] = 100 - 3 * at - 0.01
    B2 = E.Bars(bars(o, h, lo2, c))
    tr = E.simulate(B2, S, ("T1", "S1"), 0, n)
    assert tr[0]["why"] == "stop" and tr[0]["xp"] == 100 - 3 * B2.atr[20]
    # 10배 청산: 최악 −10%면 자산 0
    t = [{"d": 1, "r": 0.01, "worst": -0.10}]
    assert E.lev_path(t, 10)["ret"] == -1.0 and E.lev_path(t, 3)["liq"] == 0


if __name__ == "__main__":
    for f in (check_indicators, check_s8, check_s10, check_s1, check_s7, check_sim):
        f()
        print("ok", f.__name__)
