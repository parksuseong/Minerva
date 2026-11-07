"""pine_tv_ports·pine_tv_eval 손 계산 점검(합성 봉, 네트워크 없음).

실행: PYTHONPATH=<pyml>:docs/backtests/raw python3 docs/backtests/raw/pine_tv_ports_check.py
"""

from __future__ import annotations

import numpy as np
import pine_eval as E
import pine_tv_eval as V
import pine_tv_ports as T
from pine_ports_check import bars

T0 = 1_767_225_600_000  # 2026-01-01 00:00 UTC (목요일)


def check_frame() -> None:
    o = np.arange(10, dtype=float)
    b = bars(o, o + 1, o - 1, o + 0.5)
    X = T.frame(b, 60)
    assert len(X["c"]) == 2  # 마지막 2봉은 덜 찬 시간이라 버림
    assert X["o"][0] == 0 and X["c"][0] == 3.5 and X["h"][0] == 4 and X["l"][0] == -1
    assert X["last15"].tolist() == [3, 7]
    st = T.state15(X, np.array([1.0, 2.0]), 10)
    assert np.isnan(st[2]) and st[3] == 1 and st[6] == 1 and st[7] == 2
    lv = T.level_next(X, np.array([1.0, 2.0]), 10)
    assert np.isnan(lv[3]) and lv[4] == 1 and lv[8] == 2
    s = T.to15(X, np.array([1, -1]), 10)
    assert s.tolist() == [0, 0, 0, 1, 0, 0, 0, -1, 0, 0]


def check_indicators() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=60).cumsum()
    n = 20
    w = x[-n:]
    a1, a0 = np.polyfit(np.arange(n), w, 1)
    assert abs(T.linreg(x, n)[-1] - (a0 + a1 * (n - 1))) < 1e-9
    assert abs(T.linreg(x, n, 1)[-1] - (a0 + a1 * (n - 2))) < 1e-9
    assert abs(T.lin_slope(x, n)[-1] - a1) < 1e-9
    assert abs(T.wma(np.array([1.0, 2, 3]), 3)[-1] - (1 + 4 + 9) / 6) < 1e-12
    assert abs(T.swma(np.array([1.0, 2, 3, 4]))[-1] - (4 + 6 + 4 + 1) / 6) < 1e-12
    assert T.rsum(np.ones(5), 3).tolist()[2:] == [3, 3, 3]
    pr = T.percentrank(np.array([1.0, 2, 3, 2.5]), 3)
    assert pr[3] == 100 * 2 / 3
    assert T.dow(np.array([T0]))[0] == 5  # 목요일
    assert T.dow(np.array([T0 + 2 * T.DAY]))[0] == 7  # 토요일
    assert T.hour(np.array([T0 + 5 * T.H1 + 900_000]))[0] == 5
    assert T.barssince(np.array([False, True, False, False])).tolist()[1:] == [0, 1, 2]
    assert np.isnan(T.ffill_val(np.array([np.nan, 1.0, np.nan]))[0])


def check_sim_k() -> None:
    # 평평한 가격에서 롱 신호, 다음 봉에 +ATR 목표 도달
    n = 40
    c = np.full(n, 100.0)
    h, lo = c + 1, c - 1
    h[22] = 104
    b = bars(c, h, lo, c, t0=T0)
    B = E.Bars(b)
    sig = np.zeros(n, np.int8)
    sig[20] = 1
    tr = V.simulate(B, {"sig": sig, "native": None}, ("K", 1.0, 0.0), 0, n)
    at = B.atr[20]
    assert len(tr) == 1 and tr[0]["why"] == "tp" and tr[0]["x"] == 22
    assert abs(tr[0]["xp"] - (100 + at)) < 1e-9
    assert abs(tr[0]["r"] - (at / 100 - E.FEE * (100 + at) / 100 - E.FEE)) < 1e-12
    # 손절 2ATR
    lo2 = lo.copy()
    lo2[22] = 90
    b2 = bars(c, h, lo2, c, t0=T0)
    tr2 = V.simulate(E.Bars(b2), {"sig": sig, "native": None}, ("K", 1.0, 2.0), 0, n)
    assert tr2[0]["why"] == "stop" and abs(tr2[0]["xp"] - (100 - 2 * E.Bars(b2).atr[20])) < 1e-9
    # 자체 % 괄호 훅
    nat = {"init": T.pct_bracket(0.03, None), "reverse": False}
    tr3 = V.simulate(B, {"sig": sig, "native": nat}, ("N",), 0, n)
    assert tr3[0]["why"] == "tp" and abs(tr3[0]["xp"] - 103) < 1e-9


def check_be() -> None:
    tr = [{"r": 0.01}, {"r": 0.01}, {"r": -0.02}]
    be = V.be_win(tr)
    assert abs(be["be1"] - 2 / 3) < 1e-12 and be["be10"] > be["be1"]


def check_gate() -> None:
    S = {"a": {"sig": np.array([1, -1, 1, -1], np.int8), "native": None}}
    g = V.gated(S, "a", {"G": np.array([1, 1, -1, -1])}, "G")
    assert g["sig"].tolist() == [1, 0, 0, -1]


if __name__ == "__main__":
    check_frame()
    check_indicators()
    check_sim_k()
    check_be()
    check_gate()
    print("pine_tv checks ok")
