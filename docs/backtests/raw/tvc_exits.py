"""트레이딩뷰 구성요소 시험의 청산 모양 10개(사전 등록 그대로).

문서: docs/backtests/2026-11-08-btc-tv-components-hypotheses.md
모든 모양은 pine_tv_eval.simulate의 자체 청산(N) 훅으로 만든다: 진입 때 init 훅이 목표·손절을 정하고,
time_bars로 시간 청산, exl/exs로 신호 자체의 중립 청산(X9)을 건다. 반대 신호가 오면 청산 후 뒤집는다.
모든 모양에 손절이 있다(재난 손절 DS7 = 진입가 ∓7%, 10배면 −70% + 비용으로 청산선 −95% 안쪽).
"""

from __future__ import annotations

import math

import numpy as np

EXIT_SHAPES = ("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10")
EXIT_DESC = {
    "X1": "목표 1×ATR, 재난 손절 −7%, 96봉",
    "X2": "목표 0.5×ATR, 재난 손절 −7%, 96봉",
    "X3": "목표 1×ATR, 손절 1.5×ATR, 96봉",
    "X4": "목표 +0.6%, 손절 −1.0%, 96봉",
    "X5": "구조 손절, 목표 1R, 96봉",
    "X6": "구조 손절, 목표 0.5R, 96봉",
    "X7": "구조 손절, 1R·2R·3R에 1/3씩, 1R 닿으면 본전 손절, 192봉",
    "X8": "구조 손절, 1R에 절반, 나머지 3×ATR 추적(샹들리에식), 192봉",
    "X9": "신호 자체 중립 청산(없으면 32봉), 최대 32봉, 재난 손절 −7%",
    "X10": "16봉 시간 청산, 재난 손절 −7%",
}
DS = 0.07
R_MIN, R_MAX = 0.003, 0.03  # 구조 손절 거리(진입가 대비) 하한·상한


def make_native(shape: str, atr: np.ndarray, st_l: np.ndarray, st_s: np.ndarray,
                own_l: np.ndarray | None = None, own_s: np.ndarray | None = None) -> dict:  # fmt: skip
    """st_l/st_s: 신호 봉 j 마감에 아는 구조 손절가(롱/숏). own_*: 신호 자체 중립 청산 조건(봉 마감)."""
    n = len(atr)
    nat: dict = {"reverse": True, "sl": np.full(n, np.nan)}
    atr_l = atr.tolist()
    sl_l = st_l.tolist()
    sl_s = st_s.tolist()

    def struct_r(j: int, ep: float, dr: int) -> float:
        lvl = sl_l[j] if dr == 1 else sl_s[j]
        dist = (ep - lvl) * dr / ep if not math.isnan(lvl) else R_MAX
        return min(max(dist, R_MIN), R_MAX) * ep

    if shape in ("X1", "X2", "X3"):
        k = {"X1": 1.0, "X2": 0.5, "X3": 1.0}[shape]

        def init(p, j, ep, dr):
            a = atr_l[j]
            p["tps"] = [(1.0, ep + dr * k * a)]
            p["sl"] = ep - dr * 1.5 * a if shape == "X3" else ep * (1 - dr * DS)

        nat["time_bars"] = 96
    elif shape == "X4":

        def init(p, j, ep, dr):
            p["tps"] = [(1.0, ep * (1 + dr * 0.006))]
            p["sl"] = ep * (1 - dr * 0.010)

        nat["time_bars"] = 96
    elif shape in ("X5", "X6"):
        m = 1.0 if shape == "X5" else 0.5

        def init(p, j, ep, dr):
            r = struct_r(j, ep, dr)
            p["sl"] = ep - dr * r
            p["tps"] = [(1.0, ep + dr * m * r)]

        nat["time_bars"] = 96
    elif shape == "X7":

        def init(p, j, ep, dr):
            r = struct_r(j, ep, dr)
            p["sl"] = ep - dr * r
            p["tps"] = [(1 / 3, ep + dr * r), (1 / 3, ep + dr * 2 * r), (1 / 3, ep + dr * 3 * r)]
            p["be"] = r

        nat["time_bars"] = 192
    elif shape == "X8":

        def init(p, j, ep, dr):
            r = struct_r(j, ep, dr)
            p["sl"] = ep - dr * r
            p["tps"] = [(0.5, ep + dr * r)]
            p["trail_k"] = 3.0

        nat["time_bars"] = 192
    elif shape in ("X9", "X10"):

        def init(p, j, ep, dr):
            p["sl"] = ep * (1 - dr * DS)

        if shape == "X9":
            nat["time_bars"] = 32
            if own_l is not None:
                nat["exl"], nat["exs"] = own_l, own_s
        else:
            nat["time_bars"] = 16
    else:
        raise ValueError(shape)
    nat["init"] = init
    return nat


def label_one(o, h, lo, c, j: int, dr: int, tp: float, sl: float, tmax: int, fee: float) -> float:
    """신호 j 다음 봉 시가 진입, 목표/손절(가격 비율), 시간 청산 tmax봉. 같은 봉이면 손절 먼저.
    순손익(수수료 포함, 펀딩 제외)을 돌려준다. ML 표지(목표 먼저 = 1)용."""
    n = len(o)
    e = j + 1
    if e >= n:
        return float("nan")
    ep = o[e]
    tpx = ep * (1 + dr * tp)
    spx = ep * (1 - dr * sl)
    z = min(n, e + tmax + 1)
    for i in range(e, z):
        if dr == 1:
            if o[i] <= spx or lo[i] <= spx:
                px = min(o[i], spx)
                return dr * (px / ep - 1) - fee * (1 + px / ep)
            if o[i] >= tpx or h[i] >= tpx:
                px = max(o[i], tpx)
                return dr * (px / ep - 1) - fee * (1 + px / ep)
        else:
            if o[i] >= spx or h[i] >= spx:
                px = max(o[i], spx)
                return dr * (px / ep - 1) - fee * (1 + px / ep)
            if o[i] <= tpx or lo[i] <= tpx:
                px = min(o[i], tpx)
                return dr * (px / ep - 1) - fee * (1 + px / ep)
    x = min(z, n - 1)
    px = o[x] if x < n else c[-1]
    return dr * (px / ep - 1) - fee * (1 + px / ep)
