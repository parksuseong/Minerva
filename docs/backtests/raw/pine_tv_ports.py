"""트레이딩뷰 공개 BTC 전략 21개의 신호·자체 청산을 파이썬으로 옮긴 것(원문은 저장소 밖).

문서: docs/backtests/2026-11-08-pine-tv-hypotheses.md (해석은 그 문서의 "해석이 갈리는 곳")
- 각 `core_xx(X)`는 시간 틀 X(15분봉 또는 마감된 1h·2h·4h·일봉)의 마감 봉 값만 쓴다.
- `build(b)`가 X를 만들고 신호를 그 X 봉의 마지막 15분봉에 붙인다(다음 15분봉 시가 체결).
- 자체 청산은 pine_tv_eval.simulate의 훅으로 넘긴다:
  init(p, j, ep, dr): 진입 때 손절·목표 설정, on_close(p, i): 15분봉 i 마감에 손절 갱신·청산 사유.
- 지표 정의는 pine_ports(파인 내장 함수와 같게)를 쓴다.
"""

from __future__ import annotations

import math

import numpy as np
import pine_ports as P
from numpy.lib.stride_tricks import sliding_window_view as swv

M15 = 900_000
H1 = 3_600_000
DAY = 86_400_000


# ---------------- 시간 틀 ----------------
def frame(b: dict, minutes: int) -> dict:
    """15분봉 → minutes 봉(UTC 정렬, 다 찬 봉만). last15: 그 봉의 마지막 15분봉 인덱스."""
    n = len(b["c"])
    if minutes == 15:
        X = {k: b[k] for k in ("t", "o", "h", "l", "c", "v")}
        X["last15"] = np.arange(n)
        return X
    per = minutes * 60_000
    k = b["t"] // per
    keys, idx = np.unique(k, return_index=True)
    ends = np.r_[idx[1:], n]
    full = (ends - idx) == minutes // 15
    X = {
        "t": keys * per,
        "o": b["o"][idx],
        "h": np.maximum.reduceat(b["h"], idx),
        "l": np.minimum.reduceat(b["l"], idx),
        "c": b["c"][ends - 1],
        "v": np.add.reduceat(b["v"], idx),
        "last15": ends - 1,
    }
    return {kk: v[full] for kk, v in X.items()}


def to15(X: dict, x: np.ndarray, n: int, dtype=np.int8) -> np.ndarray:
    out = np.zeros(n, dtype)
    out[X["last15"]] = x
    return out


def state15(X: dict, x: np.ndarray, n: int) -> np.ndarray:
    """15분봉 i 마감 때 확정된 마지막 X 봉 값."""
    out = np.full(n, np.nan)
    pos = np.searchsorted(X["last15"], np.arange(n), side="right") - 1
    ok = pos >= 0
    out[ok] = np.asarray(x, float)[pos[ok]]
    return out


def level_next(X: dict, x: np.ndarray, n: int) -> np.ndarray:
    """15분봉 i가 열릴 때 이미 확정된 마지막 X 봉 값."""
    out = np.full(n, np.nan)
    pos = np.searchsorted(X["last15"], np.arange(n), side="left") - 1
    ok = pos >= 0
    out[ok] = np.asarray(x, float)[pos[ok]]
    return out


# ---------------- 추가 지표(파인 정의) ----------------
def _conv(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """out[t] = Σ_k w[k]·x[t-n+1+k] (w[-1]이 최신 봉). 창 안에 nan이 있으면 nan."""
    x = np.asarray(x, float)
    n = len(w)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    bad = np.isnan(x)
    v = np.convolve(np.where(bad, 0.0, x), w[::-1], mode="valid")
    cb = np.cumsum(np.r_[0, bad.astype(int)])
    nb = cb[n:] - cb[:-n]
    out[n - 1 :] = np.where(nb == 0, v, np.nan)
    return out


def wma(x, n: int) -> np.ndarray:
    w = np.arange(1, n + 1, dtype=float)
    return _conv(x, w / w.sum())


def hma(x, n: int) -> np.ndarray:
    return wma(2 * wma(x, n // 2) - wma(x, n), int(round(math.sqrt(n))))


def linreg(x, n: int, off: int = 0) -> np.ndarray:
    xs = np.arange(n, dtype=float)
    xb = (n - 1) / 2
    sxx = ((xs - xb) ** 2).sum()
    w = 1 / n + (xs - xb) * (n - 1 - off - xb) / sxx
    return _conv(x, w)


def lin_slope(x, n: int) -> np.ndarray:
    xs = np.arange(n, dtype=float)
    xb = (n - 1) / 2
    return _conv(x, (xs - xb) / ((xs - xb) ** 2).sum())


def swma(x) -> np.ndarray:
    return _conv(x, np.array([1, 2, 2, 1], float) / 6)


def rsum(x, n: int) -> np.ndarray:
    return _conv(x, np.ones(n))


def barssince(cond: np.ndarray) -> np.ndarray:
    out = np.full(len(cond), np.inf)
    last = -1
    for i, v in enumerate(cond.tolist()):
        if v:
            last = i
        if last >= 0:
            out[i] = i - last
    return out


def ffill_val(x: np.ndarray) -> np.ndarray:
    """ta.valuewhen(not na(x), x, 0)."""
    out = np.full(len(x), np.nan)
    last = np.nan
    for i, v in enumerate(x.tolist()):
        if v == v:
            last = v
        out[i] = last
    return out


def dmi_adx(h, lo, c, n: int = 14):
    up = np.diff(h, prepend=np.nan)
    dn = -np.diff(lo, prepend=np.nan)
    with np.errstate(invalid="ignore"):
        pdm = np.where(np.isnan(up), np.nan, np.where((up > dn) & (up > 0), up, 0.0))
        mdm = np.where(np.isnan(dn), np.nan, np.where((dn > up) & (dn > 0), dn, 0.0))
    trr = P.rma(P.tr(h, lo, c), n)
    with np.errstate(invalid="ignore", divide="ignore"):
        plus = P._fixnan(100 * P.rma(pdm, n) / trr)
        minus = P._fixnan(100 * P.rma(mdm, n) / trr)
        s = plus + minus
        adx = 100 * P.rma(np.abs(plus - minus) / np.where(s == 0, 1, s), n)
    return plus, minus, adx


def percentrank(x: np.ndarray, n: int) -> np.ndarray:
    """ta.percentrank: 직전 n봉 중 현재 값 이하인 비율(%)."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for a in range(n, len(x), 50_000):
        z = min(a + 50_000, len(x))
        w = swv(x[a - n : z - 1], n)  # 행 k: x[a+k-n .. a+k-1]
        cur = x[a:z][:, None]
        with np.errstate(invalid="ignore"):
            out[a:z] = 100 * (w <= cur).sum(1) / n
    return out


def t3(src, n: int, v: float) -> np.ndarray:
    e1 = P.ema(src, n)
    e2 = P.ema(e1, n)
    e3 = P.ema(e2, n)
    e4 = P.ema(e3, n)
    e5 = P.ema(e4, n)
    e6 = P.ema(e5, n)
    c1 = -(v**3)
    c2 = 3 * v * v + 3 * v**3
    c3 = -6 * v * v - 3 * v - 3 * v**3
    c4 = 1 + 3 * v + v**3 + 3 * v * v
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def sig_of(lc, sc, short_wins: bool = True) -> np.ndarray:
    lc = np.nan_to_num(np.asarray(lc, float)).astype(bool)
    sc = np.nan_to_num(np.asarray(sc, float)).astype(bool)
    if short_wins:
        return np.where(sc, -1, np.where(lc, 1, 0)).astype(np.int8)
    return np.where(lc & sc, 0, np.where(lc, 1, np.where(sc, -1, 0))).astype(np.int8)


def dow(t_ms: np.ndarray) -> np.ndarray:
    """파인 dayofweek: 1=일요일 … 7=토요일 (UTC)."""
    # 1970-01-01은 목요일(파인 5)
    return ((t_ms // DAY + 4) % 7) + 1


def hour(t_ms: np.ndarray) -> np.ndarray:
    return (t_ms // H1) % 24


# ---------------- 자체 청산 공통 ----------------
def pct_bracket(tp: float | None, sl: float | None):
    """진입 체결가 기준 % 목표·손절."""

    def init(p, j, ep, dr):
        if tp is not None:
            p["tps"] = [(1.0, ep * (1 + dr * tp))]
        if sl is not None:
            p["sl"] = ep * (1 - dr * sl)

    return init


# ---------------- 1. Supertrend + EMA ----------------
def core_01(X):
    h, lo, c = X["h"], X["l"], X["c"]
    d = P.supertrend_dir(h, lo, c, 1.8, 10)
    e = P.ema(c, 200)
    dd = np.diff(d, prepend=np.nan)
    with np.errstate(invalid="ignore"):
        bull, bear = dd < 0, dd > 0
        lc, sc = bull & (c > e), bear & (c < e)
    return {"sig": sig_of(lc, sc), "exl": bear, "exs": bull, "state": -d}


# ---------------- 2. RSI Volume Ladder ----------------
def core_02(X):
    c, v = X["c"], X["v"]
    r = P.rsi(c, 14)
    with np.errstate(invalid="ignore"):
        hv = v > P.sma(v, 20) * 1.5
        lc, sc = (r < 30) & hv, (r > 70) & hv
    return {"sig": sig_of(lc, sc)}


# ---------------- 3. BTC Sell Strategy (숏, 거울 롱) ----------------
def core_03(X, htf_ema15: np.ndarray, n15: int):
    """htf_ema15: 15분봉 위에 놓인 상위 틀 EMA200(마감 값). X 마감 15분봉에서 읽는다."""
    h, lo, c, t = X["h"], X["l"], X["c"], X["t"]
    hc = P.highest(c, 22)
    lc_ = P.lowest(c, 22)
    with np.errstate(invalid="ignore", divide="ignore"):
        vix_s = (hc - lo) / hc * 100 < 2.0
        vix_l = (h - lc_) / lc_ * 100 < 2.0
    r = P.rsi(c, 14)
    k = P.sma(P.stoch(r, r, r, 14), 3)
    d = P.sma(k, 3)
    k1, k2 = P.shift(k, 1), P.shift(k, 2)
    with np.errstate(invalid="ignore"):
        was_ob = barssince((k1 >= 80) & (k2 < 80)) <= 10
        was_os = barssince((k1 <= 20) & (k2 > 20)) <= 10
        dt_ = was_ob & P.cross_under(k, np.full(len(k), 80.0)) & (k < d)
        db_ = was_os & P.cross_over(k, np.full(len(k), 20.0)) & (k > d)
    ema_x = htf_ema15[X["last15"]]
    hr = hour(t)
    sess = (hr >= 8) & (hr < 22)
    with np.errstate(invalid="ignore"):
        s_raw = dt_ & vix_s & (c < ema_x) & sess
        l_raw = db_ & vix_l & (c > ema_x) & sess
    # 쿨다운 10봉(신호 봉 기준, 방향 공통)
    sig = np.zeros(len(c), np.int8)
    last = -(10**9)
    for i in np.flatnonzero(s_raw | l_raw).tolist():
        if i - last > 10:
            sig[i] = -1 if s_raw[i] else 1
            last = i
    a = P.atr(h, lo, c, 14)
    hh5, ll5 = P.highest(h, 5), P.lowest(lo, 5)
    with np.errstate(invalid="ignore"):
        stop_s = np.maximum(hh5, c + 2 * a)
        stop_l = np.minimum(ll5, c - 2 * a)
    tp_s = c - 1.5 * (stop_s - c)
    tp_l = c + 1.5 * (c - stop_l)
    return {"sig": sig, "a": a, "stop_s": stop_s, "stop_l": stop_l, "tp_s": tp_s, "tp_l": tp_l}


def native_03(X, r, n):
    """매 X 마감마다 괄호 주문 갱신(원문은 매 봉 strategy.exit 호출), 0.8R에서 한 봉만 본전·추적."""
    m = {k: state15(X, r[k], n) for k in ("a", "stop_s", "stop_l", "tp_s", "tp_l")}
    m["c"] = state15(X, X["c"], n)
    is_close = to15(X, np.ones(len(X["c"])), n, bool)

    def init(p, j, ep, dr):
        p["sl"] = float(m["stop_l"][j] if dr == 1 else m["stop_s"][j])
        p["tps"] = [(1.0, float(m["tp_l"][j] if dr == 1 else m["tp_s"][j]))]
        p["moved"] = False

    def on_close(p, i):
        if not is_close[i]:
            return None
        dr, ep = p["d"], p["ep"]
        stp = float(m["stop_l"][i] if dr == 1 else m["stop_s"][i])
        tp = float(m["tp_l"][i] if dr == 1 else m["tp_s"][i])
        if math.isnan(stp) or math.isnan(tp):
            return None
        p["sl"], p["tps"] = stp, [(1.0, tp)]
        c, a = float(m["c"][i]), float(m["a"][i])
        den = (ep - stp) if dr == 1 else (stp - ep)
        if den > 0 and not p["moved"]:
            rnow = dr * (c - ep) / den
            if rnow >= 0.8:
                p["sl"] = max(ep, c - 1.5 * a) if dr == 1 else min(ep, c + 1.5 * a)
                p["moved"] = True
        return None

    return {"init": init, "on_close": on_close, "reverse": False}


# ---------------- 5. HyperSAR Reactor ----------------
def core_05(X):
    o, h, lo, c = X["o"], X["h"], X["l"], X["c"]
    a = P.atr(h, lo, c, 16)
    with np.errstate(invalid="ignore", divide="ignore"):
        drift = np.abs(np.nan_to_num(lin_slope(c, 18))) / np.where(np.isnan(a), 1e-12, a)
        strength = 1 / (1 + np.exp(-4.5 * (drift - 0.45)))
        vol_ok = a / P.sma(a, 30) >= 1.0
        bias_dn = (c < P.ema(c, 91)) & (lin_slope(c, 54) < 0)
    step = 0.04 + 0.03 * strength
    n = len(c)
    psar_o = np.full(n, np.nan)
    up_o = np.zeros(n, bool)
    flip_o = np.zeros(n, bool)
    ol, hl, ll, cl, sl_ = o.tolist(), h.tolist(), lo.tolist(), c.tolist(), step.tolist()
    psar = ep = af = None
    up = False
    for i in range(n):
        if psar is None:
            up = cl[i] >= ol[i]
            ep = hl[i] if up else ll[i]
            psar = ll[i] if up else hl[i]
            af = 0.02
        flipped = False
        l1 = ll[i - 1] if i >= 1 else ll[i]
        l2 = ll[i - 2] if i >= 2 else ll[i]
        h1 = hl[i - 1] if i >= 1 else hl[i]
        h2 = hl[i - 2] if i >= 2 else hl[i]
        stp = sl_[i] if sl_[i] == sl_[i] else 0.04
        if up:
            nx = psar + af * (ep - psar)
            nx = min(nx, l1, l2)
            if cl[i] < nx:
                up, psar, ep, af, flipped = False, ep, ll[i], 0.02, True
            else:
                psar = max(psar, psar + 0.35 * (nx - psar))
                if hl[i] > ep:
                    ep = hl[i]
                    af = af + 0.5 * (min(af + stp, 1.0) - af)
        else:
            nx = psar + af * (ep - psar)
            nx = max(nx, h1, h2)
            if cl[i] > nx:
                up, psar, ep, af, flipped = True, ep, hl[i], 0.02, True
            else:
                psar = min(psar, psar + 0.35 * (nx - psar))
                if ll[i] < ep:
                    ep = ll[i]
                    af = af + 0.5 * (min(af + stp, 1.0) - af)
        psar_o[i], up_o[i], flip_o[i] = psar, up, flipped
    with np.errstate(invalid="ignore"):
        pl = up_o & flip_o & (c > psar_o + 0.5 * a) & vol_ok
        ps = ~up_o & flip_o & (c < psar_o - 0.5 * a) & vol_ok & bias_dn
    return {"sig": sig_of(pl, ps), "psar": psar_o, "state": np.where(up_o, 1, -1)}


# ---------------- 6. Mutanabby Algo Pro (롱, 거울 숏) ----------------
def core_06(X):
    o, h, lo, c = X["o"], X["h"], X["l"], X["c"]
    with np.errstate(invalid="ignore", divide="ignore"):
        stable = np.abs(c - o) / P.tr(h, lo, c) > 0.5
    r = P.rsi(c, 14)
    o1, c1, c5 = P.shift(o, 1), P.shift(c, 1), P.shift(c, 5)
    with np.errstate(invalid="ignore"):
        bull = (c1 < o1) & (c > o) & (c > o1)
        bear = (c1 > o1) & (c < o) & (c < o1)
        ent = bull & stable & (r < 50) & (c < c5)
        ex = bear & stable & (r > 50) & (c > c5)
    return {"sig": sig_of(ent, ex), "exl": ex, "exs": ent}


def native_close_pct(X, n, pct: float):
    """진입 신호 봉 종가 ±pct 를 X 마감 종가가 넘으면 다음 시가 청산(원문 종가 확인 손절)."""
    cx = state15(X, X["c"], n)
    is_close = to15(X, np.ones(len(X["c"])), n, bool)

    def init(p, j, ep, dr):
        p["ref"] = float(cx[j])

    def on_close(p, i):
        if not is_close[i]:
            return None
        if p["d"] == 1 and cx[i] <= p["ref"] * (1 - pct):
            return "stop"
        if p["d"] == -1 and cx[i] >= p["ref"] * (1 + pct):
            return "stop"
        return None

    return init, on_close


# ---------------- 7. TSI ----------------
def core_07(X):
    c = X["c"]
    pc = np.diff(c, prepend=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        tsi = 100 * P.ema(P.ema(pc, 25), 13) / P.ema(P.ema(np.abs(pc), 25), 13)
    hi = P.shift(P.highest(tsi, 100), 1)
    lw = P.shift(P.lowest(tsi, 100), 1)
    return {"sig": sig_of(P.cross_over(tsi, hi), P.cross_under(tsi, lw), short_wins=False)}


# ---------------- 8. T3 Fibonacci ----------------
def core_08(X):
    src = (X["h"] + X["l"] + 2 * X["c"]) / 4
    a = t3(src, 33, 0.6)
    b = t3(src, 19, 0.418)
    return {"sig": sig_of(P.cross_over(b, a), P.cross_under(b, a), short_wins=False)}


# ---------------- 9. Ultimate Stochastics ----------------
def core_09(X):
    h, lo, c = X["h"], X["l"], X["c"]
    fk = P.stoch(c, h, lo, 9)
    sk = P.sma(fk, 18)
    sd = P.sma(sk, 4)
    with np.errstate(invalid="ignore"):
        b0 = P.cross_over(sk, sd) & (sd != 0) & (sk < 90)
        s0 = P.cross_under(sk, sd) & (sd != 0) & (sk > 60)
    b1 = b0 | np.r_[False, b0[:-1]]
    s1 = s0 | np.r_[False, s0[:-1]]
    return {"sig": sig_of(b1, s1, short_wins=False)}


# ---------------- 10. Time Based DayTrade (1시간) ----------------
def core_10(X):
    h, lo, c, t = X["h"], X["l"], X["c"], X["t"]
    a13, a7 = P.atr(h, lo, c, 13), P.atr(h, lo, c, 7)
    dw, hr = dow(t), hour(t)
    when = (hr == 1) & ((dw == 2) | (dw == 4))
    with np.errstate(invalid="ignore"):
        lc = when & (a13 > a7)
        sc = when & (a13 <= a7)  # 거울 숏(포팅 때 정함): 같은 요일·시각, ATR 필터 반대
    ex = hr == 0
    return {"sig": sig_of(lc, sc), "exl": ex, "exs": ex}


# ---------------- 11. Trend Confidence ----------------
def core_11(X):
    c = X["c"]
    n = 30
    lr = linreg(c, n, 0)
    slope = lr - linreg(c, n, 1)
    dev = np.full(len(c), np.nan)
    w = swv(c, n)[:, ::-1]  # 열 i = c[t-i]
    line = lr[n - 1 :, None] - np.arange(n)[None, :] * slope[n - 1 :, None]
    dev[n - 1 :] = np.sqrt(((w - line) ** 2).mean(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        tc = (slope / c) / (dev / c)
    k = lambda v: np.full(len(c), v)  # noqa: E731
    lc = P.cross_over(tc, k(0.25))
    sc = P.cross_under(tc, k(-0.25))
    return {
        "sig": sig_of(lc, sc, short_wins=False),
        "exl": P.cross_under(tc, k(-0.10)),
        "exs": P.cross_over(tc, k(-0.05)),
        "state": np.sign(np.nan_to_num(tc)),
    }


# ---------------- 12. ATR Mean Reversion (15분봉) ----------------
def core_12(X):
    o, h, lo, c = X["o"], X["h"], X["l"], X["c"]
    a = P.atr(h, lo, c, 10)
    e = P.ema(c, 10)
    with np.errstate(invalid="ignore"):
        lc = lo < o - a
        sc = h > o + a
        exl = ((h > e) & (c < P.shift(lo, 1))) | (h > o + a)
        exs = ((lo < e) & (c > P.shift(h, 1))) | (lo < o - a)
    return {"sig": sig_of(lc, sc, short_wins=False), "exl": exl, "exs": exs, "a": a}


def native_12(b, r):
    a, lo, h = r["a"], b["l"], b["h"]

    def on_close(p, i):
        # 손절: low < 진입 봉 시가 − 1.5×ATR(현재 봉) → 다음 시가
        if p["d"] == 1 and lo[i] < p["ep"] - 1.5 * a[i]:
            return "stop"
        if p["d"] == -1 and h[i] > p["ep"] + 1.5 * a[i]:
            return "stop"
        return None

    return on_close


# ---------------- 14. RTB Momentum Breakout ----------------
def core_14(X):
    h, lo, c = X["h"], X["l"], X["c"]
    ef, es = P.ema(c, 20), P.ema(c, 50)
    r = P.rsi(c, 14)
    a = P.atr(h, lo, c, 14)
    res = P.shift(P.highest(h, 5), 1)
    sup = P.shift(P.lowest(lo, 5), 1)
    with np.errstate(invalid="ignore"):
        lc = (c > res) & (ef > es) & (r < 70)
        sc = (c < sup) & (ef < es) & (r > 30)
    return {"sig": sig_of(lc, sc), "a": a, "state": np.sign(np.nan_to_num(ef - es))}


def native_14(X, r, b, n):
    """손절 진입가 ∓ 1.5×ATR(X), 추적: 0.15×ATR 이익에서 켜지고 최고가 − 0.15×ATR(틱 단위 해석)."""
    ax = state15(X, r["a"], n)

    def init(p, j, ep, dr):
        p["ax"] = float(ax[j])
        p["sl"] = ep - dr * 1.5 * p["ax"]
        p["on"] = False
        p["ext"] = ep

    def on_close(p, i):
        dr, ep, k = p["d"], p["ep"], 0.15 * p["ax"]
        p["ext"] = max(p["ext"], b["h"][i]) if dr == 1 else min(p["ext"], b["l"][i])
        if not p["on"] and dr * (p["ext"] - ep) >= k:
            p["on"] = True
        if p["on"]:
            tr_ = p["ext"] - dr * k
            p["sl"] = max(p["sl"], tr_) if dr == 1 else min(p["sl"], tr_)
        return None

    return {"init": init, "on_close": on_close, "reverse": True}


# ---------------- 15. ADX for BTC ----------------
def core_15(X):
    h, lo, c = X["h"], X["l"], X["c"]
    _, _, adx = dmi_adx(h, lo, c, 14)
    s1, s5 = P.sma(c, 200), P.sma(c, 1000)
    k = lambda v: np.full(len(c), v)  # noqa: E731
    up = P.cross_over(adx, k(14))
    with np.errstate(invalid="ignore"):
        lc, sc = up & (s1 > s5), up & (s1 < s5)
    ex = P.cross_under(adx, k(45))
    return {"sig": sig_of(lc, sc), "exl": ex, "exs": ex}


# ---------------- 16. Fenix Wyckoff Squeeze (15분봉, 길이 그대로) ----------------
def core_16(X):
    h, lo, c, v = X["h"], X["l"], X["c"], X["v"]
    hlc3 = (h + lo + c) / 3
    e = P.ema(hlc3, 1188)
    dv = P.ema(np.abs(hlc3 - e), 1188)
    with np.errstate(invalid="ignore", divide="ignore"):
        tci = P.ema((hlc3 - e) / np.maximum(dv, 1e-6) / 0.025, 366) + 50
    ch = np.diff(hlc3, prepend=np.nan)
    with np.errstate(invalid="ignore"):
        upv = np.where(ch > 0, hlc3 * v, 0.0)
        lov = np.where(ch < 0, hlc3 * v, 0.0)
    su, sl_ = rsum(upv, 3512), rsum(lov, 3512)
    with np.errstate(invalid="ignore", divide="ignore"):
        mf = 100 - 100 / (1 + su / np.maximum(sl_, 1e-6))
    rh = P.rsi(hlc3, 3512)
    wt1 = (tci + mf + rh) / 3
    wt2 = P.sma(wt1, 6)
    wt3 = linreg(wt1, 1320, 0)
    w1p = P.shift(wt1, 1)
    with np.errstate(invalid="ignore"):
        ph_b = (wt1 > wt2) & (wt1 > w1p)
        ph_s = (wt1 < wt2) & (wt1 < w1p)
        wd = wt1 - w1p
        exl = P.cross_under(wt1, wt2) | (rsum((wd < 0).astype(float), 296) >= 296)
        exs = P.cross_over(wt1, wt2) | (rsum((wd > 0).astype(float), 296) >= 296)
    basis = P.sma(c, 138)
    sd = P.stdev(c, 138)
    a138 = P.atr(h, lo, c, 138)
    ub, lb = basis + 10.2 * sd, basis - 10.2 * sd
    uk, lk = basis + 20.5 * a138, basis - 20.5 * a138
    with np.errstate(invalid="ignore"):
        sq_on = (ub < uk) & (lb > lk)
        sq_off = (ub > uk) | (lb < lk)
    rec = rsum(np.r_[0.0, sq_on[:-1].astype(float)], 68) > 0
    fired = sq_off & rec
    hm = hma(c, 600)
    hm1 = P.shift(hm, 1)
    e2 = P.ema(c, 476 * 4)
    vr = percentrank(P.atr(h, lo, c, 14), 135)
    _, _, adx = dmi_adx(h, lo, c, 14)
    with np.errstate(invalid="ignore"):
        ok = (vr >= 4) & (vr <= 99) & (adx >= 18)
        lc = fired & (c > e2) & (hm > hm1) & ph_b & (wt3 > 50) & ok
        sc = fired & (c < e2) & (hm < hm1) & ph_s & (wt3 < 50) & ok
    return {"sig": sig_of(lc, sc), "exl": exl, "exs": exs, "a412": P.atr(h, lo, c, 412)}


def native_16(b, r):
    a412, c = r["a412"], b["c"]

    def init(p, j, ep, dr):
        p["ea"] = float(a412[j])
        p["trc"] = c[j] - dr * p["ea"] * 18.6
        p["tpc"] = c[j] + dr * p["ea"] * 38.8

    def on_close(p, i):
        dr = p["d"]
        if dr == 1:
            p["trc"] = max(p["trc"], c[i] - p["ea"] * 18.6)
            hit = c[i] <= p["trc"] or c[i] >= p["tpc"]
        else:
            p["trc"] = min(p["trc"], c[i] + p["ea"] * 18.6)
            hit = c[i] >= p["trc"] or c[i] <= p["tpc"]
        if hit:
            return "stop"
        if i - p["e"] + 1 >= 480:
            return "time"
        return None

    return init, on_close


# ---------------- 17. EMA 5/9 (15분봉 + 1시간 EMA) ----------------
def core_17(b, n):
    c, h, lo = b["c"], b["h"], b["l"]
    f, s = P.ema(c, 5), P.ema(c, 9)
    e20, e50 = P.ema(c, 20), P.ema(c, 50)
    X1 = frame(b, 60)
    hu = state15(X1, (P.ema(X1["c"], 20) > P.ema(X1["c"], 50)).astype(float), n)
    hd = state15(X1, (P.ema(X1["c"], 20) < P.ema(X1["c"], 50)).astype(float), n)
    d = P.supertrend_dir(h, lo, c, 3, 10)
    a = P.atr(h, lo, c, 14)
    bull, bear = P.cross_over(f, s), P.cross_under(f, s)
    with np.errstate(invalid="ignore"):
        ok = a / c * 100 > 0.30
        buy_ok = (e20 > e50) & (hu == 1) & (d == 1) & ok  # 원문 stBuy = dir == 1 그대로
        sell_ok = (e20 < e50) & (hd == 1) & (d == -1) & ok
    return {"sig": sig_of(bull & buy_ok, bear & sell_ok), "exl": bear, "exs": bull}


# ---------------- 18. RSI-Adaptive T3 + Squeeze ----------------
def core_18(X):
    h, lo, c = X["h"], X["l"], X["c"]
    r = P.rsi(c, 14)
    ln = np.round(5 + 45 * (1 - r / 100))
    v = 0.7

    def pine_ema(s, lens):
        out = np.empty(len(s))
        prev = math.nan
        sl, ll = s.tolist(), lens.tolist()
        for i in range(len(sl)):
            L = ll[i]
            if prev != prev:
                prev = sl[i]
            elif L == L:
                al = 2 / (L + 1)
                prev = al * sl[i] + (1 - al) * prev
            else:
                prev = math.nan  # 길이 nan이면 파인도 na
            out[i] = prev
        return out

    e1 = pine_ema(c, ln)
    e2 = pine_ema(e1, ln)
    e3 = pine_ema(e2, ln)
    e4 = pine_ema(e3, ln)
    e5 = pine_ema(e4, ln)
    e6 = pine_ema(e5, ln)
    c1, c2 = -(v**3), 3 * v * v + 3 * v**3
    c3, c4 = -6 * v * v - 3 * v - 3 * v**3, 1 + 3 * v + v**3 + 3 * v * v
    tt = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
    basis = P.sma(c, 27)
    dev = 1.5 * P.stdev(c, 27)  # 원문: BB 폭에 multKC 사용
    ma = P.sma(c, 20)
    rg = P.sma(P.tr(h, lo, c), 20)
    with np.errstate(invalid="ignore"):
        off = (basis - dev < ma - rg * 1.5) & (basis + dev > ma + rg * 1.5)
    mid = (P.highest(h, 20) + P.lowest(lo, 20)) / 2
    val = linreg(c - (mid + P.sma(c, 20)) / 2, 20, 0)
    t1 = P.shift(tt, 1)
    with np.errstate(invalid="ignore"):
        lc = P.cross_over(tt, t1) & (val > 0) & off
        sc = P.cross_under(tt, t1) & (val < 0) & off
    return {"sig": sig_of(lc, sc, short_wins=False)}


# ---------------- 19. Ichimoku ----------------
def core_19(X):
    h, lo = X["h"], X["l"]

    def don(n):
        return (P.lowest(lo, n) + P.highest(h, n)) / 2

    tk, kj = don(9), don(26)
    sa, sb = (tk + kj) / 2, don(52)
    da, db = P.shift(sa, 25), P.shift(sb, 25)
    bull, bear = P.cross_over(tk, kj), P.cross_under(tk, kj)
    with np.errstate(invalid="ignore"):
        weak_bear = bear & (tk > da) & (tk > db)
        weak_bull = bull & (tk < da) & (tk < db)
    return {
        "sig": sig_of(bull, bear),
        "exl": weak_bear,
        "exs": weak_bull,
        "state": np.sign(np.nan_to_num(tk - kj)),
    }


# ---------------- 20. Quantitative Trend (상태 신호) ----------------
def core_20(X):
    c = X["c"]
    phv = ffill_val(P.pivots(c, 46, 32, low=False))
    plv = ffill_val(P.pivots(c, 46, 32, low=True))
    with np.errstate(invalid="ignore"):
        return {"sig": sig_of(c > phv, c < plv, short_wins=False)}


# ---------------- 21. ATR_RSI v2 (상태 신호) ----------------
def core_21(X):
    h, lo, c = X["h"], X["l"], X["c"]
    a = P.atr(h, lo, c, 26)
    am = P.sma(a, 45)
    r = P.rsi(c, 15)
    with np.errstate(invalid="ignore"):
        nm = am / c * 100
        ok = (nm >= 0.3) & (nm <= 0.7) & (a > am)
        lc, sc = ok & (r > 60), ok & (r < 40)
    return {"sig": sig_of(lc, sc, short_wins=False)}


def native_21(b):
    h, lo = b["h"], b["l"]

    def init(p, j, ep, dr):
        p["ext"] = h[j] if dr == 1 else lo[j]  # 포지션 없을 때 intra_trade_high := high (신호 봉)
        p["tps"] = [(1.0, ep * (1.03 if dr == 1 else 0.94))]
        p["sl"] = p["ext"] * (1 - dr * 0.015)

    def on_close(p, i):
        if p["d"] == 1:
            p["ext"] = max(p["ext"], h[i])
            p["sl"] = p["ext"] * (1 - 0.015)
        else:
            p["ext"] = min(p["ext"], lo[i])
            p["sl"] = p["ext"] * (1 + 0.015)
        return None

    return init, on_close


# ---------------- 22. Relative Vigor Index ----------------
def core_22(X):
    o, h, lo, c = X["o"], X["h"], X["l"], X["c"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rvi = rsum(swma(c - o), 4) / rsum(swma(h - lo), 4)
    sg = swma(rvi)
    a = P.atr(h, lo, c, 19)
    e1, e2 = P.ema(c, 95), P.ema(c, 200)
    with np.errstate(invalid="ignore"):
        up = (lo > e1) & (e1 > e2)
        dn = (h < e1) & (e1 < e2)
    lc, sc = P.cross_over(rvi, sg) & up, P.cross_under(rvi, sg) & dn
    return {
        "sig": sig_of(lc, sc, short_wins=False),
        "sl_l": lo - 2 * a,
        "tp_l": h + 1 * a,
        "sl_s": h + 2 * a,
        "tp_s": lo - 1 * a,
    }


def native_22(X, r, n):
    m = {k: state15(X, r[k], n) for k in ("sl_l", "tp_l", "sl_s", "tp_s")}
    hx, lx = state15(X, X["h"], n), state15(X, X["l"], n)
    is_close = to15(X, np.ones(len(X["c"])), n, bool)

    def init(p, j, ep, dr):
        p["s"] = float(m["sl_l"][j] if dr == 1 else m["sl_s"][j])
        p["t"] = float(m["tp_l"][j] if dr == 1 else m["tp_s"][j])

    def on_close(p, i):
        if not is_close[i] or i == p["j"]:
            return None
        if p["d"] == 1:
            if lx[i] < p["s"]:
                return "stop"
            if hx[i] > p["t"]:
                return "tp"
        else:
            if hx[i] > p["s"]:
                return "stop"
            if lx[i] < p["t"]:
                return "tp"
        return None

    return {"init": init, "on_close": on_close, "reverse": False}


# ---------------- 23. Saturday (1시간) ----------------
def core_23(X, b):
    t = X["t"]
    d = frame(b, 1440)
    # 1시간 봉 시작 시각 기준 전날(금요일) 일봉: 그 날 00:00 이전에 마감된 마지막 일봉
    pos = np.searchsorted(d["t"] + DAY, t - (t % DAY), side="right") - 1
    green = np.zeros(len(t), bool)
    ok = pos >= 0
    green[ok] = d["c"][pos[ok]] > d["o"][pos[ok]]
    sat = dow(t) == 7
    ex = dow(t) < 7
    return {"sig": sig_of(sat & green, sat & ~green), "exl": ex, "exs": ex}


def native_23(X, n):
    cx = state15(X, X["c"], n)
    is_close = to15(X, np.ones(len(X["c"])), n, bool)

    def on_close(p, i):
        if is_close[i] and p["d"] == 1 and cx[i] >= p["ep"] * 1.007:
            return "tp"
        if is_close[i] and p["d"] == -1 and cx[i] <= p["ep"] * 0.093:
            return "tp"
        return None

    return on_close


# ---------------- 조립 ----------------
HTF = {"01": 60, "03": 60, "05": 60, "06": 120, "07": 120, "09": 240, "11": 240, "14": 240,
       "15": 60, "19": 60, "22": 240}  # fmt: skip
ONE = {"02": 15, "08": 15, "12": 15, "16": 15, "17": 15, "18": 15, "20": 15, "21": 15,
       "10": 60, "23": 60}  # fmt: skip


def _cond15(X, r, key, n):
    return to15(X, r[key].astype(bool) if key in r else np.zeros(len(X["c"]), bool), n, bool)


def make(b: dict, sid: str, minutes: int, scale: str) -> dict:
    """한 세트. scale='H'(원래 틀) 또는 'm'(같은 길이를 15분봉에)."""
    n = len(b["c"])
    X = frame(b, minutes)
    if sid == "01":
        r = core_01(X)
    elif sid == "02":
        r = core_02(X)
    elif sid == "03":
        Xh = frame(b, 240 if scale == "H" else 60)
        eh = state15(Xh, P.ema(Xh["c"], 200), n)
        r = core_03(X, eh, n)
    elif sid == "05":
        r = core_05(X)
    elif sid == "06":
        r = core_06(X)
    elif sid == "07":
        r = core_07(X)
    elif sid == "08":
        r = core_08(X)
    elif sid == "09":
        r = core_09(X)
    elif sid == "10":
        r = core_10(X)
    elif sid == "11":
        r = core_11(X)
    elif sid == "12":
        r = core_12(X)
    elif sid == "14":
        r = core_14(X)
    elif sid == "15":
        r = core_15(X)
    elif sid == "16":
        r = core_16(X)
    elif sid == "17":
        r = core_17(b, n)
    elif sid == "18":
        r = core_18(X)
    elif sid == "19":
        r = core_19(X)
    elif sid == "20":
        r = core_20(X)
    elif sid == "21":
        r = core_21(X)
    elif sid == "22":
        r = core_22(X)
    elif sid == "23":
        r = core_23(X, b)
    else:
        raise KeyError(sid)
    if sid == "17":
        sig, exl, exs = r["sig"], r["exl"], r["exs"]
    else:
        sig = to15(X, r["sig"], n)
        exl, exs = _cond15(X, r, "exl", n), _cond15(X, r, "exs", n)
    nat: dict = {"reverse": True}
    if sid in ("01", "06", "11", "15", "17", "19", "10", "23"):
        nat.update(exl=exl, exs=exs)
    if sid == "02":
        nat = {"init": pct_bracket(0.04, 0.03), "reverse": False}
    elif sid == "03":
        nat = native_03(X, r, n)
    elif sid == "05":
        ps = level_next(X, r["psar"], n)
        nat = {"reverse": True, "dyn_sl_s": ps, "dyn_sl_l": np.full(n, np.nan)}
    elif sid == "06":
        ini, onc = native_close_pct(X, n, 0.02)
        nat.update(init=ini, on_close=onc)
    elif sid == "08":
        nat = {"init": pct_bracket(0.15, 0.02), "reverse": True}
    elif sid == "09":
        nat = {"init": pct_bracket(0.14, 0.08), "reverse": True}
    elif sid == "10":
        nat["reverse"] = False
    elif sid == "11":
        nat["init"] = pct_bracket(None, 0.10)
    elif sid == "12":
        nat = {"exl": exl, "exs": exs, "on_close": native_12(b, r), "reverse": False}
    elif sid == "14":
        nat = native_14(X, r, b, n)
    elif sid == "16":
        ini, onc = native_16(b, r)
        nat = {"exl": exl, "exs": exs, "init": ini, "on_close": onc, "reverse": False}
    elif sid == "20":
        nat = {"init": pct_bracket(0.02, 0.0175), "reverse": False}
    elif sid == "21":
        ini, onc = native_21(b)
        nat = {"init": ini, "on_close": onc, "reverse": False}
    elif sid == "22":
        nat = native_22(X, r, n)
    elif sid == "23":
        nat.update(on_close=native_23(X, n), reverse=False)
    out = {"sig": sig.astype(np.int8), "native": nat}
    if "state" in r:
        out["state"] = state15(X, np.asarray(r["state"], float), n)
    return out


def build(b: dict) -> dict:
    S: dict = {}
    for sid, mins in HTF.items():
        S[f"T{sid}H"] = make(b, sid, mins, "H")
        S[f"T{sid}m"] = make(b, sid, 15, "m")
    for sid, mins in ONE.items():
        S[f"T{sid}"] = make(b, sid, mins, "H")
    return S


def th1_dir(d: dict) -> np.ndarray:
    """TH1 일봉 방향: CCI20(hlc3) > 0 이고 RSI14 > 50 → 1, 둘 다 반대 → −1, 그 밖 0."""
    cc = P.cci((d["h"] + d["l"] + d["c"]) / 3, 20)
    rr = P.rsi(d["c"], 14)
    with np.errstate(invalid="ignore"):
        return np.where((cc > 0) & (rr > 50), 1, np.where((cc < 0) & (rr < 50), -1, 0))
