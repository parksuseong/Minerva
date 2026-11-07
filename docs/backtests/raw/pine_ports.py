"""트레이딩뷰 파인 스크립트 10개의 신호 논리를 파이썬으로 옮긴 것(원문은 저장소 밖).

문서: docs/backtests/2026-11-08-pine-10-hypotheses.md
- 모든 함수는 마감된 봉 t의 값만 쓴다. 신호 t는 다음 봉 시가 체결(7번만 역지정가 봉 안 체결).
- 지표는 파인 내장 함수 정의를 따른다: ema·rma는 첫 값을 단순평균으로 시작, stdev는 모표준편차,
  atr = rma(true range), rsi = rma 방식.
- 반환 신호 배열: +1 롱, −1 숏, 0 없음.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

DAY = 86_400_000
M15 = 900_000


# ---------------- 파인 내장 지표 ----------------
def sma(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    bad = np.isnan(x)
    cs = np.cumsum(np.where(bad, 0.0, x))
    cb = np.cumsum(bad)
    s = cs[n - 1 :] - np.r_[0.0, cs[:-n]]
    b = cb[n - 1 :] - np.r_[0, cb[:-n]]
    out[n - 1 :] = np.where(b == 0, s / n, np.nan)
    return out


def _rec(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    ok = ~np.isnan(x)
    if not ok.any():
        return out
    i0 = int(np.argmax(ok))
    if len(x) - i0 < n:
        return out
    seed = i0 + n - 1
    out[seed] = x[i0 : seed + 1].mean()
    prev = out[seed]
    xl = x.tolist()
    o = out.tolist()
    for i in range(seed + 1, len(x)):
        v = xl[i]
        if v != v:
            o[i] = prev
            continue
        prev = alpha * v + (1 - alpha) * prev
        o[i] = prev
    return np.array(o)


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return _rec(x, n, 2.0 / (n + 1))


def rma(x: np.ndarray, n: int) -> np.ndarray:
    return _rec(x, n, 1.0 / n)


def stdev(x: np.ndarray, n: int) -> np.ndarray:
    m = sma(x, n)
    m2 = sma(np.asarray(x, float) ** 2, n)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def highest(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[n - 1 :] = swv(np.asarray(x, float), n).max(1)
    return out


def lowest(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    out[n - 1 :] = swv(np.asarray(x, float), n).min(1)
    return out


def shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def tr(h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> np.ndarray:
    pc = shift(c)
    t = np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))
    t[0] = h[0] - lo[0]
    return t


def atr(h, lo, c, n: int = 14) -> np.ndarray:
    return rma(tr(h, lo, c), n)


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(c, prepend=np.nan)
    up = rma(np.where(np.isnan(d), np.nan, np.maximum(d, 0)), n)
    dn = rma(np.where(np.isnan(d), np.nan, np.maximum(-d, 0)), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dn == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / dn)))
    r[np.isnan(up) | np.isnan(dn)] = np.nan
    return r


def stoch(c, h, lo, n: int) -> np.ndarray:
    hh, ll = highest(h, n), lowest(lo, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hh - ll > 0, 100 * (c - ll) / (hh - ll), np.nan)


def cci(src: np.ndarray, n: int = 20) -> np.ndarray:
    m = sma(src, n)
    out = np.full(len(src), np.nan)
    w = swv(np.asarray(src, float), n)
    md = np.abs(w - w.mean(1, keepdims=True)).mean(1)
    out[n - 1 :] = md
    with np.errstate(divide="ignore", invalid="ignore"):
        return (src - m) / (0.015 * out)


def cross_over(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pa, pb = shift(a), shift(b)
    with np.errstate(invalid="ignore"):
        return (a > b) & (pa <= pb)


def cross_under(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pa, pb = shift(a), shift(b)
    with np.errstate(invalid="ignore"):
        return (a < b) & (pa >= pb)


def pivots(x: np.ndarray, left: int, right: int, low: bool) -> np.ndarray:
    """파인 ta.pivotlow/high: 봉 t에 x[t-right]가 피벗이면 그 값, 아니면 nan(양쪽 엄격 비교)."""
    n = len(x)
    out = np.full(n, np.nan)
    w = left + right + 1
    if n < w:
        return out
    win = swv(np.asarray(x, float), w)
    mid = win[:, left]
    others = np.delete(win, left, axis=1)
    with np.errstate(invalid="ignore"):
        ok = (mid < others.min(1)) if low else (mid > others.max(1))
    ok &= ~np.isnan(win).any(1)
    out[w - 1 :] = np.where(ok, mid, np.nan)
    return out


# ---------------- 일봉 ----------------
def daily(b: dict) -> dict:
    """15분봉 → UTC 일봉(다 끝난 날만). last15[d]: 그날 마지막 15분봉 인덱스."""
    day = b["t"] // DAY
    keys, idx = np.unique(day, return_index=True)
    ends = np.r_[idx[1:], len(day)]
    full = (ends - idx) == 96
    keys, idx, ends = keys[full], idx[full], ends[full]
    return {
        "day": keys,
        "o": b["o"][idx],
        "h": np.array([b["h"][a:z].max() for a, z in zip(idx, ends)]),
        "l": np.array([b["l"][a:z].min() for a, z in zip(idx, ends)]),
        "c": b["c"][ends - 1],
        "v": np.array([b["v"][a:z].sum() for a, z in zip(idx, ends)]),
        "last15": ends - 1,
        "first15": idx,
    }


def daily_to_15(d: dict, sig: np.ndarray, n15: int) -> np.ndarray:
    """일봉 신호를 그날 마지막 15분봉에 붙인다(다음 봉 = 다음 날 00:00 시가 체결)."""
    out = np.zeros(n15, np.int8)
    out[d["last15"]] = sig
    return out


def daily_state_15(d: dict, state: np.ndarray, n15: int) -> np.ndarray:
    """15분봉 i 마감 시점에 확정된 마지막 일봉의 값(그날 마지막 봉 마감부터 적용)."""
    out = np.full(n15, np.nan)
    pos = np.searchsorted(d["last15"], np.arange(n15), side="right") - 1
    ok = pos >= 0
    out[ok] = np.asarray(state, float)[pos[ok]]
    return out


def daily_level_next(d: dict, level: np.ndarray, n15: int) -> np.ndarray:
    """15분봉 i가 열릴 때 이미 확정된 마지막 일봉의 값(봉 안 손절 가격 등)."""
    out = np.full(n15, np.nan)
    pos = np.searchsorted(d["last15"], np.arange(n15), side="left") - 1
    ok = pos >= 0
    out[ok] = np.asarray(level, float)[pos[ok]]
    return out


# ---------------- 1. Volume Order Block Retest ----------------
def s1_ob_retest(b: dict) -> dict:
    o, h, lo, c, v = b["o"], b["h"], b["l"], b["c"], b["v"]
    a = atr(h, lo, c, 14)
    vs = sma(v, 20)
    body = np.abs(c - o)
    a1 = np.nan_to_num(shift(a))
    vs1 = np.nan_to_num(shift(vs))
    big_body = body >= a1 * 0.8
    big_vol = v >= vs1 * 1.0
    bull_imp = (c > o) & big_body & big_vol
    bear_imp = (c < o) & big_body & big_vol
    n = len(c)
    sig = np.zeros(n, np.int8)
    lvl = np.full(n, np.nan)  # 롱: 구역 하단, 숏: 구역 상단
    zones: list = []  # [top, bottom, born, bull, used]
    ol, hl, ll, cl = o.tolist(), h.tolist(), lo.tolist(), c.tolist()
    for t in range(n):
        ls = ss = False
        for i in range(len(zones) - 1, -1, -1):
            z = zones[i]
            dead = (z[3] and cl[t] < z[1]) or ((not z[3]) and cl[t] > z[0]) or (t - z[2] > 300)
            if dead:
                zones.pop(i)
                continue
            touched = hl[t] >= z[1] and ll[t] <= z[0]
            if touched and not z[4]:
                if z[3]:
                    if cl[t] > ol[t] and cl[t] > z[1] and not ls and not ss:
                        ls = True
                        lvl[t] = z[1]
                        z[4] = True
                else:
                    if cl[t] < ol[t] and cl[t] < z[0] and not ls and not ss:
                        ss = True
                        lvl[t] = z[0]
                        z[4] = True
        if ls:
            sig[t] = 1
        elif ss:
            sig[t] = -1
        if t > 21:
            for bull, imp in ((True, bull_imp[t]), (False, bear_imp[t])):
                if not imp:
                    continue
                top, bot = hl[t - 1], ll[t - 1]
                if any(z[3] == bull and top >= z[1] and bot <= z[0] for z in zones):
                    continue
                zones.append([top, bot, t, bull, False])
                if len(zones) > 15:
                    zones.pop(0)
    # 자체 청산: 손절 = 구역 ∓ 0.2×ATR, 목표 1.5R(신호 봉 종가 기준), 1R 본전, 60봉 시간 청산
    sl = np.where(sig == 1, lvl - a * 0.2, np.where(sig == -1, lvl + a * 0.2, np.nan))
    risk = np.where(sig == 1, c - sl, np.where(sig == -1, sl - c, np.nan))
    okr = risk > 0.1
    sig = np.where(okr, sig, 0).astype(np.int8)
    tp = np.where(sig == 1, c + 1.5 * risk, np.where(sig == -1, c - 1.5 * risk, np.nan))
    native = {
        "sl": sl,
        "tp": tp,
        "be_risk": risk,
        "time_bars": 60,
        "reverse": False,
    }
    return {"sig": sig, "native": native}


# ---------------- 2. Rv-Div 정규 다이버전스 ----------------
def s2_divergence(o, h, lo, c) -> np.ndarray:
    hl2 = (h + lo) / 2
    osc = sma(hl2, 5) - sma(hl2, 34)
    a = atr(h, lo, c, 14)
    plo = pivots(osc, 7, 2, low=True)
    phi = pivots(osc, 7, 2, low=False)
    n = len(c)
    sig = np.zeros(n, np.int8)
    oL: list = []
    pL: list = []
    bL: list = []
    oH: list = []
    pH: list = []
    bH: list = []
    last_c = last_v = None
    for t in range(n):
        bull = bear = False
        if not math.isnan(plo[t]):
            px, bb, ov = lo[t - 2], t - 2, plo[t]
            chosen = -1
            for i in range(len(oL)):
                sep = bb - bL[i]
                ok = 4 <= sep <= 90 and px < pL[i] and ov > oL[i]
                ok = ok and ov < 0 and oL[i] < 0
                ok = ok and abs(px - pL[i]) >= 0.8 * a[t]
                if ok and chosen == -1:
                    chosen = i
            if chosen >= 0 and (last_c is None or t - last_c >= 3):
                bull = True
                del oL[: chosen + 1], pL[: chosen + 1], bL[: chosen + 1]
            oL.append(ov)
            pL.append(px)
            bL.append(bb)
            while len(oL) > 12:
                oL.pop(0), pL.pop(0), bL.pop(0)
        if not math.isnan(phi[t]):
            px, bb, ov = h[t - 2], t - 2, phi[t]
            chosen = -1
            for i in range(len(oH)):
                sep = bb - bH[i]
                ok = 4 <= sep <= 90 and px > pH[i] and ov < oH[i]
                ok = ok and ov > 0 and oH[i] > 0
                ok = ok and abs(px - pH[i]) >= 0.8 * a[t]
                if ok and chosen == -1:
                    chosen = i
            if chosen >= 0 and (last_v is None or t - last_v >= 3):
                bear = True
                del oH[: chosen + 1], pH[: chosen + 1], bH[: chosen + 1]
            oH.append(ov)
            pH.append(px)
            bH.append(bb)
            while len(oH) > 12:
                oH.pop(0), pH.pop(0), bH.pop(0)
        if bull:
            last_c = t
        if bear:
            last_v = t
        if bull and not bear:
            sig[t] = 1
        elif bear and not bull:
            sig[t] = -1
    return sig


# ---------------- 3. Gaussian Channel + StochRSI ----------------
def gaussian(src: np.ndarray, per: int = 144, poles: int = 4) -> np.ndarray:
    beta = (1 - math.cos(4 * math.asin(1) / per)) / (math.pow(1.414, 2 / poles) - 1)
    alpha = -beta + math.sqrt(beta * beta + 2 * beta)
    x = 1 - alpha
    assert poles == 4
    a4 = alpha**4
    c1, c2, c3, c4 = 4 * x, 6 * x * x, 4 * x**3, x**4
    s = np.nan_to_num(src).tolist()
    f = [0.0] * len(s)
    f1 = f2 = f3 = f4 = 0.0
    for i, v in enumerate(s):
        y = a4 * v + c1 * f1 - c2 * f2 + c3 * f3 - c4 * f4
        f[i] = y
        f4, f3, f2, f1 = f3, f2, f1, y
    return np.array(f)


def s3_gauss_core(h, lo, c) -> dict:
    hlc3 = (h + lo + c) / 3
    filt = gaussian(hlc3)
    ftr = gaussian(tr(h, lo, c))
    hband = filt + ftr * 1.414
    lband = filt - ftr * 1.414
    r = rsi(c, 14)
    st = stoch(r, r, r, 14)
    k = sma(st, 3)
    pf = shift(filt)
    with np.errstate(invalid="ignore"):
        up = filt > pf
        dn = filt < pf
        long_c = up & (c > hband) & ((k > 80) | (k < 15))
        short_c = dn & (c < lband) & ((k < 20) | (k > 85))
        close_l = (c < hband) | dn
        close_s = (c > lband) | up
    sig = np.where(long_c, 1, np.where(short_c, -1, 0)).astype(np.int8)
    return {
        "sig": sig,
        "close_l": close_l,
        "close_s": close_s,
        "hband": hband,
        "lband": lband,
        "up": up,
        "dn": dn,
    }


def s3_native(core: dict) -> dict:
    """15분봉 원본 그대로일 때: 조건 청산 다음 봉 시가, 손절은 직전 봉 마감의 하단(숏은 상단)."""
    return {
        "exl": core["close_l"],
        "exs": core["close_s"],
        "dyn_sl_l": shift(core["lband"]),
        "dyn_sl_s": shift(core["hband"]),
        "reverse": False,
    }


# ---------------- 4. CryptoEdge Pro ----------------
def vwap_daily(b: dict) -> np.ndarray:
    hlc3 = (b["h"] + b["l"] + b["c"]) / 3
    day = b["t"] // DAY
    pv = hlc3 * b["v"]
    out = np.empty(len(day))
    starts = np.r_[0, np.flatnonzero(np.diff(day)) + 1, len(day)]
    for a, z in zip(starts[:-1], starts[1:]):
        cv = np.cumsum(b["v"][a:z])
        out[a:z] = np.cumsum(pv[a:z]) / np.where(cv > 0, cv, np.nan)
    return out


def s4_cryptoedge(b: dict) -> dict:
    c = b["c"]
    v1 = vwap_daily(b)
    m1, m2 = ema(c, 9), ema(c, 21)
    a = atr(b["h"], b["l"], c, 14)
    with np.errstate(invalid="ignore"):
        lo_ = cross_over(m1, v1) & (m1 > m2)
        sh_ = cross_under(m1, v1) & (m1 < m2)
    sig = np.where(lo_, 1, np.where(sh_, -1, 0)).astype(np.int8)
    sl = np.where(sig == 1, c - 1.5 * a, np.where(sig == -1, c + 1.5 * a, np.nan))
    tp = np.where(sig == 1, c + 2.5 * a, np.where(sig == -1, c - 2.5 * a, np.nan))
    return {"sig": sig, "native": {"sl": sl, "tp": tp, "reverse": False}}


# ---------------- 5. Quatro SMA ----------------
def s5_quatro(b: dict) -> dict:
    c, v = b["c"], b["v"]
    vol_ok = v > sma(v, 40) * 2.5
    s1, s2, s3, s200 = sma(c, 4), sma(c, 16), sma(c, 32), sma(c, 200)
    with np.errstate(invalid="ignore"):
        long_ma = (s1 > s2) & (s2 > s3)
        short_ma = (s1 < s2) & (s2 < s3)
        long_c = long_ma & (c > s200) & vol_ok
        short_c = short_ma & (c < s200) & vol_ok
        long_close = short_ma & (c > s200)
        short_close = long_ma & (c < s200)
    sig = np.where(long_c, 1, np.where(short_c, -1, 0)).astype(np.int8)
    native = {
        "exl": long_close,
        "exs": short_close,
        "ladder": [(0.25, 0.10), (0.50, 0.20), (0.25, 0.50)],
        "sl_pct": 0.10,
        "reverse": True,
    }
    state = np.where(long_ma, 1, np.where(short_ma, -1, 0))
    return {"sig": sig, "native": native, "state": state}


# ---------------- 6. Trend Deviation ----------------
def supertrend_dir(h, lo, c, factor: float, n: int) -> np.ndarray:
    a = atr(h, lo, c, n)
    src = (h + lo) / 2
    up_b = (src - factor * a).tolist()
    dn_b = (src + factor * a).tolist()
    cl = c.tolist()
    al = a.tolist()
    nn = len(cl)
    d = [1] * nn
    lower = upper = float("nan")
    prev_super = float("nan")
    for i in range(nn):
        lb, ub = up_b[i], dn_b[i]
        if i > 0 and lower == lower:
            lb = lb if (lb > lower or cl[i - 1] < lower) else lower
        if i > 0 and upper == upper:
            ub = ub if (ub < upper or cl[i - 1] > upper) else upper
        if i == 0 or al[i - 1] != al[i - 1]:
            di = 1
        elif prev_super == upper:
            di = -1 if cl[i] > ub else 1
        else:
            di = 1 if cl[i] < lb else -1
        lower, upper = lb, ub
        prev_super = lb if di == -1 else ub
        d[i] = di
    return np.array(d, float)


def stc(c: np.ndarray, n: int = 14, fast: int = 25, slow: int = 60) -> np.ndarray:
    m = (ema(c, fast) - ema(c, slow)).tolist()
    nn = len(m)
    f = 1.5
    out = [float("nan")] * nn
    c5 = 0.0
    ddd_hist: list = []
    d6 = 0.0
    e5 = None
    ddd = None
    win: list = []
    for i in range(nn):
        win.append(m[i])
        if len(win) > n:
            win.pop(0)
        w = [x for x in win if x == x]
        if len(win) == n and len(w) == n:
            lo_, hi_ = min(w), max(w)
            rng = hi_ - lo_
            c5 = (m[i] - lo_) / rng * 100 if rng > 0 else c5
        ddd = c5 if ddd is None else ddd + f * (c5 - ddd)
        ddd_hist.append(ddd)
        if len(ddd_hist) > n:
            ddd_hist.pop(0)
        if len(ddd_hist) == n:
            lo_, hi_ = min(ddd_hist), max(ddd_hist)
            rng = hi_ - lo_
            d6 = (ddd - lo_) / rng * 100 if rng > 0 else d6
        e5 = d6 if e5 is None else e5 + f * (d6 - e5)
        out[i] = e5
    return np.array(out)


def aroon_bars(x: np.ndarray, n: int, high: bool) -> np.ndarray:
    """ta.highestbars/lowestbars(x, n): 최고(최저) 봉까지의 음수 거리, 동률은 최근 봉."""
    w = swv(np.asarray(x, float), n)[:, ::-1]
    k = (w.argmax(1) if high else w.argmin(1)).astype(float)
    out = np.full(len(x), np.nan)
    out[n - 1 :] = -k
    return out


def s6_trend_dev(h, lo, c) -> np.ndarray:
    up2 = np.diff(h, prepend=np.nan)
    dn2 = -np.diff(lo, prepend=np.nan)
    with np.errstate(invalid="ignore"):
        pdm = np.where(np.isnan(up2), np.nan, np.where((up2 > dn2) & (up2 > 0), up2, 0.0))
        mdm = np.where(np.isnan(dn2), np.nan, np.where((dn2 > up2) & (dn2 > 0), dn2, 0.0))
    trur = rma(tr(h, lo, c), 15)
    with np.errstate(invalid="ignore", divide="ignore"):
        plus = 100 * rma(pdm, 15) / trur
        minus = 100 * rma(mdm, 15) / trur
    plus = _fixnan(plus)
    minus = _fixnan(minus)
    dmi_up = cross_over(plus, minus)
    dmi_dn = cross_over(minus, plus)
    src = (h + lo + c) / 3
    basis = sma(src, 13)
    dev = 2.3 * stdev(src, 13)
    with np.errstate(invalid="ignore"):
        bblong = src < basis + dev
        bbshort = src > basis - dev
    au = 100 * (aroon_bars(h, 6, True) + 5) / 5
    al = 100 * (aroon_bars(lo, 6, False) + 5) / 5
    aroon_bull = cross_over(au, al)
    aroon_bear = cross_under(au, al)
    with np.errstate(invalid="ignore"):
        aroon_long = al > au
        aroon_short = al < au
    s = stc(c)
    s1, s2, s3 = shift(s, 1), shift(s, 2), shift(s, 3)
    with np.errstate(invalid="ignore"):
        stc_short = (s3 <= s2) & (s2 >= s1) & (s > 40)
        stc_long = (s3 >= s2) & (s2 <= s1) & (s < 60)
        stc_low = s < 25
        stc_high = s > 75
    macd = ema(c, 15) - ema(c, 200)
    amacd = ema(macd, 25)
    with np.errstate(invalid="ignore"):
        macd_long = macd > amacd
        macd_short = macd < amacd
    mom0 = c - shift(c, 10)
    mom1 = mom0 - shift(mom0)
    with np.errstate(invalid="ignore"):
        mom_long = (mom0 > 0) & (mom1 > 0)
        mom_short = (mom0 < 0) & (mom1 < 0)
    d = supertrend_dir(h, lo, c, 2.0, 200)
    dd = np.diff(d, prepend=np.nan)
    with np.errstate(invalid="ignore"):
        trendlong = dd < 0
        trendshort = dd > 0
    anyl = mom_long | macd_long | trendlong | stc_long | stc_low | aroon_long | aroon_bull
    anys = mom_short | macd_short | trendshort | stc_short | stc_high | aroon_short | aroon_bear
    lc = dmi_up & bblong & anyl
    sc = dmi_dn & bbshort & anys
    # 같은 봉이면 파인에서 숏 진입이 나중에 실행되어 숏이 남는다
    return np.where(sc, -1, np.where(lc, 1, 0)).astype(np.int8)


def _fixnan(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    last = np.nan
    for i in range(len(out)):
        if np.isnan(out[i]) or np.isinf(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def last_dir_state(sig: np.ndarray) -> np.ndarray:
    out = np.zeros(len(sig))
    cur = 0
    for i, s in enumerate(sig.tolist()):
        if s:
            cur = s
        out[i] = cur
    return out


# ---------------- 7. Session Breakout Scalper ----------------
def andean_bull(o: np.ndarray, c: np.ndarray, n: int = 50) -> np.ndarray:
    al = 2 / (n + 1)
    ol, cl = o.tolist(), c.tolist()
    up1 = up2 = dn1 = dn2 = None
    out = np.zeros(len(cl), bool)
    for i in range(len(cl)):
        C, Op = cl[i], ol[i]
        if up1 is None:
            up1, up2, dn1, dn2 = C, C * C, C, C * C
        else:
            up1 = max(C, Op, up1 - (up1 - C) * al)
            up2 = max(C * C, Op * Op, up2 - (up2 - C * C) * al)
            dn1 = min(C, Op, dn1 + (C - dn1) * al)
            dn2 = min(C * C, Op * Op, dn2 + (C * C - dn2) * al)
        bull = math.sqrt(max(dn2 - dn1 * dn1, 0.0))
        bear = math.sqrt(max(up2 - up1 * up1, 0.0))
        out[i] = bull > bear
    return out


SESSIONS_UTC = (13, 17, 23)
TP7_USD = 600.0


def s7_session(b: dict) -> dict:
    """세션 상자 돌파 역지정가. 한 상자당 먼저 닿은 쪽 한 번만(포지션 없음 가정)."""
    t, o, h, lo, c = b["t"], b["o"], b["h"], b["l"], b["c"]
    n = len(t)
    mins = (t // 60_000) % 1440
    sess = (mins % 60 == 0) & np.isin(mins // 60, SESSIONS_UTC)
    a30 = atr(h, lo, c, 30)
    hh, ll = highest(h, 10), lowest(lo, 10)
    box = np.zeros(n, bool)
    box[5:] = sess[:-5]
    bull = andean_bull(o, c)
    fill_dir = np.zeros(n, np.int8)
    fill_px = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    tp = np.full(n, np.nan)
    risk = np.full(n, np.nan)
    boxes = np.flatnonzero(box)
    hl_, ll_, ol_ = h.tolist(), lo.tolist(), o.tolist()
    for j, bi in enumerate(boxes):
        end = boxes[j + 1] if j + 1 < len(boxes) else n - 1
        bs, ls = hh[bi], ll[bi]
        if np.isnan(bs) or np.isnan(a30[bi]):
            continue
        for k in range(bi + 1, end + 1):
            hit_b = hl_[k] >= bs
            hit_s = ll_[k] <= ls
            if hit_b and hit_s:
                break  # 같은 봉 양쪽: 순서를 몰라 건너뛴다
            if hit_b:
                fill_dir[k], fill_px[k] = 1, max(ol_[k], bs)
                sl[k], tp[k] = bs - 4 * a30[bi], bs + TP7_USD
                risk[k] = 4 * a30[bi]
                break
            if hit_s:
                fill_dir[k], fill_px[k] = -1, min(ol_[k], ls)
                sl[k], tp[k] = ls + 4 * a30[bi], ls - TP7_USD
                risk[k] = 4 * a30[bi]
                break
    prev_bull = shift(bull.astype(float)) == 1
    prev_bear = shift((~bull).astype(float)) == 1
    day_first = (t % DAY) == 0
    native = {
        "sl": sl,
        "tp": tp,
        "exl": (~prev_bull) | sess,
        "exs": (~prev_bear) | sess,
        "daygain_risk": risk,
        "day_first": day_first,
        "reverse": False,
    }
    # 청산 묶음용 신호: 직전 봉 국면이 방향과 같은 돌파만
    gated = np.where((fill_dir == 1) & prev_bull, 1, np.where((fill_dir == -1) & prev_bear, -1, 0))
    return {
        "sig_all": fill_dir,
        "sig": gated.astype(np.int8),
        "epx": fill_px,
        "native": native,
    }


# ---------------- 8. Extreme Reversal Signal ----------------
def s8_extreme(c: np.ndarray) -> np.ndarray:
    basis = sma(c, 20)
    dev = stdev(c, 20)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = (c - basis) / dev
    p = shift(sd)
    with np.errstate(invalid="ignore"):
        bull = (p < -2.0) & (sd > -2.0)
        bear = (p > 2.0) & (sd < 2.0)
    return np.where(bull, 1, np.where(bear, -1, 0)).astype(np.int8)


# ---------------- 9. Hashrate Ribbons ----------------
def s9_hash(ht: np.ndarray, hr: np.ndarray, t15: np.ndarray, lag_days: int = 2) -> dict:
    """D일 값은 D+lag일 00:00 UTC부터 안다. 교차는 그 시각 직전 15분봉 마감 신호로."""
    s30, s60 = sma(hr, 30), sma(hr, 60)
    up = cross_over(s30, s60)
    dn = cross_under(s30, s60)
    known = ht + lag_days * DAY  # 이 시각에 마감되는 15분봉에 신호
    sig = np.zeros(len(t15), np.int8)
    idx = np.searchsorted(t15, known - M15)
    for k in range(len(ht)):
        i = idx[k]
        if 0 <= i < len(t15) and t15[i] == known[k] - M15:
            if up[k]:
                sig[i] = 1
            elif dn[k]:
                sig[i] = -1
    with np.errstate(invalid="ignore"):
        st = np.where(s30 > s60, 1.0, np.where(s30 < s60, -1.0, np.nan))
    state = np.full(len(t15), np.nan)
    pos = np.searchsorted(known, t15 + M15, side="right") - 1
    ok = pos >= 0
    state[ok] = st[pos[ok]]
    return {"sig": sig, "state": state}


# ---------------- 10. Fast v Slow MA ----------------
def s10_ma(c: np.ndarray) -> dict:
    f, s = ema(c, 24), ema(c, 65)
    with np.errstate(invalid="ignore"):
        mom = np.where(np.isnan(f) | np.isnan(s), np.nan, np.where(f >= s, 1.0, -1.0))
    pm = shift(mom)
    with np.errstate(invalid="ignore"):
        up = (mom > 0) & (pm < 0)
        dn = (mom < 0) & (pm > 0)
    sig = np.where(up, 1, np.where(dn, -1, 0)).astype(np.int8)
    return {"sig": sig, "state": mom}
