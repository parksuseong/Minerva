"""트레이딩뷰 공개 지표 26개에서 뽑은 구성요소(진입 촉발·필터·청산)를 파이썬으로 옮긴 것.

문서: docs/backtests/2026-11-08-btc-tv-components-hypotheses.md (사전 등록 그대로)
원문은 저장소 밖(~/.minerva/research/tv-components-2026-11-08/, index.json)에 두고, 여기에는 논리만 둔다.
규칙: 모든 값은 마감된 봉만 쓴다. 피벗은 오른쪽 봉 수만큼 늦게 확정된다. 상위 시간 틀(1h·4h·일·주)은
그 봉이 마감된 15분봉부터 쓴다. 신호는 조건이 새로 참이 된 봉에서만(+1 롱, −1 숏), 다음 15분봉 시가 체결.
"""

from __future__ import annotations

import math

import numpy as np
import pine_ports as P
import pine_tv_ports as TP

M15 = 900_000
H1 = 3_600_000
DAY = 86_400_000

SINGLE = ("C1", "C1L", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10")
COMBO = ("K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10")
NONPRICE = {"C1", "C1L", "C7", "C8", "K1", "K2", "K3", "K4", "K5", "K6", "K7"}


def price_only(name: str) -> bool:
    return name in SINGLE + COMBO and name not in NONPRICE


def edge(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, bool)
    return x & ~np.r_[False, x[:-1]]


def sig_from(lc: np.ndarray, sc: np.ndarray) -> np.ndarray:
    le, se = edge(lc), edge(sc)
    return np.where(le & ~se, 1, np.where(se & ~le, -1, 0)).astype(np.int8)


def sig_events(lv: np.ndarray, sv: np.ndarray) -> np.ndarray:
    """이미 사건(한 봉)인 배열을 신호로. 같은 봉 양방향은 버린다."""
    lv, sv = np.asarray(lv, bool), np.asarray(sv, bool)
    return np.where(lv & ~sv, 1, np.where(sv & ~lv, -1, 0)).astype(np.int8)


def nz(x: np.ndarray, v: float = 0.0) -> np.ndarray:
    return np.where(np.isnan(x), v, x)


def recent(x: np.ndarray, k: int) -> np.ndarray:
    """최근 k봉(현재 포함) 안에 x가 참이었던 적이 있나."""
    x = np.asarray(x, bool).astype(np.int64)
    cs = np.r_[0, np.cumsum(x)]
    i = np.arange(len(x))
    return (cs[i + 1] - cs[np.maximum(i + 1 - k, 0)]) > 0


# ---------------- 피벗(확정 뒤에만) ----------------
def pivot_conf(x: np.ndarray, left: int, right: int, high: bool) -> np.ndarray:
    """확정 봉 t에 피벗 값(t−right 봉), 아니면 nan. 이웃보다 엄격히 높/낮아야 한다."""
    n = len(x)
    out = np.full(n, np.nan)
    xl = np.asarray(x, float)
    for t in range(left + right, n):
        p = t - right
        v = xl[p]
        win = np.r_[xl[p - left : p], xl[p + 1 : t + 1]]
        if (high and v > win.max()) or (not high and v < win.min()):
            out[t] = v
    return out


# ---------------- 단일 구성요소 ----------------
def c2_sweep(h, lo, c, L: int = 8, R: int = 8, keep: int = 4) -> tuple:
    """01 Sweep Odds: 확정된 스윙 고점·저점(8/8, 최근 4개)을 꼬리로 넘고 종가가 되돌아오면 쓸기.
    꼬리로 넘은 수준은 쓸기든 아니든 지운다. 저점 쓸기 = 롱, 고점 쓸기 = 숏."""
    n = len(c)
    ph = pivot_conf(h, L, R, True)
    pl = pivot_conf(lo, L, R, False)
    hs: list = []
    ls: list = []
    lv = np.zeros(n, bool)
    sv = np.zeros(n, bool)
    lvl_l = np.full(n, np.nan)
    lvl_s = np.full(n, np.nan)
    hl, ll_, cl = h.tolist(), lo.tolist(), c.tolist()
    for t in range(n):
        if hs:
            hit = [x for x in hs if hl[t] > x]
            if hit:
                if cl[t] < max(hit):
                    sv[t] = True
                    lvl_s[t] = max(hit)
                hs = [x for x in hs if x not in hit]
        if ls:
            hit = [x for x in ls if ll_[t] < x]
            if hit:
                if cl[t] > min(hit):
                    lv[t] = True
                    lvl_l[t] = min(hit)
                ls = [x for x in ls if x not in hit]
        # 이 봉에서 확정된 피벗은 다음 봉부터 쓴다
        if not math.isnan(ph[t]):
            hs = (hs + [float(ph[t])])[-keep:]
        if not math.isnan(pl[t]):
            ls = (ls + [float(pl[t])])[-keep:]
    return lv, sv


def c3_session(t, h, lo, c) -> tuple:
    """07 Sessions & Sweeps: 아시아(00~08 UTC) 범위를 런던(08~16 UTC)에 처음 넘는 봉(하루 한쪽 한 번).
    원문은 꼬리만 봐서 방향이 없어 01의 쓸기 정의(종가가 범위 안으로 되돌아옴)를 더해 되돌림 진입으로 정한다."""
    n = len(c)
    day = t // DAY
    hr = (t % DAY) // H1
    lv = np.zeros(n, bool)
    sv = np.zeros(n, bool)
    cur, ahi, alo, dh, dl = -1, -np.inf, np.inf, False, False
    for i in range(n):
        if day[i] != cur:
            cur, ahi, alo, dh, dl = day[i], -np.inf, np.inf, False, False
        if hr[i] < 8:
            ahi = max(ahi, h[i])
            alo = min(alo, lo[i])
        elif hr[i] < 16 and np.isfinite(ahi):
            if not dh and h[i] > ahi:
                dh = True
                sv[i] = c[i] < ahi
            if not dl and lo[i] < alo:
                dl = True
                lv[i] = c[i] > alo
    return lv, sv


def choch(h, lo, c, L: int = 5, R: int = 5) -> tuple:
    """03 Structure Radar: 피벗 5/5, 종가 돌파. 추세 −1에서 위로 돌파 = 상승 CHoCH(롱). 반환: (롱, 숏, 추세)."""
    n = len(c)
    ph = pivot_conf(h, L, R, True)
    pl = pivot_conf(lo, L, R, False)
    swh = swl = math.nan
    ah = al = False
    tr = 0
    lv = np.zeros(n, bool)
    sv = np.zeros(n, bool)
    trend = np.zeros(n)
    for i in range(n):
        if ah and c[i] > swh:
            if tr == -1:
                lv[i] = True
            tr = 1
            ah = False
        if al and c[i] < swl:
            if tr == 1:
                sv[i] = True
            tr = -1
            al = False
        if not math.isnan(ph[i]):
            swh, ah = float(ph[i]), True
        if not math.isnan(pl[i]):
            swl, al = float(pl[i]), True
        trend[i] = tr
    return lv, sv, trend


def squeeze_breakout(o, h, lo, c, v, kc: float = 1.5, min_sq: int = 8, buf: float = 0.15,
                     arm: int = 20) -> tuple:  # fmt: skip
    """26 Squeeze Breakout(공격형, Intraday 묶음): BB(20,2) ⊂ KC(EMA20 ± 1.5·ATR20)이 min_sq봉 이상 이어진
    뒤 풀리면 arm봉 동안 상자 돌파를 기다린다. 롱: 종가 > 상자 고점 + 0.15·ATR14, 양봉, 몸통 ≥ 범위 50%,
    거래량 ≥ 1.2×20평균, 종가 > EMA200. 상자당 한 번. 반환 (롱, 숏, 스퀴즈 켜짐)."""
    n = len(c)
    basis = P.sma(c, 20)
    sd = P.stdev(c, 20)
    e20 = P.ema(c, 20)
    a20 = P.atr(h, lo, c, 20)
    a14 = P.atr(h, lo, c, 14)
    e200 = P.ema(c, 200)
    vs = P.sma(v, 20)
    with np.errstate(invalid="ignore"):
        sq = ((basis + 2 * sd) < (e20 + kc * a20)) & ((basis - 2 * sd) > (e20 - kc * a20))
    lv = np.zeros(n, bool)
    sv = np.zeros(n, bool)
    cnt, bhi, blo = 0, -np.inf, np.inf
    armed, left, ahi, alo = False, 0, 0.0, 0.0
    for i in range(n):
        if sq[i]:
            if cnt == 0:
                bhi, blo = h[i], lo[i]
            cnt += 1
            bhi, blo = max(bhi, h[i]), min(blo, lo[i])
            continue
        if cnt >= min_sq:
            armed, left, ahi, alo = True, arm, bhi, blo
        cnt = 0
        if armed:
            rng = h[i] - lo[i]
            body = abs(c[i] - o[i])
            ok = rng > 0 and body >= 0.5 * rng and vs[i] > 0 and v[i] >= 1.2 * vs[i]
            if ok and c[i] > ahi + buf * a14[i] and c[i] > o[i] and c[i] > e200[i]:
                lv[i] = True
                armed = False
            elif ok and c[i] < alo - buf * a14[i] and c[i] < o[i] and c[i] < e200[i]:
                sv[i] = True
                armed = False
            left -= 1
            if left <= 0:
                armed = False
    return lv, sv, sq


def vwz(h, lo, c, v, n: int = 20) -> np.ndarray:
    """21 MTF Z-Score: 직전 n봉(현재 제외) 거래량 가중 평균·분산(봉 안 분산 (H−L)²/12 포함)으로 종가 z."""
    hlc3 = (h + lo + c) / 3
    w = np.where(v > 0, v, 1.0)
    a = np.r_[0.0, np.cumsum(w)]
    b = np.r_[0.0, np.cumsum(w * hlc3)]
    q = np.r_[0.0, np.cumsum(w * (hlc3**2 + (h - lo) ** 2 / 12))]
    i = np.arange(len(c))
    lo_i = np.maximum(i - n, 0)
    sw = a[i] - a[lo_i]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (b[i] - b[lo_i]) / sw
        var = (q[i] - q[lo_i]) / sw - m * m
        z = (c - m) / np.sqrt(np.maximum(var, 1e-12))
    z[i < n] = np.nan
    return z


def cvd_z(o, c, v, tbv=None) -> np.ndarray:
    """19 CVD Z-Score: 원문은 봉 색으로 거래량 부호를 정하는 대용. 선물에는 실제 테이커 매수량이 있어
    델타 = 테이커 매수 − 테이커 매도를 쓴다(없으면 원문 대용). Z = EMA5((CVD − SMA20)/stdev20)."""
    if tbv is not None:
        d = 2 * tbv - v
    else:
        d = np.where(c > o, v, np.where(c < o, -v, 0.0))
    cvd = np.cumsum(d)
    with np.errstate(invalid="ignore", divide="ignore"):
        rz = (cvd - P.sma(cvd, 20)) / P.stdev(cvd, 20)
    rz[~np.isfinite(rz)] = np.nan
    return P.ema(rz, 5)


def c8_flush(h, lo, c, o, v, oi, atr14) -> tuple:
    """18 OI Pressure의 강제청산 대용: OI 15분 변화(%) ≤ −1.8×20봉 평균 |변화|, 거래량 ≥ 1.8×평균,
    범위 ≥ 1.3×ATR. 음봉 = 롱 청산(FLUSH) → 되돌림 롱, 양봉 = 숏 청산(SQUEEZE) → 되돌림 숏."""
    with np.errstate(invalid="ignore", divide="ignore"):
        oip = 100 * (oi / P.shift(oi) - 1)
    amp = P.sma(np.abs(oip), 20)
    vr = v / P.sma(v, 20)
    with np.errstate(invalid="ignore"):
        liq = (oip <= -1.8 * amp) & (vr >= 1.8) & ((h - lo) >= 1.3 * atr14)
    liq = np.where(np.isnan(oip) | np.isnan(amp), False, liq)
    return liq & (c < o), liq & (c > o)


def mfi(h, lo, c, v, n: int = 14) -> np.ndarray:
    tp = (h + lo + c) / 3
    mf = tp * v
    d = np.diff(tp, prepend=np.nan)
    pos = P.sma(np.where(d > 0, mf, 0.0), n) * n
    neg = P.sma(np.where(d < 0, mf, 0.0), n) * n
    with np.errstate(invalid="ignore", divide="ignore"):
        r = 100 - 100 / (1 + pos / neg)
    return np.where(neg == 0, 100.0, r)


def c9_divergence(o, h, lo, c, v, atr14) -> tuple:
    """08 Eye Divergence Engine(정규 다이버전스만): 피벗 5/3 확정 때 직전 같은 종류 피벗(5~80봉 전)과 비교.
    가격 신저가인데 오실레이터 7개(RSI, MACD 히스토그램, Stoch, CCI, MFI, OBV, 모멘텀) 중 4개 이상이
    더 높으면, 가격 차 ≥ 0.1·ATR, 피벗 봉 합성값 ≤ −25. 확정 뒤 6봉 안에 종가가 (피벗~확정 봉 최고가)를
    넘으면 롱, 그 전에 저가가 피벗 저가 아래면 취소. 숏은 거울. (평균 격차 ≥ 8 조건과 숨은 다이버전스는 뺐다)"""
    n = len(c)
    rsi = P.rsi(c, 14)
    hist = P.ema(c, 12) - P.ema(c, 26)
    hist = hist - P.ema(hist, 9)
    k = P.sma(P.stoch(c, h, lo, 14), 3)
    cci = P.cci((h + lo + c) / 3, 20)
    mf = mfi(h, lo, c, v, 14)
    obv = np.cumsum(np.sign(np.diff(c, prepend=c[0])) * v)
    od = obv - P.ema(obv, 20)
    mom = c - P.shift(c, 10)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = [
            (rsi - 50) * 2, hist / P.stdev(hist, 100) * 40, (k - 50) * 2, cci / 2,
            (mf - 50) * 2, od / P.stdev(od, 100) * 40, mom / (atr14 * math.sqrt(10)) * 50,
        ]  # fmt: skip
    raw = [rsi, hist, k, cci, mf, obv, mom]
    norm = [np.clip(x, -100, 100) for x in norm]
    comp = P.ema(np.nanmean(np.vstack(norm), 0), 3)
    pl = pivot_conf(lo, 5, 3, False)
    ph = pivot_conf(h, 5, 3, True)
    lv = np.zeros(n, bool)
    sv = np.zeros(n, bool)
    hl, ll_, cl = h.tolist(), lo.tolist(), c.tolist()

    def check(piv, low: bool, out):
        prev = None
        pend = None  # (만료 봉, 촉발 가격, 무효 가격)
        for t in range(n):
            if pend is not None:
                exp, trig, inval = pend
                if (ll_[t] < inval) if low else (hl[t] > inval):
                    pend = None
                elif (cl[t] > trig) if low else (cl[t] < trig):
                    out[t] = True
                    pend = None
                elif t >= exp:
                    pend = None
            if math.isnan(piv[t]):
                continue
            p = t - 3
            if prev is not None and 5 <= p - prev <= 80:
                pp, px, qx = prev, piv[t], (lo[prev] if low else h[prev])
                lower = px < qx if low else px > qx
                if lower and abs(px - qx) >= 0.1 * atr14[p]:
                    votes = 0
                    for r in raw:
                        a_, b_ = r[p], r[pp]
                        if not (math.isnan(a_) or math.isnan(b_)):
                            votes += (a_ > b_) if low else (a_ < b_)
                    zone = comp[p] <= -25 if low else comp[p] >= 25
                    if votes >= 4 and zone:
                        seg = h[p : t + 1] if low else lo[p : t + 1]
                        trig = seg.max() if low else seg.min()
                        pend = (t + 6, float(trig), float(px))
            prev = p

    check(pl, True, lv)
    check(ph, False, sv)
    return lv, sv


def darvas(h, lo, c, N: int = 3, look: int = 100) -> tuple:
    """24 Darvas Box Ladder: 100봉 신고가에서 천장, N봉 확인 뒤 바닥, 다시 N봉 확인, 높이 3~25%면 상자.
    롱: 종가 > 천장×1.001. 숏은 거울(100봉 신저가에서 바닥 상자, 종가 < 바닥×0.999)."""

    def run(hh, ll, cc, up: bool) -> np.ndarray:
        n = len(cc)
        hi_n = P.highest(hh, look) if up else -P.lowest(ll, look)
        H = hh if up else -ll
        L = ll if up else -hh
        C = cc if up else -cc
        out = np.zeros(n, bool)
        st, ceil, flo, low_s, cd = 0, 0.0, 0.0, 0.0, N
        for i in range(n):
            if math.isnan(hi_n[i]):
                continue
            if st == 0:
                if H[i] >= hi_n[i]:
                    ceil, low_s, cd, st = H[i], L[i], N, 1
            elif st == 1:
                low_s = min(low_s, L[i])
                if H[i] > ceil:
                    ceil, cd = H[i], N
                else:
                    cd -= 1
                    if cd <= 0:
                        flo, cd, st = low_s, N, 2
            elif st == 2:
                if H[i] > ceil:
                    ceil, low_s, cd, st = H[i], L[i], N, 1
                elif L[i] < flo:
                    flo, cd = L[i], N
                else:
                    cd -= 1
                    if cd <= 0:
                        hgt = (ceil - flo) / abs(flo) * 100
                        st = 3 if 3 <= hgt <= 25 else 0
            elif st == 3:
                if C[i] > ceil + 0.001 * abs(ceil):
                    out[i] = True
                    ceil, low_s, cd, st = H[i], L[i], N, 1
                elif C[i] < flo:
                    st = 0
        return out

    return run(h, lo, c, True), run(h, lo, c, False)


def chandelier_dir(h, lo, c, n: int = 22, k: float = 3.0) -> np.ndarray:
    """15 Chandelier Exit(종가 극값): 롱선 = 최고 종가22 − 3·ATR22, 숏선 = 최저 종가22 + 3·ATR22."""
    a = P.atr(h, lo, c, n)
    cl_ = P.highest(c, n) - k * a
    cs_ = P.lowest(c, n) + k * a
    d = np.zeros(len(c))
    cur = 0
    for i in range(1, len(c)):
        if not math.isnan(cs_[i - 1]) and c[i] > cs_[i - 1]:
            cur = 1
        elif not math.isnan(cl_[i - 1]) and c[i] < cl_[i - 1]:
            cur = -1
        d[i] = cur
    return d


def leg_maturity(h, lo, c, atr14, k: float = 2.5) -> tuple:
    """25 Trend Leg Maturity(ATR 지그재그 2.5): 현재 다리 방향과 성숙도(나이·폭 백분위 평균, 같은 방향 최근
    100개 다리 대비, 10개 이상일 때). 반전은 종가가 극값에서 2.5·ATR 되돌린 봉에 확정."""
    n = len(c)
    dirn = np.zeros(n)
    score = np.full(n, np.nan)
    d, ext, ext_i, st, st_i = 0, c[0], 0, c[0], 0
    legs = {1: [], -1: []}
    for i in range(n):
        a = atr14[i]
        if math.isnan(a):
            continue
        if d == 0:
            d, ext, ext_i, st, st_i = 1, h[i], i, lo[i], i
        if d == 1:
            if h[i] > ext:
                ext, ext_i = h[i], i
            if c[i] <= ext - k * a:
                legs[1].append((ext_i - st_i, (ext - st) / a))
                legs[1] = legs[1][-100:]
                st, st_i = ext, ext_i
                seg = lo[ext_i : i + 1]
                d, ext, ext_i = -1, seg.min(), ext_i + int(seg.argmin())
        else:
            if lo[i] < ext:
                ext, ext_i = lo[i], i
            if c[i] >= ext + k * a:
                legs[-1].append((ext_i - st_i, (st - ext) / a))
                legs[-1] = legs[-1][-100:]
                st, st_i = ext, ext_i
                seg = h[ext_i : i + 1]
                d, ext, ext_i = 1, seg.max(), ext_i + int(seg.argmax())
        dirn[i] = d
        hist = legs[d]
        if len(hist) >= 10:
            age = i - st_i
            stretch = abs(ext - st) / a
            ag = np.array([x[0] for x in hist])
            sg = np.array([x[1] for x in hist])
            pa = ((ag < age).sum() + 0.5 * (ag == age).sum()) / len(ag) * 100
            pe = ((sg < stretch).sum() + 0.5 * (sg == stretch).sum()) / len(sg) * 100
            score[i] = (pa + pe) / 2
    return dirn, score


# ---------------- 조립 ----------------
def _htf(b: dict, minutes: int) -> dict:
    return TP.frame(b, minutes)


def build(b: dict, design: tuple | None, thr: dict | None = None, nonprice: bool = True) -> tuple:
    t, o, h, lo, c, v = (b[k] for k in ("t", "o", "h", "l", "c", "v"))
    n = len(c)
    atr = P.atr(h, lo, c, 14)
    st_l = P.lowest(lo, 8) - 0.25 * atr
    st_s = P.highest(h, 8) + 0.25 * atr
    H = _htf(b, 60)
    H4 = _htf(b, 240)
    D = _htf(b, 1440)
    Wk = _htf(b, 10080)
    # 1h·4h 구성요소
    c4l, c4s, tr1h = choch(H["h"], H["l"], H["c"])
    c5l, c5s, sq1h = squeeze_breakout(H["o"], H["h"], H["l"], H["c"], H["v"])
    c10l, c10s = darvas(H["h"], H["l"], H["c"])
    ce4 = TP.state15(H4, chandelier_dir(H4["h"], H4["l"], H4["c"]), n)
    rk4 = TP.state15(H4, P.supertrend_dir(H4["h"], H4["l"], H4["c"], 3.0, 10), n)
    to = lambda X, x: TP.to15(X, x.astype(np.int8), n).astype(bool)  # noqa: E731
    # 21 MTF z: 15m(현재 봉 마감), 1h(75분 대신), 일·주 편향
    z15 = vwz(h, lo, c, v)
    z1h = TP.state15(H, vwz(H["h"], H["l"], H["c"], H["v"]), n)
    zD = TP.state15(D, vwz(D["h"], D["l"], D["c"], D["v"]), n)
    zW = TP.state15(Wk, vwz(Wk["h"], Wk["l"], Wk["c"], Wk["v"]), n)
    zD_ = np.where(np.isnan(zD), zW, zD)
    zW_ = np.where(np.isnan(zW), zD, zW)
    bias = np.where((zD_ > 0) & (zW_ > 0), 1, np.where((zD_ < 0) & (zW_ < 0), -1, 0))
    lomin = np.minimum(nz(z15), nz(z1h))
    lomax = np.maximum(nz(z15), nz(z1h))
    c2l, c2s = c2_sweep(h, lo, c)
    c3l, c3s = c3_session(t, h, lo, c)
    c9l, c9s = c9_divergence(o, h, lo, c, v, atr)
    legd, legs = leg_maturity(h, lo, c, atr)
    sets: dict = {}
    sets["C2"] = {"sig": sig_events(c2l, c2s)}
    sets["C3"] = {"sig": sig_events(c3l, c3s)}
    sets["C4"] = {"sig": sig_events(to(H, c4l), to(H, c4s))}
    sets["C5"] = {"sig": sig_events(to(H, c5l), to(H, c5s))}
    own6 = (nz(z15) >= 0, nz(z15) <= 0)
    sets["C6"] = {"sig": sig_from((bias == 1) & (lomin <= -2), (bias == -1) & (lomax >= 2)),
                  "own_l": own6[0], "own_s": own6[1]}  # fmt: skip
    sets["C9"] = {"sig": sig_events(c9l, c9s)}
    sets["C10"] = {"sig": sig_events(to(H, c10l), to(H, c10s))}
    with np.errstate(invalid="ignore"):
        sets["K8"] = {"sig": sig_events(c2l & (nz(z15) <= -1.5), c2s & (nz(z15) >= 1.5)),
                      "own_l": own6[0], "own_s": own6[1]}  # fmt: skip
        sets["K9"] = {"sig": sig_events(c2l & (nz(ce4) == 1), c2s & (nz(ce4) == -1))}
        sets["K10"] = {"sig": sig_events(to(H, c5l) & (nz(rk4) == 1), to(H, c5s) & (nz(rk4) == -1))}
    feat = {
        "rsi": P.rsi(c, 14) - 50, "stoch": P.stoch(c, h, lo, 14) - 50,
        "cci": P.cci((h + lo + c) / 3, 20), "atrp": atr / c, "z15": z15, "z1h": z1h,
        "bias": bias.astype(float), "ce4": ce4, "rk4": rk4,
        "tr1h": TP.state15(H, tr1h, n), "sq1h": TP.state15(H, sq1h.astype(float), n),
        "hour": ((t % DAY) // H1).astype(float),
        "dhi": (P.highest(h, 96) - c) / atr, "dlo": (c - P.lowest(lo, 96)) / atr,
        "legd": legd, "legs": legs, "r1": c / P.shift(c, 4) - 1, "r4": c / P.shift(c, 16) - 1,
    }  # fmt: skip
    signed = {"rsi", "stoch", "cci", "z15", "z1h", "bias", "ce4", "rk4", "tr1h", "legd", "r1", "r4"}
    if nonprice:
        zP = b["f_zP"]
        if thr is None:
            a, z = design
            x = zP[a:z]
            x = x[~np.isnan(x)]
            thr = {
                "zP": {q: float(np.quantile(x, float(q) / 100)) for q in ("0.5", "2", "98", "99.5")}
            }
        q = thr["zP"]
        with np.errstate(invalid="ignore"):
            pl_ = nz(zP, 0) <= q["0.5"]
            ps_ = nz(zP, 0) >= q["99.5"]
            hl_ = nz(zP, 0) <= q["2"]
            hs_ = nz(zP, 0) >= q["98"]
        own1 = (nz(zP, 0) >= 0, nz(zP, 0) <= 0)
        cz = cvd_z(o, c, v, b["tbv"])
        c8l, c8s = c8_flush(h, lo, c, o, v, b["f_oi"], atr)
        sets["C1"] = {"sig": sig_from(pl_, ps_), "own_l": own1[0], "own_s": own1[1]}
        sets["C1L"] = {"sig": sig_from(hl_, hs_), "own_l": own1[0], "own_s": own1[1]}
        sets["C7"] = {"sig": sig_from(nz(cz) <= -2, nz(cz) >= 2), "own_l": nz(cz) >= 0,
                      "own_s": nz(cz) <= 0}  # fmt: skip
        sets["C8"] = {"sig": sig_events(c8l, c8s)}
        K = {}
        K["K1"] = (hl_ & recent(c2l, 8), hs_ & recent(c2s, 8))
        K["K2"] = (hl_ & (nz(z15) <= -1.5), hs_ & (nz(z15) >= 1.5))
        K["K3"] = (hl_ & (nz(ce4) == 1), hs_ & (nz(ce4) == -1))
        K["K4"] = (hl_ & (nz(ce4) == -1), hs_ & (nz(ce4) == 1))
        K["K6"] = (hl_ & (nz(cz) <= -1.5), hs_ & (nz(cz) >= 1.5))
        K["K7"] = (hl_ & (legd == -1) & (nz(legs) > 66.7), hs_ & (legd == 1) & (nz(legs) > 66.7))
        for k_, (lc, sc) in K.items():
            sets[k_] = {"sig": sig_from(lc, sc), "own_l": own1[0], "own_s": own1[1]}
        # K5: C1L 켜진 뒤 16봉 안에서 처음 종가 > 직전 봉 고가(롱) / < 직전 봉 저가(숏), 무장당 한 번
        s1l = sets["C1L"]["sig"]
        k5 = np.zeros(n, np.int8)
        arm_d, arm_left = 0, 0
        for i in range(1, n):
            if s1l[i] != 0:
                arm_d, arm_left = int(s1l[i]), 16
                continue
            if arm_left > 0:
                arm_left -= 1
                if arm_d == 1 and c[i] > h[i - 1]:
                    k5[i], arm_left = 1, 0
                elif arm_d == -1 and c[i] < lo[i - 1]:
                    k5[i], arm_left = -1, 0
        sets["K5"] = {"sig": k5, "own_l": own1[0], "own_s": own1[1]}
        feat.update({
            "zP": zP, "prem8": b["f_prem8"], "dprem": b["f_dprem"], "zOI4": b["f_zOI4"],
            "oi1": b["f_oi1"], "tb1": b["f_tb1"] - 0.5, "tb4": b["f_tb4"] - 0.5,
            "zC": b["f_zC"], "zD": b["f_zD"], "fund": b["f_f_last"], "cvdz": cz,
        })  # fmt: skip
        signed |= {"zP", "dprem", "tb1", "tb4", "cvdz", "prem8", "fund", "zC"}
    order = [k for k in SINGLE + COMBO if k in sets]
    sets = {k: sets[k] for k in order}
    comp = {"sets": sets, "atr": atr, "st_l": st_l, "st_s": st_s, "feat": feat, "signed": signed,
            "single": [k for k in SINGLE if k in sets]}  # fmt: skip
    return comp, thr


def build_price_only(b: dict) -> tuple:
    return build(b, None, None, nonprice=False)
