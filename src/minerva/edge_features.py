"""24시간 방향 모델(연구 8차) 특징 65개를 연구 코드와 똑같이 만든다. 순수 numpy, 네트워크 없음.

연구(2026-11-20 8차 `build.py`, 11-21 F·10차가 그대로 씀)의 `symbol_feats`·단면 순위를 옮겨 왔다.
값이 연구와 같아야 모델 점수가 같으므로 계산 순서·초깃값(EMA 0 시작, 지그재그 0 시작, 2020-01-01
시간 격자)을 바꾸지 않는다. 시험 `tests/test_edge_features.py`가 연구 코드로 만든 값과 맞춰 본다.

- 시간 격자: 0번 = 2020-01-01 00:00 UTC, 1시간 단위. 결정 시각 T = 4시간 격자(0, 4, 8, …).
- T의 특징은 T-1시(마감한 마지막 1시간봉)까지, 4시간봉은 T/4-1번, 일봉은 T/24-1번(어제 UTC),
  펀딩은 T 이하에 정산된 것만 쓴다. 배열 뒤쪽(T 이후)을 바꿔도 T 값은 그대로다(시험으로 확인).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.signal import lfilter

T0_MS = 1577836800000  # 2020-01-01 00:00 UTC
HR_MS = 3_600_000

SYM_KEYS = [
    "r1h", "r4h", "r12h", "r24h", "r3d", "r7d", "rv24", "rv7d", "rvratio", "pos7d", "volz",
    "h1_stoch", "h1_cci", "h1_rsi",
    "h4_stoch", "h4_cci", "h4_rsi", "h4_rdir", "h4_rdist", "h4_rsince", "h4_rflips",
    "h4_ma5", "h4_ma10", "h4_ma20", "h4_ma50", "h4_stack", "h4_rngpct",
    "d1_stoch", "d1_cci", "d1_rsi", "d1_rdir", "d1_rdist", "d1_rsince", "d1_rflips",
    "d1_ma20", "d1_ma50", "f_last", "f_m3d", "f_chg",
]
MKT_SRC = [("r4h", "r4h"), ("r24h", "r24h"), ("r7d", "r7d"), ("rv24", "rv24"),
           ("rsi4h", "h4_rsi"), ("rsi1d", "d1_rsi"), ("rdir4h", "h4_rdir"),
           ("rdir1d", "d1_rdir"), ("rdist4h", "h4_rdist"), ("rdist1d", "d1_rdist")]
MKT_KEYS = [f"{p}_{k}" for p in ("btc", "eth") for k, _ in MKT_SRC]
XS_SRC = [("xs_r24h", "r24h"), ("xs_r7d", "r7d"), ("xs_fund", "f_last"),
          ("xs_volz", "volz"), ("xs_rv24", "rv24")]
XS_KEYS = [k for k, _ in XS_SRC] + ["xs_r24h_dm"]
NAMES = SYM_KEYS + MKT_KEYS + XS_KEYS  # 65개, 연구 rows.npz의 names 순서


def ffill(x: np.ndarray) -> np.ndarray:
    idx = np.where(np.isnan(x), 0, np.arange(len(x)))
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def ema(x: np.ndarray, a: float) -> np.ndarray:
    return lfilter([a], [1, -(1 - a)], np.nan_to_num(x))


def rsi(c: np.ndarray) -> np.ndarray:
    d = np.diff(c, prepend=c[0])
    up = ema(np.maximum(d, 0), 1 / 14)
    dn = ema(np.maximum(-d, 0), 1 / 14)
    return 100 - 100 / (1 + up / np.maximum(dn, 1e-12))


def roll(x: np.ndarray, k: int, f) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= k:
        out[k - 1:] = f(swv(x, k), axis=1)
    return out


def stoch(h: np.ndarray, low: np.ndarray, c: np.ndarray, k: int = 5) -> np.ndarray:
    hh = roll(h, k, np.max)
    ll = roll(low, k, np.min)
    return 100 * (c - ll) / np.maximum(hh - ll, 1e-12)


def cci(h: np.ndarray, low: np.ndarray, c: np.ndarray, k: int = 20) -> np.ndarray:
    tp = (h + low + c) / 3
    out = np.full(len(c), np.nan)
    if len(tp) < k:
        return out
    w = swv(tp, k)
    m = w.mean(1)
    md = np.abs(w - m[:, None]).mean(1)
    out[k - 1:] = (tp[k - 1:] - m) / np.maximum(0.015 * md, 1e-12)
    return out


def sma(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= k:
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[k - 1:] = (cs[k:] - cs[:-k]) / k
    return out


def scan(c: np.ndarray, th: float):
    """앞만 보는 지그재그(btc_regime.scan과 같은 규칙): 방향, 전환 가격, 마지막 전환 뒤 봉 수,
    최근 60봉 전환 횟수."""
    n = len(c)
    lo = hi = 0
    d = 0
    dirs = np.zeros(n)
    level = np.full(n, np.nan)
    since = np.full(n, np.nan)
    flip = np.zeros(n)
    last = -1
    for i in range(1, n):
        prev = d
        if d == 0:
            if c[i] > c[hi]:
                hi = i
            if c[i] < c[lo]:
                lo = i
            if c[i] >= c[lo] * (1 + th):
                d = 1
                hi = i
            elif c[i] <= c[hi] * (1 - th):
                d = -1
                lo = i
        elif d == 1:
            if c[i] > c[hi]:
                hi = i
            elif c[i] <= c[hi] * (1 - th):
                d = -1
                lo = i
        else:
            if c[i] < c[lo]:
                lo = i
            elif c[i] >= c[lo] * (1 + th):
                d = 1
                hi = i
        if d != prev:
            last = i
            flip[i] = 1
        dirs[i] = d
        if d == 1:
            level[i] = c[hi] * (1 - th)
        elif d == -1:
            level[i] = c[lo] * (1 + th)
        if last >= 0:
            since[i] = i - last
    fl = np.convolve(flip, np.ones(60))[:n]
    return dirs, level, since, fl


def agg(o, h, low, c, q, k: int):
    """1시간 배열을 k시간 봉으로. b번 봉 = [k*b, k*b+k) 시."""
    nb = len(c) // k
    big_o = o[:nb * k:k]
    big_c = c[k - 1:nb * k:k]
    big_h = h[:nb * k].reshape(nb, k).max(1)
    big_l = low[:nb * k].reshape(nb, k).min(1)
    big_q = q[:nb * k].reshape(nb, k).sum(1)
    return big_o, big_h, big_l, big_c, big_q


def tf_feats(h, low, c, th: float | None = None) -> dict:
    f = {"stoch": stoch(h, low, c), "cci": cci(h, low, c), "rsi": rsi(c)}
    if th is not None:
        d, lev, since, fl = scan(c, th)
        f["rdir"] = d
        with np.errstate(invalid="ignore", divide="ignore"):
            f["rdist"] = np.where(d != 0, (c / lev - 1) * d, np.nan)
        f["rsince"] = since
        f["rflips"] = fl
    return f


def symbol_feats(o, h, low, c, q, fh, fr, nh: int):
    """한 종목의 결정 시각마다 특징(연구 build.symbol_feats와 같은 계산).

    o,h,low,c,q: 길이 nh의 시간 배열(없는 시간은 nan). fh: 펀딩 정산 시간 번호, fr: 펀딩률.
    돌려줌: (T 배열, 특징 dict, 행 유효 여부). 행은 T-1시 봉이 있고 첫 봉에서 60일 지난
    뒤만 유효."""
    o, h, low, c, q = (np.array(a, dtype=float, copy=True) for a in (o, h, low, c, q))
    present = ~np.isnan(c)
    first = int(np.argmax(present)) if present.any() else nh
    c = ffill(c)
    if first < nh:
        c[:first] = c[first]
    o = np.where(np.isnan(o), c, o)
    h = np.where(np.isnan(h), c, h)
    low = np.where(np.isnan(low), c, low)
    q = np.nan_to_num(q)
    with np.errstate(invalid="ignore", divide="ignore"):
        lc = np.log(c)
        feats = {}
        big_t = np.arange(0, nh, 4)
        jj = np.maximum(big_t - 1, 0)
        for k, n in (("r1h", 1), ("r4h", 4), ("r12h", 12), ("r24h", 24), ("r3d", 72),
                     ("r7d", 168)):
            feats[k] = lc[jj] - lc[np.maximum(jj - n, 0)]
        r1 = np.diff(lc, prepend=np.nan)
        rv24 = roll(np.nan_to_num(r1), 24, np.std)
        rv7 = roll(np.nan_to_num(r1), 168, np.std)
        feats["rv24"] = rv24[jj]
        feats["rv7d"] = rv7[jj]
        feats["rvratio"] = feats["rv24"] / np.maximum(feats["rv7d"], 1e-9)
        hh7 = roll(h, 168, np.max)
        ll7 = roll(low, 168, np.min)
        feats["pos7d"] = ((c - ll7) / np.maximum(hh7 - ll7, 1e-12))[jj]
        lq = np.log1p(q)
        q24 = np.nan_to_num(sma(lq, 24))
        m30 = sma(q24, 720)
        s30 = np.sqrt(np.maximum(sma(q24 ** 2, 720) - m30 ** 2, 1e-12))
        feats["volz"] = ((q24 - np.roll(m30, 24)) / np.roll(s30, 24))[jj]
        cz = np.nan_to_num(c, nan=1.0)
        hz = np.nan_to_num(h, nan=1.0)
        lz = np.nan_to_num(low, nan=1.0)
        for k, v in tf_feats(hz, lz, cz).items():
            feats["h1_" + k] = v[jj]
        _o4, h4, l4, c4, _q4 = agg(o, h, low, c, q, 4)
        c4z = np.nan_to_num(c4, nan=1.0)
        f4 = tf_feats(np.nan_to_num(h4, nan=1.0), np.nan_to_num(l4, nan=1.0), c4z, th=0.05)
        b = np.maximum(big_t // 4 - 1, 0)
        for k, v in f4.items():
            feats["h4_" + k] = v[b]
        for n in (5, 10, 20, 50):
            feats[f"h4_ma{n}"] = (c4 / sma(c4z, n) - 1)[b]
        ma = [sma(c4z, n) for n in (5, 10, 20, 50)]
        feats["h4_stack"] = (np.sign(ma[0] - ma[1]) + np.sign(ma[1] - ma[2])
                             + np.sign(ma[2] - ma[3]))[b]
        rng = roll(np.nan_to_num(h4, nan=1.0), 20, np.max) / roll(
            np.nan_to_num(l4, nan=1.0), 20, np.min) - 1
        pr = np.full(len(rng), np.nan)
        if len(rng) >= 121:
            w = swv(rng, 121)
            pr[120:] = (w[:, :120] < w[:, 120:121]).mean(1)
        feats["h4_rngpct"] = pr[b]
        _od, hd, ld, cd, _qd = agg(o, h, low, c, q, 24)
        cdz = np.nan_to_num(cd, nan=1.0)
        fd = tf_feats(np.nan_to_num(hd, nan=1.0), np.nan_to_num(ld, nan=1.0), cdz, th=0.10)
        dd = np.maximum(big_t // 24 - 1, 0)
        for k, v in fd.items():
            feats["d1_" + k] = v[dd]
        feats["d1_ma20"] = (cd / sma(cdz, 20) - 1)[dd]
        feats["d1_ma50"] = (cd / sma(cdz, 50) - 1)[dd]
        fh = np.asarray(fh, dtype=np.int64)
        fr = np.asarray(fr, dtype=float)
        order = np.argsort(fh)
        fh = fh[order]
        fr = fr[order]
        ix = np.searchsorted(fh, big_t, side="right") - 1
        cs = np.cumsum(np.insert(fr, 0, 0.0))
        ok = ix >= 0
        feats["f_last"] = np.where(ok, fr[np.maximum(ix, 0)] if len(fr) else np.nan, np.nan)
        lo9 = np.maximum(ix - 8, 0)
        feats["f_m3d"] = np.where(ok, (cs[ix + 1] - cs[lo9]) / np.maximum(ix + 1 - lo9, 1),
                                  np.nan)
        feats["f_chg"] = feats["f_last"] - feats["f_m3d"]
    valid = present[np.maximum(big_t - 1, 0)] & (big_t - 1 >= first + 24 * 60)
    return big_t, feats, valid


def label_ret(o, c, t: np.ndarray, hold: int, nh: int) -> np.ndarray:
    """학습 정답(연구 build의 ret{H}): T 시가 진입, T+H 시가 청산의 로그 수익.
    데이터 끝을 넘으면 nan.
    중간 상장폐지는 마지막 종가."""
    present = ~np.isnan(c)
    lastp = np.maximum.accumulate(np.where(present, np.arange(nh), 0))
    e = np.minimum(t + hold, nh - 1)
    valid = (t + hold < nh) & present[np.minimum(t, nh - 1)]
    px = np.where(present[e], o[e], c[lastp[np.maximum(e - 1, 0)]])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid, np.log(px / o[np.minimum(t, nh - 1)]), np.nan)


def market_feats(feats: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": feats[src] for k, src in MKT_SRC}


def cross_section(x_sym: np.ndarray, ti: np.ndarray) -> np.ndarray:
    """같은 결정 시각 행끼리 백분위 순위 5개 + 24시간 수익 − 중앙값(연구 main과 같은 계산).
    x_sym: (행, SYM_KEYS) 정렬은 (시각, 종목) 순이어야 한다."""
    col = {k: SYM_KEYS.index(src) for k, src in XS_SRC}
    out = np.full((len(ti), len(XS_SRC) + 1), np.nan, np.float32)
    if not len(ti):
        return out
    bounds = np.flatnonzero(np.diff(ti)) + 1
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(ti)]
    r24 = SYM_KEYS.index("r24h")
    for a, b in zip(starts, ends):
        for j, (k, _src) in enumerate(XS_SRC):
            v = x_sym[a:b, col[k]]
            okk = ~np.isnan(v)
            if okk.sum() >= 2:
                r = np.full(b - a, np.nan)
                r[okk] = np.argsort(np.argsort(v[okk])) / (okk.sum() - 1)
                out[a:b, j] = r
        v = x_sym[a:b, r24]
        out[a:b, -1] = v - np.nanmedian(v) if (~np.isnan(v)).any() else np.nan
    return out


def assemble(per_sym: dict, mkt: dict, sym_order: list[str]):
    """종목별 (행 위치, 특징 dict)와 BTC·ETH 시장 특징을 연구 rows.npz 모양으로 합친다.

    per_sym: {종목: (T 인덱스 배열(결정 격자 번호), feats dict(격자 전체))}
    mkt: {'btc': 격자 전체 dict, 'eth': …}. 돌려줌 (ti, 종목 이름 배열, X float32 (행, 65))."""
    syms = [s for s in sym_order if s in per_sym and len(per_sym[s][0])]
    if not syms:
        return np.zeros(0, np.int64), np.array([], dtype=object), np.zeros((0, len(NAMES)),
                                                                          np.float32)
    ti = np.concatenate([per_sym[s][0] for s in syms])
    si = np.concatenate([np.full(len(per_sym[s][0]), k) for k, s in enumerate(syms)])
    x = np.concatenate([np.stack([per_sym[s][1][k][per_sym[s][0]].astype(np.float32)
                                  for k in SYM_KEYS], 1) for s in syms])
    m = np.stack([np.concatenate([mkt[p][f"{p}_{k}"][per_sym[s][0]] for s in syms])
                  for p in ("btc", "eth") for k, _ in MKT_SRC], 1).astype(np.float32)
    o = np.lexsort((si, ti))
    ti, si, x, m = ti[o], si[o], x[o], m[o]
    xs = cross_section(x, ti)
    xall = np.concatenate([x, m, xs], 1).astype(np.float32)
    return ti, np.array(syms, dtype=object)[si], xall
