"""139번: 지표 전체 묶음 학습 가중치(P1)와 조합 규칙 탐색(P2).

등록: 2026-11-07-btc-indicator-library-hypotheses.md
실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_indicator_library.py \
    --out <폴더> [--stage feat|p1|p2|all] [--no15]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

M5 = 300_000
B15 = 900_000
DAY = 86_400_000
R = 0.015
HOLD15 = 192  # 라벨 2일
HOLD5 = 576  # 매매 최대 2일
FEE = 0.0005
EMB = 200
FRAMES = (("15m", 3), ("1h", 12), ("4h", 48), ("1d", 288))
YEARS = range(2021, 2027)


def ts(y: int) -> float:
    return datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


# ---------- 자료 ----------
def load_fut5(path: Path) -> dict:
    cols = {k: [] for k in ("t", "o", "h", "l", "c", "v", "tb")}
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row or not row[0].isdigit():
                continue
            cols["t"].append(float(row[0]))
            for k, j in (("o", 1), ("h", 2), ("l", 3), ("c", 4), ("v", 5)):
                cols[k].append(float(row[j]))
            try:
                cols["tb"].append(float(row[9]))
            except (ValueError, IndexError):
                cols["tb"].append(np.nan)
    return {k: np.array(v, dtype=float) for k, v in cols.items()}


def resample(d: dict, factor: int) -> dict:
    if factor == 1:
        return d
    span = M5 * factor
    key = (d["t"] // span).astype(np.int64)
    edges = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    st, en = edges[:-1], edges[1:]
    full = (en - st) == factor
    st, en = st[full], en[full]
    allst = edges[:-1]
    hmax = np.maximum.reduceat(d["h"], allst)
    lmin = np.minimum.reduceat(d["l"], allst)
    vs = np.add.reduceat(d["v"], allst)
    tbs = np.add.reduceat(np.nan_to_num(d["tb"], nan=np.nan), allst)
    sel = np.flatnonzero(full)
    return {"t": key[st].astype(float) * span, "o": d["o"][st], "h": hmax[sel], "l": lmin[sel],
            "c": d["c"][en - 1], "v": vs[sel], "tb": tbs[sel]}


def load_funding(path: Path):
    t, r = [], []
    with path.open() as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            t.append(float(row[0]))
            r.append(float(row[2]))
    return np.array(t), np.array(r)


def load_oi(path: Path):
    tk, oi = [], []
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            try:
                t = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            tk.append(t.timestamp() * 1000 + M5)  # create_time은 구간 시작
            v = float(row[2]) if row[2] else np.nan
            oi.append(v if v > 0 else np.nan)
    tk = np.array(tk)
    o = np.argsort(tk, kind="stable")
    return tk[o], np.array(oi)[o]


# ---------- 지표 ----------
def ema(x, n):
    a = 2 / (n + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def wilder(x, n):
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    out[n] = np.nanmean(x[1:n + 1])
    for i in range(n + 1, len(x)):
        out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    return out


def roll_max(x, n):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = sw(x, n).max(1)
    return out


def roll_min(x, n):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = sw(x, n).min(1)
    return out


def roll_mean_std(x, n):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    m = np.full(len(x), np.nan)
    s = np.full(len(x), np.nan)
    if len(x) >= n:
        w = sw(x, n)
        m[n - 1:] = w.mean(1)
        s[n - 1:] = w.std(1)
    return m, s


def shift(x, k):
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:-k] if k > 0 else x
    return out


def frame_features(d: dict) -> tuple[list, list]:
    from btc_fractal import cci, crossed, recent, rsi, stoch
    h, l, c, v = d["h"], d["l"], d["c"], d["v"]  # noqa: E741
    names, cols = [], []

    def add(n, x):
        names.append(n)
        cols.append(np.asarray(x, dtype=float))

    st, cc, rs = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    tr = np.maximum(h - l, np.maximum(np.abs(h - shift(c, 1)), np.abs(l - shift(c, 1))))
    tr[0] = h[0] - l[0]
    atr = wilder(tr, 14)
    atrs = np.where(atr > 0, atr, np.nan)
    add("stoch", st)
    add("cci", cc)
    add("rsi", rs)
    add("stoch_sl", st - shift(st, 3))
    add("cci_sl", cc - shift(cc, 3))
    add("rsi_sl", rs - shift(rs, 3))
    l1 = recent(crossed(st, 20, True))
    l2 = l1 & recent(crossed(cc, -100, True))
    l3 = l2 & recent(crossed(rs, 30, True))
    s1 = recent(crossed(st, 80, False))
    s2 = s1 & recent(crossed(cc, 100, False))
    s3 = s2 & recent(crossed(rs, 70, False))
    lv = l1.astype(float) + l2 + l3 - (s1.astype(float) + s2 + s3)
    add("scr_lv", lv)
    m = ema(c, 12) - ema(c, 26)
    sg = ema(m, 9)
    hist = m - sg
    add("macd", m / c)
    add("macd_sig", sg / c)
    add("macd_hist", hist / c)
    add("macd_hsl", (hist - shift(hist, 3)) / c)
    ten = (roll_max(h, 9) + roll_min(l, 9)) / 2
    kij = (roll_max(h, 26) + roll_min(l, 26)) / 2
    spa = shift((ten + kij) / 2, 26)
    spb = shift((roll_max(h, 52) + roll_min(l, 52)) / 2, 26)
    top, bot = np.fmax(spa, spb), np.fmin(spa, spb)
    dist = np.where(c > top, c - top, np.where(c < bot, c - bot, 0.0))
    add("ichi_dist", dist / atrs)
    add("ichi_thick", (top - bot) / atrs)
    add("ichi_tk", (ten - kij) / atrs)
    add("ichi_chikou", c / shift(c, 26) - 1)
    up, dn = h - shift(h, 1), shift(l, 1) - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdi = 100 * wilder(pdm, 14) / atrs
    mdi = 100 * wilder(mdm, 14) / atrs
    dx = 100 * np.abs(pdi - mdi) / np.where(pdi + mdi > 0, pdi + mdi, np.nan)
    adx = wilder(np.nan_to_num(dx, nan=0.0), 14)
    add("adx", adx)
    add("pdi", pdi)
    add("mdi", mdi)
    add("di_spread", pdi - mdi)
    add("adx_sl", adx - shift(adx, 3))
    for n in (20, 50, 200):
        add(f"ema{n}", (c - ema(c, n)) / atrs)
    bm, bs = roll_mean_std(c, 20)
    add("bb_pb", (c - (bm - 2 * bs)) / np.where(bs > 0, 4 * bs, np.nan))
    add("bb_w", 4 * bs / bm)
    add("atr_pct", atr / c)
    vm, vs = roll_mean_std(shift(v, 1), 96)
    add("vol_z", (v - vm) / np.where(vs > 0, vs, np.nan))
    tb4 = np.convolve(np.nan_to_num(d["tb"], nan=np.nan), np.ones(4), "full")[:len(v)]
    v4 = np.convolve(v, np.ones(4), "full")[:len(v)]
    add("taker", tb4 / np.where(v4 > 0, v4, np.nan))
    return names, cols


DIRECTIONAL = {"stoch", "cci", "rsi", "stoch_sl", "cci_sl", "rsi_sl", "scr_lv", "macd", "macd_sig",
               "macd_hist", "macd_hsl", "ichi_dist", "ichi_tk", "ichi_chikou", "di_spread", "ema20",
               "ema50", "ema200", "bb_pb", "taker", "funding", "oi_1h", "oi_4h"}


def build(base: Path, out: Path, no15: bool) -> None:
    f5 = load_fut5(base / "futures/klines_5m.csv")
    d15 = resample(f5, 3)
    t15 = d15["t"]
    close15 = t15 + B15
    names, cols = [], []
    for fname, k in FRAMES:
        if no15 and fname == "15m":
            continue
        d = resample(f5, k)
        nm, cl = frame_features(d)
        cts = d["t"] + k * M5
        j = np.searchsorted(cts, close15, side="right") - 1
        ok = j >= 0
        for n, x in zip(nm, cl):
            y = np.full(len(t15), np.nan, dtype=np.float32)
            y[ok] = x[j[ok]]
            names.append(f"{fname}_{n}")
            cols.append(y)
        print("frame", fname, len(nm), flush=True)
    ft, fr = load_funding(base / "futures/funding.csv")
    j = np.searchsorted(ft, close15, side="right") - 1
    fund = np.where(j >= 0, fr[np.maximum(j, 0)], np.nan).astype(np.float32)
    names.append("funding")
    cols.append(fund)
    tk, oi = load_oi(base / "futures/metrics.csv")
    for hname, hms in (("oi_1h", 3_600_000), ("oi_4h", 14_400_000)):
        a = np.searchsorted(tk, close15, side="right") - 1
        b = np.searchsorted(tk, close15 - hms, side="right") - 1
        okk = (a >= 0) & (b >= 0)
        aa, bb = np.maximum(a, 0), np.maximum(b, 0)
        okk &= (close15 - tk[aa] <= 1_800_000) & (close15 - hms - tk[bb] <= 1_800_000)
        with np.errstate(invalid="ignore", divide="ignore"):
            val = np.where(okk, np.log(oi[aa] / oi[bb]), np.nan)
        names.append(hname)
        cols.append(val.astype(np.float32))
    X = np.stack(cols, 1).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    # 라벨: 다음 15분봉 시가에서 192봉 안에 +1.5% 먼저 = 1
    from numpy.lib.stride_tricks import sliding_window_view as sw
    o, h, l = d15["o"], d15["h"], d15["l"]  # noqa: E741
    n = len(o)
    y = np.full(n, np.nan, dtype=np.float32)
    H = np.r_[h, np.full(HOLD15 + 1, np.nan)]
    L = np.r_[l, np.full(HOLD15 + 1, np.nan)]
    step = 20000
    for s0 in range(0, n - 1, step):
        idx = np.arange(s0, min(n - 1, s0 + step))
        px = o[idx + 1]
        hw = sw(H[1:], HOLD15)[idx]
        lw = sw(L[1:], HOLD15)[idx]
        up = hw >= (px * (1 + R))[:, None]
        dn = lw <= (px * (1 - R))[:, None]
        fu = np.where(up.any(1), up.argmax(1), 10**6)
        fd = np.where(dn.any(1), dn.argmax(1), 10**6)
        lab = np.where(fu < fd, 1.0, np.where(fd < fu, 0.0, np.nan))  # 같은 봉이면 0(손절 먼저)
        lab = np.where((fu == fd) & (fu < 10**6), 0.0, lab)
        y[idx] = lab
    np.savez_compressed(out / "feat.npz", X=X, y=y, t=t15, names=np.array(names))
    print("features", X.shape, "labels", np.isfinite(y).sum(), flush=True)


# ---------- 매매 결과(모든 15분봉, 롱·숏) ----------
def outcomes(base: Path, out: Path, t15: np.ndarray) -> None:
    f5 = load_fut5(base / "futures/klines_5m.csv")
    t5, o5, h5, l5, c5 = f5["t"], f5["o"], f5["h"], f5["l"], f5["c"]
    ft, fr = load_funding(base / "futures/funding.csv")
    cf = np.r_[0.0, np.cumsum(fr)]
    n = len(t15)
    res = {s: (np.full(n, np.nan, np.float32), np.full(n, -1, np.int64)) for s in (1, -1)}
    ent = np.searchsorted(t5, t15 + B15 + M5)  # 신호 봉 마감 + 5분의 5분봉
    okt = (ent < len(t5)) & (t5[np.minimum(ent, len(t5) - 1)] == t15 + B15 + M5)
    for i in np.flatnonzero(okt):
        e = ent[i]
        if e + 2 >= len(t5):
            continue
        px = o5[e]
        hi, lo = h5[e:e + HOLD5], l5[e:e + HOLD5]
        for s in (1, -1):
            tp, sl = px * (1 + R * s), px * (1 - R * s)
            if s > 0:
                a, b = hi >= tp, lo <= sl
            else:
                a, b = lo <= tp, hi >= sl
            ia = a.argmax() if a.any() else 10**6
            ib = b.argmax() if b.any() else 10**6
            if ib <= ia and ib < 10**6:
                k = ib
                fill = min(o5[e + k], sl) if s > 0 else max(o5[e + k], sl)
                if k == 0:
                    fill = sl
            elif ia < 10**6:
                k = ia
                fill = tp
            else:
                k = len(hi) - 1
                fill = c5[e + k]
            x = e + k
            g = (fill / px - 1) * s
            fa = np.searchsorted(ft, t5[e], side="right")
            fb = np.searchsorted(ft, t5[x] + M5, side="right")
            g -= s * (cf[fb] - cf[fa])
            res[s][0][i] = g - 2 * FEE
            res[s][1][i] = x
    np.savez_compressed(out / "outcomes.npz", ln=res[1][0], lx=res[1][1], sn=res[-1][0], sx=res[-1][1],
                        t5=t5)
    print("outcomes done", flush=True)


# ---------- 평가 공통 ----------
def walk(sig_l, sig_s, ln, sn, lx, sx, t5, t15, i0, i1, cost_extra=0.0):
    """포지션 하나. 신호 봉 i의 결과(진입은 i 마감+5분). 반환: [(i, side, net, exit_time)]"""
    tr = []
    busy = -1.0
    idx = np.flatnonzero((sig_l | sig_s)[i0:i1]) + i0
    for i in idx:
        if t15[i] + B15 + M5 <= busy:
            continue
        s = 1 if sig_l[i] else -1
        net = ln[i] if s > 0 else sn[i]
        x = lx[i] if s > 0 else sx[i]
        if not np.isfinite(net) or x < 0:
            continue
        tr.append((int(i), s, float(net) - cost_extra, float(t5[x])))
        busy = float(t5[x])
    return tr


def stats(tr, t15, lev=1.0, yrs=None) -> dict:
    if not tr:
        return {"n": 0}
    g = np.array([z[2] for z in tr])
    s = np.array([z[1] for z in tr])
    eq = np.cumprod(1 + np.maximum(lev * g, -1))
    pk = np.maximum.accumulate(np.r_[1.0, eq])
    if yrs is None:
        yrs = max((t15[tr[-1][0]] - t15[tr[0][0]]) / (365.25 * DAY), 0.25)
    cagr = eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 else -1.0
    mdd = float((eq / pk[1:] - 1).min())
    return {"n": len(g), "win": round(float((g > 0).mean() * 100), 1), "ev": round(float(g.mean() * 100), 3),
            "long": round(float(g[s > 0].mean() * 100), 3) if (s > 0).any() else None,
            "short": round(float(g[s < 0].mean() * 100), 3) if (s < 0).any() else None,
            "nl": int((s > 0).sum()), "ns": int((s < 0).sum()),
            "cagr": round(cagr * 100, 1), "mdd": round(mdd * 100, 1),
            "ratio": round(cagr / abs(mdd), 2) if mdd < 0 else None}


def daily_mdd(tr, t15, t5, c5, lev=1.0):
    """일별 평가 낙폭: 보유 중에는 매일 마감가로 평가."""
    if not tr:
        return 0.0
    eq = 1.0
    vals = [1.0]
    t5d = t5 // DAY
    for i, s, net, xt in tr:
        e_t = t15[i] + B15 + M5
        e = int(np.searchsorted(t5, e_t))
        x = int(np.searchsorted(t5, xt))
        px = c5[e - 1] if e > 0 else c5[e]
        days = np.flatnonzero(np.r_[t5d[e + 1:x + 1] != t5d[e:x], False])
        for k in days:
            j = e + k
            vals.append(eq * (1 + lev * (c5[j] / px - 1) * s))
        eq *= 1 + max(lev * net, -1)
        vals.append(eq)
    v = np.array(vals)
    return float((v / np.maximum.accumulate(v) - 1).min())


def circ_null(sig_l, sig_s, ln, sn, lx, sx, t5, t15, i0, i1, n=1000, seed=7):
    rng = np.random.default_rng(seed)
    L, S = sig_l[i0:i1], sig_s[i0:i1]
    span = i1 - i0
    out = []
    for _ in range(n):
        k = int(rng.integers(96 * 7, span - 96 * 7))
        a, b = np.zeros_like(sig_l), np.zeros_like(sig_s)
        a[i0:i1], b[i0:i1] = np.roll(L, k), np.roll(S, k)
        tr = walk(a, b, ln, sn, lx, sx, t5, t15, i0, i1)
        out.append(np.mean([z[2] for z in tr]) if tr else 0.0)
    return np.array(out)


# ---------- P1 ----------
def p1(out: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    F = np.load(out / "feat.npz", allow_pickle=True)
    X, y, t15, names = F["X"], F["y"], F["t"], list(F["names"])
    preds = {"L1": np.full(len(t15), np.nan, np.float32)}
    for k in range(5):
        preds[f"X{k}"] = np.full(len(t15), np.nan, np.float32)
    weights = {}
    for Y in YEARS:
        a, b = int(np.searchsorted(t15, ts(Y))), int(np.searchsorted(t15, ts(Y + 1)))
        tr_end = a - EMB
        i_tr = np.arange(int(np.searchsorted(t15, ts(2020) + 60 * DAY)), tr_end, 3)
        i_tr = i_tr[np.isfinite(y[i_tr])]
        Xt, yt = X[i_tr], y[i_tr]
        med = np.nanmedian(Xt, 0)
        med = np.where(np.isfinite(med), med, 0)
        Xf = np.where(np.isfinite(Xt), Xt, med)
        mu, sd = Xf.mean(0), Xf.std(0)
        sd = np.where(sd > 0, sd, 1)
        Z = (Xf - mu) / sd
        lr = LogisticRegression(penalty="l1", C=0.05, solver="liblinear", max_iter=300)
        lr.fit(Z, yt)
        weights[Y] = lr.coef_[0].tolist()
        Xs = X[a:b]
        Zs = (np.where(np.isfinite(Xs), Xs, med) - mu) / sd
        preds["L1"][a:b] = lr.predict_proba(Zs)[:, 1]
        for k in range(5):
            m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, n_jobs=2, random_state=Y + 1000 * k,
                                  tree_method="hist", eval_metric="logloss")
            m.fit(Xt, yt)
            preds[f"X{k}"][a:b] = m.predict_proba(Xs)[:, 1]
        print("year", Y, "train", len(i_tr), "nz L1", int(np.sum(np.abs(lr.coef_[0]) > 0)), flush=True)
    preds["XGB"] = np.mean([preds[f"X{k}"] for k in range(5)], 0).astype(np.float32)
    np.savez_compressed(out / "p1_pred.npz", **preds)
    json.dump({"names": names, "weights": {str(k): v for k, v in weights.items()}},
              open(out / "p1_weights.json", "w"))


def auc(yv, p):
    ok = np.isfinite(yv) & np.isfinite(p)
    yv, p = yv[ok], p[ok]
    if yv.sum() == 0 or yv.sum() == len(yv):
        return None
    r = p.argsort().argsort() + 1
    n1 = yv.sum()
    return float((r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yv) - n1)))


def signals_from(p, t15, q):
    L = np.zeros(len(p), bool)
    S = np.zeros(len(p), bool)
    for Y in range(2022, 2027):
        pa, pb = int(np.searchsorted(t15, ts(Y - 1))), int(np.searchsorted(t15, ts(Y)))
        a, b = pb, int(np.searchsorted(t15, ts(Y + 1)))
        prev = p[pa:pb]
        prev = prev[np.isfinite(prev)]
        if len(prev) < 1000:
            continue
        hi, lo = np.quantile(prev, q), np.quantile(prev, 1 - q)
        L[a:b] = p[a:b] >= hi
        S[a:b] = p[a:b] <= lo
    return L, S


def p1_eval(out: Path, base: Path) -> dict:
    F = np.load(out / "feat.npz", allow_pickle=True)
    t15, y = F["t"], F["y"]
    O = np.load(out / "outcomes.npz")
    ln, sn, lx, sx, t5 = O["ln"], O["sn"], O["lx"], O["sx"], O["t5"]
    f5 = load_fut5(base / "futures/klines_5m.csv")
    c5 = f5["c"]
    P = np.load(out / "p1_pred.npz")
    i23, i24 = int(np.searchsorted(t15, ts(2023))), int(np.searchsorted(t15, ts(2024)))
    iend = len(t15) - 2
    rep = {"auc": {}}
    for k in ("L1", "XGB"):
        rep["auc"][k] = {Y: auc(y[int(np.searchsorted(t15, ts(Y))):int(np.searchsorted(t15, ts(Y + 1)))],
                                P[k][int(np.searchsorted(t15, ts(Y))):int(np.searchsorted(t15, ts(Y + 1)))])
                         for Y in YEARS}
    for model in ("L1", "XGB"):
        for q in (0.90, 0.95):
            name = f"{model}-q{int(q * 100)}"
            Ls, Ss = signals_from(P[model], t15, q)
            r = {}
            for per, a, b in (("2023", i23, i24), ("2024~", i24, iend)):
                tr = walk(Ls, Ss, ln, sn, lx, sx, t5, t15, a, b)
                yrs = (t15[b - 1] - t15[a]) / (365.25 * DAY)
                r[per] = stats(tr, t15, 1.0, yrs)
                r[per + "_3x"] = stats(tr, t15, 3.0, yrs)
                r[per]["mdd_daily"] = round(daily_mdd(tr, t15, t5, c5) * 100, 1)
                r[per + "_3x"]["mdd_daily"] = round(daily_mdd(tr, t15, t5, c5, 3.0) * 100, 1)
                if r[per]["n"] and r[per]["mdd_daily"] < 0:
                    r[per]["ratio_daily"] = round(r[per]["cagr"] / abs(r[per]["mdd_daily"]), 2)
                tr15 = walk(Ls, Ss, ln, sn, lx, sx, t5, t15, a, b, cost_extra=0.001)
                r[per + "_cost15"] = stats(tr15, t15, 1.0, yrs)
            if model == "XGB":
                seeds = []
                for k in range(5):
                    Lk, Sk = signals_from(P[f"X{k}"], t15, q)
                    seeds.append(stats(walk(Lk, Sk, ln, sn, lx, sx, t5, t15, i24, iend), t15))
                r["seeds_2024"] = seeds
            null = circ_null(Ls, Ss, ln, sn, lx, sx, t5, t15, i24, iend)
            act = r["2024~"].get("ev", 0) / 100
            r["null_mean_pct"] = round(float(null.mean() * 100), 3)
            r["null_p"] = round(float((np.sum(null >= act) + 1) / (len(null) + 1)), 4)
            rep[name] = r
            print(name, r["2023"], r["2024~"], r["null_p"], flush=True)
    return rep


# ---------- P2 ----------
def p2(out: Path) -> dict:
    F = np.load(out / "feat.npz", allow_pickle=True)
    X, t15, names = F["X"], F["t"], [str(x) for x in F["names"]]
    O = np.load(out / "outcomes.npz")
    ln, sn = O["ln"], O["sn"]
    i20 = int(np.searchsorted(t15, ts(2020) + 60 * DAY))
    i23, i24 = int(np.searchsorted(t15, ts(2023))), int(np.searchsorted(t15, ts(2024)))
    iend = len(t15) - 2
    nf = X.shape[1]
    qs = (0.10, 0.25, 0.75, 0.90)
    Q = np.nanquantile(X[i20:i23], qs, axis=0)  # (4, nf)
    base = [n.split("_", 1)[1] if n.split("_", 1)[0] in ("15m", "1h", "4h", "1d") else n for n in names]

    def cond(f, k):
        x = X[:, f]
        with np.errstate(invalid="ignore"):
            return x <= Q[k, f] if k < 2 else x >= Q[k, f]

    def mirror(f, k):
        b = base[f]
        if b in DIRECTIONAL:
            return f, 3 - k
        if b in ("pdi", "mdi"):
            other = names[f].replace("pdi", "mdi") if b == "pdi" else names[f].replace("mdi", "pdi")
            return names.index(other), k
        return f, k

    conds = [(f, k) for f in range(nf) for k in range(4)]
    cache_l = {}

    def cl(c):
        if c not in cache_l:
            cache_l[c] = cond(*c)
        return cache_l[c]

    def edges(m):
        return m & ~np.r_[False, m[:-1]]

    def score(rule, a, b):
        ml = np.ones(len(t15), bool)
        ms = np.ones(len(t15), bool)
        for c in rule:
            ml &= cl(c)
            ms &= cl(mirror(*c))
        el, es = edges(ml)[a:b], edges(ms)[a:b]
        g = np.r_[ln[a:b][el], sn[a:b][es]]
        g = g[np.isfinite(g)]
        if len(g) < 2:
            return None
        return {"n": int(len(g)), "ev": float(g.mean()), "t": float(g.mean() / (g.std() / np.sqrt(len(g)) + 1e-12)),
                "nl": int(np.isfinite(ln[a:b][el]).sum()), "ns": int(np.isfinite(sn[a:b][es]).sum()),
                "long": float(np.nanmean(ln[a:b][el])) if el.any() else None,
                "short": float(np.nanmean(sn[a:b][es])) if es.any() else None,
                "win": float((g > 0).mean())}

    t0 = time.time()
    singles = []
    for c in conds:
        s = score((c,), i20, i23)
        if s and s["n"] >= 100:
            singles.append((s["t"], (c,)))
    singles.sort(reverse=True)
    top_c = [r[0] for _, r in singles[:40]]
    pairs = {}
    for c1 in top_c:
        for c2 in conds:
            if c2[0] == c1[0]:
                continue
            key = tuple(sorted((c1, c2)))
            if key in pairs:
                continue
            s = score(key, i20, i23)
            if s and s["n"] >= 100:
                pairs[key] = s["t"]
    top_p = sorted(pairs.items(), key=lambda z: -z[1])[:40]
    triples = {}
    for key, _ in top_p:
        for c3 in conds:
            if c3[0] in (key[0][0], key[1][0]):
                continue
            k3 = tuple(sorted(key + (c3,)))
            if k3 in triples:
                continue
            s = score(k3, i20, i23)
            if s and s["n"] >= 100:
                triples[k3] = s["t"]
    allr = [(t, (c,)) for t, (c,) in [(x[0], x[1]) for x in singles]]
    allr += [(t, k) for k, t in pairs.items()] + [(t, k) for k, t in triples.items()]
    n_eval = len(conds) + len(pairs) + len(triples)
    allr.sort(key=lambda z: -z[0])
    top20 = [k for _, k in allr[:20]]
    print("p2 search", n_eval, "rules", round(time.time() - t0), "s", flush=True)

    def desc(rule):
        qn = ("≤10분위", "≤25분위", "≥75분위", "≥90분위")
        return " AND ".join(f"{names[f]} {qn[k]}({Q[k, f]:.4g})" for f, k in rule)

    rows = []
    for rule in top20:
        rows.append({"rule": desc(rule), "train": score(rule, i20, i23), "val2023": score(rule, i23, i24),
                     "hold": score(rule, i24, iend), "key": [[int(f), int(k)] for f, k in rule]})
    pos23 = sum(1 for r in rows if r["val2023"] and r["val2023"]["ev"] > 0)
    pos24 = sum(1 for r in rows if r["hold"] and r["hold"]["ev"] > 0)
    final = sorted([r for r in rows if r["val2023"] and r["val2023"]["n"] >= 30],
                   key=lambda r: -r["val2023"]["ev"])[:3]
    # 최종 3개: 비용 1.5배, 무작위 대조(사건 단위, 원형 이동 1000)
    rng = np.random.default_rng(7)
    for r in final:
        rule = [tuple(x) for x in r["key"]]
        ml = np.ones(len(t15), bool)
        ms = np.ones(len(t15), bool)
        for c in rule:
            ml &= cl(c)
            ms &= cl(mirror(*c))
        el, es = edges(ml), edges(ms)
        a, b = i24, iend
        g = np.r_[ln[a:b][el[a:b]], sn[a:b][es[a:b]]]
        g = g[np.isfinite(g)]
        r["hold_cost15_ev"] = float((g - 0.001).mean()) if len(g) else None
        act = g.mean() if len(g) else 0
        null = []
        L, S = el[a:b], es[a:b]
        for _ in range(1000):
            k = int(rng.integers(96 * 7, (b - a) - 96 * 7))
            gl = ln[a:b][np.roll(L, k)]
            gs = sn[a:b][np.roll(S, k)]
            gg = np.r_[gl, gs]
            gg = gg[np.isfinite(gg)]
            null.append(gg.mean() if len(gg) else 0)
        null = np.array(null)
        r["null_mean"] = float(null.mean())
        r["null_p"] = float((np.sum(null >= act) + 1) / 1001)
    return {"n_rules_evaluated": n_eval, "top20": rows, "pos2023": pos23, "pos2024": pos24, "final3": final}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--stage", default="all")
    ap.add_argument("--no15", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.data)
    t0 = time.time()
    if a.stage in ("feat", "all"):
        build(base, out, a.no15)
        F = np.load(out / "feat.npz", allow_pickle=True)
        outcomes(base, out, F["t"])
        print("feat stage", round(time.time() - t0), "s", flush=True)
    if a.stage in ("p1", "all"):
        p1(out)
        rep = p1_eval(out, base)
        json.dump(rep, open(out / "p1_result.json", "w"), ensure_ascii=False, indent=1, default=float)
        print("p1 stage", round(time.time() - t0), "s", flush=True)
    if a.stage in ("p2", "all"):
        rep = p2(out)
        json.dump(rep, open(out / "p2_result.json", "w"), ensure_ascii=False, indent=1, default=float)
        print("p2 stage", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
