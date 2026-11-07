"""131번: BTC 급락(급등) 전조 사건 연구. 등록: 2026-11-07-btc-crash-precursor-hypotheses.md

실행: PYTHONPATH=<numpy·xgboost 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_crash_precursor.py \
    --out <폴더>
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import btc_fut_daily_regime as R
from btc_fractal import cci, load5, resample, rsi, stoch
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars
from btc_swing_patterns import zigzag

M5 = 300_000
BAR = 900_000
H = 3_600_000
DAY = 86_400_000
HZ = 16  # 4시간
DROP = 0.03
GAP = 96  # 24시간 정리
COST = 0.0005


def ts(y: int, m: int = 1, d: int = 1) -> float:
    return datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000


SPLIT = ts(2024)


def taker15(path: Path, t15: np.ndarray) -> np.ndarray:
    """15분봉마다 (거래량, 시장가 매수량). OKX 봉은 nan."""
    t, v, tk = [], [], []
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(int(row[0]))
            v.append(float(row[5]))
            tk.append(float(row[9]) if row[9] not in ("", "nan") else np.nan)
    t, v, tk = np.array(t, float), np.array(v), np.array(tk)
    key = (t // BAR).astype(np.int64)
    k15 = (t15 // BAR).astype(np.int64)
    idx = np.searchsorted(k15, key)
    ok = (idx < len(k15)) & (k15[np.minimum(idx, len(k15) - 1)] == key)
    vv = np.zeros(len(t15))
    tt = np.zeros(len(t15))
    nan = np.zeros(len(t15))
    np.add.at(vv, idx[ok], v[ok])
    np.add.at(tt, idx[ok], np.nan_to_num(tk[ok]))
    np.add.at(nan, idx[ok], np.isnan(tk[ok]).astype(float))
    tt[nan > 0] = np.nan
    return vv, tt


def load_metrics(path: Path):
    tk, oi, ls = [], [], []
    with path.open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            try:
                t = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            tk.append(t.timestamp() * 1000 + M5)  # create_time은 구간 시작: 값은 +5분에 정해진다
            oi.append(float(row[2]) if row[2] else np.nan)
            ls.append(float(row[4]) if row[4] else np.nan)
    tk = np.array(tk)
    o = np.argsort(tk, kind="stable")
    return tk[o], np.array(oi)[o], np.array(ls)[o]


def asof(tk: np.ndarray, val: np.ndarray, q: np.ndarray, stale: float = 1800_000) -> np.ndarray:
    j = np.searchsorted(tk, q, side="right") - 1
    out = np.full(len(q), np.nan)
    ok = j >= 0
    jj = np.maximum(j, 0)
    ok &= (q - tk[jj]) <= stale
    out[ok] = val[jj[ok]]
    return out


def htf(d5: dict, t15: np.ndarray, k: int) -> dict:
    """마감된 상위 봉의 Stoch·CCI·RSI(현재, 직전)를 15분봉 마감 시각에 맞춘다."""
    u = resample(d5, k)
    close = u["t"] + k * M5
    j = np.searchsorted(close, t15 + BAR, side="right") - 1
    s, c, r = stoch(u["h"], u["l"], u["c"], 5), cci(u["h"], u["l"], u["c"], 20), rsi(u["c"], 14)
    out = {}
    jj = np.maximum(j, 0)
    for nm, x in (("s", s), ("c", c), ("r", r)):
        cur = np.where(j >= 0, x[jj], np.nan)
        prev = np.where(j >= 1, x[np.maximum(j - 1, 0)], np.nan)
        out[nm] = cur
        out[nm + "_d"] = cur - prev
        for lag in (1, 2, 3):
            out[f"{nm}_l{lag}"] = np.where(j >= lag, x[np.maximum(j - lag, 0)], np.nan)
    u_atr = atr(u)
    out["atr"] = np.where(j >= 0, u_atr[jj], np.nan)
    return out


def atr(u: dict, n: int = 14) -> np.ndarray:
    pc = np.r_[u["c"][0], u["c"][:-1]]
    tr = np.maximum(u["h"] - u["l"], np.maximum(abs(u["h"] - pc), abs(u["l"] - pc)))
    out = np.full(len(tr), np.nan)
    if len(tr) > n:
        out[n - 1] = tr[:n].mean()
        for i in range(n, len(tr)):
            out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def roll_prev(x: np.ndarray, n: int):
    """직전 n봉(현재 제외) 평균·표준편차."""
    cs = np.cumsum(np.r_[0.0, x])
    cs2 = np.cumsum(np.r_[0.0, x * x])
    m = np.full(len(x), np.nan)
    sd = np.full(len(x), np.nan)
    i = np.arange(n, len(x))
    s1 = cs[i] - cs[i - n]
    s2 = cs2[i] - cs2[i - n]
    m[i] = s1 / n
    sd[i] = np.sqrt(np.maximum(s2 / n - m[i] ** 2, 0))
    return m, sd


def any_back(flag: np.ndarray, n: int) -> np.ndarray:
    """지난 n봉(현재 포함) 안에 참이 있었나."""
    cs = np.cumsum(np.r_[0, flag.astype(int)])
    i = np.arange(len(flag))
    return (cs[i + 1] - cs[np.maximum(i + 1 - (n + 1), 0)]) > 0


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, c - h, c + h)


def declutter(flag: np.ndarray) -> np.ndarray:
    out, last = [], -10**9
    for i in np.flatnonzero(flag):
        if i - last >= GAP:
            out.append(i)
            last = i
    return np.array(out, int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.data) / "futures"
    d5 = load5(base / "klines_5m.csv")
    d = bars(d5, 0)
    t, o, h, l, c = d["t"], d["o"], d["h"], d["l"], d["c"]
    n = len(t)
    T = t + BAR  # 마감 시각
    vol, tkb = taker15(base / "klines_5m.csv", t)
    # 라벨
    y = np.zeros(n, bool)
    ys = np.zeros(n, bool)
    from numpy.lib.stride_tricks import sliding_window_view as sw
    lo = sw(np.r_[l[1:], np.full(HZ, np.nan)], HZ).min(axis=1)[:n]
    hi = sw(np.r_[h[1:], np.full(HZ, np.nan)], HZ).max(axis=1)[:n]
    valid = np.arange(n) < n - HZ
    y[valid] = lo[valid] <= c[valid] * (1 - DROP)
    ys[valid] = hi[valid] >= c[valid] * (1 + DROP)
    # 특징
    F: dict[str, np.ndarray] = {}
    s15, c15, r15 = stoch(h, l, c, 5), cci(h, l, c, 20), rsi(c, 14)
    F["s15"], F["c15"], F["r15"] = s15, c15, r15
    F["s15_d"], F["c15_d"], F["r15_d"] = np.r_[np.nan, np.diff(s15)], np.r_[np.nan, np.diff(c15)], np.r_[np.nan, np.diff(r15)]
    hf = {nm: htf(d5, t, k) for nm, k in (("1h", 12), ("4h", 48), ("1d", 288))}
    for nm, x in hf.items():
        for kk in ("s", "c", "r", "s_d", "c_d", "r_d"):
            F[f"{kk}{nm}"] = x[kk]
    x4 = hf["4h"]
    R4 = (np.fmax(np.fmax(x4["s_l1"], x4["s_l2"]), x4["s_l3"]) > 80) & (x4["s"] < x4["s_l1"]) & \
        (np.fmax(np.fmax(x4["c_l1"], x4["c_l2"]), x4["c_l3"]) > 100) & (x4["c"] < x4["c_l1"])
    R4m = (np.fmin(np.fmin(x4["s_l1"], x4["s_l2"]), x4["s_l3"]) < 20) & (x4["s"] > x4["s_l1"]) & \
        (np.fmin(np.fmin(x4["c_l1"], x4["c_l2"]), x4["c_l3"]) < -100) & (x4["c"] > x4["c_l1"])
    vm, vsd = roll_prev(vol, 96)
    volz = (vol - vm) / np.where(vsd > 0, vsd, np.nan)
    F["volz"] = volz
    F["volz16"] = np.convolve(np.nan_to_num(volz), np.ones(16) / 16, mode="full")[:n]
    ret24 = np.r_[np.full(96, np.nan), c[96:] / c[:-96] - 1]
    F["ret24"] = ret24
    r15b = c / o - 1
    mk, moi, mls = load_metrics(base / "metrics.csv")
    oi_now = asof(mk, moi, T)
    oi_1h = asof(mk, moi, T - H)
    oi_4h = asof(mk, moi, T - 4 * H)
    ls_now = asof(mk, mls, T)
    ls_24 = asof(mk, mls, T - DAY)
    F["oi1h"] = np.log(oi_now / oi_1h)
    F["oi4h"] = np.log(oi_now / oi_4h)
    F["ls"] = ls_now
    F["ls24"] = ls_now - ls_24
    fu = load_funding(base / "funding.csv")
    F["fund"] = asof(fu[0], fu[1], T, stale=9 * H)
    v4 = np.convolve(vol, np.ones(4), mode="full")[:n]
    t4 = np.convolve(np.nan_to_num(tkb, nan=np.nan), np.ones(4), mode="full")[:n]
    F["tk1h"] = t4 / np.where(v4 > 0, v4, np.nan)
    lr = np.r_[np.nan, np.diff(np.log(c))]
    _, rv = roll_prev(np.nan_to_num(lr), 96)
    F["rv"] = rv
    F["rng48"] = (sw(np.r_[np.full(47, np.nan), h], 48).max(axis=1) - sw(np.r_[np.full(47, np.nan), l], 48).min(axis=1)) \
        / np.where(hf["1d"]["atr"] > 0, hf["1d"]["atr"], np.nan)
    piv = zigzag(h, l, 0.02)
    lh = np.zeros(n)
    hl = np.zeros(n)
    highs = [(p[3], p[1]) for p in piv if p[2] == 1]
    lows = [(p[3], p[1]) for p in piv if p[2] == -1]
    for arr, outv, sgn in ((highs, lh, 1), (lows, hl, -1)):
        for k in range(1, len(arr)):
            ci = arr[k][0]
            end = arr[k + 1][0] if k + 1 < len(arr) else n
            outv[ci:end] = 1.0 if (arr[k][1] < arr[k - 1][1]) == (sgn == 1) else 0.0
    F["lower_high"] = lh
    F["higher_low"] = hl
    F["hour"] = ((t // H) % 24).astype(float)
    reg, _ = R.regime(d5, d, 0)
    F["dir"] = reg.astype(float)
    # 스퀴즈 급등 P와 거울 P'
    oi_bar = np.log(asof(mk, moi, T) / asof(mk, moi, T - H))
    pump = (volz >= 4) & (r15b > 0) & (ret24 >= 0.025) & (oi_bar <= 0)
    dump = (volz >= 4) & (r15b < 0) & (ret24 <= -0.025) & (oi_bar <= 0)
    P = any_back(np.nan_to_num(pump).astype(bool), 96)
    Pm = any_back(np.nan_to_num(dump).astype(bool), 96)
    F["pump24"] = P.astype(float)
    F["dump24"] = Pm.astype(float)
    F["rollover4h"] = R4.astype(float)
    F["rollup4h"] = R4m.astype(float)
    with np.errstate(invalid="ignore"):
        H1 = P & R4
        H2 = H1 & (reg <= 0)
        H3 = H2 & (F["ls24"] > 0) & (F["tk1h"] < 0.45)
        H1m = Pm & R4m
        H2m = H1m & (reg >= 0)
        H3m = H2m & (F["ls24"] < 0) & (F["tk1h"] > 0.55)
    start = int(np.searchsorted(t, ts(2020, 3)))  # 상위 봉 준비 구간
    disc = (t >= t[start]) & (t < SPLIT) & valid
    hold = (t >= SPLIT) & valid
    rep: dict = {"bars": int(n), "last": datetime.fromtimestamp(t[-1] / 1000, timezone.utc).isoformat()}
    onsets = {}
    for nm, yy in (("crash", y), ("spike", ys)):
        yi = np.flatnonzero(yy)
        onsets[nm] = yi[np.r_[True, np.diff(yi) > GAP]] if len(yi) else yi
    def evalflag(flag, yy, nm, period):
        m = disc if period == "disc" else hold
        fl = flag & m
        fires = declutter(fl)
        k = int(yy[fires].sum()) if len(fires) else 0
        p, lo_, hi_ = wilson(k, len(fires))
        base_ = float(yy[m].mean())
        ons = onsets[nm][m[onsets[nm]]]
        caught = 0
        if len(ons):
            raw = np.flatnonzero(flag)
            for e in ons:
                j = np.searchsorted(raw, e - HZ)
                if j < len(raw) and raw[j] <= e:
                    caught += 1
        return {"fires": int(len(fires)), "hits": k, "prec": round(p * 100, 2) if len(fires) else None,
                "ci": [round(lo_ * 100, 2), round(hi_ * 100, 2)] if len(fires) else None,
                "base": round(base_ * 100, 2), "lift": round(p / base_, 2) if len(fires) and base_ > 0 else None,
                "lift_lo": round(lo_ / base_, 2) if len(fires) and base_ > 0 else None,
                "events": int(len(ons)), "recall": round(caught / len(ons) * 100, 1) if len(ons) else None,
                "fire_idx": fires.tolist()}
    flags = {"H1": (H1, y, "crash", H1m), "H2": (H2, y, "crash", H2m), "H3": (H3, y, "crash", H3m)}
    rep["flags"] = {}
    for nm, (fl, yy, ev, mirror) in flags.items():
        r_ = {"disc": evalflag(fl, yy, ev, "disc"), "hold": evalflag(fl, yy, ev, "hold"),
              "mirror_disc": evalflag(mirror, ys, "spike", "disc"), "mirror_hold": evalflag(mirror, ys, "spike", "hold")}
        hh = r_["hold"]
        r_["pass"] = bool(hh["fires"] >= 15 and (hh["lift"] or 0) >= 2 and (hh["lift_lo"] or 0) > 1
                          and (r_["mirror_hold"]["lift"] or 0) > 1)
        rep["flags"][nm] = r_
    # 단변량(기술용)
    uni = []
    for nm, x in F.items():
        xd = x[disc & np.isfinite(x)]
        if len(np.unique(xd)) < 3:
            continue
        q10, q90 = np.quantile(xd, [0.1, 0.9])
        for side, fl in (("top10", x >= q90), ("bot10", x <= q10)):
            fl = np.nan_to_num(fl).astype(bool) & np.isfinite(x)
            e1 = evalflag(fl, y, "crash", "disc")
            e2 = evalflag(fl, y, "crash", "hold")
            uni.append({"feat": nm, "side": side, "disc_lift": e1["lift"], "disc_fires": e1["fires"],
                        "hold_lift": e2["lift"], "hold_fires": e2["fires"]})
    for nm in ("pump24", "rollover4h", "lower_high", "dump24", "rollup4h"):
        fl = F[nm] > 0.5
        e1, e2 = evalflag(fl, y, "crash", "disc"), evalflag(fl, y, "crash", "hold")
        uni.append({"feat": nm, "side": "flag", "disc_lift": e1["lift"], "disc_fires": e1["fires"],
                    "hold_lift": e2["lift"], "hold_fires": e2["fires"]})
    rep["univariate"] = sorted(uni, key=lambda z: -(z["disc_lift"] or 0))
    # ML
    import xgboost as xgb
    names = list(F)
    X = np.column_stack([F[k] for k in names]).astype(np.float32)
    X[~np.isfinite(X)] = np.nan  # OI 0 등으로 생긴 inf는 결측으로
    score = np.full(n, np.nan)
    for Y in range(2021, 2027):
        tr = (t >= t[start]) & (t < ts(Y) - 32 * BAR) & valid
        te = (t >= ts(Y)) & (t < ts(Y + 1)) & valid
        if te.sum() == 0:
            continue
        idx = np.flatnonzero(tr)[::2]
        pos = y[idx].mean()
        m = xgb.XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, n_jobs=2, scale_pos_weight=(1 - pos) / max(pos, 1e-6),
                              tree_method="hist", random_state=Y)
        m.fit(X[idx], y[idx])
        score[te] = m.predict_proba(X[te])[:, 1]
        # 미래 구간(마지막 봉까지) 점수도 남긴다(사례 확인용)
        if Y == 2026:
            tail = (t >= ts(Y))
            score[tail] = m.predict_proba(X[tail])[:, 1]
    def auc(yy, p):
        o_ = np.argsort(p)
        rk = np.empty(len(p))
        rk[o_] = np.arange(1, len(p) + 1)
        npos = yy.sum()
        nneg = len(yy) - npos
        return float((rk[yy].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else np.nan
    ml = {}
    for Y in range(2022, 2027):
        te = (t >= ts(Y)) & (t < ts(Y + 1)) & valid & np.isfinite(score)
        prv = (t >= ts(Y - 1)) & (t < ts(Y)) & np.isfinite(score)
        if te.sum() == 0 or prv.sum() == 0:
            continue
        r_ = {"auc": round(auc(y[te], score[te]), 3), "base": round(float(y[te].mean()) * 100, 2)}
        for q in (0.99, 0.95):
            th = np.quantile(score[prv], q)
            fl = np.zeros(n, bool)
            fl[te] = score[te] >= th
            fires = declutter(fl)
            k = int(y[fires].sum())
            p, lo_, hi_ = wilson(k, len(fires))
            r_[f"top{int(round((1 - q) * 100))}"] = {"fires": int(len(fires)), "prec": round(p * 100, 2) if len(fires) else None,
                                                    "lift": round(p / y[te].mean(), 2) if len(fires) else None,
                                                    "lift_lo": round(lo_ / y[te].mean(), 2) if len(fires) else None,
                                                    "bar_prec": round(float(y[fl].mean()) * 100, 2) if fl.any() else None}
        ml[str(Y)] = r_
    ho = hold & np.isfinite(score)
    ml["hold_auc"] = round(auc(y[ho], score[ho]), 3)
    # 확인 구간 상위 1%(전년 문턱) 통합
    fl = np.zeros(n, bool)
    for Y in range(2024, 2027):
        te = (t >= ts(Y)) & (t < ts(Y + 1)) & valid & np.isfinite(score)
        prv = (t >= ts(Y - 1)) & (t < ts(Y)) & np.isfinite(score)
        if te.sum() and prv.sum():
            fl[te] = score[te] >= np.quantile(score[prv], 0.99)
    fires = declutter(fl)
    k = int(y[fires].sum())
    p, lo_, hi_ = wilson(k, len(fires))
    bh = float(y[hold].mean())
    ml["hold_top1"] = {"fires": int(len(fires)), "prec": round(p * 100, 2), "lift": round(p / bh, 2),
                       "lift_lo": round(lo_ / bh, 2), "base": round(bh * 100, 2)}
    ml["pass"] = bool(ml["hold_top1"]["lift"] >= 2 and ml["hold_top1"]["lift_lo"] > 1)
    ml["fire_idx_hold_top1"] = fires.tolist()
    rep["ml"] = ml
    imp = sorted(zip(names, m.feature_importances_.tolist()), key=lambda z: -z[1])[:12]
    rep["ml_importance_2026"] = [[k_, round(v_, 4)] for k_, v_ in imp]
    # 매매 환산(통과한 것만)
    f5 = d5
    def short_trade(i, side):
        e5 = int(np.searchsorted(f5["t"], t[i] + BAR + M5))  # 다음 15분봉 시작 5분 뒤
        if e5 >= len(f5["t"]):
            return None
        px = f5["o"][e5]
        stop = px * (1 + 0.015) if side < 0 else px * (1 - 0.015)
        tgt = px * (1 - 0.03) if side < 0 else px * (1 + 0.03)
        end = e5 + HZ * 3
        for j in range(e5, min(end, len(f5["t"]))):
            hh, ll, oo = f5["h"][j], f5["l"][j], f5["o"][j]
            if side < 0:
                if hh >= stop:
                    fill = max(oo, stop) if j > e5 else stop
                    return (px / fill - 1) - 2 * COST
                if ll <= tgt:
                    return (px / tgt - 1) - 2 * COST
            else:
                if ll <= stop:
                    fill = min(oo, stop) if j > e5 else stop
                    return (fill / px - 1) - 2 * COST
                if hh >= tgt:
                    return (tgt / px - 1) - 2 * COST
        j = min(end, len(f5["t"]) - 1)
        g = (px / f5["o"][j] - 1) if side < 0 else (f5["o"][j] / px - 1)
        return g - 2 * COST
    trades = {}
    for nm, r_ in rep["flags"].items():
        if not r_["pass"]:
            continue
        for key, side in (("hold", -1), ("disc", -1)):
            g = [short_trade(i, side) for i in r_[key]["fire_idx"]]
            g = [x for x in g if x is not None]
            trades[f"{nm}|{key}"] = {"n": len(g), "mean_pct": round(float(np.mean(g)) * 100, 3) if g else None,
                                     "win": round(float(np.mean(np.array(g) > 0)) * 100, 1) if g else None}
    if ml["pass"]:
        g = [short_trade(i, -1) for i in ml["fire_idx_hold_top1"]]
        g = [x for x in g if x is not None]
        trades["ML|hold"] = {"n": len(g), "mean_pct": round(float(np.mean(g)) * 100, 3), "win": round(float(np.mean(np.array(g) > 0)) * 100, 1)}
    rep["trades"] = trades
    # 사용자 사례
    case = {}
    for tt in np.arange(ts(2026, 11, 5) + 10 * H, ts(2026, 11, 5) + 14 * H, BAR):
        i = int(np.searchsorted(t, tt))
        if i < n and t[i] == tt:
            case[datetime.fromtimestamp(tt / 1000, timezone.utc).strftime("%m-%d %H:%M")] = {
                "H1": bool(H1[i]), "H2": bool(H2[i]), "H3": bool(H3[i]), "P": bool(P[i]), "R4": bool(R4[i]),
                "dir": int(reg[i]), "ls24": None if not np.isfinite(F["ls24"][i]) else round(float(F["ls24"][i]), 3),
                "tk1h": None if not np.isfinite(F["tk1h"][i]) else round(float(F["tk1h"][i]), 2),
                "ml": None if not np.isfinite(score[i]) else round(float(score[i]), 3), "y": bool(y[i])}
    rep["case"] = case
    np.savez_compressed(out / "series.npz", t=t, c=c, h=h, l=l, o=o, y=y, score=score, H1=H1, H2=H2, H3=H3, P=P, R4=R4)
    for r_ in rep["flags"].values():
        for k_ in ("disc", "hold", "mirror_disc", "mirror_hold"):
            r_[k_]["fire_idx"] = r_[k_]["fire_idx"][:500]
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
