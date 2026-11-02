"""과거 SCR 3단 매매로 학습한 승리확률 모델. 등록: 2026-11-02-btc-scr-model-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_model.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import btc_book2 as K
import btc_pattern_census as P
import btc_pattern_context as C
from btc_combined_prob import side_code
import btc_swing_patterns as S
from btc_fractal import MIN5, cci, load5, resample, rsi, scr3, sma, stoch
from btc_scr_ma import fit_logit

STOP = 0.01
COST = (0.0005, 0.001)
WIN = 0.001
TRAIN_END = datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
VAL_END = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
BREAK = ("DB", "DBP", "TB", "IHS", "ATRI")


def higher_dir(d5: dict, factor: int, t30: np.ndarray) -> np.ndarray:
    """상위봉(5분봉 factor개 묶음) 마감된 봉의 방향: CCI>0·RSI>50 → +1, 둘 다 아래 → −1."""
    h = resample(d5, factor)
    cc, rr = cci(h["h"], h["l"], h["c"]), rsi(h["c"])
    dr = np.where((cc > 0) & (rr > 50), 1, np.where((cc < 0) & (rr < 50), -1, 0))
    close_t = h["t"] + factor * MIN5
    j = np.searchsorted(close_t, t30 + 1_800_000, side="right") - 1
    return np.where(j >= 0, dr[np.maximum(j, 0)], 0)


def label(d: dict, i: int, side: int, opp: np.ndarray) -> tuple:
    """i 마감 신호, i+1 시가 진입. 손절 1%(갭은 시가), 반대 SCR 원신호 다음 시가. 상한 없음."""
    o, h, l = d["o"], d["h"], d["l"]
    n = len(o)
    e = i + 1
    px = o[e]
    stop = px * (1 - STOP * side)
    for j in range(e, n - 1):
        if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
            fill = min(o[j], stop) if side > 0 else max(o[j], stop)
            return (fill / px - 1) * side, j
        if opp[j]:
            return (o[j + 1] / px - 1) * side, j + 1
    return None


def build(d5: dict, offset: int) -> tuple:
    sh = dict(d5)
    sh["t"] = d5["t"] - offset * 60_000
    d = resample(sh, 6)
    d["t"] = d["t"] + offset * 60_000
    S.CAP = K.CAP = 48
    n = len(d["c"])
    c, h, l, o, v = d["c"], d["h"], d["l"], d["o"], d["v"]
    on_l, on_s, s, cc, r = scr3(d)
    st = C.states(d)
    lr = np.r_[0.0, np.diff(np.log(c))]
    vol = np.full(n, np.nan)
    vol[47:] = np.lib.stride_tricks.sliding_window_view(lr, 48).std(1)
    med = np.full(n, np.nan)
    for i in range(1440, n, 48):
        med[i:i + 48] = np.nanmedian(vol[i - 1440:i])
    h4 = higher_dir(d5, 48, d["t"])
    d1 = higher_dir(d5, 288, d["t"])
    piv = S.zigzag(h, l, 0.02)
    va = np.full(n, np.nan)
    va[20:] = np.lib.stride_tricks.sliding_window_view(v, 20)[:-1].mean(1)
    body = np.abs(c - o) / (c * vol)
    pat = {1: np.zeros(n, bool), -1: np.zeros(n, bool)}
    for th in (0.02, 0.04):
        for side in (1, -1):
            for name, at, g, why, rr in P.rows_for(d, side, th):
                if 1 <= at < n:
                    pat[side][at - 1 if name in BREAK else at] = True
    near = {k: np.convolve(pat[k].astype(int), np.ones(13, int))[:n] > 0 for k in (1, -1)}
    lc = np.log(c)
    X, y, g, meta = [], [], [], []
    piv_lows = [p for p in piv if p[2] == -1]
    piv_highs = [p for p in piv if p[2] == 1]
    for side, on, opp in ((1, on_l, on_s), (-1, on_s, on_l)):
        for i in np.flatnonzero(on):
            if i < 3000 or i >= n - 2:
                continue
            res = label(d, i, side, opp)
            if res is None:
                continue
            gross, x = res
            sc, ma, bb = side_code(int(st[i]), side) if st[i] >= 0 else (2, 1, 1)
            f = []
            f += [(s[i] / 50 - 1) * side, np.clip(cc[i] / 100, -5, 5) * side, (r[i] / 50 - 1) * side]
            w3 = slice(i - 2, i + 1)
            f += [((s[w3].min() if side > 0 else 100 - s[w3].max()) / 50 - 1),
                  np.clip((cc[w3].min() if side > 0 else -cc[w3].max()) / 100, -5, 5),
                  ((r[w3].min() if side > 0 else 100 - r[w3].max()) / 50 - 1)]
            f += [float(sc == k) for k in range(5)] + [float(ma == k) for k in range(4)]
            f += [float(bb == k) for k in range(4)]
            for nn in (20, 60, 120, 240):
                m = sma(c[i - nn - 1:i + 1], nn)[-1]
                f.append(np.log(c[i] / m) / (vol[i] * np.sqrt(nn)) * side)
            for k in (3, 12, 48, 96):
                f.append((lc[i] - lc[i - k]) / (vol[i] * np.sqrt(k)) * side)
            f += [float(h4[i] * side > 0), float(h4[i] * side < 0),
                  float(d1[i] * side > 0), float(d1[i] * side < 0)]
            # 다이버전스(30분봉 RSI, 2% 지그재그 직전 확정 저점/고점 대비)
            pv = [p for p in (piv_lows if side > 0 else piv_highs) if p[3] <= i]
            div = 0.0
            if pv:
                pi, pp = pv[-1][0], pv[-1][1]
                w0 = max(i - 5, 0)
                if side > 0:
                    j = w0 + int(np.argmin(l[w0:i + 1]))
                    div = float(l[j] < pp and r[j] > r[pi])
                else:
                    j = w0 + int(np.argmax(h[w0:i + 1]))
                    div = float(h[j] > pp and r[j] < r[pi])
            f.append(div)
            vr = v[i] / va[i] if va[i] > 0 else 1.0
            f += [np.log(max(vr, 1e-3)), float(vr > 2 and body[i] < 0.5)]
            f += [float(near[side][i]), float(near[-side][i])]
            hour = int(((d["t"][i] // 3_600_000) + 9) % 24)
            f += [float(hour in (22, 23, 0, 1)), float(hour in (9, 10)), float(vol[i] > med[i])]
            f.append(float(side > 0))
            X.append(f)
            y.append(float(gross - 2 * COST[0] > WIN))
            g.append(gross)
            meta.append((int(i), side, float(d["t"][i]), int(x)))
    return d, np.nan_to_num(np.array(X)), np.array(y), np.array(g), meta


FEATS = (["Stoch", "CCI", "RSI", "Stoch 깊이", "CCI 깊이", "RSI 깊이"] +
         [f"SCR:{x}" for x in C.SCR] + [f"이평:{x}" for x in C.MA] + [f"볼린저:{x}" for x in C.BB] +
         ["SMA20 거리", "SMA60 거리", "SMA120 거리", "SMA240 거리", "3봉 흐름", "12봉 흐름",
          "48봉 흐름", "96봉 흐름", "4h 같은방향", "4h 반대", "일봉 같은방향", "일봉 반대",
          "다이버전스", "거래량 배수(log)", "흡수 봉", "같은방향 패턴 6h", "반대 패턴 6h",
          "시간 22~01", "시간 09~10", "변동성 큼", "롱"])


def predict_logit(fit, x):
    w, mu, s = fit
    z = np.c_[np.ones(len(x)), (x - mu) / s]
    return 1 / (1 + np.exp(-np.clip(z @ w, -30, 30)))


def knn(xtr, ytr, xq, k=60):
    mu, s = xtr.mean(0), xtr.std(0) + 1e-9
    a, b = (xtr - mu) / s, (xq - mu) / s
    dd = (b ** 2).sum(1)[:, None] + (a ** 2).sum(1)[None, :] - 2 * b @ a.T
    nn = np.argpartition(dd, k, axis=1)[:, :k]
    return ytr[nn].mean(1)


def auc(y, p):
    o = np.argsort(p)
    rk = np.empty(len(p))
    rk[o] = np.arange(1, len(p) + 1)
    pos = y == 1
    a_, b_ = pos.sum(), (~pos).sum()
    return float((rk[pos].sum() - a_ * (a_ + 1) / 2) / (a_ * b_)) if a_ and b_ else float("nan")


def day_boot(y, p, t, reps=500, seed=1):
    rng = np.random.default_rng(seed)
    day = (t // 86_400_000).astype(np.int64)
    uniq = np.unique(day)
    idx = {u: np.flatnonzero(day == u) for u in uniq}
    out = []
    for _ in range(reps):
        pick = rng.choice(uniq, len(uniq))
        ii = np.concatenate([idx[u] for u in pick])
        out.append(auc(y[ii], p[ii]))
    return [round(float(np.nanpercentile(out, 2.5)), 3), round(float(np.nanpercentile(out, 97.5)), 3)]


def trade(meta, g, p, mask, thr, cost):
    order = np.argsort([m[0] for m in meta])
    busy, out = -1, []
    for k in order:
        if not mask[k] or p[k] < thr:
            continue
        i, side, t, x = meta[k]
        if i <= busy:
            continue
        out.append(k)
        busy = x
    if not out:
        return {"n": 0}, []
    net = g[out] - 2 * cost
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    s_ = np.sort(net)
    return ({"n": len(out), "win": round(float((net > WIN).mean() * 100), 1),
             "ev": round(float(net.mean() * 100), 3), "ret": round(float((eq[-1] - 1) * 100), 1),
             "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
             "drop5": round(float(s_[:-5].mean() * 100), 3) if len(s_) > 5 else None}, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(args.data) / "5m.csv")
    d, X, y, g, meta = build(d5, args.offset)
    t = np.array([m[2] for m in meta])
    tr, va, te = t < TRAIN_END, (t >= TRAIN_END) & (t < VAL_END), t >= VAL_END
    trva = tr | va
    rep: dict = {"offset": args.offset, "n": [int(tr.sum()), int(va.sum()), int(te.sum())],
                 "base_win": [round(float(y[m].mean()) * 100, 1) for m in (tr, va, te)]}
    # 1) 검증 구간으로 고르기: 규제 세기, 모델(logit/knn/평균), 문턱
    best = None
    for lam in (3.0, 30.0, 300.0):
        fit = fit_logit(X[tr], y[tr], lam=lam)
        pl = predict_logit(fit, X[va])
        pk = knn(X[tr], y[tr], X[va])
        for mname, pv in (("logit", pl), ("knn", pk), ("avg", (pl + pk) / 2)):
            a = auc(y[va], pv)
            if best is None or a > best[0]:
                best = (a, lam, mname)
    rep["chosen"] = {"val_auc": round(best[0], 3), "lam": best[1], "model": best[2]}
    lam, mname = best[1], best[2]
    # 2) 학습+검증으로 다시 맞추고 시험
    fit = fit_logit(X[trva], y[trva], lam=lam)
    p_l = predict_logit(fit, X)
    p_k = np.zeros(len(y))
    p_k[te] = knn(X[trva], y[trva], X[te])
    p_k[trva] = knn(X[trva], y[trva], X[trva])
    p = {"logit": p_l, "knn": p_k, "avg": (p_l + p_k) / 2}[mname]
    # 문턱: 검증 구간(학습만으로 맞춘 모델)에서 거래 30회 이상 중 평균 최대
    fit_tr = fit_logit(X[tr], y[tr], lam=lam)
    pv_l = predict_logit(fit_tr, X)
    pv_k = np.zeros(len(y))
    pv_k[va] = knn(X[tr], y[tr], X[va])
    pv = {"logit": pv_l, "knn": pv_k, "avg": (pv_l + pv_k) / 2}[mname]
    thr_best = None
    for q in np.arange(0.20, 0.70, 0.02):
        s_, _ = trade(meta, g, pv, va, q, COST[0])
        if s_["n"] >= 30 and (thr_best is None or s_["ev"] > thr_best[1]):
            thr_best = (round(float(q), 2), s_["ev"], s_["n"])
    rep["threshold"] = thr_best
    rep["auc_test"] = round(auc(y[te], p[te]), 3)
    rep["auc_test_dayboot"] = day_boot(y[te], p[te], t[te])
    rep["calib_test"] = []
    for lo_, hi_ in ((0, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1.01)):
        k = te & (p >= lo_) & (p < hi_)
        rep["calib_test"].append([lo_, hi_, int(k.sum()),
                                  round(float(y[k].mean()) * 100, 1) if k.any() else None,
                                  round(float(p[k].mean()) * 100, 1) if k.any() else None])
    rep["coef"] = sorted([[FEATS[i], round(float(fit[0][1 + i]), 3)] for i in range(len(FEATS))],
                         key=lambda z: -abs(z[1]))[:15]
    base_te, _ = trade(meta, g, np.ones(len(y)), te, 0.0, COST[0])
    rep["scr_only_test"] = base_te
    rep["scr_only_test_c10"] = trade(meta, g, np.ones(len(y)), te, 0.0, COST[1])[0]
    if thr_best:
        rep["model_test"], sel = trade(meta, g, p, te, thr_best[0], COST[0])
        rep["model_test_c10"] = trade(meta, g, p, te, thr_best[0], COST[1])[0]
        rep["test_trades"] = [{"i": meta[k][0], "side": meta[k][1], "t": meta[k][2],
                               "p": round(float(p[k]), 3), "g": round(float(g[k]), 5)} for k in sel]
    rep["all_test_signals"] = [{"i": meta[k][0], "side": meta[k][1], "t": meta[k][2],
                                "p": round(float(p[k]), 3), "win": int(y[k])}
                               for k in np.flatnonzero(te)]
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("test_trades", "all_test_signals")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
