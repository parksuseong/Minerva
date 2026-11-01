"""BTCUSDT 차트 패턴 학습 앙상블. 사전 등록: docs/backtests/2026-11-01-btc-pattern-hypotheses.md

numpy, scikit-learn 필요. 실행: PYTHONPATH=src:<경로>:docs/backtests/raw python <이 파일> <결과.json>
주문과 연결되지 않는다.
"""

from __future__ import annotations

import json
import pickle
import random
import sys
import time

import btc_regress as br
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from minerva.strategy import btc_bt as bt

TARGETS = (0.01, 0.02, 0.03)
YEARS = range(2020, 2027)
TOP = 0.98
SHIFTS = 300
EPS = 1e-12


def roll_std(x, n):
    a, b = np.cumsum(np.r_[0, x]), np.cumsum(np.r_[0, x**2])
    out = np.full(len(x), np.nan)
    m = (a[n:] - a[:-n]) / n
    v = (b[n:] - b[:-n]) / n - m**2
    out[n - 1 :] = np.sqrt(np.maximum(v, EPS))
    return out


def roll_mean(x, n):
    a = np.cumsum(np.r_[0, x])
    out = np.full(len(x), np.nan)
    out[n - 1 :] = (a[n:] - a[:-n]) / n
    return out


def build(bars):
    o, h, l, c, v = [  # noqa: E741
        np.array([getattr(b, k) for b in bars], float)
        for k in ("open", "high", "low", "close", "volume")
    ]
    r = np.r_[0, np.diff(np.log(c))]
    s96 = roll_std(r, 96)
    cols = [np.r_[np.full(k, np.nan), (r / s96)[:-k]] if k else r / s96 for k in range(32)]
    span = np.maximum(h - l, EPS * c)
    body = (c - o) / span
    up = (h - np.maximum(o, c)) / span
    dn = (np.minimum(o, c) - l) / span
    for s in (body, up, dn):
        cols += [np.r_[np.full(k, np.nan), s[:-k]] if k else s for k in range(8)]
    for n in (8, 32, 96, 480):
        wh = np.lib.stride_tricks.sliding_window_view(h, n).max(1)
        wl = np.lib.stride_tricks.sliding_window_view(l, n).min(1)
        pos = np.full(len(c), np.nan)
        pos[n - 1 :] = (c[n - 1 :] - wl) / np.maximum(wh - wl, EPS * wh)
        cols.append(pos)
    cols.append(roll_mean(v, 4) / np.maximum(roll_mean(v, 96), EPS))
    cols.append(roll_std(r, 16) / s96)
    return np.c_[cols].T, o, h, l


def fit_predict(X, y, tr, te, stride, knn_stride):
    pick = np.nonzero(tr)[0][::stride]
    ys = y[pick].astype(int)
    scaler = StandardScaler().fit(X[pick])
    a = np.clip(scaler.transform(X[pick]), -8, 8)
    b = np.clip(scaler.transform(X[te]), -8, 8)
    g = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=7,
    ).fit(a, ys)
    lg = LogisticRegression(C=0.1, max_iter=300).fit(a, ys)
    pca = PCA(10, random_state=7).fit(a)
    ka = pca.transform(a[:: knn_stride // stride])
    kb = pca.transform(b)
    ky = ys[:: knn_stride // stride].astype(float)
    nn = NearestNeighbors(n_neighbors=200, algorithm="kd_tree").fit(ka)
    pk_te = ky[nn.kneighbors(kb, return_distance=False)].mean(1)
    pa = g.predict_proba(a)[:, 1]
    pl = lg.predict_proba(a)[:, 1]
    sample = np.arange(0, len(a), max(1, len(a) // 30000))
    pk_tr = ky[nn.kneighbors(pca.transform(a[sample]), return_distance=False)].mean(1)
    inner = (pa[sample] + pl[sample] + pk_tr) / 3
    pg_te = g.predict_proba(b)[:, 1]
    pl_te = lg.predict_proba(b)[:, 1]
    return pg_te, pk_te, pl_te, float(np.quantile(inner, TOP))


def run_frame(frame: str, report: dict, path: str) -> None:
    bars = bt.load(frame)
    X, o, h, l = build(bars)  # noqa: E741
    br.MAX_BARS = 500 if frame == "1h" else 2000
    br.STOP = 0.01
    stride, knn_stride = (1, 2) if frame == "1h" else (4, 8)
    ok = np.isfinite(X).all(axis=1)
    days_of = [b.day for b in bars]
    years_of = np.array([d.year for d in days_of])
    bad = np.array([br.BAD[0] <= d <= br.BAD[1] for d in days_of])
    n = len(o)
    s20 = int(np.argmax(years_of == 2020))
    s25 = int(np.argmax(years_of == 2025))
    days = (days_of[-1] - days_of[s20]).days
    days25 = (days_of[-1] - days_of[s25]).days
    print(frame, X.shape, flush=True)
    for tp in TARGETS:
        res = {s: br.outcomes(o, h, l, tp, s) for s in (1, -1)}
        keyed = {(tp, s): res[s] for s in (1, -1)}
        probs = {s: np.zeros(n) for s in (1, -1)}
        cuts, aucs = {}, {}
        for side in (1, -1):
            y = res[side][0].astype(float)
            parts: dict = {"G": [], "K": [], "L": [], "E": [], "y": []}
            for yr in YEARS:
                t0 = time.time()
                z = int(np.argmax(years_of == yr))
                tr = np.zeros(n, bool)
                tr[: z - br.MAX_BARS] = True
                tr &= ok & ~bad & (y >= 0)
                te = np.nonzero((years_of == yr) & ok)[0]
                pg, pk, pl, cut = fit_predict(X, y, tr, te, stride, knn_stride)
                pe = (pg + pk + pl) / 3
                probs[side][te] = pe
                cuts[(side, yr)] = cut
                good = y[te] >= 0
                for k, v in (("G", pg), ("K", pk), ("L", pl), ("E", pe)):
                    parts[k].append(v[good])
                parts["y"].append(y[te][good])
                print(f" {frame} tp{tp} side{side} {yr} {time.time() - t0:.0f}s", flush=True)
            ally = np.concatenate(parts["y"])
            aucs[side] = {
                k: round(float(roc_auc_score(ally, np.concatenate(parts[k]))), 4)
                for k in ("G", "K", "L", "E")
            }
        sig = []
        for i in np.nonzero(ok & (years_of >= YEARS[0]))[0][:-1]:
            yr = int(years_of[i])
            pl, ps = probs[1][i], probs[-1][i]
            hit_l, hit_s = pl >= cuts[(1, yr)], ps >= cuts[(-1, yr)]
            if hit_l and (not hit_s or pl >= ps):
                sig.append((int(i), 1, tp))
            elif hit_s:
                sig.append((int(i), -1, tp))
        trades = br.trade(sig, keyed, br.COST, o)
        got = br.score(trades, years_of, days)
        key = f"{frame}/{tp}"
        got["auc"] = aucs
        if trades:
            slip = [(e, r - 0.0004 - (0.001 if r < 0 else 0), w) for e, r, w in trades]
            rng = random.Random(7)
            m, vals = n - s20, []
            for _ in range(SHIFTS):
                d = rng.randrange(1000, m - 1000)
                moved = sorted(((i - s20 + d) % m + s20, s, t) for i, s, t in sig)
                moved = [x for x in moved if x[0] < n - 2]
                vals.append(br.score(br.trade(moved, keyed, br.COST, o), years_of, days)["final"])
            got.update(
                signals=len(sig),
                cost10=br.score(br.trade(sig, keyed, 0.001, o), years_of, days)["cagr"],
                slip=br.score(slip, years_of, days)["cagr"],
                recent=br.score(
                    br.trade([x for x in sig if x[0] >= s25], keyed, br.COST, o), years_of, days25
                ).get("final"),
                p=(sum(v >= got["final"] for v in vals) + 1) / (SHIFTS + 1),
                longs=sum(s == 1 for _, s, _ in sig),
            )
            pickle.dump({"sig": sig, "tp": tp}, open(f"{path}.{frame}.{tp}.pkl", "wb"))
        report[key] = got
        print(key, json.dumps(got, ensure_ascii=False, default=str), flush=True)


def main() -> None:
    report: dict = {}
    for frame in ("1h", "15m"):
        run_frame(frame, report, sys.argv[1])
        with open(sys.argv[1], "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
