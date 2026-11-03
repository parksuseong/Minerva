"""ML 장벽 매매: (A) 선물 시장 정보 추가, (B) 걸어두는 지정가. 등록: 2026-11-04-btc-ml-edge-cost-hypotheses.md

실행: PYTHONPATH=<numpy·sklearn·xgboost·torch 경로>:src:docs/backtests/raw \
      python docs/backtests/raw/btc_ml_edge_cost.py --base <ml15/pred.npz> --out <폴더> [--stage b|a|all]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import btc_ml_15m as M
import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
from btc_fractal import load5, scr3
from btc_futures_compare import load_funding
from btc_ml_barrier import barrier_labels
from btc_scr_litfilter import asof, load_fut

TP = SL = 0.015
OFF = 0.001
REST = 2
THRU = 0.0002
MAKER, TAKER, SLIP = 0.0002, 0.0005, 0.0002


def rest_sim(P, t, li, si, i0, i1, sides, k=1.0):
    o, h, l = P["o"], P["h"], P["l"]  # noqa: E741
    c = P["c"]
    mk, tk, sp = MAKER * k, TAKER * k, SLIP * k
    out, i = [], i0
    while i < i1 - 2:
        s = 1 if (1 in sides and li[i]) else (-1 if (-1 in sides and si[i]) else 0)
        if not s:
            i += 1
            continue
        lim = c[i] * (1 - OFF * s)
        j = None
        for b in range(i + 1, min(i + 1 + REST, len(o))):
            if (l[b] <= lim * (1 - THRU)) if s > 0 else (h[b] >= lim * (1 + THRU)):
                j = b
                break
        if j is None:
            i += 1
            continue
        px = lim
        up, dn = px * (1 + TP), px * (1 - SL)
        res, b = None, j
        while b < len(o):
            first = b == j
            if s > 0:
                if l[b] <= dn:
                    res = ((min(o[b], dn) if not first else dn) * (1 - sp) / px - 1 - mk - tk, 0)
                elif not first and h[b] >= up * (1 + THRU):
                    res = (TP - 2 * mk, 1)
            else:
                if h[b] >= up:
                    res = (-((max(o[b], up) if not first else up) * (1 + sp) / px - 1) - mk - tk, 0)
                elif not first and l[b] <= dn * (1 - THRU):
                    res = (SL - 2 * mk, 1)
            if res:
                break
            b += 1
        if res is None:
            break
        out.append((float(t[j]), res[0], s, float(t[b]), res[1]))
        i = b
    return out


def summ_net(tr, ft, fr, years):
    """rest_sim은 비용을 이미 뺀 1배 손익을 낸다."""
    if not tr:
        return {"n": 0, "x3": -100.0, "ev3": -1.0, "per_year": 0}
    cf = np.r_[0.0, np.cumsum(fr)]
    g = np.array([x[1] for x in tr])
    side = np.array([x[2] for x in tr])
    a = np.searchsorted(ft, [x[0] for x in tr], side="right")
    b = np.searchsorted(ft, [x[3] for x in tr], side="right")
    n3 = 3 * (g - side * (cf[b] - cf[a]))
    eq = np.cumprod(1 + np.maximum(n3, -1))
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    return {"n": len(tr), "per_year": round(len(tr) / years, 1), "long": int((side > 0).sum()),
            "win": round(float(np.mean([x[4] for x in tr]) * 100), 1),
            "ev3": round(float(n3.mean() * 100), 3), "x3": round(float((eq[-1] - 1) * 100), 1),
            "x3_mdd": round(float((eq / peak[1:] - 1).min() * 100), 1)}


def evaluate(tag, z, score, mode, ft, fr):
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    t = z["t"]
    n = len(t)
    i24 = int(np.searchsorted(t, E.ts(2024)))
    yrs = (t[-1] - t[i24]) / F.YEAR
    F.TP = F.SL = TP

    def run(sc, q=0.90, sides=(1, -1), delay=0, k=1.0):
        hi, lo = F.thresholds(sc, t, q)
        with np.errstate(invalid="ignore"):
            li, si = np.isfinite(sc) & (sc >= hi), np.isfinite(sc) & (sc <= lo)
        if delay:
            li, si = np.r_[False, li[:-1]], np.r_[False, si[:-1]]
        if mode == "rest":
            return summ_net(rest_sim(P, t, li, si, i24, n, sides, k), ft, fr, yrs)
        F.COST = (TAKER + SLIP) * k
        r = F.summ(F.simulate(P, t, li, si, i24, n, sides), ft, fr, yrs)
        F.COST = 0.0005
        return r

    base = run(score)
    lo_, sh_ = run(score, sides=(1,)), run(score, sides=(-1,))
    dl, fee = run(score, delay=1), run(score, k=1.5)
    q85, q95 = run(score, q=0.85), run(score, q=0.95)
    rng = np.random.default_rng(151)
    seg = score[i24:].copy()
    rand = []
    for _ in range(200):
        kk = int(rng.integers(96 * 30, len(seg) - 96 * 30))
        s2 = score.copy()
        s2[i24:] = np.roll(seg, kk)
        rand.append(run(s2)["x3"])
    p = float((np.array(rand) >= base["x3"]).mean())
    crit = [base["x3"] > 0 and base["ev3"] > 0, base["per_year"] >= 180, p < 0.01,
            lo_["ev3"] > 0 and sh_["ev3"] > 0, dl["x3"] > 0, fee["x3"] > 0,
            q85["x3"] > 0 and q95["x3"] > 0]
    print(tag, "base", base, flush=True)
    for k_, v in (("long", lo_), ("short", sh_), ("delay1", dl), ("fee1.5", fee), ("q85", q85),
                  ("q95", q95)):
        print(tag, k_, v, flush=True)
    print(tag, "p", p, "rand median", float(np.median(rand)), "crit", crit, flush=True)


def mean_score(probs):
    known = np.isfinite(probs).all(0)
    return np.where(known, np.nanmean(np.where(known, probs, 0.5), 0), np.nan)


def fut_features(t15: np.ndarray) -> np.ndarray:
    fut = load_fut(Path("~/.minerva/data/btc").expanduser())
    tc = t15 + 900_000
    oi = asof(tc, fut["t"], fut["oi"])
    cols = []
    with np.errstate(all="ignore"):
        for h in (1, 4, 24):
            prev = asof(tc - h * 3_600_000, fut["t"], fut["oi"])
            cols.append(np.log(oi / prev))
        cols.append(asof(tc, fut["t"], fut["top"]))
        cols.append(asof(tc, fut["t"], fut["glob"]))
        cols.append(np.log(asof(tc, fut["t"], fut["tk"])))
        cols.append(asof(tc, fut["ft"], fut["fr"]) * 1000)
        fr = fut["fr"]
        pct = np.array([(fr[max(0, i - 270) : i + 1] < fr[i]).mean() for i in range(len(fr))])
        cols.append(asof(tc, fut["ft"], pct))
    X = np.stack(cols, 1)
    has = np.isfinite(X).all(1).astype(float)
    X = np.where(np.isfinite(X), X, 0.0)
    return np.c_[X, has].astype(np.float32)


def predict_a(out: Path) -> None:
    B = E.build(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), M.FACTOR)
    d = B["d"]
    t = d["t"]
    X = np.c_[B["X"], fut_features(t)]
    seq = B["seq"]
    n = len(t)
    valid = np.isfinite(X).all(1) & np.isfinite(seq).all(1)
    seq_ok = np.convolve(valid.astype(int), np.ones(E.SEQ, int), "full")[:n] == E.SEQ
    y = barrier_labels(d, TP, SL)
    on_l, on_s, *_ = scr3(d)
    probs = np.full((len(E.MODELS), n), np.nan, np.float32)
    for Y in E.YEARS:
        te = np.flatnonzero((t >= E.ts(Y)) & (t < E.ts(Y + 1)) & seq_ok)
        if not len(te):
            continue
        base = (t >= E.ts(2018)) & (t < E.ts(Y) - M.EMBARGO * M.BAR) & seq_ok
        i0 = np.flatnonzero(base)
        mu, sd = X[i0].mean(0), X[i0].std(0) + 1e-6
        Xn = ((X - mu) / sd).astype(np.float32)
        Xn[~np.isfinite(Xn)] = 0
        smu, ssd = seq[i0].mean(0), seq[i0].std(0) + 1e-6
        Sn = ((seq - smu) / ssd).astype(np.float32)
        Sn[~np.isfinite(Sn)] = 0
        tr = np.flatnonzero(base & (y >= 0))[::3]
        for mi, name in enumerate(E.MODELS):
            t0 = time.time()
            if name in ("rf", "mlp", "xgb"):
                p = E.fit_predict_tab(name, Xn[tr], y[tr], Xn[te], seed=Y, ncls=2)
            else:
                p = E.fit_predict_seq(name, Sn, Xn, y, tr, te, seed=Y, ncls=2)
            probs[mi, te] = p[:, 1]
            print(Y, name, "sec", round(time.time() - t0, 1), flush=True)
    np.savez_compressed(out / "pred.npz", probs=probs, y=y, t=t, o=d["o"], h=d["h"], l=d["l"],
                        c=d["c"], on_l=on_l, on_s=on_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ft, fr = load_funding()
    z0 = np.load(a.base)
    s0 = mean_score(z0["probs"])
    if a.stage in ("b", "all"):
        evaluate("B-rest(base model)", z0, s0, "rest", ft, fr)
    if a.stage in ("a", "all"):
        if not (out / "pred.npz").exists():
            predict_a(out)
        z1 = np.load(out / "pred.npz")
        s1 = mean_score(z1["probs"])
        y, t = z1["y"], z1["t"]
        m = np.isfinite(s1) & np.isfinite(s0) & (y >= 0) & (t >= E.ts(2024))
        for nm, sc in (("base", s0), ("A", s1)):
            q = np.quantile(sc[m], 0.9)
            print("top10% hit 2024~", nm, round(float(y[m][sc[m] >= q].mean() * 100), 1),
                  "all", round(float(y[m].mean() * 100), 1), flush=True)
        evaluate("A-taker", z1, s1, "taker", ft, fr)
        evaluate("A-rest", z1, s1, "rest", ft, fr)


if __name__ == "__main__":
    main()
