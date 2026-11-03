"""15분봉 ML 확률 상·하위 봉에서 ±1.5% 목표 매매. 등록: 2026-11-04-btc-ml-barrier-freq-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_barrier_freq.py --pred <ml15/pred.npz>
"""

from __future__ import annotations

import argparse
import itertools
from datetime import datetime, timezone

import numpy as np
import btc_ml_ensemble as E
from btc_fractal import recent
from btc_futures_compare import load_funding

TP = SL = 0.015
COST = 0.0005
BE = (SL + 2 * COST) / (TP + SL)
QS = (0.90, 0.95)
SCRW = (0, 16)
YEAR = 365.25 * 86_400_000


def thresholds(pm, t, q):
    yr = np.array([datetime.fromtimestamp(x / 1000, timezone.utc).year for x in t])
    hi, lo = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for y in range(2023, 2027):
        prev = np.isfinite(pm) & (yr == y - 1)
        if not prev.any():
            continue
        m = yr == y
        hi[m] = np.quantile(pm[prev], q)
        lo[m] = np.quantile(pm[prev], 1 - q)
    return hi, lo


def simulate(P, t, long_in, short_in, i0, i1, sides):
    o, h, l = P["o"], P["h"], P["l"]  # noqa: E741
    out, i = [], max(i0, 0)
    while i < i1 - 1:
        s = 1 if (1 in sides and long_in[i]) else (-1 if (-1 in sides and short_in[i]) else 0)
        if not s:
            i += 1
            continue
        px = o[i + 1]
        up, dn = px * (1 + TP), px * (1 - SL)
        j, res = i + 1, None
        while j < len(o):
            if s > 0:
                if l[j] <= dn:
                    res = (min(o[j], dn) / px - 1, 0)
                elif h[j] >= up:
                    res = (TP if o[j] <= up else o[j] / px - 1, 1)
            else:
                if h[j] >= up:
                    res = (-(max(o[j], up) / px - 1), 0)
                elif l[j] <= dn:
                    res = (SL if o[j] >= dn else -(o[j] / px - 1), 1)
            if res:
                break
            j += 1
        if res is None:
            break
        out.append((float(t[i + 1]), res[0], s, float(t[j]), res[1]))
        i = j
    return out


def summ(tr, ft, fr, years):
    if not tr:
        return {"n": 0, "x3": -100.0}
    cf = np.r_[0.0, np.cumsum(fr)]
    g = np.array([x[1] for x in tr])
    side = np.array([x[2] for x in tr])
    a = np.searchsorted(ft, [x[0] for x in tr], side="right")
    b = np.searchsorted(ft, [x[3] for x in tr], side="right")
    fund = side * (cf[b] - cf[a])
    n1 = g - 2 * COST - fund
    n3 = 3 * g - 6 * COST - 3 * fund
    out = {"n": len(tr), "per_year": round(len(tr) / years, 1), "long": int((side > 0).sum()),
           "win": round(float(np.mean([x[4] for x in tr]) * 100), 1),
           "ev3": round(float(n3.mean() * 100), 3)}
    for k, net in (("x1", n1), ("x3", n3)):
        eq = np.cumprod(1 + np.maximum(net, -1))
        peak = np.maximum.accumulate(np.r_[1.0, eq])
        out[k] = round(float((eq[-1] - 1) * 100), 1)
        out[k + "_mdd"] = round(float((eq / peak[1:] - 1).min() * 100), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    z = np.load(a.pred)
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    t, probs = z["t"], z["probs"]
    known = np.isfinite(probs).all(0)
    pm = np.where(known, np.nanmean(np.where(known, probs, 0.5), 0), np.nan)
    ft, fr = load_funding()
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))
    n = len(t)
    yrs = {"sel": (t[i24] - t[i23]) / YEAR, "test": (t[-1] - t[i24]) / YEAR}

    def rules(pmv, q, w):
        hi, lo = thresholds(pmv, t, q)
        with np.errstate(invalid="ignore"):
            li, si = known & (pmv >= hi), known & (pmv <= lo)
        if w:
            li, si = li & recent(z["on_l"], w), si & recent(z["on_s"], w)
        return li, si

    rows = {}
    for q, w in itertools.product(QS, SCRW):
        li, si = rules(pm, q, w)
        rows[(q, w)] = {
            p: summ(simulate(P, t, li, si, a0, a1, (1, -1)), ft, fr, yrs[p])
            for p, (a0, a1) in (("sel", (i23, i24)), ("test", (i24, n)))
        }
        print(f"q{q} W{w}", rows[(q, w)], flush=True)
    best = max(rows, key=lambda k: rows[k]["sel"]["x3"])
    q, w = best
    test = rows[best]["test"]
    li, si = rules(pm, q, w)
    lo_only = summ(simulate(P, t, li, si, i24, n, (1,)), ft, fr, yrs["test"])
    sh_only = summ(simulate(P, t, li, si, i24, n, (-1,)), ft, fr, yrs["test"])
    rng = np.random.default_rng(111)
    seg = pm[i24:].copy()
    rand = []
    for _ in range(200):
        k = int(rng.integers(96 * 30, len(seg) - 96 * 30))
        p2 = pm.copy()
        p2[i24:] = np.roll(seg, k)
        l2, s2 = rules(p2, q, w)
        rand.append(summ(simulate(P, t, l2, s2, i24, n, (1, -1)), ft, fr, yrs["test"])["x3"])
    rand = np.array(rand)
    p = float((rand >= test["x3"]).mean())
    crit = [test["per_year"] >= 180, test["x3"] > 0 and test["ev3"] > 0, test["win"] >= BE * 100 + 3,
            p < 0.05, lo_only.get("ev3", -1) > 0 and sh_only.get("ev3", -1) > 0]
    print("PICK", best, "test", test, "long", lo_only, "short", sh_only, "p", p,
          "rand_median", float(np.median(rand)), "be", round(BE * 100, 1), "crit", crit, flush=True)


if __name__ == "__main__":
    main()
