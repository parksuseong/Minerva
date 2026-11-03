"""q0.90 ML 장벽 매매, 지정가 체결 가정. 등록: 2026-11-04-btc-ml-barrier-q90-maker.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_barrier_q90_maker.py --pred <ml15/pred.npz>
"""

from __future__ import annotations

import argparse

import numpy as np
import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
from btc_futures_compare import load_funding

TP = SL = 0.015
SLIP = 0.0002


def simulate(P, t, li, si, i0, i1, sides, maker, taker):
    o, h, l, c = P["o"], P["h"], P["l"], P["c"]  # noqa: E741
    out, i = [], i0
    while i < i1 - 1:
        s = 1 if (1 in sides and li[i]) else (-1 if (-1 in sides and si[i]) else 0)
        if not s:
            i += 1
            continue
        px = c[i]
        j = i + 1
        if not (l[j] <= px if s > 0 else h[j] >= px):
            i += 1
            continue
        up, dn = px * (1 + TP), px * (1 - SL)
        res = None
        while j < len(o):
            first = j == i + 1
            op = px if first else o[j]
            if s > 0:
                if l[j] <= dn:
                    fill = min(op, dn) * (1 - SLIP)
                    res = (fill / px - 1 - maker - taker, 0)
                elif h[j] >= up:
                    res = ((max(op, up) if not first else up) / px - 1 - 2 * maker, 1)
            else:
                if h[j] >= up:
                    fill = max(op, up) * (1 + SLIP)
                    res = (-(fill / px - 1) - maker - taker, 0)
                elif l[j] <= dn:
                    res = (-((min(op, dn) if not first else dn) / px - 1) - 2 * maker, 1)
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
        return {"n": 0, "x3": -100.0, "ev3": -1.0}
    cf = np.r_[0.0, np.cumsum(fr)]
    g = np.array([x[1] for x in tr])  # 비용 뺀 1배 손익
    side = np.array([x[2] for x in tr])
    a = np.searchsorted(ft, [x[0] for x in tr], side="right")
    b = np.searchsorted(ft, [x[3] for x in tr], side="right")
    fund = side * (cf[b] - cf[a])
    n3 = 3 * (g - fund)
    eq = np.cumprod(1 + np.maximum(n3, -1))
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    return {"n": len(tr), "per_year": round(len(tr) / years, 1), "long": int((side > 0).sum()),
            "win": round(float(np.mean([x[4] for x in tr]) * 100), 1),
            "ev3": round(float(n3.mean() * 100), 3), "x3": round(float((eq[-1] - 1) * 100), 1),
            "x3_mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
            "x1": round(float((np.prod(1 + g - fund) - 1) * 100), 1)}


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
    n = len(t)
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))
    yt, ys = (t[-1] - t[i24]) / F.YEAR, (t[i24] - t[i23]) / F.YEAR

    def run(pmv, sides=(1, -1), maker=0.0002, taker=0.0005, a0=i24, a1=n, years=yt):
        hi, lo = F.thresholds(pmv, t, 0.90)
        with np.errstate(invalid="ignore"):
            li, si = known & (pmv >= hi), known & (pmv <= lo)
        return summ(simulate(P, t, li, si, a0, a1, sides, maker, taker), ft, fr, years)

    test = run(pm)
    print("2023", run(pm, a0=i23, a1=i24, years=ys))
    print("2024~", test)
    lo_, sh_ = run(pm, (1,)), run(pm, (-1,))
    print("long", lo_)
    print("short", sh_)
    stress = run(pm, maker=0.0004, taker=0.00075)
    print("fees x1.5", stress)
    for y in (2024, 2025, 2026):
        a0, a1 = int(np.searchsorted(t, E.ts(y))), int(np.searchsorted(t, E.ts(y + 1)))
        print(y, run(pm, a0=a0, a1=a1, years=(t[min(a1, n - 1)] - t[a0]) / F.YEAR))
    rng = np.random.default_rng(131)
    seg = pm[i24:].copy()
    rand = []
    for _ in range(200):
        k = int(rng.integers(96 * 30, len(seg) - 96 * 30))
        p2 = pm.copy()
        p2[i24:] = np.roll(seg, k)
        rand.append(run(p2)["x3"])
    rand = np.array(rand)
    p = float((rand >= test["x3"]).mean())
    crit = [test["x3"] > 0 and test["ev3"] > 0, test["per_year"] >= 180, p < 0.01,
            lo_["ev3"] > 0 and sh_["ev3"] > 0, stress["x3"] > 0]
    print("p", p, "rand median", float(np.median(rand)), "crit", crit)


if __name__ == "__main__":
    main()
