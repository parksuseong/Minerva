"""확인: ML 확률 상·하위 10% 봉 ±1.5% 매매. 등록: 2026-11-04-btc-ml-barrier-q90-check.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_barrier_q90.py --pred <ml15/pred.npz>
"""

from __future__ import annotations

import argparse

import numpy as np
import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
from btc_futures_compare import load_funding


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
    i24 = int(np.searchsorted(t, E.ts(2024)))
    yr_test = (t[-1] - t[i24]) / F.YEAR

    def run(pmv, sides=(1, -1), a0=i24, a1=n, years=yr_test):
        hi, lo = F.thresholds(pmv, t, 0.90)
        with np.errstate(invalid="ignore"):
            li, si = known & (pmv >= hi), known & (pmv <= lo)
        return F.summ(F.simulate(P, t, li, si, a0, a1, sides), ft, fr, years)

    base = run(pm)
    print("test", base, flush=True)
    rng = np.random.default_rng(121)
    seg = pm[i24:].copy()
    rand = []
    for _ in range(1000):
        k = int(rng.integers(96 * 30, len(seg) - 96 * 30))
        p2 = pm.copy()
        p2[i24:] = np.roll(seg, k)
        rand.append(run(p2)["x3"])
    rand = np.array(rand)
    p = float((rand >= base["x3"]).mean())
    lo_, sh_ = run(pm, (1,)), run(pm, (-1,))
    F.COST = 0.001
    hi_cost = run(pm)
    F.COST = 0.0005
    years = {}
    for y in (2024, 2025, 2026):
        a0, a1 = int(np.searchsorted(t, E.ts(y))), int(np.searchsorted(t, E.ts(y + 1)))
        years[y] = run(pm, a0=a0, a1=a1, years=(t[min(a1, n - 1)] - t[a0]) / F.YEAR)
    crit = [p < 0.01, lo_["ev3"] > 0 and sh_["ev3"] > 0, hi_cost["x3"] > 0,
            sum(1 for v in years.values() if v["x3"] > 0) >= 2]
    print("p", p, "rand median", float(np.median(rand)), "p95", float(np.quantile(rand, 0.95)))
    print("long", lo_)
    print("short", sh_)
    print("cost 0.1%/side", hi_cost)
    for y, v in years.items():
        print(y, v)
    print("crit", crit)


if __name__ == "__main__":
    main()
