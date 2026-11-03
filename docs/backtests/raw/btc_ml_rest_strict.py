"""V1 걸어두는 지정가, 진입을 더 까다롭게. 문서: 2026-11-04-btc-ml-rest-strict-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_rest_strict.py --pred <ml15/pred.npz>
"""

from __future__ import annotations

import argparse

import numpy as np
import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
from btc_futures_compare import load_funding
from btc_ml_15m import BE
from btc_ml_edge_cost import mean_score, rest_sim, summ_net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    z = np.load(a.pred)
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    t, probs = z["t"], z["probs"]
    n = len(t)
    ft, fr = load_funding()
    sc = mean_score(probs)
    with np.errstate(invalid="ignore"):
        upv, dnv = (probs > BE).sum(0), (probs < 1 - BE).sum(0)
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))
    for name, q, k in (("S1 q95+4of5", 0.95, 4), ("S2 q97+4of5", 0.97, 4), ("S3 q95+5of5", 0.95, 5)):
        hi, lo = F.thresholds(sc, t, q)
        with np.errstate(invalid="ignore"):
            li = np.isfinite(sc) & (sc >= hi) & (upv >= k)
            si = np.isfinite(sc) & (sc <= lo) & (dnv >= k)
        for per, (a0, a1) in (("2023", (i23, i24)), ("2024~", (i24, n))):
            yrs = (t[a1 - 1] - t[a0]) / F.YEAR
            for lab, sides in (("both", (1, -1)), ("long", (1,)), ("short", (-1,))):
                r = summ_net(rest_sim(P, t, li, si, a0, a1, sides), ft, fr, yrs)
                print(name, per, lab, r, flush=True)


if __name__ == "__main__":
    main()
