"""(B) 걸어두는 지정가 + 확인 조건. 등록: 2026-11-04-btc-ml-rest-confirm-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_rest_confirm.py --pred <ml15/pred.npz>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
from btc_futures_compare import load_funding
from btc_ml_15m import BE
from btc_ml_edge_cost import mean_score, rest_sim, summ_net
from btc_trend_detect import detectors
from btc_xgb_trend import load_daily

D = 86_400_000


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
    dd, wk = load_daily(Path("~/.minerva/data/btc/1d.csv").expanduser()), load_daily(
        Path("~/.minerva/data/btc/1w.csv").expanduser())
    t5 = np.asarray(detectors(dd, wk)["T5"], float)
    day = np.searchsorted(dd["t"], t, side="right") - 1
    trend = np.where(day - 1 >= 0, t5[np.maximum(day - 1, 0)], 0.0)
    with np.errstate(invalid="ignore"):
        up4, dn4 = (probs > BE).sum(0) >= 4, (probs < 1 - BE).sum(0) >= 4
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))

    def signals(score, q, v):
        hi, lo = F.thresholds(score, t, q)
        with np.errstate(invalid="ignore"):
            li, si = np.isfinite(score) & (score >= hi), np.isfinite(score) & (score <= lo)
        if v in ("V1", "V3"):
            li, si = li & up4, si & dn4
        if v in ("V2", "V3"):
            li, si = li & (trend > 0), si & (trend < 0)
        return li, si

    def run(v, score=sc, q=0.90, sides=(1, -1), k=1.0, a0=i24, a1=n, delay=0):
        li, si = signals(score, q, v)
        if delay:
            li, si = np.r_[False, li[:-1]], np.r_[False, si[:-1]]
        return summ_net(rest_sim(P, t, li, si, a0, a1, sides, k), ft, fr, (t[a1 - 1] - t[a0]) / F.YEAR)

    sel = {v: run(v, a0=i23, a1=i24) for v in ("V0", "V1", "V2", "V3")}
    for v, r in sel.items():
        print("2023", v, r, "| 2024~", run(v), flush=True)
    best = max(sel, key=lambda v: sel[v]["x3"])
    base = run(best)
    lo_, sh_ = run(best, sides=(1,)), run(best, sides=(-1,))
    fee, q85, q95, dl = run(best, k=1.5), run(best, q=0.85), run(best, q=0.95), run(best, delay=1)
    for k_, r in (("long", lo_), ("short", sh_), ("fee1.5", fee), ("q85", q85), ("q95", q95),
                  ("delay1(report)", dl)):
        print(best, k_, r, flush=True)
    rng = np.random.default_rng(161)
    seg = sc[i24:].copy()
    rand = []
    for _ in range(200):
        kk = int(rng.integers(96 * 30, len(seg) - 96 * 30))
        s2 = sc.copy()
        s2[i24:] = np.roll(seg, kk)
        rand.append(run(best, score=s2)["x3"])
    p = float((np.array(rand) >= base["x3"]).mean())
    crit = [base["x3"] > 0 and base["ev3"] > 0, base["per_year"] >= 180, p < 0.01,
            lo_["ev3"] > 0 and sh_["ev3"] > 0, fee["x3"] > 0, q85["x3"] > 0 and q95["x3"] > 0]
    print("PICK", best, base, "p", p, "rand median", float(np.median(rand)), "crit", crit, flush=True)


if __name__ == "__main__":
    main()
