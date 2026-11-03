"""선물·롱숏·3배·펀딩비로 후보 비교. 문서: 2026-11-04-btc-futures-compare.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_futures_compare.py --ml <ml15/pred.npz>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from btc_fractal import load5
from btc_ml_15m_short import signals as ml_signals
from btc_scr_trend import simulate
from btc_scr_wait_fast import on_grid
from btc_trend_detect import detectors, to5, ts
from btc_xgb_trend import load_daily, sma

COST = 0.0005
LIQ = 0.32
STOP = 0.03
FUT = Path("~/.minerva/data/btc/futures").expanduser()
SPOT = Path("~/.minerva/data/btc").expanduser()


def load_funding():
    with (FUT / "funding.csv").open() as f:
        r = list(csv.reader(f))[1:]
    return np.array([float(x[0]) for x in r]), np.array([float(x[2]) for x in r])


def stats(tr, ft, fr) -> dict:
    if not tr:
        return {"n": 0}
    cf = np.r_[0.0, np.cumsum(fr)]
    g = np.array([x[1] for x in tr])
    mae = np.array([x[2] for x in tr])
    side = np.array([x[3] for x in tr])
    a = np.searchsorted(ft, [x[0] for x in tr], side="right")
    b = np.searchsorted(ft, [x[4] for x in tr], side="right")
    fund = side * (cf[b] - cf[a])
    n1 = g - 2 * COST - fund
    n3 = np.where(mae >= LIQ, -1.0, 3 * g - 6 * COST - 3 * fund)
    out = {"n": len(tr), "long": int((side > 0).sum()), "win": round(float((n1 > 0.001).mean() * 100), 1),
           "fund3": round(float((3 * fund).sum() * 100), 1)}
    for k, net in (("x1", n1), ("x3", n3)):
        eq = np.cumprod(1 + np.maximum(net, -1))
        peak = np.maximum.accumulate(np.r_[1.0, eq])
        out[k] = round(float((eq[-1] - 1) * 100), 1)
        out[k + "_mdd"] = round(float((eq / peak[1:] - 1).min() * 100), 1)
    return out


def grid_from_15m(t5, t15, mask):
    g = np.zeros(len(t5), bool)
    j = np.searchsorted(t5, t15[mask] + 900_000)
    g[j[j < len(g)]] = True
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml", required=True)
    a = ap.parse_args()
    d5 = load5(FUT / "klines_5m.csv")
    t = d5["t"]
    n = len(t) - 1
    ft, fr = load_funding()
    dd, wk = load_daily(SPOT / "1d.csv"), load_daily(SPOT / "1w.csv")
    t8 = to5(t, dd["t"], detectors(dd, wk)["T8"])
    ma = sma(dd["c"], 60)
    ma_state = np.where(np.isnan(ma), 0.0, np.where(dd["c"] > ma, 1.0, -1.0))
    ma5 = to5(t, dd["t"], ma_state)
    z0 = np.zeros(len(t), bool)
    s15, s30 = on_grid(d5, 3, 0), on_grid(d5, 6, 0)
    z = np.load(a.ml)
    li, lo, si, so = (grid_from_15m(t, z["t"], m) for m in ml_signals(z["probs"], z["on_l"], z["on_s"]))
    cand = {
        "C0 SCR15 no filter": ((s15[0], s15[1]), s15),
        "C1 T8+SCR15 trend exit": ((s15[0] & (t8 > 0), s15[1] & (t8 < 0)), (t8 >= 0, t8 < 1)),
        "C2 T8+SCR30 trend exit": ((s30[0] & (t8 > 0), s30[1] & (t8 < 0)), (t8 >= 0, t8 < 1)),
        "C3 SCR15+MA60": ((s15[0] & (ma5 > 0), s15[1] & (ma5 < 0)), s15),
        "C4 ML15+SCR v2": ((li, si), (so, lo)),
    }
    per = {"2020~23": (ts(2020), ts(2024)), "2024~": (ts(2024), t[-1] + 1)}
    for name, (ent, opp) in cand.items():
        for lab, e in (("long", (ent[0], z0)), ("short", (z0, ent[1])), ("both", ent)):
            row = []
            for p, (a0, a1) in per.items():
                if name.startswith("C4") and p == "2020~23":
                    a0 = ts(2022)
                j0, j1 = int(np.searchsorted(t, a0)), min(int(np.searchsorted(t, a1)), n)
                r = stats(simulate(d5, e, opp, "X1", STOP, None, j0, j1), ft, fr)
                row.append(f"{p}: n{r.get('n')} win{r.get('win')} 3x {r.get('x3')}% (mdd {r.get('x3_mdd')}) "
                           f"1x {r.get('x1')}% fund3 {r.get('fund3')}")
            print(name, lab, " | ".join(row), flush=True)
    # 대조: 3배 보유
    for p, (a0, a1) in per.items():
        j0, j1 = int(np.searchsorted(t, a0)), min(int(np.searchsorted(t, a1)), n)
        r = stats([(t[j0], d5["c"][j1 - 1] / d5["o"][j0] - 1, (d5["o"][j0] - d5["l"][j0:j1].min()) / d5["o"][j0], 1, t[j1 - 1])], ft, fr)
        print("hold long", p, r)


if __name__ == "__main__":
    main()
