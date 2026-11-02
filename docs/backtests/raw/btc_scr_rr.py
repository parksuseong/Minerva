"""30분봉 SCR 3단 + 짧은 손절·긴 익절 격자. 등록: 2026-11-03-btc-scr-rr-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_rr.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import btc_scr_model as M
from btc_fractal import load5, resample, scr3
from btc_swing_patterns import SPLIT

STOPS = (0.005, 0.0075, 0.01)
TPS = (0.015, 0.02, 0.03, 0.05)
COST = 0.0005


def trade(d, i, side, opp, sl, tp, use_opp):
    o, h, l = d["o"], d["h"], d["l"]
    n = len(o)
    e = i + 1
    px = o[e]
    stop = px * (1 - sl * side)
    tgt = px * (1 + tp * side)
    for j in range(e, n - 1):
        if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
            f = min(o[j], stop) if side > 0 else max(o[j], stop)
            return (f / px - 1) * side, j
        if (side > 0 and h[j] >= tgt) or (side < 0 and l[j] <= tgt):
            f = max(o[j], tgt) if side > 0 else min(o[j], tgt)
            return (f / px - 1) * side, j
        if use_opp and opp[j]:
            return (o[j + 1] / px - 1) * side, j + 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(args.data) / "5m.csv")
    sh = dict(d5)
    sh["t"] = d5["t"] - args.offset * 60_000
    d = resample(sh, 6)
    d["t"] = d["t"] + args.offset * 60_000
    n = len(d["c"])
    on_l, on_s, *_ = scr3(d)
    d1 = M.higher_dir(d5, 288, d["t"])
    sigs = sorted([(int(i), s) for s, on in ((1, on_l), (-1, on_s)) for i in np.flatnonzero(on)
                   if 3000 <= i < n - 2])
    rep: dict = {"offset": args.offset, "cells": {}}
    for gate in ("none", "D"):
        for use_opp in (False, True):
            for sl in STOPS:
                for tp in TPS:
                    busy, tr = -1, []
                    for i, side in sigs:
                        if i <= busy or (gate == "D" and d1[i] * side <= 0):
                            continue
                        res = trade(d, i, side, on_s if side > 0 else on_l, sl, tp, use_opp)
                        if res is None:
                            continue
                        g, x = res
                        tr.append((g, float(d["t"][i])))
                        busy = x
                    g = np.array([z[0] for z in tr])
                    t = np.array([z[1] for z in tr])

                    def st(m, lev=1, cost=COST):
                        if m.sum() == 0:
                            return None
                        net = lev * (g[m] - 2 * cost)
                        eq = np.cumprod(1 + net)
                        pk = np.maximum.accumulate(np.r_[1.0, eq])
                        s = np.sort(g[m] - 2 * cost)
                        return {"n": int(m.sum()),
                                "win": round(float(((g[m] - 2 * cost) > 0.001).mean() * 100), 1),
                                "ev": round(float((g[m] - 2 * cost).mean() * 100), 3),
                                "ret": round(float((eq[-1] - 1) * 100), 1),
                                "mdd": round(float((eq / pk[1:] - 1).min() * 100), 1),
                                "drop5": round(float(s[:-5].mean() * 100), 3) if len(s) > 5 else None}
                    key = f"{gate}_{'opp' if use_opp else 'hold'}_sl{sl * 100:.2f}_tp{tp * 100:.0f}"
                    rep["cells"][key] = {"old": st(t < SPLIT), "new": st(t >= SPLIT), "all": st(t > 0),
                                         "all_c10": st(t > 0, 1, 0.001), "new_3x": st(t >= SPLIT, 3),
                                         "all_3x": st(t > 0, 3)}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
