"""15분봉 ML + SCR v2에 숏을 붙인 참고 확인. 문서: 2026-11-04-btc-ml-15m-short-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_15m_short.py --pred <ml15/pred.npz>
"""

from __future__ import annotations

import argparse

import numpy as np
import btc_ml_ensemble as E
from btc_fractal import recent
from btc_ml_15m import BE, COST, STOP

W, M = 16, 0.04


def signals(probs, on_l, on_s):
    known = np.isfinite(probs).all(0)
    with np.errstate(invalid="ignore"):
        pm = np.nanmean(np.where(known, probs, 0.5), 0)
        up_v, dn_v = (probs > BE).sum(0), (probs < 1 - BE).sum(0)
        zl, zs = recent(on_l, W), recent(on_s, W)
        long_in = known & zl & (up_v >= 3) & (pm >= BE + M)
        long_out = known & zs & (pm < 0.5)
        short_in = known & zs & (dn_v >= 3) & (pm <= 1 - (BE + M))
        short_out = known & zl & (pm > 0.5)
    return long_in, long_out, short_in, short_out


def simulate(P, sig, i0, i1, sides):
    o, h, l = P["o"], P["h"], P["l"]  # noqa: E741
    li, lo, si, so = sig
    out, side, px, line = [], 0, 0.0, 0.0
    for i in range(max(i0, 1), i1):
        if side:
            exit_sig = lo[i - 1] if side > 0 else so[i - 1]
            hit = l[i] <= line if side > 0 else h[i] >= line
            if exit_sig:
                out.append((side, (o[i] / px - 1) * side))
                side = 0
            elif hit:
                fill = min(o[i], line) if side > 0 else max(o[i], line)
                out.append((side, (fill / px - 1) * side))
                side = 0
                continue
            else:
                continue
        s = 1 if (1 in sides and li[i - 1]) else (-1 if (-1 in sides and si[i - 1]) else 0)
        if s:
            side, px = s, o[i]
            line = px * (1 - STOP * s)
            if l[i] <= line if s > 0 else h[i] >= line:
                out.append((s, -STOP))
                side = 0
    return out


def summ(tr):
    if not tr:
        return {"n": 0}
    g = np.array([x[1] for x in tr])
    out = {"n": len(tr), "long": sum(1 for x in tr if x[0] > 0)}
    for k, lev in (("x1", 1), ("x3", 3)):
        net = lev * g - 2 * lev * COST
        eq = np.cumprod(1 + np.maximum(net, -1))
        peak = np.maximum.accumulate(np.r_[1.0, eq])
        out[k] = round(float((eq[-1] - 1) * 100), 1)
        out[k + "_mdd"] = round(float((eq / peak[1:] - 1).min() * 100), 1)
    out["win"] = round(float((g - 2 * COST > 0.001).mean() * 100), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    z = np.load(a.pred)
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    t = z["t"]
    sig = signals(z["probs"], z["on_l"], z["on_s"])
    i22, i24 = (int(np.searchsorted(t, E.ts(v))) for v in (2022, 2024))
    for name, (a0, a1) in (("2022~23", (i22, i24)), ("2024~", (i24, len(t)))):
        for lab, sides in (("long", (1,)), ("short", (-1,)), ("both", (1, -1))):
            print(name, lab, summ(simulate(P, sig, a0, a1, sides)), flush=True)


if __name__ == "__main__":
    main()
