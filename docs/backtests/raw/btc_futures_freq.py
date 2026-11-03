"""선물 15분봉, 이틀에 한 번 이상 매매하는 규칙 찾기. 등록: 2026-11-04-btc-futures-freq-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_futures_freq.py
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
from btc_fractal import load5
from btc_futures_compare import FUT, SPOT, load_funding, stats
from btc_scr_long_frame import atr_grid
from btc_scr_wait_fast import on_grid
from btc_trend_detect import detectors, to5, ts
from btc_xgb_trend import load_daily, sma

FILTERS = ("F1", "F2", "F3")
EXITS = {"X2": 0.01, "X3": 0.015, "X4": 0.025, "X5": None}
STOPS = (0.015, 0.03)
YEAR = 365.25 * 86_400_000


def simulate(d5, el, es, f, atr, tp, stop, j0, j1) -> list:
    o, h, l, t = d5["o"], d5["h"], d5["l"], d5["t"]  # noqa: E741
    out, side, px, line, peak, mae, ej = [], 0, 0.0, 0.0, 0.0, 0.0, 0
    for j in range(j0, j1):
        if side:
            ex = None
            if f[j] * side < 0:
                ex = o[j]
            else:
                if tp is None and (peak / px - 1) * side >= 0.01 and np.isfinite(atr[j]):
                    trail = peak - 3 * atr[j] * side
                    floor = px * (1 + 0.01 * side)
                    trail = max(trail, floor) if side > 0 else min(trail, floor)
                    line = max(line, trail) if side > 0 else min(line, trail)
                if l[j] <= line if side > 0 else h[j] >= line:
                    ex = min(o[j], line) if side > 0 else max(o[j], line)
                elif tp is not None:
                    tgt = px * (1 + tp * side)
                    if h[j] >= tgt if side > 0 else l[j] <= tgt:
                        ex = max(o[j], tgt) if side > 0 else min(o[j], tgt)
            if ex is None:
                mae = max(mae, (px - l[j]) / px if side > 0 else (h[j] - px) / px)
                peak = max(peak, h[j]) if side > 0 else min(peak, l[j])
                continue
            g = (ex / px - 1) * side
            out.append((float(t[ej]), g, max(mae, -g), side, float(t[j]), float(px)))
            side = 0
        s = 1 if (el[j] and f[j] > 0) else (-1 if (es[j] and f[j] < 0) else 0)
        if not s:
            continue
        side, px, ej = s, o[j], j
        line = px * (1 - stop * s)
        peak = px
        if l[j] <= line if s > 0 else h[j] >= line:
            out.append((float(t[j]), -stop, stop, s, float(t[j]), float(px)))
            side = 0
        else:
            mae = (px - l[j]) / px if s > 0 else (h[j] - px) / px
            peak = h[j] if s > 0 else l[j]
    return out


def per_trade3(tr, ft, fr) -> float:
    if not tr:
        return -1.0
    cf = np.r_[0.0, np.cumsum(fr)]
    g = np.array([x[1] for x in tr])
    mae = np.array([x[2] for x in tr])
    side = np.array([x[3] for x in tr])
    a = np.searchsorted(ft, [x[0] for x in tr], side="right")
    b = np.searchsorted(ft, [x[4] for x in tr], side="right")
    fund = side * (cf[b] - cf[a])
    return float(np.where(mae >= 0.32, -1.0, 3 * g - 0.003 - 3 * fund).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=3, help="진입 봉 크기(5분 배수): 3=15분, 1=5분")
    a = ap.parse_args()
    d5 = load5(FUT / "klines_5m.csv")
    t = d5["t"]
    n = len(t) - 1
    ft, fr = load_funding()
    dd, wk = load_daily(SPOT / "1d.csv"), load_daily(SPOT / "1w.csv")
    det = detectors(dd, wk)
    ma = sma(dd["c"], 60)
    states = {"F1": det["T8"], "F2": det["T5"],
              "F3": np.where(np.isnan(ma), 0.0, np.where(dd["c"] > ma, 1.0, -1.0))}
    f5 = {k: to5(t, dd["t"], v) for k, v in states.items()}
    el, es = on_grid(d5, a.factor, 0)
    atr = atr_grid(d5, max(a.factor, 3), 0)
    jj = {"sel": (int(np.searchsorted(t, ts(2020))), int(np.searchsorted(t, ts(2024)))),
          "test": (int(np.searchsorted(t, ts(2024))), n)}
    yrs = {k: (t[min(b, n)] - t[a]) / YEAR for k, (a, b) in jj.items()}
    rows = {}
    for fk, (xk, tp), st in itertools.product(FILTERS, EXITS.items(), STOPS):
        key = f"{fk}|{xk}|SL{st * 100:g}%"
        r = {}
        for p, (a0, b0) in jj.items():
            tr = simulate(d5, el, es, f5[fk], atr, tp, st, a0, b0)
            s = stats(tr, ft, fr)
            s["per_year"] = round(s["n"] / yrs[p], 1)
            s["ev3"] = round(per_trade3(tr, ft, fr) * 100, 3)
            g = np.array([x[1] for x in tr]) if tr else np.array([0.0])
            s["avg_win_px"] = round(float(g[g > 0.001].mean() * 100), 2) if (g > 0.001).any() else 0.0
            s["breakeven_win"] = round((st + 0.001) / ((tp or 0.02) + st) * 100, 1)
            r[p] = s
        rows[key] = r
        print(key, {p: (v["per_year"], v["win"], v["ev3"], v["x3"], v["x3_mdd"], v["avg_win_px"])
                    for p, v in r.items()}, flush=True)
    cand = [k for k, v in rows.items() if v["sel"]["per_year"] >= 180]
    print("candidates", len(cand), flush=True)
    if not cand:
        print("NO CANDIDATE with >=180/yr in 2020~23")
        return
    best = max(cand, key=lambda k: rows[k]["sel"]["ev3"])
    fk, xk, sl = best.split("|")
    tp, st = EXITS[xk], float(sl[2:-1]) / 100
    test = rows[best]["test"]

    def unfiltered(a, b):
        """필터 없음: 신호 방향을 그대로 받고, 반대 신호 봉에서 청산한다."""
        fl = np.where(el, 1.0, np.where(es, -1.0, 0.0))
        return simulate(d5, el, es, fl, atr, tp, st, a, b)

    base = stats(unfiltered(*jj["test"]), ft, fr)
    rng = np.random.default_rng(101)
    rand = []
    for _ in range(100):
        sh = int(rng.integers(60, len(states[fk]) - 60))
        fr5 = to5(t, dd["t"], np.roll(states[fk], sh))
        rand.append(per_trade3(simulate(d5, el, es, fr5, atr, tp, st, *jj["test"]), ft, fr))
    rand = np.array(rand)
    p = float((rand >= test["ev3"] / 100).mean())
    if a.factor > 1:
        e5, s5 = on_grid(d5, a.factor, 5)
        shifted = stats(simulate(d5, e5, s5, f5[fk], atr_grid(d5, a.factor, 5), tp, st, *jj["test"]), ft, fr)
    else:
        shifted = {"x3": float("nan")}
    crit = [test["per_year"] >= 180, test["x3"] > 0 and test["ev3"] > 0, test["x3"] > base["x3"],
            p < 0.05, shifted["x3"] > 0, test["avg_win_px"] >= 1.0]
    print("PICK", best, "test", test, "unfiltered", base, "p", p, "shifted", shifted, "crit", crit,
          flush=True)


if __name__ == "__main__":
    main()
