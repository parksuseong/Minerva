"""30분봉 SCR 3단 + 추세 추종 청산. 등록: 2026-11-03-btc-scr-trend-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_trend.py --out <폴더>
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import load5, resample
from btc_scr_wait_fast import COST, M5, on_grid, summ

ts = lambda y: datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000  # noqa: E731
EXITS = ("X1", "X2", "X3", "X4", "X5")
STOPS = (0.01, 0.02)
NRAND = 200


def atr_grid(d5: dict, offset: int) -> np.ndarray:
    sh = dict(d5)
    sh["t"] = d5["t"] - offset * 60_000
    f = resample(sh, 6)
    f["t"] = f["t"] + offset * 60_000
    pc = np.r_[f["c"][0], f["c"][:-1]]
    tr = np.maximum(f["h"], pc) - np.minimum(f["l"], pc)
    a = np.convolve(tr, np.ones(14) / 14, "full")[: len(tr)]
    a[:13] = np.nan
    j = np.searchsorted(f["t"] + 6 * M5, d5["t"], side="right") - 1
    return np.where(j >= 0, a[np.maximum(j, 0)], np.nan)


def simulate(d5, ent, opp, mode, stop, atr, j0, j1) -> list:
    o, h, l, t = d5["o"], d5["h"], d5["l"], d5["t"]  # noqa: E741
    EL, ES = ent
    XL, XS = opp
    tr, side, px, line, peak, mae, ej = [], 0, 0.0, 0.0, 0.0, 0.0, 0
    for j in range(j0, j1):
        if side:
            if mode in ("X3", "X4", "X5"):
                if mode == "X5":
                    trail = peak * (1 - 0.03 * side)
                else:
                    k = 3 if mode == "X3" else 5
                    trail = peak - k * atr[j] * side if np.isfinite(atr[j]) else line
                line = max(line, trail) if side > 0 else min(line, trail)
            ex = None
            if mode in ("X1", "X2") and (XS[j] if side > 0 else XL[j]):
                ex = o[j]
            elif l[j] <= line if side > 0 else h[j] >= line:
                ex = min(o[j], line) if side > 0 else max(o[j], line)
            if ex is None:
                mae = max(mae, (px - l[j]) / px if side > 0 else (h[j] - px) / px)
                peak = max(peak, h[j]) if side > 0 else min(peak, l[j])
                continue
            g = (ex / px - 1) * side
            tr.append((float(t[ej]), g, max(mae, -g), side, float(t[j]), float(px)))
            side = 0
        s = 1 if EL[j] else (-1 if ES[j] else 0)
        if not s:
            continue
        side, px, ej = s, o[j], j
        line = px * (1 - stop * side)
        if l[j] <= line if side > 0 else h[j] >= line:
            g = (line / px - 1) * side
            tr.append((float(t[j]), g, -g, side, float(t[j]), float(px)))
            side = 0
            continue
        peak = h[j] if side > 0 else l[j]
        mae = (px - l[j]) / px if side > 0 else (h[j] - px) / px
    return tr


def stats(tr: list) -> dict:
    r = summ(tr)
    if not tr:
        return r
    net = np.array([x[1] for x in tr]) - 2 * COST
    w, lo = net[net > 0], net[net <= 0]
    r.update(pf=round(float(w.sum() / max(-lo.sum(), 1e-12)), 2),
             avg_win=round(float(w.mean() * 100), 2) if len(w) else 0.0,
             avg_loss=round(float(lo.mean() * 100), 2) if len(lo) else 0.0,
             hold_h=round(float(np.median([(x[4] - x[0]) / 3_600_000 for x in tr])), 1))
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(a.data) / "5m.csv")
    t = d5["t"]
    n = len(t) - 1
    js = {k: int(np.searchsorted(t, ts(y))) for k, y in (("18", 2018), ("22", 2022), ("25", 2025))}
    per = {"2018~21": (js["18"], js["22"]), "train": (js["22"], js["25"]), "test": (js["25"], n)}
    rep: dict = {}
    sig = {}
    for off in (0, 15):
        e30 = on_grid(d5, 6, off)
        sig[off] = (e30, {"X1": e30, "X2": on_grid(d5, 48, off)}, atr_grid(d5, off))
    e30, opps, atr = sig[0]
    for x, st in itertools.product(EXITS, STOPS):
        key = f"{x}|SL{st * 100:g}%"
        rep[key] = {p: stats(simulate(d5, e30, opps.get(x, e30), x, st, atr, *jj))
                    for p, jj in per.items()}
        r = rep[key]
        print(key, {p: (r[p]["n"], r[p]["win"], r[p]["ev"], r[p]["x1"], r[p].get("pf"))
                    for p in per}, flush=True)
    best = max(rep, key=lambda k: rep[k]["train"]["ev"])
    x, st = best.split("|")[0], float(best.split("SL")[1][:-1]) / 100
    e15, opps15, atr15 = sig[15]
    off15 = stats(simulate(d5, e15, opps15.get(x, e15), x, st, atr15, *per["test"]))
    j0, j1 = per["test"]
    k = int(e30[0][j0:j1].sum() + e30[1][j0:j1].sum())
    rng = np.random.default_rng(3)
    rand = []
    for _ in range(NRAND):
        pick = rng.choice(np.arange(j0, j1), k, replace=False)
        sd = rng.random(k) < 0.5
        el, es = np.zeros(len(t), bool), np.zeros(len(t), bool)
        el[pick[sd]], es[pick[~sd]] = True, True
        rand.append(stats(simulate(d5, (el, es), (es, el), x, st, atr, j0, j1))["x1"])
    rand = np.array(rand)
    test = rep[best]["test"]
    p = float((rand >= test["x1"]).mean())
    rep["_pick"] = {"best": best, "off15_test": off15, "rand_x1_median": float(np.median(rand)),
                    "rand_x1_p90": float(np.quantile(rand, 0.9)), "p_random": p}
    crit = [test["x1"] > 0 and test["ev"] > 0 and test["pf"] >= 1.2, off15["x1"] > 0,
            rep[best]["2018~21"]["x1"] > 0, p < 0.05]
    rep["_pick"]["criteria"] = crit
    print("PICK", rep["_pick"], flush=True)
    tr = simulate(d5, e30, opps.get(x, e30), x, st, atr, js["25"], n)
    rep["_trades"] = [[z[0], round(z[1], 5), z[3], z[4], z[5]] for z in tr]
    (out / "result.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
