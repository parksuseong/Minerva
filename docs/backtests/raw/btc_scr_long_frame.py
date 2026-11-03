"""4시간봉·일봉 SCR 3단. 등록: 2026-11-03-btc-scr-long-frame-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_long_frame.py --out <폴더>
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import load5, resample
from btc_scr_trend import simulate, stats
from btc_scr_wait_fast import M5, on_grid

ts = lambda y: datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000  # noqa: E731
FRAMES = {"4h": (48, 120), "1d": (288, 720)}
NRAND = 200


def atr_grid(d5: dict, factor: int, offset: int) -> np.ndarray:
    sh = dict(d5)
    sh["t"] = d5["t"] - offset * 60_000
    f = resample(sh, factor)
    f["t"] = f["t"] + offset * 60_000
    pc = np.r_[f["c"][0], f["c"][:-1]]
    tr = np.maximum(f["h"], pc) - np.minimum(f["l"], pc)
    a = np.convolve(tr, np.ones(14) / 14, "full")[: len(tr)]
    a[:13] = np.nan
    j = np.searchsorted(f["t"] + factor * M5, d5["t"], side="right") - 1
    return np.where(j >= 0, a[np.maximum(j, 0)], np.nan)


def run(d5, fac, off, mode, st, j0, j1):
    e = on_grid(d5, fac, off)
    return simulate(d5, e, e, "X1" if mode == "E1" else "X3", st, atr_grid(d5, fac, off), j0, j1), e


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
    j18, j23 = (int(np.searchsorted(t, ts(y))) for y in (2018, 2023))
    end = t[-1] + M5
    cuts = {"1y": end - 365 * 86_400_000, "6m": end - 183 * 86_400_000}
    rep: dict = {}
    rng = np.random.default_rng(9)
    for fname, (fac, shift) in FRAMES.items():
        rows = {}
        for mode, st in itertools.product(("E1", "E2"), (0.03, 0.06)):
            key = f"{fname}|{mode}|SL{st * 100:g}%"
            tr_train, _ = run(d5, fac, 0, mode, st, j18, j23)
            tr_test, e = run(d5, fac, 0, mode, st, j23, n)
            r = {"train": stats(tr_train), "test": stats(tr_test)}
            r.update({k: stats([z for z in tr_test if z[0] >= c]) for k, c in cuts.items()})
            rows[key] = r
            print(key, {p: (r[p].get("n"), r[p].get("win"), r[p].get("ev"), r[p].get("x1"),
                            r[p].get("x3"), r[p].get("pf")) for p in ("train", "test", "1y", "6m")},
                  flush=True)
        best = max(rows, key=lambda k: rows[k]["train"].get("ev", -9))
        _, mode, sl = best.split("|")
        st = float(sl[2:-1]) / 100
        sh = stats(run(d5, fac, shift, mode, st, j23, n)[0])
        e = on_grid(d5, fac, 0)
        k = int(e[0][j23:n].sum() + e[1][j23:n].sum())
        atr = atr_grid(d5, fac, 0)
        rand = []
        for _ in range(NRAND):
            pick = rng.choice(np.arange(j23, n), k, replace=False)
            sd = rng.random(k) < 0.5
            el, es = np.zeros(len(t), bool), np.zeros(len(t), bool)
            el[pick[sd]], es[pick[~sd]] = True, True
            rand.append(stats(simulate(d5, (el, es), (es, el), "X1" if mode == "E1" else "X3",
                                       st, atr, j23, n)).get("x1", 0.0))
        rand = np.array(rand)
        te = rows[best]["test"]
        p = float((rand >= te["x1"]).mean())
        crit = [te["x1"] > 0 and te["ev"] > 0 and te.get("pf", 0) >= 1.2, sh.get("x1", -1) > 0,
                p < 0.05, te["n"] >= 30]
        rows["_pick"] = {"best": best, "shifted_test": sh, "p_random": p,
                         "rand_median": float(np.median(rand)), "criteria": crit}
        trs, _ = run(d5, fac, 0, mode, st, j23, n)
        rows["_trades"] = [[z[0], round(z[1], 5), z[3], z[4], z[5]] for z in trs]
        print("PICK", fname, rows["_pick"], flush=True)
        rep[fname] = rows
    (out / "result.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
