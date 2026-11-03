"""비트코인 T8 추세 필터 + 진입·청산 고르기. 등록: 2026-11-03-btc-t8-entry-exit-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_t8_entry_exit.py --out <폴더>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from btc_fractal import load5
from btc_scr_long_frame import atr_grid
from btc_scr_trend import simulate, stats
from btc_scr_wait_fast import on_grid
from btc_trend_detect import detectors, to5, ts
from btc_trend_detect_majors import ev3
from btc_xgb_trend import load_daily

FRAMES = {"15m": (3, 5), "30m": (6, 15), "1h": (12, 30)}
EXITS = ("X1", "X2", "X3", "X4")
STOPS = (0.02, 0.03)
NRAND = 100


class Case:
    def __init__(self, d5: dict, dd_t: np.ndarray, state: np.ndarray):
        self.d5, self.dd_t = d5, dd_t
        self.sig: dict = {}
        self.atr: dict = {}
        self.h4 = on_grid(d5, 48, 0)
        self.set_state(state)

    def set_state(self, state: np.ndarray) -> None:
        self.f5 = to5(self.d5["t"], self.dd_t, state)

    def signals(self, frame: str, shifted: bool) -> tuple:
        fac, off = FRAMES[frame]
        key = (frame, shifted)
        if key not in self.sig:
            self.sig[key] = on_grid(self.d5, fac, off if shifted else 0)
            self.atr[key] = atr_grid(self.d5, fac, off if shifted else 0)
        return self.sig[key], self.atr[key]

    def run(self, frame: str, ex: str, stop: float, jj: tuple, shifted: bool = False) -> list:
        (el, es), atr = self.signals(frame, shifted)
        f = self.f5
        ent = (el & (f > 0), es & (f < 0))
        if ex == "X1":
            return simulate(self.d5, ent, (el, es), "X1", stop, atr, *jj)
        if ex == "X2":
            return simulate(self.d5, ent, self.h4, "X1", stop, atr, *jj)
        if ex == "X3":
            return simulate(self.d5, ent, (el, es), "X3", stop, atr, *jj)
        return simulate(self.d5, ent, (f >= 0, f < 1), "X1", stop, atr, *jj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path("~/.minerva/data/btc").expanduser()
    d5 = load5(root / "5m.csv")
    dd, wk = load_daily(root / "1d.csv"), load_daily(root / "1w.csv")
    state = detectors(dd, wk)["T8"]
    case = Case(d5, dd["t"], state)
    t = d5["t"]
    n = len(t) - 1
    per = {"2018~21": (int(np.searchsorted(t, ts(2018))), int(np.searchsorted(t, ts(2022)))),
           "2022~": (int(np.searchsorted(t, ts(2022))), n)}
    rows, train_ev = {}, {}
    for fr, ex, st in itertools.product(FRAMES, EXITS, STOPS):
        key = f"{fr}|{ex}|SL{st * 100:g}%"
        r = {}
        for p, jj in per.items():
            tr = case.run(fr, ex, st, jj)
            r[p] = stats(tr)
            e = ev3(tr)
            r[p]["ev3"] = round(float(e.mean() * 100), 3) if len(e) else None
            if p == "2018~21":
                train_ev[key] = float(e.mean()) if len(e) else -9.0
        rows[key] = r
        print(key, {p: (r[p].get("n"), r[p].get("win"), r[p].get("ev3"), r[p].get("x3"),
                        r[p].get("x3_mdd")) for p in per}, flush=True)
    best = max(train_ev, key=train_ev.get)
    fr, ex, sl = best.split("|")
    st = float(sl[2:-1]) / 100
    test = rows[best]["2022~"]
    sh = stats(case.run(fr, ex, st, per["2022~"], shifted=True))
    rng = np.random.default_rng(41)
    rand = []
    for _ in range(NRAND):
        case.set_state(np.roll(state, int(rng.integers(60, len(state) - 60))))
        e = ev3(case.run(fr, ex, st, per["2022~"]))
        rand.append(float(e.mean()) if len(e) else -1.0)
    case.set_state(state)
    rand = np.array(rand)
    p = float((rand >= test["ev3"] / 100).mean())
    crit = [test["x3"] > 0 and test["ev3"] > 0, test["x3"] > 296.0, sh.get("x3", -1) > 0, p < 0.05,
            test["x3_mdd"] > -70]
    pick = {"best": best, "train_ev3": round(train_ev[best] * 100, 3), "test": test, "shifted": sh,
            "rand_median_ev3": round(float(np.median(rand)) * 100, 3), "p": p, "criteria": crit}
    print("PICK", json.dumps(pick), flush=True)
    tr = case.run(fr, ex, st, per["2022~"])
    rep = {"rows": rows, "pick": pick,
           "trades": [[z[0], round(z[1], 5), z[3], z[4], z[5]] for z in tr]}
    (out / "result.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
