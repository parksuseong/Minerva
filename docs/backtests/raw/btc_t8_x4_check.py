"""T8 추세 청산(X4) 15분·30분봉 확인. 등록: 2026-11-03-btc-t8-x4-check-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_t8_x4_check.py --out <폴더>
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import btc_t8_entry_exit as E
from btc_fractal import load5
from btc_scr_trend import stats
from btc_trend_detect import detectors, ts
from btc_trend_detect_majors import ev3, frames
from btc_xgb_trend import load_daily

CELLS = list(itertools.product(("15m", "30m"), (0.02, 0.03)))
SHIFTS = {"15m": (5, 10), "30m": (15,)}
NRAND = 200
COINS = ("ETH", "BNB", "SOL", "XRP")


def periods(t):
    n = len(t) - 1
    return {"2018~21": (int(np.searchsorted(t, ts(2018))), int(np.searchsorted(t, ts(2022)))),
            "2022~": (int(np.searchsorted(t, ts(2022))), n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path("~/.minerva/data").expanduser()
    rep: dict = {"btc": {}, "alts": {}}
    d5 = load5(data / "btc/5m.csv")
    dd, wk = load_daily(data / "btc/1d.csv"), load_daily(data / "btc/1w.csv")
    state = detectors(dd, wk)["T8"]
    case = E.Case(d5, dd["t"], state)
    per = periods(d5["t"])
    rng = np.random.default_rng(51)
    for fr, st in CELLS:
        key = f"{fr}|X4|SL{st * 100:g}%"
        tr = case.run(fr, "X4", st, per["2022~"])
        base = stats(tr)
        e0 = float(ev3(tr).mean())
        sh = {}
        for off in SHIFTS[fr]:
            E.FRAMES[fr] = (E.FRAMES[fr][0], off)
            case.sig.pop((fr, True), None)
            sh[off] = stats(case.run(fr, "X4", st, per["2022~"], shifted=True)).get("x3")
        rand = []
        for _ in range(NRAND):
            case.set_state(np.roll(state, int(rng.integers(60, len(state) - 60))))
            e = ev3(case.run(fr, "X4", st, per["2022~"]))
            rand.append(float(e.mean()) if len(e) else -1.0)
        case.set_state(state)
        p = float((np.array(rand) >= e0).mean())
        rep["btc"][key] = {"test": base, "ev3": e0, "shifted_x3": sh, "p": p}
        print("BTC", key, {k: base.get(k) for k in ("n", "win", "x3", "x3_mdd", "x1")},
              "ev3", round(e0 * 100, 2), "shift", sh, "p", p, flush=True)
    del d5, case
    alt = {f"{fr}|X4|SL{st * 100:g}%": {"2022~": [], "2018~21": [], "pos": 0} for fr, st in CELLS}
    for coin in COINS:
        d5 = load5(data / f"alt/{coin}USDT/5m.csv")
        dd, wk = frames(d5)
        case = E.Case(d5, dd["t"], detectors(dd, wk)["T8"])
        per = periods(d5["t"])
        for fr, st in CELLS:
            key = f"{fr}|X4|SL{st * 100:g}%"
            r = {}
            for pname, jj in per.items():
                if jj[1] - jj[0] < 288 * 30:
                    continue
                tr = case.run(fr, "X4", st, jj)
                alt[key][pname] += list(ev3(tr))
                r[pname] = stats(tr)
            if r.get("2022~", {}).get("x3", -1) > 0:
                alt[key]["pos"] += 1
            rep["alts"].setdefault(key, {})[coin] = r
            print(coin, key, {p: (v.get("n"), v.get("win"), v.get("x3"), v.get("x3_mdd"))
                              for p, v in r.items()}, flush=True)
        del d5, case
    rep["verdict"] = {}
    for fr, st in CELLS:
        key = f"{fr}|X4|SL{st * 100:g}%"
        b, al = rep["btc"][key], alt[key]
        m22 = float(np.mean(al["2022~"])) if al["2022~"] else float("nan")
        m18 = float(np.mean(al["2018~21"])) if al["2018~21"] else float("nan")
        crit = [b["p"] < 0.0125, all(v > 0 for v in b["shifted_x3"].values()),
                al["pos"] >= 3 and m22 > 0, m18 > 0]
        rep["verdict"][key] = {"alt_pos": al["pos"], "alt_ev3_2022": m22, "alt_ev3_2018": m18,
                               "criteria": crit}
        print("VERDICT", key, rep["verdict"][key], flush=True)
    (out / "result.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
