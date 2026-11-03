"""실시간 추세 판별 + 30분봉 SCR. 등록: 2026-11-03-btc-trend-detect-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_trend_detect.py --out <폴더>
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.modules.setdefault("xgboost", type(sys)("xgboost"))  # 라벨링 함수만 쓴다
from btc_fractal import load5, scr3  # noqa: E402
from btc_scr_trend import simulate, stats  # noqa: E402
from btc_scr_wait_fast import on_grid  # noqa: E402
from btc_xgb_trend import labeling, load_daily, sma  # noqa: E402

D = 86_400_000
NRAND = 100
ts = lambda y: datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000  # noqa: E731


def hold_state(up: np.ndarray, dn: np.ndarray) -> np.ndarray:
    out = np.zeros(len(up))
    cur = 0.0
    for i in range(len(up)):
        if up[i]:
            cur = 1.0
        elif dn[i]:
            cur = -1.0
        out[i] = cur
    return out


def structure(d: dict, k: int) -> np.ndarray:
    h, l, c = d["h"], d["l"], d["c"]  # noqa: E741
    n = len(c)
    out = np.zeros(n)
    hi = lo = np.nan
    cur = 0.0
    for i in range(n):
        p = i - k  # i일 마감에 확정되는 스윙 후보
        if p - k >= 0:
            if h[p] == h[p - k : i + 1].max():
                hi = h[p]
            if l[p] == l[p - k : i + 1].min():
                lo = l[p]
        if not np.isnan(hi) and c[i] > hi:
            cur = 1.0
        elif not np.isnan(lo) and c[i] < lo:
            cur = -1.0
        out[i] = cur
    return out


def detectors(dd: dict, wk: dict) -> dict:
    c = dd["c"]
    ma20, ma60 = sma(c, 20), sma(c, 60)
    t1 = np.where(np.isnan(ma60), 0, np.where(c > ma60, 1.0, -1.0))
    t2 = np.where(np.isnan(ma60), 0, np.where(ma20 > ma60, 1.0, -1.0))
    on_l, on_s, *_ = scr3(dd)
    t3 = hold_state(on_l, on_s)
    wl, ws, *_ = scr3(wk)
    ws_state = hold_state(wl, ws)
    wclose = wk["t"] + 7 * D
    j = np.searchsorted(wclose, dd["t"] + D, side="right") - 1
    t4 = np.where(j >= 0, ws_state[np.maximum(j, 0)], 0.0)
    t5, t6 = structure(dd, 5), structure(dd, 10)
    hh = np.r_[np.full(20, np.nan), np.array([dd["h"][i - 20 : i].max() for i in range(20, len(c))])]
    ll = np.r_[np.full(20, np.nan), np.array([dd["l"][i - 20 : i].min() for i in range(20, len(c))])]
    with np.errstate(invalid="ignore"):
        t7 = hold_state(c > hh, c < ll)
    t8 = np.where(t4 == t5, t4, 0.0)
    return {"T1": t1, "T2": t2, "T3": t3, "T4": t4, "T5": t5, "T6": t6, "T7": t7, "T8": t8}


def to5(d5t: np.ndarray, dd_t: np.ndarray, state: np.ndarray) -> np.ndarray:
    """그날 쓰는 값은 전날 마감 상태."""
    day = np.searchsorted(dd_t, d5t, side="right") - 1
    prev = day - 1
    return np.where(prev >= 0, state[np.maximum(prev, 0)], 0.0)


def agreement(state: np.ndarray, truth: np.ndarray, ok: np.ndarray) -> dict:
    used = np.r_[0.0, state[:-1]]
    hit = (used == truth) & ok
    turns = np.flatnonzero((np.diff(truth) != 0) & ok[1:]) + 1
    lags = []
    for tp in turns:
        w = np.flatnonzero(used[tp : tp + 120] == truth[tp])
        lags.append(int(w[0]) if len(w) else 120)
    return {"agree": round(float(hit.sum() / ok.sum()) * 100, 1),
            "neutral": round(float(((used == 0) & ok).sum() / ok.sum()) * 100, 1),
            "lag_median": float(np.median(lags)), "turns": len(turns)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path("~/.minerva/data/btc").expanduser()
    d5 = load5(root / "5m.csv")
    dd, wk = load_daily(root / "1d.csv"), load_daily(root / "1w.csv")
    det = detectors(dd, wk)
    lab, p0, p1 = labeling(dd["c"])
    truth = np.zeros(len(dd["t"]))
    truth[p0:p1] = np.where(lab[p0:p1] == 0, 1.0, -1.0)
    ok = np.zeros(len(truth), bool)
    ok[p0:p1] = True
    ok &= dd["t"] >= ts(2018)
    rep: dict = {"agreement": {k: agreement(v, truth, ok) for k, v in det.items()}}
    for k, v in rep["agreement"].items():
        print("agree", k, v, flush=True)
    t = d5["t"]
    n = len(t) - 1
    j18, j22 = int(np.searchsorted(t, ts(2018))), int(np.searchsorted(t, ts(2022)))
    per = {"2018~21": (j18, j22), "2022~": (j22, n)}
    sig = {0: on_grid(d5, 6, 0), 15: on_grid(d5, 6, 15)}
    f5 = {k: to5(t, dd["t"], v) for k, v in det.items()}

    def run(filt, st, off, jj):
        el, es = sig[off]
        ent = (el, es) if filt is None else (el & (filt > 0), es & (filt < 0))
        return stats(simulate(d5, ent, (el, es), "X1", st, None, *jj))

    rows = {}
    for name, st in itertools.product(["none", *det], (0.01, 0.03)):
        key = f"{name}|SL{st * 100:g}%"
        filt = None if name == "none" else f5[name]
        rows[key] = {p: run(filt, st, 0, jj) for p, jj in per.items()}
        r = rows[key]
        print(key, {p: (r[p]["n"], r[p]["win"], r[p]["x3"], r[p]["x3_mdd"]) for p in per}, flush=True)
    rep["rows"] = rows

    cand = {k: v for k, v in rows.items() if not k.startswith("none")}

    def ev3_of(name, st, off, jj):
        el, es = sig[off]
        filt = f5[name]
        tr = simulate(d5, (el & (filt > 0), es & (filt < 0)), (el, es), "X1", st, None, *jj)
        if not tr:
            return -9.0
        g = np.array([x[1] for x in tr])
        mae = np.array([x[2] for x in tr])
        return float(np.where(mae >= 0.32, -1.0, 3 * g - 6 * 0.0005).mean())

    scores = {k: ev3_of(k.split("|")[0], float(k.split("SL")[1][:-1]) / 100, 0, per["2018~21"])
              for k in cand}
    best = max(scores, key=scores.get)
    bname, bst = best.split("|")[0], float(best.split("SL")[1][:-1]) / 100
    test = rows[best]["2022~"]
    test_ev3 = ev3_of(bname, bst, 0, per["2022~"])
    off15 = run(f5[bname], bst, 15, per["2022~"])
    rng = np.random.default_rng(21)
    nd = len(dd["t"])
    rand = []
    orig = f5[bname]
    for _ in range(NRAND):
        sh = int(rng.integers(60, nd - 60))
        f5[bname] = to5(t, dd["t"], np.roll(det[bname], sh))
        rand.append(ev3_of(bname, bst, 0, per["2022~"]))
    f5[bname] = orig
    rand = np.array(rand)
    p = float((rand >= test_ev3).mean())
    base_none = rows[f"none|SL{bst * 100:g}%"]["2022~"]["x3"]
    base_t1 = rows[f"T1|SL{bst * 100:g}%"]["2022~"]["x3"]
    crit = [test["x3"] > 0 and test_ev3 > 0, test["x3"] > max(base_none, base_t1),
            off15["x3"] > 0, p < 0.05]
    rep["pick"] = {"best": best, "train_ev3": scores, "test": test, "test_ev3": test_ev3,
                   "off15": off15, "rand_ev3_median": float(np.median(rand)), "p_random": p,
                   "criteria": crit}
    print("PICK", best, "test", test, "ev3", round(test_ev3 * 100, 3), "off15 x3", off15["x3"],
          "p", p, "crit", crit, flush=True)
    print("train ev3", {k: round(v * 100, 3) for k, v in scores.items()}, flush=True)
    (out / "result.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
