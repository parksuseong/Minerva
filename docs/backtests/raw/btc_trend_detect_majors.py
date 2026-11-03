"""주봉 SCR + 일봉 구조 필터(T8)를 메이저 코인에서 확인. 등록: 2026-11-03-btc-trend-detect-majors-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_trend_detect_majors.py --out <폴더> [--check-btc]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import load5
from btc_scr_trend import simulate, stats
from btc_scr_wait_fast import on_grid
from btc_trend_detect import detectors, to5
from btc_xgb_trend import load_daily

D = 86_400_000
STOP = 0.03
NRAND = 100
COINS = ("ETH", "BNB", "SOL", "XRP")
ts = lambda y: datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000  # noqa: E731


def bins(d5: dict, span: int, shift: int = 0) -> dict:
    key = ((d5["t"] - shift) // span).astype(np.int64)
    st = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    en = np.r_[st[1:], len(key)]
    st, en = st[:-1], en[:-1]  # 마지막 묶음은 진행 중일 수 있어 버린다
    return {"t": key[st].astype(float) * span + shift, "o": d5["o"][st],
            "h": np.maximum.reduceat(d5["h"], st)[: len(st)], "l": np.minimum.reduceat(d5["l"], st)[: len(st)],
            "c": d5["c"][en - 1], "v": np.add.reduceat(d5["v"], st)[: len(st)]}


def frames(d5: dict) -> tuple:
    return bins(d5, D), bins(d5, 7 * D, 4 * D)


def ev3(tr: list) -> np.ndarray:
    if not tr:
        return np.array([])
    g = np.array([x[1] for x in tr])
    mae = np.array([x[2] for x in tr])
    return np.where(mae >= 0.32, -1.0, 3 * g - 6 * 0.0005)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-btc", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path("~/.minerva/data").expanduser()
    if a.check_btc:
        d5 = load5(data / "btc/5m.csv")
        dd, wk = frames(d5)
        fd, fw = load_daily(data / "btc/1d.csv"), load_daily(data / "btc/1w.csv")
        for name, x, y in (("1d", dd, fd), ("1w", wk, fw)):
            com, ia, ib = np.intersect1d(x["t"], y["t"], return_indices=True)
            diff = np.abs(x["c"][ia] / y["c"][ib] - 1)
            print(name, "rows", len(x["t"]), len(y["t"]), "common", len(com),
                  "close max diff", float(diff.max()))
        t8a = detectors(dd, wk)["T8"]
        t8b = detectors(fd, fw)["T8"]
        com, ia, ib = np.intersect1d(dd["t"], fd["t"], return_indices=True)
        print("T8 same", float((t8a[ia] == t8b[ib]).mean()))
        return
    rng = np.random.default_rng(33)
    pool = {"t8": [], "none": [], "off15": [], "aux": []}
    rand = np.zeros((NRAND, 2))  # 합, 개수
    rep: dict = {"coins": {}}
    for coin in COINS:
        d5 = load5(data / f"alt/{coin}USDT/5m.csv")
        t = d5["t"]
        n = len(t) - 1
        dd, wk = frames(d5)
        state = detectors(dd, wk)["T8"]
        f5 = to5(t, dd["t"], state)
        j18, j22 = int(np.searchsorted(t, ts(2018))), int(np.searchsorted(t, ts(2022)))
        sig0, sig15 = on_grid(d5, 6, 0), on_grid(d5, 6, 15)

        def run(sig, filt, jj):
            el, es = sig
            ent = (el, es) if filt is None else (el & (filt > 0), es & (filt < 0))
            return simulate(d5, ent, (el, es), "X1", STOP, None, *jj)

        tr = run(sig0, f5, (j22, n))
        tn = run(sig0, None, (j22, n))
        t15 = run(sig15, f5, (j22, n))
        ta = run(sig0, f5, (j18, j22)) if j22 - j18 > 288 * 30 else []
        for k, x in (("t8", tr), ("none", tn), ("off15", t15), ("aux", ta)):
            pool[k] += list(ev3(x))
        for i in range(NRAND):
            sh = int(rng.integers(60, len(state) - 60))
            fr = to5(t, dd["t"], np.roll(state, sh))
            e = ev3(run(sig0, fr, (j22, n)))
            rand[i] += (e.sum(), len(e))
        rep["coins"][coin] = {"t8": stats(tr), "none": stats(tn), "off15": stats(t15), "aux": stats(ta)}
        r = rep["coins"][coin]
        print(coin, {k: {q: r[k].get(q) for q in ("n", "win", "x3", "x3_mdd", "x1")} for k in r},
              flush=True)
        del d5
    m = {k: float(np.mean(v)) if v else float("nan") for k, v in pool.items()}
    rmean = rand[:, 0] / np.maximum(rand[:, 1], 1)
    p = float((rmean >= m["t8"]).mean())
    pos = sum(1 for c in rep["coins"].values() if c["t8"].get("x3", -1) > 0)
    win = float((np.array(pool["t8"]) > 0).mean())
    crit = [m["t8"] > 0, pos >= 3, m["t8"] > m["none"], p < 0.05, m["off15"] > 0]
    rep["pool"] = {"ev3": m, "n": {k: len(v) for k, v in pool.items()}, "win3": win,
                   "coins_pos": pos, "rand_median": float(np.median(rmean)), "p": p, "criteria": crit}
    print("POOL", json.dumps(rep["pool"]), flush=True)
    (out / "result.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
