"""메이저 코인에 SCR 규칙 재확인. 등록: 2026-11-03-btc-majors-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_majors.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import load5
from btc_scr_long_frame import atr_grid
from btc_scr_trend import simulate, stats
from btc_scr_wait_fast import COST, on_grid

ts = lambda y, m=1: datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000  # noqa: E731
DATA = Path("~/.minerva/data").expanduser()
COINS = {"BTC": DATA / "btc/5m.csv", **{c: DATA / f"alt/{c}USDT/5m.csv"
                                        for c in ("ETH", "BNB", "SOL", "XRP")}}
RULES = {"R1": (6, 15, "X1", 0.03), "R2": (6, 15, "X1", 0.01),
         "R3": (48, 120, "X1", 0.06), "R4": (48, 120, "X3", 0.03)}
NRAND = 100


def nets(tr: list) -> np.ndarray:
    return np.array([x[1] for x in tr]) - 2 * COST


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13)
    acc = {r: {"main": [], "shift": [], "aux": [], "rand": np.zeros(NRAND), "coins": {}}
           for r in RULES}
    for coin, path in COINS.items():
        d5 = load5(path)
        t = d5["t"]
        n = len(t) - 1
        j22, j20 = int(np.searchsorted(t, ts(2022))), int(np.searchsorted(t, ts(2020)))
        for rk, (fac, shift, mode, st) in RULES.items():
            A = acc[rk]
            e = on_grid(d5, fac, 0)
            atr = atr_grid(d5, fac, 0)
            tr = simulate(d5, e, e, mode, st, atr, j22, n)
            es = on_grid(d5, fac, shift)
            trs = simulate(d5, es, es, mode, st, atr_grid(d5, fac, shift), j22, n)
            tra = simulate(d5, e, e, mode, st, atr, j20, j22) if j22 > j20 + 1000 else []
            A["main"] += list(nets(tr))
            A["shift"] += list(nets(trs))
            A["aux"] += list(nets(tra))
            A["coins"][coin] = {"main": stats(tr), "aux": stats(tra)}
            k = int(e[0][j22:n].sum() + e[1][j22:n].sum())
            for i in range(NRAND):
                pick = rng.choice(np.arange(j22, n), k, replace=False)
                sd = rng.random(k) < 0.5
                el, es_ = np.zeros(len(t), bool), np.zeros(len(t), bool)
                el[pick[sd]], es_[pick[~sd]] = True, True
                A["rand"][i] += nets(simulate(d5, (el, es_), (es_, el), mode, st, atr, j22, n)).sum()
            m = A["coins"][coin]["main"]
            print(coin, rk, {k_: m.get(k_) for k_ in ("n", "win", "ev", "x1", "x3", "pf")}, flush=True)
        del d5
    rep = {}
    for rk, A in acc.items():
        mn, sh, ax = (np.array(A[k]) for k in ("main", "shift", "aux"))
        pf = lambda x: float(x[x > 0].sum() / max(-x[x <= 0].sum(), 1e-12))  # noqa: E731
        total = float(mn.sum())
        p = float((A["rand"] >= total).mean())
        pos = sum(1 for c in A["coins"].values() if c["main"].get("x1", -1) > 0)
        res = {"n": len(mn), "win": round(float((mn > 0.001).mean() * 100), 1),
               "ev": round(float(mn.mean() * 100), 3), "pf": round(pf(mn), 2),
               "coins_pos": pos, "shift_ev": round(float(sh.mean() * 100), 3),
               "aux_n": len(ax), "aux_ev": round(float(ax.mean() * 100), 3) if len(ax) else None,
               "sum_net": round(total * 100, 1), "rand_sum_median": round(float(np.median(A["rand"]) * 100), 1),
               "p_random": p}
        res["criteria"] = [res["ev"] > 0 and res["pf"] >= 1.2, pos >= 4, res["shift_ev"] > 0,
                           p < 0.05, (res["aux_ev"] or -1) > 0]
        res["coins"] = A["coins"]
        rep[rk] = res
        print(rk, {k: v for k, v in res.items() if k != "coins"}, flush=True)
    (out / "result.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
