"""ML 규칙 A·B·AS·BS·AS4를 선물 15분봉 가격으로 다시 체결 (같은 예측, 재학습 없음).
문서: 2026-11-07-btc-ml-futures-reprice.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_futures_reprice.py \
      --res ~/.minerva/research/btc-2026-11-04 --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
import numpy as np
from btc_fractal import load5
from btc_futures_compare import load_funding
from btc_ml_edge_cost import rest_sim, summ_net
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_scr_learned import bars

RULES = {"A": (0.90, 4, "s0"), "B": (0.95, 5, "s0"), "AS": (0.90, 4, "mean"),
         "BS": (0.95, 5, "mean"), "AS4": (0.90, 4, "mean")}
DAY = 86_400_000


def futures_on(t: np.ndarray, spot: dict) -> tuple:
    """현물 15분봉 시각에 맞춘 선물 OHLC. 선물에 없는 봉은 현물 값으로 두고 개수를 센다."""
    d = bars(load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser()), 0)
    j = np.searchsorted(d["t"], t)
    ok = (j < len(d["t"])) & (d["t"][np.minimum(j, len(d["t"]) - 1)] == t)
    P = {k: spot[k].copy() for k in "ohlc"}
    for k in "ohlc":
        P[k][ok] = d[k][j[ok]]
    return P, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, out = Path(a.res).expanduser(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(res / "ml15/pred.npz")
    t = z["t"]
    spot = {k: z[k] for k in "ohlc"}
    ps = [z["probs"]] + [np.load(res / f"seeds/pred_s{k}.npz")["probs"] for k in (1, 2, 3, 4)]
    probs = {"s0": z["probs"], "mean": np.mean(ps, 0)}
    fut, ok = futures_on(t, spot)
    ft, fr = load_funding()
    tr4 = trend4h(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), t)
    n = len(t)
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))
    i1y = int(np.searchsorted(t, t[-1] - 365 * DAY))
    periods = {"2023": (i23, i24), "2024~": (i24, n), "1y": (i1y, n)}
    rep: dict = {"futures_missing_2023~": int((~ok[i23:]).sum()), "end": float(t[-1])}
    trades: dict = {}
    for r, (q, k, src) in RULES.items():
        li, si = rule_masks(probs[src], t, q, k)
        if r == "AS4":
            li, si = li & (tr4 > 0), si & (tr4 < 0)
        for pn, (a0, a1) in periods.items():
            yrs = (t[a1 - 1] - t[a0]) / F.YEAR
            for px_name, P in (("spot", spot), ("fut", fut)):
                for lab, sides in (("both", (1, -1)), ("long", (1,)), ("short", (-1,))):
                    tr = rest_sim(P, t, li, si, a0, a1, sides)
                    rep[f"{r} {pn} {px_name} {lab}"] = summ_net(tr, ft, fr, yrs)
                    if lab == "both" and pn in ("2024~", "1y"):
                        trades[f"{r} {pn} {px_name}"] = tr
        # 결과가 달라진 신호 수(2023~ 전체, 양방향)
        s_tr = {x[0]: x for x in rest_sim(spot, t, li, si, i23, n, (1, -1))}
        f_tr = {x[0]: x for x in rest_sim(fut, t, li, si, i23, n, (1, -1))}
        both = set(s_tr) & set(f_tr)
        rep[f"{r} diff"] = {"spot_fills": len(s_tr), "fut_fills": len(f_tr), "same_fill": len(both),
                            "same_fill_diff_outcome": sum(s_tr[x][4] != f_tr[x][4] for x in both)}
    (out / "result.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / "trades.json").write_text(json.dumps(trades, default=float))
    for key, v in rep.items():
        print(key, v, flush=True)


if __name__ == "__main__":
    main()
