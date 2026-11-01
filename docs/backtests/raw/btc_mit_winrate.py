"""mit strategy 고승률 조합 탐색. 사전 등록: docs/backtests/2026-11-01-btc-mit-winrate-hypotheses.md

numpy 필요. 실행: PYTHONPATH=src:<numpy 경로>:docs/backtests/raw python <이 파일> <결과.json>
주문과 연결되지 않는다.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from datetime import date
from itertools import combinations

import btc_mit as mit
import btc_mit_combo as combo
import numpy as np

from minerva.strategy import btc_bt as bt

TPS = (None,) if "--opposite" in sys.argv else (0.003, 0.005, 0.007, 0.01, 0.015, 0.02)
SL = 0.01
IS, OOS = range(2020, 2025), (2025, 2026)


def tally(tr, years):
    by: dict = {}
    for t in tr:
        y = int(years[t[2]])
        b = by.setdefault(y, [0, 0, 0.0, 0.0])
        b[0] += 1
        b[1] += t[6] > 0
        b[2] += math.log1p(t[6])
        b[3] += math.log1p(t[6] - 0.001)
    return by


def agg(by, ys):
    n = sum(by[y][0] for y in ys if y in by)
    w = sum(by[y][1] for y in ys if y in by)
    return (
        n,
        (w / n if n else 0.0),
        sum(by[y][2] for y in ys if y in by),
        sum(by[y][3] for y in ys if y in by),
    )


def report(frame: str, result: dict) -> None:
    bars = bt.load(frame)
    arr = tuple(np.array([getattr(b, k) for b in bars]) for k in ("open", "high", "low", "close"))
    years = np.array([b.day.year for b in bars])
    n = len(bars)
    s20 = next(i for i, b in enumerate(bars) if b.day >= date(2020, 1, 1))
    ind = combo.indicators(bars)
    names = list(ind)
    runs = {
        k: (mit.runs(v[0]) <= combo.EXPIRY, mit.runs(v[1]) <= combo.EXPIRY) for k, v in ind.items()
    }
    rows = []
    for lead in names:
        rest = [x for x in names if x != lead]
        for conf in [()] + [(x,) for x in rest] + list(combinations(rest, 2)):
            lng, sht = ind[lead][0] & runs[lead][0], ind[lead][1] & runs[lead][1]
            for cf in conf:
                lng, sht = lng & ind[cf][0], sht & ind[cf][1]
            sig = combo.sigs(lng, sht)
            for tp in TPS:
                rows.append(((lead, conf, tp), tally(mit.run(arr, sig, s20, n, tp, SL), years)))
        print(frame, lead, len(rows), flush=True)
    pool = [r for r in rows if (a := agg(r[1], IS))[0] >= 500 and a[2] > 0]
    pool.sort(key=lambda r: -agg(r[1], IS)[1])
    top = pool[:20]
    show = lambda r: {  # noqa: E731
        "combo": str(r[0]),
        "is_n_win_ret": [
            agg(r[1], IS)[0],
            round(agg(r[1], IS)[1], 3),
            round(math.exp(agg(r[1], IS)[2]) - 1, 3),
        ],
        "oos_n_win_ret_ret10": [
            agg(r[1], OOS)[0],
            round(agg(r[1], OOS)[1], 3),
            round(math.exp(agg(r[1], OOS)[2]) - 1, 3),
            round(math.exp(agg(r[1], OOS)[3]) - 1, 3),
        ],
    }
    res: dict = {
        "tried": len(rows),
        "is_pos_500": len(pool),
        "is_win60_pos": sum(agg(r[1], IS)[1] >= 0.6 for r in pool),
        "is_win70_pos": sum(agg(r[1], IS)[1] >= 0.7 for r in pool),
        "by_tp_pool": {str(tp): sum(r[0][2] == tp for r in pool) for tp in TPS},
    }
    if top:
        oo = [math.exp(agg(r[1], OOS)[2]) - 1 for r in top]
        be = [(SL + 0.001) / ((r[0][2] or SL) + SL) for r in top]
        res |= {
            "top20_oos_win_median": statistics.median(agg(r[1], OOS)[1] for r in top),
            "top20_breakeven_median": statistics.median(be),
            "top20_oos_ret_median": statistics.median(oo),
            "top20_oos_positive": sum(v > 0 for v in oo),
            "top5": [show(r) for r in top[:5]],
            "best_oos_of_top20": show(max(top, key=lambda r: agg(r[1], OOS)[2])),
        }
    eq, eq10, picks = 0.0, 0.0, {}
    for y in range(2021, 2027):
        past = range(2020, y)
        need = 100 * len(past)
        c = [r for r in rows if (a := agg(r[1], past))[0] >= need and a[2] > 0]
        if not c:
            picks[y] = None
            continue
        b = max(c, key=lambda r: agg(r[1], past)[1])
        g = agg(b[1], [y])
        picks[y] = (str(b[0]), g[0], round(g[1], 3), round(math.exp(g[2]) - 1, 3))
        eq += g[2]
        eq10 += g[3]
    span = (bars[-1].day - date(2021, 1, 1)).days / bt.YEAR_DAYS
    res["walk"] = (math.exp(eq / span) - 1, math.exp(eq10 / span) - 1, picks)
    result[frame] = res
    print(json.dumps({frame: res}, ensure_ascii=False, default=str), flush=True)


def main() -> None:
    out: dict = {}
    for frame in ("1h", "15m"):
        report(frame, out)
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
