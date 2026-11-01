"""mit 조합에 RSI·SCR 조건 추가.

사전 등록: docs/backtests/2026-11-01-btc-mit-scr-filter-hypotheses.md

numpy 필요. 실행: PYTHONPATH=src:<numpy 경로>:docs/backtests/raw python <이 파일>
주문과 연결되지 않는다.
"""

from __future__ import annotations

import math
from datetime import date

import btc_mit as mit
import btc_mit_combo as combo
import numpy as np

from minerva.strategy import btc_bt as bt
from minerva.strategy.indicators import cci, rsi, stoch_fast_k


def nums(s):
    return np.array([np.nan if v is None else v for v in s], float)


def setup():
    bars = bt.load("1h")
    arr = tuple(np.array([getattr(b, k) for b in bars]) for k in ("open", "high", "low", "close"))
    ind = combo.indicators(bars)
    runs = {
        k: (mit.runs(v[0]) <= combo.EXPIRY, mit.runs(v[1]) <= combo.EXPIRY) for k, v in ind.items()
    }
    lng, sht = ind["DMI"][0] & runs["DMI"][0], ind["DMI"][1] & runs["DMI"][1]
    for cf in ("RQK", "ICHI"):
        lng, sht = lng & ind[cf][0], sht & ind[cf][1]
    s, c, r = nums(stoch_fast_k(bars)), nums(cci(bars)), nums(rsi([b.close for b in bars]))
    return bars, arr, lng, sht, s, c, r


def conds(s, c, r):
    with np.errstate(invalid="ignore"):
        v2 = (s < 80) & (c < 100) & (r < 70), (s > 20) & (c > -100) & (r > 30)
        v3 = (c > 0) & (r > 50), (c < 0) & (r < 50)
        return {
            "B": (np.ones(len(s), bool), np.ones(len(s), bool)),
            "V1": (r < 70, r > 30),
            "V2": v2,
            "V3": v3,
            "V4": (v2[0] & v3[0], v2[1] & v3[1]),
        }


def main() -> None:
    bars, arr, lng, sht, s, c, r = setup()
    n = len(bars)
    s20 = next(i for i, b in enumerate(bars) if b.day >= date(2020, 1, 1))
    s25 = next(i for i, b in enumerate(bars) if b.day >= date(2025, 1, 1))
    for name, (a, b) in conds(s, c, r).items():
        sig = combo.sigs(lng & a, sht & b)
        for tp in (None, 0.01, 0.02):
            tr = mit.run(arr, sig, s20, n, tp, 0.01)
            i = [t for t in tr if t[0] < s25]
            o = [t for t in tr if t[0] >= s25]
            g = lambda ts, d: math.exp(sum(math.log1p(t[6] - d) for t in ts)) - 1  # noqa: E731
            w = lambda ts: sum(t[6] > 0 for t in ts) / len(ts) if ts else 0.0  # noqa: E731
            print(
                f"{name} tp={tp} n={len(i)}/{len(o)} win={w(i):.3f}/{w(o):.3f} "
                f"is={g(i, 0):+.3f} oos={g(o, 0):+.3f} oos10={g(o, 0.001):+.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
