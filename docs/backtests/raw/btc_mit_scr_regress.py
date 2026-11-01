"""mit 조합 SCR 회귀 조건.

사전 등록: docs/backtests/2026-11-01-btc-mit-scr-regress-hypotheses.md
numpy 필요. 실행: PYTHONPATH=src:<numpy 경로>:docs/backtests/raw python <이 파일>
주문과 연결되지 않는다.
"""

from __future__ import annotations

import math
from datetime import date
from types import SimpleNamespace

import btc_mit as mit
import btc_mit_combo as combo
import btc_mit_scr as sc
import numpy as np

from minerva.strategy.indicators import cci, rsi, stoch_fast_k

H = 24
SPAN = 1464


def scr(bars, k):
    """k봉 묶음 SCR. 값은 묶음이 마감된 뒤의 봉에만 준다."""
    if k == 1:
        g = bars
    else:
        g = []
        for i in range(0, len(bars) - k + 1, k):
            w = bars[i : i + k]
            g.append(
                SimpleNamespace(
                    high=max(b.high for b in w), low=min(b.low for b in w), close=w[-1].close
                )
            )
    s, c, r = stoch_fast_k(g), cci(g), rsi([b.close for b in g])
    out = np.full((len(bars), 3), np.nan)
    for j, row in enumerate(zip(s, c, r)):
        v = [np.nan if x is None else x for x in row]
        if k == 1:
            out[j] = v
        else:
            out[(j + 1) * k : (j + 2) * k] = v
    return out


def fit(x, y):
    m, d = np.nanmean(x, 0), np.nanstd(x, 0) + 1e-9
    z = (x - m) / d
    ok = ~np.isnan(z).any(1) & ~np.isnan(y)
    z, t = z[ok], y[ok] - y[ok].mean()
    b = np.linalg.solve(z.T @ z + len(z) * np.eye(z.shape[1]), z.T @ t)
    return m, d, b


def pred(x, m, d, b):
    return np.nan_to_num((x - m) / d) @ b


def run(arr, lng, sht, a, z, d=0.0):
    tr = mit.run(arr, combo.sigs(lng, sht), a, z, None, 0.01)
    return sum(math.log1p(t[6] - d) for t in tr), tr


def main() -> None:
    bars, arr, lng, sht, *_ = sc.setup()
    c = arr[3]
    x = np.hstack([scr(bars, k) for k in (1, 4, 24)])
    y = np.r_[np.log(c[H:] / c[:-H]), np.full(H, np.nan)]
    names = [f"{f}_{k}" for k in ("1h", "4h", "1d") for f in ("S", "C", "R")]
    n = len(bars)
    a = next(i for i, b in enumerate(bars) if b.day >= date(2026, 9, 1))
    yy = y.copy()
    yy[n - H :] = np.nan
    m, d, b = fit(x[a:], yy[a:])
    p = pred(x, m, d, b)
    lg, tr = run(arr, lng & (p > 0), sht & (p < 0), a, n)
    print("fit", {k: round(float(v) * 1e4, 1) for k, v in zip(names, b)})
    print("in", len(tr), sum(t[6] > 0 for t in tr), round(math.exp(lg) - 1, 3), "base", round(math.exp(run(arr, lng, sht, a, n)[0]) - 1, 3))
    s0 = next(i for i, x_ in enumerate(bars) if x_.day >= date(2020, 1, 1))
    res = []
    for i in range(s0, n - 2 * SPAN + 1, SPAN):
        e, f = i + SPAN, min(i + 2 * SPAN, n)
        mm, dd, bb = fit(x[i : e - H], y[i : e - H])
        q = pred(x, mm, dd, bb)
        lq, lw = lng & (q > 0), sht & (q < 0)
        base = run(arr, lng, sht, e, f)[0]
        res.append((run(arr, lq, lw, e, f)[0], run(arr, lq, lw, e, f, 0.001)[0], base, bars[e].day))
    w = [r[0] for r in res]
    print("walk n", len(w), "sum", round(math.expm1(sum(w)), 3), "sum10", round(math.expm1(sum(r[1] for r in res)), 3), "base", round(math.expm1(sum(r[2] for r in res)), 3), "pos", sum(v > 0 for v in w), "beat", sum(r[0] > r[2] for r in res))
    print("last", [(str(r[3]), round(math.expm1(r[0]), 3), round(math.expm1(r[2]), 3)) for r in res[-5:]])
    print("post2025", round(math.expm1(sum(r[0] for r in res if r[3] >= date(2025, 1, 1))), 3), round(math.expm1(sum(r[2] for r in res if r[3] >= date(2025, 1, 1))), 3))
    np.save(sys_path(), p)


def sys_path():
    import sys

    return sys.argv[1]


if __name__ == "__main__":
    main()
