"""mit strategy 지표 조합 탐색. 사전 등록: docs/backtests/2026-11-01-btc-mit-combo-hypotheses.md

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
import numpy as np

from minerva.strategy import btc_bt as bt
from minerva.strategy.indicators import dmi, rsi

EXPIRY = 3
EXITS = {"E1": 0.02, "E2": None}
N = {"1h": 24, "15m": 96}


def sma(x, n):
    a = np.cumsum(np.r_[0.0, np.nan_to_num(x)])
    out = np.full(len(x), np.nan)
    out[n - 1 :] = (a[n:] - a[:-n]) / n
    return out


def wma(x, n):
    w = np.arange(1, n + 1, dtype=float)
    out = np.full(len(x), np.nan)
    out[n - 1 :] = np.convolve(x, w[::-1] / w.sum(), "valid")
    return out


def hh(x, n):
    v = np.lib.stride_tricks.sliding_window_view(x, n).max(1)
    return np.r_[np.full(n - 1, np.nan), v]


def ll(x, n):
    v = np.lib.stride_tricks.sliding_window_view(x, n).min(1)
    return np.r_[np.full(n - 1, np.nan), v]


def rma(x, n):
    k, cur, out = 1 / n, x[0], np.zeros(len(x))
    for i, v in enumerate(x):
        cur = cur + (v - cur) * k
        out[i] = cur
    return out


def supertrend(h, l, c, n=10, m=3.0):  # noqa: E741
    tr = np.maximum.reduce([h - l, abs(h - np.r_[c[0], c[:-1]]), abs(l - np.r_[c[0], c[:-1]])])
    atr = rma(tr, n)
    hl2 = (h + l) / 2
    up, dn = hl2 - m * atr, hl2 + m * atr
    fu, fd, t = up[0], dn[0], 1
    out = np.zeros(len(c), int)
    for i in range(1, len(c)):
        fu = max(up[i], fu) if c[i - 1] > fu else up[i]
        fd = min(dn[i], fd) if c[i - 1] < fd else dn[i]
        t = -1 if t == 1 and c[i] < fu else 1 if t == -1 and c[i] > fd else t
        out[i] = t
    return out


def indicators(bars):
    h, l, c = [np.array([getattr(b, k) for b in bars]) for k in ("high", "low", "close")]  # noqa: E741
    n = len(c)
    out: dict[str, tuple] = {}
    out["RF"] = mit.range_filter(c)
    out["RQK"] = mit.rqk(c)
    st = supertrend(h, l, c)
    out["ST"] = (st == 1, st == -1)
    e50, e200 = mit.ema(c, 50), mit.ema(c, 200)
    out["EMA2"] = (e50 > e200, e50 < e200)
    a, b, d = mit.ema(c, 9), mit.ema(c, 21), mit.ema(c, 55)
    out["EMA3"] = ((a > b) & (b > d), (a < b) & (b < d))
    m = mit.ema(c, 12) - mit.ema(c, 26)
    sg = mit.ema(m, 9)
    out["MACD"] = (m > sg, m < sg)
    p, q, x = dmi(bars, 14)
    p, q, x = [np.array([np.nan if v is None else v for v in s], float) for s in (p, q, x)]
    with np.errstate(invalid="ignore"):
        out["DMI"] = ((p > q) & (x > 20), (p < q) & (x > 20))
        tenkan = (hh(h, 9) + ll(l, 9)) / 2
        kijun = (hh(h, 26) + ll(l, 26)) / 2
        sa = np.r_[np.full(25, np.nan), ((tenkan + kijun) / 2)[:-25]]
        sb = np.r_[np.full(25, np.nan), ((hh(h, 52) + ll(l, 52)) / 2)[:-25]]
        top, bot = np.fmax(sa, sb), np.fmin(sa, sb)
        out["ICHI"] = (c > top, c < bot)
        tp = (h + l + c) / 3
        w = np.lib.stride_tricks.sliding_window_view(tp, 20)
        md = np.abs(w - w.mean(1)[:, None]).mean(1)
        cc = np.r_[np.full(19, np.nan), (tp[19:] - w.mean(1)) / (0.015 * np.maximum(md, 1e-12))]
        out["CCI"] = (cc > 0, cc < 0)
        roc = np.r_[np.full(9, np.nan), c[9:] / c[:-9] - 1]
        out["ROC"] = (roc > 0, roc < 0)
        hi, lo = sma(h, 10), sma(l, 10)
        st2, cur = np.zeros(n, int), 0
        for i in range(n):
            cur = 1 if c[i] > hi[i] else -1 if c[i] < lo[i] else cur
            st2[i] = cur
        out["SSL"] = (st2 == 1, st2 == -1)
        hma = wma(2 * wma(c, 27) - wma(c, 55), 7)
        pv = np.r_[np.full(2, np.nan), hma[:-2]]
        out["HULL"] = (hma > pv, hma < pv)
        tr = np.maximum.reduce([h - l, abs(h - np.r_[c[0], c[:-1]]), abs(l - np.r_[c[0], c[:-1]])])
        vp = abs(h - np.r_[l[0], l[:-1]])
        vm = abs(l - np.r_[h[0], h[:-1]])
        vip, vim = sma(vp, 14) / sma(tr, 14), sma(vm, 14) / sma(tr, 14)
        out["VORTEX"] = (vip > vim, vip < vim)
        k = (c - ll(l, 14)) / np.maximum(hh(h, 14) - ll(l, 14), 1e-12) * 100
        ks = sma(np.nan_to_num(k), 3)
        kd = sma(np.nan_to_num(ks), 3)
        out["STOCH"] = (ks > kd, ks < kd)
        r = np.array([np.nan if v is None else v for v in rsi(list(c), 14)], float)
        rm = sma(r, 14)
        out["RSI"] = (r > rm, r < rm)
    return out


def sigs(lng, sht):
    idx = np.nonzero(lng | sht)[0]
    out, state = [], 0
    for i in idx.tolist():
        s = 1 if lng[i] else -1
        if state == -s:
            out.append((i, s))
        state = s
    return out


def report(frame: str, result: dict) -> None:
    bars = bt.load(frame)
    arr = tuple(np.array([getattr(b, k) for b in bars]) for k in ("open", "high", "low", "close"))
    years = np.array([b.day.year for b in bars])
    n = len(bars)
    s20 = next(i for i, b in enumerate(bars) if b.day >= date(2020, 1, 1))
    s25 = next(i for i, b in enumerate(bars) if b.day >= date(2025, 1, 1))
    ind = indicators(bars)
    names = list(ind)
    runs = {k: (mit.runs(v[0]) <= EXPIRY, mit.runs(v[1]) <= EXPIRY) for k, v in ind.items()}
    rows = []
    for lead in names:
        rest = [x for x in names if x != lead]
        sets = [()] + [(x,) for x in rest] + list(combinations(rest, 2))
        for conf in sets:
            lng = ind[lead][0] & runs[lead][0]
            sht = ind[lead][1] & runs[lead][1]
            for cf in conf:
                lng = lng & ind[cf][0]
                sht = sht & ind[cf][1]
            sig = sigs(lng, sht)
            for ex, tp in EXITS.items():
                tr = mit.run(arr, sig, s20, n, tp, 0.01)
                if len(tr) < 100:
                    continue
                by: dict = {}
                by10: dict = {}
                for t in tr:
                    y = int(years[t[2]])
                    by[y] = by.get(y, 0.0) + math.log1p(t[6])
                    by10[y] = by10.get(y, 0.0) + math.log1p(t[6] - 0.001)
                rows.append(((lead, conf, ex), by, by10, tr))
        print(frame, lead, len(rows), flush=True)
    IS, OOS = range(2020, 2025), (2025, 2026)
    sc = lambda b, ys: sum(b.get(y, 0.0) for y in ys)  # noqa: E731
    pool = [r for r in rows if sum(1 for t in r[3] if t[2] < s25) >= 100]
    pool.sort(key=lambda r: -sc(r[1], IS))
    top = pool[:20]
    oos = [math.exp(sc(r[1], OOS)) - 1 for r in top]
    allo = [math.exp(sc(r[1], OOS)) - 1 for r in pool]
    alli = [math.exp(sc(r[1], IS)) - 1 for r in pool]
    days = (bars[-1].day - bars[s20].day).days
    best = top[0]
    got = mit.stats(best[3], years, days)
    res = {
        "tried": len(rows),
        "is_positive": sum(v > 0 for v in alli),
        "oos_positive_all": sum(v > 0 for v in allo),
        "oos_median_all": statistics.median(allo),
        "top20_oos_median": statistics.median(oos),
        "top20_oos_positive": sum(v > 0 for v in oos),
        "best": str(best[0]),
        "best_stats": got,
        "best_is": math.exp(sc(best[1], IS)) - 1,
        "best_oos": math.exp(sc(best[1], OOS)) - 1,
        "best_oos10": math.exp(sc(best[2], OOS)) - 1,
        "best_all10": math.exp(sc(best[2], range(2020, 2027))) - 1,
        "top5": [
            (
                str(r[0]),
                round(math.exp(sc(r[1], IS)) - 1, 3),
                round(math.exp(sc(r[1], OOS)) - 1, 3),
                len(r[3]),
            )
            for r in top[:5]
        ],
    }
    eq, eq10, picks = 0.0, 0.0, {}
    for y in range(2021, 2027):
        past = range(2020, y)
        cand = max(pool, key=lambda r: sc(r[1], past))
        picks[y] = (str(cand[0]), round(math.exp(cand[1].get(y, 0.0)) - 1, 3))
        eq += cand[1].get(y, 0.0)
        eq10 += cand[2].get(y, 0.0)
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
