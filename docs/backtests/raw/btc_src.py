"""SRC Adaptive TF. 사전 등록: docs/backtests/2026-11-01-btc-src-adaptive-hypotheses.md

numpy 필요. 실행: PYTHONPATH=src:<numpy 경로> python docs/backtests/raw/btc_src.py <결과.pkl>
주문과 연결되지 않는다.
"""

from __future__ import annotations

import math
import pickle
import random
import sys
from itertools import product

import numpy as np

from minerva.strategy import btc_bt as bt
from minerva.strategy.indicators import dmi, rsi, stoch_fast_k

COST = 0.0005
PREV = (1, 2, 3)
CROSS = (1, 2)
EXITS = (1, 2, 3)
TREND = (False, True)
YEARS = range(2017, 2027)
WALK = range(2020, 2027)
BASE = (2, 1, 2, False)
SHIFTS = 200
FRAMES = {"15m": 96, "30m": 48, "1h": 24}


def nan(x):
    return np.array([np.nan if v is None else v for v in x], float)


def ema(x, n):
    k, out, cur = 2 / (n + 1), np.zeros(len(x)), x[0]
    for i, v in enumerate(x):
        cur = v * k + cur * (1 - k)
        out[i] = cur
    return out


def cci_close(c, n=20):
    w = np.lib.stride_tricks.sliding_window_view(c, n)
    m = w.mean(1)
    d = np.abs(w - m[:, None]).mean(1)
    out = np.full(len(c), np.nan)
    out[n - 1 :] = (c[n - 1 :] - m) / (0.015 * np.maximum(d, 1e-12))
    return out


def up(x, lv):
    return np.r_[False, (x[:-1] <= lv) & (x[1:] > lv)]


def down(x, lv):
    return np.r_[False, (x[:-1] >= lv) & (x[1:] < lv)]


def since(ev, n=12):
    out, last = np.zeros(len(ev), bool), -(10**9)
    for i, e in enumerate(ev):
        if e:
            last = i
        out[i] = i - last <= n
    return out


def cross(a, b):
    return np.r_[False, (a[:-1] <= b[:-1]) & (a[1:] > b[1:])]


def features(bars):
    c = np.array([b.close for b in bars])
    v = np.array([b.volume for b in bars])
    s, q, r = nan(stoch_fast_k(bars, 14)), cci_close(c), nan(rsi(list(c), 14))
    with np.errstate(invalid="ignore"):
        oss = (s <= 20).astype(int) + (q <= -50).astype(int) + (r <= 36).astype(int)
        obs = (s >= 80).astype(int) + (q >= 50).astype(int) + (r >= 84).astype(int)
        cu = up(s, 20).astype(int) + up(q, -50).astype(int) + up(r, 36).astype(int)
        cd = down(s, 80).astype(int) + down(q, 50).astype(int) + down(r, 84).astype(int)
    pdi, mdi, _ = dmi(bars, 14)
    p, m = nan(pdi), nan(mdi)
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    step = np.sign(np.r_[0, np.diff(c)]) * v
    obv = np.cumsum(step)
    osig = ema(obv, 20)
    ma = np.r_[np.full(49, np.nan), np.convolve(c, np.ones(50) / 50, "valid")]
    gold = (
        since(cross(p, m)).astype(int)
        + since(cross(macd, sig)).astype(int)
        + since(cross(obv, osig)).astype(int)
    )
    dead = (
        since(cross(m, p)).astype(int)
        + since(cross(sig, macd)).astype(int)
        + since(cross(osig, obv)).astype(int)
    )
    with np.errstate(invalid="ignore"):
        tl, ts = (gold >= 2) & (c > ma), (dead >= 2) & (c < ma)
    return dict(oss=oss, obs=obs, cu=cu, cd=cd, tl=tl, ts=ts)


def opens(f, prev, crs, trend):
    p_os, p_ob = np.r_[0, f["oss"][:-1]], np.r_[0, f["obs"][:-1]]
    lo = (p_os >= prev) & (f["cu"] >= crs)
    hi = (p_ob >= prev) & (f["cd"] >= crs)
    if trend:
        lo, hi = lo & f["tl"], hi & f["ts"]
    return lo, hi


def sim(arr, f, lo, hi, ex, a, z, mode, cost=COST):
    """mode 'orig': 15%/25% 봉 마감 판정 뒤 다음 시가. 'one': 1% 봉 안 손절선."""
    o, h, l = arr  # noqa: E741
    heat, cool = f["obs"] >= ex, f["oss"] >= ex
    long_out = (heat | hi).tolist()
    short_out = (cool | lo).tolist()
    los, his = lo.tolist(), hi.tolist()
    out = []
    pos, entry, ext, e_i = 0, 0.0, 0.0, 0
    ol, hl, ll = o.tolist(), h.tolist(), l.tolist()

    def close(i, px):
        nonlocal pos
        r = (pos * (px / entry - 1) + 1) * (1 - cost) - (px / entry) * cost - 1
        out.append((i, r, e_i, pos))
        pos = 0

    pending = False
    for i in range(a, z):
        if i > a:
            s = i - 1
            if pos and (pending or (long_out[s] if pos == 1 else short_out[s])):
                close(i, ol[i])
            pending = False
            want = los[s] and not his[s]
            wants = his[s] and not los[s]
            if pos == 0 and (want or wants):
                pos, entry, ext, e_i = (1 if want else -1), ol[i], ol[i], i
        if pos:
            if mode == "one":
                line = entry * (1 - pos * 0.01)
                if (ll[i] <= line) if pos == 1 else (hl[i] >= line):
                    close(i, min(ol[i], line) if pos == 1 else max(ol[i], line))
                continue_stop = False
            else:
                ext = max(ext, hl[i]) if pos == 1 else min(ext, ll[i])
                stop = entry * (1 - pos * 0.15)
                trail = ext * (1 - pos * 0.25)
                if pos == 1:
                    pending = ll[i] <= stop or ll[i] <= trail
                else:
                    pending = hl[i] >= stop or hl[i] >= trail
                continue_stop = pending
            _ = continue_stop
    if pos:
        close(z - 1, ol[z - 1])
    return out


def agg(tr, years):
    by: dict = {}
    n: dict = {}
    for i, r, *_ in tr:
        y = int(years[i])
        by[y] = by.get(y, 0.0) + math.log(max(1 + r, 1e-9))
        n[y] = n.get(y, 0) + 1
    return by, n


def summary(tr, days):
    r = np.array([x[1] for x in tr])
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    mdd = float((eq / peak - 1).min())
    cagr = eq[-1] ** (bt.YEAR_DAYS / days) - 1
    return dict(
        n=len(r),
        win=float((r > 0).mean()),
        cagr=float(cagr),
        mdd=mdd,
        ratio=float(cagr / abs(mdd)),
        final=float(eq[-1]),
        top5=float(np.prod(np.sort(1 + r)[:-5])) if len(r) > 5 else 1.0,
    )


def main(path: str) -> None:
    store = {}
    for frame in FRAMES:
        bars = bt.load(frame)
        arr = tuple(np.array([getattr(b, k) for b in bars]) for k in ("open", "high", "low"))
        f = features(bars)
        years = np.array([b.day.year for b in bars])
        ym = np.array([b.day.year * 100 + b.day.month for b in bars])
        n = len(bars)
        s20 = int(np.argmax(years == 2020))
        s25 = int(np.argmax(years == 2025))
        days = (bars[-1].day - bars[s20].day).days
        d25 = (bars[-1].day - bars[s25].day).days
        print(frame, "hold", [round(x, 3) for x in bt.hold(bars, s20, n)], flush=True)
        for mode in ("orig", "one"):
            table, cost10 = {}, {}
            for key in product(PREV, CROSS, EXITS, TREND):
                pr, cr, ex, tf = key
                lo, hi = opens(f, pr, cr, tf)
                tr = sim(arr, f, lo, hi, ex, 0, n, mode)
                table[key] = agg(tr, years)
                cost10[key] = agg(sim(arr, f, lo, hi, ex, 0, n, mode, 0.001), years)
            eq, eq10, picks, log = 0.0, 0.0, {}, []
            for y in WALK:
                best, top = None, 0.0
                for key, (by, cnt) in table.items():
                    if sum(v for k, v in cnt.items() if k < y) < 30:
                        continue
                    sc = sum(v for k, v in by.items() if k < y)
                    if sc > top:
                        best, top = key, sc
                picks[y] = best
                if best:
                    v = table[best][0].get(y, 0.0)
                    eq += v
                    eq10 += cost10[best][0].get(y, 0.0)
                    log.append(round(math.exp(v) - 1, 3))
                else:
                    log.append(0.0)
            yrs = days / bt.YEAR_DAYS
            allc = [sum(by.get(y, 0.0) for y in WALK) for by, _ in table.values()]
            print(
                f"{frame} {mode} 연도별 선택: 연 {(math.exp(eq / yrs) - 1) * 100:+.1f}% "
                f"비용 0.10% {(math.exp(eq10 / yrs) - 1) * 100:+.1f}% 해별 {log} 고른 값 {picks}",
                flush=True,
            )
            print(
                f"  36칸 2020~ 연수익 중앙값 {(math.exp(float(np.median(allc)) / yrs) - 1) * 100:+.1f}% "
                f"최고 {(math.exp(max(allc) / yrs) - 1) * 100:+.1f}% 플러스 {sum(v > 0 for v in allc)}/36",
                flush=True,
            )
            lo, hi = opens(f, *BASE[:2], BASE[3])
            base = sim(arr, f, lo, hi, BASE[2], s20, n, mode)
            got = summary(base, days)
            c10 = summary(sim(arr, f, lo, hi, BASE[2], s20, n, mode, 0.001), days)
            rec = summary(sim(arr, f, lo, hi, BASE[2], s25, n, mode), d25)
            rng = random.Random(7)
            idx = {v: np.nonzero(ym == v)[0] for v in np.unique(ym)}
            vals = []
            for _ in range(SHIFTS):
                lo2, hi2 = np.zeros(n, bool), np.zeros(n, bool)
                for src, dst in ((lo, lo2), (hi, hi2)):
                    for i in np.nonzero(src[s20:])[0] + s20:
                        m = idx[ym[i]]
                        dst[m[rng.randrange(len(m))]] = True
                vals.append(summary(sim(arr, f, lo2, hi2, BASE[2], s20, n, mode), days)["final"])
            p = (sum(v >= got["final"] for v in vals) + 1) / (SHIFTS + 1)
            print(
                f"  원본값 {got} 비용 0.10% {c10['cagr']:.3f} 2025~ x{rec['final']:.3f} n25 {rec['n']} p {p:.3f}",
                flush=True,
            )
            store[(frame, mode)] = base
    pickle.dump(store, open(path, "wb"))


if __name__ == "__main__":
    main(sys.argv[1])
