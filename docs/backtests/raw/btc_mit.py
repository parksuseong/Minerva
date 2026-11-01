"""BTCUSDT mit strategy 기본 설정. 사전 등록: docs/backtests/2026-11-01-btc-mit-hypotheses.md

numpy가 필요하다. 실행: PYTHONPATH=src:<numpy 경로> python docs/backtests/raw/btc_mit.py
주문과 연결되지 않는다.
"""

from __future__ import annotations

import random
from datetime import date

import numpy as np

from minerva.strategy import btc_bt as bt

COST = 0.0005
SHIFTS = 300
EXPIRY = 3


def ema(x: np.ndarray, n: int) -> np.ndarray:
    k, out, cur = 2 / (n + 1), np.zeros(len(x)), x[0]
    for i, v in enumerate(x):
        cur = v * k + cur * (1 - k)
        out[i] = cur
    return out


def range_filter(c: np.ndarray, per: int = 100, mult: float = 3.0):
    d = np.abs(np.r_[0.0, np.diff(c)])
    rng = ema(ema(d, per), per * 2 - 1) * mult
    filt = np.zeros(len(c))
    prev, up, dn = 0.0, 0, 0
    long_, short = np.zeros(len(c), bool), np.zeros(len(c), bool)
    for i, x in enumerate(c):
        r = rng[i]
        f = (prev if x - r < prev else x - r) if x > prev else (prev if x + r > prev else x + r)
        filt[i] = f
        up = up + 1 if f > prev else 0 if f < prev else up
        dn = dn + 1 if f < prev else 0 if f > prev else dn
        moved = i > 0 and x != c[i - 1]
        long_[i] = x > f and moved and up > 0
        short[i] = x < f and moved and dn > 0
        prev = f
    return long_, short


def rqk(c: np.ndarray, h: float = 8.0, r: float = 8.0, window: int = 27):
    i = np.arange(window, dtype=float)
    w = (1 + i**2 / (h**2 * 2 * r)) ** (-r)
    y = np.convolve(c, w, "full")[: len(c)] / w.sum()
    y[: window - 1] = np.nan
    up = np.r_[False, y[:-1] < y[1:]]
    down = np.r_[False, y[:-1] > y[1:]]
    return up, down


def runs(flag: np.ndarray) -> np.ndarray:
    out, n = np.zeros(len(flag), int), 0
    for i, f in enumerate(flag):
        n = n + 1 if f else 0
        out[i] = n
    return out


def signals(c: np.ndarray):
    """번갈아 나오는 (봉, 쪽) 목록."""
    lead_l, lead_s = range_filter(c)
    up, down = rqk(c)
    lng = lead_l & up & (runs(lead_l) <= EXPIRY)
    sht = lead_s & down & (runs(lead_s) <= EXPIRY)
    out, state = [], 0
    for i in np.nonzero(lng | sht)[0].tolist():
        s = 1 if lng[i] else -1
        if state == -s:
            out.append((i, s))
        state = s
    return out


def run(arr, sig, a, z, tp, sl, cost=COST, next_open=False, stop_slip=0.0, tv_path=False):
    """신호 봉 종가 체결. 익절·손절이 없으면 다음 신호에서 뒤집는다. tp None이면 익절 없음."""
    o, h, l, c = arr  # noqa: E741
    out = []
    for k, (i, s) in enumerate(sig):
        if i < a or i >= z - 1:
            continue
        nxt = sig[k + 1][0] if k + 1 < len(sig) else z - 1
        nxt = min(nxt, z - 1)
        start = i + 1
        px0 = c[i]
        if next_open:
            px0 = o[i + 1]
        stop = px0 * (1 - s * sl)
        goal = None if tp is None else px0 * (1 + s * tp)
        seg_h, seg_l = h[start : nxt + 1], l[start : nxt + 1]
        hit_s = seg_l <= stop if s == 1 else seg_h >= stop
        hit_g = np.zeros(len(seg_h), bool)
        if goal is not None:
            hit_g = seg_h >= goal if s == 1 else seg_l <= goal
        any_ = hit_s | hit_g
        last, px, why = nxt, c[nxt] if not next_open else o[min(nxt + 1, z - 1)], "flip"
        if any_.any():
            j = int(any_.argmax())
            b = start + j
            lose = hit_s[j]
            if hit_s[j] and hit_g[j] and tv_path:
                near_high = h[b] - o[b] < o[b] - l[b]
                first_up = near_high
                lose = (s == 1) != first_up
            last = b
            if lose:
                bad = min(o[b], stop) if s == 1 else max(o[b], stop)
                px, why = bad * (1 - s * stop_slip), "stop"
            else:
                px, why = goal, "tp"
        ret = (s * (px / px0 - 1) + 1) * (1 - cost) - (px / px0) * cost - 1
        out.append((i, s, last, float(px0), float(px), why, float(ret)))
    return out


def stats(tr, years, days):
    if not tr:
        return {"n": 0, "final": 1.0, "cagr": 0.0, "mdd": 0.0}
    eq, peak, mdd, by = 1.0, 1.0, 0.0, {}
    for t in tr:
        eq *= 1 + t[6]
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
        y = int(years[t[2]])
        by[y] = by.get(y, 1.0) * (1 + t[6])
    rets = sorted((t[6] for t in tr), reverse=True)
    cagr = eq ** (bt.YEAR_DAYS / days) - 1
    return {
        "n": len(tr),
        "win": sum(r > 0 for r in rets) / len(rets),
        "final": eq,
        "cagr": cagr,
        "mdd": mdd,
        "ratio": cagr / abs(mdd) if mdd else 0.0,
        "no_top5": float(np.prod([1 + r for r in rets[5:]])),
        "long": float(np.sum([t[6] for t in tr if t[1] == 1])),
        "short": float(np.sum([t[6] for t in tr if t[1] == -1])),
        "stops": sum(t[5] == "stop" for t in tr),
        "tps": sum(t[5] == "tp" for t in tr),
        "years": {y: round(v - 1, 3) for y, v in sorted(by.items())},
    }


def report(frame: str) -> None:
    bars = bt.load(frame)
    arr = tuple(np.array([getattr(b, k) for b in bars]) for k in ("open", "high", "low", "close"))
    n = len(bars)
    years = np.array([b.day.year for b in bars])
    at = lambda d: next(i for i, b in enumerate(bars) if b.day >= d)  # noqa: E731
    s20, s25, s8 = at(date(2020, 1, 1)), at(date(2025, 1, 1)), at(date(2026, 8, 1))
    days = (bars[-1].day - bars[s20].day).days
    d25 = (bars[-1].day - bars[s25].day).days
    sig = signals(arr[3])
    print(
        frame,
        "signals/day",
        len([x for x in sig if x[0] >= s20]) / ((n - s20) / (24 if frame == "1h" else 96)),
        flush=True,
    )
    for name, tp in (("E1", 0.02), ("E2", None)):
        base = stats(run(arr, sig, s20, n, tp, 0.01), years, days)
        rng = random.Random(7)
        gap, vals, m = (500 if frame == "1h" else 2000), [], n - s20
        s_in = [x for x in sig if x[0] >= s20]
        for _ in range(SHIFTS):
            d = rng.randrange(gap, m - gap)
            moved = sorted(((i - s20 + d) % m + s20, s) for i, s in s_in)
            vals.append(stats(run(arr, moved, s20, n, tp, 0.01), years, days)["final"])
        p = (sum(v >= base["final"] for v in vals) + 1) / (SHIFTS + 1)
        c10 = stats(run(arr, sig, s20, n, tp, 0.01, cost=0.001), years, days)
        slip = stats(run(arr, sig, s20, n, tp, 0.01, next_open=True, stop_slip=0.001), years, days)
        tv = stats(run(arr, sig, s20, n, tp, 0.01, tv_path=True), years, days)
        r25 = stats(run(arr, sig, s25, n, tp, 0.01), years, d25)
        r8 = stats(run(arr, sig, s8, n, tp, 0.01), years, 92)
        print(f"{frame} {name}: {base}", flush=True)
        print(
            f"  cost10 {c10['cagr']:.3f} slip {slip['cagr']:.3f} tv {tv['cagr']:.3f}/{tv.get('win')} "
            f"y25 {r25['final']:.3f} n25 {r25['n']} win25 {r25.get('win')} "
            f"3m {r8['final']:.3f} n8 {r8['n']} win8 {r8.get('win')} p {p:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    report("1h")
    report("15m")
