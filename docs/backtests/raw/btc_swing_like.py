"""BTCUSDT 스윙 근사 매매. 사전 등록: docs/backtests/2026-11-01-btc-swing-like-hypotheses.md

실행: PYTHONPATH=src python docs/backtests/raw/btc_swing_like.py
주문과 연결되지 않는다.
"""

from __future__ import annotations

import math
import random
from datetime import date

from minerva.strategy import btc_bt as bt
from minerva.strategy.btc_features import rolling_mean, true_range
from minerva.strategy.indicators import dmi, rsi

LEN, OB, OS = 7, 70, 30
COST = 0.0005
ATR_N, ATR_K = 14, 2.0
SHIFTS = 300


def a1(bars):
    r = rsi([b.close for b in bars], LEN)
    n = len(bars)
    lo, hi = [0] * n, [0] * n
    for i in range(1, n):
        p, v = r[i - 1], r[i]
        if p is None or v is None:
            continue
        if p >= OB > v:
            hi[i] = 1
        if p <= OS < v:
            lo[i] = 1
    return lo, hi


def a2(bars):
    atr = rolling_mean(true_range(bars), ATR_N)
    n = len(bars)
    lo, hi = [0] * n, [0] * n
    leg, ext = 0, bars[0].close  # leg 1: 고점 찾는 중
    for i, b in enumerate(bars):
        if atr[i] is None:
            continue
        gap = atr[i] * ATR_K
        if leg >= 0 and b.high >= ext:
            ext = b.high
            leg = 1
        if leg <= 0 and b.low <= ext:
            ext = b.low
            leg = -1
        if leg == 1 and b.close <= ext - gap:
            hi[i], leg, ext = 1, -1, b.low
        elif leg == -1 and b.close >= ext + gap:
            lo[i], leg, ext = 1, 1, b.high
    return lo, hi


def a3(bars):
    r = rsi([b.close for b in bars], LEN)
    n = len(bars)
    lo, hi = [0] * n, [0] * n
    run = {1: None, -1: None}
    prev = {1: None, -1: None}
    for i in range(1, n):
        p, v = r[i - 1], r[i]
        if p is None or v is None:
            continue
        for s, on in ((1, v >= OB), (-1, v <= OS)):
            if on:
                px = bars[i].high if s == 1 else bars[i].low
                if run[s] is None:
                    run[s] = [px, v]
                elif s == 1:
                    run[s] = [max(run[s][0], px), max(run[s][1], v)]
                else:
                    run[s] = [min(run[s][0], px), min(run[s][1], v)]
        if p >= OB > v and run[1]:
            q = prev[1]
            if q and run[1][0] > q[0] and run[1][1] < q[1]:
                hi[i] = 1
            prev[1], run[1] = run[1], None
        if p <= OS < v and run[-1]:
            q = prev[-1]
            if q and run[-1][0] < q[0] and run[-1][1] > q[1]:
                lo[i] = 1
            prev[-1], run[-1] = run[-1], None
    return lo, hi


def a4(bars):
    lo, hi = a1(bars)
    adx = dmi(bars, 14)[2]
    ok = [v is not None and v < 25 for v in adx]
    return [a and b for a, b in zip(lo, ok)], [a and b for a, b in zip(hi, ok)]


RULES = {"A1": a1, "A2": a2, "A3": a3, "A4": a4}


def report(frame: str) -> None:
    bars = bt.load(frame)
    n = len(bars)
    at = lambda d: next(i for i, b in enumerate(bars) if b.day >= d)  # noqa: E731
    s20, s25, s8 = at(date(2020, 1, 1)), at(date(2025, 1, 1)), at(date(2026, 8, 1))
    gap = 500 if frame == "1h" else 2000
    print(frame, "hold", [round(x, 3) for x in bt.hold(bars, s20, n)], flush=True)
    for name, fn in RULES.items():
        lo, hi = fn(bars)
        log: list[float] = []
        r = bt.simulate(bars, lo, hi, 1, 1, COST, s20, n, stop=0.01, log=log)
        r10 = bt.simulate(bars, lo, hi, 1, 1, 0.001, s20, n, stop=0.01)
        r25 = bt.simulate(bars, lo, hi, 1, 1, COST, s25, n, stop=0.01)
        r8 = bt.simulate(bars, lo, hi, 1, 1, COST, s8, n, stop=0.01)
        free = bt.simulate(bars, lo, hi, 1, 1, COST, s20, n)
        log.sort(reverse=True)
        rest = math.prod(1 + x for x in log[5:])
        rng = random.Random(7)
        m, vals = n - s20, []
        for _ in range(SHIFTS):
            d = rng.randrange(gap, m - gap)
            a = lo[:s20] + lo[s20:][d:] + lo[s20:][:d]
            b = hi[:s20] + hi[s20:][d:] + hi[s20:][:d]
            vals.append(bt.simulate(bars, a, b, 1, 1, COST, s20, n, stop=0.01).final)
        p = (sum(v >= r.final for v in vals) + 1) / (SHIFTS + 1)
        print(
            f"{frame} {name}: cagr {r.cagr:.3f} mdd {r.mdd:.3f} ratio {r.ratio:.2f} n {r.trades} "
            f"win {r.win_rate:.3f} L {r.long_pnl:.2f} S {r.short_pnl:.2f} cost10 {r10.cagr:.3f} "
            f"no_top5 {rest:.3f} p {p:.3f} y25 x{r25.final:.3f} chart x{r8.final:.3f} "
            f"n8 {r8.trades} win8 {r8.win_rate} nostop {free.cagr:.3f}/{free.mdd:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    report("1h")
    report("15m")
