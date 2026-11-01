"""BTCUSDT 15분봉 스윙 라벨 매매 (사전 등록: docs/backtests/2026-11-01-btc-swing-label-hypotheses.md).

실행: PYTHONPATH=src python docs/backtests/raw/btc_swing_label.py
주문과 연결되지 않는다.
"""

from __future__ import annotations

import math
import random
from datetime import date

from minerva.strategy import btc_bt as bt
from minerva.strategy.indicators import rsi

LEN, OB, OS = 7, 70, 30
COST = 0.0005


def confirmed(bars):
    """(확정 봉, 쪽, 라벨, 가격). 쪽 1은 고점 라벨."""
    r = rsi([b.close for b in bars], LEN)
    out, cur, last = [], None, {1: None, -1: None}
    for i, v in enumerate(r):
        if v is None:
            continue
        b = bars[i]
        side = 1 if v >= OB else -1 if v <= OS else 0
        if side and cur and cur[0] != side:
            s, price = cur
            prev = last[s]
            if s == 1:
                text = "HH" if prev is None or price > prev else "LH"
            else:
                text = "HL" if prev is None or price > prev else "LL"
            last[s] = price
            out.append((i, s, text, price))
            cur = None
        if cur is None and side:
            cur = [side, b.high if side == 1 else b.low]
        elif cur:
            # 원본처럼 극단을 벗어난 뒤에도 고점·저점을 따라간다
            cur[1] = max(cur[1], b.high) if cur[0] == 1 else min(cur[1], b.low)
    return out


def rules(bars, ev):
    n = len(bars)
    pulses = {k: ([0] * n, [0] * n) for k in ("S1", "S2", "S3", "S4")}
    last_high, last_low = None, None
    for i, _, text, _ in ev:
        if text == "HL":
            pulses["S1"][0][i] = 1
            if last_high == "HH":
                pulses["S2"][0][i] = 1
            last_low = text
        elif text == "LL":
            last_low = text
        elif text == "LH":
            pulses["S1"][1][i] = 1
            if last_low == "LL":
                pulses["S2"][1][i] = 1
            last_high = text
        else:
            last_high = text
    hi, lo, t_hi, t_lo, k = None, None, None, None, 0
    above, below = False, False
    for i, b in enumerate(bars):
        while k < len(ev) and ev[k][0] <= i:
            _, s, text, price = ev[k]
            if s == 1:
                hi, t_hi, above = price, text, False
            else:
                lo, t_lo, below = price, text, False
            k += 1
        if hi is not None and b.close > hi and not above:
            above = True
            pulses["S3"][0][i] = 1
            if t_lo == "HL":
                pulses["S4"][0][i] = 1
        if lo is not None and b.close < lo and not below:
            below = True
            pulses["S3"][1][i] = 1
            if t_hi == "LH":
                pulses["S4"][1][i] = 1
    return pulses


def show(name, bars, lo, hi, a, z):
    log = []
    r = bt.simulate(bars, lo, hi, 1, 1, COST, a, z, stop=0.01, log=log)
    r10 = bt.simulate(bars, lo, hi, 1, 1, 0.001, a, z, stop=0.01)
    log.sort(reverse=True)
    rest = math.prod(1 + x for x in log[5:])
    print(
        f"  {name}: final {r.final:.3f} cagr {r.cagr:.3f} mdd {r.mdd:.3f} ratio {r.ratio:.2f} "
        f"n {r.trades} win {r.win_rate} long {r.long_pnl:.3f} short {r.short_pnl:.3f} "
        f"cost10 {r10.final:.3f}/{r10.cagr:.3f} no_top5 {rest:.3f}",
        flush=True,
    )
    return r


def main() -> None:
    bars = bt.load("15m")
    n = len(bars)
    ev = confirmed(bars)
    at = lambda d: next(i for i, b in enumerate(bars) if b.day >= d)  # noqa: E731
    s20, s25, s2m = at(date(2020, 1, 1)), at(date(2025, 1, 1)), at(date(2026, 9, 1))
    for name, (lo, hi) in rules(bars, ev).items():
        print(name, "signals/day", (sum(lo[s20:]) + sum(hi[s20:])) / ((n - s20) / 96), flush=True)
        base = show("2020~", bars, lo, hi, s20, n)
        show("2025~", bars, lo, hi, s25, n)
        show("2m", bars, lo, hi, s2m, n)
        rng = random.Random(7)
        m, vals = n - s20, []
        for _ in range(200):
            d = rng.randrange(2000, m - 2000)
            a = lo[:s20] + lo[s20:][d:] + lo[s20:][:d]
            b = hi[:s20] + hi[s20:][d:] + hi[s20:][:d]
            vals.append(bt.simulate(bars, a, b, 1, 1, COST, s20, n, stop=0.01).final)
        print("  shift p", (sum(v >= base.final for v in vals) + 1) / 201, flush=True)


if __name__ == "__main__":
    main()
