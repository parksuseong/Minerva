"""BTCUSDT 15분봉 RSI Swing 지표 (사전 등록: docs/backtests/2026-11-01-btc-rsi-swing-hypotheses.md).

실행: PYTHONPATH=src python docs/backtests/raw/btc_rsi_swing.py
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


def signals(bars):
    """(매수 신호, 매도 신호, 라벨 목록). 라벨은 (봉, 쪽, 가격)으로 사후 확정 위치다."""
    r = rsi([b.close for b in bars], LEN)
    buy, sell = [0] * len(bars), [0] * len(bars)
    state, swings = 0, []
    cur = None  # [쪽, 봉, 가격]: 아직 움직이는 라벨
    for i, v in enumerate(r):
        if v is None:
            continue
        ob, os_ = v >= OB, v <= OS
        if state == 2 and ob:
            buy[i] = 1
        if state == 1 and os_:
            sell[i] = 1
        if ob:
            if cur and cur[0] == 1:
                if bars[i].high >= cur[2]:
                    cur = [1, i, bars[i].high]
            else:
                if cur:
                    swings.append(tuple(cur))
                cur = [1, i, bars[i].high]
            state = 1
        if os_:
            if cur and cur[0] == -1:
                if bars[i].low <= cur[2]:
                    cur = [-1, i, bars[i].low]
            else:
                if cur:
                    swings.append(tuple(cur))
                cur = [-1, i, bars[i].low]
            state = 2
        if state == 1 and cur and cur[0] == 1 and bars[i].high >= cur[2]:
            cur = [1, i, bars[i].high]
        if state == 2 and cur and cur[0] == -1 and bars[i].low <= cur[2]:
            cur = [-1, i, bars[i].low]
    return buy, sell, swings


def show(name, bars, buy, sell, a, z, stop=0.01):
    log = []
    r = bt.simulate(bars, buy, sell, 1, 1, COST, a, z, stop=stop, log=log)
    r10 = bt.simulate(bars, buy, sell, 1, 1, 0.001, a, z, stop=stop)
    log.sort(reverse=True)
    rest = math.prod(1 + x for x in log[5:])
    print(
        f"{name}: final {r.final:.3f} cagr {r.cagr:.3f} mdd {r.mdd:.3f} ratio {r.ratio:.2f} "
        f"n {r.trades} win {r.win_rate} long {r.long_pnl:.3f} short {r.short_pnl:.3f} "
        f"cost10 {r10.final:.3f}/{r10.cagr:.3f} no_top5 {rest:.3f}",
        flush=True,
    )
    return r


def hindsight(swings, bars, a, z):
    """라벨에서 라벨까지 다 먹었다고 치면(왕복 0.10%)."""
    x = [s for s in swings if a <= s[1] < z]
    eq = 1.0
    for p, q in zip(x, x[1:]):
        eq *= 1 - (q[2] / p[2] - 1) * p[0] - 0.001  # p[0]=1은 고점 라벨이라 숏
    return eq, len(x) - 1


def main() -> None:
    bars = bt.load("15m")
    n = len(bars)
    buy, sell, swings = signals(bars)
    at = lambda d: next(i for i, b in enumerate(bars) if b.day >= d)  # noqa: E731
    s20, s25, s2m = at(date(2020, 1, 1)), at(date(2025, 1, 1)), at(date(2026, 9, 1))
    base = show("2020~", bars, buy, sell, s20, n)
    show("2020~ no stop", bars, buy, sell, s20, n, None)
    show("2025~", bars, buy, sell, s25, n)
    show("2m", bars, buy, sell, s2m, n)
    show("2m no stop", bars, buy, sell, s2m, n, None)
    print("signals/day", (sum(buy[s20:]) + sum(sell[s20:])) / ((n - s20) / 96))
    print("hindsight 2m", hindsight(swings, bars, s2m, n), "2020~", hindsight(swings, bars, s20, n))
    rng = random.Random(7)
    m = n - s20
    vals = []
    for _ in range(300):
        d = rng.randrange(2000, m - 2000)
        b = buy[:s20] + buy[s20:][d:] + buy[s20:][:d]
        s = sell[:s20] + sell[s20:][d:] + sell[s20:][:d]
        vals.append(bt.simulate(bars, b, s, 1, 1, COST, s20, n, stop=0.01).final)
    print("shift p", (sum(v >= base.final for v in vals) + 1) / 301, flush=True)


if __name__ == "__main__":
    main()
