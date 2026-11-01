"""BTCUSDT 15분봉 손절 폭 비교 (사전 등록: docs/backtests/2026-11-01-btc-stop-size-hypotheses.md).

실행: PYTHONPATH=src python docs/backtests/raw/btc_stop_size.py
주문과 연결되지 않는다.
"""

from __future__ import annotations

import math
import random
import statistics
from itertools import product

from minerva.strategy import btc_bt as bt
from minerva.strategy import btc_params as bp
from minerva.strategy.indicators import cci, rsi, stoch_fast_k

STOPS = (0.01, 0.015, 0.02, 0.03)
COST = 0.0005
SHIFTS = 200


def main() -> None:
    bars = bt.load("15m")
    closes = [b.close for b in bars]
    n = len(bars)
    st = {k: bp.crosses(stoch_fast_k(bars, k), 20, 80) for k in bp.STOCH}
    cc = {k: bp.crosses(cci(bars, k), -100, 100) for k in bp.CCI}
    rs = {k: bp.crosses(rsi(closes, k), 30, 70) for k in bp.RSI}
    spans = bp.year_spans(bars)
    s20, back = spans[2020][0], bt.split_at(bars)
    days = (bars[-1].day - bars[s20].day).days
    cagr = lambda f: f ** (bt.YEAR_DAYS / days) - 1  # noqa: E731
    base = [st[5], cc[30], rs[14]]
    lo = bp.levels([f[0] for f in base], 1)
    sh = bp.levels([f[1] for f in base], 1)
    keys = [(2, s, q, 0, w) for s, q, w in product(bp.STOCH, bp.CCI, bp.WINDOWS)]
    keys += [(3, s, q, r, w) for s, q, r, w in product(bp.STOCH, bp.CCI, bp.RSI, bp.WINDOWS)]
    for stop in STOPS:
        log: list[float] = []
        r = bt.simulate(bars, lo, sh, 3, 3, COST, s20, n, stop=stop, log=log)
        r10 = bt.simulate(bars, lo, sh, 3, 3, 0.001, s20, n, stop=stop)
        rb = bt.simulate(bars, lo, sh, 3, 3, COST, back, n, stop=stop)
        rf = bt.simulate(bars, lo, sh, 3, 3, COST, 0, back, stop=stop)
        log.sort(reverse=True)
        rest = math.prod(1 + x for x in log[5:])
        years = {
            y: round(bt.simulate(bars[a:z], lo[a:z], sh[a:z], 3, 3, COST, stop=stop).final - 1, 3)
            for y, (a, z) in spans.items()
        }
        rng = random.Random(7)
        vals = []
        for _ in range(SHIFTS):
            d = rng.randrange(2000, n - s20 - 2000)
            ll = lo[:s20] + lo[s20:][d:] + lo[s20:][:d]
            hh = sh[:s20] + sh[s20:][d:] + sh[s20:][:d]
            vals.append(bt.simulate(bars, ll, hh, 3, 3, COST, s20, n, stop=stop).final)
        p = sum(v >= r.final for v in vals) / SHIFTS
        print(
            f"stop {stop}: cagr {r.cagr:.3f} mdd {r.mdd:.3f} ratio {r.ratio:.2f} n {r.trades} "
            f"win {r.win_rate:.3f} cost10 {r10.cagr:.3f} front {rf.cagr:.3f}/{rf.mdd:.3f} "
            f"back {rb.cagr:.3f}/{rb.mdd:.3f} no_top5 {rest:.3f} p {p:.3f} "
            f"median_random {statistics.median(vals):.3f}",
            flush=True,
        )
        print(" years", years, flush=True)
        table: dict = {}
        stress: dict = {}
        for key in keys:
            k, s, q, r_, w = key
            flags = [st[s], cc[q]] + ([rs[r_]] if k == 3 else [])
            a_lo = bp.levels([f[0] for f in flags], w)
            a_sh = bp.levels([f[1] for f in flags], w)
            table[key], stress[key] = {}, {}
            for y, (a, z) in spans.items():
                for book, cost in ((table, COST), (stress, 0.001)):
                    t = bt.simulate(bars[a:z], a_lo[a:z], a_sh[a:z], k, k, cost, stop=stop)
                    book[key][y] = {"final": t.final, "mdd": t.mdd, "trades": t.trades}
        out = bp.walk(table, stress)
        allc = [cagr(math.prod(table[k][y]["final"] for y in bp.WALK)) for k in table]
        print(
            f" walk {cagr(out['final']):.3f} cost10 {cagr(out['stressed']):.3f} "
            f"worst {out['worst_year_mdd']:.3f} positive {sum(v > 0 for v in allc)}/{len(allc)}",
            flush=True,
        )
        print(" chosen", out["chosen"], flush=True)


if __name__ == "__main__":
    main()
