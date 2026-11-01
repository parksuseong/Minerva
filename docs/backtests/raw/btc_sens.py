"""CCI 기간, 손절, RSI 기준선 민감도. 등록: docs/backtests/2026-11-01-btc-stop-size-hypotheses.md 끝

실행: PYTHONPATH=src:docs/backtests/raw python docs/backtests/raw/btc_sens.py
주문과 연결되지 않는다.
"""

from __future__ import annotations

from datetime import date

import btc_swing_label as sl

from minerva.strategy import btc_bt as bt
from minerva.strategy import btc_params as bp
from minerva.strategy.indicators import cci, rsi, stoch_fast_k

COST = 0.0005


def line(tag, bars, lo, hi, k, s20, s25, n, stop):
    r = bt.simulate(bars, lo, hi, k, k, COST, s20, n, stop=stop)
    r10 = bt.simulate(bars, lo, hi, k, k, 0.001, s20, n, stop=stop)
    r25 = bt.simulate(bars, lo, hi, k, k, COST, s25, n, stop=stop)
    print(
        f"{tag}: cagr {r.cagr * 100:+.1f} mdd {r.mdd * 100:.1f} ratio {r.ratio:.2f} n {r.trades} "
        f"win {(r.win_rate or 0) * 100:.1f} cost10 {r10.cagr * 100:+.1f} y25 {r25.final:.3f}",
        flush=True,
    )


def main() -> None:
    for frame in ("1h", "15m"):
        bars = bt.load(frame)
        c = [b.close for b in bars]
        n = len(bars)
        at = lambda d: next(i for i, b in enumerate(bars) if b.day >= d)  # noqa: E731, B023
        s20, s25 = at(date(2020, 1, 1)), at(date(2025, 1, 1))
        st = bp.crosses(stoch_fast_k(bars, 5), 20, 80)
        r14 = rsi(c, 14)
        for cn in (20, 30, 50):
            cc = bp.crosses(cci(bars, cn), -100, 100)
            for lo_, hi_ in ((30, 70), (20, 80)):
                rs = bp.crosses(r14, lo_, hi_)
                fl = [st, cc, rs]
                lo = bp.levels([f[0] for f in fl], 1)
                sh = bp.levels([f[1] for f in fl], 1)
                for stop in (0.01, 0.015, 0.02):
                    tag = f"{frame} 3단 CCI{cn} RSI{hi_}/{lo_} 손절{stop * 100:g}"
                    line(tag, bars, lo, sh, 3, s20, s25, n, stop)
    bars = bt.load("1h")
    n = len(bars)
    at = lambda d: next(i for i, b in enumerate(bars) if b.day >= d)  # noqa: E731, B023
    s20, s25 = at(date(2020, 1, 1)), at(date(2025, 1, 1))
    for ob, os_ in ((70, 30), (80, 20)):
        sl.OB, sl.OS = ob, os_
        lo, hi = sl.rules(bars, sl.confirmed(bars))["S4"]
        for stop in (0.01, 0.015, 0.02):
            line(f"1h S4 RSI{ob}/{os_} 손절{stop * 100:g}", bars, lo, hi, 1, s20, s25, n, stop)


if __name__ == "__main__":
    main()
