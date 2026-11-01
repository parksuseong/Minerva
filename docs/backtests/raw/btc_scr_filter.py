"""SCR 진입 필터. 사전 등록: docs/backtests/2026-11-01-btc-combo-hypotheses.md 끝부분

실행: PYTHONPATH=src:<numpy>:docs/backtests/raw python <이 파일> <combo.json>
주문과 연결되지 않는다.
"""

from __future__ import annotations

import math
import pickle
import sys

import btc_mit as mit
import btc_regress as br
import btc_swing_label as sl
import numpy as np

from minerva.strategy import btc_bt as bt
from minerva.strategy.indicators import tm_states


def show(key, tr, years, days, days25, s25):
    """tr: (청산 봉, 수익). 수익은 비용 0.05%, 비용 0.10%는 왕복 0.10%를 더 뺀 근사."""
    if not tr:
        print(key, "거래 없음", flush=True)
        return
    r = np.array([x[1] for x in tr])
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    mdd = float((eq / peak - 1).min())
    cagr = eq[-1] ** (bt.YEAR_DAYS / days) - 1
    c10 = np.prod(1 + r - 0.001) ** (bt.YEAR_DAYS / days) - 1
    rec = np.prod([1 + x[1] for x in tr if x[0] >= s25])
    rest = np.prod(np.sort(1 + r)[:-5]) if len(r) > 5 else 1.0
    by: dict = {}
    for j, x in tr:
        y = int(years[j])
        by[y] = by.get(y, 1.0) * (1 + x)
    print(
        f"{key}: n {len(r)} win {(r > 0).mean():.3f} cagr {cagr:.3f} mdd {mdd:.3f} "
        f"ratio {cagr / abs(mdd) if mdd else 0:.2f} cost10 {c10:.3f} y25 {rec:.3f} "
        f"top5x {rest:.3f} years+ {sum(v > 1 for v in by.values())}",
        flush=True,
    )


def main(path: str) -> None:
    for frame, tp, rule in (("1h", 0.03, "T"), ("15m", 0.01, "E")):
        d = pickle.load(open(f"{path}.{frame}.{tp}.{rule}.pkl", "rb"))
        bars = bt.load(frame)
        o, h, l = [np.array([getattr(b, k) for b in bars]) for k in ("open", "high", "low")]  # noqa: E741
        st = tm_states(bars)
        lv = {1: [s.long_level for s in st], -1: [s.short_level for s in st]}
        years = np.array([b.day.year for b in bars])
        s20 = int(np.argmax(years == 2020))
        s25 = int(np.argmax(years == 2025))
        days = (bars[-1].day - bars[s20].day).days
        br.MAX_BARS = 500 if frame == "1h" else 2000
        br.STOP = 0.01
        keyed = {(tp, s): br.outcomes(o, h, l, tp, s) for s in (1, -1)}
        for k in (1, 2, 3):
            sig = [x for x in d["sig"] if lv[x[1]][x[0]] >= k]
            tr = [(e, r) for e, r, _ in br.trade(sig, keyed, br.COST, o)]
            show(f"combo {frame} tp{tp} k{k}", tr, years, days, 0, s25)
    bars = bt.load("1h")
    arr = tuple(np.array([getattr(b, k) for b in bars]) for k in ("open", "high", "low", "close"))
    st = tm_states(bars)
    lv = {1: [s.long_level for s in st], -1: [s.short_level for s in st]}
    years = np.array([b.day.year for b in bars])
    n = len(bars)
    s20 = int(np.argmax(years == 2020))
    s25 = int(np.argmax(years == 2025))
    days = (bars[-1].day - bars[s20].day).days
    sig = mit.signals(arr[3])
    for name, tp in (("mit E1", 0.02), ("mit E2", None)):
        base = mit.run(arr, sig, s20, n, tp, 0.01)
        for k in (1, 2, 3):
            tr = [(t[2], t[6]) for t in base if lv[t[1]][t[0]] >= k]
            show(f"{name} k{k}", tr, years, days, 0, s25)
    pulses = sl.rules(bars, sl.confirmed(bars))["S4"]
    for k in (0, 1, 2, 3):
        al = [v >= k for v in lv[1]]
        ash = [v >= k for v in lv[-1]]
        log: list[float] = []
        r = bt.simulate(
            bars,
            pulses[0],
            pulses[1],
            1,
            1,
            0.0005,
            s20,
            n,
            stop=0.01,
            allow_long=al,
            allow_short=ash,
            log=log,
        )
        r10 = bt.simulate(
            bars,
            pulses[0],
            pulses[1],
            1,
            1,
            0.001,
            s20,
            n,
            stop=0.01,
            allow_long=al,
            allow_short=ash,
        )
        r25 = bt.simulate(
            bars,
            pulses[0],
            pulses[1],
            1,
            1,
            0.0005,
            s25,
            n,
            stop=0.01,
            allow_long=al,
            allow_short=ash,
        )
        log.sort(reverse=True)
        print(
            f"S4 k{k}: n {r.trades} win {r.win_rate} cagr {r.cagr:.3f} mdd {r.mdd:.3f} "
            f"ratio {r.ratio:.2f} cost10 {r10.cagr:.3f} y25 {r25.final:.3f} "
            f"top5x {math.prod(1 + x for x in log[5:]):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv[1])
