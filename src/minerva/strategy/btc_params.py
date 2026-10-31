"""BTCUSDT 15분봉 SCR 기간 값 조사 (사전 등록: docs/backtests/2026-11-01-btc-params-hypotheses.md).

해마다 그 전 성적으로 값을 골라 그해에 적용한다. 주문과 연결되지 않는다.
실행: `python -m minerva.strategy.btc_params`
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from itertools import product

from minerva.strategy import btc_bt as bt
from minerva.strategy.btc_data import OUT
from minerva.strategy.indicators import Num, cci, crossed_down, crossed_up, rsi, stoch_fast_k

STOCH = (5, 9, 14, 21)
CCI = (14, 20, 30, 50)
RSI = (7, 9, 14, 21)
WINDOWS = (1, 3)
YEARS = range(2017, 2027)
WALK = range(2020, 2027)
MIN_TRADES = 30
BASE = (3, 5, 20, 14, 3)  # 현행: 3단, 5/20/14, 3봉 안


def crosses(series: Sequence[Num], low: float, high: float) -> tuple[list[bool], list[bool]]:
    idx = range(len(series))
    return [crossed_up(series, i, low) for i in idx], [crossed_down(series, i, high) for i in idx]


def recent(flags: Sequence[bool], window: int) -> list[bool]:
    """최근 window봉 안에 참이 있었는가."""
    out, last = [], -(10**9)
    for i, f in enumerate(flags):
        if f:
            last = i
        out.append(i - last < window)
    return out


def levels(flags: Sequence[Sequence[bool]], window: int) -> list[int]:
    """앞 지표부터 연속으로 인정된 단계."""
    rows = [recent(f, window) for f in flags]
    out = [0] * len(rows[0])
    for row in rows[::-1]:
        out = [(lv + 1) if on else 0 for lv, on in zip(out, row)]
    return out


def year_spans(bars) -> dict[int, tuple[int, int]]:
    spans: dict[int, list[int]] = {}
    for i, b in enumerate(bars):
        spans.setdefault(b.day.year, [i, i])[1] = i + 1
    return {y: (a, z) for y, (a, z) in spans.items()}


def pick(table: dict, year: int):
    """그 전 해들의 로그 수익 합 1위. 거래가 적거나 합이 0 이하면 관망."""
    best, top = None, 0.0
    for key, rows in table.items():
        past = [r for y, r in rows.items() if y < year]
        if sum(r["trades"] for r in past) < MIN_TRADES:
            continue
        score = sum(math.log(max(r["final"], 1e-9)) for r in past)
        if score > top:
            best, top = key, score
    return best


def walk(table: dict, cost_table: dict) -> dict:
    equity, stressed, worst, chosen = 1.0, 1.0, 0.0, {}
    for y in WALK:
        key = pick(table, y)
        chosen[y] = key
        if key is None:
            continue
        r = table[key][y]
        equity *= r["final"]
        stressed *= cost_table[key][y]["final"]
        worst = min(worst, r["mdd"])
    return {"final": equity, "stressed": stressed, "worst_year_mdd": worst, "chosen": chosen}


def run(root=OUT) -> dict:
    bars = bt.load("15m", root)
    closes = [b.close for b in bars]
    stoch = {n: crosses(stoch_fast_k(bars, n), 20, 80) for n in STOCH}
    ccis = {n: crosses(cci(bars, n), -100, 100) for n in CCI}
    rsis = {n: crosses(rsi(closes, n), 30, 70) for n in RSI}
    print("지표 계산 끝", flush=True)
    spans = year_spans(bars)
    table: dict = {}
    stress: dict = {}
    keys = [(2, s, c, 0, w) for s, c, w in product(STOCH, CCI, WINDOWS)]
    keys += [(3, s, c, r, w) for s, c, r, w in product(STOCH, CCI, RSI, WINDOWS)]
    for key in keys:
        k, s, c, r, w = key
        flags = [stoch[s], ccis[c]] + ([rsis[r]] if k == 3 else [])
        lo = levels([f[0] for f in flags], w)
        sh = levels([f[1] for f in flags], w)
        table[key], stress[key] = {}, {}
        for y, (a, z) in spans.items():
            for book, cost in ((table, 0.0005), (stress, 0.001)):
                t = bt.simulate(bars[a:z], lo[a:z], sh[a:z], k, 1, cost)
                book[key][y] = {"final": t.final, "mdd": t.mdd, "trades": t.trades}
    out = walk(table, stress)
    base = {y: table[BASE][y] for y in WALK}
    start = spans[WALK[0]][0]
    days = (bars[-1].day - bars[start].day).days
    cagr = lambda f: f ** (bt.YEAR_DAYS / days) - 1  # noqa: E731
    top = sorted(
        table,
        key=lambda q: -sum(math.log(max(table[q][y]["final"], 1e-9)) for y in spans),
    )[:5]
    return {
        "walk": {**out, "cagr": cagr(out["final"]), "cagr_cost10": cagr(out["stressed"])},
        "base_cagr": cagr(math.prod(r["final"] for r in base.values())),
        "base_years": {y: r["final"] - 1 for y, r in base.items()},
        "walk_years": {
            y: (table[k][y]["final"] - 1 if k else 0.0) for y, k in out["chosen"].items()
        },
        "after_the_fact_top5": [
            [q, sum(math.log(max(table[q][y]["final"], 1e-9)) for y in spans)] for q in top
        ],
    }


def main() -> None:
    path = OUT / "scr-params.json"
    report = run()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print("wrote", path)


if __name__ == "__main__":
    main()
