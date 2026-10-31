"""SCR v2: BTCUSDT 15분봉, 시장 국면별 조합.

사전 등록: docs/backtests/2026-11-01-btc-scr-v2-hypotheses.md

국면은 마감된 주봉·월봉으로 정하고, 조합은 해마다 그 전 거래만으로 고른다. 주문과 연결되지 않는다.
실행: `python -m minerva.strategy.btc_v2`
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from minerva.strategy.btc_bt import (
    RULES,
    YEAR_DAYS,
    align,
    close_times,
    edges,
    load,
    state_dir,
)
from minerva.strategy.btc_data import OUT
from minerva.strategy.indicators import Bar, tm_states

COST = 0.0005
UP, DOWN, SIDE = 1, -1, 0
REGIMES = (UP, DOWN, SIDE)
NAMES = {UP: "상승장", DOWN: "하락장", SIDE: "횡보장"}
WALK_YEARS = range(2020, 2027)
MIN_TRADES = 30


@dataclass(frozen=True)
class Combo:
    k: int
    c: int
    stop: float | None
    side: str  # both / long / short

    def label(self) -> str:
        stop = "손절1%" if self.stop else "손절없음"
        return f"{self.k}단·청산{self.c}단·{stop}·{self.side}"


COMBOS = [
    Combo(k, c, stop, side)
    for k, c in RULES
    for stop in (None, 0.01)
    for side in ("both", "long", "short")
]


@dataclass
class Run:
    final: float
    cagr: float
    mdd: float
    ratio: float
    trades: list[tuple[int, int, float]]  # 청산 연도, 진입 국면, 거래 수익


def run(
    bars: Sequence[Bar],
    long_levels: Sequence[int],
    short_levels: Sequence[int],
    regimes: Sequence[int],
    plan_for,
    cost: float = COST,
    start: int = 0,
) -> Run:
    """plan_for(연도)는 {국면: Combo}다. 없는 국면은 관망이다."""
    open_l = {k: edges(long_levels, k) for k in (1, 2, 3)}
    open_s = {k: edges(short_levels, k) for k in (1, 2, 3)}
    equity, peak, mdd = 1.0, 1.0, 0.0
    pos, entry, base, before, regime = 0, 0.0, 0.0, 0.0, 0
    combo: Combo | None = None
    log: list[tuple[int, int, float]] = []
    ruined = False

    def value(px: float) -> float:
        return base * (px / entry) if pos == 1 else base * (2 - px / entry)

    def leave(px: float, year: int) -> None:
        nonlocal equity, pos, combo
        after = value(px) - base * (px / entry) * cost
        log.append((year, regime, after / before - 1))
        equity, pos, combo = after, 0, None

    end = len(bars)
    for i in range(start, end):
        b = bars[i]
        y = b.day.year
        if i > start:
            s = i - 1
            now = regimes[s]
            if pos and now != regime:
                leave(b.open, y)
            if pos:
                k, c = combo.k, combo.c
                ls, ss = open_l[k][s], open_s[k][s]
                both = ls and ss
                if pos == 1 and (open_s[c][s] or (ss and not both)):
                    leave(b.open, y)
                elif pos == -1 and (open_l[c][s] or (ls and not both)):
                    leave(b.open, y)
            plan = plan_for(y).get(now)
            if pos == 0 and plan is not None:
                ls = open_l[plan.k][s] and plan.side != "short"
                ss = open_s[plan.k][s] and plan.side != "long"
                raw = open_l[plan.k][s] and open_s[plan.k][s]
                if not raw and (ls or ss):
                    pos = 1 if ls else -1
                    before, base, entry = equity, equity * (1 - cost), b.open
                    equity, combo, regime = base, plan, now
        if pos:
            if combo.stop is not None:
                line = entry * (1 - combo.stop) if pos == 1 else entry * (1 + combo.stop)
                if (b.low <= line) if pos == 1 else (b.high >= line):
                    gapped = (b.open < line) if pos == 1 else (b.open > line)
                    leave(b.open if gapped else line, y)
                    peak = max(peak, equity)
                    mdd = min(mdd, equity / peak - 1)
                    continue
            worst = value(b.low if pos == 1 else b.high)
            if worst <= 0:
                log.append((y, regime, -1.0))
                equity, pos, ruined, mdd = 0.0, 0, True, -1.0
                break
            mdd = min(mdd, worst / peak - 1)
            equity = value(b.close)
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
    if pos:
        leave(bars[end - 1].close, bars[end - 1].day.year)
    days = max((bars[end - 1].day - bars[start].day).days, 1)
    cagr = 0.0 if ruined and equity <= 0 else equity ** (YEAR_DAYS / days) - 1
    if ruined:
        cagr = -1.0
    return Run(equity, cagr, mdd, cagr / abs(mdd) if mdd else 0.0, log)


def score(trades: Sequence[tuple[int, int, float]], before_year: int, regime: int) -> float | None:
    """그 국면에서 before_year 전에 끝난 거래의 로그 수익 합. 표본이 적으면 None."""
    rets = [r for y, g, r in trades if y < before_year and g == regime]
    if len(rets) < MIN_TRADES:
        return None
    return sum(math.log(max(1 + r, 1e-9)) for r in rets)


def pick(table: dict, before_year: int, regime: int) -> Combo | None:
    """표본이 충분하고 합이 플러스인 조합 중 1위. 없으면 관망이다."""
    best, top = None, 0.0
    for combo, trades in table.items():
        s = score(trades, before_year, regime)
        if s is not None and s > top:
            best, top = combo, s
    return best


def yearly_plans(tables: dict[int, dict]) -> dict[int, dict]:
    return {
        y: {g: c for g in REGIMES if (c := pick(tables[g], y, g)) is not None} for y in WALK_YEARS
    }


def without_top(trades: Sequence[tuple[int, int, float]], n: int, days: int) -> float:
    kept = sorted(r for _, _, r in trades)[: max(len(trades) - n, 0)]
    equity = 1.0
    for r in kept:
        equity *= 1 + r
    return equity ** (YEAR_DAYS / max(days, 1)) - 1


def hold(bars: Sequence[Bar], start: int) -> tuple[float, float, float]:
    entry = bars[start].open
    peak, mdd = entry, 0.0
    for b in bars[start:]:
        mdd = min(mdd, b.low / peak - 1)
        peak = max(peak, b.high)
    days = max((bars[-1].day - bars[start].day).days, 1)
    cagr = (bars[-1].close / entry) ** (YEAR_DAYS / days) - 1
    return cagr, mdd, cagr / abs(mdd)


def evaluate(upper: str, root: Path = OUT) -> dict:
    bars = load("15m", root)
    times = close_times("15m", root)
    states = tm_states(bars)
    lo, sh = [s.long_level for s in states], [s.short_level for s in states]
    dirs = [state_dir(s) for s in tm_states(load(upper, root))]
    regimes = align(times, close_times(upper, root), dirs)
    tables: dict[int, dict] = {g: {} for g in REGIMES}
    for g in REGIMES:
        for combo in COMBOS:
            r = run(bars, lo, sh, regimes, lambda y, g=g, combo=combo: {g: combo})
            tables[g][combo] = r.trades
    plans = yearly_plans(tables)
    start = next(i for i, b in enumerate(bars) if b.day.year >= WALK_YEARS[0])
    plan_for = lambda y: plans.get(y, {})  # noqa: E731
    net = run(bars, lo, sh, regimes, plan_for, COST, start)
    high = run(bars, lo, sh, regimes, plan_for, 0.001, start)
    gross = run(bars, lo, sh, regimes, plan_for, 0.0, start)
    days = (bars[-1].day - bars[start].day).days
    share = {NAMES[g]: sum(r == g for r in regimes[start:]) / (len(bars) - start) for g in REGIMES}
    after = {}
    for g in REGIMES:
        c = pick(tables[g], 9999, g)
        after[NAMES[g]] = c.label() if c else "관망"
    return {
        "upper": upper,
        "regime_share": share,
        "hold": hold(bars, start),
        "net": _summary(net),
        "cost10": _summary(high),
        "gross": _summary(gross),
        "without_top5": without_top(net.trades, 5, days),
        "by_year": _by_year(net.trades),
        "plans": {
            str(y): {NAMES[g]: (p[g].label() if g in p else "관망") for g in REGIMES}
            for y, p in plans.items()
        },
        "after_the_fact": after,
    }


def _summary(r: Run) -> dict:
    return {
        "final": r.final,
        "cagr": r.cagr,
        "mdd": r.mdd,
        "ratio": r.ratio,
        "trades": len(r.trades),
    }


def _by_year(trades: Sequence[tuple[int, int, float]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for y, _, r in trades:
        row = out.setdefault(str(y), [1.0, 0])
        row[0] *= 1 + r
        row[1] += 1
    return out


def main() -> None:
    report = {upper: evaluate(upper) for upper in ("1w", "1M")}
    path = OUT / "scr-v2.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
