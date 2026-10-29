"""숏 규칙 백테스트 (기초자산 하락에 베팅 = 인버스 매수의 근거).

체결: 신호 봉 마감 판정 → 다음 봉 시가 진입.
청산(먼저 오는 것): 고가가 손절가(진입 × (1 + stop)) 도달 → 그 가격 /
Stoch fast 20 상향 돌파(과매도 탈출) → 그 봉 종가 / 최대 보유 봉수 → 종가.
수익 = 진입가 / 청산가 − 1 (숏). 비교 기준은 같은 보유기간의 무조건 숏 평균.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

from minerva.strategy.backtest import Trade
from minerva.strategy.indicators import Bar, crossed_up, stoch_fast_k


@dataclass(frozen=True)
class ShortRule:
    hold: int = 20
    stop: float = 0.07


Signal = Callable[[int], bool]


def run_short(bars: Sequence[Bar], flags: Sequence[bool], rule: ShortRule) -> list[Trade]:
    stoch = stoch_fast_k(bars)
    trades: list[Trade] = []
    i = 0
    while i < len(bars) - 1:
        if not flags[i]:
            i += 1
            continue
        e = i + 1
        entry = bars[e].open
        stop_px = entry * (1 + rule.stop)
        high = entry
        j = e
        while True:
            high = max(high, bars[j].high)
            if bars[j].high >= stop_px:
                exit_px, reason = max(stop_px, bars[j].open), "stop"
                break
            if j > e and crossed_up(stoch, j, 20.0):
                exit_px, reason = bars[j].close, "take"
                break
            if j - e + 1 >= rule.hold:
                exit_px, reason = bars[j].close, "time"
                break
            if j == len(bars) - 1:
                exit_px, reason = bars[j].close, "open"
                break
            j += 1
        # Trade.ret = exit/entry − 1 이므로 숏은 가격을 뒤집어 저장한다
        trades.append(
            Trade(bars[e].day, bars[j].day, exit_px, entry, j - e + 1, entry / high - 1, reason)
        )
        i = j + 1
    return trades


def short_baseline(bars: Sequence[Bar], hold: int) -> float | None:
    rets = [
        bars[i + 1].open / bars[i + hold].close - 1
        for i in range(len(bars) - hold - 1)
        if bars[i + hold].close > 0
    ]
    return statistics.mean(rets) if rets else None


def short_stats(trades: Sequence[Trade], bars: Sequence[Bar]) -> dict:
    closed = [t for t in trades if t.reason != "open"]
    if not closed:
        return {"trades": 0}
    rets = [t.ret for t in closed]
    hold = max(1, round(statistics.median(t.bars for t in closed)))
    base = short_baseline(bars, hold)
    return {
        "trades": len(closed),
        "win_rate": round(sum(r > 0 for r in rets) / len(rets), 3),
        "mean": round(statistics.mean(rets), 4),
        "worst": round(min(rets), 4),
        "median_bars": hold,
        "edge": None if base is None else round(statistics.mean(rets) - base, 4),
    }


def evaluate_short(
    bars: Sequence[Bar], flags: Sequence[bool], rule: ShortRule, split: float = 0.7
) -> dict:
    trades = run_short(bars, flags, rule)
    cut: date = bars[int(len(bars) * split)].day
    return {
        "all": short_stats(trades, bars),
        "in_sample": short_stats(
            [t for t in trades if t.entry_day < cut], [b for b in bars if b.day < cut]
        ),
        "out_of_sample": short_stats(
            [t for t in trades if t.entry_day >= cut], [b for b in bars if b.day >= cut]
        ),
    }


def short_timing_pvalue(
    bars: Sequence[Bar], flags: Sequence[bool], rule: ShortRule, n_perm: int = 300, seed: int = 7
) -> float | None:
    """숏 진입 타이밍이 무작위보다 나은지 (p). 롱의 `random_timing_pvalue`와 같은 방식:
    같은 횟수의 무작위 진입으로 같은 청산 규칙을 돌려 평균 수익이 규칙 이상인 비율."""
    import random

    actual = [t.ret for t in run_short(bars, flags, rule) if t.reason != "open"]
    k = sum(flags)
    if len(actual) < 2 or k < 2:
        return None
    target = statistics.mean(actual)
    rng = random.Random(seed)
    slots = range(len(bars) - 1)
    hits = 0
    for _ in range(n_perm):
        pick = set(rng.sample(slots, k))
        fake = [i in pick for i in range(len(bars))]
        rets = [t.ret for t in run_short(bars, fake, rule) if t.reason != "open"]
        if rets and statistics.mean(rets) >= target:
            hits += 1
    return (hits + 1) / (n_perm + 1)
