"""이벤트 분석: 신호가 난 날 이후 N봉 수익을 무조건 기대수익과 비교한다.

롱/숏 백테스트와 달리 청산 규칙 없이 "이 신호 뒤에 시장이 어떻게 움직였나"만 본다.
진입은 신호 다음 봉 시가, 수익은 h봉 뒤 종가 기준. 같은 이벤트가 겹치지 않게
신호 사이에 최소 간격(gap)을 둔다.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from datetime import date

from minerva.strategy.indicators import Bar


def forward_returns(bars: Sequence[Bar], i: int, horizons: Sequence[int]) -> dict[int, float]:
    out = {}
    if i + 1 >= len(bars) or bars[i + 1].open <= 0:
        return out
    entry = bars[i + 1].open
    for h in horizons:
        if i + h < len(bars):
            out[h] = bars[i + h].close / entry - 1
    return out


def max_rise(bars: Sequence[Bar], i: int, h: int) -> float | None:
    """신호 봉 종가 대비 이후 h봉 안의 최고가 상승률."""
    if i + h >= len(bars):
        return None
    hi = max(b.high for b in bars[i + 1 : i + h + 1])
    return hi / bars[i].close - 1


def dedupe(flags: Sequence[bool], gap: int) -> list[int]:
    idx, last = [], -(10**9)
    for i, f in enumerate(flags):
        if f and i - last >= gap:
            idx.append(i)
            last = i
    return idx


def summarize(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "hit": round(sum(v > 0 for v in values) / len(values), 3),
        "worst": round(min(values), 4),
    }


def study(
    bars: Sequence[Bar],
    flags: Sequence[bool],
    horizons: Sequence[int] = (5, 20, 60),
    gap: int = 5,
    split: float = 0.7,
) -> dict:
    """신호별 h봉 수익 요약 + 같은 h의 무조건 평균 + 표본 내/밖."""
    events = dedupe(flags, gap)
    cut: date = bars[int(len(bars) * split)].day
    out: dict = {"events": len(events), "split_at": cut.isoformat(), "by_h": {}}
    for h in horizons:
        rets = [(bars[i].day, forward_returns(bars, i, [h]).get(h)) for i in events]
        rets = [(d, r) for d, r in rets if r is not None]
        base = [forward_returns(bars, i, [h]).get(h) for i in range(len(bars))]
        base = [r for r in base if r is not None]
        base_mean = statistics.mean(base) if base else 0.0
        all_s = summarize([r for _, r in rets])
        ins = summarize([r for d, r in rets if d < cut])
        oos = summarize([r for d, r in rets if d >= cut])
        for s in (all_s, ins, oos):
            if s.get("n"):
                s["edge"] = round(s["mean"] - base_mean, 4)
        out["by_h"][h] = {
            "baseline": round(base_mean, 4),
            "all": all_s,
            "in": ins,
            "out": oos,
        }
    return out


def align(daily: Sequence[Bar], series: dict[date, float]) -> list[float | None]:
    """날짜별 값을 일봉 순서에 맞춘다 (없는 날은 직전 값)."""
    out: list[float | None] = []
    last = None
    for b in daily:
        last = series.get(b.day, last)
        out.append(last)
    return out


Flag = Callable[[int], bool]
