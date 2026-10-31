"""피보나치·엘리어트 규칙 검증 (`docs/backtests/2026-10-31-fib-elliott-hypotheses.md`).

스윙은 확정된 날부터만 쓴다. 고점은 종가가 10% 내려온 날에야 고점인 줄 알 수 있어서,
사후에 그린 파동을 쓰면 미래 정보가 들어간다.

실행: `python -m minerva.strategy.patterns [--daily 분석파일]`
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from minerva.strategy import daily as dm
from minerva.strategy.backtest import Trade, filter_mask, signals, simulate
from minerva.strategy.indicators import Bar, sma, stoch_fast_k
from minerva.strategy.report import fetch_yahoo

SWING = 0.10
FIB_LOW, FIB_HIGH = 0.382, 0.618
WAVE2_MAX = 0.786
AVOID_DAYS = 40
START = date(2010, 1, 1)
SPLIT = date(2021, 10, 29)
N_PERM = 2000
P_LIMIT = 0.05 / 4


@dataclass(frozen=True)
class Pivot:
    kind: str  # H 또는 L
    price: float
    index: int
    confirmed: int  # 이 스윙을 처음 알 수 있는 봉


def zigzag(closes: Sequence[float], pct: float = SWING) -> list[Pivot]:
    out: list[Pivot] = []
    if not closes:
        return out
    hi, lo = 0, 0
    up: bool | None = None
    for i, c in enumerate(closes):
        if up is not True and c >= closes[lo] * (1 + pct):
            out.append(Pivot("L", closes[lo], lo, i))
            up, hi = True, i
        elif up is not False and c <= closes[hi] * (1 - pct):
            out.append(Pivot("H", closes[hi], hi, i))
            up, lo = False, i
        if up is not False and c > closes[hi]:
            hi = i
        if up is not True and c < closes[lo]:
            lo = i
    return out


def _known(pivots: Sequence[Pivot], n: int) -> list[list[Pivot]]:
    """봉마다 그날까지 확정된 스윙 목록."""
    out: list[list[Pivot]] = []
    seen: list[Pivot] = []
    k = 0
    for i in range(n):
        while k < len(pivots) and pivots[k].confirmed <= i:
            seen = [*seen[-5:], pivots[k]]
            k += 1
        out.append(seen)
    return out


def fib_zones(bars: Sequence[Bar]) -> tuple[list[bool], list[bool]]:
    """(그날 종가가 38.2~61.8% 구간인가, 그 스윙에서 처음 들어온 날인가)."""
    closes = [b.close for b in bars]
    known = _known(zigzag(closes), len(bars))
    inside, first = [False] * len(bars), [False] * len(bars)
    seen: set[int] = set()
    for i, piv in enumerate(known):
        swing = next(
            (
                (a, b)
                for a, b in zip(reversed(piv[:-1]), reversed(piv))
                if a.kind == "L" and b.kind == "H"
            ),
            None,
        )
        if swing is None:
            continue
        low, high = swing
        span = high.price - low.price
        if high.price - FIB_HIGH * span <= closes[i] <= high.price - FIB_LOW * span:
            inside[i] = True
            if high.index not in seen:
                seen.add(high.index)
                first[i] = True
    return inside, first


def wave3_breakouts(bars: Sequence[Bar]) -> list[bool]:
    closes = [b.close for b in bars]
    known = _known(zigzag(closes), len(bars))
    out = [False] * len(bars)
    fired: set[int] = set()
    for i, piv in enumerate(known):
        if len(piv) < 3 or [p.kind for p in piv[-3:]] != ["L", "H", "L"]:
            continue
        p0, p1, p2 = piv[-3:]
        if p2.index in fired or p2.price <= p0.price:
            continue
        retrace = (p1.price - p2.price) / (p1.price - p0.price)
        if FIB_LOW <= retrace <= WAVE2_MAX and closes[i] > p1.price:
            fired.add(p2.index)
            out[i] = True
    return out


def impulse_done(piv: Sequence[Pivot]) -> bool:
    """마지막 여섯 스윙이 충격파 규칙을 만족하는가."""
    if len(piv) < 6 or [p.kind for p in piv[-6:]] != ["L", "H", "L", "H", "L", "H"]:
        return False
    p = [x.price for x in piv[-6:]]
    w1, w3, w5 = p[1] - p[0], p[3] - p[2], p[5] - p[4]
    return p[2] > p[0] and p[3] > p[1] and p[4] > p[1] and p[5] > p[3] and w3 >= min(w1, w5)


def wave5_blocks(bars: Sequence[Bar]) -> list[bool]:
    known = _known(zigzag([b.close for b in bars]), len(bars))
    out = [False] * len(bars)
    done: set[int] = set()
    for i, piv in enumerate(known):
        if piv and piv[-1].confirmed == i and piv[-1].index not in done and impulse_done(piv):
            done.add(piv[-1].index)
            for j in range(i + 1, min(i + 1 + AVOID_DAYS, len(bars))):
                out[j] = True
    return out


@dataclass
class Series:
    bars: list[Bar]
    cost: float
    mask: list[bool]
    tm: list[bool]
    stoch: list
    trail: list
    active: list[bool]


def prepare(symbol: str, vix: list[Bar], cost: float) -> Series | None:
    bars = dm.completed(fetch_yahoo(symbol, "max", ttl=10**9), "KR")
    if len(bars) < 300:
        return None
    ctx = dm._ctx(vix, bars)
    rule = dm.TM_RULE
    return Series(
        bars,
        cost,
        filter_mask(bars, rule, ctx),
        signals(bars, rule, ctx),
        stoch_fast_k(bars),
        sma([b.close for b in bars], rule.trail_ma),
        [b.day >= START for b in bars],
    )


def trades_of(s: Series, sig: Sequence[bool]) -> list[tuple[Trade, float]]:
    live = [x and a for x, a in zip(sig, s.active)]
    done = simulate(s.bars, dm.TM_RULE, live, s.stoch, s.trail)
    return [(t, t.ret - s.cost) for t in done if t.reason != "open"]


def summary(rows: list[tuple[Trade, float]]) -> dict:
    if not rows:
        return {"n": 0, "mean": 0.0}
    r = [x for _, x in rows]
    return {
        "n": len(r),
        "win": sum(x > 0 for x in r) / len(r),
        "mean": statistics.mean(r),
        "median": statistics.median(r),
        "mae": statistics.mean(t.mae for t, _ in rows),
        "worst": min(r),
    }


def by_segment(rows: list[tuple[Trade, float]]) -> dict:
    return {
        "전체": summary(rows),
        "앞": summary([x for x in rows if x[0].entry_day < SPLIT]),
        "뒤": summary([x for x in rows if x[0].entry_day >= SPLIT]),
    }


def random_timing_p(series: list[Series], picks: list[list[tuple[Trade, float]]]) -> tuple:
    """같은 종목·같은 해의 필터 통과일에서 같은 수를 뽑았을 때 평균이 실제 이상인 비율."""
    rng = random.Random(7)
    total, actual_rets, count = [0.0] * N_PERM, [], 0
    for s, rows in zip(series, picks):
        wanted: dict[int, int] = defaultdict(int)
        for t, _ in rows:
            wanted[t.entry_day.year] += 1
        pools: dict[int, list[float]] = defaultdict(list)
        for i, ok in enumerate(s.mask):
            year = s.bars[min(i + 1, len(s.bars) - 1)].day.year
            if not ok or not s.active[i] or year not in wanted:
                continue
            w = 23
            one = simulate(
                s.bars[i : i + w],
                dm.TM_RULE,
                [True] + [False] * (w - 1),
                s.stoch[i : i + w],
                s.trail[i : i + w],
            )
            pools[year].extend(t.ret - s.cost for t in one[:1] if t.reason != "open")
        for year, k in wanted.items():
            vals = pools[year]
            if not vals:
                continue
            k = min(k, len(vals))
            for n in range(N_PERM):
                total[n] += sum(rng.sample(vals, k))
            count += k
        actual_rets.extend(x for _, x in rows)
    actual = statistics.mean(actual_rets)
    fake = [t / count for t in total]
    return (sum(x >= actual for x in fake) + 1) / (N_PERM + 1), statistics.mean(fake)


def random_removal_p(all_rows: list[float], kept: list[float]) -> float:
    rng = random.Random(7)
    target = statistics.mean(kept)
    hits = sum(sum(rng.sample(all_rows, len(kept))) / len(kept) >= target for _ in range(N_PERM))
    return (hits + 1) / (N_PERM + 1)


def verdict(seg: dict, tm: dict) -> bool:
    return all(
        seg[k]["n"] >= 30 and seg[k]["mean"] > 0 and seg[k]["mean"] > tm[k]["mean"]
        for k in ("앞", "뒤")
    )


def main(argv: list[str] | None = None) -> None:
    default = "~/.minerva/daily/history/2026-10-29-1631.json"
    parser = argparse.ArgumentParser(description="피보나치·엘리어트 검증")
    parser.add_argument("--daily", default=str(Path(default).expanduser()))
    args = parser.parse_args(argv)
    daily = json.loads(Path(args.daily).read_text())
    vix = dm.completed(fetch_yahoo("^VIX", "max", ttl=10**9), "US")
    symbols = [("^KS11", 0.001)] + [
        (a["symbol"], 0.003)
        for a in daily["assets"]
        if "error" not in a and a.get("market") == "KR" and not a["symbol"].startswith("^")
    ]
    series = [s for s in (prepare(sym, vix, c) for sym, c in symbols) if s]
    print("종목", len(series), flush=True)

    tm, f1, f2, e1, e2 = [], [], [], [], []
    for s in series:
        inside, first = fib_zones(s.bars)
        bull = [b.close > b.open for b in s.bars]
        blocked = wave5_blocks(s.bars)
        tm.append(trades_of(s, s.tm))
        f1.append(trades_of(s, [m and x and u for m, x, u in zip(s.mask, first, bull)]))
        f2.append(trades_of(s, [t and x for t, x in zip(s.tm, inside)]))
        e1.append(trades_of(s, [m and x for m, x in zip(s.mask, wave3_breakouts(s.bars))]))
        e2.append(trades_of(s, [t and not x for t, x in zip(s.tm, blocked)]))
    flat = lambda rows: [x for r in rows for x in r]  # noqa: E731
    tm_seg = by_segment(flat(tm))
    print("TM", json.dumps(tm_seg, ensure_ascii=False, default=float), flush=True)
    tm_rets = [x for _, x in flat(tm)]
    for name, rows in (("F1", f1), ("E1", e1)):
        seg = by_segment(flat(rows))
        p, base = random_timing_p(series, rows)
        ok = verdict(seg, tm_seg) and p < P_LIMIT
        print(name, f"p={p:.4f} 무작위평균={base:.4f} 통과={ok}")
        print(json.dumps(seg, ensure_ascii=False, default=float), flush=True)
    for name, rows in (("F2", f2), ("E2", e2)):
        seg = by_segment(flat(rows))
        kept = [x for _, x in flat(rows)]
        p = random_removal_p(tm_rets, kept)
        ok = (
            all(seg[k]["n"] >= 30 and seg[k]["mean"] > tm_seg[k]["mean"] for k in ("앞", "뒤"))
            and p < P_LIMIT
        )
        print(name, f"p={p:.4f} 통과={ok}")
        print(json.dumps(seg, ensure_ascii=False, default=float), flush=True)


if __name__ == "__main__":
    main()
