"""BTCUSDT SCR 롱/숏 백테스트 (사전 등록: docs/backtests/2026-10-31-btc-scr-hypotheses.md).

봉 마감으로 신호를 보고 다음 봉 시가에 평가액 100%로 체결한다. 주문과 연결되지 않는다.
실행: `python -m minerva.strategy.btc_bt`
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from minerva.strategy.btc_data import FRAMES, OUT
from minerva.strategy.indicators import Bar, tm_states

SPLIT = date(2024, 1, 1)
COST = 0.0005
RULES = ((1, 1), (2, 1), (2, 2), (3, 1), (3, 3))
YEAR_DAYS = 365.25


@dataclass
class Result:
    final: float
    cagr: float
    mdd: float
    ratio: float
    trades: int
    win_rate: float | None
    long_pnl: float
    short_pnl: float
    exposure: float
    ruined: bool = False


def load(frame: str, root: Path = OUT) -> list[Bar]:
    with (root / f"{frame}.csv").open() as f:
        return [
            Bar(
                datetime.fromtimestamp(int(r["open_time"]) / 1000, timezone.utc).date(),
                float(r["open"]),
                float(r["high"]),
                float(r["low"]),
                float(r["close"]),
                float(r["volume"]),
            )
            for r in csv.DictReader(f)
        ]


def edges(levels: Sequence[int], k: int) -> list[bool]:
    """단계가 k 이상이 된 첫 봉."""
    return [lv >= k and (i == 0 or levels[i - 1] < k) for i, lv in enumerate(levels)]


def simulate(
    bars: Sequence[Bar],
    long_levels: Sequence[int],
    short_levels: Sequence[int],
    k: int,
    c: int,
    cost: float = COST,
    start: int = 0,
    end: int | None = None,
) -> Result:
    """[start, end) 구간을 무포지션으로 시작한다. 마지막 봉 종가에 남은 포지션을 정리한다."""
    end = len(bars) if end is None else end
    open_long, open_short = edges(long_levels, k), edges(short_levels, k)
    close_long, close_short = edges(short_levels, c), edges(long_levels, c)
    equity, peak, mdd = 1.0, 1.0, 0.0
    pos, entry, base, before = 0, 0.0, 0.0, 0.0  # base: 진입 직후, before: 진입 직전 평가액
    trades, wins, in_market = 0, 0, 0
    pnl = {1: 0.0, -1: 0.0}
    ruined = False

    def value(px: float) -> float:
        return base * (px / entry) if pos == 1 else base * (2 - px / entry)

    def leave(px: float) -> None:
        nonlocal equity, pos, trades, wins
        after = value(px) - base * (px / entry) * cost
        pnl[pos] += after - before
        trades += 1
        wins += after > before
        equity, pos = after, 0

    for i in range(start, end):
        b = bars[i]
        if i > start:
            s = i - 1  # 직전 봉 마감 신호, 이번 봉 시가 체결
            both = open_long[s] and open_short[s]
            if pos == 1 and (close_long[s] or (open_short[s] and not both)):
                leave(b.open)
            elif pos == -1 and (close_short[s] or (open_long[s] and not both)):
                leave(b.open)
            if pos == 0 and not both and (open_long[s] or open_short[s]):
                pos = 1 if open_long[s] else -1
                before, base, entry = equity, equity * (1 - cost), b.open
                equity = base
        if pos:
            in_market += 1
            worst = value(b.low if pos == 1 else b.high)
            if worst <= 0:  # 1배 숏은 가격이 두 배면 평가액이 사라진다
                pnl[pos] -= before
                trades += 1
                equity, pos, ruined = 0.0, 0, True
                mdd = -1.0
                break
            mdd = min(mdd, worst / peak - 1)
            equity = value(b.close)
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1)
    if pos:
        leave(bars[end - 1].close)
    days = max((bars[end - 1].day - bars[start].day).days, 1)
    cagr = equity ** (YEAR_DAYS / days) - 1
    return Result(
        equity,
        cagr,
        mdd,
        cagr / abs(mdd) if mdd else 0.0,
        trades,
        wins / trades if trades else None,
        pnl[1],
        pnl[-1],
        in_market / max(end - start, 1),
        ruined,
    )


def hold(bars: Sequence[Bar], start: int, end: int) -> tuple[float, float, float]:
    """BTC 보유: 연수익, 낙폭, 수익/낙폭."""
    entry = bars[start].open
    peak, mdd = entry, 0.0
    for b in bars[start:end]:
        mdd = min(mdd, b.low / peak - 1)
        peak = max(peak, b.high)
    days = max((bars[end - 1].day - bars[start].day).days, 1)
    cagr = (bars[end - 1].close / entry) ** (YEAR_DAYS / days) - 1
    return cagr, mdd, cagr / abs(mdd)


def split_at(bars: Sequence[Bar]) -> int:
    return next(i for i, b in enumerate(bars) if b.day >= SPLIT)


def run_frame(frame: str, root: Path = OUT) -> list[dict]:
    bars = load(frame, root)
    states = tm_states(bars)
    lo, sh = [s.long_level for s in states], [s.short_level for s in states]
    cut, n = split_at(bars), len(bars)
    out = []
    for k, c in RULES:
        row = {"frame": frame, "k": k, "c": c}
        for name, a, b in (("front", 0, cut), ("back", cut, n)):
            row[name] = asdict(simulate(bars, lo, sh, k, c, COST, a, b))
        row["back_cost10"] = asdict(simulate(bars, lo, sh, k, c, 0.001, cut, n))
        row["front_cost10"] = asdict(simulate(bars, lo, sh, k, c, 0.001, 0, cut))
        out.append(row)
    return out


def main() -> None:
    root = OUT
    bars = load("1d", root)
    cut = split_at(bars)
    report = {
        "hold": {"front": hold(bars, 0, cut), "back": hold(bars, cut, len(bars))},
        "rows": [],
    }
    for frame in FRAMES:
        rows = run_frame(frame, root)
        report["rows"] += rows
        best = max(rows, key=lambda r: r["front"]["ratio"])
        print(frame, best["k"], best["c"], best["front"]["cagr"], best["back"]["cagr"], flush=True)
    picks = [r for r in report["rows"] if r["front"]["trades"] >= 30]
    report["pick"] = max(picks, key=lambda r: r["front"]["ratio"]) if picks else None
    report["median_front_cagr"] = statistics.median(r["front"]["cagr"] for r in report["rows"])
    path = root / "scr-backtest.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
