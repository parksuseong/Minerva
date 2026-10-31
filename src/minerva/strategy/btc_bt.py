"""BTCUSDT SCR 롱/숏 백테스트 (사전 등록: docs/backtests/2026-10-31-btc-scr-hypotheses.md).

봉 마감으로 신호를 보고 다음 봉 시가에 평가액 100%로 체결한다. 주문과 연결되지 않는다.
실행: `python -m minerva.strategy.btc_bt`
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from minerva.strategy.btc_data import FRAMES, OUT
from minerva.strategy.indicators import Bar, TMState, tm_states

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
    stop: float | None = None,
    allow_long: Sequence[bool] | None = None,
    allow_short: Sequence[bool] | None = None,
    log: list[float] | None = None,
) -> Result:
    """[start, end) 구간을 무포지션으로 시작한다. 마지막 봉 종가에 남은 포지션을 정리한다."""
    end = len(bars) if end is None else end
    open_long, open_short = edges(long_levels, k), edges(short_levels, k)
    if allow_long is not None:
        open_long = [o and a for o, a in zip(open_long, allow_long)]
    if allow_short is not None:
        open_short = [o and a for o, a in zip(open_short, allow_short)]
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
        if log is not None:
            log.append(after / before - 1)
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
            if stop is not None:
                line = entry * (1 - stop) if pos == 1 else entry * (1 + stop)
                hit = b.low <= line if pos == 1 else b.high >= line
                if hit:
                    gapped = b.open < line if pos == 1 else b.open > line
                    leave(b.open if gapped else line)
                    peak = max(peak, equity)
                    mdd = min(mdd, equity / peak - 1)
                    continue
            worst = value(b.low if pos == 1 else b.high)
            if worst <= 0:  # 1배 숏은 가격이 두 배면 평가액이 사라진다
                pnl[pos] -= before
                trades += 1
                if log is not None:
                    log.append(-1.0)
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


def run_frame(frame: str, root: Path = OUT, stop: float | None = None) -> list[dict]:
    bars = load(frame, root)
    states = tm_states(bars)
    lo, sh = [s.long_level for s in states], [s.short_level for s in states]
    cut, n = split_at(bars), len(bars)
    out = []
    for k, c in RULES:
        row = {"frame": frame, "k": k, "c": c}
        for name, a, b in (("front", 0, cut), ("back", cut, n)):
            row[name] = asdict(simulate(bars, lo, sh, k, c, COST, a, b, stop))
        row["back_cost10"] = asdict(simulate(bars, lo, sh, k, c, 0.001, cut, n, stop))
        row["front_cost10"] = asdict(simulate(bars, lo, sh, k, c, 0.001, 0, cut, stop))
        out.append(row)
    return out


def marked_levels(states: Sequence[TMState], window: int = 3) -> tuple[list[int], list[int]]:
    """과열 후 1단 반전 (2026-11-01 사전 등록). Stoch 돌파·이탈 봉에 최근 CCI 과열을 요구한다."""
    long_lv, short_lv = [0] * len(states), [0] * len(states)
    for i in range(1, len(states)):
        prev, cur = states[i - 1].stoch, states[i].stoch
        if prev is None or cur is None:
            continue
        recent = [s.cci for s in states[max(0, i - window + 1) : i + 1] if s.cci is not None]
        if prev <= 20 < cur and any(c <= -100 for c in recent):
            long_lv[i] = 1
        if prev >= 80 > cur and any(c >= 100 for c in recent):
            short_lv[i] = 1
    return long_lv, short_lv


def close_times(frame: str, root: Path = OUT) -> list[int]:
    with (root / f"{frame}.csv").open() as f:
        return [int(r["close_time"]) for r in csv.DictReader(f)]


def state_dir(s: TMState) -> int:
    """상방 1, 하방 −1, 중립 0."""
    if s.cci is None or s.rsi is None:
        return 0
    if s.cci > 0 and s.rsi > 50:
        return 1
    return -1 if s.cci < 0 and s.rsi < 50 else 0


def align(times: Sequence[int], upper_times: Sequence[int], upper: Sequence[int]) -> list[int]:
    """매매 봉이 마감된 시각까지 마감된 마지막 상위 봉의 값. 없으면 0."""
    out, j, cur = [], 0, 0
    for t in times:
        while j < len(upper_times) and upper_times[j] <= t:
            cur = upper[j]
            j += 1
        out.append(cur)
    return out


def without_top(rets: Sequence[float], n: int, days: int) -> float:
    """거래 수익 상위 n건을 뺀 연수익."""
    kept = sorted(rets)[: max(len(rets) - n, 0)]
    equity = 1.0
    for r in kept:
        equity *= 1 + r
    return equity ** (YEAR_DAYS / max(days, 1)) - 1


def run_filter(root: Path = OUT) -> dict:
    """15분봉 + 상위 봉 방향 필터 20개 (2026-11-01 사전 등록)."""
    bars = load("15m", root)
    times = close_times("15m", root)
    states = tm_states(bars)
    lo, sh = [s.long_level for s in states], [s.short_level for s in states]
    cut, n = split_at(bars), len(bars)
    back_days = (bars[-1].day - bars[cut].day).days
    rows = []
    for up in ("4h", "1d"):
        dirs = [state_dir(s) for s in tm_states(load(up, root))]
        mask = align(times, close_times(up, root), dirs)
        allow_l, allow_s = [m == 1 for m in mask], [m == -1 for m in mask]
        for stop in (None, 0.01):
            for k, c in RULES:
                row = {"upper": up, "stop": stop, "k": k, "c": c}
                common = (allow_l, allow_s)
                row["front"] = asdict(simulate(bars, lo, sh, k, c, COST, 0, cut, stop, *common))
                rets: list[float] = []
                row["back"] = asdict(
                    simulate(bars, lo, sh, k, c, COST, cut, n, stop, *common, rets)
                )
                row["back_cost10"] = asdict(
                    simulate(bars, lo, sh, k, c, 0.001, cut, n, stop, *common)
                )
                row["front_gross"] = asdict(
                    simulate(bars, lo, sh, k, c, 0.0, 0, cut, stop, *common)
                )
                row["back_gross"] = asdict(simulate(bars, lo, sh, k, c, 0.0, cut, n, stop, *common))
                row["back_without_top5"] = without_top(rets, 5, back_days)
                rows.append(row)
                print(up, stop, k, c, row["front"]["cagr"], row["back"]["cagr"], flush=True)
    picks = [r for r in rows if r["front"]["trades"] >= 30]
    daily = load("1d", root)
    dcut = split_at(daily)
    return {
        "hold": {"front": hold(daily, 0, dcut), "back": hold(daily, dcut, len(daily))},
        "rows": rows,
        "pick": max(picks, key=lambda r: r["front"]["ratio"]) if picks else None,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stop", type=float, help="손절 비율 (예: 0.01)")
    p.add_argument("--filter", action="store_true", help="15분봉 상위 봉 필터 시험")
    args = p.parse_args(argv)
    root = OUT
    if args.filter:
        path = root / "scr-15m-filter.json"
        path.write_text(json.dumps(run_filter(root), ensure_ascii=False, indent=1))
        print("wrote", path)
        return
    bars = load("1d", root)
    cut = split_at(bars)
    report = {
        "hold": {"front": hold(bars, 0, cut), "back": hold(bars, cut, len(bars))},
        "rows": [],
    }
    for frame in FRAMES:
        rows = run_frame(frame, root, args.stop)
        report["rows"] += rows
        best = max(rows, key=lambda r: r["front"]["ratio"])
        print(frame, best["k"], best["c"], best["front"]["cagr"], best["back"]["cagr"], flush=True)
    picks = [r for r in report["rows"] if r["front"]["trades"] >= 30]
    report["pick"] = max(picks, key=lambda r: r["front"]["ratio"]) if picks else None
    report["median_front_cagr"] = statistics.median(r["front"]["cagr"] for r in report["rows"])
    path = root / (f"scr-backtest-stop{args.stop}.json" if args.stop else "scr-backtest.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
