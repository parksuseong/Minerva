"""BTCUSDT 15분봉 다른 지표 조사 (사전 등록: docs/backtests/2026-11-01-btc-features-hypotheses.md).

SCR 1단 신호 뒤 ±1% 중 유리한 쪽이 먼저 닿는 비율을 지표 구간별로 잰다. 주문과 연결되지 않는다.
실행: `python -m minerva.strategy.btc_features`
"""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone

from minerva.strategy import btc_bt as bt
from minerva.strategy.btc_data import OUT
from minerva.strategy.indicators import Bar, Num, dmi, tm_states

TOUCH = 0.01
MAX_BARS = 2000


def ema(values: Sequence[float], n: int) -> list[float]:
    k, out = 2 / (n + 1), []
    cur = values[0]
    for v in values:
        cur = v * k + cur * (1 - k)
        out.append(cur)
    return out


def rolling_mean(values: Sequence[float], n: int) -> list[Num]:
    out: list[Num] = [None] * len(values)
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= n:
            total -= values[i - n]
        if i >= n - 1:
            out[i] = total / n
    return out


def true_range(bars: Sequence[Bar]) -> list[float]:
    out = [bars[0].high - bars[0].low]
    for p, b in zip(bars, bars[1:]):
        out.append(max(b.high - b.low, abs(b.high - p.close), abs(b.low - p.close)))
    return out


def percent_b(closes: Sequence[float], n: int = 20) -> list[Num]:
    out: list[Num] = [None] * len(closes)
    for i in range(n - 1, len(closes)):
        w = closes[i - n + 1 : i + 1]
        mean = sum(w) / n
        sd = (sum((x - mean) ** 2 for x in w) / n) ** 0.5
        out[i] = (closes[i] - mean) / (2 * sd) if sd else 0.0
    return out


def first_touch(bars: Sequence[Bar], i: int, sign: int) -> tuple[bool | None, int]:
    """신호 봉 i 뒤 다음 시가 기준. (유리한 쪽이 먼저인가, 닿은 봉). 한 봉에 둘 다면 None."""
    if i + 1 >= len(bars):
        return None, i
    entry = bars[i + 1].open
    up, down = entry * (1 + TOUCH), entry * (1 - TOUCH)
    for j in range(i + 1, min(i + 1 + MAX_BARS, len(bars))):
        high, low = bars[j].high >= up, bars[j].low <= down
        if high and low:
            return None, j
        if high or low:
            return (high == (sign == 1)), j
    return None, min(i + MAX_BARS, len(bars) - 1)


def quintiles(values: Sequence[float]) -> list[float]:
    s = sorted(values)
    return [s[len(s) * q // 5] for q in range(1, 5)]


def bucket(value: float, cuts: Sequence[float]) -> int:
    return sum(value > c for c in cuts)


def summarize(events: Sequence[tuple[int, int, bool, int]], bins: int) -> list[dict]:
    """구간별 적중률. 앞 신호의 결과가 나기 전에 나온 같은 구간 신호는 겹치므로 뺀다."""
    out = []
    for g in range(bins):
        hits, n, free = 0, 0, -1
        for i, _, hit, end in sorted(e for e in events if e[1] == g):
            if i <= free:
                continue
            free = end
            n += 1
            hits += hit
        out.append({"bin": g, "n": n, "rate": hits / n if n else None})
    return out


def features(bars: Sequence[Bar], taker: Sequence[float], root=OUT) -> dict[str, tuple]:
    closes = [b.close for b in bars]
    vols = [b.volume for b in bars]
    vol20 = rolling_mean(vols, 20)
    tr14 = rolling_mean(true_range(bars), 14)
    e200 = ema(closes, 200)
    four = bt.load("4h", root)
    e4 = ema([b.close for b in four], 50)
    dist4 = [b.close / e - 1 for b, e in zip(four, e4)]
    mask4 = bt.align(bt.close_times("15m", root), bt.close_times("4h", root), dist4)
    ratio = [t / v if v else 0.5 for t, v in zip(rolling_sum(taker), rolling_sum(vols))]
    return {
        "거래량 비율": ([v / m if m else None for v, m in zip(vols, vol20)], False),
        "체결 방향": ([r - 0.5 for r in ratio], True),
        "ATR 비율": ([None if a is None else a / c for a, c in zip(tr14, closes)], False),
        "EMA200 거리": ([c / e - 1 for c, e in zip(closes, e200)], True),
        "ADX": (dmi(bars, 14)[2], False),
        "볼린저 %B": (percent_b(closes), True),
        "4시간 EMA50 거리": (mask4, True),
    }


def rolling_sum(values: Sequence[float], n: int = 20) -> list[float]:
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= n:
            total -= values[i - n]
        out.append(total)
    return out


def run(root=OUT) -> dict:
    bars = bt.load("15m", root)
    with (root / "15m.csv").open() as f:
        rows = list(csv.DictReader(f))
    taker = [float(r["taker_buy_base"]) for r in rows]
    hours = [
        datetime.fromtimestamp(int(r["open_time"]) / 1000, timezone.utc).hour // 4 for r in rows
    ]
    states = tm_states(bars)
    cut = bt.split_at(bars)
    signals = []
    for side, levels in (
        (1, [s.long_level for s in states]),
        (-1, [s.short_level for s in states]),
    ):
        edge = bt.edges(levels, 1)
        signals += [(i, side) for i in range(len(bars) - 1) if edge[i]]
    signals.sort()
    outcome = {}
    for i, side in signals:
        hit, end = first_touch(bars, i, side)
        if hit is not None:
            outcome[(i, side)] = (hit, end)
    report = {}
    for name, (values, signed) in features(bars, taker, root).items():
        pts = []
        for (i, side), (hit, end) in outcome.items():
            v = values[i]
            if v is not None:
                pts.append((i, side * v if signed else v, hit, end))
        cuts = quintiles([v for i, v, _, _ in pts if i < cut])
        rows_f = [(i, bucket(v, cuts), hit, end) for i, v, hit, end in pts if i < cut]
        rows_b = [(i, bucket(v, cuts), hit, end) for i, v, hit, end in pts if i >= cut]
        report[name] = {"cuts": cuts, "front": summarize(rows_f, 5), "back": summarize(rows_b, 5)}
    pts = [(i, hours[i], hit, end) for (i, side), (hit, end) in outcome.items()]
    report["시간대"] = {
        "front": summarize([p for p in pts if p[0] < cut], 6),
        "back": summarize([p for p in pts if p[0] >= cut], 6),
    }
    return report


def ols(xs: Sequence[Sequence[float]], ys: Sequence[float]) -> list[float]:
    """절편 포함 최소제곱. 가우스 소거로 푼다."""
    rows = [[1.0, *x] for x in xs]
    n = len(rows[0])
    m = [[sum(r[a] * r[b] for r in rows) for b in range(n)] for a in range(n)]
    v = [sum(r[a] * y for r, y in zip(rows, ys)) for a in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[p], v[i], v[p] = m[p], m[i], v[p], v[i]
        for r in range(n):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [a - f * b for a, b in zip(m[r], m[i])]
                v[r] -= f * v[i]
    return [v[i] / m[i][i] for i in range(n)]


def standardize(values: Sequence[float], ref: Sequence[float]) -> list[float]:
    mean = sum(ref) / len(ref)
    sd = (sum((x - mean) ** 2 for x in ref) / len(ref)) ** 0.5
    return [(x - mean) / sd for x in values]


def weights(root=OUT) -> dict:
    """앞 구간에서 Stoch·CCI·RSI 가중치를 맞추고 뒤 구간에 적용한다 (2026-11-01 사전 등록)."""
    bars = bt.load("15m", root)
    states = tm_states(bars)
    cut = bt.split_at(bars)
    report = {}
    for name, side, levels in (
        ("롱", 1, [s.long_level for s in states]),
        ("숏", -1, [s.short_level for s in states]),
    ):
        edge = bt.edges(levels, 1)
        pts = []
        for i in range(len(bars) - 1):
            s = states[i]
            if edge[i] and None not in (s.stoch, s.cci, s.rsi):
                hit, end = first_touch(bars, i, side)
                if hit is not None:
                    pts.append((i, (s.stoch, s.cci, s.rsi), hit, end))
        front = [p for p in pts if p[0] < cut]
        cols = [[p[1][k] for p in pts] for k in range(3)]
        refs = [[p[1][k] for p in front] for k in range(3)]
        z = list(zip(*(standardize(c, r) for c, r in zip(cols, refs))))
        n_front = len(front)
        beta = ols(z[:n_front], [float(p[2]) for p in front])
        score = [beta[0] + sum(b * x for b, x in zip(beta[1:], row)) for row in z]
        cuts = quintiles(score[:n_front])
        rows = [(p[0], bucket(sc, cuts), p[2], p[3]) for p, sc in zip(pts, score)]
        report[name] = {
            "weights": dict(zip(("절편", "Stoch", "CCI", "RSI"), beta)),
            "front": summarize([r for r in rows if r[0] < cut], 5),
            "back": summarize([r for r in rows if r[0] >= cut], 5),
        }
    return report


def main() -> None:
    if "--weights" in sys.argv:
        path = OUT / "scr-weights.json"
        path.write_text(json.dumps(weights(), ensure_ascii=False, indent=1))
    else:
        path = OUT / "scr-features.json"
        path.write_text(json.dumps(run(), ensure_ascii=False, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
