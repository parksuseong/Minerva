"""BTCUSDT 추세 청산 기록 추적. 주문과 연결되지 않고 거래소 키도 쓰지 않는다.

규칙 근거: docs/backtests/2026-11-03-btc-t8-entry-exit-hypotheses.md (사후 관찰, 미검증)
- 추세(T8): 주봉 SCR 3단 마지막 신호 방향과 일봉 구조(k=5 스윙 돌파) 방향이 같으면
  그 방향, 다르면 없음.
  그날 쓰는 추세는 전날 일봉 마감까지의 값이다.
- 진입: 추세 방향의 15분봉 또는 30분봉 SCR 3단 신호, 봉 마감 다음 5분봉 시가.
- 청산: 추세가 그 방향이 아니게 되면 그 5분봉 시가, 또는 진입가 대비 3% 손절.
수익은 1배와 3배(수익·비용 ×3)를 같이 적는다. START 이후 마감 봉으로 매번 다시 계산해
~/.minerva/btc_trend_paper/에 쓰고, 새 진입·청산·추세 변화는 텔레그램으로 알린다.
계산은 백테스트(docs/backtests/raw/btc_fractal.py, btc_trend_detect.py, btc_scr_trend.py)를
numpy 없이 옮긴 것이다.
실행: `python -m minerva.strategy.btc_trend_paper`
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from minerva import notify
from minerva.strategy.btc_paper import fetch

log = logging.getLogger(__name__)

OUT = Path("~/.minerva/btc_trend_paper").expanduser()
START = int(datetime(2026, 11, 3, 14, tzinfo=timezone.utc).timestamp() * 1000)
M5 = 300_000
DAY = 86_400_000
WEEK = 7 * DAY
WEEK_SHIFT = 4 * DAY  # 바이낸스 주봉은 월요일 00:00 UTC 시작, 1970-01-01은 목요일
WARM_5M = 3 * DAY
STOP = 0.03
COST = 0.0005
LIQ = 0.32
RULES = {"T8-15m": 3, "T8-30m": 6}
STRUCT_K = 5


@dataclass(frozen=True)
class Bar:
    t: int
    open: float
    high: float
    low: float
    close: float


@dataclass
class Trade:
    rule: str
    side: int
    entry_t: int
    entry: float
    exit_t: int | None = None
    exit: float | None = None
    why: str = ""
    ret1: float | None = None
    ret3: float | None = None
    mae: float = 0.0


# ---- 지표 (btc_fractal.py와 같은 정의) ----
def stoch(h: Sequence[float], lo: Sequence[float], c: Sequence[float], n: int = 5) -> list:
    out: list = [None] * len(c)
    for i in range(n - 1, len(c)):
        hh, ll = max(h[i - n + 1 : i + 1]), min(lo[i - n + 1 : i + 1])
        out[i] = 50.0 if hh == ll else (c[i] - ll) / (hh - ll) * 100
    return out


def cci(h: Sequence[float], lo: Sequence[float], c: Sequence[float], n: int = 20) -> list:
    tp = [(a + b + x) / 3 for a, b, x in zip(h, lo, c)]
    out: list = [None] * len(c)
    for i in range(n - 1, len(c)):
        w = tp[i - n + 1 : i + 1]
        m = sum(w) / n
        md = sum(abs(x - m) for x in w) / n
        out[i] = 0.0 if md == 0 else (tp[i] - m) / (0.015 * md)
    return out


def rsi(c: Sequence[float], n: int = 14) -> list:
    out: list = [None] * len(c)
    if len(c) <= n:
        return out
    d = [b - a for a, b in zip(c, c[1:])]
    g = [max(x, 0.0) for x in d]
    ls = [max(-x, 0.0) for x in d]
    ag, al = sum(g[:n]) / n, sum(ls[:n]) / n
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + g[i]) / n
        al = (al * (n - 1) + ls[i]) / n
        out[i + 1] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def crossed(x: Sequence, line: float, up: bool) -> list[bool]:
    out = [False] * len(x)
    for i in range(1, len(x)):
        p, v = x[i - 1], x[i]
        if p is None or v is None:
            continue
        out[i] = (p <= line < v) if up else (p >= line > v)
    return out


def recent(x: Sequence[bool], w: int = 3) -> list[bool]:
    return [any(x[max(0, i - w + 1) : i + 1]) for i in range(len(x))]


def scr3(bars: Sequence[Bar]) -> tuple[list[bool], list[bool]]:
    """SCR 3단 롱·숏 신호(처음 켜진 봉만)."""
    h = [b.high for b in bars]
    lo = [b.low for b in bars]
    c = [b.close for b in bars]
    s, cc, r = stoch(h, lo, c), cci(h, lo, c), rsi(c)

    def level(sl: float, cl: float, rl: float, up: bool) -> list[bool]:
        a = recent(crossed(s, sl, up))
        b = recent(crossed(cc, cl, up))
        d = recent(crossed(r, rl, up))
        return [x and y and z for x, y, z in zip(a, b, d)]

    out = []
    for lv in (level(20, -100, 30, True), level(80, 100, 70, False)):
        out.append([v and not (i and lv[i - 1]) for i, v in enumerate(lv)])
    return out[0], out[1]


def resample(bars: Sequence[Bar], factor: int, base: int = M5, shift: int = 0) -> list[Bar]:
    """시각 기준 묶음, 봉이 다 찬 묶음만 (btc_fractal.resample과 같음)."""
    span = base * factor
    groups: dict[int, list[Bar]] = {}
    for b in bars:
        groups.setdefault((b.t - shift) // span, []).append(b)
    out = []
    for k in sorted(groups):
        g = groups[k]
        if len(g) == factor:
            out.append(
                Bar(
                    k * span + shift,
                    g[0].open,
                    max(x.high for x in g),
                    min(x.low for x in g),
                    g[-1].close,
                )
            )
    return out


# ---- 추세 (btc_trend_detect.py의 T4·T5·T8) ----
def hold_state(up: Sequence[bool], dn: Sequence[bool]) -> list[int]:
    out, cur = [], 0
    for u, d in zip(up, dn):
        cur = 1 if u else -1 if d else cur
        out.append(cur)
    return out


def structure(bars: Sequence[Bar], k: int = STRUCT_K) -> list[int]:
    """앞뒤 k봉 최고·최저로 스윙을 정하고(k봉 뒤 확정) 종가가 넘거나 깨면 방향을 바꾼다."""
    out, cur = [], 0
    hi = lo = None
    for i, b in enumerate(bars):
        p = i - k
        if p - k >= 0:
            if bars[p].high == max(x.high for x in bars[p - k : i + 1]):
                hi = bars[p].high
            if bars[p].low == min(x.low for x in bars[p - k : i + 1]):
                lo = bars[p].low
        if hi is not None and b.close > hi:
            cur = 1
        elif lo is not None and b.close < lo:
            cur = -1
        out.append(cur)
    return out


def trend_by_day(daily: Sequence[Bar], weekly: Sequence[Bar]) -> dict[int, int]:
    """일봉 시작 시각 → 그 일봉 마감 때의 T8 (1 상승, −1 하락, 0 없음)."""
    wl, ws = scr3(weekly)
    wstate = hold_state(wl, ws)
    st = structure(daily)
    out, j = {}, -1
    for i, d in enumerate(daily):
        while j + 1 < len(weekly) and weekly[j + 1].t + WEEK <= d.t + DAY:
            j += 1
        t4 = wstate[j] if j >= 0 else 0
        out[int(d.t)] = t4 if t4 == st[i] else 0
    return out


def trend_for(t: int, by_day: dict[int, int]) -> int:
    """5분봉 시각 t에 쓰는 추세 = 전날 일봉 마감 값."""
    return by_day.get(t // DAY * DAY - DAY, 0)


# ---- 체결 (btc_scr_trend.simulate의 반대 신호 청산 방식) ----
def signal_grid(bars5: Sequence[Bar], factor: int) -> tuple[set[int], set[int]]:
    """그 봉 크기 SCR 신호를 쓸 수 있는 첫 5분봉 시각."""
    frame = resample(bars5, factor)
    on_l, on_s = scr3(frame)
    span = factor * M5
    return (
        {b.t + span for b, v in zip(frame, on_l) if v},
        {b.t + span for b, v in zip(frame, on_s) if v},
    )


def rets(side: int, entry: float, px: float, mae: float) -> tuple[float, float]:
    g = (px / entry - 1) * side
    r3 = -1.0 if mae >= LIQ else 3 * g - 6 * COST
    return g - 2 * COST, r3


def replay(
    rule: str, bars5: Sequence[Bar], by_day: dict[int, int], start: int = START, stop: float = STOP
) -> list[Trade]:
    longs, shorts = signal_grid(bars5, RULES[rule])
    out: list[Trade] = []
    cur: Trade | None = None
    line = 0.0
    for b in bars5:
        tr = trend_for(b.t, by_day)
        if cur is not None:
            px, why = None, ""
            if (tr < 1) if cur.side > 0 else (tr >= 0):
                px, why = b.open, "trend"
            elif b.low <= line if cur.side > 0 else b.high >= line:
                px, why = (min(b.open, line) if cur.side > 0 else max(b.open, line)), "stop"
            if px is None:
                adv = (
                    (cur.entry - b.low) / cur.entry
                    if cur.side > 0
                    else (b.high - cur.entry) / cur.entry
                )
                cur.mae = max(cur.mae, adv)
                continue
            g = (px / cur.entry - 1) * cur.side
            cur.mae = max(cur.mae, -g)
            cur.exit_t, cur.exit, cur.why = b.t, px, why
            cur.ret1, cur.ret3 = rets(cur.side, cur.entry, px, cur.mae)
            cur = None
        if b.t < start:
            continue
        side = 1 if (b.t in longs and tr > 0) else -1 if (b.t in shorts and tr < 0) else 0
        if not side:
            continue
        cur = Trade(rule, side, b.t, b.open)
        out.append(cur)
        line = b.open * (1 - stop * side)
        if b.low <= line if side > 0 else b.high >= line:
            cur.mae = stop
            cur.exit_t, cur.exit, cur.why = b.t, line, "stop"
            cur.ret1, cur.ret3 = rets(side, b.open, line, cur.mae)
            cur = None
        else:
            cur.mae = (b.open - b.low) / b.open if side > 0 else (b.high - b.open) / b.open
    return out


# ---- 자료·알림 ----
def to_bars(rows: Sequence[Sequence[float]]) -> list[Bar]:
    return [Bar(int(r[0]), *map(float, r[1:5])) for r in rows]


def closed(rows: list[list[float]], step: int, now: int) -> list[Bar]:
    return [b for b in to_bars(rows) if b.t + step <= now]


def update5(root: Path, now: int) -> list[Bar]:
    path = root / "5m.csv"
    rows: dict[int, list[float]] = {}
    if path.exists():
        with path.open() as f:
            for r in csv.reader(f):
                rows[int(r[0])] = [float(x) for x in r[1:]]
    since = max(rows) + 1 if rows else START - WARM_5M
    for r in fetch("5m", int(since), now):
        if r[0] + M5 <= now:
            rows[int(r[0])] = list(r[1:5])
    with path.open("w", newline="") as f:
        csv.writer(f).writerows([k, *rows[k]] for k in sorted(rows))
    keys = sorted(rows)
    gaps = sum(1 for a, b in zip(keys, keys[1:]) if b - a != M5)
    if gaps:
        log.warning("5m 봉 공백 %d곳", gaps)
    return [Bar(k, *rows[k]) for k in keys]


def when(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).astimezone().strftime("%m/%d %H:%M")


TREND_NAME = {1: "상승", -1: "하락", 0: "없음"}


def message(t: Trade, done: bool) -> str:
    side = "롱" if t.side > 0 else "숏"
    if done:
        why = "추세 전환" if t.why == "trend" else "손절"
        return (
            f"📒 기록 추적 {t.rule} {side} 청산 {when(t.exit_t)} {t.exit:,.0f} ({why}) "
            f"1배 {t.ret1 * 100:+.2f}%, 3배 {t.ret3 * 100:+.2f}%"
        )
    line = t.entry * (1 - STOP * t.side)
    return (
        f"📒 기록 추적 {t.rule} {side} 진입 {when(t.entry_t)} {t.entry:,.0f}, "
        f"손절 {line:,.0f}(3%), 추세가 꺾이면 청산. 실주문 아님"
    )


def run(root: Path = OUT, now: int | None = None) -> dict:
    now = now or int(time.time() * 1000)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "seen.json"
    seen = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    bars5 = update5(root, now)
    daily = closed(fetch("1d", now - 1000 * DAY, now), DAY, now)
    weekly = closed(fetch("1w", now - 500 * WEEK, now), WEEK, now)
    by_day = trend_by_day(daily, weekly)
    trades = [t for rule in RULES for t in replay(rule, bars5, by_day)]
    (root / "trades.json").write_text(
        json.dumps([asdict(t) for t in trades], ensure_ascii=False, indent=1)
    )
    msgs: list[tuple[str, str]] = []
    if daily:
        last = int(daily[-1].t)
        tr = by_day[last]
        key = f"trend:{last}:{tr}"
        prev = by_day.get(last - DAY)
        if last + DAY >= START and prev is not None and prev != tr:
            msgs.append(
                (
                    key,
                    f"📒 기록 추적 추세 {TREND_NAME[prev]} → {TREND_NAME[tr]} "
                    f"({when(last + DAY)} 일봉 마감 기준)",
                )
            )
    for t in sorted(trades, key=lambda x: x.entry_t):
        msgs.append((f"in:{t.rule}:{t.entry_t}", message(t, False)))
        if t.exit_t is not None:
            msgs.append((f"out:{t.rule}:{t.entry_t}", message(t, True)))
    for key, text in msgs:
        if key not in seen and notify.send(text):
            seen.add(key)
    state_path.write_text(json.dumps(sorted(seen)))
    trend_now = by_day[int(daily[-1].t)] if daily else 0
    return {
        "trend": TREND_NAME[trend_now],
        "trades": len(trades),
        "closed": sum(1 for t in trades if t.exit_t is not None),
        "open": [asdict(t) for t in trades if t.exit_t is None],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
