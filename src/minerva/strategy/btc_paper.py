"""BTCUSDT 돈 없는 기록 추적. 주문과 연결되지 않고 거래소 키도 쓰지 않는다.

규칙과 폐기 조건: docs/backtests/2026-11-01-btc-paper-tracking.md
L1(15분봉 20일 신고가 롱, 96봉 보유, 손절 1%), S4(1시간봉 구조 돌파, 뒤집기, 손절 1%)를
START 이후 마감 봉으로 다시 계산해 ~/.minerva/btc_paper/에 쓴다. 새 진입·청산은 텔레그램으로 알린다.
실행: `python -m minerva.strategy.btc_paper`
"""

from __future__ import annotations

import csv
import json
import logging
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from minerva import notify
from minerva.strategy.indicators import rsi

log = logging.getLogger(__name__)

OUT = Path("~/.minerva/btc_paper").expanduser()
KLINES = (
    "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT"
    "&interval={iv}&limit=1000&startTime={since}"
)
START = int(datetime(2026, 11, 1, 10, tzinfo=timezone.utc).timestamp() * 1000)
FRAME_MS = {"15m": 900_000, "1h": 3_600_000}
WARM_BARS = {"15m": 2100, "1h": 2000}
STOP = 0.01
COST = 0.0005
LOOK = 1920
HOLD = 96
LEN, OB, OS = 7, 70, 30


@dataclass(frozen=True)
class PBar:
    t: int
    open: float
    high: float
    low: float
    close: float


@dataclass
class Trade:
    rule: str
    side: int
    signal_t: int
    entry_t: int
    entry: float
    exit_t: int | None = None
    exit: float | None = None
    why: str = ""
    ret: float | None = None
    overshoot: float | None = None  # 손절 봉에서 손절선을 넘은 폭, 체결 밀림의 상한


def breakouts(closes: Sequence[float], look: int = LOOK) -> list[bool]:
    """종가가 직전 look봉 최고 종가를 처음 넘은 봉."""
    out, above = [False] * len(closes), False
    for i in range(look, len(closes)):
        now = closes[i] > max(closes[i - look : i])
        out[i] = now and not above
        above = now
    return out


def swings(bars: Sequence[PBar]) -> list[tuple[int, int, str, float]]:
    """확정된 스윙 라벨 (확정 봉, 쪽, 글자, 가격). 쪽 1은 고점."""
    r = rsi([b.close for b in bars], LEN)
    out: list[tuple[int, int, str, float]] = []
    cur: list | None = None
    last: dict[int, float | None] = {1: None, -1: None}
    for i, v in enumerate(r):
        if v is None:
            continue
        b = bars[i]
        side = 1 if v >= OB else -1 if v <= OS else 0
        if side and cur and cur[0] != side:
            s, price = cur
            prev = last[s]
            if s == 1:
                text = "HH" if prev is None or price > prev else "LH"
            else:
                text = "HL" if prev is None or price > prev else "LL"
            last[s] = price
            out.append((i, s, text, price))
            cur = None
        if cur is None and side:
            cur = [side, b.high if side == 1 else b.low]
        elif cur:
            cur[1] = max(cur[1], b.high) if cur[0] == 1 else min(cur[1], b.low)
    return out


def s4_pulses(bars: Sequence[PBar]) -> tuple[list[bool], list[bool]]:
    """S4: 종가가 마지막 고점 위로 처음 올라서고 마지막 저점이 HL이면 롱, 반대면 숏."""
    ev = swings(bars)
    lo, hi = [False] * len(bars), [False] * len(bars)
    top, bot, t_hi, t_lo = None, None, None, None
    above, below, k = False, False, 0
    for i, b in enumerate(bars):
        while k < len(ev) and ev[k][0] <= i:
            _, s, text, price = ev[k]
            if s == 1:
                top, t_hi, above = price, text, False
            else:
                bot, t_lo, below = price, text, False
            k += 1
        if top is not None and b.close > top and not above:
            above = True
            lo[i] = t_lo == "HL"
        if bot is not None and b.close < bot and not below:
            below = True
            hi[i] = t_hi == "LH"
    return lo, hi


def net(side: int, entry: float, px: float, cost: float = COST) -> float:
    return (side * (px / entry - 1) + 1) * (1 - cost) - (px / entry) * cost - 1


def _stop_exit(t: Trade, b: PBar) -> bool:
    line = t.entry * (1 - t.side * STOP)
    if not (b.low <= line if t.side == 1 else b.high >= line):
        return False
    px = min(b.open, line) if t.side == 1 else max(b.open, line)
    t.exit_t, t.exit, t.why, t.ret = b.t, px, "stop", net(t.side, t.entry, px)
    t.overshoot = (px - b.low) / line if t.side == 1 else (b.high - px) / line
    return True


def replay_l1(bars: Sequence[PBar], start: int = START) -> list[Trade]:
    sig = breakouts([b.close for b in bars])
    out: list[Trade] = []
    cur: Trade | None = None
    until = 0
    for i, b in enumerate(bars):
        if cur is not None:
            if i == until:
                cur.exit_t, cur.exit, cur.why = b.t, b.open, "time"
                cur.ret = net(1, cur.entry, b.open)
                cur = None
            elif _stop_exit(cur, b):
                cur = None
        if cur is None and i > 0 and sig[i - 1] and bars[i - 1].t >= start:
            cur = Trade("L1", 1, bars[i - 1].t, b.t, b.open)
            out.append(cur)
            until = i + HOLD
            if _stop_exit(cur, b):
                cur = None
    return out


def replay_s4(bars: Sequence[PBar], start: int = START) -> list[Trade]:
    lo, hi = s4_pulses(bars)
    out: list[Trade] = []
    cur: Trade | None = None
    for i, b in enumerate(bars):
        if i > 0 and bars[i - 1].t >= start:
            up, down = lo[i - 1], hi[i - 1]
            want = 0 if up == down else 1 if up else -1
            if cur is not None and want == -cur.side:
                cur.exit_t, cur.exit, cur.why = b.t, b.open, "flip"
                cur.ret = net(cur.side, cur.entry, b.open)
                cur = None
            if cur is None and want:
                cur = Trade("S4", want, bars[i - 1].t, b.t, b.open)
                out.append(cur)
        if cur is not None and _stop_exit(cur, b):
            cur = None
    return out


def fetch(frame: str, since: int, now: int) -> list[list[float]]:
    rows: list[list[float]] = []
    while since < now:
        url = KLINES.format(iv=frame, since=since)
        with urllib.request.urlopen(url, timeout=20) as res:
            page = json.load(res)
        if not page:
            break
        rows += [[int(p[0]), *map(float, p[1:5])] for p in page]
        since = page[-1][0] + 1
        if len(page) < 1000:
            break
        time.sleep(0.2)
    return rows


def update(frame: str, root: Path = OUT, now: int | None = None) -> list[PBar]:
    """마감 봉을 CSV에 이어 붙이고 전체를 돌려준다."""
    now = now or int(time.time() * 1000)
    step = FRAME_MS[frame]
    path = root / f"{frame}.csv"
    rows: dict[int, list[float]] = {}
    if path.exists():
        with path.open() as f:
            for r in csv.reader(f):
                rows[int(r[0])] = [float(x) for x in r[1:]]
    since = max(rows) + 1 if rows else START - WARM_BARS[frame] * step
    for r in fetch(frame, int(since), now):
        if r[0] + step <= now:
            rows[int(r[0])] = list(r[1:])
    with path.open("w", newline="") as f:
        csv.writer(f).writerows([k, *rows[k]] for k in sorted(rows))
    keys = sorted(rows)
    gaps = [(a, b) for a, b in zip(keys, keys[1:]) if b - a != step]
    if gaps:
        raise ValueError(f"{frame} 봉 공백 {len(gaps)}곳")
    return [PBar(k, *rows[k]) for k in keys]


def when(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).astimezone().strftime("%m/%d %H:%M")


def message(t: Trade, closed: bool) -> str:
    side = "롱" if t.side == 1 else "숏"
    if closed:
        return (
            f"📒 기록 추적 {t.rule} {side} 청산 {when(t.exit_t)} {t.exit:,.0f} "
            f"({t.why}) {t.ret * 100:+.2f}%"
        )
    return (
        f"📒 기록 추적 {t.rule} {side} 진입 {when(t.entry_t)} {t.entry:,.0f}, 손절 1%, 실주문 아님"
    )


def run(root: Path = OUT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "seen.json"
    seen = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    trades = replay_l1(update("15m", root)) + replay_s4(update("1h", root))
    (root / "trades.json").write_text(
        json.dumps([asdict(t) for t in trades], ensure_ascii=False, indent=1)
    )
    for t in sorted(trades, key=lambda x: x.entry_t):
        for key, closed in (
            (f"in:{t.rule}:{t.entry_t}", False),
            (f"out:{t.rule}:{t.entry_t}", True),
        ):
            if key in seen or (closed and t.exit_t is None):
                continue
            if notify.send(message(t, closed)):
                seen.add(key)
    state_path.write_text(json.dumps(sorted(seen)))
    done = [t for t in trades if t.ret is not None]
    return {
        "trades": len(trades),
        "closed": len(done),
        "open": [asdict(t) for t in trades if t.exit_t is None],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
