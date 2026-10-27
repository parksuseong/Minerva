"""삼중 모멘텀(TM) 시그널과 이평선 상태를 계산한다.

TM = Stochastic fast %K, CCI, RSI 세 지표. 빠른 순서대로 L1/L2/L3 라고 부른다.
- L1 Stochastic fast %K(5): 과매도 20 / 과매수 80
- L2 CCI(9 또는 20): 과매도 -100 / 과매수 +100
- L3 RSI(14, Wilder): 과매도 30 / 과매수 70

모든 함수는 리스트를 받고 같은 길이의 리스트를 돌려준다. 계산이 안 되는 앞부분은 None.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Optional

Num = Optional[float]


@dataclass(frozen=True)
class Bar:
    day: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def load_csv(path: str) -> list[Bar]:
    """Date,Open,High,Low,Close[,Volume] 헤더를 가진 CSV(야후 형식)를 읽는다."""
    bars = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("Close") or row["Close"] in ("null", ""):
                continue
            bars.append(
                Bar(
                    day=date.fromisoformat(row["Date"][:10]),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume") or 0),
                )
            )
    bars.sort(key=lambda b: b.day)
    return bars


def resample(bars: Sequence[Bar], frame: str) -> list[Bar]:
    """일봉을 주봉('W', ISO 주) 또는 월봉('M')으로 묶는다. 'D'는 그대로."""
    if frame == "D":
        return list(bars)
    if frame not in ("W", "M"):
        raise ValueError(f"unknown frame: {frame}")

    def key(d: date) -> tuple[int, int]:
        if frame == "W":
            iso = d.isocalendar()
            return (iso[0], iso[1])
        return (d.year, d.month)

    out: list[Bar] = []
    group: list[Bar] = []
    for b in bars:
        if group and key(b.day) != key(group[0].day):
            out.append(_merge(group))
            group = []
        group.append(b)
    if group:
        out.append(_merge(group))
    return out


def _merge(group: list[Bar]) -> Bar:
    return Bar(
        day=group[-1].day,
        open=group[0].open,
        high=max(b.high for b in group),
        low=min(b.low for b in group),
        close=group[-1].close,
        volume=sum(b.volume for b in group),
    )


def sma(values: Sequence[float], n: int) -> list[Num]:
    out: list[Num] = [None] * len(values)
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= n:
            total -= values[i - n]
        if i >= n - 1:
            out[i] = total / n
    return out


def stoch_fast_k(bars: Sequence[Bar], n: int = 5) -> list[Num]:
    out: list[Num] = [None] * len(bars)
    for i in range(n - 1, len(bars)):
        window = bars[i - n + 1 : i + 1]
        hi = max(b.high for b in window)
        lo = min(b.low for b in window)
        out[i] = 50.0 if hi == lo else (bars[i].close - lo) / (hi - lo) * 100
    return out


def cci(bars: Sequence[Bar], n: int = 20) -> list[Num]:
    tp = [(b.high + b.low + b.close) / 3 for b in bars]
    mean = sma(tp, n)
    out: list[Num] = [None] * len(bars)
    for i in range(n - 1, len(bars)):
        m = mean[i]
        dev = sum(abs(x - m) for x in tp[i - n + 1 : i + 1]) / n
        out[i] = 0.0 if dev == 0 else (tp[i] - m) / (0.015 * dev)
    return out


def rsi(closes: Sequence[float], n: int = 14) -> list[Num]:
    out: list[Num] = [None] * len(closes)
    if len(closes) <= n:
        return out
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0)
        losses += max(-d, 0)
    avg_g, avg_l = gains / n, losses / n
    out[n] = _rsi_value(avg_g, avg_l)
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_g = (avg_g * (n - 1) + max(d, 0)) / n
        avg_l = (avg_l * (n - 1) + max(-d, 0)) / n
        out[i] = _rsi_value(avg_g, avg_l)
    return out


def _rsi_value(avg_g: float, avg_l: float) -> float:
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    return 100 - 100 / (1 + avg_g / avg_l)


# (지표 이름, 과매도선, 과매수선)
LEVELS = (("stoch", 20.0, 80.0), ("cci", -100.0, 100.0), ("rsi", 30.0, 70.0))


def crossed_up(series: Sequence[Num], i: int, line: float) -> bool:
    prev, cur = series[i - 1] if i > 0 else None, series[i]
    return prev is not None and cur is not None and prev <= line < cur


def crossed_down(series: Sequence[Num], i: int, line: float) -> bool:
    prev, cur = series[i - 1] if i > 0 else None, series[i]
    return prev is not None and cur is not None and prev >= line > cur


@dataclass(frozen=True)
class TMState:
    day: date
    close: float
    stoch: Num
    cci: Num
    rsi: Num
    long_level: int  # 0~3: 최근 window봉 안에 과매도 상향 돌파한 단계 (L1부터 연속)
    short_level: int  # 0~3: 최근 window봉 안에 과매수 하향 이탈한 단계


def tm_states(bars: Sequence[Bar], cci_n: int = 20, window: int = 3) -> list[TMState]:
    """봉마다 TM 롱/숏 레벨을 계산한다.

    레벨 k는 L1..Lk 지표가 모두 최근 `window`봉 안에 기준선을 돌파했다는 뜻이다.
    L3(RSI)만 돌파하고 L1이 아니면 레벨은 0이다 — 빠른 지표가 먼저 확인해 줘야 한다.
    """
    series = {
        "stoch": stoch_fast_k(bars),
        "cci": cci(bars, cci_n),
        "rsi": rsi([b.close for b in bars]),
    }
    out = []
    for i, b in enumerate(bars):
        lo = max(0, i - window + 1)
        long_level = short_level = 0
        for name, oversold, _ in LEVELS:
            if any(crossed_up(series[name], j, oversold) for j in range(lo, i + 1)):
                long_level += 1
            else:
                break
        for name, _, overbought in LEVELS:
            if any(crossed_down(series[name], j, overbought) for j in range(lo, i + 1)):
                short_level += 1
            else:
                break
        out.append(
            TMState(
                day=b.day,
                close=b.close,
                stoch=series["stoch"][i],
                cci=series["cci"][i],
                rsi=series["rsi"][i],
                long_level=long_level,
                short_level=short_level,
            )
        )
    return out


MA_PERIODS = (5, 10, 20, 50, 120, 200)


def ma_snapshot(closes: Sequence[float]) -> dict:
    """마지막 봉 기준 이평선 값, 종가 위/아래, 정배열/역배열 여부."""
    mas = {n: sma(closes, n)[-1] for n in MA_PERIODS if len(closes) >= n}
    vals = [mas[n] for n in MA_PERIODS if n in mas]
    close = closes[-1]
    return {
        "close": close,
        "ma": mas,
        "above": {n: close > v for n, v in mas.items()},
        "bull_aligned": len(vals) >= 3 and all(a > b for a, b in zip(vals, vals[1:])),
        "bear_aligned": len(vals) >= 3 and all(a < b for a, b in zip(vals, vals[1:])),
    }


def dmi(bars: Sequence[Bar], n: int = 14) -> tuple[list[Num], list[Num], list[Num]]:
    """Wilder 방식 (+DI, -DI, ADX)."""
    size = len(bars)
    pdi: list[Num] = [None] * size
    mdi: list[Num] = [None] * size
    adx: list[Num] = [None] * size
    if size <= n:
        return pdi, mdi, adx
    tr_s = pdm_s = mdm_s = 0.0
    dxs: list[float] = []
    adx_v: Num = None
    for i in range(1, size):
        cur, prev = bars[i], bars[i - 1]
        up, down = cur.high - prev.high, prev.low - cur.low
        pdm = up if up > down and up > 0 else 0.0
        mdm = down if down > up and down > 0 else 0.0
        tr = max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close))
        if i <= n:
            tr_s, pdm_s, mdm_s = tr_s + tr, pdm_s + pdm, mdm_s + mdm
            if i < n:
                continue
        else:
            tr_s = tr_s - tr_s / n + tr
            pdm_s = pdm_s - pdm_s / n + pdm
            mdm_s = mdm_s - mdm_s / n + mdm
        p = 0.0 if tr_s == 0 else 100 * pdm_s / tr_s
        m = 0.0 if tr_s == 0 else 100 * mdm_s / tr_s
        pdi[i], mdi[i] = p, m
        dx = 0.0 if p + m == 0 else 100 * abs(p - m) / (p + m)
        if adx_v is None:
            dxs.append(dx)
            if len(dxs) == n:
                adx_v = sum(dxs) / n
        else:
            adx_v = (adx_v * (n - 1) + dx) / n
        adx[i] = adx_v
    return pdi, mdi, adx
