"""변동성 레짐과 상위 시간축 방향을 날짜별로 계산한다 (백테스트 필터용).

모든 값은 그날 종가까지의 정보만 쓴다. 상위 시간축은 이미 마감된 봉만 본다.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from datetime import date, timedelta

from minerva.strategy.indicators import Bar, resample, sma

SPIKE_RATIO = 1.3  # 50일선 대비 이만큼 높으면 스파이크
SPIKE_LOOKBACK = 20


def vix_weekly(vix: Sequence[Bar]) -> list[tuple[date, float | None, float | None]]:
    """VIX 주봉별 (그 주 마지막 거래일, 5주선, 10주선)."""
    wk = resample(vix, "W")
    closes = [b.close for b in wk]
    return list(zip([b.day for b in wk], sma(closes, 5), sma(closes, 10)))


def weekly_gate_at(
    weekly: list[tuple[date, float | None, float | None]],
    day: date,
    keys: list[date] | None = None,
) -> tuple:
    """그 날짜에 알 수 있던 VIX 주봉 (5주선, 10주선, 매수 허용 여부).

    지난주까지 마감된 주봉만 쓴다. 이번 주 봉은 진행 중이라 미래 정보다.
    5주선이 10주선 아래(데드크로스 상태)면 매수를 허용한다.
    """
    monday = day - timedelta(days=day.weekday())
    keys = keys if keys is not None else [w[0] for w in weekly]
    i = bisect.bisect_left(keys, monday) - 1
    if i < 0 or weekly[i][1] is None or weekly[i][2] is None:
        return None, None, False
    _, w5, w10 = weekly[i]
    return w5, w10, w5 < w10


def gated_regime(raw: str | None, calm: bool) -> str:
    """일봉 레짐에 주봉 5<10 관문을 씌운다: 관문이 열리면 R0·R1(매수 허용), 닫히면 R2·R3.

    관문이 열리면 일봉이 R2·R3여도 R1(비중 100%)로 본다. 백테스트(2010~)에서 일봉 레짐으로
    비중을 줄이는 것보다 수익이 높았다(1,000만 원 연 +4.8% vs +4.4%, 낙폭 −10.1% vs −9.3%).
    관문이 닫히면 인버스 비중도 1/2 이하가 된다.
    """
    if calm:
        return raw if raw in ("R0", "R1") else "R1"
    return raw if raw in ("R2", "R3") else "R2"


def vix_regimes(vix: Sequence[Bar]) -> dict[date, str]:
    """날짜 → R0(평온)/R1(잔물결)/R2(변동성 추세)/R3(패닉).

    - R3: 종가가 5·10·20·50일선 모두 위이고 50일선의 1.3배 이상
    - R2: 종가가 50일선 위
    - R1: 종가가 10일선 또는 20일선 위
    - R0: 그 외
    """
    closes = [b.close for b in vix]
    ma = {n: sma(closes, n) for n in (5, 10, 20, 50)}
    out = {}
    for i, b in enumerate(vix):
        m = {n: ma[n][i] for n in ma}
        if m[50] is None:
            continue
        c = b.close
        if all(c > v for v in m.values()) and c >= SPIKE_RATIO * m[50]:
            out[b.day] = "R3"
        elif c > m[50]:
            out[b.day] = "R2"
        elif c > m[10] or c > m[20]:
            out[b.day] = "R1"
        else:
            out[b.day] = "R0"
    return out


def vix_normalizing(vix: Sequence[Bar]) -> dict[date, bool]:
    """최근 20일 안에 스파이크(50일선 x1.3 이상)가 있었고, 지금은 5일선 아래로 내려온 날."""
    closes = [b.close for b in vix]
    ma5, ma50 = sma(closes, 5), sma(closes, 50)
    out = {}
    for i, b in enumerate(vix):
        if ma50[i] is None or ma5[i] is None:
            continue
        lo = max(0, i - SPIKE_LOOKBACK + 1)
        spiked = any(
            ma50[j] is not None and closes[j] >= SPIKE_RATIO * ma50[j] for j in range(lo, i + 1)
        )
        out[b.day] = spiked and b.close < ma5[i]
    return out


def htf_direction(daily: Sequence[Bar], frame: str, n: int) -> dict[date, bool]:
    """일봉 날짜 → 직전에 마감된 상위 봉(W/M)의 종가가 n봉 이평선 위인지."""
    hb = resample(daily, frame)
    ma = sma([b.close for b in hb], n)
    # 상위 봉 i는 hb[i].day(그 기간 마지막 거래일)에 마감된다
    out: dict[date, bool] = {}
    k = -1
    for b in daily:
        while k + 1 < len(hb) and hb[k + 1].day < b.day:
            k += 1
        if k >= 0 and ma[k] is not None:
            out[b.day] = hb[k].close > ma[k]
    return out
