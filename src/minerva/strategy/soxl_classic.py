"""SOXL 탭의 「SOXX 일봉 지표 분석」 카드 (사용자 요청 2026-11-11). 미검증, 보기만.
화면에는 "원문"이라고 쓰지 않는다(사용자 요청).

원문은 SOXX 일봉(1배)으로 반도체를 읽고, 숏(SOXS)은 SOXX 차트를 뒤집어 본다.
- 지표: Stoch fast %K 5(%D 3), CCI 20(시그널 10), RSI 14(시그널 6) — `tm_states`와 같은 설정.
- 선: 이평 5·10·20·50·120·200·250, 볼린저(20, 2): 가운데 = 20일선.
- 롱 타점: 볼밴 하단(침체) + SCR 3단 돌파 → 10·20일선까지 반등 뒤 다시 하락(데드캣).
- 숏 타점(뒤집은 차트의 20일선 지지 + 스토캐만 돌파 마감): SOXX로는 20일선 근처에서
  스토캐가 과매수(80)에서 아래로 꺾여 마감 → 볼밴 하단 쪽으로. 추세 우위 쪽은 CCI·RSI가
  극단을 안 만들어서 스토캐 하나만 본다.
- 숏 추세 이탈은 20일선이 아니라 50일선 위로 올라설 때(주·월봉이 숏 쪽 상방인 동안).
- 장투: 주봉·월봉 SCR 돌파.
SOXL 환산은 하루 움직임 3배 어림(SOXL = 기준 × (1 + 3 × SOXX 변화율)). 기준은 두 ETF의 같은 날 종가.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from minerva.strategy.indicators import Bar, resample, sma, tm_states

MAS = (5, 10, 20, 50, 120, 200, 250)
BB_N, BB_K = 20, 2.0
BB_TOUCH_BARS = 5  # 볼밴 하단 터치를 몇 봉 안에서 찾을지
TURN_BARS = 2  # 스토캐 꺾임 마감을 몇 봉 안에서 찾을지
NEAR_MA20 = 0.03  # 20일선 근처 = 고가가 20일선 3% 안
PROFILE_BARS, PROFILE_BINS = 60, 12  # 매물대: 최근 60봉을 12칸으로


def _r(v: float | None, n: int = 2) -> float | None:
    return None if v is None else round(v, n)


def bollinger(closes: Sequence[float]) -> tuple[list, list, list]:
    mid = sma(closes, BB_N)
    up: list = [None] * len(closes)
    lo: list = [None] * len(closes)
    for i in range(BB_N - 1, len(closes)):
        sd = statistics.pstdev(closes[i - BB_N + 1 : i + 1])
        up[i], lo[i] = mid[i] + BB_K * sd, mid[i] - BB_K * sd
    return mid, up, lo


def profile(bars: Sequence[Bar]) -> list[dict]:
    """최근 PROFILE_BARS봉 거래량을 가격 칸(대표가 = (고+저+종)/3)에 모아 많은 두 칸."""
    win = list(bars[-PROFILE_BARS:])
    lo, hi = min(b.low for b in win), max(b.high for b in win)
    total = sum(b.volume for b in win)
    if hi <= lo or total <= 0:
        return []
    step = (hi - lo) / PROFILE_BINS
    vol = [0.0] * PROFILE_BINS
    for b in win:
        k = min(int(((b.high + b.low + b.close) / 3 - lo) / step), PROFILE_BINS - 1)
        vol[k] += b.volume
    top = sorted(range(PROFILE_BINS), key=lambda k: -vol[k])[:2]
    return [
        {"lo": _r(lo + k * step), "hi": _r(lo + (k + 1) * step), "pct": _r(vol[k] / total * 100, 1)}
        for k in sorted(top)
    ]


def _frame_scr(bars: Sequence[Bar]) -> dict | None:
    if len(bars) < 30:
        return None
    states = tm_states(bars)
    s = states[-1]

    def last(side: str) -> dict | None:
        for back, x in enumerate(reversed(states)):
            if getattr(x, f"{side}_level") == 3:
                return {"day": x.day.isoformat(), "ago": back}
        return None

    return {
        "day": s.day.isoformat(),
        "stoch": _r(s.stoch, 1),
        "cci": _r(s.cci, 1),
        "rsi": _r(s.rsi, 1),
        "long_level": s.long_level,
        "short_level": s.short_level,
        "last_long3": last("long"),
        "last_short3": last("short"),
    }


def build(soxx: Sequence[Bar], soxl_close: float | None) -> dict:
    """soxx: 마감된 SOXX 일봉. soxl_close: 같은 날 SOXL ETF 종가(환산 기준)."""
    if len(soxx) < 260:
        raise ValueError(f"SOXX 일봉이 {len(soxx)}개뿐이다")
    closes = [b.close for b in soxx]
    last = soxx[-1]
    ref = last.close
    states = tm_states(soxx)
    s = states[-1]
    k = [x.stoch for x in states]
    d = sma([x if x is not None else 0.0 for x in k], 3)
    mid, up, lo = bollinger(closes)
    ma = {n: sma(closes, n)[-1] for n in MAS}

    def to_soxl(x: float | None) -> float | None:
        if x is None or not soxl_close:
            return None
        return _r(soxl_close * (1 + 3 * (x / ref - 1)))

    def level(name: str, x: float | None) -> dict:
        return {"name": name, "soxx": _r(x), "pct": _r((x / ref - 1) * 100) if x else None,
                "soxl": to_soxl(x)}

    lines = [level(f"{n}일선" + (" (볼밴 가운데)" if n == 20 else ""), ma[n]) for n in MAS]
    lines += [level("볼밴 상단", up[-1]), level("볼밴 하단", lo[-1])]

    n = len(soxx)
    touch = [i for i in range(n - BB_TOUCH_BARS, n) if lo[i] is not None and soxx[i].low <= lo[i]]
    scr_long = s.long_level
    long_ok = bool(touch) and scr_long == 3

    near20 = max(b.high for b in soxx[-3:]) >= ma[20] * (1 - NEAR_MA20)
    k_now = k[-1]
    stoch_hot = k_now is not None and k_now >= 80
    # 꺾임 마감: 최근 TURN_BARS봉 안에 %K가 80 이상에서 80 아래로 마감
    turns = [
        i for i in range(n - TURN_BARS, n)
        if k[i - 1] is not None and k[i] is not None and k[i - 1] >= 80 > k[i]
    ]
    stoch_turn = bool(turns)
    short_ok = near20 and stoch_turn

    return {
        "symbol": "SOXX",
        "day": last.day.isoformat(),
        "close": _r(ref),
        "soxl_close": _r(soxl_close),
        "ind": {
            "stoch": _r(k_now, 1), "stoch_d": _r(d[-1], 1),
            "cci": _r(s.cci, 1), "rsi": _r(s.rsi, 1),
            "long_level": s.long_level, "short_level": s.short_level,
        },
        "lines": lines,
        "profile": profile(soxx),
        "long_setup": {
            "ok": long_ok,
            "bb_touch": bool(touch),
            "bb_touch_day": soxx[touch[-1]].day.isoformat() if touch else None,
            "scr_level": scr_long,
            "targets": [level("10일선", ma[10]), level("20일선", ma[20])],
        },
        "short_setup": {
            "ok": short_ok,
            "near_ma20": near20,
            "stoch_hot": stoch_hot,
            "stoch_turn": stoch_turn,
            "turn_day": soxx[turns[-1]].day.isoformat() if turns else None,
            "ma20": level("20일선", ma[20]),
            "target": level("볼밴 하단", lo[-1]),
            "risk": level("20일선 위 마감 = 숏 단기 위험", ma[20]),
            "trend_break": level("50일선 위 = 숏 추세 이탈", ma[50]),
        },
        "weekly": _frame_scr(resample(soxx, "W")),
        "monthly": _frame_scr(resample(soxx, "M")),
        "note": "검증 안 된 판독 규칙이에요. SOXX 일봉(1배) 기준이고 SOXL 값은 하루 3배 어림이에요"
        "(며칠 넘게 가면 3배 상품의 녹아내림 때문에 어긋나요).",
    }
