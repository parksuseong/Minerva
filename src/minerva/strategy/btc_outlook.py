"""비트코인 큰 그림: 4시간·일·주·월봉 상태와 앞으로 볼 가격대 요약 (사용자 요청 2026-11-09 08:23
"거기는 4시간 일봉 주봉 월봉기준으로 분석을 해줘, 향후 미래나 방향같은거를 요약해서 보여줘야지").

표시 전용이다. 자동매매 엔진(15분 SCR ①)의 진입·방향과 연결되지 않는다.
방향 판독은 원문 SCR 읽기(시간축 일치, 3단 돌파·이탈)와 이평선 위치로 한다.
예측 확률은 붙이지 않는다.
BTC 검증에서 이런 방향 판독의 다음 움직임 적중은 50% 안팎이었다(2026-11-08 연구).
"""

from __future__ import annotations

from minerva.strategy.indicators import Bar, TMState, sma, tm_states

FRAMES = (("4h", "4시간봉"), ("1d", "일봉"), ("1w", "주봉"), ("1M", "월봉"))
# 봉마다 볼 이평선(월봉 20 = 원문 "보라 20개월선", 볼린저 중심)
MAS = {"4h": (20, 50, 200), "1d": (20, 50, 200), "1w": (10, 20, 50), "1M": (10, 20, 50)}
SWING_N = 20
ATR_N = 14
UNIT = {"4h": "4시간", "1d": "하루", "1w": "한 주", "1M": "한 달"}


def _r(v: float | None, n: int = 1) -> float | None:
    return None if v is None else round(v, n)


def atr_pct(bars: list[Bar], n: int = ATR_N) -> float | None:
    """Wilder ATR(n)을 마지막 종가 대비 %로. 다음 봉 한 개의 보통 움직임 폭."""
    if len(bars) < n + 2:
        return None
    tr = [
        max(b.high - b.low, abs(b.high - p.close), abs(b.low - p.close))
        for p, b in zip(bars, bars[1:])
    ]
    a = sum(tr[:n]) / n
    for x in tr[n:]:
        a += (x - a) / n
    return a / bars[-1].close * 100


def direction(s: TMState) -> str:
    if s.cci is None or s.rsi is None:
        return "중립"
    if s.cci > 0 and s.rsi > 50:
        return "상방"
    if s.cci < 0 and s.rsi < 50:
        return "하방"
    return "중립"


def zone(s: TMState) -> str:
    """지표 위치. 과열·침체는 세 지표 중 2개 이상이 기준선 밖일 때."""
    if s.stoch is None or s.cci is None or s.rsi is None:
        return "-"
    hot = (s.stoch > 80) + (s.cci > 100) + (s.rsi > 70)
    cold = (s.stoch < 20) + (s.cci < -100) + (s.rsi < 30)
    if hot >= 2:
        return "과열"
    if cold >= 2:
        return "침체"
    if hot or cold:
        return "과열 쪽" if hot else "침체 쪽"
    return "보통"


def _last(states: list[TMState], side: str) -> tuple[int, int] | None:
    """(몇 봉 전, 단계). 가장 최근 신호."""
    for k, s in enumerate(reversed(states)):
        level = getattr(s, f"{side}_level")
        if level:
            return k, level
    return None


def frame(key: str, name: str, bars: list[Bar], price: float) -> dict:
    """price: 지금 가격. 이평선·고저까지 거리는 지금 가격 기준(지표는 마감 봉 기준)."""
    states = tm_states(bars)
    s, prev = states[-1], states[-2]
    closes = [b.close for b in bars]
    close = price
    mas = {}
    for n in MAS[key]:
        if len(closes) >= n:
            v = sma(closes, n)[-1]
            mas[str(n)] = {"value": _r(v, 1), "gap_pct": _r((close / v - 1) * 100, 2)}
    win = bars[-SWING_N:]
    hi, lo = max(b.high for b in win), min(b.low for b in win)
    atr = atr_pct(bars)
    stoch_turn = None
    if s.stoch is not None and prev.stoch is not None:
        stoch_turn = (
            "오름" if s.stoch > prev.stoch else "내림" if s.stoch < prev.stoch else "그대로"
        )
    sig = {}
    for side in ("long", "short"):
        hit = _last(states, side)
        sig[side] = {"bars_ago": hit[0], "level": hit[1]} if hit else None
    return {
        "frame": key,
        "name": name,
        "close": _r(closes[-1], 1),
        "direction": direction(s),
        "zone": zone(s),
        "stoch": _r(s.stoch),
        "cci": _r(s.cci),
        "rsi": _r(s.rsi),
        "stoch_turn": stoch_turn,
        "long_level": s.long_level,
        "short_level": s.short_level,
        "last_long": sig["long"],
        "last_short": sig["short"],
        "ma": mas,
        "swing": {
            "n": SWING_N,
            "high": _r(hi, 1),
            "low": _r(lo, 1),
            "high_pct": _r((hi / close - 1) * 100, 2),
            "low_pct": _r((lo / close - 1) * 100, 2),
        },
        "atr_pct": _r(atr, 2),
        "range": {
            "unit": UNIT[key],
            "up": _r(close * (1 + atr / 100), 0) if atr else None,
            "down": _r(close * (1 - atr / 100), 0) if atr else None,
        },
    }


def _ma_line(f: dict) -> str:
    above = [n for n, m in f["ma"].items() if m["gap_pct"] is not None and m["gap_pct"] > 0]
    below = [n for n, m in f["ma"].items() if m["gap_pct"] is not None and m["gap_pct"] <= 0]
    if not below:
        return "이평선 모두 위"
    if not above:
        return "이평선 모두 아래"
    return f"{'·'.join(above)}선 위, {'·'.join(below)}선 아래"


def _pair(a: dict, b: dict) -> str:
    """두 시간축 묶음의 한 줄 판독."""
    da, db = a["direction"], b["direction"]
    if da == db and da != "중립":
        return f"{da} (둘 다)"
    if "중립" in (da, db):
        other = db if da == "중립" else da
        return "중립" if other == "중립" else f"{other} 쪽, 확인 덜 됨"
    return f"엇갈림 ({b['name']} {db}, {a['name']} {da})"


def levels(frames: dict[str, dict], price: float) -> tuple[list[dict], list[dict]]:
    """지금 가격 위 저항·아래 지지 후보(가까운 순). 이평선과 20봉 고저."""
    cands = []
    for key in ("4h", "1d", "1w", "1M"):
        f = frames.get(key)
        if not f:
            continue
        for n, m in f["ma"].items():
            if m["value"]:
                cands.append({"price": m["value"], "what": f"{f['name']} {n}선"})
        cands.append({"price": f["swing"]["high"], "what": f"{f['name']} 20봉 고점"})
        cands.append({"price": f["swing"]["low"], "what": f"{f['name']} 20봉 저점"})
    merged: dict[float, dict] = {}
    for c in cands:
        if c["price"] in merged:
            merged[c["price"]]["what"] += f" · {c['what']}"
        else:
            merged[c["price"]] = dict(c)
    up = sorted((c for c in merged.values() if c["price"] > price), key=lambda c: c["price"])
    dn = sorted((c for c in merged.values() if c["price"] < price), key=lambda c: -c["price"])
    for c in up + dn:
        c["pct"] = _r((c["price"] / price - 1) * 100, 2)
    return up[:3], dn[:3]


def _monthly_last(m1: dict) -> str | None:
    """월봉에서 가장 최근(3봉 안) 신호 하나만. 원문 규칙은 월봉 마감 기준."""
    hits = [(m1[f"last_{side}"], side) for side in ("long", "short") if m1[f"last_{side}"]]
    hits = [(h, side) for h, side in hits if h["bars_ago"] <= 2]
    if not hits:
        return None
    h, side = min(hits, key=lambda x: (x[0]["bars_ago"], -x[0]["level"]))
    when = "지난달 마감" if h["bars_ago"] == 0 else f"{h['bars_ago'] + 1}달 전 마감"
    if side == "short":
        return (
            f"월봉 숏 {h['level']}단 이탈({when}). 원문 규칙: 월봉 스토캐·CCI가 마감 기준으로 "
            "다시 올라서기 전의 반등은 데드캣 후보예요."
        )
    return f"월봉 롱 {h['level']}단 돌파({when}). 큰 흐름이 위로 돌아서는 쪽이에요."


def summarize(frames: dict[str, dict], price: float) -> dict:
    h4, d1, w1, m1 = (frames[k] for k in ("4h", "1d", "1w", "1M"))
    ways = [f["direction"] for f in (h4, d1, w1, m1)]
    big = _pair(w1, m1)
    mid = _pair(h4, d1)
    if all(w == "상방" for w in ways):
        tone, verdict = "up", "네 시간축 모두 상방이에요. 원문 규칙상 강한 상승 추세예요."
    elif all(w == "하방" for w in ways):
        tone, verdict = "down", "네 시간축 모두 하방이에요. 원문 규칙상 강한 하락 추세예요."
    else:
        up, dn = ways.count("상방"), ways.count("하방")
        tone = "up" if up > dn else "down" if dn > up else "mixed"
        head = f"큰 흐름(주·월) {big} · 단기(4시간·일) {mid}. "
        if big.startswith("상방") and mid.startswith("하방"):
            verdict = head + "위 흐름 속 조정이에요. 조정이 어디서 멈추는지 봐야 해요."
        elif big.startswith("하방") and mid.startswith("상방"):
            verdict = head + "아래 흐름 속 반등이에요. 원문 기준으로는 데드캣 반등 후보예요."
        else:
            mix = f"상방 {up} · 하방 {dn} · 중립 {4 - up - dn}"
            verdict = head + f"시간축이 엇갈려요({mix}). 뚜렷한 추세가 아니에요."
    up_lv, dn_lv = levels(frames, price)
    watch = []
    for f in (h4, d1, w1, m1):
        for side, word in (("long", "롱 3단 돌파"), ("short", "숏 3단 이탈")):
            if f[f"{side}_level"] == 3:
                watch.append(f"{f['name']} {word}이 지금 나왔어요.")
        if f["zone"] in ("과열", "침체"):
            watch.append(
                f"{f['name']} {f['zone']} 구간이에요 "
                f"({'되돌림' if f['zone'] == '과열' else '반등'} 주의)."
            )
    m_last = _monthly_last(m1)
    if m_last:
        watch.append(m_last)
    return {
        "tone": tone,
        "verdict": verdict,
        "big": big,
        "mid": mid,
        "ma": {k: _ma_line(frames[k]) for k in ("4h", "1d", "1w", "1M")},
        "resist": up_lv,
        "support": dn_lv,
        "watch": watch,
        "range_day": d1["range"],
        "range_week": w1["range"],
    }


def build(raw: dict[str, list[Bar]], price: float) -> dict:
    """raw: {"4h": 마감 봉, "1d": ..., "1w": ..., "1M": ...}"""
    frames = {k: frame(k, name, raw[k], price) for k, name in FRAMES if len(raw.get(k, [])) >= 30}
    out = {
        "price": _r(price, 1),
        "frames": [frames[k] for k, _ in FRAMES if k in frames],
        "note": "표시 전용 판독이에요. 엔진 진입·방향은 15분 SCR ①만 써요. "
        "이런 방향 판독의 다음 움직임 적중은 BTC 검증에서 50% 안팎이었어요(확률 아님).",
    }
    if len(frames) == len(FRAMES):
        out["summary"] = summarize(frames, price)
    return out
