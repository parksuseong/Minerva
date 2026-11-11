"""SCR 실험실의 계산(파이썬 쌍둥이). 화면 계산은 `static/scrlab_core.js`가 하고, 이 모듈은
같은 순서로 같은 계산을 해서 테스트로 JS 결과를 맞춰 본다 (사용자 요청 2026-11-08).

SCR 3단 신호: 마감된 봉만 쓴다. 최근 W봉(현재 봉 포함) 안에 Stoch fast %K·CCI(hlc3)·RSI(Wilder)가
모두 기준선을 다시 넘어오면 신호다. 조건이 처음 맞은 봉에만 표시하고,
이어서 맞는 동안은 다시 찍지 않는다.
원본은 백테스트 `btc_fractal.py`의 stoch·cci·rsi·crossed·recent·scr3.
"""

from __future__ import annotations

from typing import Optional

Num = Optional[float]

# 기본값: 사용자 지정 2026-11-08 (Stoch 9, CCI 20, RSI 21)
DEFAULTS: dict = {
    "stoch_n": 9,
    "stoch_lo": 20.0,
    "stoch_hi": 80.0,
    "cci_n": 20,
    "cci_lo": -100.0,
    "cci_hi": 100.0,
    "rsi_n": 21,
    "rsi_lo": 30.0,
    "rsi_hi": 70.0,
    "w": 3,
    "need": 3,
    "mode": "cross",  # cross = 기준선 다시 넘어옴, zone = 구간 안에 있음
}


def stoch(h: list[float], lo: list[float], c: list[float], n: int) -> list[Num]:
    out: list[Num] = [None] * len(c)
    for i in range(n - 1, len(c)):
        hh = max(h[i - n + 1 : i + 1])
        ll = min(lo[i - n + 1 : i + 1])
        rng = hh - ll
        out[i] = 50.0 if rng == 0 else (c[i] - ll) / rng * 100
    return out


def cci(h: list[float], lo: list[float], c: list[float], n: int) -> list[Num]:
    tp = [(h[i] + lo[i] + c[i]) / 3 for i in range(len(c))]
    out: list[Num] = [None] * len(c)
    for i in range(n - 1, len(c)):
        win = tp[i - n + 1 : i + 1]
        s = 0.0
        for v in win:
            s += v
        m = s / n
        d = 0.0
        for v in win:
            d += abs(v - m)
        md = d / n
        out[i] = 0.0 if md == 0 else (tp[i] - m) / (0.015 * md)
    return out


def rsi(c: list[float], n: int) -> list[Num]:
    out: list[Num] = [None] * len(c)
    if len(c) <= n:
        return out

    def val(ag: float, al: float) -> float:
        return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)

    ag = al = 0.0
    for i in range(1, n + 1):
        d = c[i] - c[i - 1]
        ag += max(d, 0.0)
        al += max(-d, 0.0)
    ag /= n
    al /= n
    out[n] = val(ag, al)
    for i in range(n + 1, len(c)):
        d = c[i] - c[i - 1]
        ag = (ag * (n - 1) + max(d, 0.0)) / n
        al = (al * (n - 1) + max(-d, 0.0)) / n
        out[i] = val(ag, al)
    return out


def crossed(x: list[Num], line: float, up: bool) -> list[bool]:
    out = [False] * len(x)
    for i in range(1, len(x)):
        p, v = x[i - 1], x[i]
        if p is None or v is None:
            continue
        out[i] = (p <= line < v) if up else (p >= line > v)
    return out


def in_zone(x: list[Num], line: float, below: bool) -> list[bool]:
    return [v is not None and (v < line if below else v > line) for v in x]


def recent(flags: list[bool], w: int) -> list[bool]:
    return [any(flags[max(0, i - w + 1) : i + 1]) for i in range(len(flags))]


def signals(h: list[float], lo: list[float], c: list[float], params: dict | None = None) -> dict:
    """롱·숏 신호 봉 번호와 지표 값. 같은 봉에 둘 다 나올 수 있다(원본과 같음)."""
    p = {**DEFAULTS, **(params or {})}
    s = stoch(h, lo, c, int(p["stoch_n"]))
    cc = cci(h, lo, c, int(p["cci_n"]))
    r = rsi(c, int(p["rsi_n"]))
    w, need = int(p["w"]), int(p["need"])
    zone = p["mode"] == "zone"

    def side(long: bool) -> list[int]:
        lines = (
            (s, p["stoch_lo"] if long else p["stoch_hi"]),
            (cc, p["cci_lo"] if long else p["cci_hi"]),
            (r, p["rsi_lo"] if long else p["rsi_hi"]),
        )
        hits = [
            recent(in_zone(x, line, long) if zone else crossed(x, line, long), w)
            for x, line in lines
        ]
        on = [sum(f[i] for f in hits) >= need for i in range(len(c))]
        return [i for i in range(len(c)) if on[i] and not (i > 0 and on[i - 1])]

    return {"long": side(True), "short": side(False), "stoch": s, "cci": cc, "rsi": r}


# ---------------------------------------------------------------- 신호까지 거리(ScrView.linePrice)


def _last_value(name: str, h: list[float], lo: list[float], c: list[float], p: dict):
    """다음 봉이 x(고·저·종 모두 x)에 마감할 때 그 지표 값을 돌려주는 함수.
    전체를 다시 계산하는 것과 같은 값을 마지막 칸만 계산해 낸다(봉 50개 × 40번 반복이라)."""
    if name == "Stoch":
        n = int(p["stoch_n"])
        if len(c) < n:
            return lambda x: stoch(h + [x], lo + [x], c + [x], n)[-1]
        hh, ll = h[len(h) - n + 1 :], lo[len(lo) - n + 1 :]
        mh = max(hh) if hh else float("-inf")
        ml = min(ll) if ll else float("inf")

        def st(x: float):
            top, bot = max(mh, x), min(ml, x)
            rng = top - bot
            return 50.0 if rng == 0 else (x - bot) / rng * 100

        return st
    if name == "CCI":
        n = int(p["cci_n"])
        if len(c) < n:
            return lambda x: cci(h + [x], lo + [x], c + [x], n)[-1]
        tp = [(h[i] + lo[i] + c[i]) / 3 for i in range(len(c) - n + 1, len(c))]

        def cc(x: float):
            win = tp + [(x + x + x) / 3]
            s = 0.0
            for v in win:
                s += v
            m = s / n
            d = 0.0
            for v in win:
                d += abs(v - m)
            md = d / n
            return 0.0 if md == 0 else (win[-1] - m) / (0.015 * md)

        return cc
    n = int(p["rsi_n"])
    if len(c) <= n:  # 앞부분은 짧으니 통째로 다시 계산한다
        return lambda x: rsi(c + [x], n)[-1]
    ag = al = 0.0
    for i in range(1, n + 1):
        d = c[i] - c[i - 1]
        ag += max(d, 0.0)
        al += max(-d, 0.0)
    ag /= n
    al /= n
    for i in range(n + 1, len(c)):
        d = c[i] - c[i - 1]
        ag = (ag * (n - 1) + max(d, 0.0)) / n
        al = (al * (n - 1) + max(-d, 0.0)) / n

    def rs(x: float):
        d = x - c[-1]
        g = (ag * (n - 1) + max(d, 0.0)) / n
        lo_ = (al * (n - 1) + max(-d, 0.0)) / n
        return 100.0 if lo_ == 0 else 100 - 100 / (1 + g / lo_)

    return rs


def line_price(
    h: list[float], lo: list[float], c: list[float], name: str, line: float, p: dict
) -> float | None:
    """다음 봉이 어디에 마감하면(꼬리 없는 봉) 지표가 기준선에 닿나. ±15% 안에서 40번 반으로 나눈다
    (화면 `ScrView.linePrice`와 같다). 범위 밖이면 None."""
    at = _last_value(name, h, lo, c, {**DEFAULTS, **p})
    last = c[-1]
    a, b = last * 0.85, last * 1.15
    va, vb = at(a), at(b)
    if va is None or vb is None or line < va or line > vb:
        return None
    for _ in range(40):
        mid = (a + b) / 2
        if at(mid) < line:
            a = mid
        else:
            b = mid
    return (a + b) / 2


def near(h: list[float], lo: list[float], c: list[float], params: dict | None = None) -> dict:
    """롱·숏 각각 신호까지 거리(화면 '신호 가까운 순'과 같은 계산).
    {"long"|"short": {"fired", "k", "hit": [이름], "need": [{"name", "price", "pct"} 또는
    {"name", "far": True}]}}"""
    p = {**DEFAULTS, **(params or {})}
    res = signals(h, lo, c, p)
    n = len(c) - 1
    last = c[-1]
    w = int(p["w"])
    out: dict = {}
    for side, long in (("long", True), ("short", False)):
        items = (
            ("Stoch", res["stoch"], p["stoch_lo"] if long else p["stoch_hi"]),
            ("CCI", res["cci"], p["cci_lo"] if long else p["cci_hi"]),
            ("RSI", res["rsi"], p["rsi_lo"] if long else p["rsi_hi"]),
        )
        hit, need = [], []
        for name, x, line in items:
            if recent(crossed(x, line, long), w)[n]:
                hit.append(name)
                continue
            v = x[n]
            if v is None or not (v < line if long else v > line):
                need.append({"name": name, "far": True})
                continue
            px = line_price(h, lo, c, name, line, p)
            need.append(
                {
                    "name": name,
                    "price": px,
                    "pct": None if px is None else round((px / last - 1) * 100, 3),
                }
            )
        out[side] = {"fired": n in res[side], "k": len(hit), "hit": hit, "need": need}
    return out
