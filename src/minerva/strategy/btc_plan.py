"""BTCUSDT 규율 도우미: 오늘의 판, 손절·크기 계산, 아침 브리핑. 주문은 내지 않는다.

사용자 원칙(2026-11-08 19:09 KST) "덜 잃고 크게 먹기": 손절·크기·추적 청산은 기계가 정하고
진입 여부만 사람이 정한다. 웹 서버와 이 모듈에는 주문 권한이 없다.

- 큰 움직임 표시: 일봉·4시간·15분 방향(CCI20>0·RSI14>50 = 상방, 둘 다 아래 = 하방)이 모두 하방.
  등록 시험(2026-11-08 deep, V5)에서 다음 24시간 범위가 무작위의 1.03~1.20배, 방향은 반반이었다.
- 일봉 방향: 추세 이탈 규칙(T6D)의 마지막 신호 방향(마감된 일봉만). 백테스트 원본
  `pine_ports.s6_trend_dev` + `daily_wave.features`의 GT 문을 numpy 없이 옮겼다.
- 실행: `python -m minerva.strategy.btc_plan brief` (텔레그램 아침 브리핑)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from minerva.strategy.indicators import Bar

NAN = float("nan")
FEE = 0.0005  # 테이커 편도
MAX_LEV = 5.0  # 계좌 5배 교차
QTY_STEP = 0.001
MIN_NOTIONAL = 100.0  # 바이낸스 BTCUSDT 최소 주문 금액(USDT)
RISK_CHOICES = (0.5, 1.0, 2.0)
BIG_FRAMES = ("1d", "4h", "15m")
T6D_LABEL = {1: "롱 우위", -1: "숏 우위", 0: "중립"}


def _nan(v: float) -> bool:
    return v != v


# ---------------- 파인 내장 지표 (pine_ports와 같은 정의) ----------------
def _sma(x: Sequence[float], n: int) -> list[float]:
    out = [NAN] * len(x)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1 : i + 1]
        if not any(_nan(v) for v in w):
            out[i] = sum(w) / n
    return out


def _rec(x: Sequence[float], n: int, alpha: float) -> list[float]:
    out = [NAN] * len(x)
    i0 = next((i for i, v in enumerate(x) if not _nan(v)), None)
    if i0 is None or len(x) - i0 < n:
        return out
    seed = i0 + n - 1
    prev = sum(x[i0 : seed + 1]) / n
    out[seed] = prev
    for i in range(seed + 1, len(x)):
        v = x[i]
        if not _nan(v):
            prev = alpha * v + (1 - alpha) * prev
        out[i] = prev
    return out


def _ema(x: Sequence[float], n: int) -> list[float]:
    return _rec(x, n, 2.0 / (n + 1))


def _rma(x: Sequence[float], n: int) -> list[float]:
    return _rec(x, n, 1.0 / n)


def _stdev(x: Sequence[float], n: int) -> list[float]:
    m = _sma(x, n)
    m2 = _sma([v * v for v in x], n)
    return [math.sqrt(max(b - a * a, 0.0)) if not _nan(a) else NAN for a, b in zip(m, m2)]


def _shift(x: Sequence[float], k: int = 1) -> list[float]:
    return [NAN] * min(k, len(x)) + list(x[: len(x) - k])


def _tr(h: Sequence[float], lo: Sequence[float], c: Sequence[float]) -> list[float]:
    out = [h[0] - lo[0]] if h else []
    for i in range(1, len(c)):
        pc = c[i - 1]
        out.append(max(h[i] - lo[i], abs(h[i] - pc), abs(lo[i] - pc)))
    return out


def atr(h: Sequence[float], lo: Sequence[float], c: Sequence[float], n: int = 14) -> list[float]:
    return _rma(_tr(h, lo, c), n)


def _cross_over(a: Sequence[float], b: Sequence[float]) -> list[bool]:
    pa, pb = _shift(a), _shift(b)
    return [x > y and p <= q for x, y, p, q in zip(a, b, pa, pb)]


def _cross_under(a: Sequence[float], b: Sequence[float]) -> list[bool]:
    pa, pb = _shift(a), _shift(b)
    return [x < y and p >= q for x, y, p, q in zip(a, b, pa, pb)]


def _fixnan(x: Sequence[float]) -> list[float]:
    out, last = [], NAN
    for v in x:
        if _nan(v) or math.isinf(v):
            out.append(last)
        else:
            last = v
            out.append(v)
    return out


def _aroon_bars(x: Sequence[float], n: int, high: bool) -> list[float]:
    """ta.highestbars/lowestbars: 최고(최저) 봉까지의 음수 거리, 동률은 최근 봉."""
    out = [NAN] * len(x)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1 : i + 1][::-1]
        best = max(w) if high else min(w)
        out[i] = -float(w.index(best))
    return out


def _supertrend_dir(h, lo, c, factor: float, n: int) -> list[float]:
    a = atr(h, lo, c, n)
    d = [1.0] * len(c)
    lower = upper = prev_super = NAN
    for i in range(len(c)):
        src = (h[i] + lo[i]) / 2
        lb, ub = src - factor * a[i], src + factor * a[i]
        if i > 0 and not _nan(lower):
            lb = lb if (lb > lower or c[i - 1] < lower) else lower
        if i > 0 and not _nan(upper):
            ub = ub if (ub < upper or c[i - 1] > upper) else upper
        if i == 0 or _nan(a[i - 1]):
            di = 1
        elif prev_super == upper:
            di = -1 if c[i] > ub else 1
        else:
            di = 1 if c[i] < lb else -1
        lower, upper = lb, ub
        prev_super = lb if di == -1 else ub
        d[i] = float(di)
    return d


def _stc(c: Sequence[float], n: int = 14, fast: int = 25, slow: int = 60) -> list[float]:
    m = [a - b for a, b in zip(_ema(c, fast), _ema(c, slow))]
    f = 1.5
    out = [NAN] * len(m)
    c5 = d6 = 0.0
    ddd = e5 = None
    win: list[float] = []
    ddd_hist: list[float] = []
    for i, mi in enumerate(m):
        win.append(mi)
        if len(win) > n:
            win.pop(0)
        w = [x for x in win if not _nan(x)]
        if len(win) == n and len(w) == n:
            lo_, hi_ = min(w), max(w)
            c5 = (mi - lo_) / (hi_ - lo_) * 100 if hi_ > lo_ else c5
        ddd = c5 if ddd is None else ddd + f * (c5 - ddd)
        ddd_hist.append(ddd)
        if len(ddd_hist) > n:
            ddd_hist.pop(0)
        if len(ddd_hist) == n:
            lo_, hi_ = min(ddd_hist), max(ddd_hist)
            d6 = (ddd - lo_) / (hi_ - lo_) * 100 if hi_ > lo_ else d6
        e5 = d6 if e5 is None else e5 + f * (d6 - e5)
        out[i] = e5
    return out


def trend_dev(h: Sequence[float], lo: Sequence[float], c: Sequence[float]) -> list[int]:
    """추세 이탈 규칙(T6D) 봉별 신호: +1 롱, −1 숏, 0 없음. 마감된 봉만 넣는다."""
    n = len(c)
    if n < 2:
        return [0] * n
    up2 = [NAN] + [h[i] - h[i - 1] for i in range(1, n)]
    dn2 = [NAN] + [lo[i - 1] - lo[i] for i in range(1, n)]
    pdm = [NAN if _nan(u) else (u if u > d and u > 0 else 0.0) for u, d in zip(up2, dn2)]
    mdm = [NAN if _nan(d) else (d if d > u and d > 0 else 0.0) for u, d in zip(up2, dn2)]
    trur = _rma(_tr(h, lo, c), 15)

    def di(dm: list[float]) -> list[float]:
        r = _rma(dm, 15)
        return _fixnan(
            [NAN if _nan(a) or _nan(t) or t == 0 else 100 * a / t for a, t in zip(r, trur)]
        )

    plus, minus = di(pdm), di(mdm)
    dmi_up, dmi_dn = _cross_over(plus, minus), _cross_over(minus, plus)
    src = [(a + b + x) / 3 for a, b, x in zip(h, lo, c)]
    basis, dev = _sma(src, 13), [2.3 * v for v in _stdev(src, 13)]
    bblong = [s < b + d for s, b, d in zip(src, basis, dev)]
    bbshort = [s > b - d for s, b, d in zip(src, basis, dev)]
    au = [100 * (v + 5) / 5 for v in _aroon_bars(h, 6, True)]
    al = [100 * (v + 5) / 5 for v in _aroon_bars(lo, 6, False)]
    aroon_bull, aroon_bear = _cross_over(au, al), _cross_under(au, al)
    s = _stc(c)
    s1, s2, s3 = _shift(s, 1), _shift(s, 2), _shift(s, 3)
    macd = [a - b for a, b in zip(_ema(c, 15), _ema(c, 200))]
    amacd = _ema(macd, 25)
    mom0 = [a - b for a, b in zip(c, _shift(c, 10))]
    mom1 = [a - b for a, b in zip(mom0, _shift(mom0))]
    d = _supertrend_dir(h, lo, c, 2.0, 200)
    dd = [NAN] + [d[i] - d[i - 1] for i in range(1, n)]
    out = []
    for i in range(n):
        anyl = (
            (mom0[i] > 0 and mom1[i] > 0)
            or macd[i] > amacd[i]
            or dd[i] < 0
            or (s3[i] >= s2[i] and s2[i] <= s1[i] and s[i] < 60)
            or s[i] < 25
            or al[i] > au[i]
            or aroon_bull[i]
        )
        anys = (
            (mom0[i] < 0 and mom1[i] < 0)
            or macd[i] < amacd[i]
            or dd[i] > 0
            or (s3[i] <= s2[i] and s2[i] >= s1[i] and s[i] > 40)
            or s[i] > 75
            or al[i] < au[i]
            or aroon_bear[i]
        )
        lc = dmi_up[i] and bblong[i] and anyl
        sc = dmi_dn[i] and bbshort[i] and anys
        # 같은 봉이면 파인에서 숏 진입이 나중에 실행되어 숏이 남는다
        out.append(-1 if sc else (1 if lc else 0))
    return out


def t6d_state(
    h: Sequence[float], lo: Sequence[float], c: Sequence[float]
) -> tuple[int, int | None]:
    """일봉 방향 문 = 마지막 T6D 신호의 방향과 그 봉 위치(신호가 없으면 0, None)."""
    sig = trend_dev(h, lo, c)
    for i in range(len(sig) - 1, -1, -1):
        if sig[i]:
            return sig[i], i
    return 0, None


def last_pivot(x: Sequence[float], low: bool, left: int = 2, right: int = 2) -> int | None:
    """좌우 봉보다 엄격히 낮은(높은) 마지막 봉 위치. 오른쪽 봉이 마감돼야 확정된다."""
    for i in range(len(x) - 1 - right, left - 1, -1):
        others = list(x[i - left : i]) + list(x[i + 1 : i + right + 1])
        if (x[i] < min(others)) if low else (x[i] > max(others)):
            return i
    return None


def big_move(frames: list[dict]) -> bool:
    """일봉·4시간·15분 방향이 모두 하방이면 True (방향 예측이 아니라 범위 예고)."""
    ways = {f["frame"]: f["direction"] for f in frames}
    return all(ways.get(k) == "하방" for k in BIG_FRAMES)


def _r(v: float | None, n: int = 1) -> float | None:
    return None if v is None or _nan(v) else round(v, n)


def today(
    frames: list[dict],
    daily: list[Bar],
    daily_at: list[datetime],
    h4: list[Bar],
    price: float,
) -> dict:
    """오늘의 판: 큰 움직임 표시, 일봉 방향(T6D), 손절 후보(일봉 ATR, 4시간 스윙)."""
    dh, dl, dc = [b.high for b in daily], [b.low for b in daily], [b.close for b in daily]
    state, at = t6d_state(dh, dl, dc)
    atr_d = atr(dh, dl, dc, 14)[-1]
    hh, hl, hc = [b.high for b in h4], [b.low for b in h4], [b.close for b in h4]
    atr_4h = atr(hh, hl, hc, 14)[-1]
    lo_i, hi_i = last_pivot(hl, low=True), last_pivot(hh, low=False)
    return {
        "price": _r(price, 1),
        "big_move": big_move(frames),
        "big_frames": {f["frame"]: f["direction"] for f in frames if f["frame"] in BIG_FRAMES},
        "t6d": state,
        "t6d_label": T6D_LABEL[state],
        "t6d_since": daily_at[at].strftime("%Y-%m-%d") if at is not None else None,
        "atr_d": _r(atr_d, 1),
        "atr_d_pct": _r(atr_d / price * 100, 2) if price and not _nan(atr_d) else None,
        "atr_4h": _r(atr_4h, 1),
        "swing_low_4h": _r(hl[lo_i], 1) if lo_i is not None else None,
        "swing_high_4h": _r(hh[hi_i], 1) if hi_i is not None else None,
        "prev_4h_low": _r(hl[-1], 1) if hl else None,
        "prev_4h_high": _r(hh[-1], 1) if hh else None,
        "stops": {
            "long": {
                "atr": _r(price - atr_d, 1),
                "swing": _r(hl[lo_i], 1) if lo_i is not None else None,
            },
            "short": {
                "atr": _r(price + atr_d, 1),
                "swing": _r(hh[hi_i], 1) if hi_i is not None else None,
            },
        },
    }


# ---------------- 손절·크기 계산 ----------------
def _floor_step(q: float, step: float = QTY_STEP) -> float:
    return round(math.floor(q / step + 1e-9) * step, 3)


def size_position(
    side: str,
    entry: float,
    stop: float,
    equity: float,
    risk_pct: float = 1.0,
    available: float | None = None,
    fee: float = FEE,
    max_lev: float = MAX_LEV,
) -> dict:
    """손절에 닿으면 왕복 수수료 포함 equity×risk_pct%를 잃는 수량. 5배를 넘으면 5배로 줄인다."""
    if side not in ("long", "short"):
        raise ValueError("side는 long 또는 short")
    if entry <= 0 or stop <= 0 or equity <= 0 or risk_pct <= 0:
        raise ValueError("가격·자금·위험은 0보다 커야 해요")
    sign = 1 if side == "long" else -1
    if (entry - stop) * sign <= 0:
        raise ValueError("롱은 손절가가 진입가보다 낮고, 숏은 높아야 해요")
    dist = abs(entry - stop)
    per_btc = dist + fee * (entry + stop)  # 손절 시 BTC 1개당 손실(왕복 수수료 포함)
    budget = equity * risk_pct / 100
    want = _floor_step(budget / per_btc)
    margin_base = min(equity, available) if available else equity
    cap = _floor_step(margin_base * max_lev / entry)
    capped = want > cap
    qty = cap if capped else want
    notional = qty * entry
    loss = qty * per_btc
    warnings = []
    if capped:
        warnings.append(
            f"레버리지가 {max_lev:g}배를 넘어 {max_lev:g}배로 줄였어요: 위험 "
            f"{risk_pct:g}% → {loss / equity * 100:.2f}%"
        )
    if qty <= 0:
        warnings.append("수량이 최소 단위 0.001 BTC보다 작아요. 손절을 좁히거나 위험을 올려야 해요")
    elif notional < MIN_NOTIONAL:
        warnings.append(
            f"주문 금액 {notional:,.1f} USDT가 바이낸스 최소 {MIN_NOTIONAL:g} USDT보다 작아요"
        )
    be = entry * (1 + fee) / (1 - fee) if side == "long" else entry * (1 - fee) / (1 + fee)
    return {
        "side": side,
        "entry": round(entry, 1),
        "stop": round(stop, 1),
        "stop_pct": round(dist / entry * 100, 2),
        "equity": round(equity, 2),
        "risk_pct": risk_pct,
        "qty": qty,
        "qty_wanted": want,
        "capped": capped,
        "notional": round(notional, 2),
        "leverage": round(notional / equity, 2),
        "leverage_wanted": round(want * entry / equity, 2),
        "loss_at_stop": round(loss, 2),
        "loss_pct": round(loss / equity * 100, 2),
        "fees_round_trip": round(qty * fee * (entry + stop), 2),
        "r": round(dist, 1),
        "r_levels": {f"{k}R": round(entry + sign * k * dist, 1) for k in (1, 2, 3)},
        "breakeven_stop": round(be, 1),
        "warnings": warnings,
    }


# ---------------- 아침 브리핑 ----------------
def brief_text(data: dict) -> str:
    p = data.get("plan") or {}
    lines = [f"☀️ BTC 오늘의 판 ({data.get('generated_at', '')[:16].replace('T', ' ')} KST)"]
    lines.append(f"현재가 {p.get('price', 0):,.1f} USDT (선물)")
    if p.get("big_move"):
        lines.append("큰 움직임 가능성 높음 (방향은 반반): 일봉·4시간·15분 모두 하방")
    else:
        lines.append("큰 움직임 표시 없음")
    since = f", {p['t6d_since']} 신호부터" if p.get("t6d_since") else ""
    lines.append(f"일봉 방향(T6D): {p.get('t6d_label', '-')}{since}")
    if p.get("atr_d_pct") is not None:
        lines.append(f"손절 거리 제안: 일봉 ATR 1배 = {p['atr_d']:,.1f} ({p['atr_d_pct']}%)")
    lines.append("손절·크기는 화면의 계산기로. 진입 여부만 직접 정하세요. 주문은 나가지 않아요.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from minerva import notify
    from minerva.strategy import btc

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["brief"])
    ap.add_argument("--dry", action="store_true", help="보내지 않고 출력만")
    args = ap.parse_args(argv)
    path = Path(os.environ.get("MINERVA_BTC_OUT", str(btc.OUT))).expanduser()
    data = json.loads(path.read_text()) if path.exists() else {}
    fresh = data.get("generated_at") and "plan" in data
    if fresh:
        age = datetime.now(btc.KST) - datetime.fromisoformat(data["generated_at"])
        fresh = age.total_seconds() < 3600
    if not fresh:
        data = btc.build()
    text = brief_text(data)
    print(text)
    if not args.dry and not notify.send(text):
        raise SystemExit("텔레그램 전송 실패")


if __name__ == "__main__":
    main()
