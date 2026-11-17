"""고서 탭 비교용 퀀트 기법 — 널리 공개된 고전 규칙 여섯 개의 기계 정의.

입력·출력만 있는 순수 함수 모음이다(파일·네트워크·시계 없음, 같은 입력이면 같은 결과).
일봉 형식은 `gose_rules`와 같다: (날짜 'YYYY-MM-DD', 시가, 고가, 저가, 종가), 날짜 오름차순.
모두 롱만, 신호는 i봉 종가에 확정, 체결은 다음 봉 시가. 손절·시간 청산 없음(각 기법의 청산만).

- Q1 터틀 돌파(Richard Dennis·William Eckhardt의 Turtle Trading, System 2):
  종가 > 직전 55일 최고가면 진입, 종가 < 직전 20일 최저가면 청산.
- Q2 골든 크로스: SMA50이 SMA200을 위로 넘으면 진입, 아래로 넘으면 청산.
- Q3 Connors RSI(2)(Larry Connors·Cesar Alvarez, "Short Term Trading Strategies That Work"):
  종가 > SMA200이고 RSI(2, Wilder) < 10이면 진입, 종가 > SMA5면 청산.
- Q4 볼린저 평균 회귀(John Bollinger의 Bollinger Bands, BB(20, 2)):
  종가 > SMA200이고 종가 < 하단 밴드면 진입, 종가 ≥ 중심선(SMA20)이면 청산.
- Q5 시계열 모멘텀(Moskowitz·Ooi·Pedersen 2012 "Time Series Momentum"):
  월말 종가에 12-1개월 수익(252거래일 전 → 21거래일 전) > 0이면 보유, ≤ 0이면 월말에 청산.
- Q6 횡단면 모멘텀(Jegadeesh·Titman 1993 "Returns to Buying Winners and Selling Losers"):
  월말마다 전 종목을 12-1개월 수익으로 줄 세워 상위 10개를 보유, 매달 바꾼다.
  종목 하나로 정할 수 없어 `momentum_portfolio`가 종목 전체를 한꺼번에 본다.

표준편차는 모집단 식(n으로 나눔), 볼린저의 원래 정의와 같다.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import deque
from datetime import date as _date
from datetime import timedelta

from minerva.gose_rules import rsi, sma

LABEL = "퀀트(검증된 공개 기법)"
LOOKBACK = 252  # 12개월
SKIP = 21  # 최근 1개월은 뺀다(단기 반전)
TOP_N = 10

STRATEGIES = {
    "Q1": {
        "name": "터틀 돌파 (55일 고가 돌파 → 20일 저가 이탈)",
        "source": "Turtle Trading (Dennis·Eckhardt), System 2",
    },
    "Q2": {
        "name": "골든 크로스 (SMA50 × SMA200)",
        "source": "이동평균 교차(고전 추세 추종)",
    },
    "Q3": {
        "name": "Connors RSI(2) (SMA200 위 + RSI2 < 10 → 종가 > SMA5)",
        "source": "Connors·Alvarez RSI(2)",
    },
    "Q4": {
        "name": "볼린저 평균 회귀 (SMA200 위 + BB(20,2) 하단 아래 → 중심선)",
        "source": "Bollinger Bands 평균 회귀",
    },
    "Q5": {
        "name": "시계열 모멘텀 (월말 12-1개월 수익 > 0이면 보유)",
        "source": "Moskowitz·Ooi·Pedersen (2012) TSMOM",
    },
    "Q6": {
        "name": "횡단면 모멘텀 (월말 12-1개월 수익 상위 10 보유)",
        "source": "Jegadeesh·Titman (1993) 모멘텀",
    },
}
for _s in STRATEGIES.values():
    _s.update(side="long", timeframe="D", max_hold=None, rule_ids=[_s["source"]])
PER_SYMBOL = ("Q1", "Q2", "Q3", "Q4", "Q5")


# ---------------------------------------------------------------- 지표


def prior_max(values: list, n: int) -> list:
    """i봉 직전 n개(i-n .. i-1)의 최댓값. 앞부분은 None."""
    out: list = [None] * len(values)
    dq: deque = deque()
    for i in range(len(values)):
        if i >= n:
            out[i] = values[dq[0]]
        while dq and values[dq[-1]] <= values[i]:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - n:
            dq.popleft()
    return out


def prior_min(values: list, n: int) -> list:
    """i봉 직전 n개(i-n .. i-1)의 최솟값. 앞부분은 None."""
    neg = prior_max([-v for v in values], n)
    return [None if v is None else -v for v in neg]


def rolling_std(values: list, n: int, mean: list) -> list:
    """모집단 표준편차(n으로 나눔). 앞부분 n-1개는 None."""
    out: list = [None] * len(values)
    for i in range(n - 1, len(values)):
        m = mean[i]
        out[i] = (sum((x - m) ** 2 for x in values[i - n + 1 : i + 1]) / n) ** 0.5
    return out


def momentum(closes: list, i: int, lookback: int = LOOKBACK, skip: int = SKIP) -> float | None:
    """12-1개월 수익: close[i-skip] / close[i-lookback] - 1. 자료가 모자라면 None."""
    if i < lookback or closes[i - lookback] <= 0:
        return None
    return closes[i - skip] / closes[i - lookback] - 1.0


def _is_last_weekday(d: str) -> bool:
    x = _date.fromisoformat(d)
    nxt = x + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt.month != x.month


def month_end(dates: list, i: int) -> bool:
    """i봉 종가가 월말 판단 시점인가. dates[:i+1]만 본다(다음 봉을 보지 않음, 미래 정보 없음).

    - 그날이 그 달 마지막 평일이면 월말.
    - 마지막 평일이 휴장이라 그 달에 판단하지 못했으면(예: 한국 12/31), 새 달 첫 봉 종가에
      늦게 판단한다(그 봉까지의 종가만 쓰고 다음 시가 체결).
    """
    if _is_last_weekday(dates[i]):
        return True
    return i > 0 and dates[i - 1][:7] != dates[i][:7] and not _is_last_weekday(dates[i - 1])


def prepare(bars: list) -> dict:
    """Q1~Q5가 쓰는 지표를 한 번에 계산한다."""
    dates = [b[0] for b in bars]
    highs = [b[2] for b in bars]
    lows = [b[3] for b in bars]
    closes = [b[4] for b in bars]
    sma20 = sma(closes, 20)
    std20 = rolling_std(closes, 20, sma20)
    return {
        "bars": bars,
        "dates": dates,
        "close": closes,
        "hh55": prior_max(highs, 55),
        "ll20": prior_min(lows, 20),
        "sma5": sma(closes, 5),
        "sma20": sma20,
        "bb_lo": [None if m is None else m - 2.0 * s for m, s in zip(sma20, std20)],
        "sma50": sma(closes, 50),
        "sma200": sma(closes, 200),
        "rsi2": rsi(closes, 2),
    }


# ---------------------------------------------------------------- 진입·청산 (i봉 종가)


def _ok(*vals) -> bool:
    return all(v is not None for v in vals)


def entry(sid: str, ctx: dict, i: int) -> str | None:
    """무포지션일 때 i봉 종가 진입 신호 사유(없으면 None)."""
    c = ctx["close"][i]
    if sid == "Q1":
        hh = ctx["hh55"][i]
        return "55일 고가 돌파" if _ok(hh) and c > hh else None
    if sid == "Q2":
        if i < 1:
            return None
        f0, s0, f1, s1 = (
            ctx["sma50"][i - 1],
            ctx["sma200"][i - 1],
            ctx["sma50"][i],
            ctx["sma200"][i],
        )
        return "골든 크로스" if _ok(f0, s0, f1, s1) and f0 <= s0 and f1 > s1 else None
    if sid == "Q3":
        s200, r2 = ctx["sma200"][i], ctx["rsi2"][i]
        return "SMA200 위 · RSI2 < 10" if _ok(s200, r2) and c > s200 and r2 < 10 else None
    if sid == "Q4":
        s200, lo = ctx["sma200"][i], ctx["bb_lo"][i]
        return "SMA200 위 · 볼린저 하단 아래" if _ok(s200, lo) and c > s200 and c < lo else None
    if sid == "Q5":
        if not month_end(ctx["dates"], i):
            return None
        m = momentum(ctx["close"], i)
        return f"월말 12-1개월 {m * 100:+.1f}%" if m is not None and m > 0 else None
    raise KeyError(sid)


def exit_(sid: str, ctx: dict, i: int) -> str | None:
    """보유 중일 때 i봉 종가 청산 신호 사유(없으면 None)."""
    c = ctx["close"][i]
    if sid == "Q1":
        ll = ctx["ll20"][i]
        return "20일 저가 이탈" if _ok(ll) and c < ll else None
    if sid == "Q2":
        if i < 1:
            return None
        f0, s0, f1, s1 = (
            ctx["sma50"][i - 1],
            ctx["sma200"][i - 1],
            ctx["sma50"][i],
            ctx["sma200"][i],
        )
        return "데드 크로스" if _ok(f0, s0, f1, s1) and f0 >= s0 and f1 < s1 else None
    if sid == "Q3":
        s5 = ctx["sma5"][i]
        return "종가 > SMA5" if _ok(s5) and c > s5 else None
    if sid == "Q4":
        mid = ctx["sma20"][i]
        return "중심선(SMA20) 회복" if _ok(mid) and c >= mid else None
    if sid == "Q5":
        if not month_end(ctx["dates"], i):
            return None
        m = momentum(ctx["close"], i)
        return f"월말 12-1개월 {m * 100:+.1f}%" if m is not None and m <= 0 else None
    raise KeyError(sid)


def _ret(fill_px: float, exit_px: float, fee: float) -> float:
    return exit_px * (1 - fee) / (fill_px * (1 + fee)) - 1.0


def simulate(sid: str, bars: list, fee: float = 0.0005, ctx: dict | None = None) -> dict:
    """Q1~Q5를 한 종목 일봉 위에서 돌린다. 반환 형식은 `gose_rules.simulate`와 같다.

    {"trades": [...], "pending": [(신호일, 종류, 종가, 사유)], "open": 보유 중 거래 또는 None}
    거래: entry_i/exit_i, entry_date/exit_date, signal_date, side, fills[(날짜, 가격, 1.0, 사유)],
          exit_px, reason_in, reason_out, ret(수수료 뒤).
    """
    if ctx is None:
        ctx = prepare(bars)
    n = len(bars)
    trades, pending = [], []
    pos = None
    order = None  # (종류, 사유, 신호일)
    for i in range(n):
        o = bars[i][1]
        if order is not None:
            kind, reason, sd = order
            order = None
            if kind == "entry":
                pos = {
                    "sid": sid,
                    "side": "long",
                    "entry_i": i,
                    "entry_date": bars[i][0],
                    "signal_date": sd,
                    "fills": [(bars[i][0], o, 1.0, reason)],
                    "reason_in": reason,
                    "avg": o,
                }
            elif pos is not None:
                pos.update(
                    exit_i=i,
                    exit_date=bars[i][0],
                    exit_px=o,
                    reason_out=reason,
                    ret=_ret(pos["avg"], o, fee),
                )
                trades.append(pos)
                pos = None
        if pos is not None:
            r = exit_(sid, ctx, i)
            if r is not None:
                order = ("exit", r, bars[i][0])
        else:
            r = entry(sid, ctx, i)
            if r is not None:
                order = ("entry", r, bars[i][0])
        if order is not None and i == n - 1:
            pending.append((order[2], order[0], bars[i][4], order[1]))
    return {"trades": trades, "pending": pending, "open": pos}


# ---------------------------------------------------------------- Q6 포트폴리오


def momentum_portfolio(
    series: dict, top: int = TOP_N, fee: float = 0.0005, calendar: list | None = None
) -> dict:
    """Q6: 월말(합친 거래일 달력 기준)마다 12-1개월 수익 상위 `top` 종목을 들고 매달 바꾼다.

    series: {티커: 일봉}. 종목마다 그 월말 날짜 이하 마지막 봉이 같은 달에 있고 252봉 이상이어야
    순위에 든다(같으면 티커 순). 순위 밖으로 나간 종목은 월말 다음 자기 첫 봉 시가에 판다.
    새로 든 종목은 그 매도 체결일(가장 늦은 것, 없으면 달력의 다음 거래일) 이후 자기 첫 봉 시가에
    산다 — 같은 날이면 매도가 먼저라 자리 10개를 넘지 않는다. 장부(종목별 규칙 보유)는 계좌가
    못 산 경우에도 보유로 친다(다른 계좌와 같은 뜻).
    반환: {"trades": [...], "open": [...], "pending": [(티커, 신호일, 종류, 종가, 사유)],
           "rebalances": [(월말 날짜, [상위 티커])]}. 거래에는 `sym`이 붙는다.
    """
    syms = sorted(series)
    dates = {s: [b[0] for b in series[s]] for s in syms}
    closes = {s: [b[4] for b in series[s]] for s in syms}
    cal = calendar or sorted({d for s in syms for d in dates[s]})
    held: dict = {}
    trades, pending, rebal = [], [], []

    def last_idx(s: str, d: str) -> int:
        return bisect_right(dates[s], d) - 1

    def first_after(s: str, d: str, inclusive: bool = False) -> int | None:
        j = (bisect_left if inclusive else bisect_right)(dates[s], d)
        return j if j < len(dates[s]) else None

    for k, d in enumerate(cal):
        if not month_end(cal, k):
            continue
        scores = []
        for s in syms:
            i = last_idx(s, d)
            if i < 0 or dates[s][i][:7] != d[:7]:
                continue
            m = momentum(closes[s], i)
            if m is not None:
                scores.append((-m, s, m, i))
        scores.sort()
        target = {s: (m, i) for _, s, m, i in scores[:top]}
        rebal.append((d, [s for _, s, _, _ in scores[:top]]))
        drop = sorted(s for s in held if s not in target)
        add = sorted(s for s in target if s not in held)
        nxt = cal[k + 1] if k + 1 < len(cal) else None
        if nxt is None:
            for s in drop:
                pending.append((s, d, "exit", closes[s][last_idx(s, d)], "모멘텀 상위 10 밖"))
            for s in add:
                m, i = target[s]
                pending.append(
                    (s, d, "entry", closes[s][i], f"모멘텀 상위 10 (12-1개월 {m * 100:+.1f}%)")
                )
            break
        exit_days = []
        for s in drop:
            t = held.pop(s)
            j = first_after(s, d)
            if j is None:  # 자료가 끝난 종목: 마지막 종가로 다음 거래일에 정리
                j = len(dates[s]) - 1
                px, xd, why = closes[s][j], nxt, "모멘텀 상위 10 밖(자료 끝, 마지막 종가)"
            else:
                px, xd, why = series[s][j][1], dates[s][j], "모멘텀 상위 10 밖"
            t.update(
                exit_i=j, exit_date=xd, exit_px=px, reason_out=why, ret=_ret(t["avg"], px, fee)
            )
            trades.append(t)
            exit_days.append(xd)
        day = max(exit_days + [nxt])
        for s in add:
            j = first_after(s, day, inclusive=True)
            if j is None:
                continue
            m, _ = target[s]
            reason = f"모멘텀 상위 10 (12-1개월 {m * 100:+.1f}%)"
            o = series[s][j][1]
            held[s] = {
                "sid": "Q6",
                "sym": s,
                "side": "long",
                "entry_i": j,
                "entry_date": dates[s][j],
                "signal_date": d,
                "fills": [(dates[s][j], o, 1.0, reason)],
                "reason_in": reason,
                "avg": o,
            }
    return {
        "trades": trades,
        "open": [held[s] for s in sorted(held)],
        "pending": pending,
        "rebalances": rebal,
    }
