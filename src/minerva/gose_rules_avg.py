"""고서 "SCR 3단 + 물타기" 규칙(5차에서 고른 판). 순수 함수만: 파일·네트워크·시계 없음.

규칙서 본문은 `gose_spec.md`(다른 에이전트가 그대로 따라 할 수 있게 쓴 것)이고, 이 모듈은
그 문서를 한 줄 한 줄 옮긴 것이다. 둘이 다르면 버그다.

일봉 형식: (날짜 'YYYY-MM-DD', 시가, 고가, 저가, 종가) 리스트, 날짜 오름차순, 마감된 봉만,
분할 조정(배당 조정 없음). 신호는 t봉 종가로 보고 체결은 t+1봉 시가.

요약
- 지표: 패스트 스토캐스틱 %K(5), CCI(20), RSI(14, 단순 평균식).
- 신호 SIG_t: 세 지표가 모두 과매도선 위, 그중 하나 이상이 t봉에 위로 돌파, 그리고
  창 [max(t−10, 직전 SIG+1), t−1] 안에 스토캐 ≤20, CCI ≤−100, RSI ≤33 인 봉이 각각 있음.
- 무포지션에서 SIG → 다음 시가 1단위(사이클 시작 때 자산/3) 매수. 보유 중 SIG → 추가 1단위
  (최대 2번). 익절: 평단(수수료 포함)×1.10 지정가, 전량. 손절·시간 제한 없음.
"""

from __future__ import annotations

from datetime import date

FEE = 0.0005  # 편도
STOCH_N, CCI_N, RSI_N = 5, 20, 14
OS = (20.0, -100.0, 30.0)  # 돌파선(스토캐, CCI, RSI)
WIN_LINE = (20.0, -100.0, 33.0)  # 창 안 과매도 확인선(RSI는 3 여유)
WINDOW = 10
TP = 0.10
MAX_ADDS = 2
UNITS = 1 + MAX_ADDS  # 사이클 시작 자산을 3등분


# ---------------------------------------------------------------- 지표


def stoch_k(h: list, lo: list, c: list, n: int = STOCH_N) -> list:
    """(C_t − 최저 L(n)) / (최고 H(n) − 최저 L(n)) × 100. 분모 0이면 50. 앞 n−1봉은 None."""
    out: list = [None] * len(c)
    for i in range(n - 1, len(c)):
        hi = max(h[i - n + 1 : i + 1])
        lw = min(lo[i - n + 1 : i + 1])
        out[i] = 50.0 if hi == lw else (c[i] - lw) / (hi - lw) * 100
    return out


def cci(h: list, lo: list, c: list, n: int = CCI_N) -> list:
    """(TP − SMA_n(TP)) / (0.015 × 평균절대편차_n). TP=(H+L+C)/3. 편차 0이면 0."""
    tp = [(a + b + d) / 3 for a, b, d in zip(h, lo, c)]
    out: list = [None] * len(tp)
    for i in range(n - 1, len(tp)):
        win = tp[i - n + 1 : i + 1]
        m = sum(win) / n
        dev = sum(abs(x - m) for x in win) / n
        out[i] = 0.0 if dev == 0 else (tp[i] - m) / (0.015 * dev)
    return out


def rsi_sma(c: list, n: int = RSI_N) -> list:
    """100 − 100/(1 + ΣU/ΣD), 최근 n개 종가 변화의 단순 합(와일더 평활 아님).

    ΣD=0이면 ΣU>0일 때 100, 둘 다 0이면 50. 앞 n봉은 None.
    """
    out: list = [None] * len(c)
    for i in range(n, len(c)):
        up = sum(max(c[j] - c[j - 1], 0.0) for j in range(i - n + 1, i + 1))
        dn = sum(max(c[j - 1] - c[j], 0.0) for j in range(i - n + 1, i + 1))
        out[i] = (100.0 if up > 0 else 50.0) if dn == 0 else 100 - 100 / (1 + up / dn)
    return out


def indicators(bars: list) -> dict:
    h = [b[2] for b in bars]
    lo = [b[3] for b in bars]
    c = [b[4] for b in bars]
    return {"S": stoch_k(h, lo, c), "C": cci(h, lo, c), "R": rsi_sma(c)}


# ---------------------------------------------------------------- 신호


def signals(ind: dict) -> list[int]:
    """SIG 봉 인덱스 목록(자료 첫 봉부터 계산, 직전 SIG는 체결 여부와 상관없이 직전 SIG)."""
    keys = ("S", "C", "R")
    n = len(ind["S"])
    out: list[int] = []
    last = -1
    for t in range(1, n):
        v = [ind[k][t] for k in keys]
        p = [ind[k][t - 1] for k in keys]
        if None in v or None in p:
            continue
        if not all(v[j] > OS[j] for j in range(3)):
            continue
        if not any(p[j] <= OS[j] < v[j] for j in range(3)):
            continue
        lo = max(t - WINDOW, last + 1, 0)
        ok = all(
            any(ind[k][i] is not None and ind[k][i] <= WIN_LINE[j] for i in range(lo, t))
            for j, k in enumerate(keys)
        )
        if ok:
            out.append(t)
            last = t
    return out


# ---------------------------------------------------------------- 계좌 흉내


def _days(d0: str, d1: str) -> int:
    return (date.fromisoformat(d1) - date.fromisoformat(d0)).days


def simulate(
    bars: list,
    start: str | None = None,
    end: str | None = None,
    capital: float = 1.0,
    tp: float = TP,
    max_adds: int = MAX_ADDS,
    fee: float = FEE,
    ind: dict | None = None,
    sigs: list[int] | None = None,
) -> dict:
    """한 종목 한 계좌. start 날짜(포함) 이후 SIG만 쓰고, 그 전에는 무포지션·현금 capital.

    봉 t의 순서: ① SIG_{t−1}이면 시가에 진입(무포지션) 또는 추가(보유, 추가 < max_adds)
    ② 보유 중이면 평단(총투입/수량, 수수료 포함)×(1+tp) 지정가: 시가 ≥ 목표면 시가 체결
    (단, 이 봉에 사이클을 시작했으면 시가 체결 없음), 아니면 고가 ≥ 목표면 목표가 체결
    ③ 종가 평가. 청산한 봉에서는 다시 사지 않는다(진입은 시가에만 있으므로 자동).

    돌려주는 것: curve [[날짜, 자산]], cycles(닫힌 사이클), open(열린 사이클 상태 또는 None),
    cash, equity, next(다음 시가에 할 일: 마지막 봉이 SIG일 때).
    """
    ind = ind if ind is not None else indicators(bars)
    sigs = sigs if sigs is not None else signals(ind)
    n = len(bars)
    a = next((i for i, b in enumerate(bars) if start is None or b[0] >= start), n)
    z = max((i for i, b in enumerate(bars) if end is None or b[0] <= end), default=-1)
    sigset = {s for s in sigs if a <= s <= z}
    cash = capital
    sh = inv = unit = 0.0
    cyc: dict | None = None
    cycles: list[dict] = []
    curve: list[list] = []
    for t in range(a, z + 1):
        d, o, h, lo, c = bars[t][:5]
        if (t - 1) in sigset:
            if sh == 0:
                unit = cash / (1 + max_adds)
                sh = unit / (o * (1 + fee))
                cash -= unit
                inv = unit
                cyc = {
                    "entry_date": d,
                    "t0": t,
                    "eq0": cash + unit,
                    "unit": unit,
                    "adds": 0,
                    "fills": [[bars[t - 1][0], d, o, unit]],
                    "mae": 0.0,
                }
            elif cyc is not None and cyc["adds"] < max_adds:
                sh += unit / (o * (1 + fee))
                cash -= unit
                inv += unit
                cyc["adds"] += 1
                cyc["fills"].append([bars[t - 1][0], d, o, unit])
        if sh > 0 and cyc is not None:
            avg = inv / sh
            tgt = avg * (1 + tp)
            px = None
            if o >= tgt and cyc["entry_date"] != d:
                px = o
            elif h >= tgt:
                px = tgt
            cyc["mae"] = min(cyc["mae"], lo / avg - 1)
            if px is not None:
                got = sh * px * (1 - fee)
                cash += got
                cycles.append(
                    dict(
                        cyc,
                        exit_date=d,
                        exit_px=px,
                        avg=avg,
                        invested=inv,
                        proceeds=got,
                        pnl=got - inv,
                        ret=got / inv - 1,
                        bars_held=t - cyc["t0"] + 1,
                        days_held=_days(cyc["entry_date"], d),
                    )
                )
                sh = inv = 0.0
                cyc = None
        curve.append([d, cash + sh * c])
    open_state = None
    if cyc is not None and z >= a:
        avg = inv / sh
        last = bars[z]
        open_state = dict(
            cyc,
            shares=sh,
            invested=inv,
            avg=avg,
            tp_px=avg * (1 + tp),
            last_date=last[0],
            last_px=last[4],
            value=sh * last[4],
            ret=sh * last[4] / inv - 1,
            mae_now=last[4] / avg - 1,
            bars_held=z - cyc["t0"] + 1,
            days_held=_days(cyc["entry_date"], last[0]),
            adds_left=max_adds - cyc["adds"],
        )
    nxt = None
    if z >= a and z in sigset:
        if cyc is None:
            nxt = {
                "kind": "entry",
                "signal_date": bars[z][0],
                "close": bars[z][4],
                "amount": cash / (1 + max_adds),
            }
        elif cyc["adds"] < max_adds:
            nxt = {"kind": "add", "signal_date": bars[z][0], "close": bars[z][4], "amount": unit}
        else:
            nxt = {"kind": "capped", "signal_date": bars[z][0], "close": bars[z][4], "amount": 0.0}
    equity = curve[-1][1] if curve else capital
    return {
        "curve": curve,
        "cycles": cycles,
        "open": open_state,
        "capital": capital,
        "cash": cash,
        "equity": equity,
        "next": nxt,
        "first": a,
        "last": z,
    }


def metrics(res: dict) -> dict:
    """연복리(CAGR, 달력 기준), 일별 최대 낙폭, 닫힌 사이클 수·승률·최악 MAE·최장 보유."""
    curve = res["curve"]
    if not curve:
        return {}
    yrs = _days(curve[0][0], curve[-1][0]) / 365.25
    e0 = res["capital"]
    e1 = curve[-1][1]
    peak = mdd = 0.0
    for _, v in curve:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    cy = res["cycles"]
    out = {
        "cagr": (e1 / e0) ** (1 / yrs) - 1 if yrs > 0 and e1 > 0 else None,
        "mdd": mdd,
        "final": e1,
        "n": len(cy),
        "open": 1 if res["open"] else 0,
    }
    if cy:
        out.update(
            win=sum(1 for x in cy if x["ret"] > 0) / len(cy),
            worst_mae=min(x["mae"] for x in cy),
            bars_max=max(x["bars_held"] for x in cy),
            days_max=max(x["days_held"] for x in cy),
        )
    return out
