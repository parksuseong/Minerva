"""고서(古書) 전략 규칙 — 룰북 후보 전략의 기계 정의.

입력·출력만 있는 순수 함수 모음이다. 파일·네트워크·시계를 쓰지 않으며
같은 입력이면 항상 같은 결과를 낸다.

일봉 형식: (날짜 'YYYY-MM-DD', 시가, 고가, 저가, 종가) 튜플의 리스트, 날짜 오름차순.

지표 설정(룰북 I-01, I-40, I-45, 충돌 C-003~C-005 처리):
- Stochastic fast %K(5), CCI(20), RSI(14, Wilder)
- 과매도 20 / -100 / 30, 과매수 80 / +100 / 70
- 상향 돌파: 전봉 < 기준선 <= 현봉, 하향 이탈: 전봉 > 기준선 >= 현봉
- k단 신호: Stoch→CCI→RSI 순서로 최근 3봉 안에 1..k번째 지표가 모두 돌파
- 확정 봉만 사용. 신호는 봉 종가에 확정되고 체결은 다음 일봉 시가.

전략 키(STRATEGIES): 1A 1B 1C 2-80 2-50 3-c50 3-c0 4 5 6 7 8 9 10 11 14
"""

from __future__ import annotations

from datetime import date as _date

FEE = 0.0005  # 편도 수수료

OS = {"stoch": 20.0, "cci": -100.0, "rsi": 30.0}
OB = {"stoch": 80.0, "cci": 100.0, "rsi": 70.0}
ORDER = ("stoch", "cci", "rsi")

STRATEGIES = {
    "1A": {
        "name": "SCR 1단 진입 + 1단 과매수 이탈 청산",
        "rule_ids": ["L-06", "L-01", "E-21", "I-01"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
    },
    "1B": {
        "name": "SCR 2단 진입 + 2단 과매수 이탈 청산",
        "rule_ids": ["L-06", "L-01", "E-21", "I-01"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
    },
    "1C": {
        "name": "SCR 3단 진입 + 3단 과매수 이탈 청산",
        "rule_ids": ["L-06", "L-01", "E-21", "E-20"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
    },
    "2-80": {
        "name": "3단 돌파 + 스토캐 80 이상 제외(재과매도 후 2단)",
        "rule_ids": ["L-04", "L-01"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
        "stoch_skip": 80.0,
    },
    "2-50": {
        "name": "3단 돌파 + 스토캐 50 이상 제외(재과매도 후 2단)",
        "rule_ids": ["L-04", "L-01", "C-009"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
        "stoch_skip": 50.0,
    },
    "3-c50": {
        "name": "일봉 1단 눌림(CCI 50·RSI 50·50일선 위, 주·월 상방)",
        "rule_ids": ["N-01", "N-02", "N-05", "T-03", "C-004"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
        "cci_mid": 50.0,
    },
    "3-c0": {
        "name": "일봉 1단 눌림(CCI 0·RSI 50·50일선 위, 주·월 상방)",
        "rule_ids": ["N-01", "N-02", "N-05", "T-03", "I-08"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
        "cci_mid": 0.0,
    },
    "4": {
        "name": "상승 추세 스토캐 1단 단타",
        "rule_ids": ["L-36"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
    },
    "5": {
        "name": "CCI 2단 단타",
        "rule_ids": ["L-37"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
    },
    "6": {
        "name": "주봉 SCR 스윙",
        "rule_ids": ["P-18", "E-20"],
        "side": "long",
        "timeframe": "W",
        "max_hold": 260,
    },
    "7": {
        "name": "월봉 SCR 장투",
        "rule_ids": ["E-03"],
        "side": "long",
        "timeframe": "M",
        "max_hold": 1260,
    },
    "8": {
        "name": "3계단 물타기(1단 0.25 → 2단 0.25 → 3단 0.5)",
        "rule_ids": ["P-29", "P-13", "E-21"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 120,
    },
    "9": {
        "name": "1단 진입 + 3단 과매수 동시 이탈까지 보유",
        "rule_ids": ["E-20"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 120,
    },
    "10": {
        "name": "VIX 주봉 3단 이탈 → 폭락 종목 롱",
        "rule_ids": ["L-89", "L-124", "V-125"],
        "side": "long",
        "timeframe": "W",
        "max_hold": 260,
        "needs_vix": True,
    },
    "11": {
        "name": "VIX 회귀 + 2단 이상 과매도 돌파",
        "rule_ids": ["L-144", "L-140"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
        "needs_vix": True,
    },
    "14": {
        "name": "(숏, 참고) CCI 200 + 3단 과매수 이탈 숏",
        "rule_ids": ["S-01", "S-43", "S-14"],
        "side": "short",
        "timeframe": "D",
        "max_hold": 60,
    },
}


# ---------------------------------------------------------------- 지표


def sma(values, n):
    """단순 이동평균. 앞부분 n-1개는 None."""
    out = [None] * len(values)
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= n:
            total -= values[i - n]
        if i >= n - 1:
            out[i] = total / n
    return out


def stoch_fast_k(highs, lows, closes, n=5):
    """Stochastic fast %K(n). 고저가 같으면 50."""
    out = [None] * len(closes)
    for i in range(n - 1, len(closes)):
        hi = max(highs[i - n + 1 : i + 1])
        lo = min(lows[i - n + 1 : i + 1])
        out[i] = 50.0 if hi == lo else (closes[i] - lo) / (hi - lo) * 100.0
    return out


def cci(highs, lows, closes, n=20):
    """CCI(n) = (TP - SMA(TP)) / (0.015 × 평균 절대편차)."""
    tp = [(h + lo + c) / 3.0 for h, lo, c in zip(highs, lows, closes)]
    mean = sma(tp, n)
    out = [None] * len(tp)
    for i in range(n - 1, len(tp)):
        m = mean[i]
        dev = sum(abs(x - m) for x in tp[i - n + 1 : i + 1]) / n
        out[i] = 0.0 if dev == 0 else (tp[i] - m) / (0.015 * dev)
    return out


def rsi(closes, n=14):
    """RSI(n), Wilder 평활."""
    out = [None] * len(closes)
    if len(closes) <= n:
        return out
    g = loss = 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        g += max(d, 0.0)
        loss += max(-d, 0.0)
    ag, al = g / n, loss / n
    out[n] = _rsi_val(ag, al)
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (n - 1) + max(d, 0.0)) / n
        al = (al * (n - 1) + max(-d, 0.0)) / n
        out[i] = _rsi_val(ag, al)
    return out


def _rsi_val(ag, al):
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def cross_up(s, i, line):
    """i봉에서 상향 돌파(전봉 < 기준선 <= 현봉, I-01)."""
    if i < 1 or s[i] is None or s[i - 1] is None:
        return False
    return s[i - 1] < line <= s[i]


def cross_down(s, i, line):
    """i봉에서 하향 이탈(전봉 > 기준선 >= 현봉, I-01)."""
    if i < 1 or s[i] is None or s[i - 1] is None:
        return False
    return s[i - 1] > line >= s[i]


def level_up(ind, i, window=3):
    """i봉의 상향 단계(0~3).

    Stoch→CCI→RSI 순으로 최근 window봉 안 과매도 상향 돌파가 이어진 개수.
    """
    lo = max(1, i - window + 1)
    k = 0
    for name in ORDER:
        s = ind[name]
        if any(cross_up(s, j, OS[name]) for j in range(lo, i + 1)):
            k += 1
        else:
            break
    return k


def level_down(ind, i, window=3):
    """i봉의 하향 단계(0~3).

    Stoch→CCI→RSI 순으로 최근 window봉 안 과매수 하향 이탈이 이어진 개수.
    """
    lo = max(1, i - window + 1)
    k = 0
    for name in ORDER:
        s = ind[name]
        if any(cross_down(s, j, OB[name]) for j in range(lo, i + 1)):
            k += 1
        else:
            break
    return k


def new_up(ind, i, k, window=3):
    """i봉에 처음으로 k단 이상 상향 신호가 된 경우 True."""
    return level_up(ind, i, window) >= k and (i < 1 or level_up(ind, i - 1, window) < k)


def new_down(ind, i, k, window=3):
    """i봉에 처음으로 k단 이상 하향(과매수 이탈) 신호가 된 경우 True."""
    return level_down(ind, i, window) >= k and (i < 1 or level_down(ind, i - 1, window) < k)


def indicators(bars):
    """봉 리스트에서 stoch·cci·rsi·sma20/50/200·종가를 계산해 dict로 돌려준다."""
    h = [b[2] for b in bars]
    lw = [b[3] for b in bars]
    c = [b[4] for b in bars]
    return {
        "stoch": stoch_fast_k(h, lw, c),
        "cci": cci(h, lw, c),
        "rsi": rsi(c),
        "sma20": sma(c, 20),
        "sma50": sma(c, 50),
        "sma200": sma(c, 200),
        "close": c,
    }


# ---------------------------------------------------------------- 주봉·월봉


def _d(s):
    return _date(int(s[0:4]), int(s[5:7]), int(s[8:10]))


def resample(bars, frame):
    """일봉을 주봉('W', ISO 주) 또는 월봉('M')으로 묶는다.

    반환: (묶은 봉 리스트, 각 묶음의 마지막 일봉 인덱스 리스트, 마지막 묶음 마감 여부).
    마지막 묶음은 날짜로 마감을 알 수 있을 때만 마감으로 본다
    (주봉: 금요일, 월봉: 그 달 마지막 평일).
    """

    def key(s):
        d = _d(s)
        if frame == "W":
            iso = d.isocalendar()
            return (iso[0], iso[1])
        return (d.year, d.month)

    out, last_idx = [], []
    start = 0
    for i in range(1, len(bars) + 1):
        if i == len(bars) or key(bars[i][0]) != key(bars[start][0]):
            g = bars[start:i]
            out.append((g[-1][0], g[0][1], max(b[2] for b in g), min(b[3] for b in g), g[-1][4]))
            last_idx.append(i - 1)
            start = i
    closed_last = False
    if bars:
        d = _d(bars[-1][0])
        if frame == "W":
            closed_last = d.weekday() == 4
        else:
            nd = d.toordinal() + 1
            while _date.fromordinal(nd).weekday() >= 5:
                nd += 1
            closed_last = _date.fromordinal(nd).month != d.month
    return out, last_idx, closed_last


def _frame_ctx(bars, frame):
    """상위 시간축 지표와 '일봉 i 종가 시점에 마감된 최신 상위 봉 번호' 표."""
    hb, last_idx, closed_last = resample(bars, frame)
    ind = indicators(hb)
    n = len(bars)
    closes_at = {}  # 일봉 인덱스 -> 그 일봉에서 마감되는 상위 봉 번호
    for k, di in enumerate(last_idx):
        if k == len(last_idx) - 1 and not closed_last:
            continue
        closes_at[di] = k
    latest = [-1] * n
    cur = -1
    for i in range(n):
        if i in closes_at:
            cur = closes_at[i]
        latest[i] = cur
    return {"bars": hb, "ind": ind, "closes_at": closes_at, "latest": latest}


def _asof(dates_src, values, dates_dst):
    """dates_dst 각 날짜에 대해 dates_src 중 그 날짜 이하 마지막 인덱스(없으면 -1)."""
    out = []
    j = -1
    for d in dates_dst:
        while j + 1 < len(dates_src) and dates_src[j + 1] <= d:
            j += 1
        out.append(j)
    return out


def prepare(bars, vix=None):
    """전략 계산에 필요한 지표를 한 번에 만든다(여러 전략에 재사용)."""
    ctx = {"bars": bars, "ind": indicators(bars)}
    ctx["W"] = _frame_ctx(bars, "W")
    ctx["M"] = _frame_ctx(bars, "M")
    if vix:
        vd = [b[0] for b in vix]
        vctx = {"bars": vix, "ind": indicators(vix), "W": _frame_ctx(vix, "W")}
        vctx["asof"] = _asof(vd, None, [b[0] for b in bars])
        ctx["vix"] = vctx
    return ctx


# ---------------------------------------------------------------- 신호 판정


def _entry(sid, ctx, i, st):
    """i봉 종가 기준 진입 신호. (단계, 사유) 또는 None. st는 전략 상태(dict, #2 대기 플래그 등)."""
    ind = ctx["ind"]
    s = ind["stoch"]
    if sid in ("1A", "4", "8", "9"):
        if not cross_up(s, i, OS["stoch"]):
            return None
        if sid == "4":
            c, m50, m200 = ind["close"][i], ind["sma50"][i], ind["sma200"][i]
            if m50 is None or m200 is None or not (c > m50 > m200):
                return None
            return (1, "상승추세 스토캐 20 돌파")
        return (1, "1단: 스토캐 20 돌파")
    if sid == "1B":
        return (2, "2단 돌파") if new_up(ind, i, 2) else None
    if sid == "1C":
        return (3, "3단 돌파") if new_up(ind, i, 3) else None
    if sid in ("2-80", "2-50"):
        thr = STRATEGIES[sid]["stoch_skip"]
        if new_up(ind, i, 3):
            if s[i] is not None and s[i] >= thr:
                st["wait"] = True
                st["seen_os"] = False
                return None
            st["wait"] = False
            return (3, f"3단 돌파(스토캐 {s[i]:.0f})")
        if st.get("wait"):
            if s[i] is not None and s[i] < OS["stoch"]:
                st["seen_os"] = True
            elif st.get("seen_os") and new_up(ind, i, 2):
                st["wait"] = False
                return (2, "재과매도 후 2단 돌파")
        return None
    if sid in ("3-c50", "3-c0"):
        mid = STRATEGIES[sid]["cci_mid"]
        if i < 5 or not cross_up(s, i, OS["stoch"]):
            return None
        for j in range(i - 4, i + 1):
            cc, rr, m50 = ind["cci"][j], ind["rsi"][j], ind["sma50"][j]
            if cc is None or rr is None or m50 is None:
                return None
            if cc < mid or rr < 50 or ind["close"][j] <= m50:
                return None
        for fr in ("W", "M"):
            f = ctx[fr]
            k = f["latest"][i]
            if k < 0:
                return None
            cc, rr = f["ind"]["cci"][k], f["ind"]["rsi"][k]
            if cc is None or rr is None or cc < mid or rr < 50:
                return None
        return (1, f"1단 눌림(CCI≥{mid:g}, RSI≥50, 50일선 위, 주·월 상방)")
    if sid == "5":
        if cross_up(ind["cci"], i, OS["cci"]) and any(
            cross_up(s, j, OS["stoch"]) for j in range(max(1, i - 2), i + 1)
        ):
            return (2, "CCI -100 돌파(스토캐 동반)")
        return None
    if sid in ("6", "7"):
        f = ctx["W" if sid == "6" else "M"]
        k = f["closes_at"].get(i)
        if k is None:
            return None
        need = 2 if sid == "6" else 3
        if new_up(f["ind"], k, need):
            return (need, ("주봉 %d단 돌파" if sid == "6" else "월봉 %d단 돌파") % need)
        return None
    if sid == "10":
        v = ctx.get("vix")
        if not v or i < 1:
            return None
        vw = v["W"]
        # 직전 일봉 이후 마감된 VIX 주봉 중 3단 이탈(창 2주)이 있었나
        a, b = v["asof"][i - 1], v["asof"][i]
        hit = None
        for vi in range(a + 1, b + 1):
            k = vw["closes_at"].get(vi)
            if k is not None and new_down(vw["ind"], k, 3, window=2):
                hit = vw["bars"][k][0]
        if hit is None:
            return None
        lo = max(0, i - 251)
        hi = max(ind["close"][lo : i + 1])
        if ind["close"][i] <= hi * 0.80:
            return (
                3,
                "VIX 주봉 3단 이탈({}) + 고점 대비 {:.0f}%".format(
                    hit, (ind["close"][i] / hi - 1) * 100
                ),
            )
        return None
    if sid == "11":
        v = ctx.get("vix")
        if not v:
            return None
        vi = v["asof"][i]
        if vi < 20:
            return None
        vr = v["ind"]["rsi"]
        if vr[vi] is None or vr[vi] >= 70:
            return None
        if not any(cross_down(vr, j, OB["rsi"]) for j in range(vi - 19, vi + 1)):
            return None
        if not new_up(ind, i, 2):
            return None
        r = ind["rsi"]
        if r[i] is None or r[i - 1] is None or r[i] <= r[i - 1]:
            return None
        vc = v["ind"]["close"]
        win = range(vi - 19, vi + 1)
        peak = max(win, key=lambda j: (vc[j], j))
        base_lo = min(vc[max(0, peak - 20) : peak]) if peak > 0 else vc[peak]
        st["vix_target"] = base_lo * 1.10
        return (
            2,
            "VIX 회귀(RSI 70 이탈) + 2단 돌파, 회귀 목표 VIX {:.2f}".format(st["vix_target"]),
        )
    if sid == "14":
        cc = ind["cci"]
        if i < 5 or not cross_down(cc, i, OB["cci"]):
            return None
        if max((cc[j] or -1e9) for j in range(i - 4, i + 1)) < 200:
            return None
        win3 = range(i - 2, i + 1)
        if not any(cross_down(s, j, OB["stoch"]) for j in win3):
            return None
        if not any(cross_down(ind["rsi"], j, OB["rsi"]) for j in win3):
            return None
        disp = 0.0
        for j in range(i - 4, i + 1):
            m = ind["sma20"][j]
            if m:
                disp = max(disp, ind["close"][j] / m)
        if disp < 1.10:
            return None
        return (3, "CCI 200 후 3단 과매수 이탈, 이격 %.0f%%" % (disp * 100))
    raise KeyError(sid)


def _exit(sid, ctx, i, pos):
    """i봉 종가 기준 신호 청산 사유 또는 None. pos는 보유 정보(dict)."""
    ind = ctx["ind"]
    s = ind["stoch"]
    stage = pos["stage"]
    if sid in ("1A", "4", "3-c50", "3-c0"):
        return "스토캐 80 이탈" if cross_down(s, i, OB["stoch"]) else None
    if sid in ("1B", "1C", "2-80", "2-50"):
        k = 3 if stage >= 3 else 2
        return (f"{k}단 과매수 이탈") if new_down(ind, i, k) else None
    if sid == "5":
        return "CCI 100 이탈" if cross_down(ind["cci"], i, OB["cci"]) else None
    if sid == "9":
        return "3단 과매수 동시 이탈" if new_down(ind, i, 3) else None
    if sid in ("6", "7"):
        f = ctx["W" if sid == "6" else "M"]
        k = f["closes_at"].get(i)
        if k is not None and new_down(f["ind"], k, 3):
            return "주봉 3단 과매수 이탈" if sid == "6" else "월봉 3단 과매수 이탈"
        return None
    if sid == "10":
        f = ctx["W"]
        k = f["closes_at"].get(i)
        if k is not None and cross_down(f["ind"]["stoch"], k, OB["stoch"]):
            return "주봉 스토캐 80 이탈"
        return None
    if sid == "8":
        c = ind["close"][i]
        avg = pos["avg"]
        if c >= avg * 1.30:
            return "평단 +30%"
        if c > avg and cross_down(s, i, OB["stoch"]):
            return "수익 중 스토캐 80 이탈"
        return None
    if sid == "11":
        v = ctx["vix"]
        vi = v["asof"][i]
        if vi >= 0 and v["ind"]["close"][vi] <= pos["vix_target"]:
            return "VIX 회귀 완료"
        return None
    if sid == "14":
        return "스토캐 20 도달" if (s[i] is not None and s[i] <= OS["stoch"]) else None
    raise KeyError(sid)


def _add(sid, ctx, i, pos):
    """#8 추가 매수 판정. (비중, 단계, 사유) 또는 None."""
    if sid != "8":
        return None
    ind = ctx["ind"]
    c = ind["close"][i]
    if c >= pos["last_px"]:
        return None
    if "add2" not in pos["done"] and new_up(ind, i, 2):
        return (0.25, 2, "2차 물타기: 2단 돌파")
    if "add3" not in pos["done"] and new_up(ind, i, 3):
        return (0.50, 3, "3차 물타기: 3단 돌파")
    return None


# ---------------------------------------------------------------- 실행


def simulate(sid, bars, vix=None, stop_pct=None, fee=FEE, ctx=None):
    """전략을 일봉 위에서 돌려 거래 목록과 미체결 신호를 돌려준다.

    - 신호는 i봉 종가에 확정, 체결은 i+1봉 시가.
    - stop_pct(예: 0.10): 롱은 저가가 진입가(#8은 평단)×(1-stop) 이하면 그 가격(갭이면 시가)에 청산,
      숏은 고가가 진입가×(1+stop) 이상이면 청산. 진입 당일 포함.
    - 시간 청산: 진입 체결 후 max_hold 일봉이 지난 봉 종가에서 확정 → 다음 시가.
    반환: {"trades": [...], "pending": [...], "open": 보유 중 포지션 또는 None}
    각 거래: entry_i/exit_i(일봉 인덱스), entry_date, exit_date, side,
             fills[(날짜, 가격, 비중, 사유)], exit_px, reason_in, reason_out,
             ret(수수료 후 1x 수익률, 예산 1 기준), mae(최대 역행, 음수).
    """
    spec = STRATEGIES[sid]
    if ctx is None:
        ctx = prepare(bars, vix if spec.get("needs_vix") else None)
    side = spec["side"]
    max_hold = spec["max_hold"]
    n = len(bars)
    trades, pending = [], []
    st = {}
    pos = None
    order = None  # (종류, 비중, 단계, 사유, 신호 날짜)

    def _close(pos, i, px, reason):
        val = _value(pos, px, fee, side, closing=True)
        pos.update(exit_i=i, exit_date=bars[i][0], exit_px=px, reason_out=reason, ret=val - 1.0)
        trades.append(pos)

    for i in range(n):
        o, h, lo, c = bars[i][1], bars[i][2], bars[i][3], bars[i][4]
        # 1) 시가 체결
        if order is not None:
            kind, w, stage, reason, sd = order
            order = None
            if kind == "entry" and pos is None:
                pos = {
                    "sid": sid,
                    "side": side,
                    "entry_i": i,
                    "entry_date": bars[i][0],
                    "stage": stage,
                    "fills": [(bars[i][0], o, w, reason)],
                    "reason_in": reason,
                    "signal_date": sd,
                    "done": set(),
                    "last_px": o,
                    "mae": 0.0,
                }
                if "vix_target" in st:
                    pos["vix_target"] = st.pop("vix_target")
                _recalc(pos)
            elif kind == "add" and pos is not None:
                pos["fills"].append((bars[i][0], o, w, reason))
                pos["done"].add(f"add{stage}")
                pos["last_px"] = o
                pos["stage"] = max(pos["stage"], stage)
                _recalc(pos)
            elif kind == "exit" and pos is not None:
                _close(pos, i, o, reason)
                pos = None
        # 2) 장중 손절
        if pos is not None and stop_pct:
            ref = pos["avg"]
            if side == "long" and lo <= ref * (1 - stop_pct):
                px = (
                    min(o, ref * (1 - stop_pct))
                    if i > pos["entry_i"] or len(pos["fills"]) > 1
                    else ref * (1 - stop_pct)
                )
                _close(pos, i, px, f"손절 {round(stop_pct * 100)}%")
                pos = None
            elif side == "short" and h >= ref * (1 + stop_pct):
                px = max(o, ref * (1 + stop_pct)) if i > pos["entry_i"] else ref * (1 + stop_pct)
                _close(pos, i, px, f"손절 {round(stop_pct * 100)}%")
                pos = None
        # 3) 역행 기록
        if pos is not None:
            worst = lo if side == "long" else h
            pos["mae"] = min(pos["mae"], _value(pos, worst, fee, side, closing=False) - 1.0)
        # 4) 종가 판정
        if pos is not None:
            reason = _exit(sid, ctx, i, pos)
            if reason is None and i - pos["entry_i"] >= max_hold:
                reason = f"시간 청산 {max_hold}봉"
            if reason is not None:
                order = ("exit", 0, 0, reason, bars[i][0])
            else:
                a = _add(sid, ctx, i, pos)
                if a is not None:
                    order = ("add", a[0], a[1], a[2], bars[i][0])
        else:
            e = _entry(sid, ctx, i, st)
            if e is not None:
                w = 0.25 if sid == "8" else 1.0
                order = ("entry", w, e[0], e[1], bars[i][0])
            else:
                st.pop("vix_target", None)
        if order is not None and i == n - 1:
            kind, w, stage, reason, sd = order
            pending.append((sd, kind, c, reason))
    return {"trades": trades, "pending": pending, "open": pos}


def _recalc(pos):
    """보유 비중·평단을 다시 계산한다."""
    w = sum(f[2] for f in pos["fills"])
    pos["weight"] = w
    pos["avg"] = sum(f[1] * f[2] for f in pos["fills"]) / w


def _value(pos, px, fee, side, closing):
    """예산 1 기준 계좌 가치. closing이면 청산 수수료까지 뺀다."""
    if side == "long":
        cash = 1.0 - sum(f[2] for f in pos["fills"])
        shares = sum(f[2] / (f[1] * (1 + fee)) for f in pos["fills"])
        return cash + shares * px * ((1 - fee) if closing else 1.0)
    # 숏: 비중 w를 가격 p에 팔았다면 손익 = w × (p - px)/p, 수수료 양쪽
    v = 1.0
    for f in pos["fills"]:
        v += f[2] * ((f[1] - px) / f[1]) - f[2] * fee
        if closing:
            v -= f[2] * fee * px / f[1]
    return v


ACTION = {"entry": "BUY", "add": "ADD", "exit": "SELL"}


def signals(sid, bars, vix=None, ctx=None):
    """일봉 종가 기준 신호 목록 [(날짜, 행동, 신호 봉 종가, 사유)] — 손절 없는 기본 규칙.

    행동: BUY/ADD/SELL(롱), SHORT/COVER(#14).
    마지막 봉의 신호는 다음 시가 체결 예정분으로 함께 나온다.
    가상매매 탭은 매일 이 함수의 마지막 날짜 신호만 보면 된다.
    """
    spec = STRATEGIES[sid]
    res = simulate(sid, bars, vix=vix, stop_pct=None, ctx=ctx)
    close_by_date = {b[0]: b[4] for b in bars}
    short = spec["side"] == "short"
    out = []
    rows = list(res["trades"])
    if res["open"] is not None:
        rows.append(res["open"])
    for t in rows:
        for k, f in enumerate(t["fills"]):
            sd = t["signal_date"] if k == 0 else _prev_date(bars, f[0])
            act = ("SHORT" if short else "BUY") if k == 0 else "ADD"
            out.append((sd, act, close_by_date[sd], f[3]))
        if "exit_date" in t and not t["reason_out"].startswith("손절"):
            sd = _prev_date(bars, t["exit_date"])
            out.append((sd, "COVER" if short else "SELL", close_by_date[sd], t["reason_out"]))
    for sd, kind, c, reason in res["pending"]:
        act = ACTION[kind]
        if short:
            act = {"BUY": "SHORT", "SELL": "COVER"}.get(act, act)
        out.append((sd, act, c, reason + " (다음 시가 체결 예정)"))
    out.sort(key=lambda x: x[0])
    return out


def _prev_date(bars, d):
    """d 직전 일봉 날짜(체결일 → 신호일)."""
    lo, hi = 0, len(bars)
    while lo < hi:
        mid = (lo + hi) // 2
        if bars[mid][0] < d:
            lo = mid + 1
        else:
            hi = mid
    return bars[lo - 1][0]
