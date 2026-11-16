"""고서 전략 2차 규칙 — 그림 판독으로 뜻을 정한 정의(결과 보기 전에 사전 등록).

순수 함수만 있다. 파일·네트워크·시계를 쓰지 않는다. 지표·주봉 묶기·체결 가치 계산은
1차 모듈(`gose_rules`)을 그대로 쓴다.

일봉 형식: (날짜 'YYYY-MM-DD', 시가, 고가, 저가, 종가) 리스트, 날짜 오름차순.
vol: 변동성 상품(UVXY, UVIX 대리) 일봉, 같은 형식. 10v2·11v2a·11v2b에만 쓴다(신호원, 매매 안 함).

전략 키(STRATEGIES): 1v2 2v2 3v2-strict 3v2-loose 10v2 11v2a 11v2b
"""

from __future__ import annotations

from minerva import gose_rules as R

FEE = R.FEE
OS, OB = R.OS, R.OB

STRATEGIES = {
    "1v2": {
        "name": "세 지표 같은 봉 동시 돌파 + 동시 꺾임 청산",
        "rule_ids": ["L-01", "E-19", "E-20"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
    },
    "2v2": {
        "name": "1v2 + 돌파 봉 스토캐 ≥80 건너뜀(≤25 후 상방 턴 재진입)",
        "rule_ids": ["L-04", "C-009", "E-19"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
    },
    "3v2-strict": {
        "name": "일봉 1단 눌림 엄격판(스토캐 20 돌파)",
        "rule_ids": ["N-01", "N-04", "E-19"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
    },
    "3v2-loose": {
        "name": "일봉 1단 눌림 느슨판(저점 ≤35 후 상방 턴)",
        "rule_ids": ["N-01", "N-04", "L-15", "E-19"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
    },
    "10v2": {
        "name": "UVXY 주봉 3지표 이탈(RSI 고점 ≥65) → 지수 ETF 롱",
        "rule_ids": ["L-109", "L-124"],
        "side": "long",
        "timeframe": "W",
        "max_hold": 260,
        "needs_vol": True,
    },
    "11v2a": {
        "name": "UVXY 일봉 마지막 지표 과매수 이탈 → 롱",
        "rule_ids": ["V-23", "VP-150", "E-19"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
        "needs_vol": True,
    },
    "11v2b": {
        "name": "UVXY 일봉 같은 봉 동시 꺾임(스토캐80·CCI100 고점) → 롱",
        "rule_ids": ["L-113", "L-107", "E-19"],
        "side": "long",
        "timeframe": "D",
        "max_hold": 60,
        "needs_vol": True,
    },
}

# 변동성 신호 전략이 사는 종목(이 밖의 종목에는 신호를 쓰지 않는다)
TICKERS = {
    "10v2": ("SOXX", "QQQ", "SPY"),
    "11v2a": ("SOXX", "QQQ", "SPY", "SOXL"),
    "11v2b": ("SOXX", "QQQ", "SPY", "SOXL"),
}

CCI_MID = 0.0
LOOSE_LOW = 35.0
REENTRY_LOW = 25.0


# ---------------------------------------------------------------- 기본 판정


def _ok(*vals):
    return all(v is not None for v in vals)


def all_cross_up(ind, i):
    """같은 봉 i에서 Stoch 20·CCI −100·RSI 30을 모두 상향 돌파(L-01 최고 타점)."""
    return all(R.cross_up(ind[k], i, OS[k]) for k in R.ORDER)


def all_falling(ind, i):
    """Stoch·CCI·RSI가 모두 전봉 대비 하락."""
    if i < 1:
        return False
    for k in R.ORDER:
        s = ind[k]
        if not _ok(s[i], s[i - 1]) or not s[i] < s[i - 1]:
            return False
    return True


def all_rising(ind, i):
    """Stoch·CCI·RSI가 모두 전봉 대비 상승."""
    if i < 1:
        return False
    for k in R.ORDER:
        s = ind[k]
        if not _ok(s[i], s[i - 1]) or not s[i] > s[i - 1]:
            return False
    return True


def e19_exit(ind, i):
    """E-19 그림 정의: 세 지표 같은 봉 하락 + 스토캐가 t−3..t−1 중 80 이상."""
    if i < 4 or not all_falling(ind, i):
        return False
    s = ind["stoch"]
    return max((v for v in s[i - 3 : i] if v is not None), default=-1) >= OB["stoch"]


def _max(s, a, b):
    vals = [v for v in s[max(0, a) : b + 1] if v is not None]
    return max(vals) if vals else None


# ---------------------------------------------------------------- 변동성 신호


def vol_triggers_weekly(vol):
    """10v2: UVXY 주봉 신호가 마감된 일봉 인덱스(vol 기준) 집합과 주봉 날짜 목록."""
    f = R._frame_ctx(vol, "W")
    ind = f["ind"]
    s, c, r = ind["stoch"], ind["cci"], ind["rsi"]
    hits, last = [], -99
    for di, k in sorted(f["closes_at"].items()):
        if k < 9 or not _ok(s[k], c[k], r[k]):
            continue
        ms, mc, mr = _max(s, k - 8, k - 1), _max(c, k - 8, k - 1), _max(r, k - 8, k - 1)
        if ms is None or mc is None or mr is None:
            continue
        if not (ms >= 80 and mc >= 100 and mr >= 65):
            continue
        if not (s[k] < 80 and c[k] < 100 and r[k] < 70 and all_falling(ind, k)):
            continue
        if k - last <= 8:
            continue
        last = k
        hits.append((di, f["bars"][k][0]))
    return hits


def vol_triggers_a(vol):
    """11v2a: 마지막 지표가 과매수 기준선 아래로 내려온 봉(셋 다 t−20..t−1에 과매수 기록)."""
    ind = R.indicators(vol)
    s, c, r = ind["stoch"], ind["cci"], ind["rsi"]
    hits = []
    for t in range(21, len(vol)):
        if not _ok(s[t], c[t], r[t], s[t - 1], c[t - 1], r[t - 1]):
            continue
        if not (s[t] < 80 and c[t] < 100 and r[t] < 70):
            continue
        if not (s[t - 1] >= 80 or c[t - 1] >= 100 or r[t - 1] >= 70):
            continue
        ms, mc, mr = _max(s, t - 20, t - 1), _max(c, t - 20, t - 1), _max(r, t - 20, t - 1)
        if ms is None or mc is None or mr is None:
            continue
        if ms >= 80 and mc >= 100 and mr >= 70:
            hits.append((t, vol[t][0]))
    return hits


def vol_triggers_b(vol):
    """11v2b: 세 지표 같은 봉 하락 + t−3..t−1 스토캐 최고 ≥80·CCI 최고 ≥100, 5봉 안 중복 없음."""
    ind = R.indicators(vol)
    s, c = ind["stoch"], ind["cci"]
    hits, last = [], -99
    for t in range(4, len(vol)):
        if not all_falling(ind, t):
            continue
        ms, mc = _max(s, t - 3, t - 1), _max(c, t - 3, t - 1)
        if ms is None or mc is None or not (ms >= 80 and mc >= 100):
            continue
        if t - last <= 5:
            continue
        last = t
        hits.append((t, vol[t][0]))
    return hits


# ---------------------------------------------------------------- 준비


def compute_vol_hits(vol, keys=("10v2", "11v2a", "11v2b")):
    """{전략 키: [(vol 인덱스, 날짜)]}. 여러 종목에 같은 값을 쓰도록 한 번만 계산한다."""
    fn = {"10v2": vol_triggers_weekly, "11v2a": vol_triggers_a, "11v2b": vol_triggers_b}
    return {k: fn[k](vol) for k in keys}


def prepare(bars, vol=None, vol_hits=None, base=None):
    """지표·상위 시간축·변동성 신호 매핑을 만든다.

    vol_hits: 미리 계산한 {키: [(vol 인덱스, 날짜)]}. base: 같은 봉으로 만든 1차 `R.prepare`
    결과(있으면 지표·주봉·월봉을 다시 계산하지 않고 빌려 쓴다).
    """
    if base is not None and base["bars"] is bars:
        ctx = {"bars": bars, "ind": base["ind"], "W": base["W"], "M": base["M"]}
    else:
        ctx = {
            "bars": bars,
            "ind": R.indicators(bars),
            "W": R._frame_ctx(bars, "W"),
            "M": R._frame_ctx(bars, "M"),
        }
    if vol:
        if vol_hits is None:
            vol_hits = compute_vol_hits(vol)
        asof = R._asof([b[0] for b in vol], None, [b[0] for b in bars])
        ctx["vol"] = {
            "asof": asof,
            "hits": {k: {vi: d for vi, d in v} for k, v in vol_hits.items()},
        }
    return ctx


def _vol_hit(ctx, key, i):
    """일봉 i 종가 시점에 새로 마감된 vol 봉 중 key 신호가 있으면 그 날짜."""
    v = ctx.get("vol")
    if not v or i < 1:
        return None
    a, b = v["asof"][i - 1], v["asof"][i]
    hs = v["hits"].get(key)
    if not hs:
        return None
    found = None
    for vi in range(a + 1, b + 1):
        if vi in hs:
            found = hs[vi]
    return found


# ---------------------------------------------------------------- 진입·청산


def _trend_ok(ctx, i):
    ind = ctx["ind"]
    if i < 5:
        return False
    for j in range(i - 4, i + 1):
        cc, rr, m50 = ind["cci"][j], ind["rsi"][j], ind["sma50"][j]
        if not _ok(cc, rr, m50) or cc < CCI_MID or rr < 50 or ind["close"][j] <= m50:
            return False
    for fr in ("W", "M"):
        f = ctx[fr]
        k = f["latest"][i]
        if k < 1 or not all_rising(f["ind"], k):
            return False
    return True


def _entry(sid, ctx, i, st):
    ind = ctx["ind"]
    s = ind["stoch"]
    if sid in ("1v2", "2v2"):
        if all_cross_up(ind, i):
            if sid == "2v2" and s[i] >= OB["stoch"]:
                st["wait"], st["seen_low"] = True, False
                return None
            st["wait"] = False
            return (3, f"세 지표 같은 봉 돌파(스토캐 {s[i]:.0f})")
        if sid == "2v2" and st.get("wait") and _ok(s[i], s[i - 1]):
            if st.get("seen_low") and s[i] > s[i - 1]:
                st["wait"] = False
                return (1, f"과매수 돌파 건너뜀 후 스토캐 재턴({s[i]:.0f})")
            if s[i] <= REENTRY_LOW:
                st["seen_low"] = True
        return None
    if sid in ("3v2-strict", "3v2-loose"):
        if i < 5:
            return None
        if sid == "3v2-strict":
            if not R.cross_up(s, i, OS["stoch"]):
                return None
        else:
            if not _ok(*s[i - 4 : i + 1]):
                return None
            if not (min(s[i - 4 : i]) <= LOOSE_LOW and s[i] > s[i - 1] and s[i - 1] <= s[i - 2]):
                return None
        if not _trend_ok(ctx, i):
            return None
        return (1, "1단 눌림(%s)" % ("20 돌파" if sid == "3v2-strict" else "저점 후 턴"))
    if sid in ("10v2", "11v2a", "11v2b"):
        d = _vol_hit(ctx, sid, i)
        if d is None:
            return None
        what = {"10v2": "주봉 이탈", "11v2a": "마지막 지표 이탈", "11v2b": "동시 꺾임"}[sid]
        return (0, f"UVXY {what} 신호({d})")
    raise KeyError(sid)


def _exit(sid, ctx, i, pos):
    ind = ctx["ind"]
    if sid == "10v2":
        f = ctx["W"]
        k = f["closes_at"].get(i)
        if k is not None and R.cross_down(f["ind"]["stoch"], k, OB["stoch"]):
            return "주봉 스토캐 80 이탈"
        return None
    if sid in STRATEGIES:
        return "세 지표 동시 꺾임" if e19_exit(ind, i) else None
    raise KeyError(sid)


# ---------------------------------------------------------------- 실행


def simulate(sid, bars, vol=None, stop_pct=None, fee=FEE, ctx=None):
    """1차 simulate와 같은 체결 규칙(종가 신호 → 다음 시가, 장중 손절, 시간 청산).

    반환: {"trades": [...], "pending": [...], "open": 보유 중 포지션 또는 None}.
    거래 필드는 1차와 같다.
    """
    spec = STRATEGIES[sid]
    if ctx is None:
        ctx = prepare(bars, vol if spec.get("needs_vol") else None)
    side, max_hold = spec["side"], spec["max_hold"]
    n = len(bars)
    trades, pending = [], []
    st = {}
    pos = None
    order = None

    def _close(p, i, px, reason):
        p.update(
            exit_i=i,
            exit_date=bars[i][0],
            exit_px=px,
            reason_out=reason,
            ret=R._value(p, px, fee, side, closing=True) - 1.0,
        )
        trades.append(p)

    for i in range(n):
        o, lo, c = bars[i][1], bars[i][3], bars[i][4]
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
                R._recalc(pos)
            elif kind == "exit" and pos is not None:
                _close(pos, i, o, reason)
                pos = None
        if pos is not None and stop_pct:
            ref = pos["avg"]
            if lo <= ref * (1 - stop_pct):
                px = min(o, ref * (1 - stop_pct)) if i > pos["entry_i"] else ref * (1 - stop_pct)
                _close(pos, i, px, f"손절 {round(stop_pct * 100)}%")
                pos = None
        if pos is not None:
            pos["mae"] = min(pos["mae"], R._value(pos, lo, fee, side, closing=False) - 1.0)
        if pos is not None:
            reason = _exit(sid, ctx, i, pos)
            if reason is None and i - pos["entry_i"] >= max_hold:
                reason = f"시간 청산 {max_hold}봉"
            if reason is not None:
                order = ("exit", 0, 0, reason, bars[i][0])
        else:
            e = _entry(sid, ctx, i, st)
            if e is not None:
                order = ("entry", 1.0, e[0], e[1], bars[i][0])
        if order is not None and i == n - 1:
            pending.append((order[4], order[0], c, order[3]))
    return {"trades": trades, "pending": pending, "open": pos}


def signals(sid, bars, vol=None, ctx=None):
    """[(신호 날짜, BUY/SELL, 신호 봉 종가, 사유)] — 손절 없는 기본 규칙.

    마지막 봉 신호는 체결 예정으로 붙는다.
    """
    res = simulate(sid, bars, vol=vol, stop_pct=None, ctx=ctx)
    cb = {b[0]: b[4] for b in bars}
    out = []
    rows = list(res["trades"]) + ([res["open"]] if res["open"] is not None else [])
    for t in rows:
        out.append((t["signal_date"], "BUY", cb[t["signal_date"]], t["reason_in"]))
        if "exit_date" in t:
            sd = R._prev_date(bars, t["exit_date"])
            out.append((sd, "SELL", cb[sd], t["reason_out"]))
    for sd, kind, c, reason in res["pending"]:
        out.append(
            (sd, {"entry": "BUY", "exit": "SELL"}[kind], c, reason + " (다음 시가 체결 예정)")
        )
    out.sort(key=lambda x: x[0])
    return out
