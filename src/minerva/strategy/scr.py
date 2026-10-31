"""블로그 SCR 규칙으로 다시 짠 전략 (`docs/strategy/2026-10-31-scr-spec.md`).

상위 시간축은 확정된 봉만 쓴다. 주봉은 그 주 마지막 거래일 종가부터 알 수 있다.
실행: `python -m minerva.strategy.scr`
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

from minerva import trade
from minerva.strategy import daily as dm
from minerva.strategy import portfolio as pf
from minerva.strategy.indicators import (
    Bar,
    TMState,
    cci,
    crossed_down,
    crossed_up,
    dmi,
    resample,
    rsi,
    sma,
    stoch_fast_k,
    tm_states,
)
from minerva.strategy.report import fetch_yahoo

STOP = 0.10
TRANCHES = (0.25, 0.25, 0.5)
MAX_POSITIONS = 10
GATE_OPEN = ("R0", "R1")
SPLIT = date(2021, 10, 29)


def confirmed(states: Sequence[TMState], day: date) -> TMState | None:
    """그날까지 마감이 확정된 마지막 상위 봉."""
    i = bisect.bisect_right([s.day for s in states], day) - 1
    return states[i] if i >= 0 else None


def is_up(s: TMState | None) -> bool:
    return s is not None and (s.cci or 0) > 0 and (s.rsi or 0) > 50


def rising(cur: TMState | None, prev: TMState | None) -> bool:
    if cur is None or prev is None:
        return False
    pairs = ((cur.stoch, prev.stoch), (cur.cci, prev.cci), (cur.rsi, prev.rsi))
    return all(a is not None and b is not None and a > b for a, b in pairs)


def frame_up(states: Sequence[TMState], day: date) -> bool:
    """원문 \"우상향하거나 SCR 지표가 양수\" (224241271674)."""
    i = bisect.bisect_right([s.day for s in states], day) - 1
    if i < 0:
        return False
    return is_up(states[i]) or (i > 0 and rising(states[i], states[i - 1]))


def is_down(s: TMState | None) -> bool:
    return s is not None and (s.cci or 0) < 0 and (s.rsi or 100) < 50


def entry_signal(
    level: int,
    stoch: float,
    close: float,
    ma5: float,
    ma10: float,
    ma20: float,
    ma50: float,
    cci_v: float,
    rsi_v: float,
    stoch_cross: bool,
    htf_down: bool,
    weekly_oversold: bool,
    pdi: float | None,
    mdi: float | None,
    adx: float | None,
) -> bool:
    if weekly_oversold:
        return False
    if adx is not None and pdi is not None and mdi is not None and adx >= 20 and mdi >= pdi * 3:
        return False
    if level >= 3 and stoch >= 50:
        return False
    above = close > ma50
    if level >= 2 and (above or ma5 < ma10 < ma20):
        return True
    return stoch_cross and above and cci_v > 0 and rsi_v > 50 and not htf_down


@dataclass
class Plan:
    code: str
    name: str
    weight: float
    bars: dict[date, Bar]
    entry: dict[date, bool]
    add2: dict[date, bool]
    add3: dict[date, bool]
    strong: dict[date, bool]
    htf_exit: dict[date, bool]
    take: dict[date, bool]
    below20: dict[date, bool]
    oversold: dict[date, bool] = field(default_factory=dict)  # 스토캐 20 이하
    overbought: dict[date, bool] = field(default_factory=dict)  # 스토캐 80 이상
    above50: dict[date, bool] = field(default_factory=dict)  # 종가가 50일선 위
    below50: dict[date, bool] = field(default_factory=dict)  # 종가가 50일선 아래
    score: dict[date, int] = field(default_factory=dict)  # 월·주·일 상방 프레임 수
    topdown: dict[date, bool] = field(default_factory=dict)  # 월봉 돌파 + 주봉 상방


def plan_for(code: str, name: str, weight: float, bars: list[Bar]) -> Plan | None:
    if len(bars) < 300:
        return None
    closes = [b.close for b in bars]
    ma = {n: sma(closes, n) for n in (5, 10, 20, 50)}
    st, cc, rs = stoch_fast_k(bars), cci(bars, 20), rsi(closes)
    daily = tm_states(bars)
    pdi, mdi, adx = dmi(bars)
    week = tm_states(resample(bars, "W"))
    month = tm_states(resample(bars, "M"))
    entry, add2, add3, strong, htf_exit, take, below = {}, {}, {}, {}, {}, {}, {}
    oversold, overbought, above50, below50, score, topdown = {}, {}, {}, {}, {}, {}
    for i, b in enumerate(bars):
        d = b.day
        w, m = confirmed(week, d), confirmed(month, d)
        take[d] = i > 0 and crossed_down(st, i, 80.0)
        strong[d] = is_up(w) and is_up(m)
        htf_exit[d] = (
            is_down(w)
            or is_down(m)
            or (w is not None and w.short_level == 3)
            or (m is not None and m.short_level == 3)
        )
        add2[d] = i > 0 and crossed_up(cc, i, -100.0)
        add3[d] = i > 0 and crossed_up(rs, i, 30.0)
        below[d] = ma[20][i] is not None and b.close < ma[20][i]
        week_up, month_up = frame_up(week, d), frame_up(month, d)
        day_up = is_up(daily[i]) or (i > 0 and rising(daily[i], daily[i - 1]))
        score[d] = week_up + month_up + day_up
        topdown[d] = week_up and m is not None and m.long_level >= 1
        oversold[d] = st[i] is not None and st[i] <= 20
        overbought[d] = st[i] is not None and st[i] >= 80
        above50[d] = ma[50][i] is not None and b.close > ma[50][i]
        below50[d] = ma[50][i] is not None and b.close < ma[50][i]
        vals = (st[i], cc[i], rs[i], ma[5][i], ma[10][i], ma[20][i], ma[50][i])
        entry[d] = all(v is not None for v in vals) and entry_signal(
            daily[i].long_level,
            st[i],
            b.close,
            ma[5][i],
            ma[10][i],
            ma[20][i],
            ma[50][i],
            cc[i],
            rs[i],
            i > 0 and crossed_up(st, i, 20.0),
            is_down(w) or is_down(m),
            w is not None and w.stoch is not None and w.stoch <= 20,
            pdi[i],
            mdi[i],
            adx[i],
        )
    return Plan(
        code,
        name,
        weight,
        {b.day: b for b in bars},
        entry,
        add2,
        add3,
        strong,
        htf_exit,
        take,
        below,
        oversold,
        overbought,
        above50,
        below50,
        score,
        topdown,
    )


@dataclass
class Pos:
    qty: int
    cost: float  # 평단
    tranches: int
    failed: bool = False  # 1단 실패(20일선 이탈)를 겪었는가
    was_strong: bool = False  # 강세로 홀딩하던 종목만 주·월봉 이탈에 판다
    opened: date | None = None
    spent: float = 0.0  # 비용 포함 총 매수액
    peaked: bool = False  # 보유 중 스토캐 80 이상을 찍었는가
    armed: bool = False  # 보유 중 종가가 50일선 위에 있었는가


def size_factor(plan: Plan, day: date, tilt: str | None) -> float:
    """일치 점수 변형 (`docs/backtests/2026-10-31-scr-frame-score-hypotheses.md`). 0이면 안 산다."""
    if tilt == "prop":
        return plan.score.get(day, 0) / 3
    if tilt == "all3":
        return float(plan.score.get(day, 0) == 3)
    if tilt == "topdown":
        return float(bool(plan.topdown.get(day)))
    return 1.0


def failed_breakout(p: Pos, plan: Plan, day: date, fail: str | None) -> bool:
    """돌파 실패 매도 변형 (`docs/backtests/2026-10-31-scr-fail-exit-hypotheses.md`)."""
    p.peaked = p.peaked or bool(plan.overbought.get(day))
    p.armed = p.armed or bool(plan.above50.get(day))
    if fail == "any":
        return bool(plan.oversold.get(day))
    if fail == "before80":
        return not p.peaked and bool(plan.oversold.get(day))
    if fail == "ma50":
        return p.armed and bool(plan.below50.get(day))
    return False


def simulate(
    plans: list[Plan],
    days: list[date],
    regime: dict[date, str],
    capital: float,
    park: dict[date, float],
    log: list[dict] | None = None,
    fail: str | None = None,
    tilt: str | None = None,
) -> tuple[pf.Result, list[float]]:
    """log를 주면 청산마다 종목·기간·사유·원화 손익을 쌓는다."""
    by = {p.code: p for p in plans}
    cash = capital
    pos: dict[str, Pos] = {}
    sells: dict[str, str] = {}
    buys: list[tuple[str, int, float]] = []
    curve, invested, trades, closed = [], [], 0, []
    last: dict[str, float] = {}
    factor: dict[str, float] = {}

    def close(code: str, p: Pos, px: float, day: date, reason: str) -> float:
        fee = pf._cost(code, "sell")
        got = p.qty * px * (1 - fee)
        closed.append(px / p.cost - 1 - pf._cost(code, "buy") - fee)
        if log is not None:
            log.append(
                {
                    "code": code,
                    "name": by[code].name,
                    "opened": p.opened,
                    "closed": day,
                    "reason": reason,
                    "tranches": p.tranches,
                    "cost": p.cost,
                    "strong": p.was_strong,
                    "failed": p.failed,
                    "score": by[code].score.get(p.opened),
                    "spent": p.spent,
                    "pnl": got - p.spent,
                    "ret": closed[-1],
                }
            )
        return got

    for day in days:
        cash0 = cash
        for code in list(sells):
            bar = by[code].bars.get(day)
            if bar is None or code not in pos:
                continue
            cash += close(code, pos.pop(code), bar.open, day, sells.pop(code))
            trades += 1
        for code, qty, ref in buys:
            bar = by[code].bars.get(day)
            limit = ref * trade.GAP_BUFFER
            if bar is None or bar.low > limit or code in sells:
                continue
            px = min(bar.open, limit)
            qty = min(qty, math.floor(cash / (px * (1 + pf._cost(code, "buy")))))
            if qty < 1:
                continue
            paid = qty * px * (1 + pf._cost(code, "buy"))
            cash -= paid
            p = pos.get(code)
            if p is None:
                pos[code] = Pos(qty, px, 1, opened=day, spent=paid)
            else:
                p.cost = (p.cost * p.qty + px * qty) / (p.qty + qty)
                p.qty += qty
                p.tranches += 1
                p.spent += paid
            trades += 1
        buys = []
        for code in list(pos):
            bar = by[code].bars.get(day)
            if bar is None:
                continue
            stop = pos[code].cost * (1 - STOP)
            if bar.low <= stop:
                cash += close(code, pos.pop(code), min(stop, bar.open), day, "stop")
                sells.pop(code, None)
                trades += 1
        cash -= abs(cash - cash0) * pf._cost(pf.PARK_CODE, "buy")
        cash *= 1 + park.get(day, 0.0) * (1 - pf.PARK_RESERVE)
        for code, p in by.items():
            if day in p.bars:
                last[code] = p.bars[day].close
        equity = cash + sum(p.qty * last.get(c, p.cost) for c, p in pos.items())
        curve.append((day, equity))
        invested.append(1 - cash / equity if equity else 0)
        gate = regime.get(day) in GATE_OPEN
        budget = cash
        for code, p in pos.items():
            plan = by[code]
            if day not in plan.bars:
                continue
            if failed_breakout(p, plan, day, fail):
                sells[code] = "fail"
                continue
            strong = bool(plan.strong.get(day))
            p.was_strong = p.was_strong or strong
            hold = strong and not plan.htf_exit.get(day)
            if p.was_strong and plan.htf_exit.get(day):
                sells[code] = "htf"
                continue
            if plan.take.get(day) and not hold:
                sells[code] = "take"
                continue
            p.failed = p.failed or bool(plan.below20.get(day))
            nxt = None
            if p.tranches == 1 and p.failed and plan.add2.get(day):
                nxt = TRANCHES[1]
            elif p.tranches == 2 and plan.add3.get(day):
                nxt = TRANCHES[2]
            if gate and nxt:
                size = equity * plan.weight * nxt * factor.get(code, 1.0)
                qty = math.floor(min(size, budget) / (last[code] * 1.03))
                if qty >= 1:
                    buys.append((code, qty, last[code]))
                    budget -= qty * last[code] * 1.03
        slots = MAX_POSITIONS - len(pos) + len(sells)
        if gate:
            for plan in plans:
                if slots <= 0:
                    break
                if plan.code in pos or day not in plan.bars or not plan.entry.get(day):
                    continue
                price = last[plan.code] * 1.03
                scale = size_factor(plan, day, tilt)
                size = equity * plan.weight * TRANCHES[0] * scale
                qty = math.floor(min(size, budget) / price)
                if qty >= 1:
                    factor[plan.code] = scale
                    buys.append((plan.code, qty, last[plan.code]))
                    budget -= qty * price
                    slots -= 1
    cagr, mdd = pf.stats(curve)
    return pf.Result(curve[-1][1], cagr, mdd, trades, sum(invested) / len(invested), curve), closed


def load(daily: dict) -> tuple[list[tuple], list[Bar]]:
    vix = dm.completed(fetch_yahoo("^VIX", "max", ttl=10**9), "US")
    rows = [("069500.KS", "KODEX 200", trade.INDEX_ETF_WEIGHT)]
    rows += [
        (a["symbol"], a["name"], trade.WEIGHT)
        for a in daily["assets"]
        if "error" not in a and a.get("market") == "KR" and not a["symbol"].startswith("^")
    ]
    return rows, vix


def verified_only(plans: list[Plan], allowed) -> list[Plan]:
    """allowed(code, day)가 참인 날에만 새로 산다. 보유 종목의 청산·추가 매수는 그대로다."""
    return [
        replace(p, entry={d: v and allowed(p.code, d) for d, v in p.entry.items()}) for p in plans
    ]


def compare_verified(plans: list[Plan], daily, vix, idx, regime, park, capital) -> None:
    """SCR-V: 앞은 오늘 목록, 뒤는 월별 워크포워드 목록으로 신규 매수를 제한한다."""
    from minerva.strategy import walkforward as wf

    today = {u.code for u in pf.load_universe(daily, vix)}
    front = [b.day for b in idx if date(2010, 1, 1) <= b.day < SPLIT]
    res, closed = simulate(
        verified_only(plans, lambda c, d: c in today), front, regime, capital, park
    )
    print("앞 SCR-V", f"{res.cagr:.1%} {res.mdd:.1%} {res.cagr / abs(res.mdd):.2f}", res.trades)
    _, months = wf.load_universe(daily, vix, SPLIT, compute=False)
    starts = sorted(months)

    def allowed(code: str, day: date) -> bool:
        i = bisect.bisect_right(starts, day) - 1
        if i < 0:
            return False
        m = months[starts[i]]
        return m["index_long"] if code == "069500" else f"{code}.KS" in m["stocks"]

    back = [b.day for b in idx if b.day >= SPLIT]
    res, closed = simulate(verified_only(plans, allowed), back, regime, capital, park)
    print("뒤 SCR-V", f"{res.cagr:.1%} {res.mdd:.1%} {res.cagr / abs(res.mdd):.2f}", res.trades)
    win = sum(x > 0 for x in closed) / max(1, len(closed))
    print("뒤 SCR-V 거래", len(closed), f"승률 {win:.1%} 최악 {min(closed):.1%}")


def live_variant(u: pf.Instrument, plan: Plan, name: str) -> pf.Instrument:
    """현행 종목에 변형을 입힌다 (`docs/backtests/2026-10-31-live-exit-hypotheses.md`)."""
    if name == "L2":
        days = set(u.take) | set(plan.htf_exit)
        take = {
            d: bool(plan.htf_exit.get(d)) or (bool(u.take.get(d)) and not plan.strong.get(d))
            for d in days
        }
        return replace(u, take=take)
    if name == "L3":
        return replace(u, entry={d: v and plan.score.get(d) == 3 for d, v in u.entry.items()})
    return u


def block_entries(u: pf.Instrument, spans: list[tuple[date, date | None]]) -> pf.Instrument:
    """월봉 제한 구간의 신규 매수를 뺀다. 신호 달 다음 날부터 해제 달 마감일까지다."""

    def blocked(d: date) -> bool:
        return any(a < d and (b is None or d <= b) for a, b in spans)

    return replace(u, entry={d: v and not blocked(d) for d, v in u.entry.items()})


def compare_live(daily: dict, vix: list[Bar], capital: float) -> None:
    from minerva.strategy import walkforward as wf

    idx = dm.completed(fetch_yahoo("^KS11", "max", ttl=10**9), "KR")
    regime = dm._regime_by_day(vix, idx)
    park = pf.park_returns()
    back_uni, _ = wf.load_universe(daily, vix, SPLIT, compute=False)
    segments = (
        (
            "앞",
            pf.load_universe(daily, vix),
            [b.day for b in idx if date(2010, 1, 1) <= b.day < SPLIT],
        ),
        ("뒤", back_uni, [b.day for b in idx if b.day >= SPLIT]),
    )
    plans: dict[str, Plan] = {}
    for _, uni, _ in segments:
        for u in uni:
            if u.code not in plans:
                bars = [u.prices[d] for d in sorted(u.prices)]
                plan = plan_for(u.code, u.name, u.cap, bars)
                if plan is not None:
                    plans[u.code] = plan
    for name, uni, days in segments:
        kept = [u for u in uni if u.code in plans]
        print(name, "종목", len(uni), "사용", len(kept), flush=True)
        spans = dm.block_spans(tm_states(resample(idx, "M")), dm.monthly_touches(idx))
        ranked = [a["symbol"].split(".")[0] for a in daily["assets"] if a["symbol"].endswith(".KS")]
        top = {"069500", *ranked[: dm.TRADE_TOP]}
        for label in ("현행", "L1", "L2", "L3", "M1", "M1-100"):
            hold = 10**6 if label in ("L1", "L2") else 20
            cooked = [live_variant(u, plans[u.code], label) for u in kept]
            if label.startswith("M1"):
                cooked = [block_entries(u, spans) for u in cooked]
            if label == "M1-100":
                cooked = [u for u in cooked if u.code in top]
            for limit in (trade.GAP_BUFFER, None):
                cfg = pf.Config(capital=capital, limit=limit, park=park, hold=hold)
                r = pf.simulate(cooked, days, regime, cfg)
                print(
                    name,
                    label,
                    "지정가" if limit else "시가",
                    f"{r.cagr:.1%} {r.mdd:.1%} {r.cagr / abs(r.mdd):.2f}",
                    r.trades,
                    f"{r.exposure:.0%}",
                    flush=True,
                )


def monthly_table(res: pf.Result, capital: float) -> list[tuple[str, float, int, int, float]]:
    """월, 수익률, 매수, 매도, 월말 평가액. 체결일 기준이다."""
    ends: dict[str, float] = {}
    for d, e in res.equity:
        ends[d.isoformat()[:7]] = e
    rows, prev = [], capital
    for month, end in ends.items():
        buys = sum(f[2] == "buy" and f[0].isoformat()[:7] == month for f in res.fills)
        sells = sum(f[2] == "sell" and f[0].isoformat()[:7] == month for f in res.fills)
        rows.append((month, end / prev - 1, buys, sells, end))
        prev = end
    return rows


def monthly_report(daily: dict, vix: list[Bar], start: date, capital: float) -> None:
    """현행 규칙(상위 100·월봉 제한)을 월별로 보여준다. 검증은 월별 워크포워드다."""
    from minerva.strategy import walkforward as wf

    idx = dm.completed(fetch_yahoo("^KS11", "max", ttl=10**9), "KR")
    regime = dm._regime_by_day(vix, idx)
    park = pf.park_returns()
    family = dm.trade_family(daily["assets"])
    pool = {**daily, "assets": family}  # 다중 검정 묶음을 실전과 같은 상위 100으로 맞춘다
    uni, _ = wf.load_universe(pool, vix, start)
    top = {"069500", *(a["symbol"].split(".")[0] for a in family)}
    spans = dm.block_spans(tm_states(resample(idx, "M")), dm.monthly_touches(idx))
    now = [block_entries(u, spans) for u in uni if u.code in top]
    days = [b.day for b in idx if b.day >= start]
    print("제한", [(str(a), str(b)) for a, b in spans if b is None or b >= start])
    print("종목", len(now), [u.name for u in now])
    for label, limit in (("지정가", trade.GAP_BUFFER), ("시가", None)):
        res = pf.simulate(now, days, regime, pf.Config(capital=capital, limit=limit, park=park))
        print(label, f"{res.cagr:.1%} {res.mdd:.1%} {res.exposure:.0%}", res.final)
        rows = monthly_table(res, capital)
        for month, ret, buys, sells, end in rows:
            print(month, f"{ret:+.2%}", buys, sells, round(end))
        print(
            "합계",
            sum(r[2] for r in rows),
            sum(r[3] for r in rows),
            f"{res.final / capital - 1:+.1%}",
        )


def main(argv: list[str] | None = None) -> None:
    default = "~/.minerva/daily/history/2026-10-29-1631.json"
    parser = argparse.ArgumentParser(description="SCR 재설계 전략 백테스트")
    parser.add_argument("--daily", default=str(Path(default).expanduser()))
    parser.add_argument("--capital", type=float, default=10_002_669)
    parser.add_argument("--verified", action="store_true", help="SCR-V: 검증 종목만 신규 매수")
    parser.add_argument("--fail", choices=("any", "before80", "ma50"), help="돌파 실패 매도 변형")
    parser.add_argument("--tilt", choices=("prop", "all3", "topdown"), help="일치 점수 변형")
    parser.add_argument("--live", action="store_true", help="현행 규칙 매도·진입 변형 비교")
    parser.add_argument("--monthly", help="현행 규칙 월별 성과 (시작일 YYYY-MM-DD)")
    args = parser.parse_args(argv)
    daily = json.loads(Path(args.daily).read_text())
    rows, vix = load(daily)
    if args.live:
        compare_live(daily, vix, args.capital)
        return
    if args.monthly:
        monthly_report(daily, vix, date.fromisoformat(args.monthly), args.capital)
        return
    idx = dm.completed(fetch_yahoo("^KS11", "max", ttl=10**9), "KR")
    regime = dm._regime_by_day(vix, idx)
    park = pf.park_returns()
    plans, base = [], []
    idx_entry, idx_take = pf._signals_for(idx, "long", vix)
    for sym, name, weight in rows:
        bars = dm.completed(fetch_yahoo(sym, "max", ttl=10**9), "KR")
        plan = plan_for(sym.split(".")[0], name, weight, bars)
        if plan is None:
            continue
        plans.append(plan)
        entry, take = pf._signals_for(bars, "long", vix)
        if sym == "069500.KS":
            entry, take = idx_entry, idx_take
        base.append(pf.Instrument(plan.code, name, "long", weight, plan.bars, entry, take))
    print("종목", len(plans), flush=True)
    if args.verified:
        compare_verified(plans, daily, vix, idx, regime, park, args.capital)
        return
    cfg = pf.Config(capital=args.capital, limit=trade.GAP_BUFFER, park=park)
    for name, days in (
        ("앞", [b.day for b in idx if date(2010, 1, 1) <= b.day < SPLIT]),
        ("뒤", [b.day for b in idx if b.day >= SPLIT]),
    ):
        res, closed = simulate(
            plans, days, regime, args.capital, park, fail=args.fail, tilt=args.tilt
        )
        was = pf.simulate(base, days, regime, cfg)
        hold = pf.index_hold(days[0], args.capital, 1.0, park)
        window = set(days)
        hold_cagr, hold_mdd = pf.stats([x for x in hold.equity if x[0] in window])
        for label, r in (("SCR", res), ("기준", was)):
            print(
                name,
                label,
                f"{r.cagr:.1%} {r.mdd:.1%} {r.cagr / abs(r.mdd):.2f}",
                r.trades,
                f"{r.exposure:.0%}",
            )
        win = sum(x > 0 for x in closed) / max(1, len(closed))
        print(
            name,
            "SCR 거래",
            len(closed),
            f"승률 {win:.1%} 평균 {sum(closed) / max(1, len(closed)):.2%} 최악 {min(closed):.1%}",
        )
        print(name, "KODEX 100%", f"{hold_cagr:.1%} {hold_mdd:.1%}", flush=True)


if __name__ == "__main__":
    main()
