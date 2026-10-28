"""포트폴리오 백테스트: 실전 운용 규칙(비중·우선순위·인버스 전환·리밸런싱·교체)을 한 계좌로 돌린다.

종목별 백테스트(backtest.py)는 "돈이 무한하고 한 종목만 산다"고 가정한다. 실전은 예수금이
한정돼 있고 정수 주만 사며, 교체·리밸런싱 같은 운용 규칙이 붙는다. 이 모듈은 그 규칙들이
"신호대로만 사는 것"보다 나은지 비교한다 (ROADMAP 13).

한계: 종목 목록·p값은 오늘 판정을 과거에 그대로 쓴다(선택 편향). 변형끼리 비교하는 용도로만 쓴다.
실행: `python -m minerva.strategy.portfolio [--capital 1000000]`
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

from minerva import trade
from minerva.strategy import daily as dm
from minerva.strategy.backtest import signals
from minerva.strategy.indicators import Bar, crossed_down, crossed_up, stoch_fast_k
from minerva.strategy.report import fetch_yahoo

FEE = 0.00015
SELL_TAX = 0.002
SLIP = 0.0005
ETFS = {"069500", "114800"}
INDEX_ETFS = trade.INDEX_ETFS
REGIME_FACTOR = trade.REGIME_FACTOR


@dataclass
class Instrument:
    code: str
    name: str
    kind: str  # long / inv
    cap: float
    prices: dict[date, Bar]
    entry: dict[date, bool]
    take: dict[date, bool]
    p: float = 1.0
    edge: float = 0.0


@dataclass
class Config:
    capital: float = 1_000_000
    max_positions: int = 10
    trim: bool = True
    rotate: bool = True
    switch: bool = True
    unswitch: bool = True  # KODEX 200 롱 신호가 오면 인버스를 판다 (switch의 반대 방향)
    priority: bool = True
    hold: int = 20
    long_stop: float = 0.10
    inv_stop: float = 0.07
    trim_at: float = 1.5
    rotate_ratio: float = 0.5
    rotate_min_days: int = 5


@dataclass
class Pos:
    qty: int
    entry: float
    day0: int  # 진입한 날의 인덱스
    reason: str = ""


@dataclass
class Result:
    final: float
    cagr: float
    mdd: float
    trades: int
    exposure: float
    equity: list[tuple[date, float]] = field(default_factory=list)


def _signals_for(src: list[Bar], kind: str, vix: list[Bar]) -> tuple[dict, dict]:
    st = stoch_fast_k(src)
    if kind == "long":
        flags = signals(src, dm.TM_RULE, dm._ctx(vix, src))
        take = [i > 0 and crossed_down(st, i, 80.0) for i in range(len(src))]
    else:
        flags = dm.short_flags(src)[0]
        take = [i > 0 and crossed_up(st, i, 20.0) for i in range(len(src))]
    return (
        {b.day: f for b, f in zip(src, flags)},
        {b.day: t for b, t in zip(src, take)},
    )


def load_universe(daily: dict, vix: list[Bar], codes: set[str] | None = None) -> list[Instrument]:
    """오늘 자동 주문 대상(검증됨)을 과거 데이터로 다시 만든다."""
    out = []
    idx = dm.completed(fetch_yahoo("^KS11", "max"), "KR")
    for p in daily.get("products", []):
        side = "short" if p.get("kind") == "inv" else "long"
        v = p.get("verify", {}).get(side, {})
        code = p["symbol"].split(".")[0]
        if v.get("label") != "검증됨":
            continue
        prices = dm.completed(fetch_yahoo(p["symbol"], "max"), "KR")
        entry, take = _signals_for(idx, "long" if side == "long" else "inv", vix)
        out.append(
            Instrument(
                code,
                p["name"],
                "long" if side == "long" else "inv",
                trade.INDEX_ETF_WEIGHT if side == "long" else trade.INVERSE_TOTAL,
                {b.day: b for b in prices},
                entry,
                take,
                v.get("p_value") or 1.0,
                v.get("net_edge") or 0.0,
            )
        )
    for a in daily.get("assets", []):
        if "error" in a or a.get("market") != "KR" or a["symbol"].startswith("^"):
            continue
        v = a["verify"]["long"]
        code = a["symbol"].split(".")[0]
        if codes is not None:
            if code not in codes:
                continue
        elif v.get("label") != "검증됨":
            continue
        bars = dm.completed(fetch_yahoo(a["symbol"], "max"), "KR")
        entry, take = _signals_for(bars, "long", vix)
        out.append(
            Instrument(
                code,
                a["name"],
                "long",
                trade.WEIGHT,
                {b.day: b for b in bars},
                entry,
                take,
                v.get("p_value") or 1.0,
                v.get("net_edge") or 0.0,
            )
        )
    return out


def _cost(code: str, side: str) -> float:
    c = FEE + SLIP
    if side == "sell" and code not in ETFS:
        c += SELL_TAX
    return c


def simulate(
    universe: list[Instrument],
    days: list[date],
    regime: dict[date, str],
    cfg: Config,
) -> Result:
    by = {u.code: u for u in universe}
    cash = cfg.capital
    pos: dict[str, Pos] = {}
    pending_sell: dict[str, tuple[int, str]] = {}
    pending_buy: list[tuple[str, int]] = []
    curve, invested, trades = [], [], 0
    last_px: dict[str, float] = {}

    def value(code: str) -> float:
        return pos[code].qty * last_px.get(code, pos[code].entry)

    for t, day in enumerate(days):
        # 1) 시가: 전날 정한 매도 → 매수
        for code, (qty, _why) in list(pending_sell.items()):
            bar = by[code].prices.get(day)
            if bar is None or code not in pos:
                continue
            qty = min(qty, pos[code].qty)
            cash += qty * bar.open * (1 - _cost(code, "sell"))
            pos[code].qty -= qty
            if pos[code].qty == 0:
                del pos[code]
            del pending_sell[code]
            trades += 1
        for code, qty in pending_buy:
            bar = by[code].prices.get(day)
            if bar is None or code in pos:
                continue
            qty = min(qty, math.floor(cash / (bar.open * (1 + _cost(code, "buy")))))
            if qty < 1:
                continue
            cash -= qty * bar.open * (1 + _cost(code, "buy"))
            pos[code] = Pos(qty, bar.open, t)
            trades += 1
        pending_buy = []
        # 2) 장중 손절
        for code in list(pos):
            bar = by[code].prices.get(day)
            if bar is None:
                continue
            stop = pos[code].entry * (
                1 - (cfg.inv_stop if by[code].kind == "inv" else cfg.long_stop)
            )
            if bar.low <= stop:
                px = min(stop, bar.open)
                cash += pos[code].qty * px * (1 - _cost(code, "sell"))
                del pos[code]
                pending_sell.pop(code, None)
                trades += 1
        # 3) 종가: 평가
        for code, u in by.items():
            if day in u.prices:
                last_px[code] = u.prices[day].close
        equity = cash + sum(value(c) for c in pos)
        curve.append((day, equity))
        invested.append(1 - cash / equity if equity else 0)
        # 4) 종가: 내일 낼 주문
        for code, p in pos.items():
            u = by[code]
            if u.take.get(day) or t - p.day0 + 1 >= cfg.hold:
                pending_sell[code] = (p.qty, "exit")
        entries = [
            u for u in universe if u.entry.get(day) and u.code not in pos and day in u.prices
        ]
        if cfg.switch and any(u.kind == "inv" for u in entries):
            for code in pos:
                if by[code].kind == "long":
                    pending_sell[code] = (pos[code].qty, "switch")
        if cfg.unswitch and any(u.code in INDEX_ETFS for u in entries):
            for code in pos:
                if by[code].kind == "inv":
                    pending_sell[code] = (pos[code].qty, "switch")
        if cfg.trim:
            for code, p in pos.items():
                cap = equity * by[code].cap
                if code not in pending_sell and value(code) > cap * cfg.trim_at:
                    qty = math.ceil((value(code) - cap) / last_px[code])
                    if 0 < qty < p.qty:
                        pending_sell[code] = (qty, "trim")
        factor = REGIME_FACTOR.get(regime.get(day, "R3"), 1 / 3)
        budget = cash + sum(
            q * last_px.get(c, 0) * (1 - _cost(c, "sell")) for c, (q, _) in pending_sell.items()
        )
        slots = cfg.max_positions - sum(
            1 for c in pos if not (c in pending_sell and pending_sell[c][0] >= pos[c].qty)
        )
        if cfg.priority:
            entries.sort(key=lambda u: (u.p, -u.edge, u.code != "114800"))
        inv_taken = any(by[c].kind == "inv" for c in pos)
        for u in entries:
            if u.kind == "inv" and inv_taken:
                continue
            alloc = equity * u.cap * factor
            price = last_px[u.code] * 1.03
            qty = math.floor(min(alloc, budget) / price)
            if (slots <= 0 or qty < 1) and cfg.rotate and alloc >= price:
                weak = sorted(
                    (
                        c
                        for c in pos
                        if c not in pending_sell and t - pos[c].day0 >= cfg.rotate_min_days
                    ),
                    key=lambda c: by[c].p,
                    reverse=True,
                )
                picks, cash2, free = [], budget, slots
                for c in weak:
                    if u.p > by[c].p * cfg.rotate_ratio or u.edge - by[c].edge <= _cost(c, "sell"):
                        break
                    picks.append(c)
                    cash2 += value(c) * (1 - _cost(c, "sell"))
                    free += 1
                    if free > 0 and math.floor(min(alloc, cash2) / price) >= 1:
                        break
                if picks and free > 0 and math.floor(min(alloc, cash2) / price) >= 1:
                    for c in picks:
                        pending_sell[c] = (pos[c].qty, "rotate")
                    budget, slots = cash2, free
                    qty = math.floor(min(alloc, budget) / price)
            if slots <= 0 or qty < 1:
                continue
            pending_buy.append((u.code, qty))
            budget -= qty * price
            slots -= 1
            inv_taken = inv_taken or u.kind == "inv"
    final = curve[-1][1] if curve else cfg.capital
    years = max((days[-1] - days[0]).days / 365.25, 1e-9) if days else 1
    peak, mdd = 0.0, 0.0
    for _, e in curve:
        peak = max(peak, e)
        mdd = min(mdd, e / peak - 1)
    return Result(
        final,
        (final / cfg.capital) ** (1 / years) - 1,
        mdd,
        trades,
        sum(invested) / len(invested) if invested else 0,
        curve,
    )


VARIANTS = {
    "신호대로만": dict(trim=False, rotate=False, priority=False),
    "+우선순위": dict(trim=False, rotate=False),
    "+리밸런싱": dict(rotate=False),
    "+교체": dict(trim=False),
    "전부(실전)": dict(),
    "전부, 인버스 전환 없음": dict(switch=False),
}


def compare(universe: list[Instrument], start: date, capital: float) -> dict[str, Result]:
    vix = dm.completed(fetch_yahoo("^VIX", "max"), "US")
    idx = dm.completed(fetch_yahoo("^KS11", "max"), "KR")
    days = [b.day for b in idx if b.day >= start]
    regime = dm._regime_by_day(vix, idx)
    base = Config(capital=capital)
    return {
        name: simulate(universe, days, regime, replace(base, **kw)) for name, kw in VARIANTS.items()
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="실전 운용 규칙 포트폴리오 백테스트")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--daily", default=str(Path("~/.minerva/daily/latest.json").expanduser()))
    parser.add_argument("--codes", help="쉼표로 구분한 종목 코드 (기본: 오늘 검증됨)")
    args = parser.parse_args(argv)
    daily = json.loads(Path(args.daily).read_text())
    vix = dm.completed(fetch_yahoo("^VIX", "max"), "US")
    codes = set(args.codes.split(",")) if args.codes else None
    uni = load_universe(daily, vix, codes)
    print("종목:", ", ".join(u.name for u in uni))
    res = compare(uni, date.fromisoformat(args.start), args.capital)
    print(f"{'변형':<22}{'최종':>14}{'연수익':>8}{'최대낙폭':>9}{'거래':>6}{'투자비율':>8}")
    for name, r in res.items():
        print(
            f"{name:<22}{r.final:>14,.0f}{r.cagr:>8.1%}{r.mdd:>9.1%}{r.trades:>6}{r.exposure:>8.0%}"
        )


if __name__ == "__main__":
    main()
