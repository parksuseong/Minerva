"""포트폴리오 백테스트: 실전 운용 규칙(비중·우선순위·리밸런싱·교체)을 한 계좌로 돌린다.

종목별 백테스트(backtest.py)는 "돈이 무한하고 한 종목만 산다"고 가정한다. 실전은 예수금이
한정돼 있고 정수 주만 사며, 교체·리밸런싱 같은 운용 규칙이 붙는다. 이 모듈은 그 규칙들이
"신호대로만 사는 것"보다 나은지 비교한다 (ROADMAP 13).

한계: 종목 목록·p값은 오늘 판정을 과거에 그대로 쓴다(선택 편향). 변형끼리 비교하는 용도로만 쓴다.
실행: `python -m minerva.strategy.portfolio [--capital 1000000] [--walk-forward]`
선택 편향을 뺀 비교는 `--walk-forward` (`strategy/walkforward.py`).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

from minerva import trade
from minerva.strategy import daily as dm
from minerva.strategy.backtest import filter_mask, signals
from minerva.strategy.indicators import Bar, crossed_down, stoch_fast_k
from minerva.strategy.report import fetch_yahoo

FEE = 0.00015
SELL_TAX = 0.002
SLIP = 0.0005
ETFS = {"069500"}
REGIME_FACTOR = trade.REGIME_FACTOR
PARK_CODE = "357870"
PARK_RESERVE = 0.02  # 실전도 평가액 2%는 현금으로 남긴다
PARK_URL = (
    f"https://fchart.stock.naver.com/sise.nhn?symbol={PARK_CODE}"
    "&timeframe=day&count=3000&requestType=0"
)
PARK_CACHE = Path("~/.minerva/data/357870_naver.xml").expanduser()


@dataclass
class Instrument:
    code: str
    name: str
    kind: str  # long (숏·인버스는 2026-10-30부터 안 한다)
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
    priority: bool = True
    hold: int = 20
    long_stop: float = 0.10
    trim_at: float = 1.5
    rotate_ratio: float = 0.5
    rotate_min_days: int = 5
    limit: float | None = None  # 매수 지정가 = 전일 종가 × limit. None이면 시가에 무조건 체결
    park: dict[date, float] | None = None  # 날짜별 파킹 ETF 수익률. None이면 현금 이자 0


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
    unfilled: list[tuple[date, str, float]] = field(default_factory=list)  # 날짜, 코드, 시가/지정가


def _signals_for(src: list[Bar], kind: str, vix: list[Bar], mode: str = "tm") -> tuple[dict, dict]:
    """mode tm: 실전 규칙(필터 + TM 눌림 진입, Stoch 80 이탈 익절).
    mode filter: 타이밍 없이 필터(50일선 위·VIX 관문·주봉/월봉 상승)가 맞으면 사고 깨지면 판다
    (ROADMAP 23: 우위가 타이밍이 아니라 필터에서 나오는지 비교)."""
    if mode == "filter":
        mask = filter_mask(src, dm.TM_RULE, dm._ctx(vix, src))
        return (
            {b.day: f for b, f in zip(src, mask)},
            {b.day: not f for b, f in zip(src, mask)},
        )
    st = stoch_fast_k(src)
    flags = signals(src, dm.TM_RULE, dm._ctx(vix, src))
    take = [i > 0 and crossed_down(st, i, 80.0) for i in range(len(src))]
    return (
        {b.day: f for b, f in zip(src, flags)},
        {b.day: t for b, t in zip(src, take)},
    )


def load_universe(daily: dict, vix: list[Bar], codes: set[str] | None = None) -> list[Instrument]:
    """오늘 자동 주문 대상(검증됨)을 과거 데이터로 다시 만든다."""
    out = []
    idx = dm.completed(fetch_yahoo("^KS11", "max"), "KR")
    for p in daily.get("products", []):
        # 숏·인버스는 하지 않는다 (2026-10-30). 예전 분석 파일의 인버스 상품은 건너뛴다
        if p.get("kind") == "inv":
            continue
        v = p.get("verify", {}).get("long", {})
        code = p["symbol"].split(".")[0]
        if v.get("label") != "검증됨":
            continue
        prices = dm.completed(fetch_yahoo(p["symbol"], "max"), "KR")
        entry, take = _signals_for(idx, "long", vix)
        out.append(
            Instrument(
                code,
                p["name"],
                "long",
                trade.INDEX_ETF_WEIGHT,
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
    factor_of=lambda r: REGIME_FACTOR.get(r, 1 / 3),
) -> Result:
    by = {u.code: u for u in universe}
    cash = cfg.capital
    pos: dict[str, Pos] = {}
    pending_sell: dict[str, tuple[int, str]] = {}
    pending_buy: list[tuple[str, int, float]] = []
    curve, invested, trades = [], [], 0
    unfilled: list[tuple[date, str, float]] = []
    last_px: dict[str, float] = {}

    def value(code: str) -> float:
        return pos[code].qty * last_px.get(code, pos[code].entry)

    for t, day in enumerate(days):
        cash0 = cash
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
        for code, qty, ref in pending_buy:
            bar = by[code].prices.get(day)
            if bar is None or code in pos:
                continue
            px = bar.open
            if cfg.limit is not None:
                limit = ref * cfg.limit
                if bar.low > limit:  # 하루 종일 지정가 위 → 미체결, 당일 만료
                    unfilled.append((day, code, bar.open / limit))
                    continue
                px = min(bar.open, limit)
            qty = min(qty, math.floor(cash / (px * (1 + _cost(code, "buy")))))
            if qty < 1:
                continue
            cash -= qty * px * (1 + _cost(code, "buy"))
            pos[code] = Pos(qty, px, t)
            trades += 1
        pending_buy = []
        # 2) 장중 손절
        for code in list(pos):
            bar = by[code].prices.get(day)
            if bar is None:
                continue
            stop = pos[code].entry * (1 - cfg.long_stop)
            if bar.low <= stop:
                px = min(stop, bar.open)
                cash += pos[code].qty * px * (1 - _cost(code, "sell"))
                del pos[code]
                pending_sell.pop(code, None)
                trades += 1
        if cfg.park is not None:
            # 파킹 이자, 그리고 매매 때마다 ETF를 사고파는 비용
            cash -= abs(cash - cash0) * _cost(PARK_CODE, "buy")
            cash *= 1 + cfg.park.get(day, 0.0) * (1 - PARK_RESERVE)
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
        if cfg.trim:
            for code, p in pos.items():
                cap = equity * by[code].cap
                if code not in pending_sell and value(code) > cap * cfg.trim_at:
                    qty = math.ceil((value(code) - cap) / last_px[code])
                    if 0 < qty < p.qty:
                        pending_sell[code] = (qty, "trim")
        factor = factor_of(regime.get(day, "R3"))
        budget = cash + sum(
            q * last_px.get(c, 0) * (1 - _cost(c, "sell")) for c, (q, _) in pending_sell.items()
        )
        slots = cfg.max_positions - sum(
            1 for c in pos if not (c in pending_sell and pending_sell[c][0] >= pos[c].qty)
        )
        if cfg.priority:
            entries.sort(key=lambda u: (u.p, -u.edge))
        for u in entries:
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
            pending_buy.append((u.code, qty, last_px[u.code]))
            budget -= qty * price
            slots -= 1
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
        unfilled,
    )


VARIANTS = {
    "신호대로만": dict(trim=False, rotate=False, priority=False),
    "+우선순위": dict(trim=False, rotate=False),
    "+리밸런싱": dict(rotate=False),
    "+교체": dict(trim=False),
    "전부(실전)": dict(),
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


def parse_naver_closes(xml: str) -> list[tuple[date, float]]:
    rows = re.findall(r'data="(\d{8})\|[^|]*\|[^|]*\|[^|]*\|(\d+)\|', xml)
    return [(date(int(d[:4]), int(d[4:6]), int(d[6:])), float(c)) for d, c in rows]


def park_returns() -> dict[date, float]:
    """파킹 ETF의 날짜별 수익률. 상장(2020-07) 전은 비어 있다.
    야후는 2024-06 이전 봉이 거의 없어(2026-10-31 확인) 네이버 일봉을 쓴다."""
    if not PARK_CACHE.exists() or time.time() - PARK_CACHE.stat().st_mtime > 86400:
        with urlopen(Request(PARK_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=20) as r:
            PARK_CACHE.write_bytes(r.read())
    closes = parse_naver_closes(PARK_CACHE.read_text(encoding="euc-kr", errors="replace"))
    if len(closes) < 1000:
        raise RuntimeError(f"파킹 ETF 일봉이 {len(closes)}개뿐이다")
    return {d: c / p - 1 for (_, p), (d, c) in zip(closes, closes[1:])}


def index_hold(
    start: date, capital: float, weight: float, park: dict[date, float] | None = None
) -> Result:
    """KODEX 200을 첫날 weight만큼 사서 끝까지 들고 있는다 (나머지는 현금). 같은 비용 모델."""
    bars = dm.completed(fetch_yahoo("069500.KS", "max"), "KR")
    days = [b.day for b in bars if b.day >= start]
    u = Instrument(
        "069500",
        "KODEX 200",
        "long",
        weight,
        {b.day: b for b in bars},
        {days[0]: True} if days else {},
        {},
    )
    cfg = Config(capital=capital, trim=False, rotate=False, hold=10**9, long_stop=1.0, park=park)
    return simulate([u], days, {}, cfg, factor_of=lambda _r: 1.0)


def compare_rules(daily: dict, vix: list[Bar], start: date, capital: float, workers: int = 2):
    """ROADMAP 23: 워크포워드 같은 종목·같은 비용·같은 비중으로 ① 현행 ② 필터만 ③ 지수 보유."""
    from minerva.strategy import walkforward

    out = {}
    for name, mode, hold in (("① 현행(필터+TM)", "tm", 20), ("② 필터만", "filter", 10**9)):
        uni, _ = walkforward.load_universe(
            daily, vix, start, workers=workers, mode=mode, compute=False
        )
        idx = dm.completed(fetch_yahoo("^KS11", "max"), "KR")
        days = [b.day for b in idx if b.day >= start]
        out[name] = simulate(
            uni, days, dm._regime_by_day(vix, idx), Config(capital=capital, hold=hold)
        )
    out[f"③ 지수 보유 {trade.INDEX_ETF_WEIGHT:.0%}"] = index_hold(
        start, capital, trade.INDEX_ETF_WEIGHT
    )
    out["③ 지수 보유 100%"] = index_hold(start, capital, 1.0)
    return out


def compare_execution(daily: dict, vix: list[Bar], start: date, capital: float, workers: int = 2):
    """ROADMAP 20②: 워크포워드에 매수 지정가와 CD 파킹을 넣어 실전 집행과 맞춘다."""
    from minerva.strategy import walkforward

    uni, _ = walkforward.load_universe(daily, vix, start, workers=workers, compute=False)
    idx = dm.completed(fetch_yahoo("^KS11", "max"), "KR")
    days = [b.day for b in idx if b.day >= start]
    regime = dm._regime_by_day(vix, idx)
    park = park_returns()
    base = Config(capital=capital)
    limit = trade.GAP_BUFFER
    out = {
        name: simulate(uni, days, regime, replace(base, **kw))
        for name, kw in (
            ("기존(시가 체결·이자 0)", {}),
            ("+지정가", dict(limit=limit)),
            ("+파킹", dict(park=park)),
            ("실전 집행(지정가+파킹)", dict(limit=limit, park=park)),
        )
    }
    for w in (trade.INDEX_ETF_WEIGHT, 1.0):
        out[f"지수 보유 {w:.0%}+파킹"] = index_hold(start, capital, w, park)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="실전 운용 규칙 포트폴리오 백테스트")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--daily", default=str(Path("~/.minerva/daily/latest.json").expanduser()))
    parser.add_argument("--codes", help="쉼표로 구분한 종목 코드 (기본: 오늘 검증됨)")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="월초마다 그 시점 데이터로 다시 검증한 종목만 산다 (선택 편향 제거, 수 시간)",
    )
    parser.add_argument(
        "--rules",
        action="store_true",
        help="워크포워드로 현행·필터만·지수 보유 비교 (ROADMAP 23, 워크포워드 캐시 사용)",
    )
    parser.add_argument(
        "--execution",
        action="store_true",
        help="워크포워드로 매수 지정가·CD 파킹 반영 전후 비교 (ROADMAP 20②, 워크포워드 캐시 사용)",
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    daily = json.loads(Path(args.daily).read_text())
    vix = dm.completed(fetch_yahoo("^VIX", "max"), "US")
    if args.execution:
        start = date.fromisoformat(args.start)
        _print(compare_execution(daily, vix, start, args.capital, args.workers))
        return
    if args.rules:
        res = compare_rules(daily, vix, date.fromisoformat(args.start), args.capital, args.workers)
        _print(res)
        return
    if args.walk_forward:
        from minerva.strategy import walkforward

        uni, months = walkforward.load_universe(
            daily, vix, date.fromisoformat(args.start), workers=args.workers
        )
        for m, v in months.items():
            print(
                m,
                f"지수 롱 {'O' if v['index_long'] else 'X'}",
                f"종목 {len(v['stocks'])}개",
            )
    else:
        codes = set(args.codes.split(",")) if args.codes else None
        uni = load_universe(daily, vix, codes)
    print("종목:", ", ".join(u.name for u in uni))
    _print(compare(uni, date.fromisoformat(args.start), args.capital))


def _print(res: dict[str, Result]) -> None:
    print(f"{'변형':<22}{'최종':>14}{'연수익':>8}{'최대낙폭':>9}{'거래':>6}{'투자비율':>8}")
    for name, r in res.items():
        print(
            f"{name:<22}{r.final:>14,.0f}{r.cagr:>8.1%}{r.mdd:>9.1%}{r.trades:>6}{r.exposure:>8.0%}"
        )


if __name__ == "__main__":
    main()
