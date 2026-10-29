"""60분봉 스윙 백테스트 (사전 등록: docs/backtests/2026-10-30-h60-hypotheses.md).

현행 일봉 규칙을 시간 범위만 줄여서 1~3일 보유로 돌리면 어떤지 본다. 야후 60분봉은 최근 약 2년치만
있어서 표본이 짧다. 결과는 예비로만 쓴다.
실행: `python -m minerva.strategy.intraday [--start 2026-01-02]`
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from minerva.strategy import daily as dm
from minerva.strategy.backtest import Rule, Trade, filter_mask, signals, simulate
from minerva.strategy.indicators import Bar, sma, stoch_fast_k
from minerva.strategy.report import CACHE_DIR, fetch_yahoo

KST = timezone(timedelta(hours=9))
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval=60m"
# 60분봉 규칙: TM L1 진입, Stoch 80 이탈 익절, 손절 −5%, 최대 18봉(약 3거래일)
RULE_A = Rule(level=1, hold=18, stop=0.05)
RULE_B = Rule(level=1, above_ma=50, hold=18, stop=0.05)
INDEX_ETF = "069500.KS"


def parse_60m(payload: dict, before: date) -> list[Bar]:
    """야후 60분봉 → Bar(day=KST 날짜). 거래량 0인 채움 봉과 before 이후(진행 중) 봉은 버린다."""
    result = payload["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    out = []
    for i, ts in enumerate(result["timestamp"]):
        o, h, lo, c, v = (q[k][i] for k in ("open", "high", "low", "close", "volume"))
        if None in (o, h, lo, c) or not v:
            continue
        day = datetime.fromtimestamp(ts, tz=KST).date()
        if day >= before:
            continue
        out.append(Bar(day, o, h, lo, c, v))
    return out


def fetch_60m(symbol: str, before: date, rng: str = "1y") -> list[Bar]:
    path = CACHE_DIR / "intraday" / f"{symbol}-60m-{rng}.json"
    if path.exists() and time.time() - path.stat().st_mtime < 12 * 3600:
        return parse_60m(json.loads(path.read_text()), before)
    url = URL.format(sym=urllib.parse.quote(symbol), rng=rng)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.load(resp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return parse_60m(payload, before)


def prev_day_mask(bars: list[Bar], daily: list[Bar], mask: list[bool]) -> list[bool]:
    """60분봉마다 그 전 거래일 마감 기준 일봉 필터 값 (당일 종가는 아직 모른다)."""
    out, j, cur = [], 0, False
    for b in bars:
        while j < len(daily) and daily[j].day < b.day:
            cur = mask[j]
            j += 1
        out.append(cur)
    return out


def _run(bars: list[Bar], rule: Rule, sig: list[bool]) -> list[Trade]:
    return simulate(bars, rule, sig, stoch_fast_k(bars), sma([b.close for b in bars], 20))


def h60_signals(bars: list[Bar], start: date, daily_ok: list[bool] | None) -> list[bool]:
    """A(daily_ok 있음): 일봉 필터 + 60분봉 TM L1. B: 60분봉 50봉선 위 + TM L1."""
    rule = RULE_A if daily_ok is not None else RULE_B
    sig = signals(bars, rule)
    return [
        s and b.day >= start and (daily_ok is None or daily_ok[i])
        for i, (b, s) in enumerate(zip(bars, sig))
    ]


def pooled_pvalue(
    cases: list[tuple[list[Bar], Rule, list[bool], list[bool], float]],
    n_perm: int = 300,
    seed: int = 7,
) -> float | None:
    """종목 합산 거래당 비용 후 평균이 무작위 진입(같은 횟수·같은 청산)보다 나은지.
    cases: (봉, 규칙, 실제 신호, 무작위로 뽑을 수 있는 봉, 비용)."""
    actual = [t.ret - cost for bars, rule, sig, _, cost in cases for t in _run(bars, rule, sig)]
    if not actual:
        return None
    target = statistics.mean(actual)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        rets = []
        for bars, rule, sig, eligible, cost in cases:
            slots = [i for i in range(len(bars) - 1) if eligible[i]]
            k = min(sum(sig), len(slots))
            pick = set(rng.sample(slots, k))
            rand = [i in pick for i in range(len(bars))]
            rets += [t.ret - cost for t in _run(bars, rule, rand)]
        if rets and statistics.mean(rets) >= target:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def _stats(rets: list[float]) -> dict:
    if not rets:
        return {"n": 0}
    return {
        "n": len(rets),
        "mean": statistics.mean(rets),
        "median": statistics.median(rets),
        "win": sum(r > 0 for r in rets) / len(rets),
        "worst": min(rets),
        "sum": sum(rets),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="60분봉 스윙 백테스트 (H60-A·B)")
    parser.add_argument("--start", default="2026-01-02")
    parser.add_argument("--daily", default=str(Path("~/.minerva/daily/latest.json").expanduser()))
    args = parser.parse_args(argv)
    start = date.fromisoformat(args.start)
    today = datetime.now(KST).date()
    latest = json.loads(Path(args.daily).read_text())
    syms = [
        a["symbol"]
        for a in latest["assets"]
        if "error" not in a
        and a.get("market") == "KR"
        and not a["symbol"].startswith("^")
        and a["verify"]["long"].get("label") == "검증됨"
    ]
    names = {a["symbol"]: a["name"] for a in latest["assets"] + latest.get("products", [])}
    vix = dm.completed(fetch_yahoo("^VIX", "max"), "US")
    rows, cases = {}, {"A": [], "B": []}
    daily_rets: list[float] = []
    for sym in [*syms, INDEX_ETF]:
        under = "^KS11" if sym == INDEX_ETF else sym  # KODEX 200은 코스피 신호로 산다
        cost = dm.COST_ETF if sym == INDEX_ETF else dm.COST_KR_STOCK
        daily = dm.completed(fetch_yahoo(under, "max"), "KR")
        ctx = dm._ctx(vix, daily)
        dmask = filter_mask(daily, dm.TM_RULE, ctx)
        bars = fetch_60m(sym, today)
        ok = prev_day_mask(bars, daily, dmask)
        in_window = [b.day >= start for b in bars]
        row = {}
        for name, daily_ok, rule in (("A", ok, RULE_A), ("B", None, RULE_B)):
            sig = h60_signals(bars, start, daily_ok)
            eligible = [w and (daily_ok is None or daily_ok[i]) for i, w in enumerate(in_window)]
            cases[name].append((bars, rule, sig, eligible, cost))
            row[name] = _stats([t.ret - cost for t in _run(bars, rule, sig)])
        # 현행 일봉 규칙: 같은 기간에 진입한 거래 (KODEX 200은 코스피 지수 수익으로 근사)
        dsig = [s and b.day >= start for s, b in zip(signals(daily, dm.TM_RULE, ctx), daily)]
        drets = [t.ret - cost for t in _run(daily, dm.TM_RULE, dsig)]
        daily_rets += drets
        row["D"] = _stats(drets)
        rows[sym] = row
    print(f"기간 {start} ~ {today - timedelta(days=1)}, 비용 후 거래당 수익")
    print(
        f"{'종목':<16}{'A 횟수':>7}{'A 평균':>8}{'B 횟수':>7}{'B 평균':>8}"
        f"{'일봉 횟수':>9}{'일봉 평균':>9}"
    )
    for sym, r in rows.items():
        cells = []
        for k in ("A", "B", "D"):
            s = r[k]
            cells.append(f"{s['n']:>7}{(s['mean'] if s['n'] else 0):>+8.2%}")
        print(f"{names.get(sym, sym)[:12]:<16}" + "".join(cells))
    print()
    for name in ("A", "B"):
        rets = [t.ret - c for bars, rule, sig, _, c in cases[name] for t in _run(bars, rule, sig)]
        s = _stats(rets)
        p = pooled_pvalue(cases[name])
        if s["n"]:
            print(
                f"H60-{name}: {s['n']}회 · 평균 {s['mean']:+.2%} · 중앙값 {s['median']:+.2%} · "
                f"승률 {s['win']:.0%} · 최악 {s['worst']:+.1%} · 합계 {s['sum']:+.1%} · p={p:.3f}"
            )
    s = _stats(daily_rets)
    if s["n"]:
        print(
            f"현행 일봉: {s['n']}회 · 평균 {s['mean']:+.2%} · 중앙값 {s['median']:+.2%} · "
            f"승률 {s['win']:.0%} · 최악 {s['worst']:+.1%} · 합계 {s['sum']:+.1%}"
        )


if __name__ == "__main__":
    main()
