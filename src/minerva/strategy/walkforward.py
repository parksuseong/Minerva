"""워크포워드 유니버스: 월초마다 그 시점까지 데이터만으로 종목 검증을 다시 해 매수 대상을 정한다.

`portfolio.load_universe`는 오늘 검증 결과를 과거 전체에 쓴다(선택 편향). 여기서는 월초 d마다
d 전날까지의 봉으로 `daily.verify`와 같은 판정(비용 차감 + BH 다중 검정)을 내고, 그달에는
통과한 종목만 새로 산다. 이미 산 종목은 판정에서 빠져도 원래 청산 규칙(익절·기간·손절)대로 판다.

한계: 후보는 오늘의 코스피 시총 상위 200이다(생존 편향). 실전은 주 1회 재검증이지만 연산량
때문에 월 1회로 한다. 우선순위용 p값·우위는 종목마다 마지막 통과 시점 값 하나만 쓴다.
검증은 종목·날짜마다 수 초 걸려서 결과를 캐시에 쌓고, 끊기면 이어서 돈다.
실행: `python -m minerva.strategy.portfolio --walk-forward --start 2021-10-29`
"""

from __future__ import annotations

import json
import os
from datetime import date
from multiprocessing import Pool
from pathlib import Path

from minerva import trade
from minerva.strategy import daily as dm
from minerva.strategy.backtest import evaluate
from minerva.strategy.indicators import Bar
from minerva.strategy.report import CACHE_DIR, fetch_yahoo
from minerva.strategy.short import evaluate_short

# 검증 규칙·자료 처리가 바뀌면 옛 결과와 섞이지 않게 버전별 파일을 쓴다
CACHE = CACHE_DIR / f"walkforward-v{dm.VERIFY_VERSION}.json"
MIN_BARS = 300
INDEX = "^KS11"
_VIX: list[Bar] = []


def month_starts(days: list[date]) -> list[date]:
    firsts: dict[tuple[int, int], date] = {}
    for d in days:
        firsts.setdefault((d.year, d.month), d)
    return sorted(firsts.values())


def window_of(day: date, starts: list[date]) -> date | None:
    past = [m for m in starts if m <= day]
    return past[-1] if past else None


def gate(flags: dict[date, bool], starts: list[date], ok) -> dict[date, bool]:
    """진입 신호를 그달 월초 판정이 통과한 날에만 남긴다."""
    out = {}
    for d, f in flags.items():
        m = window_of(d, starts)
        out[d] = bool(f and m is not None and ok(m))
    return out


def _init(vix: list[Bar]) -> None:
    global _VIX
    _VIX = vix


def verify_at(job: tuple[str, str]) -> tuple[str, dict | None]:
    """d 전날까지의 봉으로 롱 규칙(지수는 숏도) 판정. 비용 차감 전 값이다."""
    symbol, day = job
    d = date.fromisoformat(day)
    bars = [b for b in dm.completed(fetch_yahoo(symbol, "max"), "KR") if b.day < d]
    if len(bars) < MIN_BARS:
        return f"{symbol}|{day}", None
    ctx = dm._ctx(_VIX, bars)
    out = {
        "long": {
            **dm._verdict(evaluate(bars, dm.TM_RULE, ctx)),
            "p_value": dm.timing_pvalue(bars, ctx),
        }
    }
    if symbol == INDEX:
        flags = dm.short_flags(bars)[0]
        out["short"] = {
            **dm._verdict(evaluate_short(bars, flags, dm.SHORT_RULE)),
            "p_value": dm.short_pvalue(bars, flags),
        }
    return f"{symbol}|{day}", out


def _save(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    os.replace(tmp, path)


def run_verifications(
    symbols: list[str], starts: list[date], vix: list[Bar], path: Path = CACHE, workers: int = 2
) -> dict:
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [
        (s, d.isoformat()) for d in starts for s in symbols if f"{s}|{d.isoformat()}" not in cache
    ]
    print(f"검증 {len(todo)}건 남음 (캐시 {len(cache)}건)", flush=True)
    with Pool(workers, initializer=_init, initargs=(vix,)) as pool:
        for i, (key, v) in enumerate(pool.imap_unordered(verify_at, todo, chunksize=4), 1):
            cache[key] = v
            if i % 50 == 0 or i == len(todo):
                _save(path, cache)
                print(f"{i}/{len(todo)}", flush=True)
    return cache


def passed_by_month(cache: dict, stocks: list[str], starts: list[date]) -> dict[date, dict]:
    """월초별 통과: 종목 {심볼: (p, 비용 후 우위)}, 지수 롱·숏 통과 여부."""
    out = {}
    for m in starts:
        k = m.isoformat()
        assets = [
            {
                "symbol": s,
                "verify": {"long": dm.with_cost(cache[f"{s}|{k}"]["long"], dm.COST_KR_STOCK)},
            }
            for s in stocks
            if cache.get(f"{s}|{k}")
        ]
        dm.apply_fdr(assets)
        idx = cache.get(f"{INDEX}|{k}") or {}
        out[m] = {
            "stocks": {
                a["symbol"]: (
                    a["verify"]["long"].get("p_value") or 1.0,
                    a["verify"]["long"]["net_edge"] or 0.0,
                )
                for a in assets
                if a["verify"]["long"]["label"] == "검증됨"
            },
            "index_long": "long" in idx and dm.product_verdict(idx["long"])["label"] == "검증됨",
            "index_short": "short" in idx and dm.product_verdict(idx["short"])["label"] == "검증됨",
        }
    return out


def load_universe(daily: dict, vix: list[Bar], start: date, path: Path = CACHE, workers: int = 2):
    from minerva.strategy import portfolio as pf

    idx = dm.completed(fetch_yahoo(INDEX, "max"), "KR")
    starts = month_starts([b.day for b in idx if b.day >= start])
    stocks = {
        a["symbol"]: a["name"]
        for a in daily.get("assets", [])
        if "error" not in a and a.get("market") == "KR" and not a["symbol"].startswith("^")
    }
    cache = run_verifications([INDEX, *stocks], starts, vix, path, workers)
    months = passed_by_month(cache, list(stocks), starts)
    out = []
    for sym, name, kind, key, cap in (
        ("069500.KS", "KODEX 200", "long", "index_long", trade.INDEX_ETF_WEIGHT),
        ("114800.KS", "KODEX 인버스", "inv", "index_short", trade.INVERSE_TOTAL),
    ):
        prices = dm.completed(fetch_yahoo(sym, "max"), "KR")
        entry, take = pf._signals_for(idx, kind, vix)
        ok = lambda m, k=key: months[m][k]  # noqa: E731
        out.append(
            pf.Instrument(
                sym.split(".")[0],
                name,
                kind,
                cap,
                {b.day: b for b in prices},
                gate(entry, starts, ok),
                take,
            )
        )
    for s in sorted({s for m in months.values() for s in m["stocks"]}):
        bars = dm.completed(fetch_yahoo(s, "max"), "KR")
        entry, take = pf._signals_for(bars, "long", vix)
        p, edge = [months[m]["stocks"][s] for m in starts if s in months[m]["stocks"]][-1]
        ok = lambda m, s=s: s in months[m]["stocks"]  # noqa: E731
        out.append(
            pf.Instrument(
                s.split(".")[0],
                stocks[s],
                "long",
                trade.WEIGHT,
                {b.day: b for b in bars},
                gate(entry, starts, ok),
                take,
                p,
                edge,
            )
        )
    return out, months
