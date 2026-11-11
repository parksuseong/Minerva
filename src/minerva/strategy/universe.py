"""분석 대상 종목군: 코스피 시총 상위 100, 나스닥 시총 상위 10.

순위는 네이버 증권 공개 API에서 받아 하루 동안 캐시한다. 받기에 실패하면
마지막 캐시를, 그것도 없으면 아래 고정 목록을 쓴다.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import urllib.request
from pathlib import Path

from minerva.strategy.report import CACHE_DIR

TTL = 24 * 3600
KOSPI_URL = "https://m.stock.naver.com/api/stocks/marketValue/KOSPI?pageSize=80&page={page}"
MAX_PAGES = 6  # 한 쪽에 80개(ETF·우선주 포함)까지만 와서 여러 쪽을 받는다
NASDAQ_URL = "https://api.stock.naver.com/stock/exchange/NASDAQ/marketValue?page=1&pageSize=30"

FALLBACK_KOSPI = [
    ("005930.KS", "삼성전자"),
    ("000660.KS", "SK하이닉스"),
    ("373220.KS", "LG에너지솔루션"),
    ("207940.KS", "삼성바이오로직스"),
    ("005380.KS", "현대차"),
    ("105560.KS", "KB금융"),
    ("068270.KS", "셀트리온"),
    ("000270.KS", "기아"),
    ("012450.KS", "한화에어로스페이스"),
    ("035420.KS", "NAVER"),
]
FALLBACK_NASDAQ = [
    ("NVDA", "엔비디아"),
    ("AAPL", "애플"),
    ("MSFT", "마이크로소프트"),
    ("AMZN", "아마존닷컴"),
    ("GOOGL", "알파벳"),
    ("AVGO", "브로드컴"),
    ("TSLA", "테슬라"),
    ("META", "메타"),
]


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def _read(path: Path) -> list[tuple[str, str]] | None:
    """캐시 목록. 없거나 깨졌으면 None (깨진 캐시 하나로 일일 분석 전체가 멈추지 않게)."""
    try:
        return [tuple(x) for x in json.loads(path.read_text())] or None
    except (OSError, ValueError, TypeError):
        return None


def _write(path: Path, items: list[tuple[str, str]]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(items, ensure_ascii=False))
        tmp.replace(path)
    except OSError:  # 캐시를 못 써도 방금 받은 순위는 쓴다
        with contextlib.suppress(OSError):
            tmp.unlink()


def _cached(name: str, fetch, fallback: list[tuple[str, str]]) -> list[tuple[str, str]]:
    path = CACHE_DIR / f"universe-{name}.json"
    if path.exists() and time.time() - path.stat().st_mtime < TTL:
        items = _read(path)
        if items:
            return items
    try:
        items = fetch()
    except Exception:  # noqa: BLE001 — 순위를 못 받아도 분석은 돈다
        items = None
    if items:
        _write(path, items)
        return items
    return _read(path) or fallback


def pick_kospi(stocks: list[dict], n: int) -> list[tuple[str, str]]:
    """보통주만 (ETF·우선주 제외). 쪽을 나눠 받는 사이 순위가 바뀌면 같은 종목이 또 올 수 있다."""
    out, seen = [], set()
    for s in stocks:
        name = s["stockName"]
        if s.get("stockEndType") != "stock" or name.endswith(("우", "우B", "우C")):
            continue
        if s["itemCode"] in seen:
            continue
        seen.add(s["itemCode"])
        out.append((f"{s['itemCode']}.KS", name))
        if len(out) == n:
            break
    return out


def pick_nasdaq(stocks: list[dict], n: int) -> list[tuple[str, str]]:
    """보통주만, 같은 회사의 다른 주식 종류(Class A/C)는 하나만."""
    out, seen = [], set()
    for s in stocks:
        if s.get("stockEndType") != "stock":
            continue
        base = s["stockName"].split(" Class")[0]
        if base in seen:
            continue
        seen.add(base)
        out.append((s["symbolCode"], base))
        if len(out) == n:
            break
    return out


def _kospi_stocks(n: int) -> list[dict]:
    stocks: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        batch = _get_json(KOSPI_URL.format(page=page))["stocks"]
        stocks += batch
        if not batch or len(pick_kospi(stocks, n)) >= n:
            break
    return stocks


def kospi_top(n: int = 100) -> list[tuple[str, str]]:
    return _cached(f"kospi{n}", lambda: pick_kospi(_kospi_stocks(n), n), FALLBACK_KOSPI)


def nasdaq_top(n: int = 10) -> list[tuple[str, str]]:
    return _cached(
        f"nasdaq{n}", lambda: pick_nasdaq(_get_json(NASDAQ_URL)["stocks"], n), FALLBACK_NASDAQ
    )
