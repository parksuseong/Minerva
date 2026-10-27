"""종목의 일·주·월봉 TM 레벨과 이평선 상태를 한 번에 출력한다.

사용: python -m minerva.strategy.report SOXX ^VIX 005930.KS
      python -m minerva.strategy.report --csv data.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from minerva.strategy.indicators import (
    LEVELS,
    Bar,
    dmi,
    load_csv,
    ma_snapshot,
    resample,
    tm_states,
)
from minerva.strategy.regime import vix_regimes

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?{span}&interval=1d"


def parse_yahoo(payload: dict) -> list[Bar]:
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    bars = []
    for i, ts in enumerate(result["timestamp"]):
        o, h, lo, c = (quote[k][i] for k in ("open", "high", "low", "close"))
        if None in (o, h, lo, c):
            continue
        day = datetime.fromtimestamp(ts, tz=timezone.utc).date()
        bars.append(Bar(day, o, h, lo, c, quote["volume"][i] or 0))
    return bars


CACHE_DIR = Path(os.environ.get("MINERVA_DATA_CACHE", "~/.minerva/data")).expanduser()
CACHE_TTL = float(
    os.environ.get("MINERVA_DATA_TTL", 1800)
)  # 초. 분석·백테스트가 같은 원자료를 재사용


def _cache_file(symbol: str, rng: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-." else "_" for c in symbol)
    return CACHE_DIR / f"{safe}_{rng}.json"


def fetch_yahoo(symbol: str, rng: str = "10y", ttl: float | None = None) -> list[Bar]:
    """야후 일봉. 원자료 JSON을 캐시해서 백테스트·분석이 다시 받지 않게 한다."""
    ttl = CACHE_TTL if ttl is None else ttl
    path = _cache_file(symbol, rng)
    if ttl > 0 and path.exists() and time.time() - path.stat().st_mtime < ttl:
        return parse_yahoo(json.loads(path.read_text()))
    # range=max는 야후가 월봉으로 내려주므로 기간을 직접 지정한다
    span = "period1=0&period2=9999999999" if rng == "max" else f"range={rng}"
    url = YAHOO_URL.format(sym=urllib.parse.quote(symbol), span=span)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.load(resp)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    except OSError:
        pass  # 캐시 실패는 조회 결과에 영향이 없다
    return parse_yahoo(payload)


def summarize(bars: list[Bar], cci_n: int = 20) -> dict:
    out: dict = {}
    for frame in ("D", "W", "M"):
        fb = resample(bars, frame)
        if len(fb) < 20:
            continue
        s = tm_states(fb, cci_n=cci_n)[-1]
        ma = ma_snapshot([b.close for b in fb])
        pdi, mdi, adx = (x[-1] for x in dmi(fb))
        out[frame] = {
            "date": s.day.isoformat(),
            "close": round(s.close, 4),
            "stoch": _r(s.stoch),
            "cci": _r(s.cci),
            "rsi": _r(s.rsi),
            "oversold": [n for n, lo, _ in LEVELS if _in(getattr(s, n), lo, below=True)],
            "overbought": [n for n, _, hi in LEVELS if _in(getattr(s, n), hi, below=False)],
            "long_level": s.long_level,
            "short_level": s.short_level,
            "pdi": _r(pdi),
            "mdi": _r(mdi),
            "adx": _r(adx),
            "above_ma": [n for n, up in ma["above"].items() if up],
            "ma_values": {n: round(v, 4) for n, v in ma["ma"].items()},
            "bull_aligned": ma["bull_aligned"],
            "bear_aligned": ma["bear_aligned"],
        }
    return out


def with_vix_regime(name: str, bars: list[Bar], summary: dict) -> dict:
    """VIX 계열이면 regime.py 기준 레짐 코드를 붙인다."""
    if name.upper() in ("^VIX", "VIX") and bars:
        summary["regime"] = vix_regimes(bars).get(bars[-1].day)
    return summary


def _in(value, line: float, below: bool) -> bool:
    if value is None:
        return False
    return value <= line if below else value >= line


def _r(v):
    return None if v is None else round(v, 1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("symbols", nargs="*")
    p.add_argument("--csv")
    p.add_argument("--cci", type=int, default=20)
    args = p.parse_args(argv)
    targets = [(args.csv, load_csv(args.csv))] if args.csv else []
    targets += [(s, fetch_yahoo(s)) for s in args.symbols]
    for name, bars in targets:
        summary = with_vix_regime(name, bars, summarize(bars, args.cci))
        print(json.dumps({name: summary}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
