"""종목의 일·주·월봉 TM 레벨과 이평선 상태를 한 번에 출력한다.

사용: python -m minerva.strategy.report SOXX ^VIX 005930.KS
      python -m minerva.strategy.report --csv data.csv
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from minerva.strategy.indicators import (
    LEVELS,
    Bar,
    load_csv,
    ma_snapshot,
    resample,
    tm_states,
)

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range={rng}&interval=1d"


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


def fetch_yahoo(symbol: str, rng: str = "10y") -> list[Bar]:
    url = YAHOO_URL.format(sym=urllib.parse.quote(symbol), rng=rng)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return parse_yahoo(json.load(resp))


def summarize(bars: list[Bar], cci_n: int = 20) -> dict:
    out = {}
    for frame in ("D", "W", "M"):
        fb = resample(bars, frame)
        if len(fb) < 20:
            continue
        s = tm_states(fb, cci_n=cci_n)[-1]
        ma = ma_snapshot([b.close for b in fb])
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
            "above_ma": [n for n, up in ma["above"].items() if up],
            "bull_aligned": ma["bull_aligned"],
            "bear_aligned": ma["bear_aligned"],
        }
    return out


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
        print(json.dumps({name: summarize(bars, args.cci)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
