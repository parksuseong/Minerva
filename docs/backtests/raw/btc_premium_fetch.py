"""136번 자료 받기: Coinbase BTC-USD 1시간봉, Upbit KRW-BTC 1시간봉, ECB USD/KRW 일별, Deribit BTC DVOL 1시간.

저장: ~/.minerva/data/btc/{coinbase,upbit,deribit,fx}/ (저장소 밖). 이어받기 가능.
실행: python docs/backtests/raw/btc_premium_fetch.py
"""

from __future__ import annotations

import csv
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

UA = {"User-Agent": "minerva-data/1.0"}
BASE = Path("~/.minerva/data/btc").expanduser()
START = datetime(2020, 1, 1, tzinfo=timezone.utc)
H = 3_600_000


def get(url: str, tries: int = 5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:  # 재시도 후 마지막엔 올린다
            if k == tries - 1:
                raise
            time.sleep(1.5 * (k + 1))
            last = e  # noqa: F841
    return None


def last_t(path: Path) -> int | None:
    if not path.exists():
        return None
    rows = path.read_text().strip().splitlines()
    return int(rows[-1].split(",")[0]) if len(rows) > 1 else None


def append(path: Path, header: list, rows: list) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def coinbase(now_ms: int) -> int:
    p = BASE / "coinbase/btcusd_1h.csv"
    lt = last_t(p)
    t = (lt + H) if lt else int(START.timestamp() * 1000)
    n = 0
    while t < now_ms - H:
        end = min(t + 300 * H, now_ms - H)
        s = datetime.fromtimestamp(t / 1000, timezone.utc).isoformat()
        e = datetime.fromtimestamp(end / 1000, timezone.utc).isoformat()
        data = get(f"https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600&start={s}&end={e}")
        rows = sorted([int(r[0]) * 1000, r[3], r[2], r[1], r[4], r[5]] for r in data if int(r[0]) * 1000 >= t)
        rows = [r for r in rows if r[0] + H <= now_ms]  # 마감된 봉만
        if rows:
            append(p, ["open_time", "open", "high", "low", "close", "volume"], rows)
            n += len(rows)
            t = rows[-1][0] + H
        else:
            t = end
        time.sleep(0.25)
    return n


def upbit(now_ms: int) -> int:
    """뒤에서부터 받는다(to=). 파일은 시간 순서로 다시 쓴다."""
    p = BASE / "upbit/krwbtc_1h.csv"
    have = {}
    if p.exists():
        for r in list(csv.reader(p.open()))[1:]:
            have[int(r[0])] = r
    lo = int(START.timestamp() * 1000)
    newest = max(have) if have else None
    oldest = min(have) if have else None
    got = 0

    def pull(to_ms: int, stop_ms: int):
        nonlocal got
        cur = to_ms
        while cur > stop_ms:
            to = datetime.fromtimestamp(cur / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            data = get(f"https://api.upbit.com/v1/candles/minutes/60?market=KRW-BTC&count=200&to={to}")
            if not data:
                break
            for c in data:
                ts = int(datetime.strptime(c["candle_date_time_utc"], "%Y-%m-%dT%H:%M:%S")
                         .replace(tzinfo=timezone.utc).timestamp() * 1000)
                if ts + H <= now_ms and ts >= lo and ts not in have:
                    have[ts] = [ts, c["opening_price"], c["high_price"], c["low_price"], c["trade_price"],
                                c["candle_acc_trade_volume"]]
                    got += 1
            cur = min(int(datetime.strptime(c["candle_date_time_utc"], "%Y-%m-%dT%H:%M:%S")
                          .replace(tzinfo=timezone.utc).timestamp() * 1000) for c in data)
            time.sleep(0.15)

    pull(now_ms, newest if newest else lo)
    if oldest and oldest > lo:
        pull(oldest, lo)
    rows = [have[k] for k in sorted(have)]
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        w.writerows(rows)
    return got


def fx() -> int:
    p = BASE / "fx/usdkrw_daily.csv"
    end = datetime.now(timezone.utc).date().isoformat()
    data = get(f"https://api.frankfurter.dev/v1/2019-12-01..{end}?from=USD&to=KRW")
    rows = sorted((d, v["KRW"]) for d, v in data["rates"].items())
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "usdkrw"])
        w.writerows(rows)
    return len(rows)


def dvol(now_ms: int) -> int:
    p = BASE / "deribit/btc_dvol_1h.csv"
    lt = last_t(p)
    t = (lt + H) if lt else int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    n = 0
    while t < now_ms - H:
        end = min(t + 900 * H, now_ms - H)
        data = get("https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC"
                   f"&start_timestamp={t}&end_timestamp={end}&resolution=3600")["result"]["data"]
        rows = sorted([int(r[0]), r[1], r[2], r[3], r[4]] for r in data if int(r[0]) >= t and int(r[0]) + H <= now_ms)
        if rows:
            append(p, ["t", "open", "high", "low", "close"], rows)
            n += len(rows)
            t = rows[-1][0] + H
        else:
            t = end
        time.sleep(0.2)
    return n


def main() -> None:
    now = int(time.time() * 1000)
    out = {"coinbase": coinbase(now), "upbit": upbit(now), "fx": fx(), "dvol": dvol(now)}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
