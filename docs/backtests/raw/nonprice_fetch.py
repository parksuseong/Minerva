"""바이낸스 USDT-M BTCUSDT 일별 metrics 파일(5분 OI·롱숏 비율·테이커 비율)을 체크섬과 함께 받는다.

문서: docs/backtests/2026-11-08-btc-nonprice-hypotheses.md
- 출처: data.binance.vision futures/um/daily/metrics/BTCUSDT (공개 아카이브)
- 저장: ~/.minerva/data/btc/futures/metrics/BTCUSDT-metrics-YYYY-MM-DD.zip (저장소 밖)
- 한 번에 하나씩, 요청 사이 0.25초 쉰다. 이미 받아 체크섬이 맞는 파일은 건너뛴다.
- 결과 목록은 같은 폴더의 fetch_log.json (받은 날, 없는 날, 체크섬 실패).
실행: python3 -I docs/backtests/raw/nonprice_fetch.py --start 2021-12-01
표준 라이브러리만 쓴다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/"
DST = Path("~/.minerva/data/btc/futures/metrics").expanduser()


def get(url: str) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "minerva-research"})
    for k in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if k == 3:
                raise
        except OSError:
            if k == 3:
                raise
        time.sleep(2 + 3 * k)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-12-01")
    ap.add_argument("--end", default=None, help="마지막 날(포함), 기본 오늘(UTC)")
    a = ap.parse_args()
    DST.mkdir(parents=True, exist_ok=True)
    d = dt.date.fromisoformat(a.start)
    end = dt.date.fromisoformat(a.end) if a.end else dt.datetime.now(dt.timezone.utc).date()
    log = {"ok": [], "skipped": [], "missing": [], "bad_checksum": []}
    while d <= end:
        name = f"BTCUSDT-metrics-{d.isoformat()}.zip"
        f = DST / name
        ck = get(BASE + name + ".CHECKSUM")
        time.sleep(0.25)
        if ck is None:
            log["missing"].append(d.isoformat())
            d += dt.timedelta(days=1)
            continue
        want = ck.decode().split()[0].strip().lower()
        if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == want:
            log["skipped"].append(d.isoformat())
            d += dt.timedelta(days=1)
            continue
        body = get(BASE + name)
        time.sleep(0.25)
        if body is None:
            log["missing"].append(d.isoformat())
        elif hashlib.sha256(body).hexdigest() != want:
            log["bad_checksum"].append(d.isoformat())
        else:
            f.write_bytes(body)
            log["ok"].append(d.isoformat())
        if d.day == 1:
            print(d, len(log["ok"]), len(log["missing"]), len(log["bad_checksum"]), flush=True)
        d += dt.timedelta(days=1)
    (DST / "fetch_log.json").write_text(json.dumps(log, indent=1))
    print({k: len(v) for k, v in log.items()}, "missing:", log["missing"][-5:])


if __name__ == "__main__":
    main()
