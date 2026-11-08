"""바이낸스 USDT-M BTCUSDT 1분봉 내려받기 (data.binance.vision 월별·일별 + fapi 공개 시세).

월마다 npz 하나(t int64 ms, o/h/l/c float64)로 ~/.minerva/data/btc/futures/1m/에 둔다. 이미 있으면 건너뛴다.
실행: python3 docs/backtests/raw/fut1m_data.py [--start 2020-02]
"""

from __future__ import annotations

import argparse
import io
import json
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

DIR = Path("~/.minerva/data/btc/futures/1m").expanduser()
BASE = "https://data.binance.vision/data/futures/um"
UA = {"User-Agent": "minerva-research"}


def get(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def parse_zip(raw: bytes) -> np.ndarray:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        txt = z.read(z.namelist()[0]).decode()
    rows = [ln.split(",") for ln in txt.splitlines() if ln and ln[0].isdigit()]
    return np.array([[float(x) for x in r[:5]] for r in rows])


def save(path: Path, a: np.ndarray) -> None:
    a = a[np.argsort(a[:, 0])]
    _, k = np.unique(a[:, 0], return_index=True)
    a = a[k]
    np.savez_compressed(path, t=a[:, 0].astype(np.int64), o=a[:, 1], h=a[:, 2], l=a[:, 3], c=a[:, 4])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-02")
    args = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    y, m = map(int, args.start.split("-"))
    while (y, m) < (now.year, now.month):
        p = DIR / f"{y:04d}-{m:02d}.npz"
        if not p.exists():
            raw = get(f"{BASE}/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-{y:04d}-{m:02d}.zip")
            if raw is None:
                print("missing", y, m, flush=True)
            else:
                save(p, parse_zip(raw))
                print("ok", p.name, flush=True)
            time.sleep(0.5)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    # 이번 달: 일별 압축 + 오늘은 fapi
    parts = []
    d = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    while d.date() < now.date():
        raw = get(f"{BASE}/daily/klines/BTCUSDT/1m/BTCUSDT-1m-{d:%Y-%m-%d}.zip")
        if raw is not None:
            parts.append(parse_zip(raw))
        time.sleep(0.3)
        d += timedelta(days=1)
    start = int(parts[-1][-1, 0]) + 60_000 if parts else int(datetime(now.year, now.month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end = int(now.timestamp() * 1000)
    rows = []
    while start < end:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&startTime={start}&limit=1500"
        r = json.loads(get(url))
        if not r:
            break
        rows += [[float(x) for x in k[:5]] for k in r if k[6] < end]
        start = r[-1][0] + 60_000
        time.sleep(0.3)
    if rows:
        parts.append(np.array(rows))
    if parts:
        save(DIR / f"{now.year:04d}-{now.month:02d}.npz", np.vstack(parts))
        print("ok current month", flush=True)


if __name__ == "__main__":
    main()
