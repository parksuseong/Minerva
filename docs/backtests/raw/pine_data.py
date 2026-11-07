"""파인 스크립트 10개 검증용 자료: 바이낸스 USDT-M 선물 5분봉 → 15분봉, 펀딩, 해시레이트.

문서: docs/backtests/2026-11-08-pine-10-hypotheses.md
- 저장 이력(`klines_5m.csv`)의 바이낸스 행 + 직전 작업이 fapi로 다시 받은 최근 130일 캐시
  + 그 뒤 지금까지 fapi 공개 시세. OKX 행은 버린다.
- 2020-01-01 이후 빈 5분봉을 세고 fapi로 다시 받아 채운다. 그래도 비면 목록으로 남긴다.
- 결과는 저장소 밖 out 폴더의 npz(15분봉)와 data.json(점검 내역)에 쓴다.
실행: PYTHONPATH=<pyml>:docs/backtests/raw python3 docs/backtests/raw/pine_data.py --out <폴더>
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

M5 = 300_000
M15 = 900_000
DAY = 86_400_000
T2020 = 1_577_836_800_000
HOME = Path("~/.minerva").expanduser()
FUT_HIST = HOME / "data/btc/futures/klines_5m.csv"
FUND_HIST = HOME / "data/btc/futures/funding.csv"
FAPI_CACHE = HOME / "research/btc-lastmonth-2026-11-07/fut5m_fapi.csv"
FAPI = "https://fapi.binance.com/fapi/v1"
HASH_URL = "https://api.blockchain.info/charts/hash-rate?timespan=all&format=json&sampled=false"


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "minerva-research"})
    for k in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except OSError:
            if k == 3:
                raise
            time.sleep(2)


def fetch_5m(since: int, until: int) -> list:
    """[since, until) 구간의 마감된 5분봉."""
    rows, s = [], since
    now = int(time.time() * 1000)
    while s < until:
        q = f"symbol=BTCUSDT&interval=5m&limit=1500&startTime={s}&endTime={until - 1}"
        k = get(f"{FAPI}/klines?{q}")
        k = [x for x in k if int(x[6]) < now]
        if not k:
            break
        rows += [(int(x[0]), *(float(v) for v in x[1:6])) for x in k]
        s = int(k[-1][0]) + M5
        if len(k) < 1500:
            break
    return rows


def missing(ts: np.ndarray, start: int) -> list:
    t = ts[ts >= start]
    d = np.diff(t)
    bad = np.flatnonzero(d != M5)
    return [(int(t[i]) + M5, int(t[i + 1])) for i in bad]


def build(out: Path) -> dict:
    rows: dict = {}
    okx = 0
    with FUT_HIST.open() as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            if x[-1] == "okx":
                okx += 1
                continue
            rows[int(x[0])] = tuple(float(v) for v in x[1:6])
    n_hist = len(rows)
    n_cache = 0
    with FAPI_CACHE.open() as f:
        for x in csv.reader(f):
            rows[int(x[0])] = tuple(float(v) for v in x[1:6])
            n_cache += 1
    last = max(rows)
    now = int(time.time() * 1000)
    new = fetch_5m(last + M5, now)
    for x in new:
        rows[x[0]] = x[1:]
    ts = np.array(sorted(rows), dtype=np.int64)
    gaps0 = missing(ts, T2020)
    filled = 0
    for a, b in gaps0:
        for x in fetch_5m(a, b):
            rows[x[0]] = x[1:]
            filled += 1
    ts = np.array(sorted(rows), dtype=np.int64)
    gaps1 = missing(ts, T2020)
    ts = ts[ts >= T2020]
    v = np.array([rows[int(t)] for t in ts])
    # 5분봉 → 15분봉 (빈 5분봉이 있으면 있는 것만으로)
    g = ts - ts % M15
    keys, idx = np.unique(g, return_index=True)
    ends = np.r_[idx[1:], len(ts)]
    o = v[idx, 0]
    c = v[ends - 1, 3]
    h = np.maximum.reduceat(v[:, 1], idx)
    lo = np.minimum.reduceat(v[:, 2], idx)
    vol = np.add.reduceat(v[:, 4], idx)
    # 마지막 15분봉이 덜 찼으면 버린다
    if ends[-1] - idx[-1] < 3 and keys[-1] + M15 > ts[-1] + M5 - 1:
        keys, o, h, lo, c, vol = keys[:-1], o[:-1], h[:-1], lo[:-1], c[:-1], vol[:-1]
    d15 = np.diff(keys)
    # 펀딩
    fund = {}
    with FUND_HIST.open() as f:
        r = csv.reader(f)
        next(r)
        for x in r:
            fund[int(x[0])] = float(x[2])
    s = max(fund) + 1
    while True:
        k = get(f"{FAPI}/fundingRate?symbol=BTCUSDT&limit=1000&startTime={s}")
        if not k:
            break
        for x in k:
            fund[int(x["fundingTime"])] = float(x["fundingRate"])
        s = int(k[-1]["fundingTime"]) + 1
        if len(k) < 1000:
            break
    ft = np.array(sorted(fund), dtype=np.int64)
    fr = np.array([fund[int(t)] for t in ft])
    # 해시레이트
    hj = get(HASH_URL)
    hv = hj["values"]
    ht = np.array([int(x["x"]) * 1000 for x in hv], dtype=np.int64)
    hr = np.array([float(x["y"]) for x in hv])
    np.savez(out / "data15.npz", t=keys, o=o, h=h, l=lo, c=c, v=vol, ft=ft, fr=fr, ht=ht, hr=hr)
    info = {
        "built_utc_ms": now,
        "hist_binance_rows": n_hist,
        "okx_rows_dropped": okx,
        "fapi_cache_rows": n_cache,
        "fapi_new_rows": len(new),
        "gaps_since_2020_before_fill": len(gaps0),
        "gap_bars_before_fill": int(sum((b - a) // M5 for a, b in gaps0)),
        "fapi_rows_fetched_for_gaps": filled,
        "gaps_after_fill": [[a, b, (b - a) // M5] for a, b in gaps1],
        "bars15": len(keys),
        "first15": int(keys[0]),
        "last15": int(keys[-1]),
        "missing15_after": int((d15 != M15).sum()),
        "funding_last": int(ft[-1]),
        "hash_last_day": int(ht[-1]),
    }
    (out / "data.json").write_text(json.dumps(info, indent=1))
    return info


def load(out: Path) -> dict:
    z = np.load(out / "data15.npz")
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p = Path(a.out).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    print(json.dumps(build(p), indent=1))
