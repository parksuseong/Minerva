"""시험용 선물 5분봉·펀딩 최신화 (~/.minerva/data/btc/futures).

바이낸스 공식 아카이브(data.binance.vision, CHECKSUM 확인)를 먼저 쓰고, 아직 나오지 않은 날만
OKX BTC-USDT-SWAP로 채운다(마지막 칸 "okx"). 실행할 때마다 "okx" 표시가 남은 지난 날을 다시 보고,
바이낸스 아카이브가 나왔으면 바이낸스 봉으로 바꾼다. fapi.binance.com은 이 서버(미국)에서 451로
막혀 있고, 우회 경로는 쓰지 않는다.

실행: python3 docs/backtests/raw/btc_fut_update.py [--backup <폴더>]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

B = Path.home() / ".minerva/data/btc/futures"
K, F = B / "klines_5m.csv", B / "funding.csv"
BNV = "https://data.binance.vision/data/futures/um"
OKX = "https://www.okx.com/api/v5"
M5 = 300_000
DAY = 86_400_000
UA = {"User-Agent": "minerva-data/1.0"}


def get(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def vision(url: str) -> list | None:
    blob = get(url)
    if blob is None:
        return None
    want = get(url + ".CHECKSUM").decode().split()[0]
    if hashlib.sha256(blob).hexdigest() != want:
        raise ValueError(f"checksum mismatch {url}")
    z = zipfile.ZipFile(io.BytesIO(blob))
    rows = list(csv.reader(io.TextIOWrapper(z.open(z.namelist()[0]))))
    return [x for x in rows if x and x[0].isdigit()]


def day_name(dk: int) -> str:
    return datetime.fromtimestamp(dk * 86400, timezone.utc).strftime("%Y-%m-%d")


def okx_after(after_t: int, now_ms: int) -> list:
    out, cur = {}, None
    while True:
        u = f"{OKX}/market/history-candles?instId=BTC-USDT-SWAP&bar=5m&limit=100"
        if cur:
            u += f"&after={cur}"
        data = json.loads(get(u))["data"]
        if not data:
            break
        for r in data:
            t = int(r[0])
            if t > after_t and r[8] == "1" and t + M5 <= now_ms:
                out[t] = [t, *r[1:5], r[6], t + M5 - 1, r[7], "nan", "nan", "nan", "okx"]
        cur = data[-1][0]
        if int(cur) <= after_t:
            break
        time.sleep(0.15)
    return [out[t] for t in sorted(out)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", default="")
    args = ap.parse_args()
    if args.backup:
        Path(args.backup).mkdir(parents=True, exist_ok=True)
        shutil.copy2(K, Path(args.backup) / K.name)
        shutil.copy2(F, Path(args.backup) / F.name)
    with K.open() as f:
        r = csv.reader(f)
        head = next(r)
        rows = {int(x[0]): x for x in r if x and x[0].isdigit()}
    now = int(time.time() * 1000)
    today = now // DAY
    rep = {"binance_days": [], "okx_days_left": [], "okx_rows_added": 0, "funding_added": 0}
    # 1) okx로 채운 지난 날 → 바이낸스로 교체
    okx_days = sorted({t // DAY for t, x in rows.items() if x[-1] == "okx" and t // DAY < today})
    # 2) 마지막 봉 뒤로 빠진 날: 바이낸스 먼저
    last = max(rows)
    dk = (last + M5) // DAY
    todo = sorted(set(okx_days) | set(range(dk, today)))
    for d in todo:
        got = vision(f"{BNV}/daily/klines/BTCUSDT/5m/BTCUSDT-5m-{day_name(d)}.zip")
        if not got or len(got) != 288:
            continue
        for t in [t for t in rows if t // DAY == d]:
            rows.pop(t)
        for x in got:
            rows[int(x[0])] = x[: len(head)]
        rep["binance_days"].append(day_name(d))
    # 3) 그래도 빈 끝부분은 OKX
    last = max(rows)
    for x in okx_after(last, now):
        rows[int(x[0])] = x
        rep["okx_rows_added"] += 1
    rep["okx_days_left"] = sorted({day_name(t // DAY) for t, x in rows.items() if x[-1] == "okx"})
    ts = sorted(rows)
    gaps = [(a, b) for a, b in zip(ts, ts[1:]) if b - a != M5]
    tmp = K.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for t in ts:
            w.writerow(rows[t])
    tmp.replace(K)
    # 펀딩: 지난 달은 바이낸스 월별, 이번 달은 OKX
    with F.open() as f:
        r = csv.reader(f)
        fhead = next(r)
        fr = {int(x[0]): x for x in r if x and x[0].isdigit()}
    cur = datetime.now(timezone.utc).strftime("%Y-%m")
    lf = datetime.fromtimestamp(max(fr) / 1000, timezone.utc)
    ym = lf.strftime("%Y-%m")
    while ym < cur:
        got = vision(f"{BNV}/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{ym}.zip")
        if got:
            for x in got:
                if int(x[0]) not in fr:
                    fr[int(x[0])] = x[:3]
                    rep["funding_added"] += 1
        ym = (datetime.strptime(ym + "-01", "%Y-%m-%d") + timedelta(days=32)).strftime("%Y-%m")
    url = f"{OKX}/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=100"
    data = json.loads(get(url))["data"]
    for x in data:
        t = int(x["fundingTime"])
        if t > max(fr) and t <= now:
            fr[t] = [t, 8, x["realizedRate"]]
            rep["funding_added"] += 1
    tmp = F.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fhead)
        for t in sorted(fr):
            w.writerow(fr[t])
    tmp.replace(F)
    rep["kline_last"] = datetime.fromtimestamp(ts[-1] / 1000, timezone.utc).isoformat()
    rep["gaps"] = len(gaps)
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
