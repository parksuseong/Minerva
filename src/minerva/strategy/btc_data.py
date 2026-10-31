"""BTCUSDT 현물 봉 수집 (2017-09~).

공식 월별 파일을 체크섬과 맞춰 받고, 파일이 없는 최근 구간은 API로 채운다.

결과는 저장소 밖 `~/.minerva/data/btc/<봉>.csv`다. 마감된 봉만 남긴다.
실행: `python -m minerva.strategy.btc_data`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

from minerva.strategy.btc import SYMBOL
from minerva.strategy.btc import URL as KLINES

BASE = "https://data.binance.vision/data/spot/monthly/klines/{sym}/{iv}/{sym}-{iv}-{ym}.zip"
START = date(2017, 9, 1)
OUT = Path("~/.minerva/data/btc").expanduser()
# 화면 이름: (API 이름, 파일 폴더 이름, 봉 길이 분). 월봉은 길이가 달라 빈 구간을 세지 않는다
FRAMES = {
    "5m": ("5m", "5m", 5),
    "15m": ("15m", "15m", 15),
    "30m": ("30m", "30m", 30),
    "1h": ("1h", "1h", 60),
    "4h": ("4h", "4h", 240),
    "1d": ("1d", "1d", 1440),
    "1w": ("1w", "1w", 10080),
    "1M": ("1M", "1mo", None),
}
# 월별 파일은 주봉·월봉이 일부 빠져 있다(2026-10-31 확인). 양이 적어 API로 전부 받는다
API_ONLY = ("1w", "1M")
COLUMNS = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,trades,"
    "taker_buy_base,taker_buy_quote"
).split(",")


def get(url: str, tries: int = 4) -> bytes | None:
    """404는 없는 파일이라 None. 그 밖의 오류는 다시 시도하고 끝내 안 되면 올린다."""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if i == tries - 1:
                raise
        except OSError:
            if i == tries - 1:
                raise
        time.sleep(2**i)
    return None


def months(start: date, end: date) -> list[str]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def ms(value: str | int) -> int:
    """2025년부터 현물 파일의 시각은 마이크로초다. 밀리초로 맞춘다."""
    v = int(value)
    return v // 1000 if v >= 10**14 else v


def rows_of(lines: list[list]) -> list[list]:
    out = []
    for r in lines:
        if not str(r[0]).isdigit():  # 머리글이 있는 파일
            continue
        out.append([ms(r[0]), *r[1:6], ms(r[6]), *r[7:11]])
    return out


def read_zip(blob: bytes, checksum: bytes) -> list[list]:
    want = checksum.decode().split()[0]
    got = hashlib.sha256(blob).hexdigest()
    if got != want:
        raise ValueError(f"체크섬 불일치 {got} != {want}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        text = z.read(z.namelist()[0]).decode()
    return rows_of(list(csv.reader(io.StringIO(text))))


def month_file(job: tuple[str, str]) -> tuple[str, list[list] | None]:
    folder, ym = job
    url = BASE.format(sym=SYMBOL, iv=folder, ym=ym)
    blob = get(url)
    if blob is None:
        return ym, None
    sums = get(url + ".CHECKSUM")
    if sums is None:
        raise ValueError(f"체크섬 파일 없음 {url}")
    return ym, read_zip(blob, sums)


def fill_recent(api: str, after: int, now_ms: int) -> list[list]:
    """파일이 없는 구간. 1,000봉씩 받는다."""
    out: list[list] = []
    since = after + 1
    while True:
        url = KLINES.format(sym=SYMBOL, iv=api, n=1000) + f"&startTime={since}"
        page = json.loads(get(url) or b"[]")
        if not page:
            return out
        out += rows_of([r[:11] for r in page])
        last = page[-1][0]
        if len(page) < 1000 or last >= now_ms:
            return out
        since = last + 1


def gaps(times: list[int], minutes: int | None) -> list[tuple[int, int]]:
    """(앞 봉, 다음 봉) 사이가 한 봉보다 긴 곳."""
    if minutes is None:
        return []
    step = minutes * 60_000
    return [(a, b) for a, b in zip(times, times[1:]) if b - a != step]


def bad_rows(rows: list[list]) -> int:
    n = 0
    for r in rows:
        o, h, lo, c = (float(x) for x in r[1:5])
        if not (lo <= min(o, c) and h >= max(o, c) and lo > 0):
            n += 1
    return n


def collect(frame: str, out_dir: Path, now_ms: int, pool: ThreadPoolExecutor) -> dict:
    api, folder, minutes = FRAMES[frame]
    today = datetime.fromtimestamp(now_ms / 1000, timezone.utc).date()
    if frame in API_ONLY:
        missing: list[str] = []
        begin = int(datetime(START.year, START.month, START.day, tzinfo=timezone.utc).timestamp())
        merged = {r[0]: r for r in fill_recent(api, begin * 1000 - 1, now_ms)}
    else:
        results = list(pool.map(month_file, [(folder, ym) for ym in months(START, today)]))
        missing = [ym for ym, r in results if r is None]
        merged = {r[0]: r for _, rows in results if rows for r in rows}
        for r in fill_recent(api, max(merged), now_ms):
            merged[r[0]] = r
    rows = [merged[t] for t in sorted(merged) if merged[t][6] < now_ms]
    path = out_dir / f"{frame}.csv"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    tmp.replace(path)
    times = [r[0] for r in rows]
    holes = gaps(times, minutes)
    stamp = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).isoformat()[:16]  # noqa: E731
    big = sorted(holes, key=lambda x: x[0] - x[1])[:5]
    return {
        "frame": frame,
        "rows": len(rows),
        "first": stamp(times[0]),
        "last": stamp(times[-1]),
        "months_without_file": missing,
        "gaps": len(holes),
        "gap_minutes": sum((b - a) // 60_000 - minutes for a, b in holes) if minutes else 0,
        "largest_gaps": [(stamp(a), stamp(b)) for a, b in big],
        "bad_rows": bad_rows(rows),
        "bytes": path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--frames", nargs="*", default=list(FRAMES))
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    report = []
    with ThreadPoolExecutor(4) as pool:
        for frame in args.frames:
            item = collect(frame, out, now_ms, pool)
            report.append(item)
            print(json.dumps(item, ensure_ascii=False), flush=True)
    path = out / "report.json"
    old = json.loads(path.read_text()) if path.exists() else []
    done = {r["frame"] for r in report}
    merged = [r for r in old if r["frame"] not in done] + report
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
