import hashlib
import io
import zipfile
from datetime import date

import pytest

from minerva.strategy import btc_data as bd


def _zip(text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.csv", text)
    blob = buf.getvalue()
    return blob, f"{hashlib.sha256(blob).hexdigest()}  a.zip\n".encode()


LINE = "{},1,2,0.5,1.5,10,{},100,5,3,30,0\n"


def test_months_cover_year_end():
    assert bd.months(date(2017, 11, 1), date(2018, 2, 3)) == [
        "2017-11",
        "2017-12",
        "2018-01",
        "2018-02",
    ]


def test_read_zip_converts_microseconds_and_skips_header():
    text = "open_time,x\n" + LINE.format(1735689600000000, 1735689899999999)
    blob, sums = _zip(text)
    rows = bd.read_zip(blob, sums)
    assert rows[0][0] == 1735689600000 and rows[0][6] == 1735689899999


def test_read_zip_keeps_milliseconds():
    blob, sums = _zip(LINE.format(1504224000000, 1504224299999))
    assert bd.read_zip(blob, sums)[0][0] == 1504224000000


def test_read_zip_rejects_bad_checksum():
    blob, _ = _zip(LINE.format(1, 2))
    with pytest.raises(ValueError):
        bd.read_zip(blob, b"00  a.zip\n")


def test_month_file_needs_checksum(monkeypatch):
    monkeypatch.setattr(bd, "get", lambda url: None if url.endswith("CHECKSUM") else b"x")
    with pytest.raises(ValueError):
        bd.month_file(("5m", "2017-09"))


def test_gaps_find_missing_bars_only():
    m = 300_000
    assert bd.gaps([0, m, 3 * m], 5) == [(m, 3 * m)]
    assert bd.gaps([0, m], None) == []


def test_bad_rows_counts_broken_candle():
    good = [0, "1", "2", "0.5", "1.5"]
    broken = [0, "1", "0.9", "0.5", "1.5"]
    assert bd.bad_rows([good, broken]) == 1


def test_fill_recent_pages_until_short_page(monkeypatch):
    calls = []

    def fake(url):
        calls.append(url)
        n = 1000 if len(calls) == 1 else 2
        base = len(calls) * 10_000
        return str(
            [[base + i, 1, 2, 0.5, 1.5, 9, base + i + 1, 1, 1, 1, 1, 0] for i in range(n)]
        ).encode()

    monkeypatch.setattr(bd, "get", fake)
    rows = bd.fill_recent("5m", 5, 10**12)
    assert len(rows) == 1002 and "startTime=6" in calls[0]


def test_weekly_and_monthly_come_from_api_only(monkeypatch, tmp_path):
    day = 86_400_000
    row = [
        1_600_000_000_000,
        "1",
        "2",
        "0.5",
        "1.5",
        "9",
        1_600_000_000_000 + day,
        "1",
        1,
        "1",
        "1",
    ]
    since = []

    def api(iv, after, now):
        since.append((iv, after))
        return [row]

    def no_files(job):
        raise AssertionError("월별 파일을 읽으면 안 된다")

    monkeypatch.setattr(bd, "fill_recent", api)
    monkeypatch.setattr(bd, "month_file", no_files)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(1) as pool:
        got = bd.collect("1M", tmp_path, 1_700_000_000_000, pool)
    assert got["rows"] == 1 and since == [("1M", 1504224000000 - 1)]
