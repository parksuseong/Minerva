from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from minerva import routes
from minerva.main import app
from minerva.strategy import scr_lab

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "btc_15m_scr.json"
CORE_JS = ROOT / "src" / "minerva" / "static" / "scrlab_core.js"
KST = timezone(timedelta(hours=9))
STEP = 15 * 60_000


def bars() -> list[list]:
    return json.loads(FIXTURE.read_text())["bars"]


def hlc(rows):
    return [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows]


def close_kst(open_ms: int) -> str:
    return datetime.fromtimestamp((open_ms + STEP) / 1000, KST).strftime("%m-%d %H:%M")


def test_default_reproduces_last_seven_days_of_15m_signals():
    rows = bars()
    res = scr_lab.signals(*hlc(rows))
    # 픽스처 끝(11/8 20:00 KST) 기준 최근 7일
    cut = rows[-1][0] + STEP - 7 * 86_400_000
    got = sorted(
        (close_kst(rows[i][0]), side, rows[i][4])
        for side in ("long", "short")
        for i in res[side]
        if rows[i][0] >= cut
    )
    assert got == [
        ("11-03 03:15", "long", 77216.0),
        ("11-03 23:15", "short", 77229.8),
        ("11-04 07:15", "short", 77792.6),
        ("11-04 13:00", "long", 76465.7),
        ("11-04 16:45", "short", 78150.2),
        ("11-04 18:15", "short", 78414.6),
    ]


def test_cross_needs_previous_at_or_below_line():
    assert scr_lab.crossed([None, 10, 20, 25, 20, 19], 20, True) == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert scr_lab.crossed([90, 80, 79], 80, False) == [False, False, True]


def test_recent_window_includes_current_bar():
    assert scr_lab.recent([True, False, False, False], 3) == [True, True, True, False]
    assert scr_lab.recent([False, True], 1) == [False, True]


def test_signal_marked_once_while_condition_holds():
    rows = bars()
    res = scr_lab.signals(*hlc(rows), {"w": 6})
    for side in ("long", "short"):
        idx = res[side]
        assert all(b - a > 1 for a, b in zip(idx, idx[1:]))


def test_looser_settings_give_more_signals():
    h, lo, c = hlc(bars())
    strict = scr_lab.signals(h, lo, c)
    two = scr_lab.signals(h, lo, c, {"need": 2})
    zone = scr_lab.signals(h, lo, c, {"mode": "zone"})
    assert len(two["long"]) > len(strict["long"])
    assert len(zone["long"]) > len(strict["long"])


def test_rsi_flat_and_short_input():
    assert scr_lab.rsi([1.0, 2.0], 14) == [None, None]
    assert scr_lab.rsi([float(i) for i in range(20)], 14)[-1] == 100.0
    assert scr_lab.stoch([1.0] * 5, [1.0] * 5, [1.0] * 5, 5)[-1] == 50.0
    assert scr_lab.cci([1.0] * 20, [1.0] * 20, [1.0] * 20, 20)[-1] == 0.0


def bun() -> str | None:
    found = shutil.which("bun")
    if found:
        return found
    home = Path(os.path.expanduser("~/.bun/bin/bun"))
    return str(home) if home.exists() else None


PARAM_SETS = [
    {},
    {"need": 2},
    {"mode": "zone", "w": 2},
    {"stoch_n": 9, "stoch_lo": 25, "stoch_hi": 75, "cci_n": 14, "rsi_n": 7, "w": 5},
    {"cci_lo": -50, "cci_hi": 50, "rsi_lo": 40, "rsi_hi": 60, "need": 2, "mode": "zone"},
]


@pytest.mark.skipif(bun() is None, reason="JS 실행기(bun)가 없다")
def test_js_core_matches_python_twin(tmp_path):
    script = tmp_path / "run.js"
    script.write_text(
        f"const core = require({json.dumps(str(CORE_JS))});\n"
        f"const d = require({json.dumps(str(FIXTURE))});\n"
        f"const sets = {json.dumps(PARAM_SETS)};\n"
        "const col = (k) => d.bars.map((b) => b[k]);\n"
        "const h = col(2), l = col(3), c = col(4);\n"
        "console.log(JSON.stringify(sets.map((p) => core.signals(h, l, c, p))));\n"
    )
    out = subprocess.run([bun(), str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    h, lo, c = hlc(bars())
    for params, js in zip(PARAM_SETS, got):
        py = scr_lab.signals(h, lo, c, params)
        assert js["long"] == py["long"], params
        assert js["short"] == py["short"], params
        for k in ("stoch", "cci", "rsi"):
            for a, b in zip(js[k], py[k]):
                assert (a is None) == (b is None)
                if a is not None:
                    assert a == pytest.approx(b, abs=1e-9)


@pytest.mark.skipif(bun() is None, reason="JS 실행기(bun)가 없다")
def test_js_after_is_signed_and_needs_whole_bars(tmp_path):
    script = tmp_path / "after.js"
    script.write_text(
        f"const core = require({json.dumps(str(CORE_JS))});\n"
        "const c = [100, 101, 99, 110];\n"
        "console.log(JSON.stringify([core.after(c, 0, 1, 1), core.after(c, 0, 2, -1),"
        " core.after(c, 0, 0.25, 1), core.after(c, 1, 5, 1)]));\n"
    )
    out = subprocess.run([bun(), str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    one, two, part, past = json.loads(out.stdout)
    assert one == pytest.approx(1.0)
    assert two == pytest.approx(1.0)
    assert part is None and past is None


# ---- /api/btc/klines ----


def kline(open_ms: int, step: int, px: float = 100.0) -> list:
    return [open_ms, str(px), str(px + 1), str(px - 1), str(px), "1", open_ms + step - 1]


@pytest.fixture
def fake_binance(monkeypatch):
    calls = []
    routes._kline_cache.clear()

    def fake(interval, start_ms, limit):
        calls.append((interval, start_ms, limit))
        step = routes.LAB_INTERVALS[interval] * 60_000
        now = fake.now
        first = start_ms - start_ms % step
        return [kline(t, step) for t in range(first, now + 1, step)][:limit]

    fake.now = 1_794_134_700_000 + 5 * 60_000  # 마지막 15분봉이 진행 중인 시각
    monkeypatch.setattr(routes, "_get_klines", fake)
    monkeypatch.setattr(routes, "_now_ms", lambda: fake.now)
    yield calls
    routes._kline_cache.clear()


def test_klines_drops_open_bar_and_adds_warmup(fake_binance):
    client = TestClient(app)
    d = client.get("/api/btc/klines?interval=15m&days=7").json()
    now = 1_794_134_700_000 + 5 * 60_000
    assert d["interval"] == "15m" and d["days"] == 7
    assert all(b[0] + STEP - 1 < now for b in d["bars"])
    assert d["bars"][-1][0] == 1_794_134_700_000 - STEP
    assert d["show_from"] == now - 7 * 86_400_000
    before = [b for b in d["bars"] if b[0] < d["show_from"]]
    assert len(before) >= routes.LAB_WARMUP - 1
    assert d["bars"][0][1:5] == [100.0, 101.0, 99.0, 100.0]


def test_klines_pages_through_long_requests(fake_binance):
    client = TestClient(app)
    d = client.get("/api/btc/klines?interval=5m&days=30").json()
    assert len(fake_binance) >= 6
    times = [b[0] for b in d["bars"]]
    assert times == sorted(set(times))


def test_klines_caches_same_request(fake_binance):
    client = TestClient(app)
    client.get("/api/btc/klines?interval=1h&days=3")
    n = len(fake_binance)
    client.get("/api/btc/klines?interval=1h&days=3")
    assert len(fake_binance) == n
    client.get("/api/btc/klines?interval=1h&days=7")
    assert len(fake_binance) > n


def test_klines_caps_days_and_rejects_other_intervals(fake_binance):
    client = TestClient(app)
    assert client.get("/api/btc/klines?interval=1d&days=999").json()["days"] == routes.LAB_MAX_DAYS
    assert client.get("/api/btc/klines?interval=1w").status_code == 400
    assert client.get("/api/btc/klines?interval=1M").status_code == 400


def test_klines_upstream_failure_is_502(monkeypatch):
    routes._kline_cache.clear()

    def boom(*_):
        raise OSError("down")

    monkeypatch.setattr(routes, "_get_klines", boom)
    with pytest.raises(HTTPException) as exc:
        routes.btc_klines("15m", 7)
    assert exc.value.status_code == 502


def test_page_has_lab_tab_and_scripts():
    client = TestClient(app)
    page = client.get("/").text
    assert 'data-tab="scrlab"' in page and 'id="tab-scrlab"' in page
    assert "/static/scrlab_core.js" in page and "/static/scrlab.js" in page
    assert "수수료·손절 없음" in page
    assert client.get("/static/scrlab.js").status_code == 200
