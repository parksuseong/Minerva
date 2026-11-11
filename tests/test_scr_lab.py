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
    # 처음 정한 규칙(Stoch 5, RSI 14). 기본값은 2026-11-08에 Stoch 9·RSI 21로 바뀌었다.
    res = scr_lab.signals(*hlc(rows), {"stoch_n": 5, "rsi_n": 14})
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


def test_defaults_are_user_values():
    d = scr_lab.DEFAULTS
    assert (d["stoch_n"], d["stoch_lo"], d["stoch_hi"]) == (9, 20, 80)
    assert (d["cci_n"], d["cci_lo"], d["cci_hi"]) == (20, -100, 100)
    assert (d["rsi_n"], d["rsi_lo"], d["rsi_hi"]) == (21, 30, 70)
    assert (d["w"], d["need"], d["mode"]) == (3, 3, "cross")
    js = CORE_JS.read_text()
    assert "stoch_n: 9," in js and "rsi_n: 21," in js


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
def test_js_switch_sim_stop_flip_and_fees(tmp_path):
    script = tmp_path / "sim.js"
    # 봉: 0 롱 신호 → 1 시가 100 진입 → 2 고가 110 → 3 숏 신호(종가 108) → 4 시가 108 뒤집기
    #     → 5 고가 111.3 (108×1.03=111.24) 숏 손절
    script.write_text(
        f"const core = require({json.dumps(str(CORE_JS))});\n"
        "const o = [100, 100, 105, 109, 108, 109], h = [100, 101, 110, 109, 108, 111.3];\n"
        "const l = [100, 99, 104, 107, 107, 108], c = [100, 100, 109, 108, 108, 110];\n"
        "const sigs = [{ i: 0, dir: 1 }, { i: 3, dir: -1 }];\n"
        "const a = core.switchSim(o, h, l, c, sigs, { stop: 0.03, lev: 3, fee: 0.0005 });\n"
        "const b = core.switchSim(o, h, l, c, sigs, { stop: 0.03, lev: 1, fee: 0 });\n"
        "console.log(JSON.stringify([a, b]));\n"
    )
    out = subprocess.run([bun(), str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    a, b = json.loads(out.stdout)
    assert (b["trades"], b["wins"], b["stops"], b["holding"]) == (2, 1, 1, False)
    assert b["ret"] == pytest.approx((1.08 * 0.97 - 1) * 100)
    long_net = 3 * 0.08 - 3 * 0.0005 * (1 + 1.08)
    short_net = 3 * -0.03 - 3 * 0.0005 * (1 + 1.03)
    assert a["ret"] == pytest.approx(((1 + long_net) * (1 + short_net) - 1) * 100)
    assert a["mdd"] > 0


@pytest.mark.skipif(bun() is None, reason="JS 실행기(bun)가 없다")
def test_js_switch_sim_take_profit_and_open_lines(tmp_path):
    script = tmp_path / "tp.js"
    # 봉: 0 롱 신호 → 1 시가 100 진입 → 2 고가 105 (익절 4% = 104) → 3 숏 신호 → 4 시가 100 숏 진입
    #     → 5 고가 103.5, 저가 99 (손절 3% 103, 익절 4% 96 안 닿음) 손절
    #     두 번째 계산: 같은 봉들인데 마지막 숏을 열어 둔 채 끝나 열린 선을 돌려준다
    script.write_text(
        f"const core = require({json.dumps(str(CORE_JS))});\n"
        "const o = [100, 100, 103, 101, 100, 101], h = [100, 101, 105, 101, 100, 103.5];\n"
        "const l = [100, 99, 102, 100, 99, 99], c = [100, 100, 103, 100, 100, 102];\n"
        "const sigs = [{ i: 0, dir: 1 }, { i: 3, dir: -1 }];\n"
        "const opt = { stop: 0.03, tp: 0.04, lev: 1, fee: 0 };\n"
        "const a = core.switchSim(o, h, l, c, sigs, opt);\n"
        "const k = (x) => x.slice(0, 5);\n"
        "const b = core.switchSim(k(o), k(h), k(l), k(c), sigs, opt);\n"
        "const n = core.switchSim(o, h, l, c, sigs, { stop: 0.03, lev: 1, fee: 0 });\n"
        "console.log(JSON.stringify([a, b, n]));\n"
    )
    out = subprocess.run([bun(), str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    a, b, n = json.loads(out.stdout)
    assert (a["trades"], a["tps"], a["stops"], a["holding"]) == (2, 1, 1, False)
    assert [x["kind"] for x in a["exits"]] == ["tp", "stop"]
    assert [x["from"] for x in a["exits"]] == [1, 4]
    assert a["exits"][0]["px"] == pytest.approx(104)
    assert a["ret"] == pytest.approx((1.04 * 0.97 - 1) * 100)
    assert b["open"]["dir"] == -1 and b["open"]["entry"] == 100
    assert b["open"]["stop"] == pytest.approx(103) and b["open"]["tp"] == pytest.approx(96)
    # 익절 없음: 롱은 숏 신호에서 뒤집기로 닫힌다
    assert n["tps"] == 0 and n["exits"][0]["kind"] == "flip" and n["open"] is None


@pytest.mark.skipif(bun() is None, reason="JS 실행기(bun)가 없다")
def test_js_switch_sim_without_stop_rides_until_liquidation(tmp_path):
    script = tmp_path / "nostop.js"
    # 봉: 0 롱 신호 → 1 시가 100 진입 → 2 저가 90 (손절 3%면 걸림, 끄면 버팀)
    #     → 3 저가 66 (3배 청산가 66.67)
    script.write_text(
        f"const core = require({json.dumps(str(CORE_JS))});\n"
        "const o = [100, 100, 95, 80], h = [100, 101, 96, 81], l = [100, 99, 90, 66];\n"
        "const c = [100, 100, 92, 70];\n"
        "const sigs = [{ i: 0, dir: 1 }];\n"
        "const off = core.switchSim(o, h, l, c, sigs, { stop: 0, lev: 3, fee: 0 });\n"
        "const on = core.switchSim(o, h, l, c, sigs, { stop: 0.03, lev: 3, fee: 0 });\n"
        "const k = (x) => x.slice(0, 3);\n"
        "const held = core.switchSim(k(o), k(h), k(l), k(c), sigs, { stop: 0, lev: 3, fee: 0 });\n"
        "console.log(JSON.stringify([off, on, held]));\n"
    )
    out = subprocess.run([bun(), str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    off, on, held = json.loads(out.stdout)
    assert (off["stops"], off["liqs"], off["exits"][0]["kind"]) == (0, 1, "liq")
    assert off["exits"][0]["i"] == 3 and off["ret"] == pytest.approx(-100, abs=1e-6)
    assert (on["stops"], on["liqs"], on["exits"][0]["i"]) == (1, 0, 2)
    assert held["open"]["stop"] is None and held["open"]["liq"] == pytest.approx(100 * (1 - 1 / 3))


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
    class Calls(list):
        fake = None

    calls = Calls()
    routes._kline_cache.clear()

    def fake(interval, start_ms, limit, symbol="BTCUSDT"):
        calls.append((interval, start_ms, limit) if symbol == "BTCUSDT" else (symbol, interval))
        step = routes.LAB_INTERVALS[interval] * 60_000
        now = fake.now
        first = start_ms - start_ms % step
        return [kline(t, step) for t in range(first, min(now + 1, first + limit * step), step)]

    fake.now = 1_794_134_700_000 + 5 * 60_000  # 마지막 15분봉이 진행 중인 시각
    fake.mono = 1000.0
    monkeypatch.setattr(routes, "_get_klines", fake)
    monkeypatch.setattr(routes, "_now_ms", lambda: fake.now)
    monkeypatch.setattr(routes, "_mono", lambda: fake.mono)
    # 실제 엔진 종목 범위 파일과 무관하게(시험마다 네 종목만)
    monkeypatch.setattr(routes, "_universe_symbols", lambda: ())
    routes._uni_cache.clear()
    calls.fake = fake
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


def test_klines_refreshes_only_the_last_chunk(fake_binance):
    client = TestClient(app)
    client.get("/api/btc/klines?interval=15m&days=365")
    first = len(fake_binance)
    assert first >= 24  # 15분봉 1년 ≈ 3만 5천 봉 ÷ 1500
    fake_binance.fake.mono += routes.LAB_TTL + 1
    client.get("/api/btc/klines?interval=15m&days=365")
    assert len(fake_binance) == first + 1
    fake_binance.fake.mono += routes.LAB_OLD_TTL + 1
    client.get("/api/btc/klines?interval=15m&days=365")
    assert len(fake_binance) == 2 * first + 1


def test_klines_one_year_of_15m_bars(fake_binance):
    client = TestClient(app)
    d = client.get("/api/btc/klines?interval=15m&days=365").json()
    assert d["days"] == 365 and d["capped"] is False
    assert len(d["bars"]) == 365 * 96 + routes.LAB_WARMUP - 1
    times = [b[0] for b in d["bars"]]
    assert all(b - a == STEP for a, b in zip(times, times[1:]))


def test_klines_caps_days_and_rejects_other_intervals(fake_binance):
    client = TestClient(app)
    d = client.get("/api/btc/klines?interval=1d&days=999").json()
    assert d["days"] == routes.LAB_MAX_DAYS == 365 and d["capped"] is True
    d = client.get("/api/btc/klines?interval=5m&days=365").json()
    assert d["days"] == 90 and d["capped"] is True
    assert client.get("/api/btc/klines?interval=1w").status_code == 400
    assert client.get("/api/btc/klines?interval=1M").status_code == 400


def test_klines_gzip_when_asked(fake_binance):
    client = TestClient(app)
    r = client.get("/api/btc/klines?interval=4h&days=30", headers={"Accept-Encoding": "gzip"})
    assert r.headers["content-encoding"] == "gzip"
    assert r.json()["interval"] == "4h"


def test_cache_is_bounded(fake_binance):
    client = TestClient(app)
    for iv in ("5m", "15m", "30m", "1h", "4h", "1d"):
        client.get(f"/api/btc/klines?interval={iv}&days=365")
    assert len(routes._kline_cache) <= routes.LAB_MAX_CHUNKS


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
    assert 'data-days="365"' in page and 'data-days="180"' in page
    assert client.get("/static/scrlab.js").status_code == 200


def test_klines_other_symbol_is_whitelisted_and_cached_apart(fake_binance):
    client = TestClient(app)
    assert client.get("/api/btc/klines?symbol=DOGEUSDT").status_code == 400
    d = client.get("/api/btc/klines?symbol=SOXLUSDT&interval=15m&days=3").json()
    assert d["symbol"] == "SOXLUSDT" and d["bars"]
    assert any(c[0] == "SOXLUSDT" for c in fake_binance)
    assert any(k[0] == "SOXLUSDT" for k in routes._kline_cache)
    # 비트코인 덩어리 키는 예전 모양 그대로
    client.get("/api/btc/klines?interval=15m&days=3")
    assert any(len(k) == 2 for k in routes._kline_cache)


def test_klines_whitelist_adds_engine_universe(fake_binance, monkeypatch):
    client = TestClient(app)
    assert client.get("/api/btc/klines?symbol=DOGEUSDT").status_code == 400
    monkeypatch.setattr(routes, "_universe_symbols", lambda: ("DOGEUSDT",))
    assert client.get("/api/btc/klines?symbol=DOGEUSDT").status_code == 400  # 1분 캐시
    fake_binance.fake.mono += routes.UNIVERSE_TTL + 1
    assert client.get("/api/btc/klines?symbol=DOGEUSDT").status_code == 200
    assert client.get("/api/btc/klines?symbol=PEPEUSDT").status_code == 400

    def broken():
        raise ValueError("깨진 파일")

    monkeypatch.setattr(routes, "_universe_symbols", broken)
    fake_binance.fake.mono += routes.UNIVERSE_TTL + 1
    assert client.get("/api/btc/klines?symbol=ETHUSDT").status_code == 200
