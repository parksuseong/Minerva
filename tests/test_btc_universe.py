"""종목 범위(거래량 상위 50)와 백테스트 포팅 시험. 네트워크 없음."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from minerva import btc_universe as bu
from minerva.kis import KST
from minerva.strategy import scr_lab

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "btc_15m_scr.json"
CORE_JS = ROOT / "src" / "minerva" / "static" / "scrlab_core.js"
VIEW_JS = ROOT / "src" / "minerva" / "static" / "scrview.js"
NOW = datetime(2026, 11, 12, 6, 5, tzinfo=KST)
NOW_MS = int(NOW.timestamp() * 1000)


def _bun():
    from test_scr_lab import bun

    return bun()


def bars() -> list[list]:
    return json.loads(FIXTURE.read_text())["bars"]


def write(path: Path, at: datetime = NOW, **over) -> None:
    data = {
        "updated_at": at.strftime("%Y-%m-%d %H:%M"),
        "rule": "x",
        "symbols": ["BTCUSDT", "DOGEUSDT"],
        "stats": {"BTCUSDT": {"winrate": 50}, "DOGEUSDT": {"winrate": 60}},
        **over,
    }
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------- 파일 읽기


def test_load_valid_missing_stale_corrupt(tmp_path):
    p = tmp_path / "universe.json"
    assert bu.load(p, NOW_MS) is None  # 없음
    write(p)
    assert bu.load(p, NOW_MS)["symbols"] == ["BTCUSDT", "DOGEUSDT"]
    write(p, NOW - timedelta(days=15))
    assert bu.load(p, NOW_MS) is None  # 14일 넘음
    write(p, NOW - timedelta(days=13))
    assert bu.load(p, NOW_MS) is not None
    p.write_text("{깨짐")
    assert bu.load(p, NOW_MS) is None
    for bad in ({"symbols": []}, {"symbols": ["BTC"]}, {"stats": []}, {"updated_at": "어제"}):
        write(p, **bad)
        assert bu.load(p, NOW_MS) is None, bad
    p.write_text("[1, 2]")
    assert bu.load(p, NOW_MS) is None


def test_priority_by_winrate_then_volume_low_sample_last():
    stats = {
        "A": {"rank": 1, "winrate": 50, "trades": 100, "days": 365},
        "B": {"rank": 2, "winrate": 60, "trades": 100, "days": 365},
        "C": {"rank": 3, "winrate": 60, "trades": 100, "days": 365},
        "D": {"rank": 4, "winrate": 90, "trades": 19, "days": 365},  # 거래 적음
        "E": {"rank": 5, "winrate": 95, "trades": 100, "days": 59},  # 기간 짧음
        "F": {"rank": 6, "winrate": 40, "trades": 20, "days": 60},  # 경계는 정상
    }
    assert bu.priority(list(stats), stats) == ["B", "C", "A", "F", "E", "D"]
    assert bu.low_sample({}) and bu.low_sample(None)


# ---------------------------------------------------------------- 상위 50 고르기


def test_select_top_filters_contract_status_quote():
    info = {
        "symbols": [
            {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT",
             "contractType": "PERPETUAL", "underlyingType": "COIN"},
            {"symbol": "NVDAUSDT", "status": "TRADING", "quoteAsset": "USDT",
             "contractType": "TRADIFI_PERPETUAL", "underlyingType": "EQUITY"},
            {"symbol": "BTCUSDT_261225", "status": "TRADING", "quoteAsset": "USDT",
             "contractType": "CURRENT_QUARTER"},
            {"symbol": "OLDUSDT", "status": "SETTLING", "quoteAsset": "USDT",
             "contractType": "PERPETUAL"},
            {"symbol": "ETHUSDC", "status": "TRADING", "quoteAsset": "USDC",
             "contractType": "PERPETUAL"},
        ]
    }  # fmt: skip
    tickers = [
        {"symbol": s, "quoteVolume": v}
        for s, v in (
            ("BTCUSDT", "100"),
            ("NVDAUSDT", "300"),
            ("BTCUSDT_261225", "999"),
            ("OLDUSDT", "999"),
            ("ETHUSDC", "999"),
        )
    ]
    top = bu.select_top(info, tickers)
    assert [r["symbol"] for r in top] == ["NVDAUSDT", "BTCUSDT"]
    assert top[0]["contract"] == "TRADIFI_PERPETUAL" and top[0]["underlying"] == "EQUITY"
    assert len(bu.select_top(info, tickers, 1)) == 1
    # 상위 밖 extra 종목은 뒤에 붙고, 거래 불가 종목은 extra여도 안 붙는다
    one = bu.select_top(info, tickers, 1, extra=("BTCUSDT", "OLDUSDT"))
    assert [r["symbol"] for r in one] == ["NVDAUSDT", "BTCUSDT"]


def test_refresh_writes_stats_one_symbol_at_a_time(tmp_path, monkeypatch):
    monkeypatch.setattr(bu, "_sleep", lambda s: None)
    rows = bars()
    t_end = rows[-1][0] + bu.BAR_MS
    info = {
        "symbols": [
            {"symbol": s, "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL"}
            for s in ("BTCUSDT", "NEWUSDT", "BADUSDT")
        ]
    }
    tickers = [
        {"symbol": "BTCUSDT", "quoteVolume": "3"},
        {"symbol": "NEWUSDT", "quoteVolume": "2"},
        {"symbol": "BADUSDT", "quoteVolume": "1"},
    ]
    asked = []

    def get(path, params=None):
        if path == "/fapi/v1/exchangeInfo":
            return info
        if path == "/fapi/v1/ticker/24hr":
            return tickers
        asked.append(params)
        if params["symbol"] == "BADUSDT":
            raise OSError("down")
        src = rows if params["symbol"] == "BTCUSDT" else rows[-200:]  # 새 종목: 봉이 짧다
        page = [r for r in src if r[0] >= params["startTime"]][: params["limit"]]
        return [[r[0], r[1], r[2], r[3], r[4], 1, r[0] + bu.BAR_MS - 1] for r in page]

    out = tmp_path / "u.json"
    data = bu.refresh(out, scr_lab.DEFAULTS, now_ms=t_end + 60_000, get=get)
    assert json.loads(out.read_text()) == data
    assert data["symbols"] == ["BTCUSDT", "NEWUSDT", "BADUSDT"]
    assert all(p["limit"] == bu.PAGE for p in asked)
    st = data["stats"]
    assert st["BTCUSDT"]["rank"] == 1 and st["BTCUSDT"]["trades"] > 0
    assert 0 <= st["BTCUSDT"]["winrate"] <= 100 and st["BTCUSDT"]["days"] > 0
    assert st["NEWUSDT"]["trades"] == 0  # 300봉 워밍업도 안 됨
    assert st["BADUSDT"]["trades"] == 0 and "OSError" in st["BADUSDT"]["error"]
    assert bu.load(out, t_end + 60_000) is not None
    assert "BTCUSDT" in bu.table(data)


def test_refresh_stops_on_rate_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(bu, "_sleep", lambda s: None)

    def get(path, params=None):
        if path == "/fapi/v1/exchangeInfo":
            return {"symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT",
                                 "contractType": "PERPETUAL"}]}  # fmt: skip
        if path == "/fapi/v1/ticker/24hr":
            return [{"symbol": "BTCUSDT", "quoteVolume": "1"}]
        raise bu.RateLimited("HTTP 429")

    out = tmp_path / "u.json"
    with pytest.raises(bu.RateLimited):
        bu.refresh(out, scr_lab.DEFAULTS, now_ms=NOW_MS, get=get)
    assert not out.exists()  # 반쪽 파일을 쓰지 않는다


def test_engine_window_avoids_timer_minutes():
    base = 1_794_000_000 - 1_794_000_000 % 900  # 15분 경계(:00, :15 …)
    assert bu._engine_window(base + 20)  # 엔진 실행 시각
    assert bu._engine_window(base - 30) and bu._engine_window(base + 75)
    assert not bu._engine_window(base + 120) and not bu._engine_window(base + 450)


# ---------------------------------------------------------------- switchSim 포팅


def test_switch_sim_hand_cases():
    # 롱 신호(0) → 1 시가 100 진입 → 2 저가 96.9 손절(97) → 쉼 → 숏 신호(3) → 4 시가 100 진입,
    # 끝까지 보유(종가 98로 정리)
    o = [100, 100, 99, 98, 100, 99]
    h = [100, 101, 99, 98, 101, 99]
    lo = [100, 99, 96.9, 97, 99, 98]
    c = [100, 100, 97.5, 98, 100, 98]
    r = bu.switch_sim(o, h, lo, c, [(0, 1), (3, -1)], stop=0.03, lev=3, fee=0)
    assert [x["kind"] for x in r["exits"]] == ["stop", "end"]
    assert r["exits"][0]["net"] == pytest.approx(-9.0)  # 3배 × −3%
    assert r["exits"][1]["net"] == pytest.approx(6.0)  # 숏 100→98, 3배
    assert (r["trades"], r["wins"], r["stops"]) == (2, 1, 1)
    assert r["ret"] == pytest.approx((0.91 * 1.06 - 1) * 100)
    # 같은 봉 롱·숏은 건너뛴다
    both = bu.switch_sim(o, h, lo, c, [(0, 1), (0, -1)], stop=0.03, lev=3, fee=0)
    assert both["trades"] == 0
    # 반대 신호에 뒤집기, 수수료는 편도 × 레버리지 × 2쪽
    flip = bu.switch_sim([1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 2, 2],
                         [(0, 1), (1, -1)], stop=0.5, lev=1, fee=0.001)  # fmt: skip
    assert flip["exits"][0]["kind"] == "flip"
    assert flip["exits"][0]["net"] == pytest.approx((2 / 1 - 1) * 100 - 0.1 * 3)


@pytest.mark.skipif(_bun() is None, reason="JS 실행기(bun)가 없다")
def test_switch_sim_matches_js_on_fixture(tmp_path):
    rows = bars()
    h, lo, c = [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows]
    o = [r[1] for r in rows]
    sig = scr_lab.signals(h, lo, c, {"need": 2})
    sigs = sorted([(i, 1) for i in sig["long"]] + [(i, -1) for i in sig["short"]],
                  key=lambda x: (x[0], -x[1]))  # fmt: skip
    cases = [{"stop": 0.03, "tp": 0}, {"stop": 0.01, "tp": 0.02}, {"stop": 0, "tp": 0}]
    script = tmp_path / "sim.js"
    script.write_text(
        f"const core = require({json.dumps(str(CORE_JS))});\n"
        f"const d = {json.dumps({'o': o, 'h': h, 'l': lo, 'c': c})};\n"
        f"const sigs = {json.dumps([{'i': i, 'dir': x} for i, x in sigs])};\n"
        f"const cases = {json.dumps(cases)};\n"
        "console.log(JSON.stringify(cases.map((k) => core.switchSim(d.o, d.h, d.l, d.c, sigs,"
        " { stop: k.stop, tp: k.tp, lev: 3, fee: 0.0005 }))));\n"
    )
    out = subprocess.run([_bun(), str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    for k, js in zip(cases, json.loads(out.stdout)):
        py = bu.switch_sim(o, h, lo, c, sigs, stop=k["stop"], tp=k["tp"], lev=3, fee=0.0005)
        assert js["trades"] > 5
        for key in ("trades", "wins", "stops", "tps", "liqs", "holding"):
            assert py[key] == js[key], (k, key)
        assert py["ret"] == pytest.approx(js["ret"], rel=1e-9, abs=1e-9)
        assert py["mdd"] == pytest.approx(js["mdd"], rel=1e-9, abs=1e-9)
        assert [(x["i"], x["kind"]) for x in py["exits"]] == [
            (x["i"], x["kind"]) for x in js["exits"]
        ]
        assert [x["net"] for x in py["exits"]] == pytest.approx([x["net"] for x in js["exits"]])


# ---------------------------------------------------------------- 신호까지 거리(linePrice)


def test_line_price_lands_indicator_on_line():
    rows = bars()[-400:]
    h, lo, c = [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows]
    p = scr_lab.DEFAULTS
    for name, line in (("RSI", 30.0), ("RSI", 70.0), ("CCI", 100.0), ("CCI", -100.0)):
        px = scr_lab.line_price(h, lo, c, name, line, p)
        if px is None:
            continue
        hh, ll, cc = h + [px], lo + [px], c + [px]
        v = scr_lab.rsi(cc, 21)[-1] if name == "RSI" else scr_lab.cci(hh, ll, cc, 20)[-1]
        assert v == pytest.approx(line, abs=1e-3)


@pytest.mark.skipif(_bun() is None, reason="JS 실행기(bun)가 없다")
def test_line_price_matches_scrview_js(tmp_path):
    rows = bars()
    p = scr_lab.DEFAULTS
    cuts = [len(rows) - k * 37 for k in range(12)]
    asks = [(cut, name, line) for cut in cuts for name, line in
            (("Stoch", 20.0), ("Stoch", 80.0), ("CCI", -100.0), ("CCI", 100.0),
             ("RSI", 30.0), ("RSI", 70.0))]  # fmt: skip
    script = tmp_path / "line.js"
    script.write_text(
        "globalThis.window = globalThis;\n"
        "window.MinervaUI = { esc: (s) => s, pad: (n) => n, kst: (t) => t, num: (x) => x };\n"
        f"require({json.dumps(str(VIEW_JS))});\n"
        f"const core = require({json.dumps(str(CORE_JS))});\n"
        f"const rows = {json.dumps([r[:5] for r in rows])};\n"
        f"const asks = {json.dumps(asks)};\n"
        f"const p = {json.dumps(p)};\n"
        "console.log(JSON.stringify(asks.map(([cut, name, line]) =>"
        " window.ScrView.linePrice(core, rows.slice(0, cut), name, line, p))));\n"
    )
    out = subprocess.run([_bun(), str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert sum(x is not None for x in got) >= 10
    for (cut, name, line), js in zip(asks, got):
        r = rows[:cut]
        py = scr_lab.line_price([x[2] for x in r], [x[3] for x in r], [x[4] for x in r],
                                name, line, p)  # fmt: skip
        assert (py is None) == (js is None), (cut, name, line)
        if py is not None:
            assert py == pytest.approx(js, rel=1e-9)


def test_near_reports_hits_and_distances():
    rows = bars()[-300:]
    h, lo, c = [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows]
    out = scr_lab.near(h, lo, c)
    for side in ("long", "short"):
        r = out[side]
        assert r["k"] == len(r["hit"]) and r["k"] + len(r["need"]) == 3
        for x in r["need"]:
            assert x.get("far") or "price" in x
    sig = scr_lab.signals(h, lo, c)
    if sig["long"] and sig["long"][-1] == len(c) - 1:
        assert out["long"]["fired"]
