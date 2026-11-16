"""고서 가상매매(minerva.gose)와 /api/gose. 네트워크 없음, 폴더는 conftest가 임시로 둔다."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from minerva import gose
from minerva.main import app


@pytest.fixture
def gdir(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_GOSE_DIR", str(tmp_path / "gose"))
    return tmp_path / "gose"


def _chart(days, quotes, reg=None, off=0):
    ts = [int(datetime(d.year, d.month, d.day, 14, tzinfo=timezone.utc).timestamp()) for d in days]
    meta = {"gmtoffset": off}
    if reg:
        meta["currentTradingPeriod"] = {"regular": reg}
    q = {k: [x[i] for x in quotes] for i, k in enumerate(("open", "high", "low", "close"))}
    return {"chart": {"result": [{"meta": meta, "timestamp": ts, "indicators": {"quote": [q]}}]}}


def test_parse_chart_skips_bad_rows_and_unfinished_bar():
    days = [date(2026, 11, 12), date(2026, 11, 13), date(2026, 11, 16)]
    quotes = [(10, 11, 9, 10.5), (None, 11, 9, 10), (10, 12, 9, 11)]
    start = datetime(2026, 11, 16, 14, 30, tzinfo=timezone.utc).timestamp()
    reg = {"start": start, "end": start + 6.5 * 3600, "gmtoffset": 0}
    raw = _chart(days, quotes, reg)
    # 장중: 오늘 봉은 버린다
    assert [b[0] for b in gose.parse_chart(raw, start + 3600)] == ["2026-11-12"]
    # 장 마감 뒤: 남긴다. 고가·저가는 시가·종가를 감싼다
    bars = gose.parse_chart(raw, start + 7 * 3600)
    assert [b[0] for b in bars] == ["2026-11-12", "2026-11-16"]
    assert bars[-1] == ["2026-11-16", 10.0, 12.0, 9.0, 11.0]


def test_update_bars_incremental_and_refetch_on_adjust(gdir, monkeypatch):
    days = [date(2026, 11, 2) + timedelta(days=i) for i in range(5)]
    full = _chart(days, [(10, 11, 9, 10 + i) for i in range(5)])
    calls = []

    def fake(ticker, p1, p2):
        calls.append(p1)
        return replies.pop(0)

    monkeypatch.setattr(gose, "fetch_chart", fake)
    now = datetime(2026, 11, 10, tzinfo=timezone.utc).timestamp()
    replies = [full]
    bars, mode = gose.update_bars("AAA", now)
    assert mode == "full" and len(bars) == 5 and calls[0] == gose.HISTORY_START
    # 겹치는 날이 같으면 이어 붙인다
    more = days[3:] + [date(2026, 11, 9)]
    replies = [_chart(more, [(10, 11, 9, 13), (10, 11, 9, 14), (10, 11, 9, 15)])]
    bars, mode = gose.update_bars("AAA", now)
    assert mode == "incremental" and [b[4] for b in bars] == [10, 11, 12, 13, 14, 15]
    # 분할로 옛 종가가 바뀌면 전부 다시 받는다
    replies = [
        _chart(more, [(5, 6, 4, 6.5), (5, 6, 4, 7), (5, 6, 4, 7.5)]),
        _chart(days, [(5, 6, 4, 5 + i / 2) for i in range(5)]),
    ]
    bars, mode = gose.update_bars("AAA", now)
    assert mode == "full" and calls[-1] == gose.HISTORY_START
    assert gose.load_bars("AAA") == bars


def test_state_start_is_never_rewritten(gdir):
    assert gose.load_state("2026-11-16")["start"] == "2026-11-16"
    assert gose.load_state("2026-11-20")["start"] == "2026-11-16"
    (gdir / "state.json").write_text("{broken")
    with pytest.raises(SystemExit):
        gose.load_state("2026-11-21")


def _trade(sym, d_in, px_in, d_out=None, px_out=None, side="long", stop=False, fills=None):
    return {
        "sym": sym,
        "side": side,
        "signal_date": d_in,
        "fills": fills or [[d_in, px_in, 1.0, "in"]],
        "exit_date": d_out,
        "exit_px": px_out,
        "reason_in": "in",
        "reason_out": "out" if d_out else None,
        "stop_exit": stop,
    }


def test_account_fees_and_sizing():
    cal = ["d1", "d2", "d3"]
    closes = {"A": {"d1": 100.0, "d2": 110.0, "d3": 120.0}}
    acct = gose.run_account([_trade("A", "d1", 100.0, "d3", 120.0)], closes, cal)
    shares = 1000 / (100 * (1 + gose.FEE))
    pnl = shares * 120 * (1 - gose.FEE) - 1000
    assert acct["equity"] == pytest.approx(10_000 + pnl, abs=0.01)
    assert acct["curve"][1][1] == pytest.approx(9_000 + shares * 110, abs=0.01)
    assert acct["n_closed"] == 1 and acct["win_rate"] == 100.0 and not acct["open"]
    assert acct["closed"][0]["ret_pct"] == pytest.approx(pnl / 10, abs=0.01)


def test_account_slots_full_skips_by_symbol_order():
    syms = [f"S{i:02d}" for i in range(12)]
    closes = {s: {"d1": 10.0, "d2": 10.0} for s in syms}
    trades = [_trade(s, "d1", 10.0) for s in reversed(syms)]
    acct = gose.run_account(trades, closes, ["d1", "d2"])
    assert [p["sym"] for p in acct["open"]] == syms[:10]
    assert acct["skipped"] == 2
    # 같은 날 시가 청산이 자리를 먼저 비운다
    trades = [_trade(s, "d1", 10.0, "d2", 10.0) for s in syms[:10]]
    trades.append(_trade("Z", "d2", 10.0))
    acct = gose.run_account(trades, closes | {"Z": {"d2": 10.0}}, ["d1", "d2"])
    assert [p["sym"] for p in acct["open"]] == ["Z"] and acct["skipped"] == 0


def test_account_stop_exit_after_open_entry_and_short():
    closes = {"A": {"d1": 10.0, "d2": 9.0}, "B": {"d2": 50.0}}
    stop = _trade("A", "d1", 10.0, "d2", 9.0, stop=True)
    short = _trade("B", "d2", 50.0, side="short")
    acct = gose.run_account([stop, short], closes, ["d1", "d2"])
    assert acct["n_closed"] == 1 and acct["win_rate"] == 0.0
    o = acct["open"][0]
    assert o["sym"] == "B" and o["value"] == pytest.approx(o["invested"] / (1 + gose.FEE))


def test_account_scale_in_uses_entry_budget():
    fills = [["d1", 10.0, 0.25, "1"], ["d2", 8.0, 0.25, "2"]]
    closes = {"A": {"d1": 10.0, "d2": 8.0}}
    acct = gose.run_account([_trade("A", "d1", 10.0, fills=fills)], closes, ["d1", "d2"])
    assert acct["open"][0]["invested"] == pytest.approx(500.0)
    assert acct["open"][0]["entry_px"] == pytest.approx(9.0)


def _wave_bars(n=400):
    out, d = [], date(2024, 1, 1)
    prev = 100.0
    for i in range(n):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        c = 100 * (1 + 0.15 * math.sin(2 * math.pi * i / 40))
        out.append([d.isoformat(), prev, max(prev, c) * 1.01, min(prev, c) * 0.99, c])
        prev = c
        d += timedelta(days=1)
    return out


def test_collect_keeps_only_signals_from_start():
    bars = _wave_bars()
    start = bars[250][0]
    got = gose.collect("W", bars, None, start, bars[-1][0])
    rows = got["1A"]["trades"]
    assert rows and all(t["signal_date"] >= start for t in rows)
    full = gose.collect("W", bars, None, "0000", bars[-1][0])["1A"]["trades"]
    assert len(full) > len(rows)
    assert set(got) == {k for k, _, _ in gose.ACCOUNTS} and "3-c0/S10" in got


def test_build_paper_and_run_without_fetch(gdir):
    bars = _wave_bars()
    gose._write(gdir / "bars" / "W.json", {"bars": bars})
    gose._write(gdir / "bars" / "SHORT.json", {"bars": bars[:10]})
    paper = gose.build_paper(["W", "SHORT", "NONE"], "^VIX", bars[250][0], bars[-1][0])
    assert paper["virtual"] and paper["symbols_used"] == 1
    assert paper["symbols_short"] == ["SHORT", "NONE"]
    assert len(paper["accounts"]) == len(gose.ACCOUNTS)
    a = next(x for x in paper["accounts"] if x["key"] == "1A")
    assert a["curve"][0][0] == bars[250][0] and a["n_closed"] > 0


def test_build_backtest_counts_and_caveats():
    v = {
        "main": {"n": 40, "win": 55.0, "mean": 1.0},
        "port_main_1x": {"cagr": 10.0, "mdd": -20.0},
        "random": {"pct": 96.0},
        "feed_syms_main": {"n": 5, "mean": -0.2},
        "bh_same_syms_main": {"cagr": 12.0, "mdd": -30.0},
        "pass": True,
        "fail_reasons": [],
        "bonferroni_pass": False,
    }
    bad = {"main": {"n": 3}, "pass": False, "fail_reasons": ["거래<30"], "random": None}
    out = gose.build_backtest(
        {"meta": {"bonferroni_pct": 99.84}, "variants": {"3-c0/S10": v, "1A/N": bad}}
    )
    assert out["verified"] == 0 and out["basic_pass"] == 1 and out["variants"] == 2
    row = next(r for r in out["rows"] if r["variant"] == "3-c0/S10")
    assert row["verdict"].startswith("근접") and row["caveats"]
    assert next(r for r in out["rows"] if r["variant"] == "1A/N")["reason"] == "거래<30"


def test_route_pending_when_empty(gdir):
    r = TestClient(app).get("/api/gose")
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "pending" and d["paper"] == {"status": "pending"}
    assert d["rulebook"]["label"] == "원문 규칙, 미검증"


def test_route_ok_and_broken_file(gdir):
    gdir.mkdir(parents=True)
    (gdir / "backtest.json").write_text(json.dumps({"status": "ok", "rows": []}))
    (gdir / "paper.json").write_text("{half")
    d = TestClient(app).get("/api/gose").json()
    assert d["status"] == "pending" and d["backtest"]["status"] == "ok"
    (gdir / "paper.json").write_text(json.dumps({"status": "ok", "accounts": []}))
    assert TestClient(app).get("/api/gose").json()["status"] == "ok"


def test_rulebook_has_no_feed_identifiers():
    text = gose.RULEBOOK.read_text() + gose.UNIVERSE_SEED.read_text()
    for bad in ("http", "feed", "logNo", "2241", "2242", "2243"):
        assert bad not in text
