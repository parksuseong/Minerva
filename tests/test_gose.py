"""고서 가상매매(minerva.gose)와 /api/gose. 네트워크 없음, 폴더는 conftest가 임시로 둔다."""

from __future__ import annotations

import json
import math
import random
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
    assert len(paper["accounts"]) == len(gose.ACCOUNTS) + len(gose.AVG_ACCOUNTS)
    a = next(x for x in paper["accounts"] if x["key"] == "1A")
    assert a["curve"][0][0] == bars[250][0] and a["n_closed"] > 0


def _vol_bars(n=400, like=None):
    """UVXY 대신 쓰는 합성 변동성 일봉(like 또는 _wave_bars와 같은 날짜, 더 짧은 주기)."""
    out = []
    for i, b in enumerate(like or _wave_bars(n)):
        c = 20 * (1 + 0.5 * math.sin(2 * math.pi * i / 33)) ** 2 + 5
        prev = out[-1][4] if out else c
        out.append([b[0], prev, max(prev, c) * 1.01, min(prev, c) * 0.99, c])
    return out


def _trend_bars(n=900, seed=4, drift=0.002):
    """오르는 무작위 걸음(고정 시드). 월봉 지표가 차도록 길게 — 3v2 추세 조건이 가끔 맞는다."""
    rng = random.Random(seed)
    out, d, c = [], date(2022, 1, 3), 100.0
    for _ in range(n):
        while d.weekday() >= 5:
            d += timedelta(days=1)
        prev = c
        c *= 1 + drift + rng.gauss(0, 0.02)
        out.append([d.isoformat(), prev, max(prev, c) * 1.01, min(prev, c) * 0.99, c])
        d += timedelta(days=1)
    return out


def test_v2_accounts_registered():
    keys = [k for k, _, _ in gose.ACCOUNTS]
    assert keys[-2:] == ["3v2-loose", "11v2a"] and len(keys) == len(set(keys))
    assert gose._rules("11v2a") is gose.V and gose._rules("3-c0") is gose.R
    assert gose._allowed("3v2-loose", "AAPL")  # 3v2는 전 종목
    assert gose._allowed("11v2a", "SOXL") and not gose._allowed("11v2a", "AAPL")
    assert not gose._allowed("11v2a", "UVXY")  # 신호원은 사지 않는다
    assert gose._allowed("1A", "AAPL")


def test_11v2a_uses_vol_signal_and_traded_symbols_only():
    bars, vol = _wave_bars(), [tuple(b) for b in _vol_bars()]
    hits = gose.V.compute_vol_hits(vol, ("11v2a",))
    assert hits["11v2a"]
    end = bars[-1][0]
    soxl = gose.collect("SOXL", bars, None, "0000", end, vol=vol, vol_hits=hits)["11v2a"]["trades"]
    assert soxl and all(t["reason_in"].startswith("UVXY") for t in soxl)
    # 신호일은 UVXY 신호 날짜(같은 달력이라 같은 날)
    sig = {d for _, d in hits["11v2a"]}
    assert {t["signal_date"] for t in soxl} <= sig
    # 허용 밖 종목은 같은 신호가 있어도 거래 없음, 3v2-loose는 그대로 계산
    aapl = gose.collect("AAPL", bars, None, "0000", end, vol=vol, vol_hits=hits)
    assert aapl["11v2a"] == {"trades": [], "pending": []}
    # 신호원이 없으면 11v2a는 거래 없음
    assert gose.collect("SOXL", bars, None, "0000", end)["11v2a"]["trades"] == []


def test_v2_accounts_respect_start():
    bars = _trend_bars()
    vol = [tuple(b) for b in _vol_bars(like=bars)]
    end = bars[-1][0]
    for key in ("3v2-loose", "11v2a"):
        full = gose.collect("SOXL", bars, None, "0000", end, vol=vol)[key]["trades"]
        assert len(full) >= 2, key
        start = sorted(t["signal_date"] for t in full)[1]  # 첫 거래는 시작 전
        cut = gose.collect("SOXL", bars, None, start, end, vol=vol)[key]["trades"]
        assert cut and all(t["signal_date"] >= start for t in cut)
        assert len(full) > len(cut), key
        assert [t for t in full if t["signal_date"] >= start] == cut


def test_build_paper_v2_accounts_with_vol_cache(gdir):
    bars = _wave_bars()
    for sym in ("SOXL", "AAPL"):
        gose._write(gdir / "bars" / f"{sym}.json", {"bars": bars})
    gose._write(gdir / "bars" / "UVXY.json", {"bars": _vol_bars()})
    paper = gose.build_paper(["SOXL", "AAPL"], "^VIX", bars[150][0], bars[-1][0])
    by = {a["key"]: a for a in paper["accounts"]}
    assert paper["symbols_used"] == 2  # UVXY는 종목 수에 안 들어간다
    a = by["11v2a"]
    assert a["round"] == 2 and a["label"] == gose.V2_LABEL and a["signal_source"] == "UVXY"
    assert a["symbols"] == ["SOXX", "QQQ", "SPY", "SOXL"]
    assert a["signal_source_through"] == bars[-1][0]
    traded = {c["sym"] for c in a["closed"]} | {p["sym"] for p in a["open"]}
    assert traded == {"SOXL"}
    assert by["3v2-loose"]["round"] == 2 and "signal_source" not in by["3v2-loose"]
    assert "round" not in by["1A"]
    # 신호원 캐시가 없어도 돌고, 11v2a만 비어 있다
    (gdir / "bars" / "UVXY.json").unlink()
    paper = gose.build_paper(["SOXL"], "^VIX", bars[150][0], bars[-1][0])
    a = next(x for x in paper["accounts"] if x["key"] == "11v2a")
    assert a["n_closed"] == 0 and not a["open"] and a["signal_source_through"] is None


def test_build_backtest_with_round2():
    r1 = {
        "meta": {"bonferroni_pct": 99.84},
        "variants": {"1A/N": {"main": {"n": 3}, "pass": False, "fail_reasons": ["거래<30"]}},
    }
    v_loose = {
        "main": {"n": 900, "mean": 1.5},
        "subs": {"2021-22": {"n": 275, "mean": -0.999}},
        "random": {"pct": 99.4},
        "pass": False,
        "fail_reasons": ["하위기간 평균≤0: 2021-22"],
    }
    v_a = {
        "main": {"n": 29, "mean": 7.5},
        "subs": {"2021-22": {"n": 0}},
        "per_sym_main": {
            "SOXX": {"n": 8, "mean": 5.2},
            "QQQ": {"n": 7, "mean": 2.1},
            "SPY": {"n": 7, "mean": 0.9},
            "SOXL": {"n": 7, "mean": 22.2},
        },
        "pass": False,
        "fail_reasons": ["거래<30"],
    }
    r2 = {
        "meta": {"bonferroni_pct": 99.89, "variants": 46},
        "variants": {"3v2-loose/N": v_loose, "11v2a/N": v_a},
    }
    out = gose.build_backtest(r1, r2)
    assert out["variants"] == 3 and out["verified"] == 0
    assert out["meta"]["bonferroni_pct"] == 99.89 and out["meta"]["tested"] == 46
    assert "46개" in out["gates"][-1]
    assert out["rounds"]["2"] == {
        "variants": 2,
        "verified": 0,
        "label": gose.V2_LABEL,
        "vol_series": None,
    }
    rows = {r["variant"]: r for r in out["rows"]}
    assert [r["variant"] for r in out["rows"]] == ["1A/N", "3v2-loose/N", "11v2a/N"]
    assert "round" not in rows["1A/N"]
    assert any("2021-22" in c and "손실" in c for c in rows["3v2-loose/N"]["caveats"])
    cav = rows["11v2a/N"]["caveats"]
    assert any("신호 0번" in c for c in cav) and any("SOXL 쏠림" in c for c in cav)
    # 2차가 없으면 예전과 같다
    assert gose.build_backtest(r1)["rounds"] is None


def test_build_backtest_counts_and_caveats():
    v = {
        "main": {"n": 40, "win": 55.0, "mean": 1.0},
        "port_main_1x": {"cagr": 10.0, "mdd": -20.0},
        "random": {"pct": 96.0},
        "src_syms_main": {"n": 5, "mean": -0.2},
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
    text += gose.AVG_SPEC.read_text() + gose.AVG_INFO.read_text()
    for bad in ("http", "".join(("b", "log")), "logNo", "2241", "2242", "2243"):
        assert bad not in text


def test_avg_accounts_one_symbol_each_and_own_sizing(gdir):
    bars = _trend_bars(n=700, seed=7, drift=0.0)
    for sym in ("SOXL", "SOXX", "W"):
        gose._write(gdir / "bars" / f"{sym}.json", {"bars": bars})
    start = bars[300][0]
    paper = gose.build_paper(["W"], "^VIX", start, bars[-1][0])
    keys = [a["key"] for a in paper["accounts"]]
    assert keys[-2:] == ["avg-SOXL", "avg-SOXX"]
    a = paper["accounts"][-2]
    assert a["strategy"] == "avg" and a["symbols"] == ["SOXL"] and a["curve"][0][0] == start
    assert a["n_closed"] > 0 and a["win_rate"] == 100.0
    for c in a["closed"]:
        assert c["exit_date"] >= c["entry_date"] >= start and 0 <= c["adds"] <= 2
        assert abs(c["exit_px"] / c["entry_px"] - 1.10) < 1e-3 or c["exit_px"] > c["entry_px"] * 1.1
    for p in a["open"]:
        assert p["tp_px"] == round(p["avg"] * 1.1, 4) or abs(p["tp_px"] - p["avg"] * 1.1) < 1e-3
        for k in ("adds", "invested", "avg", "mae_now_pct", "mae_worst_pct", "days_held"):
            assert k in p
        # 첫 체결 = 시작 자산/3 (사이클 시작 때 현금)
        assert p["fills"][0][3] <= a["equity"]
    # 오늘 신호는 SIG일 때만, 종목도 그 계좌 것만
    assert all(s["sym"] == "SOXL" for s in a["signals"])
    assert a["reference"]["from"] == gose.AVG_REF_START


def test_avg_account_without_bars(gdir):
    a = gose.build_avg_account("avg-SOXL", "SOXL", None, "2026-11-16", "2026-11-17")
    assert a["data"] is False and a["equity"] == gose.CAPITAL and a["open"] == []


def test_route_returns_spec_markdown(gdir):
    d = TestClient(app).get("/api/gose").json()
    s = d["spec"]
    assert s["status"] == "ok" and s["accounts"] == ["avg-SOXL", "avg-SOXX"]
    assert "SIG_t" in s["markdown"] and "평단 × 1.10" in s["markdown"]
    assert s["params"]["max_adds"] == 2 and s["params"]["stop"] is None
