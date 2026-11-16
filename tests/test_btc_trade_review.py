import json
import os
import subprocess
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from minerva import btc_review
from minerva import btc_trade as bt
from minerva import btc_trade_review as tr
from minerva.kis import KST
from minerva.main import app

M = 15 * 60_000
NOW = datetime(2026, 11, 16, 22, 0, tzinfo=KST)
NEXT = datetime(2026, 11, 17, 9, 20, tzinfo=KST)  # 다음 날 아침 복기


def ms(s: str) -> int:
    return btc_review._ms(s)


def bar(t: str, o, h, lo, c):
    return [ms(t), str(o), str(h), str(lo), str(c), "0"]


def seq(start: str, rows):
    t = ms(start)
    return [[t + i * M, str(o), str(h), str(lo), str(c), "0"]
            for i, (o, h, lo, c) in enumerate(rows)]


NVDA = {"at": "2026-11-16 20:10:20", "mode": "live", "symbol": "NVDAUSDT", "side": "short",
        "qty": 2, "entry": 100.0, "exit": 97.0, "pnl": 6.0, "pct": 3.0, "why": "tp",
        "opened_at": "2026-11-16 17:15:20", "manual": True, "tp_custom": True, "tp": 97.0,
        "stop": 103.0, "trail": False, "best": 97.5}
BTC = {"at": "2026-11-16 10:30:20", "mode": "live", "symbol": "BTCUSDT", "side": "long",
       "qty": 0.03, "entry": 100.0, "exit": 97.0, "pnl": -9.0, "pct": -3.0, "why": "stop",
       "opened_at": "2026-11-16 10:00:20"}


@pytest.fixture
def files(tmp_path, monkeypatch):
    d = tmp_path / "bt"
    d.mkdir()
    monkeypatch.setattr(bt, "DIR", d)
    cap = tmp_path / "binance_capital.json"
    cap.write_text(json.dumps({"live": True, "since_at": "2026-11-16 00:00", "base": 100}))
    monkeypatch.setattr(bt, "CAPITAL_FILE", cap)
    monkeypatch.setenv("MINERVA_ACCOUNT_OUT", str(tmp_path / "acct" / "latest.json"))
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN_FILE", str(tmp_path / "none"))
    monkeypatch.delenv("MINERVA_ACCOUNT_TOKEN", raising=False)
    closed = [
        BTC,
        NVDA,
        {**BTC, "mode": "dry", "at": "2026-11-16 11:30:20"},
        {**BTC, "at": "2026-11-15 10:30:20", "opened_at": "2026-11-15 10:00:20"},  # 실전 전
        # 복기 날(11-17) 닫힌 거래: 다음 날까지 기다린다
        {**BTC, "at": "2026-11-17 08:30:20", "opened_at": "2026-11-17 08:00:20"},
        {"at": "2026-11-16 12:00:00", "mode": "live", "pnl": None},  # opened_at 없음
    ]
    (d / "closed.jsonl").write_text("\n".join(json.dumps(c) for c in closed) + "\n{broken\n")
    events = [
        {"at": "2026-11-16 10:00:20", "mode": "live", "event": "run",
         "decision": "BTC 롱 진입"},
        {"at": "2026-11-16 10:00:20", "mode": "live", "event": "ai_choose",
         "choice": "a", "ai_raw": '{"reason": "승률 높음"}'},
        {"at": "2026-11-16 17:15:20", "mode": "live", "event": "adopt_manual",
         "symbol": "NVDAUSDT", "side": "short", "why": "손으로 연 포지션"},
        {"at": "2026-11-16 18:00:20", "mode": "live", "event": "tp_custom_kept",
         "symbol": "NVDAUSDT", "tp": 97.0},
        {"at": "2026-11-16 19:00:20", "mode": "live", "event": "adopt_manual",
         "symbol": "NVDAUSDT", "why": "수량 변경"},
        {"at": "2026-11-16 19:00:20", "mode": "live", "event": "trail_on", "symbol": "BTCUSDT"},
    ]
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return d


def flat_bars(sym, a, b):
    # 진입가 근처에서 거의 안 움직이는 봉(가정 결과는 결론 안 남)
    out, t = [], a - a % M
    while t <= b:
        out.append([t, "100", "100.5", "99.5", "100", "0"])
        t += M
    return out


def test_trade_id_stable_and_safe():
    assert tr.trade_id(NVDA) == "20261116-1715_NVDAUSDT_20261116-2010"
    assert tr.trade_id({**NVDA, "symbol": "a/b..USDT"}) == "20261116-1715_ABUSDT_20261116-2010"
    assert tr.ID.match(tr.trade_id(BTC))


def test_due_waits_skips_existing_and_limits(files):
    rows = tr.live_rows()
    assert sorted(r["at"] for r in rows) == [
        "2026-11-16 10:30:20", "2026-11-16 20:10:20", "2026-11-17 08:30:20"]
    # 사용자 2026-11-16 "다음날하자": 같은 날에는 아무것도 안 쓴다
    assert tr.due(rows, NOW) == []
    nxt = NEXT
    got = tr.due(rows, nxt)
    assert [r["symbol"] for r in got] == ["BTCUSDT", "NVDAUSDT"]  # 오래된 것부터, 11-17은 아직
    assert tr.due(rows, nxt, limit=1)[0]["at"] == "2026-11-16 10:30:20"
    tr.trade_dir().mkdir(parents=True)
    (tr.trade_dir() / f"{tr.trade_id(BTC)}.json").write_text("{}")
    assert BTC["at"] not in [r["at"] for r in tr.due(rows, nxt)]


def test_live_rows_needs_live_capital(files):
    bt.CAPITAL_FILE.write_text(json.dumps({"live": False, "since_at": "2026-11-16 00:00"}))
    assert tr.live_rows() == []


def test_facts_manual_and_tp_custom(files):
    f = tr.facts(NVDA, NOW, flat_bars)
    assert f["manual"] and f["tp_custom"] and f["why_ko"] == "익절"
    assert f["entry_way"].startswith("사용자 수동 진입")
    assert f["hold_min"] == 175
    kinds = [e["event"] for e in f["hold_events"]]
    assert kinds == ["tp_custom_kept", "adopt_manual"]  # BTC trail_on은 다른 종목
    assert [e["event"] for e in f["entry_events"]] == ["adopt_manual"]
    joined = " / ".join(f["interventions"])
    assert "손으로 연 포지션" in joined and "익절가 97.0" in joined and "수량·방향" in joined
    assert f["counterfactual"]["fixed_tp"]["result"] == "open"
    assert f["counterfactual"]["rule_now"]["result"] == "open"
    assert f["after"]["partial"] is True and 0 < f["after"]["bars"] < tr.AFTER_BARS_NEXT_DAY


def test_facts_engine_entry_and_no_interventions(files):
    f = tr.facts(BTC, NOW, flat_bars)
    assert f["entry_way"] == "엔진 신호 진입 · BTC 롱 진입"
    assert {e["event"] for e in f["entry_events"]} == {"run", "ai_choose"}
    assert f["entry_events"][1]["reason"] == "승률 높음" and "ai_raw" not in f["entry_events"][1]
    assert f["interventions"] == [] and f["counterfactual"]["fixed_tp"] is None
    manual_close = tr.facts({**BTC, "why": "manual"}, NOW, flat_bars)
    assert "엔진 밖에서 닫힘" in manual_close["interventions"][0]


def test_facts_survives_fetch_error_and_missing_exit(files):
    def boom(*a):
        raise OSError("down")

    f = tr.facts(BTC, NOW, boom)
    assert f["bars_error"] == "OSError" and f["counterfactual"] is None
    g = tr.facts({**BTC, "exit": None, "pnl": None, "pct": None}, NOW, flat_bars)
    assert g["mfe_pct"] is None and g["counterfactual"]["rule_now"]["result"] == "open"


def test_simulate_stop_checked_first_long_and_short():
    b = seq("2026-11-16 10:00:00", [(100, 104, 96.5, 101)])
    got = tr.simulate("long", 100.0, b, tp_pct=3.0)
    assert got["result"] == "stop" and got["pct"] == -3.0 and got["price"] == 97.0
    b = seq("2026-11-16 10:00:00", [(100, 103.5, 96, 99)])
    got = tr.simulate("short", 100.0, b, tp_pct=3.0)
    assert got["result"] == "stop" and got["pct"] == -3.0


def test_simulate_gap_through_stop_fills_at_open():
    b = seq("2026-11-16 10:00:00", [(100, 100.5, 99.5, 100), (95, 95.5, 94, 95)])
    got = tr.simulate("long", 100.0, b, trail=True)
    assert got["result"] == "stop" and got["price"] == 95.0 and got["at"] == "2026-11-16 10:15"


def test_simulate_trail_long():
    b = seq("2026-11-16 10:00:00", [
        (100, 103.5, 99.5, 103),  # +3% 닿음 → 손절 max(100.2, 103.5×0.97=100.395)
        (103, 106, 101, 105),  # 100.395 안 닿음 → 106×0.97 = 102.82
        (105, 105, 102, 102.5),  # 102.82 닿음
    ])
    got = tr.simulate("long", 100.0, b, trail=True)
    assert got["result"] == "trail" and got["price"] == pytest.approx(102.82)
    assert got["pct"] == 2.82 and got["trail_on_at"] == "2026-11-16 10:00"
    assert got["at"] == "2026-11-16 10:30"


def test_simulate_trail_short_and_close_beyond_new_stop():
    b = seq("2026-11-16 10:00:00", [(100, 100.5, 96.5, 97), (97, 100, 96.8, 99)])
    got = tr.simulate("short", 100.0, b, trail=True)
    # 96.5×1.03 = 99.395 vs 본전+0.2% 99.8 → 99.395, 다음 봉 고가 100이 닿음
    assert got["result"] == "trail" and got["price"] == pytest.approx(99.395)
    assert got["pct"] == pytest.approx(0.6, abs=0.01)
    b = seq("2026-11-16 10:00:00", [(100, 104, 99.9, 100.1)])
    got = tr.simulate("long", 100.0, b, trail=True)  # 켜진 봉이 손절 밖에서 마감
    assert got["result"] == "trail" and got["price"] == 100.1 and "종가" in got["note"]


def test_simulate_fixed_tp_long_short_and_open():
    b = seq("2026-11-16 10:00:00", [(100, 101, 99, 100), (100, 103.5, 99.5, 103)])
    got = tr.simulate("long", 100.0, b, tp_pct=3.0)
    assert got["result"] == "tp" and got["price"] == 103.0 and got["at"] == "2026-11-16 10:15"
    b = seq("2026-11-16 10:00:00", [(100, 100, 96.9, 97)])
    got = tr.simulate("short", 100.0, b, tp_pct=3.0)
    assert got["result"] == "tp" and got["pct"] == 3.0
    got = tr.simulate("long", 100.0, seq("2026-11-16 10:00:00", [(100, 101, 99, 100.5)]),
                      tp_pct=3.0)
    assert got["result"] == "open" and got["pct"] == 0.5
    assert tr.simulate("long", 100.0, [])["result"] == "open"


def test_counterfactual_fixed_tp_alt_for_custom_tp(files):
    # 숏, 손으로 정한 익절 97: 고정 +3%(97)와 같음 → 첫 봉 저가 96.9에 익절
    b = seq("2026-11-16 17:15:00", [(100, 100, 96.9, 97)])
    cf = tr.counterfactuals(NVDA, b)
    assert cf["fixed_tp"]["result"] == "tp" and cf["fixed_tp"]["pct"] == 3.0
    assert cf["rule_now"]["rule"].startswith("손절 3%")


def test_parse_clamps_and_marks_lesson():
    raw = ('x {"headline": "h", "what": "w", "good": ["a", "b", "c", "d"], "bad": "no",'
           ' "lesson": "늦게 닫음", "manual_note": "손 익절이 빨랐다", "rule_check": 1} y')
    got = tr.parse(raw)
    assert got["good"] == ["a", "b", "c"] and got["bad"] == []
    assert got["lesson"] == "늦게 닫음 (검증 안 됨)" and got["rule_check"] == "1"
    assert got["manual_note"] == "손 익절이 빨랐다"
    assert tr.parse(raw, manual=False)["manual_note"] == ""
    assert tr.parse("not json") is None and tr.parse('{"good": []}') is None


def test_ask_fallbacks():
    def slow(p):
        raise subprocess.TimeoutExpired("claude", 1)

    assert tr.ask({}, slow)[0] is None
    assert tr.ask({}, lambda p: "??")[1] == "답을 읽지 못함"
    got, err = tr.ask({"interventions": []}, lambda p: '{"headline": "ok", "manual_note": "x"}')
    assert got["headline"] == "ok" and got["manual_note"] == "" and err is None


def test_run_saves_sends_and_skips_next_time(files):
    sent, asked = [], []

    def caller(p):
        asked.append(p)
        return '{"headline": "손 익절이 규칙과 같았다", "lesson": "관찰"}'

    paths = tr.run(NEXT, caller=caller, fetch=flat_bars, send=sent.append)
    assert [p.name for p in paths] == [f"{tr.trade_id(BTC)}.json", f"{tr.trade_id(NVDA)}.json"]
    assert all(oct(os.stat(p).st_mode & 0o777) == "0o600" for p in paths)
    assert "interventions" in asked[1] and "NVDAUSDT" in asked[1]
    assert sent[1].startswith("📝 거래 복기 · NVDA 숏 +3.0% (익절) · 수동\n")
    assert "교훈: 관찰 (검증 안 됨)" in sent[1]
    assert tr.run(NEXT, caller=caller, fetch=flat_bars, send=sent.append) == []
    assert len(sent) == 2


def test_run_no_ai_keeps_numbers_and_limit(files):
    paths = tr.run(NEXT, ai=False, limit=1, fetch=flat_bars)
    rec = json.loads(paths[0].read_text())
    assert rec["review"] is None and rec["review_error"] == "AI 끔"
    assert tr.message(rec).endswith("AI 복기 없음: AI 끔")


def test_load_trades_order_and_api(files):
    tr.run(NEXT, ai=False, fetch=flat_bars)
    (tr.trade_dir() / "notes.json").write_text("{}")
    (tr.trade_dir() / "20261101-0000_XUSDT_20261101-0100.json").write_text("{broken")
    got = tr.load_trades()
    assert [r["symbol"] for r in got] == ["NVDAUSDT", "BTCUSDT"]  # 늦게 닫힌 것이 위
    assert len(tr.load_trades(limit=1)) == 1
    api = TestClient(app).get("/api/btc/reviews").json()
    assert [t["symbol"] for t in api["trades"]] == ["NVDAUSDT", "BTCUSDT"]
    assert api["reviews"] == []


def test_record_close_keeps_manual_and_exit_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "DIR", tmp_path / "bt")
    sent = []
    ctx = bt.Ctx(None, None, sent.append, NOW, True)
    pos = {"symbol": "NVDAUSDT", "side": "short", "qty": 2, "entry": 100.0, "stop": 101.0,
           "tp": 95.0, "tp_custom": True, "manual": True, "trail": True, "best": 96.0,
           "opened_at": "2026-11-16 17:15:20"}
    bt._record_close(ctx, pos, 95.0, 10.0, "tp")
    bt._record_close(ctx, {k: pos[k] for k in ("side", "qty", "entry")}, 99.0, 2.0, "signal")
    rows = [json.loads(x) for x in (bt.DIR / "closed.jsonl").read_text().splitlines()]
    assert {k: rows[0][k] for k in ("manual", "tp_custom", "tp", "stop", "trail", "best")} == {
        "manual": True, "tp_custom": True, "tp": 95.0, "stop": 101.0, "trail": True, "best": 96.0}
    assert rows[0]["pct"] == pytest.approx(5.0)
    assert {k: rows[1][k] for k in ("manual", "tp_custom", "tp", "stop", "trail", "best")} == {
        "manual": False, "tp_custom": None, "tp": None, "stop": None, "trail": False,
        "best": None}
