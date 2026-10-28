import json

from fastapi.testclient import TestClient

from minerva import kis, perf
from minerva.main import app

CAPITAL = {"since": "2026-10-29", "base": 10_000_000, "deposits": []}


def test_returns_on_start_day_compare_with_base():
    acct = {"equity": 10_100_000, "updated_at": "2026-10-29 15:30:00"}
    r = perf.account_returns(acct, {"2026-10-28": 1_000_000}, CAPITAL, 50_000)
    assert r["total"] == 100_000 and r["total_pct"] == 1.0
    assert r["today"] == 100_000  # 시작 전 기록(10/28)은 쓰지 않는다
    assert r["bot_pct"] == 0.5


def test_deposits_excluded_from_gain():
    capital = {**CAPITAL, "deposits": [{"date": "2026-11-02", "amount": 5_000_000}]}
    acct = {"equity": 15_300_000, "updated_at": "2026-11-02 10:00:00"}
    history = {"2026-10-29": 10_100_000, "2026-10-30": 10_200_000}
    r = perf.account_returns(acct, history, capital, 0)
    assert r["invested"] == 15_000_000 and r["total"] == 300_000
    assert r["today"] == 100_000 and round(r["today_pct"], 4) == round(
        100_000 / 15_200_000 * 100, 4
    )


def test_today_unknown_without_previous_close():
    acct = {"equity": 10_000_000, "updated_at": "2026-10-30 09:00:00"}
    assert "today" not in perf.account_returns(acct, {}, CAPITAL, 0)


def test_record_equity_keeps_last_value_per_day(tmp_path):
    path = tmp_path / "equity.json"
    kis.record_equity(path, {"status": "ok", "equity": 1, "updated_at": "2026-10-29 09:00:00"})
    kis.record_equity(path, {"status": "ok", "equity": 2, "updated_at": "2026-10-29 15:30:00"})
    kis.record_equity(path, {"status": "error", "equity": 9, "updated_at": "2026-10-30 09:00:00"})
    assert json.loads(path.read_text()) == {"2026-10-29": 2}


def test_account_api_adds_returns(tmp_path, monkeypatch):
    snap = tmp_path / "latest.json"
    snap.write_text(
        json.dumps(
            {
                "status": "ok",
                "equity": 10_000_000,
                "updated_at": "2026-10-29 08:30:00",
                "positions": [],
            }
        )
    )
    (tmp_path / "capital.json").write_text(json.dumps(CAPITAL))
    monkeypatch.setenv("MINERVA_ACCOUNT_OUT", str(snap))
    monkeypatch.setenv("MINERVA_TRADE_DIR", str(tmp_path / "trade"))
    monkeypatch.delenv("MINERVA_ACCOUNT_TOKEN", raising=False)
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN_FILE", str(tmp_path / "none"))
    r = TestClient(app).get("/api/account").json()["returns"]
    assert r["total"] == 0 and r["today"] == 0 and r["bot"] == 0
