import pytest

from minerva import audit, trade

DAY = "2026-10-29"
ACCOUNT = {"equity": 1_000_000, "positions": []}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(trade, "TRADE_DIR", tmp_path)


def _e(**kw):
    base = {
        "key": f"{DAY}:005930:buy",
        "side": "buy",
        "code": "005930",
        "name": "A",
        "action": "LONG_ENTRY",
        "result": "sent",
        "kind": "long",
        "at": f"{DAY}T08:50:00",
    }
    return {**base, **kw}


def levels(results):
    return [r["level"] for r in results if r["level"] != "info"]


def test_clean_day_has_no_findings():
    acc = {**ACCOUNT, "positions": [{"code": "005930", "qty": 2, "avg": 50000, "price": 51000}]}
    owned = {"005930": {"name": "A", "kind": "long", "stop": 45000}}
    results = audit.audit([_e()], owned, acc, "live", DAY)
    assert levels(results) == []
    assert "이상 없음" in audit.report(results)


def test_flags_duplicates_foreign_codes_and_rejections():
    journal = [
        _e(),
        _e(),
        _e(key="k2", code="NVDA", name="N"),
        _e(key="k3", result="rejected", msg="잔고 부족"),
    ]
    results = audit.audit(journal, {}, ACCOUNT, "live", DAY)
    msgs = " ".join(r["msg"] for r in results)
    assert "중복 주문 2회" in msgs and "국내 종목 코드가 아님" in msgs and "잔고 부족" in msgs


def test_flags_oversized_position_and_wrong_or_missed_stop():
    acc = {**ACCOUNT, "positions": [{"code": "005930", "qty": 10, "avg": 50000, "price": 44000}]}
    owned = {"005930": {"name": "A", "kind": "long", "stop": 40000}}
    msgs = " ".join(r["msg"] for r in audit.audit([], owned, acc, "live", DAY))
    assert "계좌의 50%" in msgs and "≠ 규칙 45,000" in msgs
    owned["005930"]["stop"] = 45000
    msgs = " ".join(r["msg"] for r in audit.audit([], owned, acc, "live", DAY))
    assert "손절가인데 아직 보유" in msgs


def test_ignores_selftest_and_other_days():
    journal = [_e(kind="test", result="failed"), _e(at="2026-10-28T08:50:00", result="failed")]
    assert levels(audit.audit(journal, {}, ACCOUNT, "dry", DAY)) == []


def test_reports_bot_position_missing_from_account():
    owned = {"000660": {"name": "H", "kind": "long", "stop": 1}}
    assert levels(audit.audit([], owned, ACCOUNT, "live", DAY)) == ["warn"]
