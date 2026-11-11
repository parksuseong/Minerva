"""웹 화면 API가 깨진·빈 파일이나 이상한 입력에도 500을 내지 않는지."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from minerva import btc_trade as bt
from minerva import btc_view
from minerva.main import app

SCRIPTS = Path(__file__).parent.parent / "scripts"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trade_status_skips_half_written_journal_line(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_TRADE_DIR", str(tmp_path))
    (tmp_path / "journal.jsonl").write_text('{"key": "a"}\n\n[1]\n{"key": "b"}\n{"key": "c", "na')
    r = TestClient(app).get("/api/trade")
    assert r.status_code == 200
    assert [o["key"] for o in r.json()["orders"]] == ["b", "a"]


def test_tail_jsonl_keeps_only_objects(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n"str"\nnot json\n{"a": 2}\n')
    assert btc_view.tail_jsonl(path, 10) == [{"a": 1}, {"a": 2}]
    assert btc_view.tail_jsonl(path, 1) == [{"a": 2}]
    assert btc_view.tail_jsonl(tmp_path / "none.jsonl", 5) == []


def test_engine_view_shows_hand_made_empty_no_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "DIR", tmp_path)
    monkeypatch.setattr(bt, "CAPITAL_FILE", tmp_path / "capital.json")
    assert btc_view.engine_view()["no_entry"] is None
    (tmp_path / "NO_ENTRY").write_text("")
    assert btc_view.engine_view()["no_entry"]  # 빈 파일도 정지로 보인다


def test_signals_with_null_vix(tmp_path, monkeypatch):
    path = tmp_path / "daily.json"
    path.write_text(json.dumps({"generated_at": "x", "vix": None, "actions": []}))
    monkeypatch.setenv("MINERVA_DAILY_OUT", str(path))
    r = TestClient(app).get("/api/signals")
    assert r.status_code == 200 and r.json()["regime"] is None


def test_office_token_non_ascii_is_403_not_500(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_OFFICE_LOG", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("MINERVA_OFFICE_TOKEN", "secret")
    r = TestClient(app).get("/api/office/events", params={"token": "비밀"})
    assert r.status_code == 403


@pytest.mark.parametrize(
    "route", ["/api/account", "/api/account/binance", "/api/account/binance/series"]
)
def test_account_token_non_ascii_is_403_not_500(tmp_path, monkeypatch, route):
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN", "secret")
    monkeypatch.setenv("MINERVA_ACCOUNT_OUT", str(tmp_path / "latest.json"))
    r = TestClient(app).get(route, headers={"X-Minerva-Token": b"\xed\x95\x9c"})
    assert r.status_code == 403
    ok = TestClient(app).get(route, headers={"X-Minerva-Token": "secret"})
    assert ok.status_code == 200


def test_redirect_never_leaves_the_host():
    mod = _script("http_redirect")
    host = mod.HOST
    assert mod.location("/a?b=1") == f"https://{host}/a?b=1"
    assert mod.location("//evil.com/x") == f"https://{host}//evil.com/x"
    # 경로를 그대로 붙이면 https://host@evil.com 이 되어 evil.com으로 간다
    assert mod.location("@evil.com") == f"https://{host}/"
    assert mod.location("http://evil.com/") == f"https://{host}/"


def test_hook_trims_big_log_in_place(tmp_path, monkeypatch):
    mod = _script("office_hook")
    log = tmp_path / "events.jsonl"
    monkeypatch.setattr(mod, "LOG", log)
    monkeypatch.setattr(mod, "MAX_BYTES", 100)
    log.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(3000)))
    mod.append({"i": "new"})
    lines = log.read_text().splitlines()
    assert len(lines) == 2001
    assert json.loads(lines[0]) == {"i": 1000} and json.loads(lines[-1]) == {"i": "new"}
    assert [p.name for p in tmp_path.iterdir()] == ["events.jsonl"]
