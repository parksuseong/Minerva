import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from minerva import kis
from minerva.main import app

NOW = datetime(2026, 10, 28, 11, 0, tzinfo=kis.KST)
CFG = {"cano": "12345678", "prd": "01", "app_key": "k", "app_secret": "s"}


def test_load_config_splits_account(tmp_path):
    env = tmp_path / "kis.env"
    env.write_text("# c\nKIS_ACCOUNT=12345678-01\nKIS_APP_KEY=abc\nKIS_APP_SECRET=x/y+z=\n")
    cfg = kis.load_config(env)
    assert cfg == {"cano": "12345678", "prd": "01", "app_key": "abc", "app_secret": "x/y+z="}


def test_parse_balance_skips_sold_positions():
    pages = [
        {
            "output1": [
                {
                    "prdt_name": "삼성전자 ",
                    "pdno": "005930",
                    "hldg_qty": "3",
                    "pchs_avg_pric": "70000.0",
                    "prpr": "72000",
                    "evlu_amt": "216000",
                    "evlu_pfls_amt": "6000",
                    "evlu_pfls_rt": "2.86",
                },
                {"prdt_name": "매도완료", "pdno": "000660", "hldg_qty": "0"},
            ],
            "output2": [
                {
                    "dnca_tot_amt": "1000",
                    "prvs_rcdl_excc_amt": "900",
                    "scts_evlu_amt": "216000",
                    "tot_evlu_amt": "217000",
                    "evlu_pfls_smtl_amt": "6000",
                }
            ],
        }
    ]
    b = kis.parse_balance(pages)
    assert [p["code"] for p in b["positions"]] == ["005930"]
    assert b["positions"][0]["name"] == "삼성전자" and b["positions"][0]["qty"] == 3
    assert b["equity"] == 217000 and b["cash_d2"] == 900


def test_parse_orders_status_and_trades():
    rows = [
        {
            "ord_dt": "20261027",
            "ord_tmd": "090102",
            "prdt_name": "A",
            "pdno": "1",
            "sll_buy_dvsn_cd_name": "매수",
            "ord_qty": "10",
            "ord_unpr": "100",
            "tot_ccld_qty": "10",
            "avg_prvs": "99",
            "cncl_yn": "N",
        },
        {
            "ord_dt": "20261028",
            "ord_tmd": "100000",
            "prdt_name": "B",
            "pdno": "2",
            "sll_buy_dvsn_cd_name": "매도",
            "ord_qty": "5",
            "ord_unpr": "200",
            "tot_ccld_qty": "2",
            "avg_prvs": "201",
            "cncl_yn": "N",
        },
        {
            "ord_dt": "20261028",
            "ord_tmd": "100500",
            "prdt_name": "C",
            "pdno": "3",
            "ord_qty": "1",
            "tot_ccld_qty": "0",
            "cncl_yn": "Y",
        },
    ]
    orders, trades = kis.parse_orders([{"output1": rows}])
    assert [o["status"] for o in orders] == ["취소", "부분체결 2", "체결"]
    assert orders[0]["time"] == "2026-10-28 10:05"
    assert [(t["name"], t["qty"], t["price"]) for t in trades] == [("B", 2, 201), ("A", 10, 99)]


def test_access_token_reuses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(kis, "STATE_DIR", tmp_path)
    (tmp_path / "token.json").write_text(
        json.dumps({"access_token": "cached", "access_token_token_expired": "2026-10-29 10:00:00"})
    )
    monkeypatch.setattr(kis, "_request", lambda req: pytest.fail("should not request"))
    assert kis.access_token(CFG, NOW) == "cached"


def test_access_token_refreshes_near_expiry(tmp_path, monkeypatch):
    monkeypatch.setattr(kis, "STATE_DIR", tmp_path)
    (tmp_path / "token.json").write_text(
        json.dumps({"access_token": "old", "access_token_token_expired": "2026-10-28 11:10:00"})
    )
    fresh = {"access_token": "new", "access_token_token_expired": "2026-10-29 11:00:00"}
    monkeypatch.setattr(kis, "_request", lambda req: (fresh, {}))
    assert kis.access_token(CFG, NOW) == "new"
    assert (tmp_path / "token.json").stat().st_mode & 0o777 == 0o600


def test_get_raises_on_api_error(monkeypatch):
    monkeypatch.setattr(kis, "_request", lambda req: ({"rt_cd": "1", "msg1": "bad"}, {}))
    with pytest.raises(kis.KisError, match="bad"):
        kis._get(CFG, "t", "/x", "TR", {})


def test_get_follows_pages(monkeypatch):
    calls = []

    def fake(req):
        calls.append(req.headers.get("Tr_cont"))
        more = "M" if len(calls) == 1 else "D"
        return {"rt_cd": "0", "output1": [len(calls)], "ctx_area_nk100": "n"}, {"tr_cont": more}

    monkeypatch.setattr(kis, "_request", fake)
    pages = kis._get(CFG, "t", "/x", "TR", {"CTX_AREA_NK100": ""})
    assert len(pages) == 2 and calls == ["", "N"]


def test_run_keeps_last_snapshot_on_failure(tmp_path, monkeypatch):
    out = tmp_path / "latest.json"
    out.write_text(json.dumps({"status": "ok", "cash": 5}))
    monkeypatch.setattr(kis, "load_config", lambda: CFG)

    def boom(cfg, now):
        raise kis.KisError("down")

    monkeypatch.setattr(kis, "snapshot", boom)
    with pytest.raises(kis.KisError):
        kis.run(out)
    data = json.loads(out.read_text())
    assert data["cash"] == 5 and data["error"] == "down"


def test_account_api_requires_view_token(tmp_path, monkeypatch):
    snap = tmp_path / "account.json"
    snap.write_text(json.dumps({"status": "ok", "positions": []}))
    monkeypatch.setenv("MINERVA_ACCOUNT_OUT", str(snap))
    monkeypatch.delenv("MINERVA_ACCOUNT_TOKEN", raising=False)
    token_file = tmp_path / "view.token"
    token_file.write_text("secret\n")
    monkeypatch.setenv("MINERVA_ACCOUNT_TOKEN_FILE", str(token_file))
    client = TestClient(app)
    assert client.get("/api/account").status_code == 403
    assert client.get("/api/account", headers={"X-Minerva-Token": "nope"}).status_code == 403
    ok = client.get("/api/account", headers={"X-Minerva-Token": "secret"})
    assert ok.json()["status"] == "ok"


def test_snapshot_reissues_token_when_server_rejects_it(tmp_path, monkeypatch):
    monkeypatch.setattr(kis, "STATE_DIR", tmp_path)
    (tmp_path / "token.json").write_text("{}")
    calls = []

    def fake(cfg, now):
        calls.append((tmp_path / "token.json").exists())
        if len(calls) == 1:
            raise kis.KisError('HTTP 500: {"msg_cd":"EGW00123","msg1":"기간이 만료된 token"}')
        return {"status": "ok"}

    monkeypatch.setattr(kis, "_snapshot", fake)
    assert kis.snapshot(CFG, NOW) == {"status": "ok"}
    assert calls == [True, False]


def test_snapshot_does_not_retry_other_errors(monkeypatch):
    def fake(cfg, now):
        raise kis.KisError("TTTC8434R X: 기타")

    monkeypatch.setattr(kis, "_snapshot", fake)
    with pytest.raises(kis.KisError):
        kis.snapshot(CFG, NOW)


def test_request_retries_rate_limit(monkeypatch):
    import io
    import urllib.error

    monkeypatch.setattr(kis.time, "sleep", lambda s: None)
    calls = []

    class Resp(io.BytesIO):
        headers = {"tr_cont": "D"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        calls.append(1)
        if len(calls) < 3:
            body = io.BytesIO(b'{"rt_cd":"1","msg_cd":"EGW00215"}')
            raise urllib.error.HTTPError("u", 500, "x", {}, body)
        return Resp(b'{"rt_cd":"0"}')

    monkeypatch.setattr(kis.urllib.request, "urlopen", fake_urlopen)
    data, headers = kis._request(kis.urllib.request.Request("https://x"))
    assert data == {"rt_cd": "0"} and len(calls) == 3
