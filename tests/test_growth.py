import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from minerva import growth as g
from minerva.main import app


def _row(sym, name, cap, industry="Software", volume="1000", sector="Technology"):
    return {"symbol": sym, "name": name, "marketCap": str(cap), "industry": industry,
            "sector": sector, "country": "United States", "volume": volume}


def _ts(annual, quarterly=(), net=None, op=None):
    def series(t, pts):
        return {"meta": {"type": [t]},
                t: [{"asOfDate": d, "reportedValue": {"raw": v}} for d, v in pts]}

    res = [series("annualTotalRevenue", annual), series("quarterlyTotalRevenue", quarterly)]
    if net is not None:
        res.append(series("trailingNetIncome", [("2026-09-30", net)]))
    if op is not None:
        res.append(series("trailingOperatingIncome", [("2026-09-30", op)]))
    return {"timeseries": {"result": res}}


def test_universe_keeps_common_in_cap_range_and_drops_junk():
    rows = [
        _row("AAA", "Alpha Inc. Common Stock", 5e9),
        _row("BIG", "Big Corp Common Stock", 900e9),  # 시총 위
        _row("TINY", "Tiny Corp Common Stock", 5e8),  # 시총 아래
        _row("AAAP", "Alpha Inc. 7.5% Series A Cumulative Preferred Stock", 5e9),
        _row("AAAN", "Alpha Inc. 6.0% Senior Notes due 2030", 5e9),
        _row("SPCW", "Space Acquisition Corp Warrant", 2e9),
        _row("SPC", "Space Acquisition Corp Class A Ordinary Shares", 2e9),
        _row("BKX", "Blank Check Holdings Common Stock", 2e9, industry="Blank Checks"),
        _row("PFX", "Bank Co Depositary Shares each representing 1/1000th", 3e9),
        _row("ADR", "Foreign plc American Depositary Shares", 50e9),
        _row("CFND", "Income Fund Inc. Common Stock", 2e9),
        _row("DUOA", "Duo Corp Class A Common Stock", 10e9, volume="100"),
        _row("DUOB", "Duo Corp Class B Common Stock", 10e9, volume="900"),
        _row("BRK/B", "Odd Symbol Common Stock", 5e9),
    ]
    got = {it["symbol"]: it for it in g.universe(rows)}
    assert set(got) == {"AAA", "ADR", "DUOB"}
    assert got["AAA"]["name"] == "Alpha Inc" and got["ADR"]["name"] == "Foreign plc"


def test_parse_fund_growth_profit_and_small_base():
    f = g.parse_fund(_ts([("2024-12-31", 100e6), ("2025-12-31", 150e6)],
                         [("2025-09-30", 30e6), ("2025-12-31", 40e6), ("2026-03-31", 40e6),
                          ("2026-06-30", 45e6), ("2026-09-30", 60e6)], net=-5e6, op=2e6))
    assert round(f["growth"], 4) == 0.5 and round(f["q_growth"], 4) == 1.0
    assert f["profit"] == "영업흑자"
    small = g.parse_fund(_ts([("2024-12-31", 1e6), ("2025-12-31", 90e6)], net=1.0))
    assert small["growth"] is None and small["profit"] == "흑자"  # 작은 수로 나눈 성장은 버림
    gap = g.parse_fund(_ts([("2022-12-31", 100e6), ("2025-12-31", 200e6)]))
    assert gap["growth"] is None  # 연속한 두 해가 아니면 버림
    assert g.parse_fund({"timeseries": {"result": []}})["empty"] is True


def test_empty_answers_are_not_cached_and_streak_stops():
    cache, calls = {}, []

    def get(url):
        calls.append(url)
        return {"timeseries": {"result": []}}

    st = g.refresh_funds([f"S{i}" for i in range(40)], cache, get=get, sleep=lambda s: None,
                         log=lambda *a: None)
    assert cache == {} and st["failed"] == 15 and len(calls) == 15


def test_rank_sorts_by_growth_per_bucket_and_drops_missing():
    items = [
        {"symbol": s, "name": s, "sector": None, "industry": None, "country": None, "cap": c}
        for s, c in [("A", 50e9), ("B", 100e9), ("C", 2e9), ("D", 3e9), ("E", 40e9)]
    ]
    funds = {"A": {"growth": 0.2}, "B": {"growth": 0.9}, "C": {"growth": None},
             "D": {"growth": 0.4}, "E": {"growth": 0.5}}
    r = g.rank(items, funds)
    assert [x["symbol"] for x in r["x10"]] == ["B", "E", "A"]  # $40B는 10배 칸
    assert [x["symbol"] for x in r["x100"]] == ["D"]
    assert len(g.rank(items * 1, funds, top=1)["x10"]) == 1


def _fake_get(fail_ts=False, nasdaq_rows=None):
    rows = nasdaq_rows if nasdaq_rows is not None else (
        [_row("AAA", "Alpha Inc. Common Stock", 50e9), _row("CCC", "Gamma Inc. Common Stock", 2e9)]
        + [_row(f"Z{i:04d}"[:5], f"Filler {i} Corp Common Stock", 5e8) for i in range(1000)]
    )

    def get(url):
        if "nasdaq" in url:
            return {"data": {"rows": rows}}
        if "timeseries" in url:
            if fail_ts:
                raise OSError("down")
            sym = url.split("symbol=")[1].split("&")[0]
            base = 100e6 if sym == "AAA" else 50e6
            return _ts([("2024-12-31", base), ("2025-12-31", base * 2)], net=1.0)
        if "chart" in url:
            return {"chart": {"result": [{"meta": {"regularMarketPrice": 50.0,
                                                    "fiftyTwoWeekHigh": 100.0}}]}}
        raise AssertionError(url)

    return get


def test_run_writes_file_and_api_serves_it(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_GROWTH_DIR", str(tmp_path))
    client = TestClient(app)
    assert client.get("/api/growth").json()["status"] == "pending"
    data = g.run(get=_fake_get(), sleep=lambda s: None, log=lambda *a: None)
    assert data["buckets"][0]["rows"][0]["symbol"] == "AAA"
    assert data["buckets"][0]["rows"][0]["drawdown"] == -0.5
    assert data["buckets"][1]["rows"][0]["symbol"] == "CCC"
    r = client.get("/api/growth").json()
    assert r["status"] == "ok" and r["stale"] is False and "매수 추천이 아닙니다" in r["note"]
    # 다음 실행: 재무 캐시가 살아 있으면 재무는 다시 받지 않는다
    calls = []
    get = _fake_get()
    g.run(get=lambda u: calls.append(u) or get(u), sleep=lambda s: None, log=lambda *a: None)
    assert not any("timeseries" in u for u in calls)


def test_source_failure_keeps_previous_file_and_marks_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_GROWTH_DIR", str(tmp_path))
    g.run(get=_fake_get(), sleep=lambda s: None, log=lambda *a: None)
    before = json.loads((tmp_path / "latest.json").read_text())

    def broken(url):
        raise OSError("blocked")

    # 목록은 지난 universe.json으로, 재무 캐시도 그대로라 목록은 내되 경고가 붙는다
    data = g.run(get=broken, sleep=lambda s: None, log=lambda *a: None)
    assert data is not None and data["warnings"]
    # 캐시도 목록도 없으면 지난 파일을 그대로 두고 stale만
    (tmp_path / "universe.json").unlink()
    assert g.run(get=broken, sleep=lambda s: None, log=lambda *a: None) is None
    after = json.loads((tmp_path / "latest.json").read_text())
    assert after["stale"] is True and after["buckets"] == data["buckets"]
    assert before["buckets"][0]["rows"][0]["symbol"] == after["buckets"][0]["rows"][0]["symbol"]
    r = TestClient(app).get("/api/growth").json()
    assert r["status"] == "ok" and r["stale"] is True


def test_too_few_financials_keeps_previous(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_GROWTH_DIR", str(tmp_path))
    assert g.run(get=_fake_get(fail_ts=True), sleep=lambda s: None, log=lambda *a: None) is None
    assert not (tmp_path / "latest.json").exists()


def test_api_marks_old_file_stale_and_survives_broken_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_GROWTH_DIR", str(tmp_path))
    old = (datetime.now(timezone(timedelta(hours=9))) - timedelta(days=5)).isoformat()
    (tmp_path / "latest.json").write_text(json.dumps(
        {"generated_at": old, "stale": False, "buckets": []}))
    client = TestClient(app)
    assert client.get("/api/growth").json()["stale"] is True
    (tmp_path / "latest.json").write_text("{broken")
    assert client.get("/api/growth").json()["status"] == "pending"


def _data():
    rows = [{"symbol": "AAA", "name": "Alpha", "cap": 50e9, "growth": 1.0, "profit": "흑자",
             "drawdown": -0.6}]
    return {"generated_at": "2026-11-21T07:40+09:00", "buckets": [
        {"key": "x10", "label": "10배 후보", "rows": rows},
        {"key": "x100", "label": "100배 후보(위험 높음)", "rows": []}]}


def test_fear_alert_sends_once_per_episode(tmp_path):
    state = tmp_path / "alert.json"
    sent = []

    def send(text):
        sent.append(text)
        return True

    fear = {"last_signal": "2026-11-19", "elapsed": 1, "premium": 0.16}
    assert g.maybe_alert(_data(), fear, state, send=send) is True
    assert g.maybe_alert(_data(), fear, state, send=send) is False  # 같은 신호는 다시 안 보냄
    assert len(sent) == 1 and "AAA" in sent[0] and "매수 추천이 아닙니다" in sent[0]
    new = {"last_signal": "2027-03-02", "elapsed": 0, "premium": 0.07}
    assert g.maybe_alert(_data(), new, state, send=send) is True and len(sent) == 2


def test_fear_alert_skips_old_or_no_signal_and_retries_failed_send(tmp_path):
    state = tmp_path / "alert.json"
    never = lambda t: (_ for _ in ()).throw(AssertionError("sent"))  # noqa: E731
    assert g.maybe_alert(_data(), {"state": "없음"}, state, send=never) is False
    old = {"last_signal": "2026-08-01", "elapsed": g.FEAR_ALERT_DAYS + 1}
    assert g.maybe_alert(_data(), old, state, send=never) is False
    fear = {"last_signal": "2026-11-19", "elapsed": 0}
    assert g.maybe_alert(_data(), fear, state, send=lambda t: False) is False
    assert not state.exists()  # 보내기 실패는 기록하지 않아 다음 실행에 다시
    assert g.maybe_alert(_data(), fear, state, send=lambda t: True) is True
