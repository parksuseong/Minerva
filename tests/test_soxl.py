from datetime import date, datetime

from fastapi.testclient import TestClient

from minerva import routes
from minerva.main import app
from minerva.strategy import btc, soxl
from minerva.strategy.indicators import Bar


def _bar(d):
    return Bar(d, 1, 2, 0.5, 1.5, 100)


def test_ny_clock_follows_daylight_saving():
    # 11/11 21:00 KST = 12:00 UTC → 겨울(EST) 07:00
    assert soxl._ny_now(datetime(2026, 11, 11, 21, tzinfo=btc.KST)).hour == 7
    # 7/1 21:00 KST = 12:00 UTC → 여름(EDT) 08:00
    assert soxl._ny_now(datetime(2026, 7, 1, 21, tzinfo=btc.KST)).hour == 8


def test_closed_days_drops_today_until_us_close():
    bars = [_bar(date(2026, 11, 10)), _bar(date(2026, 11, 11))]
    # 뉴욕 11/11 10:00 (KST 11/12 00:00): 오늘 봉은 아직 진행 중
    open_now = datetime(2026, 11, 12, 0, tzinfo=btc.KST)
    assert soxl.closed_days(bars, open_now)[-1].day == date(2026, 11, 10)
    # 뉴욕 11/11 17:00 (KST 11/12 07:00): 마감됨
    closed_now = datetime(2026, 11, 12, 7, tzinfo=btc.KST)
    assert soxl.closed_days(bars, closed_now)[-1].day == date(2026, 11, 11)


def test_soxl_route_caches_and_survives_errors(monkeypatch):
    routes._soxl_cache.clear()
    calls = []

    def ok(**kw):
        calls.append(1)
        return {"status": "ok", "outlook": {"price": 65.0}}

    clock = [1000.0]
    monkeypatch.setattr(routes.soxl, "build", ok)
    monkeypatch.setattr(routes, "_mono", lambda: clock[0])
    client = TestClient(app)
    assert client.get("/api/soxl").json()["outlook"]["price"] == 65.0
    client.get("/api/soxl")
    assert len(calls) == 1  # 5분 캐시

    def boom(**kw):
        raise OSError("down")

    monkeypatch.setattr(routes.soxl, "build", boom)
    clock[0] += routes.SOXL_TTL + 1
    assert client.get("/api/soxl").json()["outlook"]["price"] == 65.0  # 지난 값 유지
    routes._soxl_cache.clear()
    assert client.get("/api/soxl").json()["status"] == "error"
    routes._soxl_cache.clear()


def test_soxl_route_rejects_unknown_symbol_and_caches_per_symbol(monkeypatch):
    routes._soxl_cache.clear()
    seen = []
    monkeypatch.setattr(routes.soxl, "build", lambda symbol: seen.append(symbol) or {"s": symbol})
    client = TestClient(app)
    assert client.get("/api/soxl?symbol=TSLAUSDT").status_code == 400
    assert client.get("/api/soxl?symbol=NVDAUSDT").json() == {"s": "NVDAUSDT"}
    assert client.get("/api/soxl").json() == {"s": "SOXLUSDT"}
    assert seen == ["NVDAUSDT", "SOXLUSDT"]
    routes._soxl_cache.clear()
