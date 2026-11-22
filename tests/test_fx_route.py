"""계좌 탭 원화 보기: /api/fx/usdkrw 캐시와 실패 처리(네트워크 없이)."""

from __future__ import annotations

from minerva import routes


def _reset(monkeypatch):
    monkeypatch.setattr(routes, "_fx", {"rate": None, "at": None, "ts": 0.0})


def test_fx_caches_and_reuses(monkeypatch):
    _reset(monkeypatch)
    calls = []

    def fake():
        calls.append(1)
        return 1400.5, 1_795_000_000

    monkeypatch.setattr(routes, "_fetch_usdkrw", fake)
    a = routes.fx_usdkrw()
    b = routes.fx_usdkrw()
    assert a["status"] == "ok" and a["rate"] == 1400.5 and not a["stale"]
    assert b["rate"] == 1400.5 and len(calls) == 1


def test_fx_failure_returns_last_value_as_stale(monkeypatch):
    _reset(monkeypatch)

    def boom():
        raise OSError("down")

    monkeypatch.setattr(routes, "_fetch_usdkrw", boom)
    assert routes.fx_usdkrw()["status"] == "error"
    routes._fx.update(rate=1390.0, at=1_795_000_000, ts=0.0)
    out = routes.fx_usdkrw()
    assert out["status"] == "ok" and out["rate"] == 1390.0 and out["stale"] is True
