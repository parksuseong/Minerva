"""barrier_paper: 장벽 판정·신호·실행 흐름(네트워크 없이)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from minerva import barrier_paper as bp

H = bp.HR_MS
D = bp.DAY_MS


def bar(t, o, h, low, c):
    return [t, o, h, low, c]


def test_barrier_long_win_loss_and_same_bar_loss():
    assert bp.barrier_exit("L", 100, [bar(0, 100, 106, 99, 105)]) == ("win", 105.0, 0)
    assert bp.barrier_exit("L", 100, [bar(0, 100, 101, 96, 97)]) == ("loss", 97.0, 0)
    # 같은 봉에서 둘 다 닿으면 손절
    assert bp.barrier_exit("L", 100, [bar(0, 100, 106, 96, 100)])[0] == "loss"
    # 둘째 봉 시가가 이미 손절 밖이면 그 시가
    r = bp.barrier_exit("L", 100, [bar(0, 100, 101, 99, 100), bar(H, 90, 91, 89, 90)])
    assert r == ("loss", 90, 1)
    assert bp.barrier_exit("L", 100, [bar(0, 100, 101, 99, 100)]) is None


def test_barrier_short_and_timeout():
    assert bp.barrier_exit("S", 100, [bar(0, 100, 101, 94, 95)]) == ("win", 95.0, 0)
    assert bp.barrier_exit("S", 100, [bar(0, 100, 104, 99, 103)])[0] == "loss"
    flat = [bar(k * H, 100, 101, 99, 100.5) for k in range(bp.CAP_H)]
    assert bp.barrier_exit("S", 100, flat) == ("timeout", 100.5, bp.CAP_H - 1)


def test_net_return_costs_and_funding():
    assert bp.net_return("L", 100, 105, 0.001) == pytest.approx(0.05 - 0.0014 - 0.001)
    # 숏은 양의 펀딩을 받는다
    assert bp.net_return("S", 100, 95, 0.001) == pytest.approx(0.05 - 0.0014 + 0.001)


def test_brk_first_day_only():
    low = [100.0] * 60 + [90.0, 85.0]
    c = [101.0] * 60 + [95.0, 88.0]
    h = [102.0] * 62
    assert bp.brk_hit(h, low, c, 60)
    assert not bp.brk_hit(h, low, c, 61)  # 이틀째는 첫날 아님


def test_dump_needs_btc_ok():
    c = [100.0] * 10 + [55.0]
    assert bp.dump_hit(c, 10, -0.02)
    assert not bp.dump_hit(c, 10, -0.15)


def test_pump_trigger_waits_for_drop():
    # 0~9일 100, 10일 170(7일 +70%) → 감시, 11일 160, 12일 140(고점 175의 85% = 148.75 아래) → 숏
    c = [100.0] * 10 + [170.0, 160.0, 140.0, 130.0]
    h = [101.0] * 10 + [175.0, 165.0, 150.0, 135.0]

    def btc(_j):
        return 0.0

    assert bp.pump_trigger(h, c, 10, btc) is None
    assert bp.pump_trigger(h, c, 11, btc) is None
    assert bp.pump_trigger(h, c, 12, btc) == 10
    assert bp.pump_trigger(h, c, 13, btc) is None  # 감시당 한 번


def test_pump_needs_btc_flat():
    c = [100.0] * 10 + [170.0, 140.0]
    h = [101.0] * 10 + [175.0, 150.0]
    assert bp.pump_trigger(h, c, 11, lambda _j: 0.2) is None


def test_netflow_z_spike():
    vals = [float((i * 37) % 11 - 5) for i in range(200)] + [500.0]
    z = bp.netflow_z(vals)
    assert z[-1] is not None and z[-1] > 2
    assert z[0] is None


def _daily(start_ms, closes, highs=None, lows=None, qv=1e9):
    out = []
    for k, c in enumerate(closes):
        h = highs[k] if highs else c * 1.01
        low = lows[k] if lows else c * 0.99
        out.append([start_ms + k * D, c, h, low, c, qv])
    return out


def test_scan_day_dump_signal_and_universe_rank():
    d0 = bp.day_ms(date(2026, 6, 1))
    n = 150
    btc = _daily(d0, [100.0] * n)
    alt = _daily(d0, [10.0] * (n - 1) + [5.0], qv=2e8)
    daily = {"BTCUSDT": btc, "AAAUSDT": alt}
    onboard = {"BTCUSDT": 0, "AAAUSDT": 0}
    last = date(2026, 6, 1) + timedelta(days=n - 1)
    new = bp.scan_day(last, daily, onboard, [])
    rules = {(t["rule"], t["sym"]) for t in new}
    assert ("DUMP", "AAAUSDT") in rules
    t = next(t for t in new if t["rule"] == "DUMP")
    assert t["entry_ms"] == bp.day_ms(last) + D
    # 이미 들고 있으면 다시 안 들어감
    held = [{"rule": "DUMP", "sym": "AAAUSDT", "result": None}]
    assert not [x for x in bp.scan_day(last, daily, onboard, held) if x["rule"] == "DUMP"]
    # 상장 90일 안 된 종목은 범위 밖
    onboard["AAAUSDT"] = bp.day_ms(last) - 10 * D
    assert not bp.scan_day(last, daily, onboard, [])


def test_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_BARRIER_PAPER_DIR", str(tmp_path))
    monkeypatch.setattr(bp, "START", date(2026, 11, 20))
    now = bp.day_ms(date(2026, 11, 22)) + 5 * H
    d0 = bp.day_ms(date(2026, 11, 22)) - 149 * D
    btc = _daily(d0, [100.0] * 150)
    alt = _daily(d0, [10.0] * 148 + [5.0, 5.0], qv=2e8)  # 11/21 마감에 급락 → 11/22 00시 진입

    def fake_get(url):
        if "exchangeInfo" in url:
            return {
                "symbols": [
                    {
                        "symbol": s,
                        "contractType": "PERPETUAL",
                        "quoteAsset": "USDT",
                        "status": "TRADING",
                        "underlyingType": "COIN",
                        "baseAsset": s[:-4],
                        "onboardDate": 0,
                    }
                    for s in ("BTCUSDT", "AAAUSDT")
                ]
            }
        if "interval=1d" in url:
            src = btc if "BTCUSDT" in url else alt
            return [[r[0], r[1], r[2], r[3], r[4], 0, r[0] + D - 1, r[5]] for r in src]
        if "interval=1h" in url:
            t0 = bp.day_ms(date(2026, 11, 22))
            px = [5.0, 5.1, 5.3]  # 3번째 봉 고가가 +5% 넘음
            return [
                [t0 + k * H, p, p * 1.01, p * 0.995, p, 0, t0 + (k + 1) * H - 1]
                for k, p in enumerate(px)
            ]
        if "fundingRate" in url:
            return [{"fundingTime": bp.day_ms(date(2026, 11, 22)) + 1, "fundingRate": "0.0001"}]
        if "coinmetrics" in url:
            return {"data": []}
        raise AssertionError(url)

    out = bp.run(get=fake_get, now_ms=now)
    assert out["trades"] >= 1
    trades = bp.load_trades()
    dump = [t for t in trades if t["rule"] == "DUMP"]
    assert dump and dump[0]["result"] == "win" and dump[0]["entry"] == 5.0
    summ = (tmp_path / "summary.json").read_text()
    assert "급락주 반등 롱" in summ
    # 두 번째 실행은 이미 훑은 날을 다시 보지 않음
    out2 = bp.run(get=fake_get, now_ms=now + H)
    assert out2["trades"] == out["trades"]


def test_api_route_pending_and_stale(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone

    from minerva import routes

    monkeypatch.setenv("MINERVA_BARRIER_PAPER_DIR", str(tmp_path))
    assert routes.barrier_paper_view()["status"] == "pending"
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {"status": "ok", "rules": [], "generated_at": datetime.now(timezone.utc).isoformat()}
        )
    )
    out = routes.barrier_paper_view()
    assert out["status"] == "ok" and out["stale"] is False


def test_non_ascii_symbol_is_quoted():
    seen = []

    def fake(url):
        seen.append(url)
        return []

    bp.daily_klines("牛来USDT", get=fake)
    assert seen[0].isascii()
