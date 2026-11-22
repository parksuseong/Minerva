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


# ---------------------------------------------------------------- 규칙별 장벽·자동 교체·주문흐름


def test_breakeven_per_geometry():
    assert bp.breakeven(0.05, 0.03) == pytest.approx(0.3925)
    assert bp.breakeven(0.03, 0.03) == pytest.approx(0.5233, abs=1e-4)
    assert bp.breakeven(0.06, 0.03) == pytest.approx(0.3489, abs=1e-4)


def test_wilson_upper():
    assert bp.wilson_upper(0, 0) == 1.0
    # 30건 중 9승(30%) → 95% 상한 약 47.9%
    assert bp.wilson_upper(9, 30) == pytest.approx(0.479, abs=0.002)
    assert bp.wilson_upper(30, 30) == pytest.approx(1.0)


def _closed(results, nets, exit_ms):
    return [{"result": r, "net": n, "exit_ms": exit_ms} for r, n in zip(results, nets)]


def test_retire_reason_rules():
    now = bp.day_ms(date(2026, 12, 31))
    old = now - 30 * D
    be = bp.breakeven(0.03, 0.03)
    # 30건 미만이고 최근 손실 작으면 그대로
    assert bp.retire_reason(_closed(["loss"] * 29, [-0.031] * 29, old), now, be) is None
    # 30건 · 승률 30% → 상한 < 52.3%
    tr = _closed(["win"] * 9 + ["loss"] * 21, [0.03] * 9 + [-0.03] * 21, old)
    assert "손익분기" in bp.retire_reason(tr, now, be)
    # 승률은 괜찮지만 거래당 평균 < 0
    tr = _closed(["win"] * 20 + ["loss"] * 10, [0.001] * 20 + [-0.03] * 10, old)
    assert "평균" in bp.retire_reason(tr, now, bp.breakeven(0.05, 0.03))
    # 건수 무관: 최근 14일 합 < −15%
    tr = _closed(["loss"] * 6, [-0.031] * 6, now - 2 * D)
    assert "최근 14일" in bp.retire_reason(tr, now, be)
    # 같은 손실이 14일 전이면 아직
    assert bp.retire_reason(_closed(["loss"] * 6, [-0.031] * 6, old), now, be) is None


def test_auto_retire_logs_and_activates_queue(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_BARRIER_PAPER_DIR", str(tmp_path))
    monkeypatch.setattr(bp, "QUEUE", ["QQQ"])
    monkeypatch.setitem(bp.RULES, "QQQ", {**bp.RULES["FLW"], "name": "대기 후보"})
    now = bp.day_ms(date(2026, 11, 25)) + 5 * H
    st: dict = {}
    assert not bp.rule_open("QQQ", st, now)  # 대기 후보는 꺼져 있음
    trades = [
        {
            "rule": "DUMP",
            "sym": f"S{k}USDT",
            "side": "L",
            "entry_ms": now - 5 * H,
            "entry": 1.0,
            "exit": 0.97,
            "result": "loss",
            "net": -0.031,
            "exit_ms": now - H,
        }
        for k in range(6)
    ]
    out = bp.auto_retire(trades, st, now)
    assert [x["rule"] for x in out] == ["DUMP"]
    assert out[0]["auto"] is True and out[0]["replaced_by"] == "QQQ"
    assert not bp.rule_open("DUMP", st, now + H)
    assert bp.rule_open("QQQ", st, bp.day_ms(date(2026, 11, 26)))
    assert not bp.rule_open("QQQ", st, bp.day_ms(date(2026, 11, 25)))
    log_rows = (tmp_path / "retire.jsonl").read_text().strip().splitlines()
    assert len(log_rows) == 1 and "DUMP" in log_rows[0]
    # 두 번 교체하지 않음
    assert bp.auto_retire(trades, st, now + H) == []
    summ = bp.build_summary(trades, st, now)
    keys = {r["key"]: r for r in summ["rules"]}
    assert keys["DUMP"]["status"] == "retired" and keys["DUMP"]["retired"]["auto"] is True
    assert any(r["key"] == "DUMP" and r["auto"] for r in summ["retired"])
    assert summ["queue"] == []


def test_fixed_retirement_brk():
    assert bp.rule_open("BRK", {}, bp.day_ms(date(2026, 11, 21)))
    assert not bp.rule_open("BRK", {}, bp.day_ms(date(2026, 11, 22)))
    assert bp.rule_open("ABS", {}, bp.day_ms(date(2026, 11, 1)))
    assert not bp.rule_open("ABS", {}, bp.day_ms(date(2026, 10, 31)))


def test_flow_series_and_pct():
    i7 = bp.flow_series([100.0] * 10, [60.0] * 10)
    assert i7[5] is None and i7[6] == pytest.approx(0.2)
    vals = [None] * 6 + [k / 200 for k in range(200)]
    vals.append(-1.0)
    assert bp.flow_pct(vals, len(vals) - 1) == 0.0
    assert bp.flow_pct(vals, 50) is None  # 직전 값 120개 미만


def _flow_bars(n_days, closes, tb_ratio):
    """tb_ratio[k] = 그날 테이커 매수 비율."""
    d0 = bp.day_ms(date(2026, 1, 1))
    out = []
    for k in range(n_days):
        c = closes[k]
        q = 1e9
        out.append([d0 + k * D, c, c * 1.01, c * 0.99, c, q, q * tb_ratio[k]])
    return out


def test_flow_sides_abs_and_flw():
    n = 200
    ratio = [0.5 + 0.01 * ((k * 7) % 5 - 2) for k in range(n - 7)] + [0.40] * 7
    bars = _flow_bars(n, [100.0] * n, ratio)
    i7 = bp.flow_series([b[5] for b in bars], [b[6] for b in bars])
    c = [b[4] for b in bars]
    i = n - 7  # 강한 매도 흐름이 들어간 첫날(7일 합에 처음 반영)
    assert bp.flow_pct(i7, i) == 0.0
    assert bp.flow_sides("ABS", c, i7, i) == ["L"]  # 가격은 안 빠짐 → 흡수 롱
    assert bp.flow_sides("ABS", c, i7, i + 1) == []  # 첫날만
    assert bp.flow_sides("FLW", c, i7, i) == []  # 7일 수익률 0 → 추종 숏 아님
    c2 = c[:i] + [95.0] * (n - i)
    assert bp.flow_sides("FLW", c2, i7, i) == ["S"]
    assert bp.flow_sides("ABS", c2, i7, i) == []
    # 매수 흐름 상위 10% + 상승 → 추종 롱, 하락 → 흡수 숏
    up = [0.5 + 0.01 * ((k * 7) % 5 - 2) for k in range(n - 7)] + [0.62] * 7
    i7u = bp.flow_series([1e9] * n, [1e9 * x for x in up])
    assert bp.flow_sides("FLW", c[:i] + [105.0] * (n - i), i7u, i) == ["L"]
    assert bp.flow_sides("ABS", c2, i7u, i) == ["S"]


def test_scan_day_flow_rule_stores_geometry():
    n = 200
    ratio = [0.5 + 0.01 * ((k * 7) % 5 - 2) for k in range(n - 1)] + [0.05]
    alt = _flow_bars(n, [10.0] * n, ratio)
    btc = _flow_bars(n, [100.0] * n, [0.5] * n)
    daily = {"BTCUSDT": btc, "AAAUSDT": alt}
    onboard = {"BTCUSDT": 0, "AAAUSDT": 0}
    last = bp.day_of(alt[-1][0])
    new = bp.scan_day(last, daily, onboard, [], rules={"ABS"})
    assert [(t["rule"], t["side"], t["tp"], t["sl"]) for t in new] == [("ABS", "L", 0.03, 0.03)]
    # 옛 6칸 일봉(테이커 값 없음)은 주문흐름 규칙에서 건너뜀
    old = {k: [b[:6] for b in v] for k, v in daily.items()}
    assert bp.scan_day(last, old, onboard, [], rules={"ABS"}) == []


def test_resolve_uses_trade_geometry():
    t0 = bp.day_ms(date(2026, 11, 2))
    trades = [
        {"rule": "ABS", "sym": "AAAUSDT", "side": "L", "entry_ms": t0, "tp": 0.03, "sl": 0.03}
    ]

    def fake_get(url):
        if "interval=1h" in url:
            return [[t0, 10.0, 10.35, 9.9, 10.3, 0, t0 + H - 1]]
        if "fundingRate" in url:
            return []
        raise AssertionError(url)

    bp.resolve(trades, t0 + 2 * H, get=fake_get)
    assert trades[0]["result"] == "win" and trades[0]["exit"] == pytest.approx(10.3)


def test_old_state_migrates_and_backfills_new_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_BARRIER_PAPER_DIR", str(tmp_path))
    import json as _json

    (tmp_path / "state.json").write_text(_json.dumps({"last_scan": "2026-11-21"}))
    now = bp.day_ms(date(2026, 11, 22)) + 5 * H
    asked = []

    def fake_get(url):
        asked.append(url)
        if "exchangeInfo" in url:
            return {"symbols": []}
        if "coinmetrics" in url:
            return {"data": []}
        raise AssertionError(url)

    bp.run(get=fake_get, now_ms=now)
    st = _json.loads((tmp_path / "state.json").read_text())
    assert st["scanned"]["PUMP"] == "2026-11-21"
    assert st["scanned"]["ABS"] == "2026-11-21"
    assert st["scanned"]["BRK"] == "2026-11-21"
    assert any("exchangeInfo" in u for u in asked)  # 새 규칙 백필 때문에 일봉을 받음
