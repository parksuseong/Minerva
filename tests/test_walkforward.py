from datetime import date

from minerva.strategy import walkforward as wf

DAYS = [date(2026, 1, 2), date(2026, 1, 5), date(2026, 2, 2), date(2026, 2, 3), date(2026, 3, 3)]


def test_month_starts_are_first_trading_days():
    assert wf.month_starts(DAYS) == [date(2026, 1, 2), date(2026, 2, 2), date(2026, 3, 3)]


def test_gate_keeps_entries_only_in_passing_months():
    starts = wf.month_starts(DAYS)
    flags = {d: True for d in DAYS} | {date(2025, 12, 30): True}
    out = wf.gate(flags, starts, lambda m: m.month == 2)
    assert [d for d, f in out.items() if f] == [date(2026, 2, 2), date(2026, 2, 3)]
    assert out[date(2025, 12, 30)] is False  # 첫 판정 전에는 사지 않는다


def _v(trades=40, edge=0.01, p=0.001):
    return {
        "label": "검증됨",
        "trades": trades,
        "edge": edge,
        "edge_in": edge,
        "edge_out": edge,
        "p_value": p,
    }


def test_passed_by_month_applies_cost_and_fdr():
    m = date(2026, 2, 2)
    cache = {
        "A.KS|2026-02-02": {"long": _v()},
        "B.KS|2026-02-02": {"long": _v(edge=0.002)},  # 비용 0.3% 빼면 우위 없음
        "C.KS|2026-02-02": {"long": _v(p=0.9)},  # 다중 검정 미통과
        "D.KS|2026-02-02": None,  # 봉이 모자라 판정 없음
        "^KS11|2026-02-02": {"long": _v(), "short": _v(edge=0.0005)},
    }
    out = wf.passed_by_month(cache, ["A.KS", "B.KS", "C.KS", "D.KS"], [m])[m]
    assert list(out["stocks"]) == ["A.KS"]
    assert out["index_long"] is True and out["index_short"] is False


def test_short_pvalue_detects_timed_entries():
    from datetime import timedelta

    from minerva.strategy.indicators import Bar
    from minerva.strategy.short import ShortRule, short_timing_pvalue

    # 10일마다 고점 다음 날 급락: 그 고점에서 숏한 규칙은 무작위보다 낫다
    bars, flags, px = [], [], 100.0
    for i in range(400):
        top = i % 10 == 5
        nxt = px * (0.95 if i % 10 == 6 else 1.006)
        bars.append(
            Bar(date(2020, 1, 1) + timedelta(days=i), px, max(px, nxt), min(px, nxt), nxt, 1)
        )
        flags.append(top)
        px = nxt
    assert short_timing_pvalue(bars, flags, ShortRule(hold=2, stop=0.5), n_perm=100) < 0.05
