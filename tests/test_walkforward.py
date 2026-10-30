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
        "^KS11|2026-02-02": {"long": _v()},
    }
    out = wf.passed_by_month(cache, ["A.KS", "B.KS", "C.KS", "D.KS"], [m])[m]
    assert list(out["stocks"]) == ["A.KS"]
    assert out["index_long"] is True and "index_short" not in out


def test_hysteresis_keeps_listed_stocks_on_looser_bar():
    ms = [date(2026, 1, 2), date(2026, 2, 2), date(2026, 3, 3), date(2026, 4, 1)]
    cache = {}
    for i, m in enumerate(ms):
        k = m.isoformat()
        cache[f"A.KS|{k}"] = {"long": _v(p=0.001 if i == 0 else 0.3)}
        cache[f"B.KS|{k}"] = {"long": _v(p=0.3)}  # 처음부터 경계: 새로 넣지 않는다

    def lists(**kw):
        out = wf.passed_by_month(cache, ["A.KS", "B.KS"], ms, **kw)
        return [list(v["stocks"]) for v in out.values()]

    assert lists() == [["A.KS"], [], [], []]
    assert lists(stay_q=0.4) == [["A.KS"]] * 4
    assert lists(sticky=2) == [["A.KS"], ["A.KS"], ["A.KS"], []]
