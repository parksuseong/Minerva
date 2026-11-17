"""gose_quant(퀀트 기법 Q1~Q6) 점검 — 작은 합성 일봉, 네트워크 없음."""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

import pytest

from minerva import gose
from minerva import gose_quant as Q


def make_bars(closes, start=date(2015, 1, 5), spread=0.01):
    """평일만, 시가 = 전날 종가, 고저는 시가·종가를 spread만큼 감싼다."""
    out, d, prev = [], start, closes[0]
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append(
            (d.isoformat(), prev, max(prev, c) * (1 + spread), min(prev, c) * (1 - spread), c)
        )
        prev = c
        d += timedelta(days=1)
    return out


def walk(n=900, seed=1, drift=0.0005, vol=0.02):
    rng = random.Random(seed)
    c, out = 100.0, []
    for _ in range(n):
        c *= 1 + drift + rng.gauss(0, vol)
        out.append(c)
    return out


def wave(n=900, period=120, amp=0.3, drift=0.0008):
    return [
        100 * (1 + amp * math.sin(2 * math.pi * i / period)) * (1 + drift) ** i for i in range(n)
    ]


# ---------------------------------------------------------------- 지표


def test_prior_windows_exclude_today():
    v = [1, 5, 2, 3, 9, 4]
    assert Q.prior_max(v, 3) == [None, None, None, 5, 5, 9]
    assert Q.prior_min(v, 3) == [None, None, None, 1, 2, 2]


def test_momentum_skips_last_month():
    closes = [100.0] * 300
    closes[300 - 1 - Q.LOOKBACK] = 50.0  # 252봉 전
    closes[300 - 1 - Q.SKIP] = 75.0  # 21봉 전
    closes[-1] = 1.0  # 최근 달은 안 본다
    assert Q.momentum(closes, 299) == pytest.approx(0.5)
    assert Q.momentum(closes, 100) is None


def test_month_end_is_calendar_only():
    dates = ["2026-10-29", "2026-10-30", "2026-11-02", "2026-11-27", "2026-11-30"]
    assert [Q.month_end(dates, i) for i in range(5)] == [False, True, False, False, True]
    # 말일 휴장(마지막 평일 12/31 없음) → 새 달 첫 봉에서 늦게 판단
    kr = ["2025-12-29", "2025-12-30", "2026-01-02", "2026-01-05"]
    assert [Q.month_end(kr, i) for i in range(4)] == [False, False, True, False]
    # 봉을 더 붙여도 지난 판단이 바뀌지 않는다(다음 봉 날짜를 보지 않음)
    for days in (dates, kr):
        for k in range(1, len(days) + 1):
            assert [Q.month_end(days[:k], i) for i in range(k)] == [
                Q.month_end(days, i) for i in range(k)
            ]


# ---------------------------------------------------------------- 신호


def _ctx_at(sid, closes, i, **over):
    ctx = Q.prepare(make_bars(closes))
    for k, v in over.items():
        ctx[k] = v
    return Q.entry(sid, ctx, i), Q.exit_(sid, ctx, i)


def test_q1_turtle_breakout_and_exit():
    closes = [100.0] * 60 + [103.0]  # 직전 55일 고가 101(spread 1%) 위로 마감
    bars = make_bars(closes)
    ctx = Q.prepare(bars)
    assert Q.entry("Q1", ctx, 60) == "55일 고가 돌파"
    assert Q.entry("Q1", ctx, 59) is None
    closes = [100.0] * 60 + [97.0]  # 직전 20일 저가 99 아래
    ctx = Q.prepare(make_bars(closes))
    assert Q.exit_("Q1", ctx, 60) == "20일 저가 이탈"
    res = Q.simulate("Q1", make_bars([100.0] * 60 + [103.0, 104.0] + [104.0] * 25 + [90.0, 91.0]))
    (t,) = res["trades"]
    assert t["signal_date"] == make_bars([1] * 61)[60][0] and t["entry_i"] == 61
    assert t["reason_out"] == "20일 저가 이탈" and t["exit_i"] == 88 and t["exit_px"] == 90.0


def test_q2_golden_and_death_cross():
    n = 260
    sma50 = [None] * n
    sma200 = [None] * n
    sma50[249], sma200[249] = 99.0, 100.0
    sma50[250], sma200[250] = 101.0, 100.0
    sma50[251], sma200[251] = 102.0, 100.0
    sma50[252], sma200[252] = 99.0, 100.0
    ctx = Q.prepare(make_bars([100.0] * n))
    ctx.update(sma50=sma50, sma200=sma200)
    assert Q.entry("Q2", ctx, 250) == "골든 크로스"
    assert Q.entry("Q2", ctx, 251) is None  # 위에 머무는 건 교차가 아님
    assert Q.exit_("Q2", ctx, 252) == "데드 크로스"
    assert Q.exit_("Q2", ctx, 251) is None


def test_q3_connors_rsi2():
    closes = [100 + i * 0.1 for i in range(220)] + [119.0, 117.0, 115.0]
    ctx = Q.prepare(make_bars(closes))
    i = len(closes) - 1
    assert ctx["rsi2"][i] < 10 and closes[i] > ctx["sma200"][i]
    assert Q.entry("Q3", ctx, i) == "SMA200 위 · RSI2 < 10"
    ctx2 = Q.prepare(make_bars(closes + [130.0]))
    assert Q.exit_("Q3", ctx2, i + 1) == "종가 > SMA5"
    assert Q.exit_("Q3", ctx2, i) is None
    # SMA200 아래면 진입 없음
    down = [200 - i * 0.3 for i in range(220)] + [130.0, 125.0]
    ctx3 = Q.prepare(make_bars(down))
    assert Q.entry("Q3", ctx3, len(down) - 1) is None


def test_q4_bollinger_reversion():
    closes = [100 + i * 0.05 + (0.5 if i % 2 else -0.5) for i in range(240)]
    closes += [closes[-1] - 2.5]
    ctx = Q.prepare(make_bars(closes))
    i = len(closes) - 1
    assert closes[i] < ctx["bb_lo"][i] and closes[i] > ctx["sma200"][i]
    assert Q.entry("Q4", ctx, i) == "SMA200 위 · 볼린저 하단 아래"
    ctx2 = Q.prepare(make_bars(closes + [ctx["sma20"][i] + 1]))
    assert Q.exit_("Q4", ctx2, i + 1) == "중심선(SMA20) 회복"


def test_q4_band_uses_population_std():
    closes = [1.0, 3.0] * 10
    ctx = Q.prepare(make_bars(closes))
    assert ctx["sma20"][19] == pytest.approx(2.0)
    assert ctx["bb_lo"][19] == pytest.approx(0.0)  # 표준편차 1(모집단)


def test_q5_tsmom_only_at_month_end():
    up = [100 * 1.002**i for i in range(400)]
    bars = make_bars(up)
    ctx = Q.prepare(bars)
    ends = [i for i in range(300, 400) if Q.month_end(ctx["dates"], i)]
    assert ends
    for i in range(300, 400):
        got = Q.entry("Q5", ctx, i)
        assert (got is not None) == (i in ends)
    res = Q.simulate("Q5", bars)
    assert res["open"] is not None and Q.month_end(
        ctx["dates"], ctx["dates"].index(res["open"]["signal_date"])
    )
    down = [100 * 0.998**i for i in range(400)]
    dctx = Q.prepare(make_bars(down))
    assert all(Q.entry("Q5", dctx, i) is None for i in range(400))
    assert any(Q.exit_("Q5", dctx, i) for i in ends)


# ---------------------------------------------------------------- Q6


def _series(n=600):
    return {
        "UP": make_bars([100 * 1.003**i for i in range(n)]),
        "MID": make_bars([100 * 1.001**i for i in range(n)]),
        "DOWN": make_bars([100 * 0.999**i for i in range(n)]),
        "LATE": make_bars([100 * 1.004**i for i in range(n - 300)], start=date(2016, 3, 1)),
    }


def test_q6_ranks_and_holds_top_n():
    res = Q.momentum_portfolio(_series(), top=2)
    assert res["rebalances"]
    assert all(len(top) <= 2 for _, top in res["rebalances"])
    assert [t for _, t in res["rebalances"][:5]] == [[]] * 5  # 252봉 전에는 순위 없음
    top = next(t for _, t in res["rebalances"] if t)
    assert top == ["UP", "MID"]
    # LATE가 252봉이 차면 MID를 밀어낸다(MID는 다음 시가에 팔고, LATE는 그 뒤에 산다)
    swap = next(k for k, (_, t) in enumerate(res["rebalances"]) if "LATE" in t)
    assert res["rebalances"][swap][1] == ["LATE", "UP"]
    mid = next(t for t in res["trades"] if t["sym"] == "MID")
    late = next(t for t in res["trades"] + res["open"] if t["sym"] == "LATE")
    assert mid["reason_out"] == "모멘텀 상위 10 밖" and mid["exit_date"] <= late["entry_date"]
    held = {t["sym"] for t in res["open"]}
    assert held == {"LATE", "UP"}


def test_q6_rebalance_fills_next_open_and_keeps_slots():
    series = _series()
    res = Q.momentum_portfolio(series, top=2)
    for t in res["trades"] + res["open"]:
        b = series[t["sym"]]
        i = t["entry_i"]
        assert b[i][0] > t["signal_date"] and t["fills"][0][1] == b[i][1]  # 다음 시가
    # 계좌로 돌려도 자리 때문에 건너뛰는 일이 없다(매도가 먼저)
    rows = [gose._trade_row(t["sym"], t, False) for t in res["trades"] + res["open"]]
    closes = {s: {x[0]: x[4] for x in b} for s, b in series.items()}
    cal = sorted({d for c in closes.values() for d in c})
    acct = gose.run_account(rows, closes, cal, keep=None)
    assert acct["skipped"] == 0 and len(acct["open"]) == 2


def test_q6_pending_on_last_month_end():
    series = _series()
    cal = sorted({b[0] for s in series.values() for b in s})
    k = max(i for i in range(len(cal)) if Q.month_end(cal, i))
    cut = {s: [b for b in bars if b[0] <= cal[k]] for s, bars in series.items()}
    res = Q.momentum_portfolio(cut, top=2)
    assert res["rebalances"][-1][0] == cal[k]
    assert all(p[1] == cal[k] for p in res["pending"])


# ---------------------------------------------------------------- 미래 봉을 쓰지 않음


@pytest.mark.parametrize("sid", Q.PER_SYMBOL)
def test_appending_future_bar_keeps_past_signals(sid):
    """봉을 하나 더 붙여도 지난 봉의 진입·청산 판단과 거래가 그대로다(그날까지 자료만 씀)."""
    closes = [a * b / 100 for a, b in zip(walk(700, seed=7), wave(700, period=400, amp=0.35))]
    bars = make_bars(closes)
    assert any(t["exit_i"] > 300 for t in Q.simulate(sid, bars)["trades"])  # 시험할 거래가 있다
    for k in range(300, 700, 5):
        short = Q.prepare(bars[:k])
        longer = Q.prepare(bars[: k + 1])
        for i in range(k - 40, k):
            assert Q.entry(sid, short, i) == Q.entry(sid, longer, i), (sid, k, i)
            assert Q.exit_(sid, short, i) == Q.exit_(sid, longer, i), (sid, k, i)
        a = Q.simulate(sid, bars[:k])
        b = Q.simulate(sid, bars[: k + 1])
        done = [t for t in b["trades"] if t["exit_i"] < k]
        assert [(t["entry_i"], t["exit_i"]) for t in a["trades"]] == [
            (t["entry_i"], t["exit_i"]) for t in done
        ]
        # 마지막 봉의 대기 신호는 다음 봉에서 그대로 체결된다
        for sd, kind, _, _ in a["pending"]:
            if kind == "entry":
                got = b["open"] or next(t for t in b["trades"] if t["entry_i"] == k)
                assert got["signal_date"] == sd and got["entry_i"] == k
            else:
                assert any(t["exit_i"] == k for t in b["trades"])


def test_q6_appending_future_bar_keeps_past_rebalances():
    series = _series()
    cal = sorted({b[0] for s in series.values() for b in s})
    for k in range(300, len(cal) - 1, 23):
        cut = {s: [b for b in bars if b[0] <= cal[k]] for s, bars in series.items()}
        more = {s: [b for b in bars if b[0] <= cal[k + 1]] for s, bars in series.items()}
        a = Q.momentum_portfolio(cut, top=2)
        b = Q.momentum_portfolio(more, top=2)
        assert a["rebalances"] == b["rebalances"][: len(a["rebalances"])]
        done = [(t["sym"], t["entry_i"]) for t in a["trades"] + a["open"]]
        later = [(t["sym"], t["entry_i"]) for t in b["trades"] + b["open"]]
        assert set(done) <= set(later)


# ---------------------------------------------------------------- 가상 계좌·백테스트


def test_build_paper_has_quant_accounts(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_GOSE_DIR", str(tmp_path / "gose"))
    d = tmp_path / "gose"
    series = _series(700)
    for s, b in series.items():
        gose._write(d / "bars" / f"{s}.json", {"bars": [list(x) for x in b]})
    end = series["UP"][-1][0]
    start = series["UP"][400][0]
    paper = gose.build_paper(list(series), "^VIX", start, end)
    by = {a["key"]: a for a in paper["accounts"]}
    for sid in Q.STRATEGIES:
        a = by[sid]
        assert a["group"] == "quant" and a["label"] == Q.LABEL and a["curve"][0][0] >= start
    q6 = by["Q6"]
    assert q6["open"] and all(p["entry_date"] > start for p in q6["open"])


def test_quant_backtest_rows_and_baselines():
    series = {s: b for s, b in _series(1500).items()}
    out = gose.quant_backtest(series, years=2, runs=5)
    assert out["status"] == "ok" and [r["key"] for r in out["rows"]] == list(Q.STRATEGIES)
    assert out["buy_hold"]["symbols"] == 4 and out["buy_hold"]["cagr"] is not None
    for r in out["rows"]:
        assert r["random"]["runs"] == 5
        assert {"cagr", "mdd", "trades", "win_rate", "avg_hold_days"} <= set(r)
    assert out["window"][0] < out["window"][1]
    # 같은 씨앗이면 같은 결과
    again = gose.quant_backtest(series, years=2, runs=5)
    assert [r["random"] for r in again["rows"]] == [r["random"] for r in out["rows"]]
