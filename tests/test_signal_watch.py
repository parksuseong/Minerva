"""signal_watch: 확정 봉 판정·조건·새로 뜬 순간만 알림(네트워크 없이)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from minerva import signal_watch as sw
from minerva.strategy.indicators import Bar


def _days(start: date, closes: list[float]) -> list[Bar]:
    out, d = [], start
    for c in closes:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        out.append(Bar(d, c, c * 1.01, c * 0.99, c))
        d += timedelta(days=1)
    return out


def test_partial_daily_week_month_are_dropped():
    bars = _days(date(2026, 11, 2), [100.0] * 15)  # 11/2(월)~11/20(금)
    assert bars[-1].day == date(2026, 11, 20)
    during = datetime(2026, 11, 20, 18, 0, tzinfo=timezone.utc)  # 장중
    after = datetime(2026, 11, 20, 22, 0, tzinfo=timezone.utc)  # 마감 뒤
    assert sw.confirmed_daily(bars, during)[-1].day == date(2026, 11, 19)
    assert sw.confirmed_daily(bars, after)[-1].day == date(2026, 11, 20)
    # 금요일 마감 뒤면 그 주는 확정, 장중이면 지난주까지
    assert sw.frame_bars(bars, "W", after)[-1].day == date(2026, 11, 20)
    assert sw.frame_bars(bars, "W", during)[-1].day == date(2026, 11, 13)
    # 11월은 아직 안 끝남 → 10월까지(여기선 자료가 11월뿐이라 빈 목록)
    assert sw.frame_bars(bars, "M", after) == []


def test_month_confirmed_on_last_weekday():
    bars = _days(date(2026, 11, 2), [100.0] * 21)  # 11/30(월)까지
    assert bars[-1].day == date(2026, 11, 30)
    now = datetime(2026, 11, 30, 22, 0, tzinfo=timezone.utc)
    assert sw.frame_bars(bars, "M", now)[-1].day == date(2026, 11, 30)


def _vshape() -> list[float]:
    # 흔들리며 오르내리는 고정 난수 경로: 과매도 뒤 3단 돌파가 어딘가에서 난다
    import random

    rng = random.Random(3)
    out, p = [], 100.0
    for _ in range(400):
        p *= 1 + rng.gauss(0, 0.02)
        out.append(p)
    return out


def test_long3_condition_and_first_run_silent(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_SIGNAL_WATCH_DIR", str(tmp_path))
    closes = _vshape()
    bars = _days(date(2026, 8, 3), closes)
    hits = [i for i in range(30, len(bars)) if sw.condition("long3", bars[: i + 1])[0]]
    assert hits, "V자 반등에서 3단 돌파가 한 번은 나야 한다"
    k = hits[0]
    sent = []

    def fake_fetch(sym, rng, ttl=None):
        return bars[:k] if state["n"] == 0 else bars[: k + 1]

    state = {"n": 0}
    monkeypatch.setattr(sw, "RULES", (("SPY_D3", "SPY", "D", "long3", "설명"),))
    now = datetime.combine(bars[k].day, datetime.min.time(), timezone.utc) + timedelta(days=2)
    out = sw.run(now=now, fetch=fake_fetch, send=lambda t: sent.append(t) or True)
    assert out["first_run"] and not sent  # 첫 실행은 기억만
    state["n"] = 1
    out = sw.run(now=now, fetch=fake_fetch, send=lambda t: sent.append(t) or True)
    assert out["sent"] == ["SPY_D3"] and len(sent) == 1 and "롱 신호" in sent[0]
    # 같은 봉으로 다시 돌면 또 보내지 않음
    sw.run(now=now, fetch=fake_fetch, send=lambda t: sent.append(t) or True)
    assert len(sent) == 1


def test_failed_send_retries_next_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MINERVA_SIGNAL_WATCH_DIR", str(tmp_path))
    closes = _vshape()
    bars = _days(date(2026, 8, 3), closes)
    k = next(i for i in range(30, len(bars)) if sw.condition("long3", bars[: i + 1])[0])
    monkeypatch.setattr(sw, "RULES", (("SPY_D3", "SPY", "D", "long3", "설명"),))
    now = datetime.combine(bars[k].day, datetime.min.time(), timezone.utc) + timedelta(days=2)
    cur = {"bars": bars[:k]}
    fetch = lambda s, r, ttl=None: cur["bars"]  # noqa: E731
    sw.run(now=now, fetch=fetch, send=lambda t: True)
    cur["bars"] = bars[: k + 1]
    assert sw.run(now=now, fetch=fetch, send=lambda t: False)["sent"] == []
    got = []
    assert sw.run(now=now, fetch=fetch, send=lambda t: got.append(t) or True)["sent"] == ["SPY_D3"]


def test_vix_bb_return():
    closes = [20.0 + (i % 3) * 0.3 for i in range(30)] + [40.0, 22.0]
    bars = _days(date(2026, 1, 5), closes)
    on, info = sw.condition("vixbb", bars)
    assert on and info["bb_upper"] > 22.0
    on2, _ = sw.condition("vixbb", bars[:-1])
    assert not on2
