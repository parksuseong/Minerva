from datetime import date, datetime, timedelta

from minerva import kr_align as K
from minerva.strategy.indicators import Bar


def bars_from(closes, start=date(2025, 1, 1)):
    return [Bar(start + timedelta(days=i), c, c * 1.01, c * 0.99, c) for i, c in enumerate(closes)]


def test_align():
    assert K.align([4, 3, 2, 1]) == "정배"
    assert K.align([1, 2, 3, 4]) == "역배"
    assert K.align([2, 3, 1, 4]) == "혼합"
    assert K.align([None, 1, 2, 3]) is None


def test_analyze_down_then_up_turns_forward():
    closes = [200 - i for i in range(150)] + [50 + 3 * i for i in range(120)]
    r = K.analyze(bars_from(closes))
    assert r["state"] == "정배"
    assert r["last_reverse"] is not None
    assert r["pairs_up"] == 3


def test_confirmed_drops_today_before_close():
    b = bars_from([1.0] * 3, start=date(2026, 11, 26))
    now = datetime(2026, 11, 28, 10, 0, tzinfo=K.KST)
    assert len(K.confirmed(b, now)) == 2
    assert len(K.confirmed(b, now.replace(hour=16))) == 3


def test_flips_and_notify_once(tmp_path):
    row = {
        "code": "000001",
        "name": "가",
        "state": "정배",
        "last_reverse": "2026-08-01",
        "date": "2026-11-27",
        "close": 1000.0,
        "state_since": "2026-11-27",
        "stoch": 70.0,
        "caution": False,
        "buy_zone": False,
    }
    plain = dict(row, code="000002", last_reverse=None)
    data = {"rows": [row, plain]}
    sent = []
    st = tmp_path / "a.json"
    out = K.notify(data, st, send=lambda t: sent.append(t) or True)
    assert [r["code"] for r in out] == ["000001"] and len(sent) == 1
    assert K.notify(data, st, send=lambda t: sent.append(t) or True) == []
    # 정배가 풀렸다가 다시 정배가 되면 다시 알린다
    K.notify({"rows": [dict(row, state="혼합")]}, st, send=lambda t: True)
    assert len(K.notify(data, st, send=lambda t: sent.append(t) or True)) == 1


def test_notify_send_fail_keeps_state(tmp_path):
    row = {
        "code": "1",
        "name": "가",
        "state": "정배",
        "last_reverse": "x",
        "date": "d",
        "close": 1.0,
        "state_since": "d",
        "stoch": 1.0,
        "caution": False,
        "buy_zone": False,
    }
    st = tmp_path / "a.json"
    assert K.notify({"rows": [row]}, st, send=lambda t: False) == []
    assert not st.exists()


def test_fetch_falls_back_to_kq():
    def fb(t):
        if t.endswith(".KS"):
            raise OSError("404")
        return bars_from([1.0] * 130)

    t, b = K.fetch("123456", fb)
    assert t == "123456.KQ" and len(b) == 130
