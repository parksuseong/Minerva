from datetime import date, timedelta

from minerva.strategy import patterns as pt
from minerva.strategy.indicators import Bar


def _bars(closes):
    d0 = date(2024, 1, 1)
    return [Bar(d0 + timedelta(days=i), c, c, c, c, 0) for i, c in enumerate(closes)]


def test_zigzag_confirms_only_after_reversal():
    closes = [100, 120, 115, 107, 106, 125]
    piv = pt.zigzag(closes)
    assert piv[0] == pt.Pivot("L", 100, 0, 1)
    assert piv[1] == pt.Pivot("H", 120, 1, 3)  # 120의 −10%는 108: 3번째 봉에 확정
    assert piv[2] == pt.Pivot("L", 106, 4, 5)


def test_fib_zone_not_known_before_high_is_confirmed():
    closes = [100, 200, 170, 150, 140]
    inside, first = pt.fib_zones(_bars(closes))
    # 200 고점은 170에서 확정. 구간은 138.2~161.8
    assert inside == [False, False, False, True, True]
    assert first == [False, False, False, True, False]


def test_fib_ignores_future_prices():
    a, _ = pt.fib_zones(_bars([100, 200, 170, 150]))
    b, _ = pt.fib_zones(_bars([100, 200, 170, 150, 300, 10]))
    assert a == b[:4]


def test_wave3_breakout_needs_higher_low_and_new_high():
    closes = [100, 150, 125, 140, 151, 160]
    # 되돌림 50%, 저점 상승, 151이 150을 처음 넘는 날
    assert pt.wave3_breakouts(_bars(closes)) == [False] * 4 + [True, False]


def test_wave3_rejects_lower_low():
    assert not any(pt.wave3_breakouts(_bars([100, 150, 90, 120, 151, 160])))


def _piv(prices):
    kinds = "LHLHLH"
    return [pt.Pivot(k, p, i, i) for i, (k, p) in enumerate(zip(kinds, prices))]


def test_impulse_rules():
    assert pt.impulse_done(_piv([100, 120, 110, 150, 130, 160]))
    assert not pt.impulse_done(_piv([100, 120, 110, 150, 115, 160]))  # 4파가 1파 고점 아래
    assert not pt.impulse_done(_piv([100, 130, 120, 135, 125, 150]))  # 3파가 가장 짧다


def test_random_removal_p_small_when_keeping_best():
    vals = [0.0] * 90 + [1.0] * 10
    assert pt.random_removal_p(vals, [1.0] * 10) < 0.01
