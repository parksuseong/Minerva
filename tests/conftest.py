"""시험이 실전 엔진 기록 폴더(~/.minerva/btc_trade)를 절대 건드리지 않게 한다.

btc_trade.DIR은 불러올 때 MINERVA_BTC_TRADE_DIR로 정해진다. minerva를 불러오기 전에 임시 폴더로
정해 두면, 어떤 시험이 DIR을 바꾸지 않거나(monkeypatch가 풀린 뒤 남은 스레드 포함) 실수해도
쓰기는 임시 폴더로 간다. 2026-11-16 시험 데이터가 실전 state.json을 덮은 사고 뒤에 넣었다.
"""

import os
import tempfile
from pathlib import Path

_REAL = Path("~/.minerva/btc_trade").expanduser().resolve()
os.environ["MINERVA_BTC_TRADE_DIR"] = tempfile.mkdtemp(prefix="minerva-test-btc-")
_TMP = Path(tempfile.mkdtemp(prefix="minerva-test-"))
# 실제 키·텔레그램·계좌 파일도 못 읽게 없는 경로로(시험은 각자 mock을 쓴다)
os.environ["MINERVA_BINANCE_ENV"] = str(_TMP / "no-binance.env")
os.environ["MINERVA_TELEGRAM_ENV"] = str(_TMP / "no-telegram.env")
os.environ["MINERVA_ACCOUNT_DIR"] = str(_TMP / "account")
# 고서 가상매매 폴더(~/.minerva/gose)도 임시로
os.environ["MINERVA_GOSE_DIR"] = str(_TMP / "gose")
# 원전 비공개 자료(~/.minerva/feed)도 임시로
os.environ["MINERVA_FEED_DIR"] = str(_TMP / "feed")
# 성장 후보 폴더(~/.minerva/growth: 목록·재무 캐시·공포 알림 기록)도 임시로
os.environ["MINERVA_GROWTH_DIR"] = str(_TMP / "growth")


def pytest_sessionstart(session):
    from minerva import btc_trade

    assert Path(btc_trade.DIR).resolve() != _REAL, "시험이 실전 엔진 폴더를 가리킨다"


import pytest  # noqa: E402

# 2026-11-17 실전 배율 1배·창 365일로 바꿨다. 예전 시험들은 3배·182일 숫자로 짜여 있어 그대로 두고,
# 실전 값은 test_btc_trade.py의 test_live_leverage_1x_window_365에서 따로 확인한다.
LEGACY_LEVERAGE = 3
LEGACY_WINDOW = 182


@pytest.fixture(autouse=True)
def _legacy_engine_numbers(request, monkeypatch):
    if request.node.get_closest_marker("live_constants"):
        return
    from minerva import btc_trade, btc_universe

    monkeypatch.setattr(btc_trade, "LEVERAGE", LEGACY_LEVERAGE)
    monkeypatch.setattr(btc_universe, "LEVERAGE", LEGACY_LEVERAGE)
    monkeypatch.setattr(btc_universe, "WINDOW_DAYS", LEGACY_WINDOW)


def pytest_configure(config):
    config.addinivalue_line("markers", "live_constants: 실전 상수 그대로 시험")
