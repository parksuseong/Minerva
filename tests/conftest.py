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


def pytest_sessionstart(session):
    from minerva import btc_trade

    assert Path(btc_trade.DIR).resolve() != _REAL, "시험이 실전 엔진 폴더를 가리킨다"
