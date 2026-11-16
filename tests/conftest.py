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


def pytest_sessionstart(session):
    from minerva import btc_trade

    assert Path(btc_trade.DIR).resolve() != _REAL, "시험이 실전 엔진 폴더를 가리킨다"
