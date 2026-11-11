"""종목 분석 > SOXL 탭: SOXL(반도체 3배 ETF) 큰 그림. 보기만 하고 주문과 연결되지 않는다
(사용자 요청 2026-11-11 "soxl 동일한 조건으로").

4시간봉은 바이낸스 SOXLUSDT 무기한 선물(2026-05 상장이라 기록이 짧다). 일·주·월봉은 미국 SOXL ETF
주가(야후 일봉 10년, 주·월은 일봉을 묶음)로 계산한다. 지금 가격은 선물 최근 체결가.
판독 규칙은 비트코인 큰 그림(`btc_outlook`)과 같다.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, time, timedelta, timezone

from minerva.strategy import btc, btc_outlook, soxl_classic
from minerva.strategy.indicators import Bar, resample
from minerva.strategy.report import fetch_yahoo

SYMBOL = "SOXLUSDT"
STOCK = "SOXL"
NOTE = (
    "표시 전용 판독이에요. 일·주·월봉은 미국 SOXL ETF 주가, 4시간봉은 바이낸스 선물이에요. "
    "미국 장이 닫힌 밤·주말에도 선물은 거래돼 월요일 개장 때 갭이 날 수 있어요. "
    "이런 방향 판독의 다음 움직임 적중은 BTC 검증에서 50% 안팎이었어요(확률 아님)."
)


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def fetch(interval: str) -> list[list]:
    """바이낸스 선물 봉(공개 시세, GET만). 비트코인 탭과 같은 주소에 종목만 바꾼다."""
    return _get(btc.URL.format(sym=SYMBOL, iv=interval, n=btc.LIMIT))


def fetch_price() -> float:
    return float(_get(btc.PRICE_URL.format(sym=SYMBOL))["price"])


def _ny_now(now: datetime) -> datetime:
    """뉴욕 시각(서머타임: 3월 둘째 일요일 ~ 11월 첫째 일요일, 미국 동부 시간)."""
    utc = now.astimezone(timezone.utc)
    y = utc.year
    mar = datetime(y, 3, 8, 7, tzinfo=timezone.utc)  # 3월 둘째 일요일 02:00 EST = 07:00 UTC
    start = mar + timedelta(days=(6 - mar.weekday()) % 7)
    nov = datetime(y, 11, 1, 6, tzinfo=timezone.utc)  # 11월 첫째 일요일 02:00 EDT = 06:00 UTC
    end = nov + timedelta(days=(6 - nov.weekday()) % 7)
    return utc + timedelta(hours=-4 if start <= utc < end else -5)


def closed_days(bars: list[Bar], now: datetime) -> list[Bar]:
    """정규장이 아직 안 끝난 오늘 일봉은 뺀다(16:00 뉴욕, 여유 15분)."""
    ny = _ny_now(now)
    if bars and bars[-1].day >= ny.date() and ny.time() < time(16, 15):
        return bars[:-1]
    return bars


def build(now: datetime | None = None) -> dict:
    now = now or datetime.now(btc.KST)
    now_ms = int(now.timestamp() * 1000)
    h4, _ = btc.parse(fetch("4h"), now_ms)
    days = closed_days(fetch_yahoo(STOCK), now)
    raw = {"4h": h4, "1d": days, "1w": resample(days, "W"), "1M": resample(days, "M")}
    price = fetch_price()
    out = btc_outlook.build(raw, price)
    out["note"] = NOTE
    try:
        soxx = closed_days(fetch_yahoo("SOXX"), now)
        same = [b.close for b in days if b.day == soxx[-1].day]
        classic = soxl_classic.build(soxx, same[0] if same else None)
    except (OSError, ValueError, KeyError) as exc:
        classic = {"error": f"SOXX 원문 방식 계산 실패 ({type(exc).__name__})"}
    return {
        "status": "ok",
        "generated_at": now.isoformat(timespec="seconds"),
        "symbol": SYMBOL,
        "outlook": out,
        "classic": classic,
    }
