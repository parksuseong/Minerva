"""종목 분석 > SOXL·NVDA·ETH 탭: 무기한 선물 큰 그림. 보기만 하고 주문과 연결되지 않는다
(사용자 요청 2026-11-11 SOXL "동일한 조건으로", 같은 날 NVDA 추가).

4시간봉은 바이낸스 무기한 선물(2026년 상장이라 기록이 짧다). 일·주·월봉은 미국 주가(야후 일봉 10년,
주·월은 일봉을 묶음)로 계산한다. 지금 가격은 선물 최근 체결가. 판독 규칙은 비트코인 큰 그림과 같다.
일봉 지표 분석은 `base` 종목 일봉(SOXL은 1배 SOXX, NVDA는 자기 자신)으로 하고 `lev`배로 환산한다.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, time, timedelta, timezone

from minerva.strategy import btc, btc_outlook, soxl_classic
from minerva.strategy.indicators import Bar, resample
from minerva.strategy.report import fetch_yahoo

# 선물 종목 → 미국 주가 종목, 일봉 지표 분석 기준 종목과 배수
US: dict[str, dict] = {
    "SOXLUSDT": {"stock": "SOXL", "base": "SOXX", "lev": 3},
    "NVDAUSDT": {"stock": "NVDA", "base": "NVDA", "lev": 1},
    # 코인은 일봉도 바이낸스 선물(마감 봉만, 최근 1000일). 2026-11-11 사용자 "이더리움도 추가"
    "ETHUSDT": {"stock": "ETH", "base": "ETH", "lev": 1, "crypto": True},
}
SYMBOL = "SOXLUSDT"  # 기본


def note(stock: str, crypto: bool = False) -> str:
    if crypto:
        return (
            f"표시 전용 판독이에요. 4시간·일·주·월봉 모두 바이낸스 {stock} 선물이에요"
            "(일봉은 최근 1000일). "
            "이런 방향 판독의 다음 움직임 적중은 BTC 검증에서 50% 안팎이었어요(확률 아님)."
        )
    return (
        f"표시 전용 판독이에요. 일·주·월봉은 미국 {stock} 주가, 4시간봉은 바이낸스 선물이에요. "
        "미국 장이 닫힌 밤·주말에도 선물은 거래돼 월요일 개장 때 갭이 날 수 있어요. "
        "이런 방향 판독의 다음 움직임 적중은 BTC 검증에서 50% 안팎이었어요(확률 아님)."
    )


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def fetch(interval: str, symbol: str = SYMBOL) -> list[list]:
    """바이낸스 선물 봉(공개 시세, GET만). 비트코인 탭과 같은 주소에 종목만 바꾼다."""
    return _get(btc.URL.format(sym=symbol, iv=interval, n=btc.LIMIT))


def fetch_price(symbol: str = SYMBOL) -> float:
    return float(_get(btc.PRICE_URL.format(sym=symbol))["price"])


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


def build(now: datetime | None = None, symbol: str = SYMBOL) -> dict:
    cfg = US[symbol]
    now = now or datetime.now(btc.KST)
    now_ms = int(now.timestamp() * 1000)
    h4, _ = btc.parse(fetch("4h", symbol), now_ms)
    crypto = bool(cfg.get("crypto"))
    if crypto:
        days, _ = btc.parse(fetch("1d", symbol), now_ms)
    else:
        days = closed_days(fetch_yahoo(cfg["stock"]), now)
    raw = {"4h": h4, "1d": days, "1w": resample(days, "W"), "1M": resample(days, "M")}
    price = fetch_price(symbol)
    out = btc_outlook.build(raw, price)
    out["note"] = note(cfg["stock"], crypto)
    # 화면에 "원문"이라고 쓰지 않는다(사용자 요청 2026-11-11). 판독 문장은 비트코인과 같이 쓴다.
    if out.get("summary"):
        out["summary"]["watch"] = [
            w.replace("원문 규칙: ", "").replace("원문 규칙상 ", "")
            for w in out["summary"].get("watch", [])
        ]
        if out["summary"].get("verdict"):
            out["summary"]["verdict"] = (
                out["summary"]["verdict"]
                .replace("원문 규칙상 ", "")
                .replace("원문 기준으로는 ", "")
            )
    try:
        if crypto or cfg["base"] == cfg["stock"]:
            base = days
        else:
            base = closed_days(fetch_yahoo(cfg["base"]), now)
        same = [b.close for b in days if b.day == base[-1].day]
        classic = soxl_classic.build(
            base,
            same[0] if same else None,
            lev=cfg["lev"],
            base_name=cfg["base"],
            stock_name=cfg["stock"],
        )
    except (OSError, ValueError, KeyError) as exc:
        classic = {"error": f"{cfg['base']} 일봉 지표 분석 계산 실패 ({type(exc).__name__})"}
    return {
        "status": "ok",
        "generated_at": now.isoformat(timespec="seconds"),
        "symbol": symbol,
        "stock": cfg["stock"],
        "outlook": out,
        "classic": classic,
    }
