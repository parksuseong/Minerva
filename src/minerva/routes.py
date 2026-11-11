"""웹 서버가 보여주기만 하는 읽기 전용 API.

웹 서버는 계산도 주문도 하지 않는다. 각 데이터는 별도 프로세스가 파일로 쓴다.
- 종목 분석: 배치 `minerva-daily.timer` → `~/.minerva/daily/latest.json`
- 계좌: `python -m minerva.kis` (`minerva-account.timer`) → `~/.minerva/account/latest.json`
  증권사 키는 그 프로세스만 읽는다. 웹 서버에는 키가 없다.
  기본은 공개(사용자 요청). 열람 토큰을 설정하면 `X-Minerva-Token`이 있어야 보여준다.
  수익률: 같은 폴더의 `capital.json`(시작 원금·입금)과 `equity.json`(날짜별 평가액)으로 계산.
- 바이낸스 선물: `python -m minerva.binance` (`minerva-binance.timer`)
  → `~/.minerva/account/binance.json`. 시드는 `binance_capital.json`, 날짜별 평가액은
  `binance_equity.json` (USDT). 열람 토큰은 계좌와 같다.
- 자동매매: `python -m minerva.trade` (`minerva-trade-*.timer`) → `~/.minerva/trade/`
- BTC 자동매매: `python -m minerva.btc_trade` (`minerva-btc-trade.timer`) → `~/.minerva/btc_trade/`.
  `/api/btc/engine`이 그 파일을 읽기만 한다(`btc_view.py`). 주문은 엔진만 낸다.
- SCR 실험실: `/api/btc/klines`가 바이낸스 선물 공개 시세(GET, 키 없음)를 덩어리째 캐시해 넘긴다.
  계산은 화면(`static/scrlab_core.js`)이 한다.
"""

from __future__ import annotations

import gzip
import hmac
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Response

from minerva import btc_view, perf
from minerva.strategy import soxl

router = APIRouter()


def btc_path() -> Path:
    return Path(os.environ.get("MINERVA_BTC_OUT", "~/.minerva/btc/latest.json")).expanduser()


def daily_path() -> Path:
    return Path(os.environ.get("MINERVA_DAILY_OUT", "~/.minerva/daily/latest.json")).expanduser()


def account_path() -> Path:
    default = "~/.minerva/account/latest.json"
    return Path(os.environ.get("MINERVA_ACCOUNT_OUT", default)).expanduser()


def account_view_token() -> str | None:
    """MINERVA_ACCOUNT_TOKEN 또는 `~/.minerva/account/view.token`. 둘 다 없으면 잠그지 않는다."""
    if os.environ.get("MINERVA_ACCOUNT_TOKEN"):
        return os.environ["MINERVA_ACCOUNT_TOKEN"]
    default = "~/.minerva/account/view.token"
    path = Path(os.environ.get("MINERVA_ACCOUNT_TOKEN_FILE", default)).expanduser()
    if not path.exists():
        return None
    return path.read_text().strip() or None


def _check_account_token(token: str) -> None:
    want = account_view_token()
    # 비 ASCII str은 compare_digest가 TypeError(500)를 내서 바이트로 비교한다
    if want and not hmac.compare_digest(token.encode(), want.encode()):
        raise HTTPException(status_code=403, detail="invalid token")


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@router.get("/api/daily")
def daily_analysis() -> dict:
    data = _read(daily_path())
    if data is None:
        return {"status": "pending", "message": "아직 오늘 분석이 없어요 (09:10·16:30 자동 실행)"}
    return data


@router.get("/api/btc")
def btc_analysis() -> dict:
    data = _read(btc_path())
    if data is None:
        return {"status": "pending", "message": "아직 비트코인 분석이 없어요 (15분마다 자동 실행)"}
    return data


@router.get("/api/btc/engine")
def btc_engine() -> dict:
    """BTC 자동매매 엔진 상태(읽기 전용): 모드·포지션·마지막 신호·AI 손절익절·정지 게이지·거래."""
    return btc_view.engine_view()


KLINE_URL = "https://fapi.binance.com/fapi/v1/klines"
# SCR 실험실에서 고를 수 있는 봉(분 단위)
# 봉을 내줄 수 있는 선물 종목(공개 시세). SOXL은 종목 분석 > SOXL 탭(보기만, 2026-11-11)
KLINE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOXLUSDT", "NVDAUSDT")
LAB_INTERVALS = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
LAB_MAX_DAYS = 365
# 5분봉 1년은 10만 봉이 넘어 휴대폰 화면이 버거워 90일까지만 (사용자 요청 2026-11-08 1년 보기)
LAB_DAY_CAP = {"5m": 90}
# 지표가 자리 잡도록 보여줄 기간 앞에 더 받는 봉 수
# (RSI 와일더 평균은 앞 값에 기대어 천천히 수렴한다)
LAB_WARMUP = 300
# 바이낸스 한 번 요청 최대 봉 수. 시각을 이 단위로 잘라 덩어리째 캐시한다.
LAB_CHUNK = 1500
LAB_TTL = 45.0  # 아직 끝나지 않은 마지막 덩어리
LAB_OLD_TTL = 6 * 3600.0  # 다 끝난 덩어리(봉이 더 바뀌지 않는다)
LAB_MAX_CHUNKS = 80  # 15분봉 1년 25개 + 5분봉 90일 18개 + 나머지, 봉 약 12만 개 이하
_kline_cache: OrderedDict = OrderedDict()
_kline_lock = threading.Lock()


def _get_klines(interval: str, start_ms: int, limit: int, symbol: str = "BTCUSDT") -> list[list]:
    query = urllib.parse.urlencode(
        {"symbol": symbol, "interval": interval, "startTime": start_ms, "limit": limit}
    )
    req = urllib.request.Request(f"{KLINE_URL}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


def _mono() -> float:
    return time.monotonic()


def _chunk(interval: str, k: int, now_ms: int, symbol: str = "BTCUSDT") -> list[tuple]:
    """k번째 덩어리(LAB_CHUNK봉)의 봉 (시작 ms, 시, 고, 저, 종). 거래량은 화면에서 안 쓴다."""
    span = LAB_CHUNK * LAB_INTERVALS[interval] * 60_000
    key = (interval, k) if symbol == "BTCUSDT" else (symbol, interval, k)
    with _kline_lock:
        hit = _kline_cache.get(key)
        if hit:
            at, rows, done = hit
            if _mono() - at < (LAB_OLD_TTL if done else LAB_TTL):
                _kline_cache.move_to_end(key)
                return rows
    page = (
        _get_klines(interval, k * span, LAB_CHUNK)
        if symbol == "BTCUSDT"
        else _get_klines(interval, k * span, LAB_CHUNK, symbol)
    )
    if not isinstance(page, list):
        raise ValueError(f"바이낸스 봉 응답이 목록이 아니다: {str(page)[:100]}")
    # 시세 공백이 있으면 1500봉이 덩어리 끝을 넘는다: 이웃 덩어리와 겹치지 않게 자른다
    lo, hi = k * span, (k + 1) * span
    rows = [
        (int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]))
        for r in page
        if lo <= int(r[0]) < hi and int(r[6]) < now_ms
    ]
    done = (k + 1) * span <= now_ms
    with _kline_lock:
        _kline_cache[key] = (_mono(), rows, done)
        _kline_cache.move_to_end(key)
        while len(_kline_cache) > LAB_MAX_CHUNKS:
            _kline_cache.popitem(last=False)
    return rows


def lab_klines(interval: str, days: int, now_ms: int, symbol: str = "BTCUSDT") -> dict:
    """보여줄 기간 + 앞쪽 여유 봉. 진행 중인 마지막 봉은 뺀다."""
    step = LAB_INTERVALS[interval] * 60_000
    span = LAB_CHUNK * step
    show_from = now_ms - days * 86_400_000
    start = show_from - LAB_WARMUP * step
    bars: list[tuple] = []
    for k in range(start // span, (now_ms - 1) // span + 1):
        bars.extend(
            b
            for b in _chunk(interval, k, now_ms, symbol)
            if b[0] >= start and b[0] + step <= now_ms
        )
    return {
        "symbol": symbol,
        "interval": interval,
        "days": days,
        "show_from": show_from,
        "bars": bars,
        "fetched_at": now_ms,
    }


def _now_ms() -> int:
    return int(time.time() * 1000)


@router.get("/api/btc/klines")
def btc_klines(
    interval: str = "15m", days: int = 7, symbol: str = "BTCUSDT", accept_encoding: str = Header("")
) -> Response:
    """SCR 실험실·SOXL 탭용 선물 봉(공개 시세, GET만). 덩어리째 캐시한다."""
    if symbol not in KLINE_SYMBOLS:
        raise HTTPException(status_code=400, detail="이 종목 봉은 내줄 수 없어요")
    if interval not in LAB_INTERVALS:
        raise HTTPException(status_code=400, detail="봉은 5m·15m·30m·1h·4h·1d 중 하나예요")
    want = max(1, int(days))
    days = min(want, LAB_MAX_DAYS, LAB_DAY_CAP.get(interval, LAB_MAX_DAYS))
    try:
        data = lab_klines(interval, days, _now_ms(), symbol)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=502, detail="바이낸스 시세를 받지 못했어요") from exc
    data["capped"] = days < want
    # 15분봉 1년이면 약 2MB라 브라우저가 받으면 압축해서 보낸다(라우트 하나라 미들웨어 대신 여기서)
    body = json.dumps(data, separators=(",", ":")).encode()
    if isinstance(accept_encoding, str) and "gzip" in accept_encoding:
        return Response(
            gzip.compress(body, 5),
            media_type="application/json",
            headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding"},
        )
    return Response(body, media_type="application/json")


SOXL_TTL = 300.0  # 큰 그림은 5분마다 새로 계산(야후·바이낸스 공개 시세)
_soxl_cache: dict = {}
_soxl_lock = threading.Lock()


@router.get("/api/soxl")
def soxl_analysis(symbol: str = "SOXLUSDT") -> dict:
    """종목 분석 > SOXL·NVDA: 큰 그림(4시간·일·주·월) + 일봉 지표 분석. 보기만, 주문 없음."""
    if symbol not in soxl.US:
        raise HTTPException(status_code=400, detail="이 종목은 볼 수 없어요")
    with _soxl_lock:
        hit = _soxl_cache.get(symbol)
        if hit and _mono() - hit[0] < SOXL_TTL:
            return hit[1]
        try:
            data = soxl.build(symbol=symbol)
        except (OSError, ValueError, KeyError) as exc:
            if hit:
                return hit[1]
            why = type(exc).__name__
            name = soxl.US[symbol]["stock"]
            return {"status": "error", "message": f"{name} 시세를 받지 못했어요 ({why})"}
        _soxl_cache[symbol] = (_mono(), data)
        return data


@router.get("/api/signals")
def signals() -> dict:
    """자동매매용: 오늘 낼 주문만. 같은 분석 결과에서 뽑는다."""
    data = _read(daily_path())
    if data is None:
        return {"status": "pending", "actions": []}
    vix = data.get("vix") or {}
    return {
        "status": "ok",
        "generated_at": data.get("generated_at"),
        "regime": vix.get("regime"),
        "size_limit": vix.get("size_limit"),
        "actions": data.get("actions", []),
    }


@router.get("/api/account")
def account(x_minerva_token: str = Header("")) -> dict:
    _check_account_token(x_minerva_token)
    data = _read(account_path())
    if data is None:
        return {
            "status": "not_connected",
            "message": "증권사 계좌가 아직 연결되지 않았어요 (한국투자증권 스냅숏 대기 중)",
        }
    capital = _read(account_path().parent / "capital.json")
    if capital and data.get("status") == "ok":
        history = _read(account_path().parent / "equity.json") or {}
        owned = _read(trade_dir() / "positions.json") or {}
        bot = perf.bot_pnl(trade_dir(), owned, data)
        data["returns"] = perf.account_returns(data, history, capital, bot)
    return data


@router.get("/api/account/binance")
def binance_account(x_minerva_token: str = Header("")) -> dict:
    _check_account_token(x_minerva_token)
    folder = account_path().parent
    data = _read(folder / "binance.json")
    if data is None:
        return {"status": "not_connected", "message": "바이낸스 계좌 스냅숏 대기 중이에요"}
    capital = _read(folder / "binance_capital.json")
    if capital and capital.get("live") is False:
        # 실전 전 수동 매매 기간: 잔고만 보이고 손익·체결은 화면에서 비운다 (사용자 결정 2026-11-08)
        data = {
            k: data[k]
            for k in (
                "status",
                "broker",
                "market",
                "updated_at",
                "wallet",
                "available",
                "error",
                "error_at",
            )
            if k in data
        }
        data.update(live=False, positions=[], orders=[], trades=[])
        return data
    if capital and data.get("status") == "ok":
        history = _read(folder / "binance_equity.json") or {}
        # 수익률은 수수료(낸 것 + 닫을 때 낼 추정치)를 뺀 순평가액으로 낸다
        net = {**data, "equity": data.get("net_equity", data["equity"])}
        returns = perf.account_returns(net, history, capital, 0.0)
        returns.pop("bot", None)
        returns.pop("bot_pct", None)
        data["returns"] = returns
    return data


@router.get("/api/account/binance/series")
def binance_series(x_minerva_token: str = Header("")) -> dict:
    """자산 차트: 5분 단위 순평가액(USDT)과 시드."""
    _check_account_token(x_minerva_token)
    folder = account_path().parent
    capital = _read(folder / "binance_capital.json") or {}
    if capital.get("live") is False:
        return {"seed": None, "points": [], "live": False}
    points = (_read(folder / "binance_series.json") or {}).get("points", [])
    # 점 시각은 KST 벽시계를 UTC 자리에 둔 값이라 시드 시각도 같은 방식으로 바꿔 비교한다
    start = capital.get("since_at") or capital.get("since")
    if start:
        fmt = "%Y-%m-%d %H:%M" if " " in start else "%Y-%m-%d"
        cut = datetime.strptime(start, fmt).replace(tzinfo=timezone.utc).timestamp()
        points = [p for p in points if p["t"] >= cut]
    return {"seed": capital.get("base"), "points": points}


def profile_path() -> Path:
    return Path(os.environ.get("MINERVA_PROFILE", "~/.minerva/profile.json")).expanduser()


@router.get("/api/profile")
def profile() -> dict:
    """화면에 쓰는 사용자 호칭. 실명은 저장소에 넣지 않고 서버 파일에서 읽는다."""
    data = _read(profile_path()) or {}
    return {"boss": data.get("boss") or "사장님"}


def trade_dir() -> Path:
    return Path(os.environ.get("MINERVA_TRADE_DIR", "~/.minerva/trade")).expanduser()


@router.get("/api/trade")
def trade_status() -> dict:
    """자동매매 상태: 모드·중지 여부·봇 보유 종목·최근 주문 결과 (`minerva.trade`가 쓴 파일)."""
    d = trade_dir()
    mode_file = d / "mode"
    # 주문 기록은 덧붙여 쓰는 파일이라 마지막 줄이 쓰는 중일 수 있다: 깨진 줄은 건너뛴다
    orders = btc_view.tail_jsonl(d / "journal.jsonl", 30)
    return {
        "mode": "live" if mode_file.exists() and mode_file.read_text().strip() == "live" else "dry",
        "halted": (d / "HALT").exists(),
        "positions": _read(d / "positions.json") or {},
        "orders": list(reversed(orders)),
    }
