"""바이낸스 USDⓈ-M 선물 계좌 스냅숏 (조회 전용).

키는 저장소 밖 `~/.minerva/binance.env`(BINANCE_API_KEY·BINANCE_API_SECRET)에서 이 프로세스만
읽는다. 웹 서버는 결과 파일 `~/.minerva/account/binance.json`만 읽는다. 주문 API는 부르지 않는다
(GET만 보낸다).

수익률: `binance_capital.json`(시드·입출금, USDT)과 `binance_equity.json`(날짜별 마지막 순평가액).
순평가액 = 지갑 잔고 + 미실현 손익 - 지금 시장가로 닫을 때 낼 수수료(테이커 요율 추정).
지갑 잔고에는 이미 낸 수수료·펀딩이 빠져 있다. 자산 차트는 `binance_series.json`(5분 단위).

실전 전(capital `live: false`, 사용자 결정 2026-11-08): 수동 매매 기간은 기록하지 않는다.
잔고만 남기고 체결·손익·자산 기록은 비운다.
실전 자동매매를 시작하면 그 시각 잔고를 시드로 새로 잡는다.

실행: `python -m minerva.binance` (minerva-binance.timer가 1분마다 실행)
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from minerva.kis import KST, _write_private

log = logging.getLogger(__name__)

BASE = "https://fapi.binance.com"
ENV_FILE = Path(os.environ.get("MINERVA_BINANCE_ENV", "~/.minerva/binance.env")).expanduser()
STATE_DIR = Path(os.environ.get("MINERVA_ACCOUNT_DIR", "~/.minerva/account")).expanduser()
SYMBOL = "BTCUSDT"
# 엔진이 매매하는 종목(btc_trade.SYMBOLS와 같게). 체결 내역은 이 네 종목을 합쳐 보여준다
TRADE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOXLUSDT", "NVDAUSDT")
TRADES = 30
SERIES_STEP = 300  # 자산 차트 점 간격(초). 1분마다 돌지만 5분에 한 점만 남긴다.
KST_SHIFT = 9 * 3600  # 차트 라이브러리는 UTC로 그려서 KST 시각으로 옮겨 저장한다
INCOME_TYPES = {"REALIZED_PNL": "realized", "FUNDING_FEE": "funding", "COMMISSION": "commission"}


class BinanceError(RuntimeError):
    pass


def load_config(path: Path = ENV_FILE) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip()
    return {"key": values["BINANCE_API_KEY"], "secret": values["BINANCE_API_SECRET"]}


def _get(cfg: dict, path: str, **params) -> dict | list:
    """서명한 GET 요청. 이 모듈은 GET만 보낸다(주문·설정 변경 없음)."""
    params["timestamp"] = int(time.time() * 1000)
    params["recvWindow"] = 10000
    query = urllib.parse.urlencode(params)
    sig = hmac.new(cfg["secret"].encode(), query.encode(), hashlib.sha256).hexdigest()
    req = urllib.request.Request(
        f"{BASE}{path}?{query}&signature={sig}", headers={"X-MBX-APIKEY": cfg["key"]}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read()[:200].decode("utf-8", "replace")
        raise BinanceError(f"HTTP {e.code}: {body}") from e


def _ms(ms) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, KST).strftime("%Y-%m-%d %H:%M:%S")


def parse_account(acct: dict, risk: list[dict]) -> dict:
    marks = {r["symbol"]: r for r in risk}
    positions = []
    for p in acct.get("positions", []):
        qty = float(p["positionAmt"])
        if not qty:
            continue
        r = marks.get(p["symbol"], {})
        entry = float(p["entryPrice"])
        pnl = float(p["unrealizedProfit"])
        margin = float(p.get("initialMargin") or 0) or abs(qty) * entry / float(p["leverage"])
        positions.append(
            {
                "symbol": p["symbol"],
                "side": "롱" if qty > 0 else "숏",
                "qty": abs(qty),
                "entry": entry,
                "mark": float(r.get("markPrice", 0)) or None,
                "liq": float(r.get("liquidationPrice", 0)) or None,
                "leverage": int(float(p["leverage"])),
                "margin_type": "격리" if p.get("isolated") else "교차",
                "pnl": pnl,
                "pnl_pct": pnl / margin * 100 if margin else None,
            }
        )
    return {
        "wallet": float(acct["totalWalletBalance"]),
        "unrealized": float(acct["totalUnrealizedProfit"]),
        "equity": float(acct["totalMarginBalance"]),
        "available": float(acct["availableBalance"]),
        "can_trade": bool(acct.get("canTrade")),
        "positions": positions,
    }


def parse_orders(orders: list[dict]) -> list[dict]:
    out = [
        {
            "time": _ms(o["time"]),
            "symbol": o["symbol"],
            "side": "매수" if o["side"] == "BUY" else "매도",
            "type": o["type"],
            "qty": float(o["origQty"]),
            "price": float(o["price"]) or float(o.get("stopPrice") or 0) or None,
            "status": o["status"],
        }
        for o in orders
    ]
    return sorted(out, key=lambda o: o["time"], reverse=True)


def parse_trades(trades: list[dict]) -> list[dict]:
    out = [
        {
            "time": _ms(t["time"]),
            "symbol": t["symbol"],
            "side": "매수" if t["side"] == "BUY" else "매도",
            "qty": float(t["qty"]),
            "price": float(t["price"]),
            "realized": float(t["realizedPnl"]),
            "fee": float(t["commission"]),
        }
        for t in trades
    ]
    return sorted(out, key=lambda t: t["time"], reverse=True)[:TRADES]


def sum_income(rows: list[dict]) -> dict:
    """시드 이후 실현손익·펀딩·수수료 합계(USDT)."""
    out = {v: 0.0 for v in INCOME_TYPES.values()}
    for r in rows:
        key = INCOME_TYPES.get(r.get("incomeType"))
        if key and r.get("asset") == "USDT":
            out[key] += float(r["income"])
    return out


def close_fee(positions: list[dict], taker: float) -> float:
    """열린 포지션을 지금 시장가로 닫을 때 낼 수수료 추정(현재가 × 수량 × 테이커 요율)."""
    return sum(p["qty"] * (p["mark"] or p["entry"]) * taker for p in positions)


def _since_ms(capital: dict | None) -> int | None:
    """시드 시각. 시드를 잡은 날에도 그 전 거래가 있을 수 있어 `since_at`(분 단위)을 먼저 쓴다."""
    if not capital or not capital.get("since"):
        return None
    if capital.get("since_at"):
        at = datetime.strptime(capital["since_at"], "%Y-%m-%d %H:%M")
    else:
        at = datetime.strptime(capital["since"], "%Y-%m-%d")
    return int(at.replace(tzinfo=KST).timestamp() * 1000)


def snapshot(cfg: dict, now: datetime, capital: dict | None = None) -> dict:
    acct = _get(cfg, "/fapi/v2/account")
    risk = [r for r in _get(cfg, "/fapi/v2/positionRisk") if float(r["positionAmt"])]
    data = {
        "status": "ok",
        "broker": "바이낸스",
        "market": "USDⓈ-M 선물",
        "updated_at": now.strftime("%Y-%m-%d %H:%M"),
        **parse_account(acct, risk),
        "orders": parse_orders(_get(cfg, "/fapi/v1/openOrders")),
    }
    rate = _get(cfg, "/fapi/v1/commissionRate", symbol=SYMBOL)
    data["fee_rate"] = {
        "maker": float(rate["makerCommissionRate"]),
        "taker": float(rate["takerCommissionRate"]),
    }
    data["close_fee_est"] = close_fee(data["positions"], data["fee_rate"]["taker"])
    data["net_equity"] = data["equity"] - data["close_fee_est"]
    if capital is not None and capital.get("live") is False:
        data["live"] = False
        data["trades"] = []
        return data
    since = _since_ms(capital)
    trades = list(_get(cfg, "/fapi/v1/userTrades", symbol=SYMBOL, limit=100))
    for sym in TRADE_SYMBOLS:
        if sym == SYMBOL:
            continue
        try:
            trades += _get(cfg, "/fapi/v1/userTrades", symbol=sym, limit=100)
        except BinanceError as e:  # 한 종목(TradFi 약관 미동의 등) 실패로 화면을 멈추지 않는다
            log.warning("userTrades %s failed: %s", sym, e)
    # 시드 전 거래는 시드 금액에 이미 들어가 있으니 보여주지 않는다 (사용자 요청 2026-11-07)
    data["trades"] = parse_trades([t for t in trades if since is None or t["time"] >= since])
    if since is not None:
        # 최근 7일 이전 시작이면 바이낸스가 기간을 나눠 달라고 하므로 3개월 이내만 담는다
        start = max(since, int(now.timestamp() * 1000) - 89 * 86400_000)
        data["income"] = sum_income(_get(cfg, "/fapi/v1/income", startTime=start, limit=1000))
    return data


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def record_equity(path: Path, data: dict) -> None:
    """날짜별 마지막 평가액(KST). 오늘 수익률이 전날 값을 읽는다."""
    if data.get("status") != "ok" or data.get("live") is False:
        return
    history = _read(path) or {}
    history[data["updated_at"][:10]] = data.get("net_equity", data["equity"])
    _write_private(path, json.dumps(history, sort_keys=True))


def record_series(path: Path, data: dict, now: datetime) -> None:
    """자산 차트용 순평가액. 5분 칸마다 한 점, 같은 칸이면 최신 값으로 덮는다."""
    if data.get("status") != "ok" or data.get("live") is False:
        return
    points = (_read(path) or {}).get("points", [])
    t = int(now.timestamp()) // SERIES_STEP * SERIES_STEP + KST_SHIFT
    point = {"t": t, "v": round(data.get("net_equity", data["equity"]), 4)}
    if points and points[-1]["t"] == t:
        points[-1] = point
    else:
        points.append(point)
    _write_private(path, json.dumps({"points": points}))


def run(out: Path) -> dict:
    """스냅숏을 쓴다. 실패하면 직전 스냅숏을 남기고 오류만 덧붙인다."""
    now = datetime.now(KST)
    try:
        data = snapshot(load_config(), now, _read(out.parent / "binance_capital.json"))
    except (BinanceError, OSError, KeyError, ValueError) as e:
        log.error("binance snapshot failed: %s", e)
        data = _read(out) or {"status": "error"}
        data["error"] = str(e)
        data["error_at"] = now.strftime("%Y-%m-%d %H:%M")
        _write_private(out, json.dumps(data, ensure_ascii=False))
        raise
    _write_private(out, json.dumps(data, ensure_ascii=False))
    record_equity(out.parent / "binance_equity.json", data)
    record_series(out.parent / "binance_series.json", data, now)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="바이낸스 선물 계좌 스냅숏 (조회 전용)")
    parser.add_argument("--out", type=Path, default=STATE_DIR / "binance.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        data = run(args.out)
    except (BinanceError, OSError, KeyError, ValueError):
        return 1
    print(
        f"ok {data['updated_at']} 순평가 {data['net_equity']:.2f} USDT "
        f"포지션 {len(data['positions'])} 주문 {len(data['orders'])} 체결 {len(data['trades'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
