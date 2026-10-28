"""국내 주식 자동 주문 (한국투자증권, 시장가).

기본은 모의 실행(dry)이다. 주문을 내지 않고 "이렇게 냈을 것"만 기록하고 알린다.
`~/.minerva/trade/mode` 파일 내용이 `live`일 때만 실제 주문을 낸다.
`~/.minerva/trade/HALT` 파일이 있으면 아무것도 하지 않는다 (매매 중지 스위치).

- `open`  (평일 08:50): 전날 일일 분석의 국내·검증됨 신호로 장전 동시호가 시장가 주문
- `watch` (장중 30초): 이 봇이 산 종목만 손절가 감시, 닿으면 시장가 매도
- `reconcile` (평일 15:45): 봇 보유 목록과 실제 잔고를 맞춘다 (미체결·직접 매도 정리)
- `selftest --yes` (장중, 수동): 1주 사고 바로 팔아 주문 경로를 확인한다 (모드와 무관하게 실주문)

계좌 평가액이 고점 대비 DRAWDOWN_LIMIT 넘게 빠지면 `NO_ENTRY` 파일을 만들어 신규 매수만 막는다.
손절·청산 매도는 계속한다. 입출금도 평가액을 바꾸므로 출금 뒤엔 `peak.json`을 지운다.

사용자가 직접 산 종목은 건드리지 않는다. 봇이 산 종목은 `positions.json`에 적어 둔다.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from minerva import kis, notify

log = logging.getLogger(__name__)

TRADE_DIR = Path(os.environ.get("MINERVA_TRADE_DIR", "~/.minerva/trade")).expanduser()
DAILY_FILE = Path(os.environ.get("MINERVA_DAILY_OUT", "~/.minerva/daily/latest.json")).expanduser()

WEIGHT = 0.20  # 종목당 계좌 평가액 대비
MAX_POSITIONS = 5
REGIME_FACTOR = {"R0": 1.0, "R1": 1.0, "R2": 1 / 2, "R3": 1 / 3}
GAP_BUFFER = 1.03  # 시가가 전일 종가보다 높게 열려도 예수금을 넘지 않게
LONG_STOP = 0.10  # 롱: 매입가 −10%
SHORT_STOP = 0.07  # 인버스: 기초자산 +7% ≈ 인버스 −7%×배수
LEVERAGE = {"252670": 2}
ENTRY = ("LONG_ENTRY", "BUY")
EXIT = ("LONG_EXIT", "SELL")
RETRIES = 3
RETRY_GAP = 10
MAX_SIGNAL_AGE = timedelta(days=4)  # 주말·연휴를 넘긴 낡은 신호로는 주문하지 않는다
DRAWDOWN_LIMIT = 0.15
SELFTEST_CODE = "114800"  # KODEX 인버스: 1주가 싸고 거래가 많다


def _load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _save(path: Path, data) -> None:
    kis._write_private(path, json.dumps(data, ensure_ascii=False, indent=1))


def mode() -> str:
    path = TRADE_DIR / "mode"
    return "live" if path.exists() and path.read_text().strip() == "live" else "dry"


def halted() -> bool:
    return (TRADE_DIR / "HALT").exists()


def entries_blocked() -> bool:
    return (TRADE_DIR / "NO_ENTRY").exists()


def check_drawdown(account: dict) -> bool:
    """고점 평가액을 갱신하고, 한도를 넘게 빠졌으면 신규 매수를 막는다. 막혔으면 True."""
    equity = account.get("equity") or 0
    if equity <= 0:
        return entries_blocked()
    peak_file = TRADE_DIR / "peak.json"
    peak = _load(peak_file, {}).get("equity", 0)
    if equity > peak:
        _save(peak_file, {"equity": equity, "at": account.get("updated_at")})
    elif equity < peak * (1 - DRAWDOWN_LIMIT) and not entries_blocked():
        kis._write_private(TRADE_DIR / "NO_ENTRY", f"{account.get('updated_at')} {equity}\n")
        notify.send(
            f"🚨 계좌 평가액 {equity:,.0f}원, 고점 {peak:,.0f}원 대비 "
            f"{equity / peak - 1:.1%} → 신규 매수를 멈춰요 (손절·청산은 계속)\n"
            "다시 열려면 ~/.minerva/trade/NO_ENTRY 삭제"
        )
    return entries_blocked()


def is_open_day(cfg: dict, day: datetime) -> bool | None:
    """한국투자 휴장일 조회. 모르면 None (그땐 주문을 막지 않고 증권사 거부에 맡긴다)."""
    ymd = day.strftime("%Y%m%d")
    pages = kis._get(
        cfg,
        kis.access_token(cfg),
        "/uapi/domestic-stock/v1/quotations/chk-holiday",
        "CTCA0903R",
        {"BASS_DT": ymd, "CTX_AREA_NK": "", "CTX_AREA_FK": ""},
        max_pages=1,
    )
    for r in pages[0].get("output") or []:
        if r.get("bass_dt") == ymd:
            return r.get("opnd_yn") == "Y"
    return None


def code_of(symbol: str) -> str:
    return symbol.split(".")[0]


def plan(daily: dict, account: dict, owned: dict) -> tuple[list[dict], list[str]]:
    """일일 분석 + 계좌 → (낼 주문, 건너뛴 이유). 매도를 먼저 둔다."""
    regime = daily.get("vix", {}).get("regime", "R3")
    held = {p["code"]: p for p in account.get("positions", [])}
    actions = [a for a in daily.get("actions", []) if a.get("market") == "KR" and a.get("verified")]
    orders, skipped = [], []
    for a in actions:
        code = code_of(a["symbol"])
        if a["action"] in EXIT and code in owned and code in held:
            orders.append(_order("sell", a, code, held[code]["qty"]))
    open_after = len(held) - len(orders)
    alloc = account.get("equity", 0) * WEIGHT * REGIME_FACTOR.get(regime, 1 / 3)
    budget = account.get("cash_d2", 0)
    for a in actions:
        if a["action"] not in ENTRY:
            continue
        code = code_of(a["symbol"])
        if code in held:
            skipped.append(f"{a['name']}: 이미 보유")
            continue
        if open_after >= MAX_POSITIONS:
            skipped.append(f"{a['name']}: 동시 보유 {MAX_POSITIONS}종목 한도")
            continue
        if not a.get("ref_price"):
            skipped.append(f"{a['name']}: 기준가 없음")
            continue
        price = a["ref_price"] * GAP_BUFFER
        qty = math.floor(min(alloc, budget) / price)
        if qty < 1:
            skipped.append(f"{a['name']}: 1주 {price / GAP_BUFFER:,.0f}원 > 배정 {alloc:,.0f}원")
            continue
        orders.append(_order("buy", a, code, qty))
        budget -= qty * price
        open_after += 1
    return orders, skipped


def _order(side: str, a: dict, code: str, qty: int) -> dict:
    return {
        "side": side,
        "code": code,
        "name": a["name"],
        "qty": int(qty),
        "action": a["action"],
        "signal_date": a["date"],
        "kind": "inverse" if a["action"] in ("BUY", "SELL") else "long",
    }


def order_key(o: dict) -> str:
    return f"{o['signal_date']}:{o['code']}:{o['side']}"


def _post(cfg: dict, token: str, path: str, tr_id: str, body: dict) -> dict:
    req = urllib.request.Request(
        kis.BASE + path,
        data=json.dumps(body).encode(),
        method="POST",
        headers={
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": cfg["app_key"],
            "appsecret": cfg["app_secret"],
            "tr_id": tr_id,
            "custtype": "P",
        },
    )
    data, _ = kis._request(req)
    return data


def buyable_qty(cfg: dict, token: str, code: str) -> int:
    """시장가 기준 미수 없이 살 수 있는 수량 (조회 전용)."""
    pages = kis._get(
        cfg,
        token,
        "/uapi/domestic-stock/v1/trading/inquire-psbl-order",
        "TTTC8908R",
        {
            "CANO": cfg["cano"],
            "ACNT_PRDT_CD": cfg["prd"],
            "PDNO": code,
            "ORD_UNPR": "",
            "ORD_DVSN": "01",
            "CMA_EVLU_AMT_ICLD_YN": "N",
            "OVRS_ICLD_YN": "N",
        },
    )
    return kis._int((pages[0].get("output") or {}).get("nrcvb_buy_qty"))


def _already_sent(cfg: dict, token: str, o: dict, since: str) -> bool:
    """응답을 못 받은 주문이 실제로 들어갔는지 오늘 주문 내역으로 확인한다."""
    snap = kis._snapshot(cfg)
    side = "매수" if o["side"] == "buy" else "매도"
    return any(
        r["code"] == o["code"] and side in r["side"] and r["time"] >= since for r in snap["orders"]
    )


def send_order(cfg: dict, o: dict, live: bool) -> dict:
    """시장가 주문. 응답이 불확실하면 내역을 확인하고 안 들어갔을 때만 다시 낸다.

    증권사가 거부한 주문(잔고 부족·거래정지 등, rt_cd != 0)은 다시 내도 같으니 재시도하지 않는다.
    """
    if not live:
        return {**o, "result": "dry", "msg": "모의 실행 (주문 안 냄)"}
    token = kis.access_token(cfg)
    since = datetime.now(kis.KST).strftime("%Y-%m-%d %H:%M")
    body = {
        "CANO": cfg["cano"],
        "ACNT_PRDT_CD": cfg["prd"],
        "PDNO": o["code"],
        "ORD_DVSN": "01",  # 시장가
        "ORD_QTY": str(o["qty"]),
        "ORD_UNPR": "0",
    }
    tr_id = "TTTC0012U" if o["side"] == "buy" else "TTTC0011U"
    last = ""
    for attempt in range(RETRIES + 1):
        if attempt:
            time.sleep(RETRY_GAP)
            try:
                if _already_sent(cfg, token, o, since):
                    return {**o, "result": "sent", "msg": "응답은 놓쳤지만 주문 내역에서 확인"}
            except (kis.KisError, OSError) as e:
                last = f"내역 확인 실패: {e}"
                continue
        try:
            data = _post(cfg, token, "/uapi/domestic-stock/v1/trading/order-cash", tr_id, body)
        except (kis.KisError, OSError, urllib.error.URLError) as e:
            last = str(e)
            log.warning("order %s attempt %d failed: %s", order_key(o), attempt + 1, e)
            continue
        if data.get("rt_cd") == "0":
            odno = (data.get("output") or {}).get("ODNO", "")
            return {**o, "result": "sent", "odno": odno, "attempts": attempt + 1}
        msg = f"{data.get('msg_cd')}: {str(data.get('msg1', '')).strip()}"
        return {**o, "result": "rejected", "msg": msg}
    return {**o, "result": "failed", "msg": last, "attempts": RETRIES + 1}


def _journal(entry: dict) -> None:
    TRADE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRADE_DIR / "journal.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _line(r: dict) -> str:
    side = "매수" if r["side"] == "buy" else "매도"
    icon = {"dry": "🧪", "sent": "✅", "rejected": "⛔", "failed": "❌"}[r["result"]]
    tail = f" — {r['msg']}" if r.get("msg") else ""
    return f"{icon} {r['name']}({r['code']}) {side} {r['qty']}주 시장가{tail}"


def run_open(now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(kis.KST)
    if halted():
        notify.send("⏸ 매매 중지 상태라 오늘 주문을 내지 않았어요")
        return []
    daily = _load(DAILY_FILE, None)
    if not daily or not daily.get("generated_at"):
        notify.send("⚠️ 일일 분석이 없어 오늘 주문을 내지 않았어요")
        return []
    made = datetime.fromisoformat(daily["generated_at"])
    if now - made > MAX_SIGNAL_AGE:
        notify.send(f"⚠️ 일일 분석이 오래돼서({daily['generated_at']}) 주문을 내지 않았어요")
        return []
    cfg = kis.load_config()
    try:
        open_day = is_open_day(cfg, now)
    except kis.KisError as e:
        log.warning("holiday check failed: %s", e)
        open_day = None
    if open_day is False:
        notify.send(f"📅 {now:%m/%d}은 휴장일이라 주문을 내지 않아요")
        return []
    account = kis._snapshot(cfg, now)
    owned = _load(TRADE_DIR / "positions.json", {})
    done = {e["key"] for e in _read_journal() if e.get("result") in ("sent", "dry")}
    live = mode() == "live"
    orders, skipped = plan(daily, account, owned)
    if check_drawdown(account):
        skipped += [
            f"{o['name']}: 계좌 손실 한도로 신규 매수 중지" for o in orders if o["side"] == "buy"
        ]
        orders = [o for o in orders if o["side"] == "sell"]
    results = []
    for o in orders:
        key = order_key(o)
        if key in done:
            continue
        if o["side"] == "buy" and live:
            # 증권사 기준 매수 가능 수량을 넘지 않게 줄인다
            o["qty"] = min(o["qty"], buyable_qty(cfg, kis.access_token(cfg), o["code"]))
            if o["qty"] < 1:
                skipped.append(f"{o['name']}: 매수 가능 수량 0")
                continue
        r = send_order(cfg, o, live)
        r["key"], r["at"] = key, now.isoformat(timespec="seconds")
        _journal(r)
        results.append(r)
        if r["result"] == "sent":
            _track(owned, r)
    _save(TRADE_DIR / "positions.json", owned)
    head = "🛒 오늘 주문" + ("" if live else " (모의 실행: 실제 주문 안 냄)")
    lines = [_line(r) for r in results] or ["낼 주문 없음"]
    if skipped:
        lines += ["", "건너뜀:"] + [f"· {s}" for s in skipped]
    notify.send("\n".join([head, f"레짐 {daily.get('vix', {}).get('regime')}", *lines]))
    return results


def _track(owned: dict, r: dict) -> None:
    if r["side"] == "buy":
        owned[r["code"]] = {
            "name": r["name"],
            "kind": r["kind"],
            "opened": r["at"][:10],
            "stop": None,  # 체결 평균가가 나오면 watch가 정한다
        }
    else:
        owned.pop(r["code"], None)


def _read_journal() -> list[dict]:
    path = TRADE_DIR / "journal.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stop_price(kind: str, code: str, avg: float) -> float:
    if kind == "inverse":
        return avg * (1 - SHORT_STOP * LEVERAGE.get(code, 1))
    return avg * (1 - LONG_STOP)


def run_watch(account: dict | None = None) -> list[dict]:
    """봇이 산 종목만 손절가 감시. 계좌 스냅숏(30초마다 갱신)의 현재가를 쓴다."""
    if halted():
        return []
    latest = kis.STATE_DIR / "latest.json"
    if account is None and latest.exists():
        check_drawdown(json.loads(latest.read_text()))  # 30초 스냅숏으로 API 호출 없이
    owned = _load(TRADE_DIR / "positions.json", {})
    if not owned:
        return []
    cfg = kis.load_config()
    account = account or kis._snapshot(cfg)
    held = {p["code"]: p for p in account.get("positions", [])}
    live = mode() == "live"
    results = []
    for code, pos in list(owned.items()):
        p = held.get(code)
        if p is None:
            continue  # 아직 체결 전이거나 직접 판 종목
        if pos.get("stop") is None and p["avg"] > 0:
            pos["stop"] = round(stop_price(pos["kind"], code, p["avg"]))
            notify.send(
                f"🎯 {pos['name']} 체결 확인 · 평균 {p['avg']:,.0f}원 · 손절 {pos['stop']:,}원"
            )
        if pos.get("stop") and p["price"] and p["price"] <= pos["stop"]:
            o = {
                "side": "sell",
                "code": code,
                "name": pos["name"],
                "qty": p["qty"],
                "action": "STOP",
                "signal_date": datetime.now(kis.KST).strftime("%Y-%m-%d"),
                "kind": pos["kind"],
            }
            r = send_order(cfg, o, live)
            r["key"] = order_key(o)
            r["at"] = datetime.now(kis.KST).isoformat(timespec="seconds")
            _journal(r)
            results.append(r)
            if r["result"] in ("sent", "dry"):
                owned.pop(code)
            elif pos.get("stop_alerted"):
                continue  # 30초마다 다시 시도하되 실패 알림은 한 번만
            pos["stop_alerted"] = True
            notify.send(f"🛑 손절가 {pos['stop']:,}원 도달 (현재 {p['price']:,.0f}원)\n{_line(r)}")
    _save(TRADE_DIR / "positions.json", owned)
    return results


def run_reconcile(now: datetime | None = None) -> list[str]:
    """장 마감 뒤 봇 보유 목록을 실제 잔고와 맞춘다. 맞춘 내용을 알린다."""
    now = now or datetime.now(kis.KST)
    cfg = kis.load_config()
    account = kis._snapshot(cfg, now)
    check_drawdown(account)
    held = {p["code"]: p for p in account.get("positions", [])}
    status = {o["code"]: o["status"] for o in reversed(account.get("orders", []))}
    owned = _load(TRADE_DIR / "positions.json", {})
    notes = []
    for code, pos in list(owned.items()):
        if code not in held:
            owned.pop(code)
            why = status.get(code, "오늘 주문 내역 없음 → 직접 매도한 것으로 봄")
            notes.append(f"• {pos['name']}: 잔고에 없음 ({why}) → 봇 목록에서 뺌")
    today = now.strftime("%Y-%m-%d")
    for e in _read_journal():
        sold_today = e.get("side") == "sell" and e.get("result") == "sent"
        if sold_today and str(e.get("at", "")).startswith(today) and e["code"] in held:
            if e["code"] not in owned:
                owned[e["code"]] = {
                    "name": e["name"],
                    "kind": e.get("kind", "long"),
                    "opened": "?",
                    "stop": None,  # watch가 평균가로 다시 정한다
                }
            left = held[e["code"]]["qty"]
            notes.append(f"• {e['name']}: 매도 접수했지만 {left}주 남음 → 다시 감시 (확인 필요)")
    _save(TRADE_DIR / "positions.json", owned)
    if notes:
        notify.send("🧾 장 마감 대사\n" + "\n".join(notes))
    return notes


def run_selftest(code: str = SELFTEST_CODE, wait: float = 90) -> list[dict]:
    """1주 시장가 매수 → 체결 확인 → 1주 시장가 매도. 주문 경로를 실제로 확인한다."""
    now = datetime.now(kis.KST)
    if not (9 * 60 + 5 <= now.hour * 60 + now.minute <= 15 * 60 + 15):
        raise ValueError("시험 주문은 장중(09:05~15:15)에만 한다")
    cfg = kis.load_config()
    before = {p["code"]: p["qty"] for p in kis._snapshot(cfg)["positions"]}.get(code, 0)
    results = []
    for side, target in (("buy", before + 1), ("sell", before)):
        o = {
            "side": side,
            "code": code,
            "name": "시험주문",
            "qty": 1,
            "action": "SELFTEST",
            "signal_date": now.strftime("%Y-%m-%d"),
            "kind": "test",
        }
        r = send_order(cfg, o, live=True)
        r["key"], r["at"] = f"selftest:{order_key(o)}", datetime.now(kis.KST).isoformat()
        _journal(r)
        results.append(r)
        if r["result"] != "sent":
            break
        deadline = time.time() + wait
        qty = None
        while time.time() < deadline:
            time.sleep(5)
            qty = {p["code"]: p["qty"] for p in kis._snapshot(cfg)["positions"]}.get(code, 0)
            if qty == target:
                break
        r["filled"] = qty == target
        if not r["filled"]:
            break
    ok = len(results) == 2 and all(r.get("filled") for r in results)
    lines = [_line(r) + (" · 체결 확인" if r.get("filled") else " · 체결 미확인") for r in results]
    notify.send(("✅ 시험 주문 성공" if ok else "❌ 시험 주문 확인 필요") + "\n" + "\n".join(lines))
    return results


def run_preview() -> list[dict]:
    """16:30 분석 뒤: 내일 08:50에 낼 주문을 미리 알린다 (주문 안 냄)."""
    daily = _load(DAILY_FILE, {})
    cfg = kis.load_config()
    orders, skipped = plan(daily, kis._snapshot(cfg), _load(TRADE_DIR / "positions.json", {}))
    state = "⏸ 매매 중지 상태" if halted() else ("🔴 실전" if mode() == "live" else "🧪 모의 실행")
    lines = [
        f"📋 내일 08:50 주문 미리보기 · {state}",
        f"레짐 {daily.get('vix', {}).get('regime')} · 분석 {daily.get('generated_at')}",
    ]
    lines += [_line({**o, "result": "dry", "msg": ""}).replace("🧪", "•") for o in orders] or [
        "낼 주문 없음"
    ]
    if skipped:
        lines += ["", "건너뜀:"] + [f"· {x}" for x in skipped]
    lines += ["", "멈추려면 텔레그램에 '매매중지'"]
    notify.send("\n".join(lines))
    return orders


COMMANDS = {
    "open": run_open,
    "watch": run_watch,
    "preview": run_preview,
    "reconcile": run_reconcile,
    "selftest": run_selftest,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="국내 주식 자동 주문 (mode 파일이 live일 때만 실주문)"
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--yes", action="store_true", help="selftest: 실주문을 낸다는 확인")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.command == "selftest" and not args.yes:
        parser.error("selftest는 실제 주문을 낸다. --yes 를 붙여야 실행한다")
    try:
        results = COMMANDS[args.command]()
    except (kis.KisError, OSError, KeyError, ValueError) as e:
        log.error("%s failed: %s", args.command, e)
        if args.command != "watch":  # watch는 30초마다 돌아 일시 오류로 알림이 쏟아진다
            notify.send(f"❌ 자동 주문({args.command}) 오류: {e}")
        return 1
    print(f"{args.command} {mode()} {len(results)}건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
