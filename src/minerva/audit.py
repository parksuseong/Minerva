"""자동매매 운용 점검 (조회 전용). 주문·보유가 규칙과 맞는지 매일 확인한다.

규칙의 원본은 `minerva.trade`의 상수다. 여기서는 결과 파일만 읽고 판정한다.
- 입력: `~/.minerva/trade/{journal.jsonl,positions.json}`, `~/.minerva/account/latest.json`
- 실행: `python -m minerva.audit [--notify]` (평일 16:00, minerva-trade-audit.timer)
- 깊은 원인 분석은 `.claude/agents/trade-auditor.md` 에이전트가 이 출력을 근거로 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime

from minerva import kis, notify, trade

WEIGHT_TOLERANCE = 1.10  # 체결가가 기준가보다 높게 열린 만큼은 봐준다
STOP_TOLERANCE = 0.01


def audit(journal: list[dict], owned: dict, account: dict, mode: str, day: str) -> list[dict]:
    """점검 결과 목록. level은 error(규칙 위반) · warn(확인 필요) · info."""
    out = []

    def add(level: str, msg: str) -> None:
        out.append({"level": level, "msg": msg})

    today = [e for e in journal if str(e.get("at", "")).startswith(day)]
    real = [e for e in today if e.get("kind") != "test"]
    sent = [e for e in real if e.get("result") in ("sent", "dry")]

    for e in sent:
        if e["side"] == "buy" and e.get("action") not in trade.ENTRY:
            add("error", f"{e['name']}: 진입 신호가 아닌 매수 ({e.get('action')})")
        if not str(e.get("code", "")).isdigit():
            add("error", f"{e['name']}: 국내 종목 코드가 아님 ({e.get('code')})")
    for key, n in Counter(e["key"] for e in sent).items():
        if n > 1:
            add("error", f"중복 주문 {n}회: {key}")
    for e in real:
        if e.get("result") in ("rejected", "failed"):
            add("warn", f"{e['name']} {e['side']} {e['result']}: {e.get('msg', '')}")

    equity = account.get("equity") or 0
    held = {p["code"]: p for p in account.get("positions", [])}
    bot = {c: held[c] for c in owned if c in held}
    if len(bot) > trade.MAX_POSITIONS:
        add("error", f"봇 보유 {len(bot)}종목 > 한도 {trade.MAX_POSITIONS}")
    for code, p in bot.items():
        pos = owned[code]
        cost = p["qty"] * p["avg"]
        if equity and cost > equity * trade.WEIGHT * WEIGHT_TOLERANCE:
            add("error", f"{pos['name']}: 매입금액 {cost:,.0f}원이 계좌의 {cost / equity:.0%}")
        want = trade.stop_price(pos.get("kind", "long"), code, p["avg"])
        if pos.get("stop") is None:
            add("warn", f"{pos['name']}: 손절가 미설정 (체결 확인 전?)")
        elif abs(pos["stop"] - want) > want * STOP_TOLERANCE:
            add("error", f"{pos['name']}: 손절가 {pos['stop']:,} ≠ 규칙 {want:,.0f}")
        if pos.get("stop") and p["price"] and p["price"] <= pos["stop"]:
            add("error", f"{pos['name']}: 현재가 {p['price']:,.0f} ≤ 손절가인데 아직 보유")
    for code, pos in owned.items():
        if code not in held:
            add("warn", f"{pos['name']}: 봇 목록에 있는데 잔고에 없음 (15:45 대사 확인)")

    flags = [
        name
        for name, on in (("매매중지", trade.halted()), ("신규매수중지", trade.entries_blocked()))
        if on
    ]
    state = "실전" if mode == "live" else "모의"
    add("info", f"모드 {state}" + (f" · {', '.join(flags)}" if flags else ""))
    add("info", f"오늘 주문 {len(sent)}건 · 봇 보유 {len(bot)}종목 · 평가액 {equity:,.0f}원")
    return out


def report(results: list[dict]) -> str:
    bad = [r for r in results if r["level"] == "error"]
    warn = [r for r in results if r["level"] == "warn"]
    head = "🔎 운용 점검: " + (
        f"❌ 규칙 위반 {len(bad)}건" if bad else "⚠️ 확인 필요" if warn else "✅ 이상 없음"
    )
    icon = {"error": "❌", "warn": "⚠️", "info": "•"}
    return "\n".join([head] + [f"{icon[r['level']]} {r['msg']}" for r in results])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="자동매매 운용 점검 (조회 전용)")
    parser.add_argument("--notify", action="store_true", help="텔레그램으로 보고")
    parser.add_argument("--day", help="YYYY-MM-DD (기본 오늘)")
    args = parser.parse_args(argv)
    day = args.day or datetime.now(kis.KST).strftime("%Y-%m-%d")
    latest = kis.STATE_DIR / "latest.json"
    account = json.loads(latest.read_text()) if latest.exists() else {}
    results = audit(
        trade._read_journal(),
        trade._load(trade.TRADE_DIR / "positions.json", {}),
        account,
        trade.mode(),
        day,
    )
    text = report(results)
    print(text)
    if args.notify:
        notify.send(text)
    return 1 if any(r["level"] == "error" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
