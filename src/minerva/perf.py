"""실전 성과: 봇이 끝낸 거래(청산 기록)와 손익, 주간 리포트.

- 청산 기록 `~/.minerva/trade/closed.jsonl`: 15:45 대사가 그날 체결된 봇 매도를 적는다.
- 봇 손익 = 청산 손익 합(비용 차감) + 봇 보유분 평가손익. 입출금과 무관하다.
- 주간 리포트(금 17:00): 이번 주·누적 성과를 백테스트 기대치와 비교한다.
실행: `python -m minerva.perf weekly`
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 실제 비용 근사: 수수료 0.015%×2, 매도 거래세(주식만) 0.20%
FEE = 0.00015
SELL_TAX = 0.002
ETF_CODES = {"114800", "252670"}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cost_rate(code: str) -> float:
    return FEE * 2 + (0 if code in ETF_CODES else SELL_TAX)


def closed_trade(sell: dict, fills: list[dict]) -> dict | None:
    """봇 매도 기록 + 그날 체결 내역 → 청산 한 건. 체결이 없으면 None."""
    day = str(sell.get("at", ""))[:10]
    mine = [
        f
        for f in fills
        if f["code"] == sell["code"] and "매도" in f["side"] and f["time"].startswith(day)
    ]
    qty = sum(f["qty"] for f in mine)
    entry = sell.get("entry_avg")
    if not qty or not entry:
        return None
    exit_px = sum(f["qty"] * f["price"] for f in mine) / qty
    gross = exit_px / entry - 1
    net = gross - cost_rate(sell["code"])
    return {
        "key": sell["key"],
        "code": sell["code"],
        "name": sell["name"],
        "kind": sell.get("kind", "long"),
        "reason": sell.get("action"),
        "opened": sell.get("opened"),
        "closed": day,
        "qty": qty,
        "entry": round(entry, 2),
        "exit": round(exit_px, 2),
        "gross_ret": round(gross, 5),
        "net_ret": round(net, 5),
        "pnl": round(qty * entry * net),
    }


def record_closed(trade_dir: Path, journal: list[dict], account: dict, day: str) -> list[dict]:
    """그날 체결된 봇 매도를 청산 기록에 더한다. 이미 적은 건 건너뛴다."""
    path = trade_dir / "closed.jsonl"
    seen = {r["key"] for r in _read_jsonl(path)}
    new = []
    for e in journal:
        is_sell = e.get("side") == "sell" and e.get("result") == "sent"
        if not is_sell or e.get("kind") == "test" or not str(e.get("at", "")).startswith(day):
            continue
        if e["key"] in seen:
            continue
        row = closed_trade(e, account.get("trades", []))
        if row:
            new.append(row)
            seen.add(row["key"])
    _append(path, new)
    return new


def bot_pnl(trade_dir: Path, owned: dict, account: dict) -> float:
    realized = sum(r["pnl"] for r in _read_jsonl(trade_dir / "closed.jsonl"))
    held = {p["code"]: p for p in account.get("positions", [])}
    unrealized = sum(held[c].get("pnl", 0) for c in owned if c in held)
    return realized + unrealized


def _summary(rows: list[dict]) -> str:
    if not rows:
        return "청산 0건"
    rets = [r["net_ret"] for r in rows]
    wins = sum(x > 0 for x in rets)
    return (
        f"청산 {len(rows)}건 · 승률 {wins / len(rows):.0%} · 평균 {statistics.mean(rets):+.2%} "
        f"(비용 후) · 손익 {sum(r['pnl'] for r in rows):+,}원"
    )


def weekly_report(trade_dir: Path, daily: dict, account: dict, owned: dict, now: datetime) -> str:
    rows = _read_jsonl(trade_dir / "closed.jsonl")
    monday = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")
    week = [r for r in rows if r["closed"] >= monday]
    held = {p["code"]: p for p in account.get("positions", [])}
    open_pnl = sum(held[c].get("pnl", 0) for c in owned if c in held)
    lines = [
        f"📈 주간 실전 성과 ({monday} ~ {now:%m/%d})",
        f"이번 주: {_summary(week)}",
        f"누적: {_summary(rows)}",
        f"보유 중 {sum(c in held for c in owned)}종목 · 평가손익 {open_pnl:+,.0f}원",
        f"계좌 평가액 {account.get('equity', 0):,.0f}원",
    ]
    # 백테스트 기대치: 이번에 거래한 종목들의 검증 승률·비용 후 우위
    ver = {}
    for a in daily.get("assets", []) + daily.get("products", []):
        side = "short" if a.get("kind") else "long"
        v = a.get("verify", {}).get(side, {})
        ver[a["symbol"].split(".")[0]] = v
    exp = [ver[r["code"]] for r in rows if ver.get(r["code"], {}).get("win_rate") is not None]
    if exp:
        lines.append(
            f"백테스트 기대: 승률 {statistics.mean(v['win_rate'] for v in exp):.0%} · "
            f"비용 후 우위 {statistics.mean(v.get('net_edge') or 0 for v in exp):+.2%}/거래"
        )
    if len(rows) < 30:
        lines.append(f"※ 실전 거래 {len(rows)}건이라 아직 우연과 구분하기 어려워요 (30건부터 비교)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from minerva import kis, notify, trade

    args = argv if argv is not None else sys.argv[1:]
    if args[:1] != ["weekly"]:
        print("usage: python -m minerva.perf weekly")
        return 2
    latest = kis.STATE_DIR / "latest.json"
    account = json.loads(latest.read_text()) if latest.exists() else {}
    text = weekly_report(
        trade.TRADE_DIR,
        trade._load(trade.DAILY_FILE, {}),
        account,
        trade._load(trade.TRADE_DIR / "positions.json", {}),
        datetime.now(kis.KST),
    )
    print(text)
    notify.send(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
