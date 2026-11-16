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
ETF_CODES = {"069500", "357870"}  # 매도 거래세 없음


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


def closed_trade(sell: dict, fills: list[dict], used: int = 0) -> dict | None:
    """봇 매도 기록 + 그날 체결 내역 → 청산 한 건. 체결이 없으면 None.

    `used`: 같은 날 같은 종목의 앞선 봇 매도가 이미 가져간 체결 수량 (TRIM 뒤 STOP 등).
    """
    day = str(sell.get("at", ""))[:10]
    odno = (sell.get("odno") or "").lstrip("0")
    exact = [f for f in fills if odno and (f.get("odno") or "").lstrip("0") == odno]
    if exact:
        # 주문번호가 맞으면 그 주문의 체결만 쓴다 (같은 날 다른 매도·직접 매도와 섞이지 않는다)
        mine, used = exact, 0
    else:
        mine = [
            f
            for f in fills
            if f["code"] == sell["code"] and "매도" in f["side"] and f["time"].startswith(day)
        ]
    filled = sum(f["qty"] for f in mine)
    # 같은 날 사용자가 직접 판 수량이 섞이지 않게 봇이 낸 수량까지만 센다
    qty = min(filled - used, sell.get("qty") or filled - used)
    entry = sell.get("entry_avg")
    if qty <= 0 or not entry:
        return None
    # 체결가는 그날 매도 체결 전체의 평균. 합계를 봇 수량으로 나누면 다른 매도분이 섞여 부풀려진다
    exit_px = sum(f["qty"] * f["price"] for f in mine) / filled
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
    old = _read_jsonl(path)
    seen = {r["key"] for r in old}
    used: dict[str, int] = {}  # 종목별로 그날 이미 청산 기록에 쓴 체결 수량
    for r in old:
        if r.get("closed") == day:
            used[r["code"]] = used.get(r["code"], 0) + r["qty"]
    new = []
    for e in journal:
        is_sell = e.get("side") == "sell" and e.get("result") == "sent"
        if (
            not is_sell
            or e.get("kind") in ("test", "park")
            or not str(e.get("at", "")).startswith(day)
        ):
            continue
        if e["key"] in seen:
            continue
        row = closed_trade(e, account.get("trades", []), used.get(e["code"], 0))
        if row:
            new.append(row)
            seen.add(row["key"])
            used[row["code"]] = used.get(row["code"], 0) + row["qty"]
    _append(path, new)
    return new


def bot_pnl(trade_dir: Path, owned: dict, account: dict) -> float:
    realized = sum(r["pnl"] for r in _read_jsonl(trade_dir / "closed.jsonl"))
    held = {p["code"]: p for p in account.get("positions", [])}
    unrealized = sum(held[c].get("pnl", 0) for c in owned if c in held)
    return realized + unrealized


def account_returns(account: dict, history: dict, capital: dict, bot: float) -> dict:
    """계좌 수익률. 입금은 `capital.json`의 deposits로 빼서 손익에 섞이지 않게 한다.

    capital: {"since": "YYYY-MM-DD", "base": 시작 평가액, "deposits": [{"date", "amount"}]}
    history: {날짜: 그날 마지막 평가액} (`kis.record_equity`)
    """
    equity = account["equity"]
    today = account["updated_at"][:10]
    deposits = capital.get("deposits", [])
    invested = capital["base"] + sum(d["amount"] for d in deposits)
    # 전액 출금 뒤에는 순투입액이 0 이하가 되므로 넣은 돈 합계로 나눈다
    base = invested if invested > 0 else capital["base"] + sum(
        d["amount"] for d in deposits if d["amount"] > 0
    )
    out = {
        "since": capital["since"],
        "invested": invested,
        "total": equity - invested,
        "total_pct": (equity - invested) / base * 100,
        "bot": bot,
        "bot_pct": bot / base * 100,
    }
    prev_days = [d for d in history if capital["since"] <= d < today]
    # 시작일 당일은 전날 마감 대신 시작 평가액과 비교한다
    prev = (
        history[max(prev_days)]
        if prev_days
        else (capital["base"] if today == capital["since"] else None)
    )
    added = sum(d["amount"] for d in deposits if d["date"] == today)
    if prev and prev + added > 0:
        out["today"] = equity - prev - added
        out["today_pct"] = out["today"] / (prev + added) * 100
    return out


INCOME_FILE = "binance_income.json"


def merge_income(
    stored: dict | None, income: dict | None, income_range: list | None
) -> tuple[dict | None, list | None]:
    """쌓아 둔 날짜별 income(`binance_income.json`: {"from", "days"})과 지금 스냅숏 값을 합친다.

    스냅숏(`binance.snapshot`)은 최근 89일만 받아서, 그 전 날은 쌓아 둔 파일에서 읽는다.
    범위 안 날은 스냅숏 값이 이긴다. 돌려주는 범위는 [가장 이른 첫 날, 스냅숏의 마지막 날]."""
    lo, hi = (list(income_range or []) + [None, None])[:2]
    days = dict((stored or {}).get("days") or {})
    if income is not None:
        for d, v in income.items():
            if (lo is None or d >= lo) and (hi is None or d < hi):
                days[d] = v
    first = (stored or {}).get("from")
    if lo is not None:
        first = min(first, lo) if first else lo
    if income is None and not first:
        return None, None
    return days, [first, hi]


def daily_returns(
    capital: dict,
    today: str,
    income: dict | None = None,
    income_range: list | None = None,
    unrealized: float | None = None,
) -> list[dict]:
    """일별 수익률(최신이 위), **확정 손익 기준**(사용자 2026-11-17 "일별수익률은 매매 확정된거
    기준으로 다시 해. 포지션 들고있는거로 하면 안되지").

    그날(KST) 손익 = 바이낸스 income의 그날 실현손익(REALIZED_PNL, 그날 체결된 청산분) + 수수료
    (COMMISSION, 음수) + 펀딩(FUNDING_FEE). 들고 있는 포지션의 평가손익은 넣지 않는다.
    수익률 = 그날 손익 ÷ 그날 시작 확정 평가액(시드 + 그날까지 입출금 + 전날까지 확정 손익 누적).
    입출금(TRANSFER)은 income에 넣지 않으므로 손익에 섞이지 않는다. 시드 날은 `since_at` 뒤만.
    오늘 줄은 지금까지 확정분이고, 보유 중 평가손익은 `unrealized`에 따로 둔다(수익률에 안 넣음).

    income: {날짜: {"realized", "funding", "commission"}} (`merge_income`).
    income_range: [첫 날, 마지막 날 또는 None]. 범위 밖 날은 손익 None(모름)이고 누적에 안 들어간다.
    """
    since = capital.get("since")
    if not since or today < since:
        return []
    lo, hi = (list(income_range or []) + [None, None])[:2]
    deps = capital.get("deposits", [])
    base = float(capital.get("base") or 0)
    out, cum = [], 0.0
    day = datetime.strptime(since, "%Y-%m-%d")
    end = datetime.strptime(today, "%Y-%m-%d")
    while day <= end:
        d = day.strftime("%Y-%m-%d")
        # 첫날은 시드 전 날짜로 적힌 입출금도 시드에 더한다(예전 계산과 같음)
        added = sum(x["amount"] for x in deps if x["date"] == d or (d == since and x["date"] < d))
        invested = base + sum(x["amount"] for x in deps if x["date"] <= d)
        start = invested + cum
        known = income is not None and (lo is None or d >= lo) and (hi is None or d < hi)
        inc = (income or {}).get(d, {}) if known else {}
        parts = {k: (float(inc.get(k, 0.0)) if known else None)
                 for k in ("realized", "commission", "funding")}
        pnl = sum(parts.values()) if known else None
        row = {
            "date": d,
            "basis": "realized",
            "start": start,
            "pnl": pnl,
            "pct": pnl / start * 100 if pnl is not None and start > 0 else None,
            "equity": start + (pnl or 0.0),
            "deposit": added,
            **parts,
        }
        if pnl is not None:
            cum += pnl
        row["cum_pnl"] = cum
        if d == today:
            row["open"] = True
            row["unrealized"] = unrealized
        out.append(row)
        day += timedelta(days=1)
    return out[::-1]


def binance_daily(folder: Path, data: dict, capital: dict) -> list[dict]:
    """계좌 탭·일일 복기가 같이 쓰는 바이낸스 일별(확정 손익). data = `binance.json` 스냅숏."""
    path = folder / INCOME_FILE
    try:
        stored = json.loads(path.read_text()) if path.exists() else None
    except (OSError, ValueError):
        stored = None
    income, rng = merge_income(stored, data.get("income_daily"), data.get("income_range"))
    return daily_returns(capital, data["updated_at"][:10], income, rng, data.get("unrealized"))


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
        if a.get("kind") == "inv":  # 숏·인버스는 하지 않는다 (2026-10-30)
            continue
        v = a.get("verify", {}).get("long", {})
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
