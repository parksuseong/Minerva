"""BTC 자동매매 엔진 상태(읽기 전용). 웹 서버가 `/api/btc/engine`으로 보여준다.

엔진(`btc_trade.py`)이 쓴 `~/.minerva/btc_trade/` 파일과 시드(`binance_capital.json`)만 읽는다.
키·네트워크·주문 없음. 비트코인 탭·오늘 할 일·계좌 탭이 모두 이 하나를 본다.
"""

from __future__ import annotations

import json
from datetime import datetime

from minerva import btc_trade as bt
from minerva.kis import KST

MAX_TRADES = 50


def _tail(name: str, n: int) -> list[dict]:
    path = bt.DIR / name
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _last(events: list[dict], pred) -> dict | None:
    return next((e for e in reversed(events) if pred(e)), None)


def engine_view(now: datetime | None = None) -> dict:
    now = now or datetime.now(KST)
    if not bt.DIR.exists():
        return {"status": "pending", "message": "자동매매 엔진이 아직 돌지 않았어요"}
    mode = bt.mode()
    state = bt._load(bt.DIR / "state.json", {})
    events = _tail("events.jsonl", 2000)
    runs = [e for e in events if e.get("event") == "run"]
    last_run = runs[-1] if runs else None
    last_sig = _last(runs, lambda e: e.get("signal"))
    keys = ("at", "mode", "side", "bar", "used", "fallback", "secs")
    ai = [{k: e.get(k) for k in keys} for e in events if e.get("event") == "ai_exits"]
    ai = [a for a in ai if a.get("mode") == mode][-10:]
    closed = [r for r in _tail("closed.jsonl", 500) if r.get("mode") == mode]
    capital = bt._load(bt.CAPITAL_FILE, {}) or {}
    seed = bt.seed(mode, state)
    n, wins = bt.winrate(mode, now)
    equity = last_run.get("equity") if last_run else None
    no_entry = (bt.DIR / "NO_ENTRY").read_text().strip() if bt.entries_blocked() else None
    return {
        "status": "ok",
        "mode": mode,
        "halt": bt.halted(),
        "no_entry": no_entry,
        "live_since": capital.get("since_at") if capital.get("live") else None,
        "params": {
            "scr": bt.SCR_PARAMS,
            "interval": bt.INTERVAL,
            "leverage": bt.LEVERAGE,
            "default_stop_pct": bt.STOP_PCT * 100,
            "ai_stop_range": list(bt.btc_ai.STOP_RANGE),
            "ai_tp_range": list(bt.btc_ai.TP_RANGE),
            "margin_frac": bt.margin_frac(),
            "working_type": bt.WORKING_TYPE,
        },
        "position": state.get("position"),
        "last_run": last_run,
        "last_signal": last_sig,
        "last_ai": ai[-1] if ai else None,
        "ai_recent": list(reversed(ai)),
        "gauges": {
            "seed": seed,
            "equity": equity,
            "floor": seed * bt.EQUITY_FLOOR if seed else None,
            "winrate_n": n,
            "winrate_wins": wins,
            "winrate_min": bt.WINRATE_MIN,
            "winrate_min_trades": bt.WINRATE_MIN_TRADES,
            "winrate_days": bt.WINRATE_DAYS,
        },
        "trades": list(reversed(closed[-MAX_TRADES:])),
        "dry_orders": list(reversed(_tail("dry.jsonl", 30))) if mode == "dry" else [],
    }
