"""BTC 자동매매 엔진 상태(읽기 전용). 웹 서버가 `/api/btc/engine`으로 보여준다.

엔진(`btc_trade.py`)이 쓴 `~/.minerva/btc_trade/` 파일과 시드(`binance_capital.json`)만 읽는다.
키·네트워크·주문 없음. 비트코인 탭·오늘 할 일·계좌 탭이 모두 이 하나를 본다.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from minerva import btc_trade as bt
from minerva import btc_universe
from minerva.kis import KST
from minerva.strategy import soxl

MAX_TRADES = 50


def tail_jsonl(path: Path, n: int) -> list[dict]:
    """JSONL 파일의 마지막 n줄(객체만). 쓰는 중이라 잘린 줄·깨진 줄은 건너뛴다(500 방지)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines()[-n:]:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _tail(name: str, n: int) -> list[dict]:
    return tail_jsonl(bt.DIR / name, n)


def _no_entry_reason() -> str | None:
    # exists() 확인 뒤 읽기 전에 엔진이나 사용자가 파일을 지울 수 있어 바로 읽는다
    try:
        text = (bt.DIR / "NO_ENTRY").read_text().strip()
    except FileNotFoundError:
        return None
    # 손으로 `touch`한 빈 파일도 정지다. 빈 문자열이면 화면이 정지 표시를 빠뜨린다.
    return text or "NO_ENTRY 파일 (사유 없음)"


def _last(events: list[dict], pred) -> dict | None:
    return next((e for e in reversed(events) if pred(e)), None)


def universe_symbols(now: datetime | None = None) -> list[str]:
    """엔진이 지금 쓰는 종목 범위(universe.json이 없거나 오래되면 네 종목)."""
    now = now or datetime.now(KST)
    return list(bt.universe(int(now.timestamp() * 1000))["symbols"])


_SYMBOL = re.compile(r"^[A-Z0-9]{1,20}USDT$")


def held_symbol() -> str | None:
    """엔진이 지금 든 종목(state.json, 읽기만). 옛 기록에 symbol이 없으면 BTCUSDT."""
    try:
        state = bt._load(bt.DIR / "state.json", {})
    except (OSError, ValueError, TypeError):
        return None  # 깨진 상태 파일이어도 화면은 열린다
    pos = state.get("position") if isinstance(state, dict) else None
    if not isinstance(pos, dict):
        return None
    sym = pos.get("symbol") or "BTCUSDT"
    return sym if isinstance(sym, str) and _SYMBOL.match(sym) else None


# 매매와 상관없이 바이낸스 화면에 늘 보여 줄 종목(사용자 2026-11-13 "앤트로픽 차트는 꼭 넣어놓자
# 매매를 하던 안하던", 상장 전 선물). 보기만 한다: 엔진 진입 대상(tradable)과는 따로다.
WATCH_SYMBOLS = ("ANTHROPICUSDT",)


def traded_view(now: datetime | None = None) -> dict:
    """엔진이 지금 매매하는 종목 = 새로 들어갈 수 있는 종목(tradable) + 들고 있는 종목.
    든 종목을 맨 앞에, 나머지는 엔진 우선순위(백테스트 승률 순)."""
    now = now or datetime.now(KST)
    u = bt.universe(int(now.timestamp() * 1000))
    tradable = set(u.get("tradable") or ())
    order = [s for s in u.get("priority") or () if s in tradable]
    order += sorted(tradable - set(order))
    held = held_symbol()
    if held:
        order = [held] + [s for s in order if s != held]
    order += [s for s in WATCH_SYMBOLS if s not in order]
    stats = u.get("stats") or {}
    rows = []
    for s in order:
        st = stats.get(s) or {}
        wr = st.get("winrate", u.get("winrate", {}).get(s))
        ret = st.get("ret", bt.DEFAULT_RET.get(s) if u.get("source") == "default" else None)
        rows.append(
            {
                "symbol": s,
                "name": s[:-4],
                "winrate": wr,
                "ret": ret,
                "held": s == held,
                "tradable": s in tradable,
                "watch": s in WATCH_SYMBOLS and s not in tradable and s != held,
                # COIN·EQUITY·COMMODITY 등(바이낸스 분류). 범위 파일에 없으면 None
                "underlying": st.get("underlying"),
                # /api/soxl이 큰 그림·일봉 지표 분석을 내주는 종목이면 그 설명
                "analysis": soxl.US.get(s),
            }
        )
    return {
        "status": "ok",
        "source": u.get("source"),
        "updated_at": u.get("updated_at"),
        "window_days": btc_universe.WINDOW_DAYS,
        "symbols": rows,
    }


def traded_symbols(now: datetime | None = None) -> list[str]:
    return [r["symbol"] for r in traded_view(now)["symbols"]]


def near_view(now: datetime | None = None) -> dict:
    """`near.json`(엔진이 매 실행 씀) 중 지금 매매하는 종목 줄만. 없거나 깨졌으면 pending."""
    try:
        data = json.loads((bt.DIR / "near.json").read_text())
    except FileNotFoundError:
        return {"status": "pending", "message": "엔진이 아직 신호 거리를 쓰지 않았어요"}
    except (OSError, ValueError):
        return {"status": "pending", "message": "신호 거리 파일을 읽지 못했어요"}
    if not isinstance(data, dict):
        return {"status": "pending", "message": "신호 거리 파일 모양이 달라요"}
    traded = traded_view(now)
    want = {r["symbol"] for r in traded["symbols"] if not r.get("watch")}
    held = next((r["symbol"] for r in traded["symbols"] if r["held"]), None)
    rows = data.get("rows") if isinstance(data.get("rows"), list) else []
    rows = [
        {**r, "held": r.get("symbol") == held}
        for r in rows
        if isinstance(r, dict) and isinstance(r.get("symbol"), str) and r["symbol"] in want
    ]
    return {
        **data,
        "status": "ok",
        "rows": rows,
        "traded": sorted(want, key=[r["symbol"] for r in traded["symbols"]].index),
        "window_days": traded["window_days"],
    }


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
    no_entry = _no_entry_reason()
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
            "symbols": universe_symbols(now),
        },
        # TradFi 약관 미동의로 쉬는 종목 {종목: {"at", "until_ms", "code", ...}}
        "unavailable": state.get("unavailable") or {},
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
