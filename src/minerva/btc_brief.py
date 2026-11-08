"""BTC 자동매매 아침 브리핑(텔레그램, 매일 08:30 KST, `minerva-btc-brief.timer`).

엔진 상태(`btc_view.engine_view`: state.json·events.jsonl·closed.jsonl·시드)와
공개 시세(GET)만 읽는다.
주문·키 없음. 지난 브리핑 뒤로 닫힌 거래를 모으려고 마지막 보낸 시각만 `brief.json`에 적는다.
옛 "오늘의 판" 브리핑을 대신한다(사용자 결정 2026-11-08: 화면·알림 모두 실전 엔진 기준).

실행: `python -m minerva.btc_brief [--dry]` (--dry: 보내지 않고 출력만, 시각도 안 적음)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta

from minerva import btc_trade as bt
from minerva import btc_view
from minerva.kis import KST, _write_private
from minerva.strategy import btc
from minerva.strategy.indicators import tm_states

SIDE = {"long": "롱", "short": "숏"}
WHY = {
    "signal": "반대 신호",
    "stop": "손절",
    "tp": "익절",
    "manual": "수동·강제",
    "protect": "손절 실패 정리",
}


def _brief_file():
    return bt.DIR / "brief.json"


def _pct(v: float | None, d: int = 2) -> str:
    return "-" if v is None else f"{round(v, d) + 0.0:+.{d}f}%"


def directions(now: datetime) -> dict:
    """1시간·4시간·일봉 TM 방향(마감 봉, 공개 시세). 못 받으면 '-'."""
    now_ms = int(now.timestamp() * 1000)
    out = {}
    for k in ("1h", "4h", "1d"):
        try:
            bars, _ = btc.parse(btc.fetch(k), now_ms)
            out[k] = btc.direction(tm_states(bars)[-1]) if len(bars) >= 30 else "-"
        except (OSError, ValueError, KeyError):
            out[k] = "-"
    return out


def brief_text(v: dict, now: datetime, price: float | None, ways: dict, since: str) -> str:
    if v.get("status") != "ok":
        return f"☀️ BTC 자동매매 아침 브리핑\n엔진 상태를 읽지 못했어요: {v.get('message', '')}"
    lines = [f"☀️ BTC 자동매매 아침 브리핑 ({now.strftime('%m-%d %H:%M')} KST)"]
    mode = "🔴 실전" if v["mode"] == "live" else "🧪 드라이런(주문 없음)"
    stops = []
    if v.get("halt"):
        stops.append("HALT")
    if v.get("no_entry"):
        stops.append(f"신규 진입 정지: {v['no_entry']}")
    lines.append(f"모드: {mode}" + (f" · ⛔ {' · '.join(stops)}" if stops else " · 정지 없음"))

    p = v.get("position")
    if p:
        sign = 1 if p["side"] == "long" else -1
        u = (price / p["entry"] - 1) * 100 * sign if price else None
        tp = f" · 익절 {p['tp']:,.1f} (+{p.get('tp_pct', 0):g}%)" if p.get("tp") else " · 익절 없음"
        prm = v.get("params") or {}
        lev = prm.get("leverage", 3)
        now_px = (
            f" · 지금 {price:,.1f} ({_pct(u)}, {lev}배 {_pct(u * lev if u is not None else None)})"
            if price
            else ""
        )
        lines.append(f"포지션: {SIDE[p['side']]} {p['qty']} BTC @ {p['entry']:,.1f}{now_px}")
        stop_pct = p.get("stop_pct", prm.get("default_stop_pct", 3))
        lines.append(f"  손절 {p['stop']:,.1f} (−{stop_pct:g}%){tp}")
    else:
        lines.append("포지션: 없음 (다음 신호 대기)")

    night = [t for t in v.get("trades", []) if (t.get("at") or "") > since]
    if night:
        total = sum(t.get("pnl") or 0 for t in night)
        parts = [
            f"{SIDE.get(t['side'], t['side'])} {_pct(t.get('pct'))} "
            f"({WHY.get(t.get('why'), t.get('why'))})"
            for t in reversed(night)
        ]
        lines.append(
            f"밤사이 청산 {len(night)}건: " + " · ".join(parts) + f" · 합계 {total:+.2f} USDT"
        )
    else:
        lines.append("밤사이 청산: 없음")

    g = v.get("gauges") or {}
    eq, seed, floor = g.get("equity"), g.get("seed"), g.get("floor")
    if eq is not None and seed:
        lines.append(
            f"평가액 {eq:,.2f} USDT · 시드 {seed:,.2f} 대비 "
            f"{_pct((eq / seed - 1) * 100)} ({round(eq - seed, 2) + 0.0:+.2f} USDT)"
        )
    n, w = g.get("winrate_n", 0), g.get("winrate_wins", 0)
    floor_txt = (
        f"−20%선 {floor:,.2f} (여유 {_pct((eq / floor - 1) * 100, 1)})"
        if eq and floor
        else "−20%선 -"
    )
    wr = f"{w}/{n} = {w / n:.0%}" if n else "거래 없음"
    lines.append(
        f"정지까지: {floor_txt} · {g.get('winrate_days', 30)}일 승률 {wr} "
        f"(기준 {g.get('winrate_min', 0.5):.0%}, {g.get('winrate_min_trades', 8)}회부터)"
    )

    s = v.get("last_signal")
    if s:
        lines.append(
            f"마지막 신호: {s.get('bar')} 봉 {SIDE.get(s.get('signal'), '-')} → {s.get('decision')}"
        )
    else:
        lines.append("마지막 신호: 아직 없음")
    run = v.get("last_run") or {}
    ind = run.get("indicators") or {}
    scr = (
        (
            f"Stoch {ind.get('stoch', '-')} / CCI {ind.get('cci', '-')} / RSI {ind.get('rsi', '-')}"
            f" ({run.get('bar', '-')} 봉)"
        )
        if ind
        else "-"
    )
    lines.append(
        f"방향 1h {ways.get('1h', '-')} · 4h {ways.get('4h', '-')} · 1d {ways.get('1d', '-')}"
        f" · SCR① {scr}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from minerva import notify

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true", help="보내지 않고 출력만")
    args = ap.parse_args(argv)
    now = datetime.now(KST)
    last = json.loads(_brief_file().read_text()) if _brief_file().exists() else {}
    since = last.get("last_at") or (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        price = btc.fetch_price()
    except (OSError, ValueError, KeyError):
        price = None
    text = brief_text(btc_view.engine_view(now), now, price, directions(now), since)
    print(text)
    if args.dry:
        return 0
    if not notify.send(text):
        raise SystemExit("텔레그램 전송 실패")
    _write_private(_brief_file(), json.dumps({"last_at": now.strftime("%Y-%m-%d %H:%M:%S")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
