"""자동매매 거래별 복기(사용자 요청 2026-11-16 "매 거래당 복기를 해라 내가 수동으로 진입/청산을
한것도 교훈으로 삼을 수 있도록").

실전 시작(`binance_capital.json`의 `since_at`) 뒤 닫힌 실전 거래(`closed.jsonl`)마다 한 번,
다음 날 아침에 `reviews/trades/<거래 ID>.json`을 쓰고 텔레그램으로 짧게 알린다(사용자 2026-11-16
"복기는 매번하지말고 다음날하자 이유는 그다음에 어떻게움직인지를 알아야지"). 청산 날짜(KST)가
오늘보다 이전인 거래만, 청산 뒤 최대 24시간 움직임을 같이 본다.
`minerva-btc-trade-review.timer`(매일 09:20 KST, 엔진 :15:20 실행과 겹치지 않게).

읽는 것: 엔진 파일(closed.jsonl·events.jsonl, 키 없음), 시드 파일(since_at), 공개 시세 15분봉(GET,
거래마다 1번). 주문·키 없음. 사용자가 손으로 한 것(손으로 연 포지션을 엔진이 이어받음, 손으로
정한 익절, 보유 중 수량 변경, 엔진 밖 청산)은 따로 모아 AI가 정직하게 평가하게 한다.
엔진 두 실행 사이에 손으로 열고 닫은 거래는 엔진이 보지 못해 복기에도 없다.

가정 결과(counterfactual)는 계산이다: 같은 진입에서 지금 규칙(따라가는 손절 또는 고정 익절)과,
손으로 익절을 정했다면 고정 +3% 익절이 어떻게 됐을지 15분봉으로 본다. 한 봉 안 순서는 모르므로
불리한 쪽(손절)을 먼저 본다. 복기에서 나온 교훈은 규칙이 아니다("검증 안 됨").

실행: `python -m minerva.btc_trade_review [--no-ai] [--limit N]`
(--no-ai는 AI·텔레그램 없이 숫자 기록만 쓴다)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

from minerva import btc_ai, btc_review
from minerva import btc_trade as bt
from minerva.btc_review import FMT, _jsonl, _ms, _read, fetch_bars
from minerva.kis import KST, _write_private

MODEL = os.environ.get("MINERVA_REVIEW_MODEL", "opus")
TIMEOUT = 150  # 5개 × 150초가 systemd 900초 안에 들게
BUDGET_USD = "0.60"
AFTER_BARS_NEXT_DAY = 96  # 청산 뒤 24시간(15분봉 96개)까지 본다(다음 날 복기)
LIMIT = 12  # 하루 한 번 쓰는 복기 수(AI 비용 상한, 하루 거래가 이보다 많으면 다음 날 이어서)
KEEP = 60  # 화면에 보내는 최근 거래 복기 수
EVENTS_CAP = 15
ID = re.compile(r"^\d{8}-\d{4}_[A-Z0-9]+_\d{8}-\d{4}$")
WHY = {**btc_review.WHY, "trail": "따라간 손절"}
SIDE = {"long": "롱", "short": "숏"}
ENTRY_EVENTS = {
    "ai_choose", "ai_exits", "adopt_manual", "adopt_pending", "entry_from_exchange",
    "fill_mismatch",
}
HOLD_EVENTS = {
    "adopt_manual", "tp_custom_kept", "trail_on", "trail_move", "to_trail", "tp_retarget",
    "tp_wait", "pre_close", "close_chase", "exit_triggered_open",
}


def trade_dir() -> Path:
    return bt.DIR / "reviews" / "trades"


def _stamp(kst: str) -> str:
    return datetime.strptime(kst, FMT).strftime("%Y%m%d-%H%M")


def trade_id(row: dict) -> str:
    """(종목, 진입 시각, 청산 시각)으로 정하는 파일 이름용 ID(분 단위)."""
    sym = re.sub(r"[^A-Z0-9]", "", str(row.get("symbol") or bt.SYMBOL).upper())
    return f"{_stamp(row['opened_at'])}_{sym}_{_stamp(row['at'])}"


def _kst(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, KST).strftime("%Y-%m-%d %H:%M")


def live_rows() -> list[dict]:
    """실전 시작 뒤 닫힌 실전 거래(btc_view와 같은 기준)."""
    capital = _read(bt.CAPITAL_FILE) or {}
    since_at = capital.get("since_at") if capital.get("live") else None
    if not since_at:
        return []
    out = []
    for r in _jsonl(bt.DIR / "closed.jsonl"):
        if r.get("mode") != "live" or (r.get("at") or "") < since_at:
            continue
        try:
            _ms(r["at"]), _ms(r["opened_at"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(r)
    return out


def due(rows: list[dict], now: datetime, limit: int = LIMIT) -> list[dict]:
    """아직 복기가 없고 청산 날짜(KST)가 오늘보다 이전인 거래(다음 날 복기), 오래된 것부터
    limit개."""
    today = now.astimezone(KST).strftime("%Y-%m-%d")
    folder = trade_dir()
    todo = [
        r for r in rows
        if str(r["at"])[:10] < today
        and not (folder / f"{trade_id(r)}.json").exists()
    ]
    todo.sort(key=lambda r: r["at"])
    return todo[:limit]


def simulate(side: str, entry: float, bars: list[list], tp_pct: float | None = None,
             trail: bool = False) -> dict:
    """같은 진입에서 손절 STOP_PCT와 (익절 tp_pct 또는 따라가는 손절)이 어떻게 됐을지.

    한 봉 안 순서는 모르므로 손절을 먼저 본다(보수적). 따라가는 손절은 엔진처럼 봉이 마감한 뒤에
    그 봉 최고가(숏은 최저가)로 옮기고, 그 봉 종가가 이미 새 손절 밖이면 종가에 닫는다."""
    long_ = side == "long"
    sign = 1 if long_ else -1
    base = entry * (1 - sign * bt.STOP_PCT)
    stop, best, on, on_at = base, entry, False, None
    tp = entry * (1 + sign * tp_pct / 100) if tp_pct else None

    def done(result: str, b: list, px: float, note: str = "") -> dict:
        moved = abs(stop - base) > entry * 1e-6
        return {
            "result": "trail" if result == "stop" and moved else result,
            "at": _kst(int(b[0])),
            "price": round(px, 8),
            "pct": round((px / entry - 1) * sign * 100, 2),
            "trail_on_at": on_at,
            "note": note,
        }

    for b in bars:
        o, hi, lo, c = (float(x) for x in b[1:5])
        if (lo <= stop) if long_ else (hi >= stop):
            gap = (o < stop) if long_ else (o > stop)
            return done("stop", b, o if gap else stop, "시가가 이미 손절 밖" if gap else "")
        if tp is not None and ((hi >= tp) if long_ else (lo <= tp)):
            return done("tp", b, tp)
        if trail:
            best = max(best, hi) if long_ else min(best, lo)
            on_now, new = bt.trail_stop(side, entry, best, on, None)
            if on_now and not on:
                on_at = _kst(int(b[0]))
            on = on_now
            if new is not None and bt._tighter(side, new, stop):
                stop = new
            if on and ((c <= stop) if long_ else (c >= stop)):
                return done("stop", b, c, "봉 마감 때 이미 새 손절 밖이라 종가 청산")
    last = bars[-1] if bars else None
    return {
        "result": "open",
        "at": _kst(int(last[0])) if last else None,
        "price": float(last[4]) if last else None,
        "pct": round((float(last[4]) / entry - 1) * sign * 100, 2) if last else None,
        "trail_on_at": on_at,
        "note": "자료 창 안에서 결론 안 남",
    }


def rule_text() -> str:
    if bt.trail_rule():
        return f"손절 {bt.STOP_PCT * 100:g}% · {bt.trail_text()}"
    tp = bt.fixed_tp_pct()
    return f"손절 {bt.STOP_PCT * 100:g}% · " + (f"익절 +{tp:g}%" if tp else "익절 없음")


def counterfactuals(row: dict, bars: list[list]) -> dict:
    entry = float(row["entry"])
    held = [b for b in bars if int(b[0]) + bt.BAR_MS > _ms(row["opened_at"])]
    if bt.trail_rule():
        now_rule = simulate(row["side"], entry, held, trail=True)
    else:
        now_rule = simulate(row["side"], entry, held, tp_pct=bt.fixed_tp_pct())
    out = {"rule_now": {"rule": rule_text(), **now_rule}, "fixed_tp": None}
    if row.get("tp_custom"):
        alt = simulate(row["side"], entry, held, tp_pct=bt.FIXED_TP_PCT)
        out["fixed_tp"] = {"rule": f"손절 {bt.STOP_PCT * 100:g}% · 익절 +{bt.FIXED_TP_PCT:g}%",
                           **alt}
    return out


def _brief(e: dict) -> dict:
    out = {k: v for k, v in e.items() if k not in ("mode", "ai_raw")}
    if e.get("ai_raw"):
        out["reason"] = btc_review._reason(e["ai_raw"])
    return {k: (v[:200] if isinstance(v, str) else v) for k, v in out.items()}


def _mine(e: dict, sym: str) -> bool:
    return e.get("symbol") in (None, sym) and e.get("mode", "live") == "live"


def related_events(row: dict, events: list[dict]) -> tuple[list[dict], list[dict]]:
    """(진입 무렵 기록, 보유 중 기록). 진입 실행의 run 기록은 진입 시각과 같은 시각이다."""
    sym = row.get("symbol") or bt.SYMBOL
    opened, closed = row["opened_at"], row["at"]
    o_ms = _ms(opened)
    lo, hi = _kst(o_ms - bt.BAR_MS - 60_000) + ":00", _kst(o_ms + 120_000) + ":59"
    entry, hold = [], []
    for e in events:
        at = str(e.get("at") or "")
        if not _mine(e, sym):
            continue
        kind = e.get("event")
        if (kind == "run" and at == opened) or (kind in ENTRY_EVENTS and lo <= at <= hi):
            entry.append(_brief(e))
        elif kind in HOLD_EVENTS and opened < at <= closed:
            hold.append(_brief(e))
    return entry[:EVENTS_CAP], hold[-EVENTS_CAP:]


def interventions(row: dict, hold: list[dict]) -> list[str]:
    out = []
    if row.get("manual"):
        out.append("진입: 사용자가 손으로 연 포지션을 엔진이 이어받음(손절·청산은 엔진 규칙)")
    if row.get("tp_custom"):
        out.append(f"익절: 사용자가 손으로 정한 익절가 {row.get('tp')}")
    for e in hold:
        if e.get("event") == "adopt_manual":
            out.append(f"보유 중 {str(e.get('at'))[5:16]} 사용자가 수량·방향을 바꿔 엔진이 다시 "
                       f"이어받음({e.get('why') or ''})")
    if row.get("why") == "manual":
        out.append("청산: 엔진 밖에서 닫힘(사용자 수동 청산 또는 거래소 강제 청산)")
    return out


def facts(row: dict, now: datetime, fetch: Callable = fetch_bars,
          events: list[dict] | None = None) -> dict:
    if events is None:
        events = _jsonl(bt.DIR / "events.jsonl")
    sym = row.get("symbol") or bt.SYMBOL
    why = row.get("why")
    rec = {k: row.get(k) for k in ("side", "qty", "entry", "exit", "pnl", "pct", "why",
                                   "tp", "stop", "best")}
    rec.update(
        id=trade_id(row), symbol=sym, opened_at=row["opened_at"], closed_at=row["at"],
        why_ko=WHY.get(why, why), manual=bool(row.get("manual")),
        # 승/패/무승부(진입가 ±DRAW_PCT% 안 손절은 무승부, btc_trade.outcome)
        result=btc_review.RESULT_KO.get(bt.outcome(row)),
        tp_custom=bool(row.get("tp_custom")), trail=bool(row.get("trail")),
        hold_min=round((_ms(row["at"]) - _ms(row["opened_at"])) / 60000),
        generated_at=now.strftime("%Y-%m-%d %H:%M"),
        rules={"leverage": bt.LEVERAGE, "now": rule_text()},
    )
    entry_ev, hold_ev = related_events(row, events)
    rec["entry_way"] = ("사용자 수동 진입(엔진이 이어받음)" if rec["manual"]
                        else "엔진 신호 진입")
    run = next((e for e in entry_ev if e.get("event") == "run"), None)
    if run and run.get("decision"):
        rec["entry_way"] += f" · {str(run['decision'])[:120]}"
    rec.update(entry_events=entry_ev, hold_events=hold_ev,
               interventions=interventions(row, hold_ev))
    rec.update(mfe_pct=None, mae_pct=None, after=None, counterfactual=None, bars_error=None)
    try:
        now_ms = int(now.timestamp() * 1000)
        bars = fetch(sym, _ms(row["opened_at"]) - bt.BAR_MS,
                     _ms(row["at"]) + (AFTER_BARS_NEXT_DAY + 2) * bt.BAR_MS)
        bars = [b for b in bars if int(b[0]) + bt.BAR_MS <= now_ms]  # 마감한 봉만
        if row.get("exit") and row.get("entry"):
            rec.update(
                btc_review.excursion(row, bars, partial=True, after_bars=AFTER_BARS_NEXT_DAY)
            )
        if row.get("entry"):
            rec["counterfactual"] = counterfactuals(row, bars)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        rec["bars_error"] = f"{type(e).__name__}"[:100]
    return rec


PROMPT = """너는 바이낸스 USDⓈ-M 선물 자동매매의 거래 하나를 복기하는 트레이더다.
엔진 규칙(바꾸지 않음): 15분봉 SCR ① 신호, 3배, 손절 진입가 -3%, 지금 청산 규칙은 rules.now,
한 번에 한 포지션, 반대 신호면 바꿈.
사용자가 손으로 연 포지션은 엔진이 이어받아 같은 규칙으로 관리한다.
자료(JSON) 칸: pct=진입가 대비 가격 % (3배 전), pnl=USDT, why_ko=청산 이유,
result=승/패/무승부(손절이 진입가 ±0.5% 안에서 났으면 무승부, 손익은 그대로),
closed_at=엔진이 기록한 시각(손절·익절 실제 체결은 최대 15분 이를 수 있음),
mfe_pct/mae_pct=들고 있는 동안 가장 좋았던/나빴던 %,
after=청산 뒤 가격 변화 %(최대 24시간, partial이면 그보다 짧음, bars=15분봉 수),
entry_way·entry_events=어떻게 들어갔나, hold_events=보유 중 엔진 기록,
interventions=사용자가 손으로 한 것, manual=손으로 연 포지션, tp_custom=손으로 정한 익절,
counterfactual.rule_now=같은 진입에 지금 규칙이었다면(계산, 봉 안 순서는 손절 먼저),
counterfactual.fixed_tp=손으로 정한 익절 대신 고정 +3% 익절이었다면(계산).
result: stop=손절, trail=따라간 손절, tp=익절, open=자료 창 안에서 결론 안 남.

규칙:
- 한국어, 짧고 쉬운 문장. 숫자는 자료에 있는 것만 쓴다. 추측하지 않는다.
- 결과가 아니라 과정을 본다: 규칙대로였나, 운이었나, 반복될 문제인가.
- interventions가 있으면 manual_note에 사용자의 손 결정(진입·익절·청산)을 정직하게 평가한다.
  잘했으면 잘했다, 엔진 규칙이나 가정 결과보다 나빴으면 나빴다고 숫자로 쓴다. 없으면 빈 문자열.
- 거래 하나로 규칙을 바꾸자고 하지 않는다. lesson은 관찰로 쓰고 끝에 "(검증 안 됨)"을 붙인다.

다른 말 없이 JSON 하나만 답한다:
{"headline": "한 줄 요약", "what": "무슨 일이 있었나 두세 문장", "good": ["잘한 점"],
 "bad": ["아쉬운 점"], "lesson": "... (검증 안 됨)", "manual_note": "",
 "rule_check": "규칙대로였나 한 문장"}
good·bad는 0~3개.

자료:
"""

KEYS = ("headline", "what", "good", "bad", "lesson", "manual_note", "rule_check")


def parse(raw: str, manual: bool = True) -> dict | None:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("headline"), str):
        return None
    out = {"headline": obj["headline"].strip()[:200]}
    for k, cap in (("what", 600), ("lesson", 300), ("manual_note", 500), ("rule_check", 300)):
        v = obj.get(k)
        out[k] = str(v).strip()[:cap] if isinstance(v, (str, int, float)) else ""
    for k in ("good", "bad"):
        v = obj.get(k) or []
        out[k] = [str(x)[:300] for x in v[:3]] if isinstance(v, list) else []
    if out["lesson"] and "검증 안 됨" not in out["lesson"]:
        out["lesson"] += " (검증 안 됨)"
    if not manual:
        out["manual_note"] = ""
    return out


def ask(data: dict, caller: Callable[[str], str] | None = None) -> tuple[dict | None, str | None]:
    caller = caller or (lambda p: btc_ai.call_claude(
        p, TIMEOUT, cwd=bt.DIR, model=MODEL, budget=BUDGET_USD
    ))
    try:
        raw = caller(PROMPT + json.dumps(data, ensure_ascii=False))
    except subprocess.TimeoutExpired:
        return None, f"시간 초과({TIMEOUT}초)"
    except (OSError, RuntimeError, subprocess.SubprocessError) as e:
        return None, f"호출 실패: {e}"[:200]
    got = parse(raw, manual=bool(data.get("interventions")))
    return got, None if got else "답을 읽지 못함"


def build(row: dict, now: datetime, ai: bool = True, caller=None, fetch=fetch_bars,
          events: list[dict] | None = None) -> dict:
    data = facts(row, now, fetch, events)
    review, err = ask(data, caller) if ai else (None, "AI 끔")
    return {**data, "review": review, "review_error": err}


def save(rec: dict) -> Path:
    path = trade_dir() / f"{rec['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_private(path, json.dumps(rec, ensure_ascii=False, indent=1))
    return path


def message(rec: dict) -> str:
    pct = rec.get("pct")
    head = (f"📝 거래 복기 · {bt._name(rec['symbol'])} {SIDE.get(rec['side'], rec['side'])}"
            + (f" {pct:+.1f}%" if isinstance(pct, (int, float)) else "")
            + f" ({rec.get('why_ko') or '-'})"
            + (f" · {rec['result']}" if rec.get("result") else "")
            + (" · 수동" if rec.get("interventions") else ""))
    rv = rec.get("review")
    if not rv:
        return f"{head}\nAI 복기 없음: {rec.get('review_error') or '-'}"
    lines = [head, rv["headline"]]
    if rv.get("lesson"):
        lines.append(f"교훈: {rv['lesson']}")
    return "\n".join(lines)


def run(now: datetime, ai: bool = True, limit: int = LIMIT, caller=None, fetch=fetch_bars,
        send: Callable[[str], bool] | None = None) -> list[Path]:
    rows = due(live_rows(), now, limit)
    if not rows:
        return []
    events = _jsonl(bt.DIR / "events.jsonl")
    out = []
    for row in rows:
        rec = build(row, now, ai=ai, caller=caller, fetch=fetch, events=events)
        out.append(save(rec))
        if send:
            send(message(rec))
    return out


def load_trades(limit: int = KEEP) -> list[dict]:
    """최근 거래 복기(청산이 늦은 것이 위). 깨진 파일은 건너뛴다."""
    folder = trade_dir()
    if not folder.exists():
        return []
    out = []
    for p in folder.glob("*.json"):
        if not ID.match(p.stem):
            continue
        rec = _read(p)
        if isinstance(rec, dict) and isinstance(rec.get("closed_at"), str):
            out.append(rec)
    out.sort(key=lambda r: (r["closed_at"], r.get("opened_at") or ""), reverse=True)
    return out[:limit]


def main(argv: list[str] | None = None) -> int:
    from minerva import notify

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-ai", action="store_true", help="AI·텔레그램 없이 숫자 기록만")
    ap.add_argument("--limit", type=int, default=LIMIT)
    args = ap.parse_args(argv)
    if args.limit < 1:
        ap.error("--limit은 1 이상")
    send = None if args.no_ai else notify.send
    for p in run(datetime.now(KST), ai=not args.no_ai, limit=args.limit, send=send):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
