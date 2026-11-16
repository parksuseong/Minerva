"""자동매매 일일 복기(사용자 요청 2026-11-15 "일일 복기 페이지도 만들어라 너가 복기해봐").

하루(KST 0시~24시)의 엔진 기록을 모아 `reviews/<날짜>.json`에 쓰고, 웹 "📝 복기" 탭이 보여준다.
매일 04:05 KST에 전날 것을 쓴다(`minerva-btc-review.timer`, 늦은 밤 거래의 청산 뒤 4시간까지 보이게)

읽는 것: 엔진 파일(closed.jsonl·events.jsonl, 키 없음), 계좌 스냅숏(binance.json 등, 키 없음),
공개 시세 15분봉(GET, 거래한 종목만, 거래마다 1번). 주문·키 없음.

AI 복기글은 `claude -p`(도구·MCP 없음)로 받는다. 실패하면 숫자 요약만 남긴다.
복기에서 나온 생각은 규칙이 아니다. 글에는 "검증 안 됨"으로 붙고, 규칙 변경은
백테스트·레드팀·사용자 결정을 거친다.

실행: `python -m minerva.btc_review [--day YYYY-MM-DD] [--no-ai] [--dry]`
(날짜를 안 주면 어제, --dry는 파일을 쓰지 않고 출력만)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.request
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from minerva import btc_ai, perf
from minerva import btc_trade as bt
from minerva.kis import KST, _write_private

MODEL = os.environ.get("MINERVA_REVIEW_MODEL", "opus")
TIMEOUT = 240
BUDGET_USD = "1.00"
CAP = 30  # 목록마다 상한(질문은 명령줄 한 줄로 넘어가서 너무 길면 실패한다)
KEEP = 60  # 화면에 보내는 최근 복기 수
AFTER_BARS = 16  # 청산 뒤 4시간(15분봉 16개)
FMT = "%Y-%m-%d %H:%M:%S"
DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# 승패(btc_trade.outcome): 진입가 ±DRAW_PCT% 안에서 난 손절은 무승부
RESULT_KO = {"win": "승", "loss": "패", "draw": "무승부"}

WHY = {
    "signal": "반대 신호",
    "stop": "손절",
    "trail": "따라간 손절",
    "tp": "익절",
    "manual": "수동·강제",
    "protect": "손절 실패 정리",
    "weekend": "휴장 전 정리",
}


def review_dir() -> Path:
    return bt.DIR / "reviews"


def account_dir() -> Path:
    default = "~/.minerva/account/latest.json"
    return Path(os.environ.get("MINERVA_ACCOUNT_OUT", default)).expanduser().parent


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _on(day: str, rec: dict) -> bool:
    return str(rec.get("at", "")).startswith(day) and rec.get("mode", "live") == "live"


def _ms(kst: str) -> int:
    return int(datetime.strptime(kst, FMT).replace(tzinfo=KST).timestamp() * 1000)


def fetch_bars(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """15분봉(공개 시세). 엔진과 같은 IP라 필요한 만큼만(499개 이하 무게 2, 1000개 무게 5).

    1000개(약 10일)보다 긴 보유는 앞쪽이 잘린다. 429·418은 HTTPError로 올라가 그 거래만 빈칸."""
    need = (end_ms - start_ms) // bt.BAR_MS + 2
    limit = 499 if need <= 499 else 1000
    if need > limit:
        start_ms = end_ms - (limit - 1) * bt.BAR_MS
    url = (
        f"{bt.BASE}/fapi/v1/klines?symbol={symbol}&interval=15m"
        f"&startTime={start_ms}&endTime={end_ms}&limit={limit}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "minerva-review"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def excursion(
    trade: dict, bars: list[list], partial: bool = False, after_bars: int = AFTER_BARS
) -> dict:
    """들고 있는 동안 가장 좋았던·나빴던 가격(진입가 대비 %, 내 방향 기준)과 청산 뒤 4시간.

    청산 시각은 엔진이 기록한 때라 손절·익절 실제 체결보다 늦을 수 있다. 그래서 손절·익절은
    진입 뒤 청산가에 처음 닿은 봉까지(청산가를 지난 움직임은 세지 않음)를 보유 구간으로 본다.
    그 밖의 청산(신호·수동 등)과 못 닿은 손절·익절은 기록 시각 전에 마감한 봉까지 본다
    (엔진은 봉 마감 20초 뒤에 청산하므로 기록 시각이 든 봉은 청산 뒤다).
    partial이면 청산 뒤 봉이 4시간보다 적어도 있는 만큼 보고 `bars`(봉 수)·`partial`을 붙인다
    (거래별 복기는 청산 1시간 뒤에 쓴다). bars에는 마감한 봉만 넣는다.
    """
    entry, exit_px = float(trade["entry"]), float(trade["exit"])
    sign = 1 if trade["side"] == "long" else -1
    start, rec = _ms(trade["opened_at"]), _ms(trade["at"])
    by_level = trade.get("why") in ("stop", "trail", "tp")
    held = [b for b in bars if int(b[0]) + bt.BAR_MS > start]
    best = worst = 0.0
    after_i = len(held)  # 청산 뒤 첫 봉
    for i, b in enumerate(held):
        # 엔진은 봉 마감 20초 뒤에 돌아 기록하므로, 기록 1분 안에 마감한 봉까지가 보유 구간이다
        if int(b[0]) + bt.BAR_MS > rec + 60_000:
            after_i = i
            break
        hi, lo = float(b[2]), float(b[3])
        good = ((hi if sign > 0 else lo) / entry - 1) * sign * 100
        bad = ((lo if sign > 0 else hi) / entry - 1) * sign * 100
        touched = by_level and i > 0 and lo <= exit_px <= hi
        if touched:
            got = (exit_px / entry - 1) * sign * 100
            good, bad = (got, bad) if got > 0 else (good, got)
        best, worst = max(best, good), min(worst, bad)
        if touched:
            after_i = i + 1
            break
    out = {"mfe_pct": round(best, 2), "mae_pct": round(worst, 2), "after": None}
    nxt = held[after_i : after_i + after_bars]
    if len(nxt) == after_bars or (partial and nxt):
        hi = max(float(b[2]) for b in nxt)
        lo = min(float(b[3]) for b in nxt)
        last = float(nxt[-1][4])
        out["after"] = {
            "high_pct": round((hi / exit_px - 1) * 100, 2),
            "low_pct": round((lo / exit_px - 1) * 100, 2),
            "close_pct": round((last / exit_px - 1) * 100, 2),
        }
        if partial:
            out["after"].update(bars=len(nxt), partial=len(nxt) < after_bars)
    return out


def account_day(day: str, folder: Path | None = None) -> dict | None:
    """계좌 탭 일별 수익률과 같은 계산(확정 손익 기준: 실현손익 − 수수료 + 펀딩)으로 그날 한 줄."""
    folder = folder or account_dir()
    data = _read(folder / "binance.json")
    capital = _read(folder / "binance_capital.json")
    if not data or not capital or data.get("status") != "ok" or capital.get("live") is False:
        return None
    rows = perf.binance_daily(folder, data, capital)
    return next((r for r in rows if r["date"] == day), None)


def facts(
    day: str, now: datetime, fetch: Callable = fetch_bars, folder: Path | None = None
) -> dict:
    events = [e for e in _jsonl(bt.DIR / "events.jsonl") if _on(day, e)]
    runs = [e for e in events if e.get("event") == "run"]
    trades = []
    for t in _jsonl(bt.DIR / "closed.jsonl"):
        if not _on(day, t) or not t.get("opened_at"):
            continue
        row = {k: t.get(k) for k in ("symbol", "side", "qty", "entry", "exit", "pnl", "pct", "why")}
        row["leverage"] = bt.pos_leverage(t)  # 그 거래를 연 배율(옛 기록은 3배)
        why = t.get("why")
        row.update(opened_at=t["opened_at"], closed_at=t["at"], why_ko=WHY.get(why, why))
        row["result"] = RESULT_KO.get(bt.outcome(t))
        row["hold_min"] = round((_ms(t["at"]) - _ms(t["opened_at"])) / 60000)
        try:
            bars = fetch(t["symbol"], _ms(t["opened_at"]) - bt.BAR_MS,
                         _ms(t["at"]) + (AFTER_BARS + 2) * bt.BAR_MS)
            row.update(excursion(t, bars))
        except (OSError, ValueError, KeyError, IndexError):
            row.update(mfe_pct=None, mae_pct=None, after=None)
        trades.append(row)
    passed = Counter(e["symbol"] for e in events if e.get("event") == "signal_passed")
    skipped = [
        {k: e.get(k) for k in ("at", "symbol", "side", "why")}
        for e in events
        if e.get("event") == "signal_skip"
    ]
    choices = [
        {"at": e["at"], "options": e.get("options"), "choice": e.get("choice"),
         "reason": _reason(e.get("ai_raw"))}
        for e in events
        if e.get("event") == "ai_choose"
    ]
    entries = [{"at": e["at"], "decision": e.get("decision")} for e in runs if _entered(e)]
    tl = bt.tally(trades)  # 진입가 ±DRAW_PCT% 안 손절은 무승부
    return {
        "day": day,
        "partial": day >= now.strftime("%Y-%m-%d"),
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "rules": {
            "leverage": bt.LEVERAGE,  # 엔진이 지금 새로 여는 배율, 거래마다는 trades[].leverage
            "stop_pct": bt.STOP_PCT * 100,
            "tp_pct": bt.fixed_tp_pct(),
            "exit_rule": "trail" if bt.trail_rule() else "fixed",
        },
        "totals": {"n": len(trades), "wins": tl["wins"], "losses": tl["losses"],
                   "draws": tl["draws"], "winrate": tl["winrate"],
                   "pnl": round(sum(t["pnl"] or 0 for t in trades), 2)},
        # 계좌 탭과 같은 값: 확정 손익 기준(그날 체결된 실현손익 − 수수료 + 펀딩,
        # 보유 중 평가손익 제외)
        "account": account_day(day, folder),
        "trades": trades,
        "entries": entries[:CAP],
        "choices": choices[:CAP],
        "skipped": skipped[:CAP],
        "skipped_total": len(skipped),
        "passed": dict(passed.most_common(15)),
        "passed_total": sum(passed.values()),
        "runs": len(runs),
        "halts": sorted({h for e in runs for h in (e.get("halts") or [])}),
        "end_position": runs[-1].get("position") if runs else None,
    }


ENTERED = re.compile(r"^\S+ (롱|숏) 진입$")


def _entered(run: dict) -> bool:
    """실제로 들어간 실행만: 엔진이 진입을 확인했을 때 쓰는 "<종목> <방향> 진입"(btc_trade).
    "… 신호, 진입 안 함"·"진입 못 함" 같은 문구는 맞지 않는다."""
    head = str(run.get("decision") or "").split(" · ")[0]
    return bool(ENTERED.match(head))


def _reason(raw) -> str | None:
    try:
        return str(json.loads(raw).get("reason"))[:200]
    except (TypeError, ValueError, AttributeError):
        return None


PROMPT = """너는 바이낸스 USDⓈ-M 선물 자동매매 엔진의 하루를 복기하는 트레이더다.
엔진 규칙(바꾸지 않음): 15분봉 SCR ① 신호, 배율은 거래마다 trades[].leverage(2026-11-17부터 새
진입은 rules.leverage배, 그 전 3배), 손절 진입가 -3%, 익절 +3%
(2026-11-16부터: 익절 없이 +2% 닿으면 손절을 본전으로, 그 뒤 최고가에서 3% 따라감, why=trail),
한 번에 한 포지션,
반대 신호면 바꿈, 백테스트 손실·7일 평균 하루 변동폭 2% 미만 종목은 신호만 표시
(2026-11-16부터 승률은 안 봄).
자료(JSON) 칸: trades(그날 닫힌 거래, mfe_pct=들고 있는 동안 가장 좋았던 %, mae_pct=가장 나빴던 %,
after=청산 뒤 4시간 가격 변화 %, result=승/패/무승부: 진입가 ±0.5% 안 손절은 무승부),
totals(wins·losses·draws, winrate=이긴 수/(이긴 수+진 수), 무승부 제외),
account(확정 손익 기준 일 손익: 그날 체결된 실현손익 − 수수료 + 펀딩, 보유 중 평가손익은 안 넣음.
계좌 전체라 손으로 한 거래·펀딩만큼 trades 합계와 다를 수 있다), entries(진입),
choices(AI 종목 선택),
closed_at은 엔진이 기록한 시각(손절·익절 실제 체결은 그보다 최대 15분 이를 수 있음),
skipped(시장 쉬는 시간 등으로 못 들어간 신호), passed(후보 아닌 종목 신호 수),
partial(하루가 안 끝남).

규칙:
- 한국어, 짧고 쉬운 문장. 숫자는 자료에 있는 것만 쓴다. 추측하지 않는다.
- 결과가 아니라 과정을 본다: 규칙대로 했는지, 운이었는지, 반복될 문제인지.
- 하루 결과로 규칙을 바꾸자고 하지 않는다.
  생각이 있으면 ideas에만 쓰고 끝에 "(검증 안 됨)"을 붙인다.
- 거래가 없으면 없다고 쓰고 신호·대기 상황만 본다.

다른 말 없이 JSON 하나만 답한다:
{"headline": "한 줄 요약", "good": ["잘한 점"], "bad": ["아쉬운 점"],
 "rule_check": "규칙대로 했는지 한두 문장", "tomorrow": ["내일 볼 것"],
 "ideas": ["... (검증 안 됨)"]}
각 목록은 0~4개.

자료:
"""

KEYS = ("headline", "good", "bad", "rule_check", "tomorrow", "ideas")


def parse(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("headline"), str):
        return None
    out = {
        "headline": obj["headline"].strip()[:200],
        "rule_check": str(obj.get("rule_check", ""))[:400],
    }
    for k in ("good", "bad", "tomorrow", "ideas"):
        v = obj.get(k) or []
        out[k] = [str(x)[:300] for x in v[:4]] if isinstance(v, list) else []
    out["ideas"] = [x if "검증 안 됨" in x else f"{x} (검증 안 됨)" for x in out["ideas"]]
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
    got = parse(raw)
    return got, None if got else "답을 읽지 못함"


def build(day: str, now: datetime, ai: bool = True, caller=None, fetch=fetch_bars,
          folder: Path | None = None) -> dict:
    data = facts(day, now, fetch, folder)
    review, err = ask(data, caller) if ai else (None, "AI 끔")
    return {**data, "review": review, "review_error": err}


def save(rec: dict) -> Path:
    path = review_dir() / f"{rec['day']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_private(path, json.dumps(rec, ensure_ascii=False, indent=1))
    return path


def load_all(limit: int = KEEP) -> list[dict]:
    """최근 복기(최신이 위). 깨진 파일은 건너뛴다."""
    folder = review_dir()
    if not folder.exists():
        return []
    out = []
    for p in sorted(folder.glob("*.json"), reverse=True):
        if not DAY.match(p.stem):
            continue
        rec = _read(p)
        if isinstance(rec, dict):
            out.append(rec)
        if len(out) >= limit:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", help="YYYY-MM-DD (KST), 기본 어제")
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args(argv)
    now = datetime.now(KST)
    day = args.day or (now - timedelta(days=1)).strftime("%Y-%m-%d")
    if not DAY.match(day):
        ap.error("날짜는 YYYY-MM-DD")
    rec = build(day, now, ai=not args.no_ai)
    if args.dry:
        print(json.dumps(rec, ensure_ascii=False, indent=1))
    else:
        print(save(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
