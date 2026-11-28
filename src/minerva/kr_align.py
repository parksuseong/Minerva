"""국장 역배→정배 감시(🌱 성장 후보 탭의 국장 칸). 주문 없음, 매수 추천 아님.

사용자 2026-11-28: 장기 역배열에서 정배열을 시도하는 국장 종목을 성장 후보에 넣고
"역배에서 정배로 바뀌면 알려줘".

- 종목: `~/.minerva/growth/kr_watch.json`(저장소에 두지 않는다) `{"stocks": [{"code", "name"}]}`.
  야후 `<code>.KS`를 먼저, 없으면 `.KQ`.
- 이평선: 일봉 종가 단순이평 5·20·50·120(원전 차트의 검정·보라·빨강·주황 선).
  정배 = 5 > 20 > 50 > 120, 역배 = 5 < 20 < 50 < 120, 나머지는 혼합.
- 알림: 마지막 확정 봉이 정배이고, 지난 `LOOKBACK`봉 안에 역배였던 날이 있고,
  알림 상태(`kr_alert.json`)의 직전 상태가 정배가 아니면 텔레그램 한 번.
  정배가 풀리면 상태를 되돌려 다음 전환 때 다시 알린다.
- 참고 칸(지표 분석 참고, 미검증):
  - 주의: 최근 3봉 고가가 볼린저(20,2) 상단 이상이고
    스토캐(5)가 최근 3봉 안에 80 아래로 꺾임
    → "볼밴 상단 + SCR 과매수 이탈" 자리. 접근하면 물릴 수 있다.
  - 매수각: 종가가 20일선 위(단기 지지 유지)이고 최근 3봉 안에 스토캐가 20 위로 돌파.
- 장 마감 뒤(평일 16:20 KST) 타이머로 돈다. 장중에 돌리면 오늘 봉은 미확정이라 빼고 판단한다.

실행: `python -m minerva.kr_align [--dry] [--notify]`
결과: `~/.minerva/growth/kr.json` → `/api/growth`의 `kr` → 화면 🌱 성장 후보 탭.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from minerva.strategy.indicators import Bar, sma, stoch_fast_k

KST = timezone(timedelta(hours=9))
MAS = (5, 20, 50, 120)
LOOKBACK = 250
BB_N, BB_K = 20, 2.0
RECENT = 3
CLOSE_HHMM = (16, 0)  # 이 시각(KST) 전에는 오늘 봉을 미확정으로 본다(15:30 마감 + 야후 지연)
MAX_AGE_DAYS = 7  # 마지막 봉이 이보다 오래되면 거래 정지·상장폐지로 보고 뺀다
SLEEP = 0.3
NOTE = (
    "관심 목록이지 매수 추천이 아닙니다."
    " 정배 전환 알림과 참고 칸은 지표 분석 참고이고 미검증입니다."
)


def growth_dir() -> Path:
    from minerva import growth

    return growth.growth_dir()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


def align(vals: list) -> str | None:
    """이평 값(짧은 것부터) → 정배·역배·혼합. 하나라도 없으면 None."""
    if any(v is None for v in vals):
        return None
    if all(a > b for a, b in zip(vals, vals[1:])):
        return "정배"
    if all(a < b for a, b in zip(vals, vals[1:])):
        return "역배"
    return "혼합"


def confirmed(bars: list[Bar], now: datetime) -> list[Bar]:
    """장 마감 전이면 오늘 봉을 뺀다(미확정)."""
    now = now.astimezone(KST)
    if bars and bars[-1].day == now.date() and (now.hour, now.minute) < CLOSE_HHMM:
        return bars[:-1]
    return bars


def analyze(bars: list[Bar]) -> dict:
    closes = [b.close for b in bars]
    ma = {n: sma(closes, n) for n in MAS}
    states = [align([ma[n][i] for n in MAS]) for i in range(len(bars))]
    k = stoch_fast_k(bars, 5)
    mid = sma(closes, BB_N)
    i = len(bars) - 1
    win = closes[i - BB_N + 1 : i + 1]
    upper = None
    if mid[i] is not None and len(win) == BB_N:
        sd = (sum((x - mid[i]) ** 2 for x in win) / BB_N) ** 0.5
        upper = mid[i] + BB_K * sd
    lo = max(0, i - LOOKBACK + 1)
    rev = [j for j in range(lo, i + 1) if states[j] == "역배"]
    since = None  # 지금 상태가 몇 봉째인지
    if states[i] is not None:
        j = i
        while j > 0 and states[j - 1] == states[i]:
            j -= 1
        since = bars[j].day.isoformat()
    recent = range(max(1, i - RECENT + 1), i + 1)
    k_cross_up = any(
        k[j - 1] is not None and k[j] is not None and k[j - 1] <= 20 < k[j] for j in recent
    )
    k_drop = any(
        k[j - 1] is not None and k[j] is not None and k[j - 1] >= 80 > k[j] for j in recent
    )
    bb_top = (
        upper is not None and max(b.high for b in bars[max(0, i - RECENT + 1) : i + 1]) >= upper
    )
    above20 = ma[20][i] is not None and closes[i] > ma[20][i]
    return {
        "date": bars[i].day.isoformat(),
        "close": closes[i],
        "state": states[i],
        "state_since": since,
        "prev_state": states[i - 1] if i > 0 else None,
        "last_reverse": bars[rev[-1]].day.isoformat() if rev else None,
        "ma": {str(n): (round(ma[n][i], 2) if ma[n][i] is not None else None) for n in MAS},
        "pairs_up": sum(
            1
            for a, b in zip(MAS, MAS[1:])
            if ma[a][i] is not None and ma[b][i] is not None and ma[a][i] > ma[b][i]
        ),
        "stoch": round(k[i], 1) if k[i] is not None else None,
        "bb_upper": round(upper, 2) if upper is not None else None,
        "caution": bool(bb_top and k_drop),
        "buy_zone": bool(above20 and k_cross_up),
    }


def fetch(code: str, fetch_bars: Callable | None = None) -> tuple[str, list[Bar]]:
    from minerva.strategy.report import fetch_yahoo

    fetch_bars = fetch_bars or (lambda t: fetch_yahoo(t, "2y", ttl=0))
    last: Exception | None = None
    for suf in (".KS", ".KQ"):
        try:
            bars = fetch_bars(code + suf)
        except Exception as e:  # noqa: BLE001 — 시장 접미사가 틀리면 야후가 404
            last = e
            continue
        if bars and (datetime.now(KST).date() - bars[-1].day).days > MAX_AGE_DAYS:
            last = RuntimeError(f"마지막 봉 {bars[-1].day} (오래됨)")
            continue
        if len(bars) >= max(MAS) + 2:
            return code + suf, bars
    raise RuntimeError(f"자료 없음({type(last).__name__ if last else '봉 부족'})")


def run(
    now: datetime | None = None, fetch_bars: Callable | None = None, sleep: Callable = time.sleep
) -> dict:
    now = now or datetime.now(KST)
    watch = read_json(growth_dir() / "kr_watch.json")
    if not isinstance(watch, dict):
        watch = {"stocks": []}
    rows, errors = [], []
    for s in watch.get("stocks", []):
        try:
            ticker, bars = fetch(s["code"], fetch_bars)
            r = analyze(confirmed(bars, now))
            rows.append(
                {
                    "code": s["code"],
                    "name": s.get("name", ""),
                    "ticker": ticker,
                    "added": s.get("added"),
                    **r,
                }
            )
        except Exception as e:  # noqa: BLE001 — 한 종목 실패가 목록 전체를 막지 않게
            errors.append({"code": s.get("code"), "name": s.get("name", ""), "error": str(e)[:80]})
        sleep(SLEEP)
    order = {"정배": 0, "혼합": 1, "역배": 2, None: 3}
    rows.sort(key=lambda r: (order.get(r["state"], 3), -r["pairs_up"]))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": NOTE,
        "mas": list(MAS),
        "rows": rows,
        "errors": errors,
    }


def flips(data: dict, state: dict) -> list[dict]:
    """알림 대상: 지금 정배, 지난 LOOKBACK봉 안에 역배 있었음, 직전 상태가 정배 아님,
    그 역배가 지난번 알린 역배보다 뒤(정배↔혼합 흔들림으로 같은 전환을 다시 알리지 않게).
    처음 보는 종목(상태에 없음)은 기억만 한다(오래전 전환을 한꺼번에 보내지 않게)."""
    out = []
    for r in data.get("rows", []):
        st = state.get(r["code"])
        if st is None or r["state"] != "정배" or not r.get("last_reverse"):
            continue
        if st.get("state") != "정배" and r["last_reverse"] > (st.get("alerted_rev") or ""):
            out.append(r)
    return out


def alert_text(rows: list[dict]) -> str:
    lines = ["🌱 국장 역배→정배 전환(일봉 5>20>50>120, 확정 봉)"]
    for r in rows:
        tag = (
            " · ⚠️볼밴 상단 과매수 이탈"
            if r["caution"]
            else " · 매수각(20일선 위+스토캐 돌파)"
            if r["buy_zone"]
            else ""
        )
        lines.append(
            f"- {r['name']}({r['code']}) {r['close']:,.0f}원, {r['state_since']}부터 정배, "
            f"마지막 역배 {r['last_reverse']}, 스토캐 {r['stoch']}{tag}"
        )
    lines.append(
        "지표 분석 참고·미검증. 볼밴 상단+과매수 이탈 자리는 피하고,"
        " 눌림 뒤 지지선 위 스토캐 돌파에서 분할로."
    )
    lines.append("관심 목록이지 매수 추천이 아닙니다.")
    return "\n".join(lines)


def notify(data: dict, state_path: Path, send: Callable | None = None) -> list[dict]:
    """전환이 있으면 한 번 보내고, 보내기에 성공해야 상태를 남긴다(실패면 다음 실행에 다시)."""
    state = read_json(state_path) or {}
    todo = flips(data, state)
    if todo:
        if send is None:
            from minerva import notify as nt

            send = nt.send
        if not send(alert_text(todo)):
            return []
    # 지금 상태를 남긴다. 역배 이력 없이 정배인 종목도 정배로 남겨
    # 같은 정배 구간에 다시 알리지 않는다
    sent = {r["code"] for r in todo}
    for r in data.get("rows", []):
        if r["state"] is None:
            continue
        old = state.get(r["code"]) or {}
        rev = r.get("last_reverse") if r["code"] in sent else old.get("alerted_rev")
        state[r["code"]] = {"state": r["state"], "date": r["date"], "alerted_rev": rev}
    write_json(state_path, state)
    return todo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry", action="store_true", help="파일·알림 없이 요약만")
    ap.add_argument("--notify", action="store_true", help="역배→정배 전환이면 텔레그램")
    a = ap.parse_args(argv)
    data = run()
    for r in data["rows"]:
        print(
            f"{r['code']} {r['name']:10} {r['state'] or '-':3} since {r['state_since']} "
            f"rev {r['last_reverse']} pairs {r['pairs_up']}/3 k {r['stoch']} "
            f"{'caution ' if r['caution'] else ''}{'buy_zone' if r['buy_zone'] else ''}"
        )
    for e in data["errors"]:
        print(f"ERR {e}", file=sys.stderr)
    if a.dry:
        return 0
    if not data["rows"] and data["errors"]:
        return 1  # 전부 실패면 지난 파일을 그대로 둔다
    write_json(growth_dir() / "kr.json", data)
    if a.notify:
        sent = notify(data, growth_dir() / "kr_alert.json")
        if sent:
            print(f"알림 보냄: {', '.join(r['name'] for r in sent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
