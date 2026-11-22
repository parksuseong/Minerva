"""SCR 롱 신호 알림(원전 규칙, 미검증). 확정 봉에서 새로 뜬 순간에만 텔레그램 한 통. 주문 없음.

미국장 마감 뒤(화~토 07:40 KST) `minerva-signal-watch.timer`가 돈다. 봉은 야후 일봉을 주·월로 묶고,
아직 안 끝난 주·월봉은 쓰지 않는다. 상태는 `~/.minerva/signal_watch/state.json`
(`MINERVA_SIGNAL_WATCH_DIR`). 처음 실행은 지금 상태만 기억하고 보내지 않는다.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from minerva.strategy.indicators import Bar, resample, sma, tm_states
from minerva.strategy.report import fetch_yahoo

log = logging.getLogger(__name__)

LABEL = "원전 규칙, 미검증 · 알림만, 주문 없음"
FRAME_KO = {"D": "일봉", "W": "주봉", "M": "월봉"}

# (키, 종목, 시간축, 종류, 설명)
# 종류: long3 = SCR 3단 돌파(스토캐 80 미만), turn = 월봉 스토캐 20 상향(꺾임),
#       vix3 = VIX SCR 3단 이탈(과매수에서), vixbb = VIX 주봉 종가가 볼밴 상단 위 → 안으로 복귀,
#       short3 = SCR 3단 이탈 경고
RULES: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "VIX_D3",
        "^VIX",
        "D",
        "vix3",
        "VIX 일봉 SCR 3단 이탈 → VIX 회귀 시작, 주식 롱 단타 구간(공격적)",
    ),
    (
        "VIX_WBB",
        "^VIX",
        "W",
        "vixbb",
        "VIX 주봉 종가가 볼밴 상단 안으로 복귀 → 변동성 축소, 증시 반등 가능",
    ),
    ("VIX_W3", "^VIX", "W", "vix3", "VIX 주봉 SCR 3단 이탈 확정 → VIX 회귀 시그널, 주식 롱 유리"),
    ("SPY_D3", "SPY", "D", "long3", "SPY 일봉 SCR 3단 돌파 → 공격적 롱(바닥 근처, 불확실성 큼)"),
    ("QQQ_D3", "QQQ", "D", "long3", "QQQ 일봉 SCR 3단 돌파 → 공격적 롱(바닥 근처, 불확실성 큼)"),
    ("SPY_W3", "SPY", "W", "long3", "SPY 주봉 SCR 3단 돌파 확정 → 중기 추세 전환 확인"),
    ("QQQ_W3", "QQQ", "W", "long3", "QQQ 주봉 SCR 3단 돌파 확정 → 중기 추세 전환 확인"),
    (
        "SOXX_W3",
        "SOXX",
        "W",
        "long3",
        "SOXX 주봉 SCR 3단 돌파 확정 → 원전이 기다리라던 SOXL 롱 타점",
    ),
    ("SOXX_M3", "SOXX", "M", "long3", "SOXX 월봉 SCR 3단 돌파 확정 → 원전의 '진짜 추세 전환'"),
    ("SPY_M3", "SPY", "M", "long3", "SPY 월봉 SCR 3단 돌파 확정 → 원전의 '진짜 추세 전환'"),
    (
        "SOXX_MT",
        "SOXX",
        "M",
        "turn",
        "SOXX 월봉 스토캐 20 상향 돌파 마감(꺾임) → 데드캣 졸업 첫 신호",
    ),
    (
        "AAPL_M3S",
        "AAPL",
        "M",
        "short3",
        "⚠️ 애플 월봉 SCR 3단 이탈 확정 → 원전: 아주 강력한 하락 시그널",
    ),
)

US_CLOSE_UTC = (21, 15)  # 미국장 마감(겨울 21:00 UTC) 뒤 여유


def watch_dir() -> Path:
    return Path(os.environ.get("MINERVA_SIGNAL_WATCH_DIR", "~/.minerva/signal_watch")).expanduser()


def us_today(now: datetime) -> date:
    return (now - timedelta(hours=5)).date()


def session_done(day: date, now: datetime) -> bool:
    """그 미국 거래일 일봉이 끝났나(마감 뒤이거나 이미 다음 날)."""
    if us_today(now) > day:
        return True
    return (now.hour, now.minute) >= US_CLOSE_UTC and now.date() == day


def confirmed_daily(bars: list[Bar], now: datetime) -> list[Bar]:
    if bars and not session_done(bars[-1].day, now):
        return bars[:-1]
    return bars


def _next_weekday(d: date) -> date:
    n = d + timedelta(days=1)
    while n.weekday() >= 5:
        n += timedelta(days=1)
    return n


def frame_bars(daily: list[Bar], frame: str, now: datetime) -> list[Bar]:
    """확정된 봉만. 주봉은 금요일(또는 지난 주), 월봉은 그달 마지막 평일(또는 지난 달)
    까지 끝났을 때."""
    daily = confirmed_daily(daily, now)
    if frame == "D" or not daily:
        return daily
    out = resample(daily, frame)
    last = daily[-1].day
    today = us_today(now)
    if frame == "W":
        done = last.weekday() == 4 or today.isocalendar()[:2] > last.isocalendar()[:2]
    else:
        done = _next_weekday(last).month != last.month or (today.year, today.month) > (
            last.year,
            last.month,
        )
    return out if done else out[:-1]


def condition(kind: str, bars: list[Bar]) -> tuple[bool, dict]:
    """마지막 확정 봉에서 조건이 맞는지와 보여 줄 값."""
    if len(bars) < 25:
        return False, {}
    s = tm_states(bars)
    cur = s[-1]
    info = {
        "date": cur.day.isoformat(),
        "close": round(cur.close, 2),
        "stoch": None if cur.stoch is None else round(cur.stoch, 1),
        "cci": None if cur.cci is None else round(cur.cci, 1),
        "rsi": None if cur.rsi is None else round(cur.rsi, 1),
    }
    if kind == "long3":
        return cur.long_level == 3 and (cur.stoch or 0) < 80, info
    if kind in ("vix3", "short3"):
        return cur.short_level == 3, info
    if kind == "turn":
        prev = s[-2].stoch
        return prev is not None and cur.stoch is not None and prev <= 20 < cur.stoch, info
    if kind == "vixbb":
        closes = [b.close for b in bars]
        mid = sma(closes, 20)
        up = []
        for i in range(len(closes)):
            if mid[i] is None:
                up.append(None)
                continue
            w = closes[i - 19 : i + 1]
            sd = (sum((x - mid[i]) ** 2 for x in w) / 20) ** 0.5
            up.append(mid[i] + 2 * sd)
        if up[-1] is None or up[-2] is None:
            return False, info
        info["bb_upper"] = round(up[-1], 2)
        return closes[-2] > up[-2] and closes[-1] <= up[-1], info
    raise ValueError(kind)


def message(rule: tuple, info: dict) -> str:
    key, sym, frame, kind, desc = rule
    vals = f"스토캐 {info.get('stoch')} · CCI {info.get('cci')} · RSI {info.get('rsi')}"
    bb = f" · 볼밴 상단 {info['bb_upper']}" if "bb_upper" in info else ""
    head = "🔔 롱 신호" if kind != "short3" else "🔔 하락 경고"
    return (
        f"{head} ({FRAME_KO[frame]} 확정 {info.get('date')})\n{desc}\n"
        f"{sym} 종가 {info.get('close')} · {vals}{bb}\n{LABEL}"
    )


def run(
    now: datetime | None = None, fetch: Callable = fetch_yahoo, send: Callable | None = None
) -> dict:
    now = now or datetime.now(timezone.utc)
    path = watch_dir() / "state.json"
    try:
        st = json.loads(path.read_text())
    except (OSError, ValueError):
        st = {}
    first = not st.get("rules")
    rules_st = st.setdefault("rules", {})
    cache: dict[str, list[Bar]] = {}
    sent, out = [], {}
    for rule in RULES:
        key, sym, frame, kind, _ = rule
        try:
            if sym not in cache:
                cache[sym] = fetch(sym, "10y", ttl=600)
            bars = frame_bars(cache[sym], frame, now)
            on, info = condition(kind, bars)
        except Exception as e:  # noqa: BLE001 - 한 종목 실패로 나머지를 멈추지 않는다
            log.warning("%s 점검 실패: %s", key, e)
            out[key] = {"error": type(e).__name__}
            continue
        prev = rules_st.get(key, {})
        new_bar = info.get("date") and info.get("date") != prev.get("date")
        fire = on and new_bar and not prev.get("on") and not first
        if new_bar or not prev:
            rules_st[key] = {"date": info.get("date"), "on": on}
        out[key] = {"on": on, **info}
        if fire:
            text = message(rule, info)
            if send is None:
                from minerva import notify

                send = notify.send
            if send(text):
                sent.append(key)
            else:
                rules_st[key] = prev  # 못 보냈으면 다음 실행에 다시
    st["checked_at"] = now.isoformat()
    st["last"] = out
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    tmp.replace(path)
    return {"first_run": first, "sent": sent, "state": out}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
