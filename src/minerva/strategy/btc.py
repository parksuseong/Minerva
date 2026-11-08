"""비트코인 탭: BTCUSDT 롱/숏 분석. 표시 전용이고 주문과 연결되지 않는다 (사용자 요청 2026-10-31).

봉마다 마감된 봉으로 삼중 모멘텀을 계산한다. 방향은 CCI·RSI가 기준선 위아래에 있는지로 본다.
자료는 바이낸스 USDⓈ-M 선물 공개 시세다(사용자 결정 2026-11-07: 모든 분석은 선물가격).
추격 한도(사용자 결정 2026-11-08): 신호 봉 종가에서 1% 넘게 벌어지면 들어가지 않는다.
일봉·4시간·15분 방향이 모두 같으면(큰 움직임 조건) 3%까지 허용한다.
실행: `python -m minerva.strategy.btc`
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from minerva.strategy import btc_plan
from minerva.strategy.indicators import Bar, TMState, tm_states

SYMBOL = "BTCUSDT"
URL = "https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={iv}&limit={n}"
PRICE_URL = "https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}"
LIMIT = 1000
FRAMES = (("15m", "15분봉"), ("4h", "4시간봉"), ("1d", "일봉"), ("1w", "주봉"), ("1M", "월봉"))
OUT = Path(os.environ.get("MINERVA_BTC_OUT", "~/.minerva/btc/latest.json")).expanduser()
KST = timezone(timedelta(hours=9))


def fetch(interval: str) -> list[list]:
    url = URL.format(sym=SYMBOL, iv=interval, n=LIMIT)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def fetch_price() -> float:
    """선물 최근 체결가(공개 시세, GET만)."""
    req = urllib.request.Request(
        PRICE_URL.format(sym=SYMBOL), headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return float(json.load(r)["price"])


def parse(rows: list[list], now_ms: int) -> tuple[list[Bar], list[datetime]]:
    """마감된 봉과 그 시작 시각(KST). 진행 중인 봉은 뺀다."""
    bars, opened = [], []
    for r in rows:
        if r[6] >= now_ms:
            continue
        t = datetime.fromtimestamp(r[0] / 1000, KST)
        bars.append(Bar(t.date(), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
        opened.append(t)
    return bars, opened


def direction(s: TMState) -> str:
    if s.cci is None or s.rsi is None:
        return "중립"
    if s.cci > 0 and s.rsi > 50:
        return "상방"
    if s.cci < 0 and s.rsi < 50:
        return "하방"
    return "중립"


def _r(v: float | None, n: int = 1) -> float | None:
    return None if v is None else round(v, n)


def last_signal(states: list[TMState], opened: list[datetime], side: str) -> dict | None:
    for s, t in zip(reversed(states), reversed(opened)):
        level = getattr(s, f"{side}_level")
        if level:
            return {"level": level, "at": t.strftime("%Y-%m-%d %H:%M"), "close": _r(s.close, 2)}
    return None


def frame_status(key: str, name: str, bars: list[Bar], opened: list[datetime]) -> dict:
    states = tm_states(bars)
    s = states[-1]
    return {
        "frame": key,
        "name": name,
        "at": opened[-1].strftime("%Y-%m-%d %H:%M"),
        "close": _r(s.close, 2),
        "stoch": _r(s.stoch),
        "cci": _r(s.cci),
        "rsi": _r(s.rsi),
        "long_level": s.long_level,
        "short_level": s.short_level,
        "direction": direction(s),
        "last_long": last_signal(states, opened, "long"),
        "last_short": last_signal(states, opened, "short"),
    }


def summary(frames: list[dict]) -> list[str]:
    def names(way: str) -> tuple[int, str]:
        hit = [f["name"] for f in frames if f["direction"] == way]
        return len(hit), ", ".join(hit) or "없음"

    out = ["상방 {}개({}), ".format(*names("상방")) + "하방 {}개({})예요.".format(*names("하방"))]
    for side, label in (("long", "롱"), ("short", "숏")):
        now = [f"{f['name']} {f[side + '_level']}단" for f in frames if f[side + "_level"]]
        out.append(f"{label} 신호: {', '.join(now)}." if now else f"{label} 신호는 지금 없어요.")
    return out


CHASE_PCT, CHASE_BIG_PCT = 1.0, 3.0
BIG_FRAMES = ("15m", "4h", "1d")


def entry_gate(frames: list[dict]) -> dict:
    """추격 한도. 큰 움직임 조건 = 15분·4시간·일봉 방향이 모두 상방이거나 모두 하방."""
    ways = {f["frame"]: f["direction"] for f in frames}
    picked = [ways.get(k) for k in BIG_FRAMES]
    big = picked[0] in ("상방", "하방") and all(w == picked[0] for w in picked)
    pct = CHASE_BIG_PCT if big else CHASE_PCT
    close = next((f["close"] for f in frames if f["frame"] == "15m"), None)
    return {
        "big": big,
        "big_way": picked[0] if big else None,
        "pct": pct,
        "ref_close": close,
        "long_max": _r(close * (1 + pct / 100), 1) if close else None,
        "short_min": _r(close * (1 - pct / 100), 1) if close else None,
    }


def build(now: datetime | None = None) -> dict:
    now = now or datetime.now(KST)
    now_ms = int(now.timestamp() * 1000)
    frames, raw = [], {}
    for key, name in FRAMES:
        bars, opened = parse(fetch(key), now_ms)
        if len(bars) < 30:
            raise ValueError(f"{key} 봉이 {len(bars)}개뿐이다")
        frames.append(frame_status(key, name, bars, opened))
        raw[key] = (bars, opened)
    price = fetch_price()
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "symbol": SYMBOL,
        "frames": frames,
        "summary": summary(frames),
        "entry_gate": entry_gate(frames),
        "plan": btc_plan.today(frames, *raw["1d"], raw["4h"][0], price),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)
    data = build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
