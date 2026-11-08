"""SCR ① 15분 갈아타기의 과거 거래 MFE/MAE 분위수 → ~/.minerva/btc_trade/ref_stats.json.

엔진이 AI에게 손절·익절 폭을 물을 때 참고로 넘긴다(실거래 기록이 짧은 동안).
신호는 엔진과 같은 `btc_trade.latest_signal` 계산(`scr_lab.signals` + `SCR_PARAMS`),
진입은 신호 다음 봉 시가, 손절 3%, 반대 신호에서 갈아타기(화면 `switchSim`과 같은 규칙).
MFE·MAE는 진입부터 청산까지 진입가 대비 최대 유리·불리 폭(%). 3% 손절이 MAE를 자른다.
국면은 진입 전날까지의 일봉 TM 방향(`btc.direction`).
자료: 바이낸스 USDⓈ-M 선물 공개 시세(GET, 키 없음), 최근 1년(BTC 1년 창 규칙).

실행: .venv/bin/python scripts/btc_ref_stats.py [--days 365]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from minerva import btc_trade as bt
from minerva.strategy import btc, scr_lab
from minerva.strategy.indicators import tm_states

URL = (
    "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval={iv}&startTime={s}&limit=1500"
)


def fetch(iv: str, start_ms: int, end_ms: int) -> list[list]:
    out: list[list] = []
    s = start_ms
    while s < end_ms:
        req = urllib.request.Request(URL.format(iv=iv, s=s), headers={"User-Agent": "minerva"})
        with urllib.request.urlopen(req, timeout=20) as r:
            page = json.load(r)
        if not page:
            break
        out += [p for p in page if int(p[6]) < end_ms]
        s = int(page[-1][0]) + 1
        time.sleep(0.2)
    return out


def q(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))], 2)


def summarize(trades: list[dict]) -> dict:
    mfe = [t["mfe"] for t in trades]
    mae = [t["mae"] for t in trades]
    ret = [t["ret"] for t in trades]
    return {
        "n": len(trades),
        "win_rate": round(sum(1 for r in ret if r > 0) / len(ret), 2) if ret else None,
        "stopped": sum(1 for t in trades if t["why"] == "stop"),
        "mfe_pct_q": {
            k: q(mfe, p) for k, p in (("25", 0.25), ("50", 0.5), ("75", 0.75), ("90", 0.9))
        },
        "mae_pct_q": {
            k: q(mae, p) for k, p in (("25", 0.25), ("50", 0.5), ("75", 0.75), ("90", 0.9))
        },
        "ret_pct_median": q(ret, 0.5),
        "bars_held_median": q([t["bars"] for t in trades], 0.5),
    }


def simulate(rows: list[list], daily: list[list]) -> list[dict]:
    h = [float(r[2]) for r in rows]
    lo = [float(r[3]) for r in rows]
    c = [float(r[4]) for r in rows]
    o = [float(r[1]) for r in rows]
    sig = scr_lab.signals(h, lo, c, bt.SCR_PARAMS)
    at: dict[int, int] = {}
    for d, k in ((1, "long"), (-1, "short")):
        for i in sig[k]:
            at[i] = 0 if i in at and at[i] != d else d
    dbars, dopen = btc.parse(daily, 1 << 62)
    dstates = tm_states(dbars)
    dmap = [(int(t.timestamp() * 1000), btc.direction(s)) for t, s in zip(dopen, dstates)]

    def regime(ms: int) -> str:
        prev = [w for t, w in dmap if t + 86_400_000 <= ms]
        return prev[-1] if prev else "중립"

    trades: list[dict] = []
    pos, entry, start, best, worst = 0, 0.0, 0, 0.0, 0.0
    reg = "중립"

    def close(j: int, px: float, why: str) -> None:
        trades.append({
            "side": "long" if pos > 0 else "short", "regime": reg, "why": why, "bars": j - start,
            "mfe": round(best * 100, 3), "mae": round(worst * 100, 3),
            "ret": round(pos * (px / entry - 1) * 100, 3),
        })  # fmt: skip

    for j in range(1, len(c)):
        d = at.get(j - 1)
        if d and d != pos:
            if pos:
                close(j, o[j], "signal")
            pos, entry, start, best, worst = d, o[j], j, 0.0, 0.0
            reg = regime(int(rows[j][0]))
        if not pos:
            continue
        fav = (h[j] / entry - 1) if pos > 0 else (1 - lo[j] / entry)
        adv = (1 - lo[j] / entry) if pos > 0 else (h[j] / entry - 1)
        best, worst = max(best, fav), max(worst, min(adv, bt.STOP_PCT))
        sp = entry * (1 - pos * bt.STOP_PCT)
        if (lo[j] <= sp) if pos > 0 else (h[j] >= sp):
            close(j, min(o[j], sp) if pos > 0 else max(o[j], sp), "stop")
            pos = 0
    return trades


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--out", type=Path, default=bt.DIR / "ref_stats.json")
    a = ap.parse_args(argv)
    end = int(time.time() * 1000)
    start = end - a.days * 86_400_000
    rows = fetch("15m", start - 300 * bt.BAR_MS, end)
    daily = fetch("1d", start - 120 * 86_400_000, end)
    trades = [t for t in simulate(rows, daily)]
    by: dict[str, dict] = {}
    for side in ("long", "short"):
        mine = [t for t in trades if t["side"] == side]
        by[f"{side}|all"] = summarize(mine)
        for reg in ("상방", "하방", "중립"):
            sub = [t for t in mine if t["regime"] == reg]
            if sub:
                by[f"{side}|{reg}"] = summarize(sub)
    fmt = "%Y-%m-%d"
    out = {
        "source": "SCR ① 15분 갈아타기 백테스트(손절 3%, 다음 봉 시가 진입, 비용 제외)",
        "period": [
            datetime.fromtimestamp(start / 1000, timezone.utc).strftime(fmt),
            datetime.fromtimestamp(end / 1000, timezone.utc).strftime(fmt),
        ],
        "regime_def": "진입 전날까지 일봉 TM 방향(CCI·RSI)",
        "made_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "by": by,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v["n"] for k, v in by.items()}, ensure_ascii=False), "→", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
