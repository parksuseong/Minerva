"""BTCUSDT 일봉에 국내 실전 규칙(SCR 1단 + VIX 관문 + 주·월봉 정렬) 롱과 그 거울 숏.

등록: 2026-11-06-btc-stock-rule-vix-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:<Minerva>/src:docs/backtests/raw python docs/backtests/raw/btc_stock_rule.py \
    --out <폴더> [--offset 8] [--futures] [--perm]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import load5

from minerva.strategy import backtest as bt
from minerva.strategy import daily as dm
from minerva.strategy.indicators import Bar, crossed_down, crossed_up, sma, stoch_fast_k, tm_states
from minerva.strategy.report import parse_yahoo

DAY = 86_400_000
H = 3_600_000
SPLIT = date(2024, 1, 1)
RULE = dm.TM_RULE
GATE_OPEN = ("R0", "R1")


def daily_bars(d5: dict, offset_h: int) -> tuple[list[Bar], list[int]]:
    """5분봉 → (UTC 0시 + offset_h) 시작 일봉. 다 차지 않은 날(288개 미만)은 버린다."""
    g = ((d5["t"] - offset_h * H) // DAY).astype(np.int64)
    u, first = np.unique(g, return_index=True)
    last = np.r_[first[1:], len(g)] - 1
    hi = np.maximum.reduceat(d5["h"], first)
    lo = np.minimum.reduceat(d5["l"], first)
    bars, start = [], []
    for k in range(len(u)):
        if last[k] - first[k] + 1 < 288:
            continue
        day = datetime.fromtimestamp(int(u[k]) * DAY / 1000, timezone.utc).date()
        bars.append(Bar(day, float(d5["o"][first[k]]), float(hi[k]), float(lo[k]), float(d5["c"][last[k]])))
        start.append(int(u[k]) * DAY + offset_h * H)
    return bars, start


def signals(bars: list[Bar], ctx: bt.Context, side: int, vix: str) -> list[bool]:
    """side=1 롱(실전 규칙), -1 숏(거울). vix: stock(롱 calm, 숏 mirror) / calm / mirror / none."""
    closes = [b.close for b in bars]
    ma50 = sma(closes, RULE.above_ma)
    st = tm_states(bars, cci_n=RULE.cci_n, window=RULE.window)
    out = []
    for i, b in enumerate(bars):
        lvl = st[i].long_level if side > 0 else st[i].short_level
        prev = (st[i - 1].long_level if side > 0 else st[i - 1].short_level) if i else 0
        ok = lvl >= RULE.level and prev < RULE.level and ma50[i] is not None
        if ok:
            ok = b.close > ma50[i] if side > 0 else b.close < ma50[i]
        mode = vix if vix != "stock" else ("calm" if side > 0 else "mirror")
        reg = ctx.regime.get(b.day)
        if ok and mode == "calm":
            ok = reg in GATE_OPEN
        if ok and mode == "mirror":
            ok = reg is not None and reg not in GATE_OPEN
        if ok:
            w, m = ctx.weekly_up.get(b.day), ctx.monthly_up.get(b.day)
            ok = (w is True and m is True) if side > 0 else (w is False and m is False)
        out.append(ok)
    return out


def walk(bars: list[Bar], sig_l: list[bool], sig_s: list[bool], stoch: list) -> list[tuple]:
    """포지션 하나. 비어 있을 때 롱 신호를 먼저 본다(롱·숏 조건은 50일선으로 서로 배타).

    청산은 backtest.simulate와 같은 순서: 손절(그 봉 포함) → 익절(진입 다음 봉부터, 종가) → 만료 → 끝.
    반환: (진입 봉, 청산 봉, 방향, 진입가, 청산가, 이유)
    """
    out, i, n = [], 0, len(bars)
    while i < n - 1:
        side = 1 if sig_l[i] else (-1 if sig_s[i] else 0)
        if not side:
            i += 1
            continue
        e = i + 1
        entry = bars[e].open
        stop_px = entry * (1 - side * RULE.stop)
        j = e
        while True:
            b = bars[j]
            if side > 0 and b.low <= stop_px:
                px, why = min(stop_px, b.open), "stop"
                break
            if side < 0 and b.high >= stop_px:
                px, why = max(stop_px, b.open), "stop"
                break
            take = crossed_down(stoch, j, 80.0) if side > 0 else crossed_up(stoch, j, 20.0)
            if j > e and take:
                px, why = b.close, "take"
                break
            if j - e + 1 >= RULE.hold:
                px, why = b.close, "time"
                break
            if j == n - 1:
                px, why = b.close, "open"
                break
            j += 1
        out.append((e, j, side, entry, px, why))
        i = j + 1
    return out


def load_funding(path: Path) -> tuple:
    t, r = [], []
    with path.open() as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            t.append(float(row[0]))
            r.append(float(row[2]))
    return np.array(t), np.array(r)


def gross(tr: tuple, start: list[int], fund) -> float:
    e, j, side, entry, px, _ = tr
    g = side * (px / entry - 1)
    if fund is not None:  # 진입 봉 시가 ~ 청산 봉 마감 사이 펀딩
        m = (fund[0] > start[e]) & (fund[0] <= start[j] + DAY)
        g -= side * float(fund[1][m].sum())
    return g


def summ(rows: list[tuple], cost: float = 0.0005) -> dict:
    """rows: (진입일, 방향, 총손익). 순손익 = 총손익 − 2×비용."""
    if not rows:
        return {"n": 0}
    net = np.array([r[2] for r in rows]) - 2 * cost
    eq = np.cumprod(1 + net)
    pk = np.maximum.accumulate(np.r_[1.0, eq])
    eq3 = np.cumprod(1 + np.maximum(3 * net, -1))
    pk3 = np.maximum.accumulate(np.r_[1.0, eq3])
    s = np.sort(net)
    return {
        "n": len(rows), "long": sum(1 for r in rows if r[1] > 0),
        "win": round(float((net > 0).mean() * 100), 1),
        "ev": round(float(net.mean() * 100), 3),
        "drop3": round(float(s[:-3].mean() * 100), 3) if len(s) > 3 else None,
        "ret": round(float((eq[-1] - 1) * 100), 1), "mdd": round(float((eq / pk[1:] - 1).min() * 100), 1),
        "ret3": round(float((eq3[-1] - 1) * 100), 1), "mdd3": round(float((eq3 / pk3[1:] - 1).min() * 100), 1),
    }


TARGETS = {  # 이름: (롱 VIX, 숏 VIX), None이면 그쪽은 안 함
    "L": ("calm", None), "S": (None, "mirror"), "LS": ("calm", "mirror"),
    "L_novix": ("none", None), "S_calm": (None, "calm"), "S_none": (None, "none"), "LS_novix": ("none", "none"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--futures", action="store_true")
    ap.add_argument("--perm", action="store_true", help="주 판정 대상에 무작위 진입 대조")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / ("btc/futures/klines_5m.csv" if args.futures else "btc/5m.csv"))
    fund = load_funding(base / "btc/futures/funding.csv") if args.futures else None
    vix = parse_yahoo(json.loads((base / "_VIX_max.json").read_text()))
    bars, start = daily_bars(d5, args.offset)
    ctx = bt.live_context(bars, vix)
    stoch = stoch_fast_k(bars)
    none = [False] * len(bars)

    # 검산: 롱 신호·거래가 실전 엔진(backtest.run)과 같아야 한다
    ref = bt.run(bars, RULE, ctx)
    mine = walk(bars, signals(bars, ctx, 1, "calm"), none, stoch)
    assert [(bars[t[0]].day, round(t[4], 6)) for t in mine] == [(t.entry_day, round(t.exit, 6)) for t in ref], \
        "long side differs from backtest.run"

    end = bars[-1].day
    rep: dict = {"offset": args.offset, "futures": args.futures, "first": str(bars[0].day), "last": str(end),
                 "vix_last": str(vix[-1].day), "check_vs_run": len(ref)}
    trades_by = {}
    for name, (lv, sv) in TARGETS.items():
        sl = signals(bars, ctx, 1, lv) if lv else none
        ss = signals(bars, ctx, -1, sv) if sv else none
        tr = walk(bars, sl, ss, stoch)
        closed = [t for t in tr if t[5] != "open"]
        rows = [(bars[t[0]].day, t[2], gross(t, start, fund)) for t in closed]
        trades_by[name] = (tr, rows)
        y1 = date.fromordinal(end.toordinal() - 365)
        rep[name] = {
            "all": summ(rows), "all_c10": summ(rows, 0.001),
            "old": summ([r for r in rows if r[0] < SPLIT]), "new": summ([r for r in rows if r[0] >= SPLIT]),
            "1y": summ([r for r in rows if r[0] >= y1]),
            "open": [[str(bars[t[0]].day), t[2], round(t[3], 1)] for t in tr if t[5] == "open"],
            "trades": [[str(bars[t[0]].day), str(bars[t[1]].day), t[2], round(t[3], 2), round(t[4], 2), t[5],
                        round(gross(t, start, fund) * 100, 2)] for t in tr],
        }

    if args.perm:  # 같은 수·같은 방향 구성의 무작위 진입, 같은 청산 규칙, 2,000회
        rng = random.Random(7)
        for name in ("L", "S", "LS"):
            _, rows = trades_by[name]
            if len(rows) < 2:
                continue
            actual = float(np.mean([r[2] for r in rows]))
            nl = sum(1 for r in rows if r[1] > 0)
            ns = len(rows) - nl
            hits = 0
            for _ in range(2000):
                sl, ss = [False] * len(bars), [False] * len(bars)
                for k in rng.sample(range(300, len(bars) - 2), nl):
                    sl[k] = True
                for k in rng.sample(range(300, len(bars) - 2), ns):
                    ss[k] = True
                tr = [t for t in walk(bars, sl, ss, stoch) if t[5] != "open"]
                if tr and np.mean([gross(t, start, fund) for t in tr]) >= actual:
                    hits += 1
            rep[name]["perm_p"] = round(hits / 2000, 4)

    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    for name in TARGETS:
        r = rep[name]
        print(name, "all", r["all"], "c10", r["all_c10"].get("ev"), "old", r["old"].get("n"), r["old"].get("ev"),
              "new", r["new"].get("n"), r["new"].get("ev"), "1y", r["1y"], r.get("perm_p", ""), flush=True)


if __name__ == "__main__":
    main()
