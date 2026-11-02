"""30분봉 SCR 3단을 일·주·월 SCR 방향 정렬로 거른다. 등록: 2026-11-02-btc-scr-mtf-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_mtf.py --out <폴더>
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
from btc_scr_ma import COSTS, WIN, YEAR, load, raw_signals, run, stats

from minerva.strategy.indicators import Bar, cci, resample, rsi

DAY_MS = 86_400_000
HALF_MS = 1_800_000
FILTERS = ("F0", "F3", "WM", "M", "W", "D")
NEED = {"F0": "", "F3": "DWM", "WM": "WM", "M": "M", "W": "W", "D": "D"}


def daily_bars(data: Path, d30: dict) -> List[Bar]:
    """1d 파일에 30분봉으로 뒤를 채운다. 48봉이 다 찬 날만."""
    out: List[Bar] = []
    with (data / "1d.csv").open() as f:
        for r in csv.DictReader(f):
            day = datetime.fromtimestamp(int(r["open_time"]) / 1000, timezone.utc).date()
            out.append(Bar(day, float(r["open"]), float(r["high"]), float(r["low"]),
                           float(r["close"]), 0.0))
    last = out[-1].day
    groups: Dict[date, list] = {}
    for t, o, h, l, c in zip(d30["t"], d30["o"], d30["h"], d30["l"], d30["c"]):
        day = datetime.fromtimestamp(t / 1000, timezone.utc).date()
        if day > last:
            groups.setdefault(day, []).append((o, h, l, c))
    for day in sorted(groups):
        g = groups[day]
        if len(g) == 48:
            out.append(Bar(day, g[0][0], max(x[1] for x in g), min(x[2] for x in g), g[-1][3], 0.0))
    return out


def complete(bars: List[Bar], frame: str) -> List[Bar]:
    """마지막 묶음이 끝나지 않았으면 버린다."""
    if not bars:
        return bars
    d = bars[-1].day
    if frame == "W" and d.weekday() != 6:
        return bars[:-1]
    if frame == "M" and d.day != calendar.monthrange(d.year, d.month)[1]:
        return bars[:-1]
    return bars


def direction(bars: List[Bar]) -> tuple:
    """(마감 시각 ms, 상방, 하방). 상방은 CCI 20 > 0이고 RSI 14 > 50."""
    cc = np.array([np.nan if v is None else v for v in cci(bars, 20)], dtype=float)
    rr = np.array([np.nan if v is None else v for v in rsi([b.close for b in bars])], dtype=float)
    close_ms = np.array(
        [datetime.combine(b.day + timedelta(days=1), datetime.min.time(), timezone.utc).timestamp()
         * 1000 for b in bars]
    )
    return close_ms, (cc > 0) & (rr > 50), (cc < 0) & (rr < 50)


def states(t30: np.ndarray, frames: dict) -> dict:
    """30분봉 i가 마감될 때 이미 마감된 상위 봉의 방향."""
    out = {}
    for name, (close_ms, up, dn) in frames.items():
        j = np.searchsorted(close_ms, t30 + HALF_MS, side="right") - 1
        ok = j >= 0
        u = np.zeros(len(t30), bool)
        w = np.zeros(len(t30), bool)
        u[ok], w[ok] = up[j[ok]], dn[j[ok]]
        out[name] = (u, w)
    return out


def perm(flt: str, st: dict, n: int) -> tuple:
    lo = np.ones(n, bool)
    sh = np.ones(n, bool)
    for k in NEED[flt]:
        lo &= st[k][0]
        sh &= st[k][1]
    no = np.zeros(n, bool)
    return lo, sh, no, no


def build(d: dict, data: Path) -> dict:
    day = daily_bars(data, d)
    frames = {
        "D": direction(day),
        "W": direction(complete(resample(day, "W"), "W")),
        "M": direction(complete(resample(day, "M"), "M")),
    }
    return states(d["t"], frames)


def card(tr: tuple, a: int, b: int, cut: int) -> dict:
    ei, _, _, gr = tr
    k = ei >= a
    net = np.sort(gr[k].astype(float) - 2 * COSTS[0])
    drop1 = float((np.prod(1 + net[:-1]) - 1) * 100) if len(net) > 1 else 0.0
    return {
        "year": stats(tr, a, b, COSTS[0]),
        "year_c10": stats(tr, a, b, COSTS[1]),
        "last3": stats(tr, cut, b, COSTS[0]),
        "drop_best": round(drop1, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    p.add_argument("--rotations", type=int, default=200)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)

    d = load(data / "30m.csv")
    n = len(d["c"])
    a, b = n - YEAR, n
    cut = a + 9 * (YEAR // 12)
    sig = raw_signals(d)["S3"]
    st = build(d, data)
    rep: dict = {"window": [
        datetime.fromtimestamp(d["t"][a] / 1000, timezone.utc).isoformat(),
        datetime.fromtimestamp(d["t"][-1] / 1000, timezone.utc).isoformat(),
    ]}
    rep["share"] = {
        k: [round(float(st[k][0][a:b].mean() * 100), 1), round(float(st[k][1][a:b].mean() * 100), 1)]
        for k in "DWM"
    }
    res = {}
    for flt in FILTERS:
        res[flt] = run(d, sig, perm(flt, st, n), "O", a, b)
        rep[flt] = card(res[flt], a, b, cut)
        ei, _, sd, gr = res[flt]
        net = gr.astype(float) - 2 * COSTS[0]
        rep[flt]["long"] = [int((sd > 0).sum()), round(float((np.prod(1 + net[sd > 0]) - 1) * 100), 1)]
        rep[flt]["short"] = [int((sd < 0).sum()), round(float((np.prod(1 + net[sd < 0]) - 1) * 100), 1)]
    print(json.dumps({k: rep[k]["year"] for k in FILTERS}, ensure_ascii=False), flush=True)

    d2 = load(data / "15m.csv", offset_min=15, merge=True)
    n2 = len(d2["c"])
    a2 = n2 - YEAR
    sig2 = raw_signals(d2)["S3"]
    st2 = build(d2, data)
    rep["offset"] = {
        flt: stats(run(d2, sig2, perm(flt, st2, n2), "O", a2, n2), a2, n2, COSTS[0])
        for flt in ("F0", "F3")
    }
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))

    real = rep["F3"]["year"]["ret"]
    rng = random.Random(20261104)
    beat = 0
    vals = []
    for _ in range(args.rotations):
        sh = rng.randrange(30 * 48, (b - a) - 30 * 48)
        rolled = {}
        for k, (u, w) in st.items():
            u2, w2 = u.copy(), w.copy()
            u2[a:b], w2[a:b] = np.roll(u[a:b], sh), np.roll(w[a:b], sh)
            rolled[k] = (u2, w2)
        v = stats(run(d, sig, perm("F3", rolled, n), "O", a, b), a, b, COSTS[0])
        vals.append(v["ret"])
        beat += v["ret"] >= real
    rep["rotation"] = {"p": round((beat + 1) / (args.rotations + 1), 3),
                       "median": float(np.median(vals))}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print("done", rep["rotation"], flush=True)


if __name__ == "__main__":
    main()
