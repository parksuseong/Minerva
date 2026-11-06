"""15분봉 SCR 진입 완화 × 손절 위치. 등록: 2026-11-06-btc-scr15-entry-stop-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_entry_stop.py \
    --out <폴더> [--offset 5] [--futures] [--perm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from btc_fractal import cci, crossed, load5, recent, resample, rsi, stoch
from btc_scr_gate15 import load_funding
from btc_swing_patterns import SPLIT

ENTRIES = ("E3", "E2", "E3r")
STOPS = ("S1", "SW")
COMBOS = [(e, s) for e in ENTRIES for s in STOPS]
SW_BARS = 12
SW_BUF = 0.001
DAY = 86_400_000


def entry_signals(d: dict, kind: str) -> tuple:
    s, cc, r = stoch(d["h"], d["l"], d["c"]), cci(d["h"], d["l"], d["c"]), rsi(d["c"])
    lo, hi = (40, 60) if kind == "E3r" else (30, 70)
    lv_l = recent(crossed(s, 20, True)) & recent(crossed(cc, -100, True))
    lv_s = recent(crossed(s, 80, False)) & recent(crossed(cc, 100, False))
    if kind != "E2":
        lv_l &= recent(crossed(r, lo, True))
        lv_s &= recent(crossed(r, hi, False))
    on_l = lv_l & ~np.r_[False, lv_l[:-1]]
    on_s = lv_s & ~np.r_[False, lv_s[:-1]]
    return on_l, on_s


def stop_price(d: dict, i: int, side: int, kind: str) -> float:
    px = d["o"][i + 1]
    if kind == "S1":
        return px * (1 - 0.01 * side)
    a = max(0, i - SW_BARS + 1)
    if side > 0:
        return float(d["l"][a:i + 1].min()) * (1 - SW_BUF)
    return float(d["h"][a:i + 1].max()) * (1 + SW_BUF)


def label(d: dict, i: int, side: int, opp: np.ndarray, stop: float):
    """btc_scr_model.label과 같은 순서. 손절가만 인자로 받는다."""
    o, h, l = d["o"], d["h"], d["l"]
    n = len(o)
    e = i + 1
    px = o[e]
    if (side > 0 and stop >= px) or (side < 0 and stop <= px):
        return None
    for j in range(e, n - 1):
        if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
            fill = min(o[j], stop) if side > 0 else max(o[j], stop)
            return (fill / px - 1) * side, j, abs(stop / px - 1), True
        if opp[j]:
            return (o[j + 1] / px - 1) * side, j + 1, abs(stop / px - 1), False
    return None


def with_funding(g: float, d: dict, i: int, x: int, side: int, fund) -> float:
    if fund is None:
        return g
    m = (fund[0] > d["t"][i + 1]) & (fund[0] <= d["t"][x])
    return g - side * float(fund[1][m].sum())


def summ(ts: list, cost: float = 0.0005) -> dict:
    if not ts:
        return {"n": 0}
    net = np.array([x[2] for x in ts]) - 2 * cost
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    eq3 = np.cumprod(1 + np.maximum(3 * net, -1))
    peak3 = np.maximum.accumulate(np.r_[1.0, eq3])
    s = np.sort(net)
    return {"n": len(ts), "win": round(float((net > 0).mean() * 100), 1),
            "ev": round(float(net.mean() * 100), 3), "ret": round(float((eq[-1] - 1) * 100), 1),
            "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
            "ret3": round(float((eq3[-1] - 1) * 100), 1),
            "mdd3": round(float((eq3 / peak3[1:] - 1).min() * 100), 1),
            "drop5": round(float(s[:-5].mean() * 100), 3) if len(s) > 5 else None,
            "stopped": round(float(np.mean([x[5] for x in ts]) * 100), 1),
            "stop_dist": round(float(np.mean([x[6] for x in ts]) * 100), 2),
            "long": int(sum(1 for x in ts if x[1] > 0))}


def walk(cands: list) -> list:
    """cands: (i, side, g, x, t, stopped, dist) 시각순. 포지션 하나."""
    busy, tr = -1, []
    for c in cands:
        if c[0] <= busy:
            continue
        tr.append(c)
        busy = c[3]
    return tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--futures", action="store_true")
    ap.add_argument("--perm", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / ("futures/klines_5m.csv" if args.futures else "5m.csv"))
    fund = load_funding(base / "futures/funding.csv") if args.futures else None
    sh = dict(d5)
    sh["t"] = d5["t"] - args.offset * 60_000
    d = resample(sh, 3)
    d["t"] = d["t"] + args.offset * 60_000
    n = len(d["c"])
    end = d["t"][-1] + 900_000
    y1 = end - 365 * DAY
    rep: dict = {"offset": args.offset, "futures": args.futures}
    sigs_by = {ek: entry_signals(d, ek) for ek in ENTRIES}
    for ek, sk in COMBOS:
        on_l, on_s = sigs_by[ek]
        cands = []
        for side, on, opp in ((1, on_l, on_s), (-1, on_s, on_l)):
            for i in np.flatnonzero(on):
                if i < 6000 or i >= n - 2:
                    continue
                res = label(d, int(i), side, opp, stop_price(d, int(i), side, sk))
                if res is None:
                    continue
                g, x, dist, stopped = res
                g = with_funding(g, d, int(i), x, side, fund)
                cands.append((int(i), side, float(g), int(x), float(d["t"][i]), stopped, dist))
        cands.sort()
        tr = walk(cands)
        name = f"{ek}-{sk}"
        rep[name] = {"all": summ(tr), "all_c10": summ(tr, 0.001),
                     "old": summ([z for z in tr if z[4] < SPLIT]), "new": summ([z for z in tr if z[4] >= SPLIT]),
                     "1y": summ([z for z in tr if z[4] >= y1]),
                     "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
    basev = rep["E3-S1"]["all"]["ev"]
    for ek, sk in COMBOS[1:]:
        r = rep[f"{ek}-{sk}"]
        ok = (r["old"].get("n", 0) >= 30 and r["new"].get("n", 0) >= 30 and r["old"]["ev"] > 0 and r["new"]["ev"] > 0
              and r["all_c10"]["ev"] > 0 and (r["all"]["drop5"] or -1) > 0 and r["all"]["ev"] > basev)
        r["pass_1245"] = bool(ok)
        if args.perm and ok:  # 무작위 봉 진입, 같은 손절·청산, 같은 수·방향 구성
            on_l, on_s = sigs_by[ek]
            rng = np.random.default_rng(7)
            nl = sum(1 for z in r["trades"] if z[1] > 0)
            ns = len(r["trades"]) - nl
            actual = float(np.mean([z[2] for z in r["trades"]]))
            cache: dict = {}

            def outcome(i: int, side: int):
                if (i, side) not in cache:
                    res = label(d, i, side, on_s if side > 0 else on_l, stop_price(d, i, side, sk))
                    cache[(i, side)] = None if res is None else (
                        with_funding(res[0], d, i, res[1], side, fund), res[1])
                return cache[(i, side)]

            hits = 0
            for _ in range(2000):
                picks = [(int(k), 1) for k in rng.choice(np.arange(6000, n - 2), nl, replace=False)] + \
                        [(int(k), -1) for k in rng.choice(np.arange(6000, n - 2), ns, replace=False)]
                picks.sort()
                busy, gs = -1, []
                for i, side in picks:
                    if i <= busy:
                        continue
                    oc = outcome(i, side)
                    if oc is None:
                        continue
                    gs.append(oc[0])
                    busy = oc[1]
                if gs and np.mean(gs) >= actual:
                    hits += 1
            r["perm_p"] = round(hits / 2000, 4)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    for ek, sk in COMBOS:
        r = rep[f"{ek}-{sk}"]
        print(f"{ek}-{sk}", r["all"], "c10", r["all_c10"].get("ev"), "old", r["old"].get("n"), r["old"].get("ev"),
              "new", r["new"].get("n"), r["new"].get("ev"), "1y", r["1y"], r.get("pass_1245", ""), r.get("perm_p", ""),
              flush=True)


if __name__ == "__main__":
    main()
