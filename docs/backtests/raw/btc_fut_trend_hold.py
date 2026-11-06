"""15분봉 눌림 진입 + 큰 추세 보유. 등록: 2026-11-07-btc-fut15-trend-hold-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_fut_trend_hold.py \
    --out <폴더> [--offset 5] [--perm]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import crossed, load5, resample, stoch
from btc_scr_entry_stop import summ, walk, with_funding
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, direction, ystart

M5 = 300_000
STOP = 0.01
VARIANTS = ("T4", "TD", "TB")
DAY = 86_400_000


def upper_dir(d5: dict, d: dict, k: int) -> np.ndarray:
    u = resample(d5, k)
    j = np.searchsorted(u["t"] + k * M5, d["t"] + 900_000, side="right") - 1
    v = np.zeros(len(d["t"]), int)
    v[j >= 0] = direction(u)[j[j >= 0]]
    return v


def trade(d: dict, i: int, side: int, reg: np.ndarray):
    o, h, l = d["o"], d["h"], d["l"]
    e = i + 1
    px = o[e]
    stop = px * (1 - STOP * side)
    for j in range(e, len(o) - 1):
        if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
            fill = min(o[j], stop) if side > 0 else max(o[j], stop)
            return (fill / px - 1) * side, j, True
        if reg[j] != side:
            return (o[j + 1] / px - 1) * side, j + 1, False
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--perm", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / "futures/klines_5m.csv")
    fund = load_funding(base / "futures/funding.csv")
    d = bars(d5, args.offset)
    n = len(d["t"])
    t = d["t"]
    s = stoch(d["h"], d["l"], d["c"])
    up, dn = crossed(s, 20, True), crossed(s, 80, False)
    r4, rd = upper_dir(d5, d, 48), upper_dir(d5, d, 288)
    regs = {"T4": r4, "TD": rd, "TB": np.where(r4 == rd, r4, 0)}
    years = range(2020, 2027)
    y1 = t[-1] + 900_000 - 365 * DAY
    rep: dict = {"offset": args.offset}
    for name in VARIANTS:
        reg = regs[name]
        cands = []
        for i in range(6000, n - 2):
            side = 1 if (reg[i] == 1 and up[i]) else (-1 if (reg[i] == -1 and dn[i]) else 0)
            if not side:
                continue
            res = trade(d, i, side, reg)
            if res is None:
                continue
            g, x, stopped = res
            g = with_funding(g, d, i, x, side, fund)
            cands.append((i, side, float(g), int(x), float(t[i]), stopped, STOP))
        tr = walk(cands)
        hold = float(np.mean([(z[3] - z[0] - 1) * 0.25 for z in tr])) if tr else None
        r = {"all": summ(tr), "all_c10": summ(tr, 0.001),
             "old": summ([z for z in tr if z[4] < ystart(2024)]), "new": summ([z for z in tr if z[4] >= ystart(2024)]),
             "long": summ([z for z in tr if z[1] > 0]), "short": summ([z for z in tr if z[1] < 0]),
             "1y": summ([z for z in tr if z[4] >= y1]), "hold_h": round(hold, 2) if hold else None,
             "years": {y: summ([z for z in tr if ystart(y) <= z[4] < ystart(y + 1)]) for y in years},
             "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
        if args.perm and tr:
            rng = np.random.default_rng(7)
            actual = float(np.mean([z[2] for z in tr]))
            cache: dict = {}
            plan = []
            for y in years:
                zs = [z for z in tr if ystart(y) <= z[4] < ystart(y + 1)]
                idx = np.arange(n)
                m = (t >= ystart(y)) & (t < ystart(y + 1)) & (idx >= 6000) & (idx < n - 2)
                plan.append((np.flatnonzero(m & (reg == 1)), np.flatnonzero(m & (reg == -1)),
                             sum(1 for z in zs if z[1] > 0), sum(1 for z in zs if z[1] < 0)))
            hits = 0
            for _ in range(2000):
                picks = []
                for pl, ps, nl, ns in plan:
                    if nl:
                        picks += [(int(k), 1) for k in rng.choice(pl, min(nl, len(pl)), replace=False)]
                    if ns:
                        picks += [(int(k), -1) for k in rng.choice(ps, min(ns, len(ps)), replace=False)]
                picks.sort()
                busy, gs = -1, []
                for i, sd in picks:
                    if i <= busy:
                        continue
                    if (i, sd) not in cache:
                        q = trade(d, i, sd, reg)
                        cache[(i, sd)] = None if q is None else (with_funding(q[0], d, i, q[1], sd, fund), q[1])
                    oc = cache[(i, sd)]
                    if oc is None:
                        continue
                    gs.append(oc[0])
                    busy = oc[1]
                if gs and np.mean(gs) >= actual:
                    hits += 1
            r["perm_p"] = round(hits / 2000, 4)
        rep[name] = r
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
