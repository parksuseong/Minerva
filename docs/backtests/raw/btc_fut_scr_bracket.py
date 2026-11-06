"""SCR + 큰 추세 문, 익절 1%·손절 1%. 등록: 2026-11-07-btc-fut15-scr-bracket-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_fut_scr_bracket.py \
    --out <폴더> [--offset 5] [--perm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from btc_fractal import crossed, load5, stoch
from btc_fut_trend_hold import upper_dir
from btc_scr_entry_stop import entry_signals, summ, walk, with_funding
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, ystart

TP = SL = 0.01
MAXH = 96
DAY = 86_400_000


def trade(d: dict, i: int, side: int):
    o, h, l = d["o"], d["h"], d["l"]
    e = i + 1
    px = o[e]
    tp, sl = px * (1 + TP * side), px * (1 - SL * side)
    for j in range(e, min(e + MAXH, len(o) - 1)):
        if side > 0:
            if l[j] <= sl:
                return (min(o[j], sl) / px - 1), j, True
            if h[j] >= tp:
                return (max(o[j], tp) / px - 1), j, False
        else:
            if h[j] >= sl:
                return -(max(o[j], sl) / px - 1), j, True
            if l[j] <= tp:
                return -(min(o[j], tp) / px - 1), j, False
    x = e + MAXH
    if x >= len(o):
        return None
    return (o[x] / px - 1) * side, x, False


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
    t = d["t"]
    n = len(t)
    s = stoch(d["h"], d["l"], d["c"])
    ents = {"E3": entry_signals(d, "E3"), "E2": entry_signals(d, "E2"),
            "ES": (crossed(s, 20, True), crossed(s, 80, False))}
    rd, r4 = upper_dir(d5, d, 288), upper_dir(d5, d, 48)
    gates = {"G0": None, "GD": rd, "GDH": np.where(rd == r4, rd, 0)}
    days_old = (ystart(2024) - t[6000]) / DAY
    days_new = (t[-1] - ystart(2024)) / DAY
    rep: dict = {"offset": args.offset}
    for en, (on_l, on_s) in ents.items():
        for gn, g in gates.items():
            cands = []
            for side, on in ((1, on_l), (-1, on_s)):
                for i in np.flatnonzero(on):
                    if i < 6000 or i >= n - 2 or (g is not None and g[i] != side):
                        continue
                    res = trade(d, int(i), side)
                    if res is None:
                        continue
                    gg, x, stopped = res
                    gg = with_funding(gg, d, int(i), x, side, fund)
                    cands.append((int(i), side, float(gg), int(x), float(t[i]), stopped, SL))
            cands.sort()
            tr = walk(cands)
            old = [z for z in tr if z[4] < ystart(2024)]
            new = [z for z in tr if z[4] >= ystart(2024)]
            r = {"all": summ(tr), "old": summ(old), "new": summ(new),
                 "long": summ([z for z in tr if z[1] > 0]), "short": summ([z for z in tr if z[1] < 0]),
                 "per_day_old": round(len(old) / days_old, 2), "per_day_new": round(len(new) / days_new, 2),
                 "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
            ok = (r["per_day_old"] >= 1 and r["per_day_new"] >= 1 and
                  all(r[k].get("win", 0) >= 70 and r[k].get("ev", -1) > 0 for k in ("old", "new", "long", "short")))
            r["pass_1235"] = bool(ok)
            if args.perm and ok:
                rng = np.random.default_rng(7)
                actual = float(np.mean([z[2] for z in tr]))
                idx = np.arange(6000, n - MAXH - 2)
                pl = idx if g is None else idx[g[idx] == 1]
                ps = idx if g is None else idx[g[idx] == -1]
                nl, ns = sum(1 for z in tr if z[1] > 0), sum(1 for z in tr if z[1] < 0)
                cache: dict = {}
                hits = 0
                for _ in range(2000):
                    picks = sorted([(int(k), 1) for k in rng.choice(pl, nl, replace=False)] +
                                   [(int(k), -1) for k in rng.choice(ps, ns, replace=False)])
                    busy, gs = -1, []
                    for i, sd in picks:
                        if i <= busy:
                            continue
                        if (i, sd) not in cache:
                            q = trade(d, i, sd)
                            cache[(i, sd)] = None if q is None else (with_funding(q[0], d, i, q[1], sd, fund), q[1])
                        oc = cache[(i, sd)]
                        if oc is None:
                            continue
                        gs.append(oc[0])
                        busy = oc[1]
                    if gs and np.mean(gs) >= actual:
                        hits += 1
                r["perm_p"] = round(hits / 2000, 4)
            rep[f"{en}-{gn}"] = r
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
