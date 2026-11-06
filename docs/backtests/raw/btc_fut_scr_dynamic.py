"""사용자 Pine "SCR Dynamic (from PDF)". 등록: 2026-11-07-btc-fut15-scr-dynamic-hypotheses.md"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import btc_fut_scr_bracket as BR
from btc_fractal import cci, crossed, load5, rsi, stoch
from btc_fut_blog_combo import ema
from btc_scr_entry_stop import label, summ, walk, with_funding
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, ystart

DAY = 86_400_000


def strict(k, c, r):
    os_, ob = (k < 20) & (c < -100) & (r < 30), (k > 80) & (c > 100) & (r > 70)
    lx = crossed(k, 20, True) & crossed(c, -100, True) & crossed(r, 30, True)
    sx = crossed(k, 80, False) & crossed(c, 100, False) & crossed(r, 70, False)
    al = ash = False
    L, S = np.zeros(len(k), bool), np.zeros(len(k), bool)
    for i in range(len(k)):
        al = True if os_[i] else (False if lx[i] else al)
        ash = True if ob[i] else (False if sx[i] else ash)
        L[i], S[i] = al and lx[i], ash and sx[i]
    return L, S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / "futures/klines_5m.csv")
    fund = load_funding(base / "futures/funding.csv")
    d = bars(d5, args.offset)
    h, l, c, t = d["h"], d["l"], d["c"], d["t"]
    n = len(t)
    k, cc, r, e50 = stoch(h, l, c, 5), cci(h, l, c, 9), rsi(c, 14), ema(c, 50)
    with np.errstate(invalid="ignore"):
        tl, ts = c > e50, c < e50
    sL, sS = strict(k, cc, r)
    modes = {"Strict": (sL & tl, sS & ts), "StochOnly": (crossed(k, 20, True) & tl, crossed(k, 80, False) & ts)}
    days_old = (ystart(2024) - t[6000]) / DAY
    days_new = (t[-1] - ystart(2024)) / DAY
    rep: dict = {"offset": args.offset}
    for mn, (on_l, on_s) in modes.items():
        for xn in ("X1", "X2"):
            cands = []
            for side, on, opp in ((1, on_l, on_s), (-1, on_s, on_l)):
                for i in np.flatnonzero(on):
                    if i < 6000 or i >= n - 2:
                        continue
                    if xn == "X1":
                        res = label(d, int(i), side, opp, d["o"][i + 1] * (1 - 0.01 * side))
                        if res is None:
                            continue
                        g, x, _, stopped = res
                    else:
                        res = BR.trade(d, int(i), side)
                        if res is None:
                            continue
                        g, x, stopped = res
                    g = with_funding(g, d, int(i), x, side, fund)
                    cands.append((int(i), side, float(g), int(x), float(t[i]), stopped, 0.01))
            cands.sort()
            tr = walk(cands)
            old = [z for z in tr if z[4] < ystart(2024)]
            new = [z for z in tr if z[4] >= ystart(2024)]
            rep[f"{mn}-{xn}"] = {"all": summ(tr), "old": summ(old), "new": summ(new),
                                 "long": summ([z for z in tr if z[1] > 0]), "short": summ([z for z in tr if z[1] < 0]),
                                 "per_day_old": round(len(old) / days_old, 2), "per_day_new": round(len(new) / days_new, 2),
                                 "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
