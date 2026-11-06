"""15분봉 SCR + ATR 상황표. 등록: 2026-11-06-btc-scr15-atr-states-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_atr.py \
    --out <폴더> [--offset 5] [--futures] [--perm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from btc_fractal import crossed, load5, scr3, stoch
from btc_scr_entry_stop import summ, walk, with_funding
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars
from btc_swing_patterns import SPLIT

N = 20
LOW_WIN, LOW_Q = 960, 20
VARIANTS = ("A1", "AT")


def atr(d: dict, n: int = 14) -> np.ndarray:
    h, l, c = d["h"], d["l"], d["c"]
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    out = np.empty(len(tr))
    out[0] = tr[0]
    for i in range(1, len(tr)):
        out[i] = out[i - 1] + (tr[i] - out[i - 1]) / n
    return out


def states(d: dict) -> dict:
    c, v = d["c"], d["v"]
    a = atr(d)
    n = len(c)
    p = np.full(n, 0.0)
    p[N:] = np.sign(c[N:] - c[:-N])
    cv = np.cumsum(np.r_[0.0, v])
    s1 = np.full(n, np.nan)
    s1[N - 1:] = cv[N:] - cv[:-N]  # 최근 N봉 합(i 포함)
    vr = np.full(n, np.nan)
    vr[2 * N - 1:] = s1[2 * N - 1:] / np.where(s1[N - 1:-N] > 0, s1[N - 1:-N], np.nan)
    ar = np.full(n, np.nan)
    ar[N:] = a[N:] / a[:-N]
    low = np.zeros(n, bool)
    for i in range(LOW_WIN + N, n):
        ref = a[i - N]
        win = a[i - N - LOW_WIN:i - N]
        low[i] = ref <= np.percentile(win, LOW_Q)
    return {"p": p, "vr": vr, "ar": ar, "atr": a, "low": low}


def signals(d: dict, st: dict) -> tuple:
    s = stoch(d["h"], d["l"], d["c"])
    up20 = crossed(s, 20, True)
    dn80 = crossed(s, 80, False)
    _, on_s, *_ = scr3(d)
    p, vr, ar, low = st["p"], st["vr"], st["ar"], st["low"]
    with np.errstate(invalid="ignore"):
        l1 = up20 & (p < 0) & (vr > 1) & (ar < 1)
        l2 = up20 & (p > 0) & (vr > 1) & (ar > 1) & low
        sh = on_s & (p > 0) & (vr >= 2) & (ar >= 2)
    return l1, l2, sh, up20, dn80


def trade(d: dict, i: int, side: int, kind: str, st: dict, up20, dn80):
    o, h, l = d["o"], d["h"], d["l"]
    n = len(o)
    e = i + 1
    if e >= n - 1:
        return None
    px = o[e]
    a0 = st["atr"][i]
    dist = 0.01 * px if kind == "A1" else a0
    stop = px - side * dist
    exit_sig = dn80 if side > 0 else up20
    for j in range(e, n - 1):
        if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
            fill = min(o[j], stop) if side > 0 else max(o[j], stop)
            return (fill / px - 1) * side, j, True, dist / px
        if exit_sig[j] or st["atr"][j] >= 2 * a0:
            return (o[j + 1] / px - 1) * side, j + 1, False, dist / px
    return None


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
    d = bars(load5(base / ("futures/klines_5m.csv" if args.futures else "5m.csv")), args.offset)
    fund = load_funding(base / "futures/funding.csv") if args.futures else None
    st = states(d)
    l1, l2, sh, up20, dn80 = signals(d, st)
    n = len(d["c"])
    rep: dict = {"offset": args.offset, "futures": args.futures,
                 "n_signals": {"L1": int(l1[6000:].sum()), "L2": int(l2[6000:].sum()), "S": int(sh[6000:].sum())}}
    for kind in VARIANTS:
        cands = []
        for tag, side, on in (("L1", 1, l1), ("L2", 1, l2), ("S", -1, sh)):
            for i in np.flatnonzero(on):
                if i < 6000 or i >= n - 2:
                    continue
                r = trade(d, int(i), side, kind, st, up20, dn80)
                if r is None:
                    continue
                g = with_funding(r[0], d, int(i), r[1], side, fund)
                cands.append((int(i), side, float(g), int(r[1]), float(d["t"][i]), r[2], r[3], tag))
        cands.sort()
        tr = walk(cands)
        res = {"all": summ(tr), "all_c10": summ(tr, 0.001),
               "old": summ([z for z in tr if z[4] < SPLIT]), "new": summ([z for z in tr if z[4] >= SPLIT]),
               "by_tag": {t: summ([z for z in tr if z[7] == t]) for t in ("L1", "L2", "S")},
               "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5], z[7]] for z in tr]}
        if args.perm and tr:
            rng = np.random.default_rng(7)
            pool = np.arange(6000, n - 2)
            nl = sum(1 for z in tr if z[1] > 0)
            actual = float(np.mean([z[2] for z in tr]))
            cache: dict = {}
            hits = 0
            for _ in range(2000):
                picks = sorted([(int(k), 1) for k in rng.choice(pool, nl, replace=False)]
                               + [(int(k), -1) for k in rng.choice(pool, len(tr) - nl, replace=False)])
                busy, gs = -1, []
                for i, sd in picks:
                    if i <= busy:
                        continue
                    if (i, sd) not in cache:
                        r = trade(d, i, sd, kind, st, up20, dn80)
                        cache[(i, sd)] = None if r is None else (with_funding(r[0], d, i, r[1], sd, fund), r[1])
                    oc = cache[(i, sd)]
                    if oc is None:
                        continue
                    gs.append(oc[0])
                    busy = oc[1]
                if gs and np.mean(gs) >= actual:
                    hits += 1
            res["perm_p"] = round(hits / 2000, 4)
        rep[kind] = res
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
