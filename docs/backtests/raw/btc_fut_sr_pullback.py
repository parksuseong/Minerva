"""1시간봉 지지·저항 되돌림 + 구조 손절 + 익절 1%. 등록: 2026-11-07-btc-fut-sr-pullback-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_fut_sr_pullback.py \
    --out <폴더> [--offset 5] [--perm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from btc_fractal import crossed, load5, recent, resample, stoch
from btc_fut_blog_combo import dmi
from btc_fut_trend_hold import upper_dir
from btc_scr_entry_stop import summ, walk, with_funding
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, ystart

M5 = 300_000
H1 = 3_600_000
BAR = 900_000
DAY = 86_400_000
TP = 0.01
MAXH = 96
K = 3
LOOK = 120 * H1


def pivots(u: dict):
    """확정된 1시간봉 저점·고점: (확정 시각, 가격) 배열."""
    h, l, t = u["h"], u["l"], u["t"]
    n = len(l)
    lo_t, lo_p, hi_t, hi_p = [], [], [], []
    for i in range(K, n - K):
        w = slice(i - K, i + K + 1)
        if l[i] == l[w].min() and (l[w] == l[i]).sum() == 1:
            lo_t.append(t[i + K] + H1); lo_p.append(l[i]); lo_t[-1] = lo_t[-1]
        if h[i] == h[w].max() and (h[w] == h[i]).sum() == 1:
            hi_t.append(t[i + K] + H1); hi_p.append(h[i])
    return (np.array(lo_t), np.array(lo_p), np.array([t[0]] * 0)), (np.array(hi_t), np.array(hi_p))


def last_level(conf_t, price, piv_t_src, T):
    j = np.searchsorted(conf_t, T, side="right") - 1
    out = np.full(len(T), np.nan)
    ok = j >= 0
    out[ok] = price[j[ok]]
    return out, j


def trade(d: dict, i: int, side: int, stop: float):
    o, h, l = d["o"], d["h"], d["l"]
    e = i + 1
    px = o[e]
    dist = abs(stop / px - 1)
    if (side > 0 and stop >= px) or (side < 0 and stop <= px) or dist < 0.003 or dist > 0.03:
        return None
    tp = px * (1 + TP * side)
    for j in range(e, min(e + MAXH, len(o) - 1)):
        if side > 0:
            if l[j] <= stop:
                return min(o[j], stop) / px - 1, j, True, dist
            if h[j] >= tp:
                return max(o[j], tp) / px - 1, j, False, dist
        else:
            if h[j] >= stop:
                return -(max(o[j], stop) / px - 1), j, True, dist
            if l[j] <= tp:
                return -(min(o[j], tp) / px - 1), j, False, dist
    x = e + MAXH
    if x >= len(o):
        return None
    return (o[x] / px - 1) * side, x, False, dist


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
    t, l, h, c = d["t"], d["l"], d["h"], d["c"]
    n = len(t)
    T = t + BAR
    u = resample(d5, 12)
    (lo_t, lo_p, _), (hi_t, hi_p) = pivots(u)
    sup, js = last_level(lo_t, lo_p, None, T)
    res_, jr = last_level(hi_t, hi_p, None, T)
    sup_age = T - np.where(js >= 0, lo_t[np.maximum(js, 0)], -np.inf)
    res_age = T - np.where(jr >= 0, hi_t[np.maximum(jr, 0)], -np.inf)
    _, _, adx = dmi(u)
    ja = np.searchsorted(u["t"] + H1, T, side="right") - 1
    adx15 = np.where(ja >= 0, adx[np.maximum(ja, 0)], np.nan)
    reg = upper_dir(d5, d, 288)
    s = stoch(h, l, c)
    up3, dn3 = recent(crossed(s, 20, True)), recent(crossed(s, 80, False))
    with np.errstate(invalid="ignore"):
        long_sig = (reg == 1) & (sup_age <= LOOK) & (sup < c) & (l <= sup * 1.003) & up3
        short_sig = (reg == -1) & (res_age <= LOOK) & (res_ > c) & (h >= res_ * 0.997) & dn3
    stop_l = np.minimum(sup, l) * (1 - 0.001)
    stop_s = np.maximum(res_, h) * (1 + 0.001)
    days_old = (ystart(2024) - t[6000]) / DAY
    days_new = (t[-1] - ystart(2024)) / DAY
    rep: dict = {"offset": args.offset}
    for name, gate in (("SR", np.ones(n, bool)), ("SRA", adx15 >= 25)):
        cands = []
        for side, sig, stp in ((1, long_sig, stop_l), (-1, short_sig, stop_s)):
            on = sig & gate
            on = on & ~np.r_[False, on[:-1]]
            for i in np.flatnonzero(on):
                if i < 6000 or i >= n - 2:
                    continue
                r = trade(d, int(i), side, float(stp[i]))
                if r is None:
                    continue
                g, x, stopped, dist = r
                g = with_funding(g, d, int(i), x, side, fund)
                cands.append((int(i), side, float(g), int(x), float(t[i]), stopped, dist))
        cands.sort()
        tr = walk(cands)
        old = [z for z in tr if z[4] < ystart(2024)]
        new = [z for z in tr if z[4] >= ystart(2024)]
        r = {"all": summ(tr), "old": summ(old), "new": summ(new),
             "long": summ([z for z in tr if z[1] > 0]), "short": summ([z for z in tr if z[1] < 0]),
             "per_day_old": round(len(old) / days_old, 2), "per_day_new": round(len(new) / days_new, 2),
             "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
        ok = r["all"].get("n", 0) > 0 and all(r[k].get("win", 0) >= 70 and r[k].get("ev", -1) > 0
                                               for k in ("old", "new", "long", "short"))
        r["pass_24"] = bool(ok)
        if args.perm and tr:
            rng = np.random.default_rng(7)
            actual = float(np.mean([z[2] for z in tr]))
            dists = np.array([z[6] for z in tr])
            idx = np.arange(6000, n - MAXH - 2)
            pl, ps = idx[(reg[idx] == 1) & gate[idx]], idx[(reg[idx] == -1) & gate[idx]]
            nl, ns = sum(1 for z in tr if z[1] > 0), sum(1 for z in tr if z[1] < 0)
            hits = 0
            for _ in range(2000):
                picks = sorted([(int(k), 1) for k in rng.choice(pl, nl, replace=False)] +
                               [(int(k), -1) for k in rng.choice(ps, ns, replace=False)])
                ds = rng.choice(dists, len(picks))
                busy, gs = -1, []
                for (i, sd), dd in zip(picks, ds):
                    if i <= busy:
                        continue
                    px = d["o"][i + 1]
                    q = trade(d, i, sd, px * (1 - dd * sd))
                    if q is None:
                        continue
                    gs.append(with_funding(q[0], d, i, q[1], sd, fund))
                    busy = q[1]
                if gs and np.mean(gs) >= actual:
                    hits += 1
            r["perm_p"] = round(hits / 2000, 4)
        rep[name] = r
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
