"""BTC 선물 15분봉: 블로그 규칙 조합(트리거 × 필터), 롱·숏 대칭. 등록: 2026-11-06-btc-fut15-blog-combo-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_fut_blog_combo.py \
    --out <폴더> [--offset 5] [--perm 조합1,조합2]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import cci, crossed, load5, recent, resample, rsi, scr3, stoch
from btc_scr_entry_stop import label, summ, walk, with_funding
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, direction

TRIG = ("T3", "T2", "TP", "TB")
FILT = ("none", "MA", "DMI", "HTF", "MACD", "DISP")
SPLIT = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
STOP = 0.01
M5 = 300_000


def sma(x, n):
    c = np.cumsum(np.r_[0.0, x])
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def ema(x, n):
    a = 2 / (n + 1)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


def wilder(x, n):
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    return out


def dmi(d, n=14):
    h, l, c = d["h"], d["l"], d["c"]
    up = np.r_[0.0, h[1:] - h[:-1]]
    dn = np.r_[0.0, l[:-1] - l[1:]]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = wilder(tr, n)
    pdi = 100 * wilder(pdm, n) / np.where(atr > 0, atr, np.nan)
    mdi = 100 * wilder(mdm, n) / np.where(atr > 0, atr, np.nan)
    dx = 100 * np.abs(pdi - mdi) / np.where(pdi + mdi > 0, pdi + mdi, np.nan)
    adx = wilder(np.nan_to_num(dx), n)
    return pdi, mdi, adx


def build(d: dict, d5: dict) -> dict:
    h, l, c = d["h"], d["l"], d["c"]
    s, cc, r = stoch(h, l, c), cci(h, l, c), rsi(c)
    up20, dn80 = crossed(s, 20, True), crossed(s, 80, False)
    t3l, t3s, *_ = scr3(d)
    l2 = recent(crossed(s, 20, True)) & recent(crossed(cc, -100, True))
    s2 = recent(crossed(s, 80, False)) & recent(crossed(cc, 100, False))
    t2l, t2s = l2 & ~np.r_[False, l2[:-1]], s2 & ~np.r_[False, s2[:-1]]
    pdi, mdi, adx = dmi(d)
    with np.errstate(invalid="ignore"):
        tpl, tps = up20 & (cc >= 0) & (r >= 50), dn80 & (cc <= 0) & (r <= 50)
        tbl, tbs = up20 & (adx < 20), dn80 & (adx < 20)
    m50 = sma(c, 50)
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    d50 = c / m50 * 100
    lo = np.full(len(c), np.nan)
    hi = np.full(len(c), np.nan)
    W = 2880
    for i in range(W + 50, len(c)):  # 지난 30일(현재 봉 제외) 분위수
        w = d50[i - W:i]
        lo[i], hi[i] = np.nanpercentile(w, 10), np.nanpercentile(w, 90)
    u = resample(d5, 48)
    j = np.searchsorted(u["t"] + 48 * M5, d["t"] + 900_000, side="right") - 1
    hd = np.zeros(len(c), int)
    hd[j >= 0] = direction(u)[j[j >= 0]]
    with np.errstate(invalid="ignore"):
        filt = {"none": (np.ones(len(c), bool), np.ones(len(c), bool)),
                "MA": (c > m50, c < m50), "DMI": (pdi > mdi, mdi > pdi), "HTF": (hd == 1, hd == -1),
                "MACD": (hist > 0, hist < 0), "DISP": (d50 <= lo, d50 >= hi)}
    trig = {"T3": (t3l, t3s, t3s, t3l), "T2": (t2l, t2s, t2s, t2l),
            "TP": (tpl, tps, dn80, up20), "TB": (tbl, tbs, dn80, up20)}
    return {"trig": trig, "filt": filt}


def run_combo(d, ind, tk, fk, fund):
    lon, son, lex, sex = ind["trig"][tk]
    fl, fs = ind["filt"][fk]
    n = len(d["c"])
    cands = []
    for side, on, ex in ((1, lon & fl, lex), (-1, son & fs, sex)):
        for i in np.flatnonzero(on):
            if i < 6000 or i >= n - 2:
                continue
            r = label(d, int(i), side, ex, d["o"][i + 1] * (1 - STOP * side))
            if r is None:
                continue
            g = with_funding(r[0], d, int(i), r[1], side, fund)
            cands.append((int(i), side, float(g), int(r[1]), float(d["t"][i]), r[3], STOP))
    cands.sort()
    return walk(cands)


def side_summ(tr):
    return {"all": summ(tr), "all_c10": summ(tr, 0.001), "long": summ([z for z in tr if z[1] > 0]),
            "short": summ([z for z in tr if z[1] < 0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--perm", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / "futures/klines_5m.csv")
    fund = load_funding(base / "futures/funding.csv")
    d = bars(d5, args.offset)
    ind = build(d, d5)
    n = len(d["c"])
    rep: dict = {"offset": args.offset}
    perm = [p for p in args.perm.split(",") if p]
    for tk in TRIG:
        for fk in FILT:
            tr = run_combo(d, ind, tk, fk, fund)
            name = f"{tk}-{fk}"
            r = {"train": side_summ([z for z in tr if z[4] < SPLIT]), "test": side_summ([z for z in tr if z[4] >= SPLIT]),
                 "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
            if name in perm:
                te = [z for z in tr if z[4] >= SPLIT]
                lon, son, lex, sex = ind["trig"][tk]
                rng = np.random.default_rng(7)
                pool = np.flatnonzero(d["t"] >= SPLIT)
                pool = pool[pool < n - 2]
                nl = sum(1 for z in te if z[1] > 0)
                actual = float(np.mean([z[2] for z in te])) if te else 0.0
                cache: dict = {}
                hits = 0
                for _ in range(2000):
                    picks = sorted([(int(k), 1) for k in rng.choice(pool, nl, replace=False)]
                                   + [(int(k), -1) for k in rng.choice(pool, len(te) - nl, replace=False)])
                    busy, gs = -1, []
                    for i, sd in picks:
                        if i <= busy:
                            continue
                        if (i, sd) not in cache:
                            rr = label(d, i, sd, lex if sd > 0 else sex, d["o"][i + 1] * (1 - STOP * sd))
                            cache[(i, sd)] = None if rr is None else (with_funding(rr[0], d, i, rr[1], sd, fund), rr[1])
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
