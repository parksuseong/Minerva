"""15분봉 SCR 신호 + 큰 파도(30분~월봉) SCR 학습 가중치. 등록: 2026-11-06-btc-scr15-bigwave-weights-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_bigwave.py \
    --out <폴더> [--offset 5] [--futures] [--perm]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import load5
from btc_scr_entry_stop import entry_signals, label, summ, walk, with_funding
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, six

YEARS = range(2021, 2027)
BASES = {"B3": "E3", "B2": "E2"}
COST_RT = 0.001
STOP = 0.01
DAY = 86_400_000
MIN = 60_000


def ystart(y: int) -> float:
    return datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


def period_keys(t: np.ndarray, kind: str) -> tuple:
    """5분봉 시각마다 (그 봉이 속한 기간의 시작, 끝)."""
    if kind == "W":  # 월요일 UTC 0시 시작 (1970-01-05가 월요일)
        s = (t - 4 * DAY) // (7 * DAY) * (7 * DAY) + 4 * DAY
        return s, s + 7 * DAY
    if kind == "M":
        m = t.astype("datetime64[ms]").astype("datetime64[M]")
        return m.astype("datetime64[ms]").astype(float), (m + 1).astype("datetime64[ms]").astype(float)
    span = {"30m": 30, "1h": 60, "4h": 240, "1d": 1440}[kind] * MIN
    s = t // span * span
    return s, s + span


def agg(d5: dict, kind: str) -> tuple:
    """빈칸이 있어도 묶는다. 반환: 봉 자료, 각 봉이 끝나는 시각."""
    s, e = period_keys(d5["t"], kind)
    edges = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    st, en = edges[:-1], edges[1:]
    b = {"t": s[st], "o": d5["o"][st], "h": np.maximum.reduceat(d5["h"], st), "l": np.minimum.reduceat(d5["l"], st),
         "c": d5["c"][en - 1]}
    return b, e[st]


UPPER = ("30m", "1h", "4h", "1d", "W", "M")


def features(d5: dict, d: dict) -> np.ndarray:
    close15 = d["t"] + 900_000
    cols = [six(d)]
    for k in UPPER:
        b, end = agg(d5, k)
        f = six(b)
        j = np.searchsorted(end, close15, side="right") - 1
        o = np.full((len(close15), 6), np.nan)
        ok = j >= 0
        o[ok] = f[j[ok]]
        cols.append(o)
    return np.hstack(cols)


def ridge(X, y, lam_frac=0.1):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam_frac * len(y) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return mu, sd, w, y.mean()


def signals(d: dict, kind: str, fund=None) -> list:
    on_l, on_s = entry_signals(d, kind)
    n = len(d["c"])
    out = []
    for side, on, opp in ((1, on_l, on_s), (-1, on_s, on_l)):
        for i in np.flatnonzero(on):
            if i < 100 or i >= n - 2:
                continue
            r = label(d, int(i), side, opp, d["o"][i + 1] * (1 - STOP * side))
            if r is None:
                continue
            g = with_funding(r[0], d, int(i), r[1], side, fund)
            out.append((int(i), side, float(g), int(r[1]), float(d["t"][i]), r[3], STOP))
    out.sort()
    return out


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
    s5 = load5(base / "5m.csv")
    d = bars(s5, args.offset)
    X = features(s5, d)
    pos = {v: k for k, v in enumerate(d["t"])}
    if args.futures:
        dt = bars(load5(base / "futures/klines_5m.csv"), args.offset)
        fund = load_funding(base / "futures/funding.csv")
    rep: dict = {"offset": args.offset, "futures": args.futures}
    for bname, kind in BASES.items():
        sig = signals(d, kind)  # 학습은 늘 현물
        F = np.array([X[z[0]] * z[1] for z in sig])
        y = np.array([z[2] for z in sig])
        tsig = np.array([z[4] for z in sig])
        texit = np.array([d["t"][z[3]] for z in sig])
        ok = ~np.isnan(F).any(1)
        models = {}
        for yr in YEARS:
            m = ok & (texit < ystart(yr))
            models[yr] = ridge(F[m], y[m])
        if args.futures:
            tsg = signals(dt, kind, fund)
            Ft = np.array([X[pos[dt["t"][z[0]]]] * z[1] if dt["t"][z[0]] in pos else np.full(X.shape[1], np.nan) for z in tsg])
        else:
            tsg, Ft = sig, F
        tt = np.array([z[4] for z in tsg])
        pred = np.full(len(tsg), np.nan)
        for yr in YEARS:
            m = (tt >= ystart(yr)) & (tt < ystart(yr + 1)) & ~np.isnan(Ft).any(1)
            mu, sd, w, b = models[yr]
            pred[m] = ((Ft[m] - mu) / sd) @ w + b
        oos = [z for z, p in zip(tsg, pred) if not np.isnan(p)]
        take = [z for z, p in zip(tsg, pred) if not np.isnan(p) and p > COST_RT]
        tr, base_tr = walk(take), walk(oos)
        r = {"all": summ(tr), "all_c10": summ(tr, 0.001),
             "years": {yr: summ([z for z in tr if ystart(yr) <= z[4] < ystart(yr + 1)]) for yr in YEARS},
             "base_all": summ(base_tr), "n_oos_signals": len(oos), "n_taken_signals": len(take),
             "weights": {yr: [round(float(v), 6) for v in models[yr][2]] for yr in YEARS},
             "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
        if args.perm and tr:
            rng = np.random.default_rng(7)
            actual = float(np.mean([z[2] for z in tr]))
            groups = []
            for yr in YEARS:
                pool = [z for z in oos if ystart(yr) <= z[4] < ystart(yr + 1)]
                k = sum(1 for z in take if ystart(yr) <= z[4] < ystart(yr + 1))
                groups.append((pool, k))
            hits = 0
            for _ in range(2000):
                pick = []
                for pool, k in groups:
                    if k:
                        pick += [pool[j] for j in rng.choice(len(pool), k, replace=False)]
                pick.sort()
                g = walk(pick)
                if g and np.mean([z[2] for z in g]) >= actual:
                    hits += 1
            r["perm_p"] = round(hits / 2000, 4)
        rep[bname] = r
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
