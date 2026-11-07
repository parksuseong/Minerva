"""122번: 1시간·4시간봉 RSI 7 스윙 다리를 맞히는 XGBoost, 선물 15분봉 체결.
등록: 2026-11-07-btc-ml-swing-leg-htf-hypotheses.md

실행: PYTHONPATH=<pyml>:<numpy>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_swing_leg_htf.py \
    --out <폴더> [--stage predict|trade|all] [--circ 200]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import btc_ml_ensemble as E
from btc_fractal import load5, rsi
from btc_ml_swing_leg import fit, leg_labels, pivots, simulate, swing_feats, zigzag
from btc_scr_entry_stop import summ
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, ystart

M5 = 300_000
YEARS = (2022, 2023, 2024, 2025, 2026)
SEEDS = 5
FRAMES = {"L1h": 12, "L4h": 48}  # 5분봉 개수
QS = (0.80, 0.20)  # 121번 H2와 같다


def predict(out: Path, name: str, factor: int) -> None:
    d5 = load5(Path("~/.minerva/data/btc/5m.csv").expanduser())
    B = E.build(d5, factor)
    d, X = B["d"], B["X"]
    t, c, h, l = d["t"], d["c"], d["h"], d["l"]
    n = len(t)
    r7 = rsi(c, 7)
    pv = pivots(h, l, r7)
    zz = zigzag(pv)
    y, reach = leg_labels(n, zz)
    X = np.concatenate([X, swing_feats(c, r7, pv, n)], 1)
    look = np.where(reach >= 0, reach - np.arange(n), 0)
    leg = np.diff([z[1] for z in zz])
    emb = int(max(look.max(), leg.max()) + 1)
    valid = np.isfinite(X).all(1)
    probs = np.full((SEEDS, n), np.nan, np.float32)
    prev = np.full((SEEDS, n), np.nan, np.float32)
    for Y in YEARS:
        te = np.flatnonzero((t >= E.ts(Y)) & (t < E.ts(Y + 1)))
        if not len(te):
            continue
        base = (t >= E.ts(2018)) & (np.arange(n) < int(np.searchsorted(t, E.ts(Y))) - emb) & valid
        i0 = np.flatnonzero(base)
        mu, sd = X[i0].mean(0), X[i0].std(0) + 1e-6
        Xn = ((X - mu) / sd).astype(np.float32)
        Xn[~np.isfinite(Xn)] = 0
        tr = np.flatnonzero(base & (y >= 0))
        pr = np.flatnonzero((t >= E.ts(Y - 1)) & (t < E.ts(Y))) if Y == YEARS[0] else np.array([], int)
        for s in range(SEEDS):
            t0 = time.time()
            idx = np.r_[te, pr]
            p = fit(Xn[tr], y[tr], Xn[idx], seed=Y + 1000 * s)
            probs[s, te] = p[: len(te)]
            if len(pr):
                prev[s, pr] = p[len(te):]
            print(name, Y, s, "train", len(tr), "sec", round(time.time() - t0, 1), flush=True)
    np.savez_compressed(out / f"pred_{name}.npz", probs=probs, prev=prev, y=y, t=t, factor=factor,
                        emb=emb, max_leg=int(leg.max()), max_look=int(look.max()),
                        zz=np.array([(a, b, p_, k) for a, b, p_, k in zz], float))
    print(name, "embargo", emb, "max_leg", int(leg.max()), "pivots", len(zz), flush=True)


def thresholds(t, p, p_prev):
    out = {}
    for Y in YEARS:
        src = p_prev if Y == YEARS[0] else p
        m = (t >= E.ts(Y - 1)) & (t < E.ts(Y)) & np.isfinite(src)
        out[Y] = (float(np.quantile(src[m], QS[0])), float(np.quantile(src[m], QS[1])))
    return out


def report(tr, days):
    y23 = [z for z in tr if ystart(2023) <= z[4] < ystart(2024)]
    new = [z for z in tr if z[4] >= ystart(2024)]
    g = sorted(z[2] for z in tr)
    return {"all": summ(tr), "all_gross": summ(tr, 0.0), "y2022": summ([z for z in tr if z[4] < ystart(2023)]),
            "y2023": summ(y23), "new": summ(new), "new_gross": summ(new, 0.0),
            "new_long": summ([z for z in new if z[1] > 0]), "new_short": summ([z for z in new if z[1] < 0]),
            "new_c15": summ(new, 0.00075),
            "drop5": round((float(np.mean(g[:-5])) - 0.001) * 100, 4) if len(g) > 5 else None,
            "per_day": round(len(tr) / days, 3),
            "hold_15m": round(float(np.mean([z[3] - z[0] - 1 for z in tr])), 1) if tr else None}


def trade_stage(out: Path, circ: int) -> None:
    F = bars(load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser()), 0)
    fund = load_funding(Path("~/.minerva/data/btc/futures/funding.csv").expanduser())
    tf = F["t"]
    close15 = tf + 900_000
    start = int(np.searchsorted(tf, E.ts(YEARS[0])))
    days = (tf[-1] - tf[start]) / 86_400_000
    yr = np.array([int(np.datetime64(int(x), "ms").astype("datetime64[Y]").astype(int) + 1970) for x in tf])
    from sklearn.metrics import roc_auc_score

    rep: dict = {"fut_end": int(tf[-1])}
    trades = {}
    for name, factor in FRAMES.items():
        P = np.load(out / f"pred_{name}.npz")
        ts, y = P["t"], P["y"]
        hclose = ts + factor * M5
        pm, pp = np.nanmean(P["probs"], 0), np.nanmean(P["prev"], 0)
        # 15분봉 마감 때 마감된 마지막 상위 봉의 확률
        j = np.searchsorted(hclose, close15, side="right") - 1
        ok = j >= 0

        def align(arr):
            a = np.full(len(tf), np.nan)
            a[ok] = arr[j[ok]]
            return a

        # 닮은 정도용 꼬리표: 15분봉이 속한 상위 봉(사후 다리)
        k = np.searchsorted(ts, tf, side="right") - 1
        yf = np.where(k >= 0, y[np.maximum(k, 0)], -1).astype(float)
        m = np.isfinite(pm) & (y >= 0)
        r: dict = {"embargo_bars": int(P["emb"]), "max_leg": int(P["max_leg"]), "max_look": int(P["max_look"]),
                   "auc": {str(Y): round(float(roc_auc_score(y[m & (ts >= E.ts(Y)) & (ts < E.ts(Y + 1))],
                                                             pm[m & (ts >= E.ts(Y)) & (ts < E.ts(Y + 1))])), 4)
                           for Y in YEARS if (m & (ts >= E.ts(Y))).any()}}
        th = thresholds(ts, pm, pp)
        hi = np.array([th.get(v, (np.nan, np.nan))[0] for v in yr])
        lo = np.array([th.get(v, (np.nan, np.nan))[1] for v in yr])
        pf = align(pm)
        tr = simulate(F, fund, pf, hi, lo, start)
        r.update(report(tr, days))
        r["thresholds"] = {str(a): [round(b, 4), round(c, 4)] for a, (b, c) in th.items()}
        hold = np.zeros(len(tf))
        for z in tr:
            hold[z[0] + 1:z[3]] = z[1]
        mm = (yf >= 0) & (np.arange(len(tf)) >= start)
        r["resemble"] = {"up_long": round(float((hold[mm & (yf == 1)] > 0).mean() * 100), 1),
                         "down_short": round(float((hold[mm & (yf == 0)] < 0).mean() * 100), 1),
                         "flat": round(float((hold[mm] == 0).mean() * 100), 1)}
        seeds = []
        for s in range(SEEDS):
            ths = thresholds(ts, P["probs"][s], P["prev"][s])
            his = np.array([ths.get(v, (np.nan, np.nan))[0] for v in yr])
            los = np.array([ths.get(v, (np.nan, np.nan))[1] for v in yr])
            trs = simulate(F, fund, align(P["probs"][s]), his, los, start)
            seeds.append(summ([z for z in trs if z[4] >= ystart(2024)]).get("ev"))
        r["seeds_new_ev"] = seeds
        if circ and tr:
            rng = np.random.default_rng(7)
            span = np.flatnonzero(np.isfinite(pf))
            null = []
            for _ in range(circ):
                sh = int(rng.integers(2000, len(span) - 2000))
                pr = pf.copy()
                pr[span] = np.roll(pf[span], sh)
                null.append(summ(simulate(F, fund, pr, hi, lo, start)).get("ev", 0.0))
            null = np.array(null)
            r["circ_null_ev"] = round(float(null.mean()), 4)
            r["circ_p"] = round(float((null >= r["all"]["ev"]).mean()), 4)
        crit = [
            all(r[q].get("ev", -1) > 0 and r[q].get("ret3", -1) > 0 for q in ("y2023", "new")),
            r["new_long"].get("ev", -1) > 0 and r["new_short"].get("ev", -1) > 0,
            r["new_c15"].get("ev", -1) > 0,
            (r["drop5"] or -1) > 0,
            r.get("circ_p", 1) < 0.025,
            sum(1 for v in seeds if v is not None and v > 0) >= 3,
            r["per_day"] >= 0.5,
        ]
        r["criteria"] = [bool(x) for x in crit]
        r["pass"] = bool(all(crit))
        rep[name] = r
        trades[name] = [[int(z[0]), int(z[1]), round(z[2], 5), int(z[3]), bool(z[5])] for z in tr]
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    (out / "trades.json").write_text(json.dumps(trades))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--circ", type=int, default=200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("predict", "all"):
        for name, factor in FRAMES.items():
            predict(out, name, factor)
    if a.stage in ("trade", "all"):
        trade_stage(out, a.circ)


if __name__ == "__main__":
    main()
