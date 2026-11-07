"""143번: BTC ML 진입 우위가 ETH에서 재현되는가. 등록: 2026-11-07-eth-ml-replication-hypotheses.md

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python docs/backtests/raw/eth_ml_replication.py --out <폴더> \
    [--stage predict|eval|all] [--seeds 5]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import btc_fut_daily_regime as R
import btc_ml_ensemble as E
import numpy as np
from btc_best_entry_vote import HOR, k2_first, measure
from btc_fractal import load5
from btc_ml_barrier import barrier_labels
from btc_ml_edge_cost import mean_score
from btc_ml_seeds import rule_masks
from btc_ml_tracker import trend4h
from btc_r1_as4_entry import Grid, Model, evaluate, lim_fill, run_k, run_ml

ETH = Path("~/.minerva/data/eth").expanduser()
FACTOR = 3
BAR = FACTOR * 300_000
EMBARGO = 600
SPLICE = E.ts(2020)


def load_d5():
    sp = load5(ETH / "5m.csv")
    fu = load5(ETH / "futures/klines_5m.csv")
    a = sp["t"] < SPLICE
    b = fu["t"] >= SPLICE
    return {k: np.r_[sp[k][a], fu[k][b]] for k in sp}, fu


def load_funding():
    with (ETH / "futures/funding.csv").open() as f:
        r = list(csv.reader(f))[1:]
    return np.array([float(x[0]) for x in r]), np.array([float(x[2]) for x in r])


def predict(out: Path, nseeds: int) -> None:
    d5, _ = load_d5()
    B = E.build(d5, FACTOR)
    X, seq, d = B["X"], B["seq"], B["d"]
    t = d["t"]
    n = len(t)
    valid = np.isfinite(X).all(1) & np.isfinite(seq).all(1)
    seq_ok = np.convolve(valid.astype(int), np.ones(E.SEQ, int), "full")[:n] == E.SEQ
    t0 = time.time()
    y = barrier_labels(d, 0.015, 0.015)
    print("labels sec", round(time.time() - t0, 1), flush=True)
    probs = np.full((nseeds, n), np.nan, np.float32)
    for Y in E.YEARS:
        te = np.flatnonzero((t >= E.ts(Y)) & (t < E.ts(Y + 1)) & seq_ok)
        if not len(te):
            continue
        base = (t >= E.ts(2018)) & (t < E.ts(Y) - EMBARGO * BAR) & seq_ok
        i0 = np.flatnonzero(base)
        mu, sd = X[i0].mean(0), X[i0].std(0) + 1e-6
        Xn = ((X - mu) / sd).astype(np.float32)
        Xn[~np.isfinite(Xn)] = 0
        tr = np.flatnonzero(base & (y >= 0))[::3]
        for k in range(nseeds):
            t1 = time.time()
            p = E.fit_predict_tab("xgb", Xn[tr], y[tr], Xn[te], seed=Y + 1000 * k, ncls=2)
            probs[k, te] = p[:, 1]
            print(Y, "seed", k, "train", len(tr), "sec", round(time.time() - t1, 1), flush=True)
    np.savez_compressed(out / "pred.npz", probs=probs, y=y, t=t, o=d["o"], h=d["h"], l=d["l"], c=d["c"])


def auc(score, y):
    from sklearn.metrics import roc_auc_score

    m = np.isfinite(score) & (y >= 0)
    return round(float(roc_auc_score(y[m], score[m])), 4) if m.sum() > 100 and len(set(y[m])) == 2 else None


def edge(G, P, sig, i0, model, periods, month, by_month, rng, nboot=2000):
    """엔트리 하나 뒤 2일 측정 동안 다음 신호는 보지 않는다(133과 같다). 무작위도 같은 지정가 체결."""
    tp = P["t"]
    ents, i = [], i0
    while i < G.n - HOR - 2:
        s = int(sig[i])
        if not s:
            i += 1
            continue
        f = lim_fill(G, i, s, model)
        if f is None:
            i += 1
            continue
        e, px = f
        ents.append((e, s, float(px)))
        i = e + HOR
    res = {}
    for pn, (p0, p1) in periods.items():
        es = [x for x in ents if p0 <= tp[x[0]] < p1]
        if not es:
            res[pn] = {"n": 0}
            continue
        hits, rmeans, months = [], [], []
        for e, s, px in es:
            pool = by_month.get(month[e])
            rr = []
            if pool is not None and len(pool):
                for r_ in rng.choice(pool, 40):
                    f = lim_fill(G, int(r_), s, model)
                    if f is not None and f[0] < G.n - HOR - 1:
                        rr.append(measure(P, f[0], s, f[1]))
                    if len(rr) >= 20:
                        break
            if not rr:
                continue
            hits.append(measure(P, e, s, px))
            rmeans.append(float(np.mean(rr)))
            months.append(month[e])
        hits, rmeans, months = np.array(hits), np.array(rmeans), np.array(months)
        diff = hits - rmeans
        um = np.unique(months)
        idx = {m: np.flatnonzero(months == m) for m in um}
        boots = []
        for _ in range(nboot):
            pick = rng.choice(um, len(um))
            sel = np.concatenate([idx[m] for m in pick])
            boots.append(diff[sel].mean())
        boots = np.array(boots)
        sd = np.array([s for _, s, _ in es])
        res[pn] = {"n": len(hits), "n_long": int((sd > 0).sum()), "P": round(float(hits.mean()) * 100, 1),
                   "P_rand": round(float(rmeans.mean()) * 100, 1), "edge": round(float(diff.mean()) * 100, 1),
                   "boot_p": round(float((boots <= 0).mean()), 4),
                   "ci95": [round(float(np.quantile(boots, 0.025)) * 100, 1),
                            round(float(np.quantile(boots, 0.975)) * 100, 1)]}
    return res


def evaluate_all(out: Path) -> None:
    z = np.load(out / "pred.npz")
    probs, y, tp = z["probs"], z["y"], z["t"]
    P = {k: z[k].astype(float) for k in "ohlc"}
    P["t"] = tp
    d5, fu = load_d5()
    end = float(tp[-1])
    fuk = {k: v[fu["t"] <= end + BAR] for k, v in fu.items()}
    ft, fr = load_funding()
    reg, _ = R.regime(fuk, P, 0)
    tr4 = trend4h(d5, tp)
    G = Grid(P, fuk)
    i0 = int(np.searchsorted(tp, E.ts(2023)))
    model = Model(True, 0.0005)
    periods = {"2023": (E.ts(2023), E.ts(2024)), "2024~": (E.ts(2024), end + 1)}
    rep: dict = {"end": datetime.fromtimestamp(end / 1000, timezone.utc).isoformat(), "seeds": int(len(probs))}
    sc = mean_score(probs)
    yr = np.array([datetime.fromtimestamp(x / 1000, timezone.utc).year for x in tp])
    rep["auc"] = {str(Y): auc(np.where(yr == Y, sc, np.nan), y) for Y in E.YEARS}
    rep["auc_seed"] = {str(Y): [auc(np.where(yr == Y, probs[k], np.nan), y) for k in range(len(probs))]
                       for Y in E.YEARS}
    print("auc", rep["auc"], flush=True)
    k_all = len(probs)
    AS = rule_masks(probs, tp, 0.90, min(4, k_all))
    BS = rule_masks(probs, tp, 0.95, k_all)
    AS4 = (AS[0] & (tr4 > 0), AS[1] & (tr4 < 0))
    masks = {"AS-x": AS, "BS-x": BS, "AS4-x": AS4}
    sigs = {k: np.where(m[0], 1, np.where(m[1], -1, 0)) for k, m in masks.items()}
    sigs["K2-x"] = k2_first(reg, AS4[0], AS4[1], i0)
    rng = np.random.default_rng(7)
    month = np.array([datetime.fromtimestamp(x / 1000, timezone.utc).strftime("%Y%m") for x in tp])
    valid = np.arange(i0, G.n - HOR - 3)
    by_month = {mm: valid[month[valid] == mm] for mm in dict.fromkeys(month[valid])}
    rep["edge"], rep["trade"], groups = {}, {}, {}
    for name, sg in sigs.items():
        rep["edge"][name] = edge(G, P, sg, i0, model, periods, month, by_month, rng)
        print("edge", name, rep["edge"][name], flush=True)
        if name == "K2-x":
            g = run_k(G, reg, AS4[0], AS4[1], i0, model, False)
        else:
            g = run_ml(G, masks[name][0], masks[name][1], i0, model)
        row = evaluate(g, G, ft, fr, periods)
        row.pop("_series")
        rep["trade"][name] = row
        groups[name] = [[int(e), int(s), [[float(v) for v in sg_] for sg_ in segs], int(x)] for e, s, segs, x in g]
        print("trade", name, {p: row[f"{p} trades"] for p in periods}, {p: row[f"{p} 1x"] for p in periods},
              flush=True)
    crit = {}
    for name in ("AS4-x", "BS-x"):
        e24, e23 = rep["edge"][name]["2024~"], rep["edge"][name]["2023"]
        tr = rep["trade"][name]["2024~ trades"]
        c1 = e24.get("n", 0) > 0 and e24["edge"] > 0 and e24["boot_p"] < 0.05
        c2 = e23.get("n", 0) > 0 and e23["edge"] > 0
        c3 = tr.get("n", 0) > 0 and tr["avg_pct"] > 0 and (tr.get("long_pct") or 0) >= 0 and \
            (tr.get("short_pct") or 0) >= 0
        crit[name] = {"edge24": bool(c1), "edge23": bool(c2), "trade24": bool(c3), "pass": bool(c1 and c2 and c3)}
    rep["crit"] = crit
    rep["pass"] = any(v["pass"] for v in crit.values())
    print("crit", crit, flush=True)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    (out / "groups.json").write_text(json.dumps(groups))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("predict", "all"):
        predict(out, a.seeds)
    if a.stage in ("eval", "all"):
        evaluate_all(out)


if __name__ == "__main__":
    main()
