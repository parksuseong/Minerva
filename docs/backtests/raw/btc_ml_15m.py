"""15분봉 마감마다 모델 확률 + SCR로 롱 진입·청산. 등록: 2026-11-03-btc-ml-15m-hypotheses.md

실행: PYTHONPATH=<numpy·sklearn·xgboost·torch 경로>:src:docs/backtests/raw \
      python docs/backtests/raw/btc_ml_15m.py --out <폴더> [--stage predict|trade|all]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import btc_ml_ensemble as E
from btc_fractal import load5, recent, scr3
from btc_ml_barrier import barrier_labels

FACTOR = 3
BAR = FACTOR * 300_000
TP = SL = 0.015
BE = (SL + 0.001) / (TP + SL)
MS = (0.03, 0.06, 0.10)
COST = 0.0005
STOP = 0.03
EMBARGO = 600
WIN = 4


def predict(out: Path) -> None:
    B = E.build(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), FACTOR)
    X, seq, d = B["X"], B["seq"], B["d"]
    t = d["t"]
    n = len(t)
    valid = np.isfinite(X).all(1) & np.isfinite(seq).all(1)
    seq_ok = np.convolve(valid.astype(int), np.ones(E.SEQ, int), "full")[:n] == E.SEQ
    y = barrier_labels(d, TP, SL)
    on_l, on_s, *_ = scr3(d)
    probs = np.full((len(E.MODELS), n), np.nan, np.float32)
    for Y in E.YEARS:
        te = np.flatnonzero((t >= E.ts(Y)) & (t < E.ts(Y + 1)) & seq_ok)
        if not len(te):
            continue
        base = (t >= E.ts(2018)) & (t < E.ts(Y) - EMBARGO * BAR) & seq_ok
        i0 = np.flatnonzero(base)
        mu, sd = X[i0].mean(0), X[i0].std(0) + 1e-6
        Xn = ((X - mu) / sd).astype(np.float32)
        Xn[~np.isfinite(Xn)] = 0
        smu, ssd = seq[i0].mean(0), seq[i0].std(0) + 1e-6
        Sn = ((seq - smu) / ssd).astype(np.float32)
        Sn[~np.isfinite(Sn)] = 0
        tr = np.flatnonzero(base & (y >= 0))[::3]
        for mi, name in enumerate(E.MODELS):
            t0 = time.time()
            if name in ("rf", "mlp", "xgb"):
                p = E.fit_predict_tab(name, Xn[tr], y[tr], Xn[te], seed=Y, ncls=2)
            else:
                p = E.fit_predict_seq(name, Sn, Xn, y, tr, te, seed=Y, ncls=2)
            probs[mi, te] = p[:, 1]
            print(Y, name, "train", len(tr), "sec", round(time.time() - t0, 1), flush=True)
    np.savez_compressed(out / "pred.npz", probs=probs, y=y, t=t, o=d["o"], h=d["h"], l=d["l"],
                        c=d["c"], on_l=on_l, on_s=on_s)


def simulate(P, ent, ex, i0, i1) -> list:
    o, l = P["o"], P["l"]  # noqa: E741
    out, inpos, px, line, e = [], False, 0.0, 0.0, 0
    for i in range(max(i0, 1), i1):
        if inpos:
            if ex[i - 1]:
                out.append((e, o[i] / px - 1))
                inpos = False
            elif l[i] <= line:
                out.append((e, min(o[i], line) / px - 1))
                inpos = False
                continue
            else:
                continue
        if ent[i - 1]:
            inpos, px, e = True, o[i], i
            line = px * (1 - STOP)
            if l[i] <= line:
                out.append((e, line / px - 1))
                inpos = False
    if inpos:
        out.append((e, P["c"][i1 - 1] / px - 1))
    return out


def summ(tr) -> dict:
    if not tr:
        return {"n": 0, "x1": 0.0}
    net = np.array([g for _, g in tr]) - 2 * COST
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    w, lo = net[net > 0], net[net <= 0]
    return {"n": len(tr), "win": round(float((net > 0.001).mean() * 100), 1),
            "ev": round(float(net.mean() * 100), 3), "x1": round(float((eq[-1] - 1) * 100), 1),
            "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
            "avg_win": round(float(w.mean() * 100), 2) if len(w) else 0.0,
            "avg_loss": round(float(lo.mean() * 100), 2) if len(lo) else 0.0}


def rules(probs, on_l, on_s, m):
    known = np.isfinite(probs).all(0)
    pm = np.where(known, np.nanmean(probs, 0), np.nan)
    votes = (probs > BE).sum(0)
    zl, zs = recent(on_l, WIN), recent(on_s, WIN)
    with np.errstate(invalid="ignore"):
        ent = known & zl & (votes >= 3) & (pm >= BE + m)
        ex = zs & known & (pm < 0.5)
    return ent, ex


def trade_stage(out: Path) -> None:
    z = np.load(out / "pred.npz")
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    probs, t, on_l, on_s, y = z["probs"], z["t"], z["on_l"], z["on_s"], z["y"]
    n = len(t)
    i22, i24 = (int(np.searchsorted(t, E.ts(v))) for v in (2022, 2024))
    known = np.isfinite(probs).all(0)
    m24 = known & (y >= 0) & (np.arange(n) >= i24)
    acc = {"base": round(float(y[m24].mean() * 100), 1)}
    for mi, name in enumerate(E.MODELS):
        pv = probs[mi][m24]
        acc[name] = round(float(y[m24][pv >= np.quantile(pv, 0.9)].mean() * 100), 1)
    pm = np.nanmean(probs, 0)[m24]
    acc["mean"] = round(float(y[m24][pm >= np.quantile(pm, 0.9)].mean() * 100), 1)
    print("top10% 2024~", acc, "be", round(BE * 100, 1), flush=True)
    sel = {m: summ(simulate(P, *rules(probs, on_l, on_s, m), i22, i24)) for m in MS}
    for m, r in sel.items():
        print("select m", m, r, flush=True)
    best = max(MS, key=lambda m: sel[m]["x1"] if sel[m]["n"] else -1e9)
    ent, ex = rules(probs, on_l, on_s, best)
    test = summ(simulate(P, ent, ex, i24, n))
    zl, zs = recent(on_l, WIN), recent(on_s, WIN)
    scr_only = summ(simulate(P, zl, zs, i24, n))
    rng = np.random.default_rng(81)
    rand = []
    seg = probs[:, i24:].copy()
    L = seg.shape[1]
    for _ in range(200):
        k = int(rng.integers(96 * 30, L - 96 * 30))
        p2 = probs.copy()
        p2[:, i24:] = np.roll(seg, k, axis=1)
        rand.append(summ(simulate(P, *rules(p2, on_l, on_s, best), i24, n))["x1"])
    rand = np.array(rand)
    p = float((rand >= test["x1"]).mean())
    crit = [test["n"] >= 30 and test["x1"] > 0 and test.get("ev", -1) > 0,
            test["x1"] > scr_only["x1"], p < 0.05]
    rep = {"top10": acc, "select": sel, "m": best, "test": test, "scr_only": scr_only,
           "rand_median": float(np.median(rand)), "p": p, "criteria": crit}
    print("PICK m", best, "test", test, "scr_only", scr_only, "p", p, "crit", crit, flush=True)
    (out / "result.json").write_text(json.dumps(rep, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("predict", "all"):
        predict(out)
    if a.stage in ("trade", "all"):
        trade_stage(out)


if __name__ == "__main__":
    main()
