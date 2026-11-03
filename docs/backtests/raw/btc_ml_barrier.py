"""머신러닝 5모델로 "먼저 +X%냐 −Y%냐" 예측 + 30분봉 SCR. 등록: 2026-11-03-btc-ml-barrier-hypotheses.md

실행: PYTHONPATH=<numpy·sklearn·xgboost·torch 경로>:src:docs/backtests/raw \
      python docs/backtests/raw/btc_ml_barrier.py --out <폴더> [--stage predict|trade|all]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import btc_ml_ensemble as E
from btc_fractal import load5, scr3

BARRIERS = {"B1": (0.03, 0.01), "B2": (0.02, 0.01), "B3": (0.015, 0.015)}
MS = (0.05, 0.10, 0.15)
COST = 0.0005
EMBARGO = 300


def breakeven(tp: float, sl: float) -> float:
    return (sl + 2 * COST) / (tp + sl)


def barrier_labels(d: dict, tp: float, sl: float) -> np.ndarray:
    o, h, l = d["o"], d["h"], d["l"]  # noqa: E741
    n = len(o)
    y = np.full(n, -1, np.int64)
    for i in range(n - 1):
        px = o[i + 1]
        up, dn = px * (1 + tp), px * (1 - sl)
        for j in range(i + 1, n):
            if l[j] <= dn:
                y[i] = 0
                break
            if h[j] >= up:
                y[i] = 1
                break
    return y


def predict(out: Path) -> None:
    B = E.build(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()))
    X, seq, t, d = B["X"], B["seq"], B["d"]["t"], B["d"]
    n = len(t)
    valid = np.isfinite(X).all(1) & np.isfinite(seq).all(1)
    seq_ok = np.convolve(valid.astype(int), np.ones(E.SEQ, int), "full")[:n] == E.SEQ
    on_l, *_ = scr3(d)
    labs = {k: barrier_labels(d, *v) for k, v in BARRIERS.items()}
    probs = np.full((len(BARRIERS), len(E.MODELS), n), np.nan, np.float32)
    for Y in E.YEARS:
        te = np.flatnonzero((t >= E.ts(Y)) & (t < E.ts(Y + 1)) & seq_ok)
        if not len(te):
            continue
        base_tr = (t >= E.ts(2018)) & (t < E.ts(Y) - EMBARGO * E.M30) & seq_ok
        idx0 = np.flatnonzero(base_tr)
        mu, sd = X[idx0].mean(0), X[idx0].std(0) + 1e-6
        Xn = ((X - mu) / sd).astype(np.float32)
        Xn[~np.isfinite(Xn)] = 0
        smu, ssd = seq[idx0].mean(0), seq[idx0].std(0) + 1e-6
        Sn = ((seq - smu) / ssd).astype(np.float32)
        Sn[~np.isfinite(Sn)] = 0
        for bi, bk in enumerate(BARRIERS):
            y = labs[bk]
            tr = np.flatnonzero(base_tr & (y >= 0))[::2]
            for mi, name in enumerate(E.MODELS):
                t0 = time.time()
                if name in ("rf", "mlp", "xgb"):
                    p = E.fit_predict_tab(name, Xn[tr], y[tr], Xn[te], seed=Y, ncls=2)
                else:
                    p = E.fit_predict_seq(name, Sn, Xn, y, tr, te, seed=Y, ncls=2)
                probs[bi, mi, te] = p[:, 1]
                print(Y, bk, name, "train", len(tr), "pos", round(float(y[tr].mean()), 3),
                      "sec", round(time.time() - t0, 1), flush=True)
    np.savez_compressed(out / "pred.npz", probs=probs, t=t, o=d["o"], h=d["h"], l=d["l"],
                        c=d["c"], scr=on_l, **{f"y_{k}": v for k, v in labs.items()})


def simulate(P: dict, entry: np.ndarray, tp: float, sl: float, i0: int, i1: int) -> list:
    o, h, l = P["o"], P["h"], P["l"]  # noqa: E741
    out, i = [], i0
    while i < i1 - 1:
        if not entry[i]:
            i += 1
            continue
        px = o[i + 1]
        up, dn = px * (1 + tp), px * (1 - sl)
        j, res = i + 1, None
        while j < len(o):
            if l[j] <= dn:
                res = (i, min(o[j], dn) / px - 1, 0)
                break
            if h[j] >= up:
                res = (i, max(o[j], up) / px - 1 if o[j] > up else tp, 1)
                break
            j += 1
        if res is None:
            break
        out.append(res)
        i = j
    return out


def summ(tr: list) -> dict:
    if not tr:
        return {"n": 0, "hit": 0.0, "x1": 0.0}
    net = np.array([g for _, g, _ in tr]) - 2 * COST
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    return {"n": len(tr), "hit": round(float(np.mean([w for *_, w in tr]) * 100), 1),
            "ev": round(float(net.mean() * 100), 3), "x1": round(float((eq[-1] - 1) * 100), 1),
            "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1)}


def trade_stage(out: Path) -> None:
    z = np.load(out / "pred.npz")
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    probs, t, scr = z["probs"], z["t"], z["scr"]
    n = len(t)
    i22, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2022, 2024))
    known = np.isfinite(probs).all(1)
    rng = np.random.default_rng(71)
    rep: dict = {}
    for ev_name, ev in (("E1_scr", scr), ("E2_all", np.ones(n, bool))):
        rows = {}
        for bi, (bk, (tp, sl)) in enumerate(BARRIERS.items()):
            be = breakeven(tp, sl)
            pb = probs[bi]
            votes = (pb > be).sum(0)
            conf = np.nanmean(pb, 0)
            for m in MS:
                ent = ev & known[bi] & (votes >= 3) & (conf >= be + m)
                rows[(bk, m)] = {"sel": summ(simulate(P, ent, tp, sl, i22, i24)),
                                 "entry": ent, "be": be}
        best = max(rows, key=lambda k: rows[k]["sel"]["x1"] if rows[k]["sel"]["n"] else -1e9)
        bk, m = best
        tp, sl = BARRIERS[bk]
        be = rows[best]["be"]
        ent = rows[best]["entry"]
        tr = simulate(P, ent, tp, sl, i24, n)
        test = summ(tr)
        y = z[f"y_{bk}"]
        cand = np.flatnonzero(ev & (y >= 0) & (np.arange(n) >= i24) & known[list(BARRIERS).index(bk)])
        all_hit = float(y[cand].mean() * 100) if len(cand) else float("nan")
        k = test["n"]
        draws = np.array([y[rng.choice(cand, k, replace=False)].mean() * 100
                          for _ in range(1000)]) if 0 < k <= len(cand) else np.array([100.0])
        p = float((draws >= test["hit"]).mean())
        crit = [k >= 30 and test["hit"] >= be * 100 + 5 and test["x1"] > 0,
                test["hit"] >= all_hit + 5, p < 0.05]
        sel_table = {f"{b}|m{mm}": r["sel"] for (b, mm), r in rows.items()}
        rep[ev_name] = {"select": sel_table, "best": f"{bk}|m{m}", "breakeven": round(be * 100, 1),
                        "test": test, "all_signals_hit": round(all_hit, 1),
                        "n_candidates": int(len(cand)), "p": p, "criteria": crit}
        print(ev_name, "select", {kk: (v["n"], v["hit"], v["x1"]) for kk, v in sel_table.items()},
              flush=True)
        print(ev_name, "PICK", f"{bk}|m{m}", "be", round(be * 100, 1), "test", test, "all-hit",
              round(all_hit, 1), "p", p, "crit", crit, flush=True)
    # 모델별 판별력: 2024~ 적중 확률 상위 10% 봉의 실제 적중률
    rep["top_decile"] = {}
    for bi, bk in enumerate(BARRIERS):
        y = z[f"y_{bk}"]
        m = known[bi] & (y >= 0) & (np.arange(n) >= i24)
        row = {"base": round(float(y[m].mean() * 100), 1)}
        for mi, name in enumerate(E.MODELS):
            pv = probs[bi, mi][m]
            q = np.quantile(pv, 0.9)
            row[name] = round(float(y[m][pv >= q].mean() * 100), 1)
        rep["top_decile"][bk] = row
        print("top10%", bk, row, flush=True)
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
