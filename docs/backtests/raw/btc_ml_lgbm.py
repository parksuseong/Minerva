"""LightGBM·로지스틱 회귀 추가 확인. 문서: 2026-11-04-btc-ml-lgbm-hypotheses.md

실행: PYTHONPATH=<pyml>:src:docs/backtests/raw python3 docs/backtests/raw/btc_ml_lgbm.py \
      --pred <ml15/pred.npz> --out <폴더> [--stage predict|eval|all]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
import numpy as np
from btc_fractal import load5
from btc_futures_compare import load_funding
from btc_ml_15m import BAR as BAR15
from btc_ml_15m import BE, EMBARGO, FACTOR, SL, TP
from btc_ml_barrier import barrier_labels
from btc_ml_edge_cost import mean_score, rest_sim, summ_net

NEW = ("lgbm", "lr")


def fit(name: str, Xtr, ytr, Xte, seed: int) -> np.ndarray:
    if name == "lgbm":
        import lightgbm as lgb

        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                               min_child_samples=50, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, n_jobs=2, random_state=seed, verbose=-1)
    else:
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=1.0, max_iter=1000)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def predict(out: Path) -> None:
    B = E.build(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), FACTOR)
    X, seq, d = B["X"], B["seq"], B["d"]
    t = d["t"]
    n = len(t)
    valid = np.isfinite(X).all(1) & np.isfinite(seq).all(1)
    seq_ok = np.convolve(valid.astype(int), np.ones(E.SEQ, int), "full")[:n] == E.SEQ
    y = barrier_labels(d, TP, SL)
    probs = np.full((len(NEW), n), np.nan, np.float32)
    for Y in E.YEARS:
        te = np.flatnonzero((t >= E.ts(Y)) & (t < E.ts(Y + 1)) & seq_ok)
        if not len(te):
            continue
        base = (t >= E.ts(2018)) & (t < E.ts(Y) - EMBARGO * BAR15) & seq_ok
        i0 = np.flatnonzero(base)
        mu, sd = X[i0].mean(0), X[i0].std(0) + 1e-6
        Xn = ((X - mu) / sd).astype(np.float32)
        Xn[~np.isfinite(Xn)] = 0
        tr = np.flatnonzero(base & (y >= 0))[::3]
        for mi, name in enumerate(NEW):
            t0 = time.time()
            probs[mi, te] = fit(name, Xn[tr], y[tr], Xn[te], Y)
            print(Y, name, "train", len(tr), "sec", round(time.time() - t0, 1), flush=True)
    np.savez_compressed(out / "pred_new.npz", probs=probs, t=t)


def evaluate(pred: Path, out: Path) -> None:
    z = np.load(pred)
    w = np.load(out / "pred_new.npz")
    t = z["t"]
    assert np.array_equal(t, w["t"]), "봉 시각이 다르다"
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    n = len(t)
    ft, fr = load_funding()
    p5 = z["probs"]
    lg, lr = w["probs"][0], w["probs"][1]
    p6 = np.vstack([p5, lg[None]])
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))

    def votes(p):
        with np.errstate(invalid="ignore"):
            return (p > BE).sum(0), (p < 1 - BE).sum(0)

    def rule(sc, q, k=0, p=None):
        hi, lo = F.thresholds(sc, t, q)
        with np.errstate(invalid="ignore"):
            li = np.isfinite(sc) & (sc >= hi)
            si = np.isfinite(sc) & (sc <= lo)
        if k:
            up, dn = votes(p)
            li, si = li & (up >= k), si & (dn >= k)
        return li, si

    var = {
        "A (5, q90, 4/5)": rule(mean_score(p5), 0.90, 4, p5),
        "A6 (6, q90, 5/6)": rule(mean_score(p6), 0.90, 5, p6),
        "B (5, q95, 5/5)": rule(mean_score(p5), 0.95, 5, p5),
        "B6 (6, q95, 6/6)": rule(mean_score(p6), 0.95, 6, p6),
        "ref 5-mean q90": rule(mean_score(p5), 0.90),
        "ref lgbm q90": rule(lg, 0.90),
        "ref xgb q90": rule(p5[2], 0.90),
        "ref lr q90": rule(lr, 0.90),
    }
    res = {}
    for name, (li, si) in var.items():
        res[name] = {}
        for per, (a0, a1) in (("2023", (i23, i24)), ("2024~", (i24, n))):
            yrs = (t[a1 - 1] - t[a0]) / F.YEAR
            for lab, sides in (("both", (1, -1)), ("long", (1,)), ("short", (-1,))):
                r = summ_net(rest_sim(P, t, li, si, a0, a1, sides), ft, fr, yrs)
                res[name][f"{per} {lab}"] = r
                print(name, per, lab, r, flush=True)
    # 모델 정확도(참고): 라벨이 있는 2023~ 봉의 AUC
    from sklearn.metrics import roc_auc_score

    y = z["y"]
    m = (t >= E.ts(2023)) & (y >= 0)
    auc = {}
    for nm, p in (("rf", p5[0]), ("mlp", p5[1]), ("xgb", p5[2]), ("lstm", p5[3]), ("cnn", p5[4]),
                  ("lgbm", lg), ("lr", lr), ("mean5", mean_score(p5)), ("mean6", mean_score(p6))):
        k = m & np.isfinite(p)
        auc[nm] = round(float(roc_auc_score(y[k], p[k])), 4)
    print("auc", auc, flush=True)
    res["auc"] = auc
    (out / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=("predict", "eval", "all"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("predict", "all"):
        predict(out)
    if a.stage in ("eval", "all"):
        evaluate(Path(a.pred), out)


if __name__ == "__main__":
    main()
