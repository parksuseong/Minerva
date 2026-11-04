"""A·B 씨드 바꿔 다시 학습 + MLP 자신감 점검. 문서: 2026-11-04-btc-ml-seeds-hypotheses.md

실행: PYTHONPATH=<pyml>:src:docs/backtests/raw python3 docs/backtests/raw/btc_ml_seeds.py \
      --pred <ml15/pred.npz> --out <폴더> [--stage predict|eval|mlp|all]
씨드마다 pred_s<k>.npz를 따로 저장한다. 중간에 끊겨도(02시 서버 정지) 끝난 씨드는 건너뛴다.
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
from btc_ml_15m import BAR, BE, EMBARGO, FACTOR, SL, TP
from btc_ml_barrier import barrier_labels
from btc_ml_edge_cost import mean_score, rest_sim, summ_net

SEEDS = (1, 2, 3, 4)
RULES = {"A": (0.90, 4), "B": (0.95, 5)}


def predict(out: Path) -> None:
    B = E.build(load5(Path("~/.minerva/data/btc/5m.csv").expanduser()), FACTOR)
    X, seq, d = B["X"], B["seq"], B["d"]
    t = d["t"]
    n = len(t)
    valid = np.isfinite(X).all(1) & np.isfinite(seq).all(1)
    seq_ok = np.convolve(valid.astype(int), np.ones(E.SEQ, int), "full")[:n] == E.SEQ
    y = barrier_labels(d, TP, SL)
    for k in SEEDS:
        f = out / f"pred_s{k}.npz"
        if f.exists():
            continue
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
            seed = Y + 1000 * k
            for mi, name in enumerate(E.MODELS):
                t0 = time.time()
                if name in ("rf", "mlp", "xgb"):
                    p = E.fit_predict_tab(name, Xn[tr], y[tr], Xn[te], seed=seed, ncls=2)
                else:
                    p = E.fit_predict_seq(name, Sn, Xn, y, tr, te, seed=seed, ncls=2)
                probs[mi, te] = p[:, 1]
                print("seed", k, Y, name, "sec", round(time.time() - t0, 1), flush=True)
        tmp = out / f"pred_s{k}.tmp.npz"
        np.savez_compressed(tmp, probs=probs, t=t)
        tmp.replace(f)


def rule_masks(probs, t, q, k):
    sc = mean_score(probs)
    hi, lo = F.thresholds(sc, t, q)
    with np.errstate(invalid="ignore"):
        up, dn = (probs > BE).sum(0), (probs < 1 - BE).sum(0)
        li = np.isfinite(sc) & (sc >= hi) & (up >= k)
        si = np.isfinite(sc) & (sc <= lo) & (dn >= k)
    return li, si


def run_rules(z, probs, ft, fr) -> dict:
    t = z["t"]
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    n = len(t)
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))
    res = {}
    for r, (q, k) in RULES.items():
        li, si = rule_masks(probs, t, q, k)
        for per, (a0, a1) in (("2023", (i23, i24)), ("2024~", (i24, n))):
            yrs = (t[a1 - 1] - t[a0]) / F.YEAR
            for lab, sides in (("both", (1, -1)), ("long", (1,)), ("short", (-1,))):
                res[f"{r} {per} {lab}"] = summ_net(rest_sim(P, t, li, si, a0, a1, sides), ft, fr,
                                                   yrs)
    return res


def evaluate(pred: Path, out: Path) -> None:
    z = np.load(pred)
    ft, fr = load_funding()
    allr = {"s0": run_rules(z, z["probs"], ft, fr)}
    for k in SEEDS:
        f = out / f"pred_s{k}.npz"
        if not f.exists():
            print("missing seed", k, flush=True)
            continue
        w = np.load(f)
        assert np.array_equal(w["t"], z["t"])
        allr[f"s{k}"] = run_rules(z, w["probs"], ft, fr)
    for s, r in allr.items():
        for key, v in r.items():
            print(s, key, v, flush=True)
    verdict = {}
    for r in RULES:
        new = [allr[s] for s in allr if s != "s0"]
        ok = sum(x[f"{r} 2023 both"]["x3"] > 0 and x[f"{r} 2024~ both"]["x3"] > 0 for x in new)
        ev0 = allr["s0"][f"{r} 2024~ both"]["ev3"]
        evm = float(np.median([x[f"{r} 2024~ both"]["ev3"] for x in new])) if new else None
        verdict[r] = {"seeds": len(new), "both_plus": ok, "ev3_orig": ev0, "ev3_median": evm,
                      "holds": bool(len(new) == 4 and ok >= 3 and evm is not None
                                    and evm >= ev0 / 2)}
    print("verdict", verdict, flush=True)
    allr["verdict"] = verdict
    (out / "seeds.json").write_text(json.dumps(allr, indent=1, default=float))


def mlp_check(pred: Path, out: Path) -> None:
    z = np.load(pred)
    t, y, probs = z["t"], z["y"], z["probs"]
    m = (t >= E.ts(2023)) & (y >= 0)
    cal = {}
    for mi, name in enumerate(E.MODELS):
        p = probs[mi]
        k = m & np.isfinite(p)
        rows = []
        for lo in np.arange(0.0, 1.0, 0.1):
            b = k & (p >= lo) & (p < lo + 0.1)
            if b.sum() >= 200:
                rows.append({"bin": round(float(lo), 1), "n": int(b.sum()),
                             "pred": round(float(p[b].mean()), 3),
                             "real": round(float(y[b].mean()), 3)})
        hi = k & (p >= 0.7)
        cal[name] = {"bins": rows, "n_ge70": int(hi.sum()),
                     "real_ge70": round(float(y[hi].mean()), 3) if hi.any() else None,
                     "mean_pred": round(float(p[k].mean()), 3), "base": round(float(y[k].mean()), 3)}
        print(name, cal[name], flush=True)
    mlp_i = E.MODELS.index("mlp")
    verdict = {"overconfident": bool(cal["mlp"]["real_ge70"] is not None
                                     and cal["mlp"]["real_ge70"] < 0.6)}
    # 참고: MLP 동의가 없었으면 빠졌을 신호 비율, MLP 뺀 4개 평균의 A·B
    rest = np.delete(probs, mlp_i, 0)
    dep = {}
    for r, (q, k) in RULES.items():
        li, si = rule_masks(probs, t, q, k)
        sel = (li | si) & (t >= E.ts(2023))
        with np.errstate(invalid="ignore"):
            up4, dn4 = (rest > BE).sum(0), (rest < 1 - BE).sum(0)
        need = (li & (up4 < k)) | (si & (dn4 < k))
        dep[r] = {"signals": int(sel.sum()), "need_mlp_vote": int((need & sel).sum())}
    ft, fr = load_funding()
    no_mlp = {}
    t4 = {"A": (0.90, 3), "B": (0.95, 4)}  # 4개 중 같은 비율: A 3/4, B 4/4
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    n = len(t)
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))
    for r, (q, k) in t4.items():
        li, si = rule_masks(rest, t, q, k)
        for per, (a0, a1) in (("2023", (i23, i24)), ("2024~", (i24, n))):
            yrs = (t[a1 - 1] - t[a0]) / F.YEAR
            no_mlp[f"{r} {per}"] = summ_net(rest_sim(P, t, li, si, a0, a1, (1, -1)), ft, fr, yrs)
    print("verdict", verdict, "dep", dep, "no_mlp", no_mlp, flush=True)
    (out / "mlp.json").write_text(json.dumps({"cal": cal, "verdict": verdict, "dep": dep,
                                              "no_mlp": no_mlp}, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=("predict", "eval", "mlp", "all"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("mlp", "all"):
        mlp_check(Path(a.pred), out)
    if a.stage in ("predict", "all"):
        predict(out)
    if a.stage in ("eval", "all"):
        evaluate(Path(a.pred), out)


if __name__ == "__main__":
    main()
