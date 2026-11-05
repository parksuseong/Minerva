"""15분 ML + 30분·1시간·4시간·일봉 추세 입력. 등록: 2026-11-05-btc-ml-mtf-feat-hypotheses.md

실행: PYTHONPATH=<pyml>:src:docs/backtests/raw python3 docs/backtests/raw/btc_ml_mtf_feat.py \
      --pred <ml15/pred.npz> --seeds <seeds 폴더> --out <폴더> [--stage predict|eval|all]
씨드마다 pred_k<k>.npz를 따로 저장한다. 중간에 끊겨도(02시 서버 정지) 끝난 씨드는 건너뛴다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
import numpy as np
from btc_fractal import load5, resample
from btc_futures_compare import load_funding
from btc_ml_15m import BAR, EMBARGO, FACTOR, SL, TP
from btc_ml_barrier import barrier_labels
from btc_ml_edge_cost import rest_sim, summ_net
from btc_ml_mtf_trend import filt, load_frame, trend_at
from btc_ml_seeds import RULES, rule_masks

KS = (0, 1, 2, 3, 4)  # 씨드 Y + 1000·k
FRAMES = ((6, "30m"), (12, "1h"), (48, "4h"), (288, "1d"))
EMA_N, SLOPE = 50, 5
YEAR_MS = 365 * 86_400_000


def ema(c: np.ndarray, n: int) -> np.ndarray:
    a = 2 / (n + 1)
    out = np.empty_like(c)
    out[0] = c[0]
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def mtf_cols(d5: dict, tc: np.ndarray) -> np.ndarray:
    cols = []
    for fac, _ in FRAMES:
        f = resample(d5, fac)
        c = f["c"]
        e = ema(c, EMA_N)
        prev = np.r_[np.full(SLOPE, np.nan), e[:-SLOPE]]
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = c / e - 1
            slope = np.log(e / prev)
            tr = np.where((c > e) & (e > prev), 1.0, np.where((c < e) & (e < prev), -1.0, 0.0))
        dist[:EMA_N], slope[:EMA_N], tr[:EMA_N] = np.nan, np.nan, np.nan
        close = f["t"] + fac * 300_000
        for v in (dist, slope, tr):
            cols.append(E.asof(close, v, tc))
    return np.stack(cols, 1).astype(np.float32)


def predict(out: Path) -> None:
    d5 = load5(Path("~/.minerva/data/btc/5m.csv").expanduser())
    B = E.build(d5, FACTOR)
    d = B["d"]
    t = d["t"]
    X = np.concatenate([B["X"], mtf_cols(d5, t + BAR)], 1)
    seq = B["seq"]
    n = len(t)
    valid = np.isfinite(X).all(1) & np.isfinite(seq).all(1)
    seq_ok = np.convolve(valid.astype(int), np.ones(E.SEQ, int), "full")[:n] == E.SEQ
    y = barrier_labels(d, TP, SL)
    for k in KS:
        f = out / f"pred_k{k}.npz"
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
                print("k", k, Y, name, "sec", round(time.time() - t0, 1), flush=True)
        tmp = out / f"pred_k{k}.tmp.npz"
        np.savez_compressed(tmp, probs=probs, t=t, y=y)
        tmp.replace(f)


def auc(p: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(p) & (y >= 0)
    p, y = p[m], y[m]
    r = np.argsort(np.argsort(p)) + 1
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(pred: Path, seeds: Path, out: Path) -> None:
    z = np.load(pred)
    t = z["t"]
    n = len(t)
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    ft, fr = load_funding()
    old = [z["probs"]] + [np.load(seeds / f"pred_s{k}.npz")["probs"] for k in (1, 2, 3, 4)]
    new = []
    for k in KS:
        w = np.load(out / f"pred_k{k}.npz")
        assert len(w["t"]) >= n and np.array_equal(w["t"][:n], t)
        new.append(w["probs"][:, :n])
    yl = np.load(out / "pred_k0.npz")["y"][:n]
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))
    iy = int(np.searchsorted(t, t[-1] - YEAR_MS))
    W = {"2023": (i23, i24), "2024~": (i24, n), "1y": (iy, n)}
    T = {"4h": trend_at(t, load_frame("4h"))}
    # filt는 세 프레임 이름을 쓰지만 F4h만 쓴다
    T.update({"30m": T["4h"], "1h": T["4h"]})

    def run(li, si, per, sides=(1, -1), k=1.0):
        a0, a1 = W[per]
        return summ_net(rest_sim(P, t, li, si, a0, a1, sides, k), ft, fr,
                        (t[a1 - 1] - t[a0]) / F.YEAR)

    m23 = t >= E.ts(2023)
    res = {"auc": {"old": {m: auc(np.mean(old, 0)[i][m23], yl[m23])
                           for i, m in enumerate(E.MODELS)},
                   "new": {m: auc(np.mean(new, 0)[i][m23], yl[m23])
                           for i, m in enumerate(E.MODELS)}}}
    print("auc", res["auc"], flush=True)
    avg_old, avg_new = np.mean(old, 0), np.mean(new, 0)
    rng = np.random.default_rng(1106)
    for r, (q, kk) in RULES.items():
        rule = r + "S"
        lo, so = rule_masks(avg_old, t, q, kk)
        ln, sn = rule_masks(avg_new, t, q, kk)
        base = {per: run(lo, so, per) for per in W}
        cand = {per: run(ln, sn, per) for per in W}
        long_, short_ = run(ln, sn, "2024~", (1,)), run(ln, sn, "2024~", (-1,))
        fee = run(ln, sn, "2024~", k=1.5)
        seg_l, seg_s = ln[i24:].copy(), sn[i24:].copy()
        rand = []
        for _ in range(200):
            kk2 = int(rng.integers(96 * 30, len(seg_l) - 96 * 30))
            rl, rs = ln.copy(), sn.copy()
            rl[i24:], rs[i24:] = np.roll(seg_l, kk2), np.roll(seg_s, kk2)
            rand.append(run(rl, rs, "2024~")["x3"])
        p = float((np.array(rand) >= cand["2024~"]["x3"]).mean())
        beat, per_seed = 0, {}
        for k, (po, pn) in enumerate(zip(old, new)):
            a, b = rule_masks(po, t, q, kk), rule_masks(pn, t, q, kk)
            xo, xn = run(*a, "2024~")["x3"], run(*b, "2024~")["x3"]
            per_seed[f"k{k}"] = {"old": xo, "new": xn}
            beat += xn > xo
        lf, sf = filt(ln, sn, "F4h", T)
        ref4 = {per: run(lf, sf, per) for per in W}
        crit = [
            cand["2023"]["x3"] > 0 and cand["2024~"]["x3"] > 0
            and cand["2023"]["x3"] > base["2023"]["x3"]
            and cand["2024~"]["x3"] > base["2024~"]["x3"],
            cand["1y"]["x3"] > 0 and cand["1y"]["per_year"] >= 50,
            long_["ev3"] > 0 and short_["ev3"] > 0,
            fee["x3"] > 0,
            p < 0.01,
            beat >= 3,
        ]
        res[rule] = {"old": base, "new": cand, "long": long_, "short": short_, "fee1.5": fee,
                     "p": p, "rand_median": float(np.median(rand)), "seeds": per_seed,
                     "seed_beat": beat, "ref_new_F4h": ref4, "crit": crit,
                     "pass": bool(all(crit))}
        print(rule, json.dumps(res[rule], default=float), flush=True)
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=("predict", "eval", "all"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("predict", "all"):
        predict(out)
    if a.stage in ("eval", "all"):
        evaluate(Path(a.pred), Path(a.seeds), out)


if __name__ == "__main__":
    main()
