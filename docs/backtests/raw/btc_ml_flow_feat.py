"""15분 ML XGBoost 단독: 기존 28개 vs + 체결 흐름·선물 포지션 9개.
등록: 2026-11-07-btc-ml-flow-feat-hypotheses.md (118번)

실행: PYTHONPATH=<pyml>:<pylib>:src:docs/backtests/raw python3 docs/backtests/raw/btc_ml_flow_feat.py \
      --out <폴더> [--stage predict|eval|all]
씨드·모델마다 pred_<base|flow>_k<k>.npz를 따로 저장하고, 이미 있으면 건너뛴다.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
import numpy as np
from btc_fractal import load5
from btc_futures_compare import load_funding
from btc_ml_15m import BAR, EMBARGO, FACTOR, SL, TP
from btc_ml_barrier import barrier_labels
from btc_ml_edge_cost import rest_sim, summ_net
from btc_ml_seeds import RULES, rule_masks

KS = (0, 1, 2, 3, 4)
M5 = 300_000
DATA = Path("~/.minerva/data/btc").expanduser()
YEAR_MS = 365 * 86_400_000
NEW = ("tb15", "tb1h", "tb4h", "f_last", "f_mean3", "p_last", "p_1h", "oi_1h", "oi_4h")


def last_idx(t_avail: np.ndarray, tc: np.ndarray) -> np.ndarray:
    """tc 시각까지 알려진 마지막 행 번호(없으면 −1). t_avail은 값이 알려지는 시각."""
    return np.searchsorted(t_avail, tc, side="right") - 1


def win_sum(x: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    cs = np.r_[0.0, np.cumsum(x)]
    out = np.full(len(j), np.nan)
    ok = j - n + 1 >= 0
    out[ok] = cs[j[ok] + 1] - cs[j[ok] + 1 - n]
    return out


def taker_cols(tc: np.ndarray) -> list:
    t, v, tk = [], [], []
    with (DATA / "5m.csv").open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(int(row[0]))
            v.append(float(row[5]))
            tk.append(float(row[9]))
    t, v, tk = np.array(t, float), np.array(v), np.array(tk)
    j = last_idx(t + M5, tc)
    cols = []
    for n in (3, 12, 48):
        # 연속한 5분봉 n개만: 중간에 빈 시간이 있으면 결측
        span_ok = np.zeros(len(j), bool)
        ok = j - n + 1 >= 0
        span_ok[ok] = (t[j[ok]] - t[j[ok] - n + 1]) == (n - 1) * M5
        with np.errstate(invalid="ignore", divide="ignore"):
            r = win_sum(tk, j, n) / win_sum(v, j, n)
        r[~span_ok | ~np.isfinite(r)] = np.nan
        cols.append(r)
    return cols


def funding_cols(tc: np.ndarray) -> list:
    ft, fr = load_funding()
    j = last_idx(ft, tc)
    last = np.where(j >= 0, fr[np.maximum(j, 0)], np.nan)
    m3 = win_sum(fr, j, 3) / 3
    return [last * 1000, m3 * 1000]


def premium_cols(tc: np.ndarray) -> list:
    t, c = [], []
    with (DATA / "futures/premium_5m.csv").open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            t.append(int(row[0]))
            c.append(float(row[4]))
    t, c = np.array(t, float), np.array(c)
    j = last_idx(t + M5, tc)
    fresh = (j >= 0) & (tc - (t[np.maximum(j, 0)] + M5) <= 3600_000)  # 1시간 넘게 끊기면 결측
    last = np.where(fresh, c[np.maximum(j, 0)], np.nan)
    m12 = win_sum(c, j, 12) / 12
    m12[~fresh] = np.nan
    return [last * 1000, m12 * 1000]


def oi_cols(tc: np.ndarray) -> list:
    ts, oi = [], []
    with (DATA / "futures/metrics.csv").open() as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if not row[2]:
                continue
            ts.append(row[0].replace(" ", "T"))
            oi.append(float(row[2]))
    t = np.array(ts, dtype="datetime64[ms]").astype(np.int64).astype(float)
    order = np.argsort(t, kind="stable")
    t, oi = t[order], np.array(oi)[order]
    cols = []
    for h in (1, 4):
        # metrics create_time은 5분 구간 시작이다(레드팀 11/7). 값은 create_time+5분에 정해진다.
        a, b = last_idx(t + M5, tc), last_idx(t + M5, tc - h * 3_600_000)
        ok = (a >= 0) & (b >= 0)
        ok &= (tc - t[np.maximum(a, 0)] <= 1800_000) & (tc - h * 3_600_000 - t[np.maximum(b, 0)] <= 1800_000)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.log(oi[np.maximum(a, 0)] / oi[np.maximum(b, 0)])
        v[~ok | ~np.isfinite(v)] = np.nan  # 미결제약정 0 기록은 결측
        cols.append(v)
    return cols


def new_features(t: np.ndarray) -> np.ndarray:
    tc = t + BAR
    cols = taker_cols(tc) + funding_cols(tc) + premium_cols(tc) + oi_cols(tc)
    return np.stack(cols, 1).astype(np.float32)


def predict(out: Path) -> None:
    d5 = load5(DATA / "5m.csv")
    B = E.build(d5, FACTOR)
    d = B["d"]
    t = d["t"]
    X0, seq = B["X"], B["seq"]
    XN = new_features(t)
    n = len(t)
    valid = np.isfinite(X0).all(1) & np.isfinite(seq).all(1)
    seq_ok = np.convolve(valid.astype(int), np.ones(E.SEQ, int), "full")[:n] == E.SEQ
    y = barrier_labels(d, TP, SL)
    cover = {nm: str(np.datetime64(int(t[np.flatnonzero(np.isfinite(XN[:, i]))[0]]), "ms"))
             for i, nm in enumerate(NEW)}
    print("first available", cover, flush=True)
    np.savez_compressed(out / "prices.npz", t=t, y=y, o=d["o"], h=d["h"], l=d["l"], c=d["c"])
    (out / "coverage.json").write_text(json.dumps(cover, indent=1))
    for kind in ("base", "flow"):
        for k in KS:
            f = out / f"pred_{kind}_k{k}.npz"
            if f.exists():
                continue
            probs = np.full((1, n), np.nan, np.float32)
            gain = {}
            for Y in E.YEARS:
                te = np.flatnonzero((t >= E.ts(Y)) & (t < E.ts(Y + 1)) & seq_ok)
                if not len(te):
                    continue
                base = (t >= E.ts(2018)) & (t < E.ts(Y) - EMBARGO * BAR) & seq_ok
                i0 = np.flatnonzero(base)
                mu, sd = X0[i0].mean(0), X0[i0].std(0) + 1e-6
                Xn = ((X0 - mu) / sd).astype(np.float32)
                Xn[~np.isfinite(Xn)] = 0
                if kind == "flow":
                    nmu, nsd = np.nanmean(XN[i0], 0), np.nanstd(XN[i0], 0) + 1e-6
                    Xn = np.c_[Xn, ((XN - nmu) / nsd).astype(np.float32)]  # 결측은 NaN 그대로
                tr = np.flatnonzero(base & (y >= 0))[::3]
                seed = Y + 1000 * k
                t0 = time.time()
                p = E.fit_predict_tab("xgb", Xn[tr], y[tr], Xn[te], seed=seed, ncls=2)
                probs[0, te] = p[:, 1]
                print(kind, "k", k, Y, "train", len(tr), "sec", round(time.time() - t0, 1), flush=True)
                if kind == "flow" and k == 0:
                    gain[Y] = importance(Xn[tr], y[tr], seed)
            tmp = out / f"pred_{kind}_k{k}.tmp.npz"
            np.savez_compressed(tmp, probs=probs)
            tmp.replace(f)
            if gain:
                (out / "gain_flow_k0.json").write_text(json.dumps(gain, indent=1))


def importance(Xtr, ytr, seed) -> dict:
    import xgboost as xgb

    prm = {"max_depth": 5, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "nthread": 2,
           "seed": seed, "objective": "binary:logistic"}
    bst = xgb.train(prm, xgb.DMatrix(Xtr, label=ytr), num_boost_round=300)
    g = bst.get_score(importance_type="gain")
    tot = sum(g.values())
    names = [f"b{i}" for i in range(Xtr.shape[1] - len(NEW))] + list(NEW)
    share = {names[int(kk[1:])]: v / tot for kk, v in g.items()}
    return {"new_share": round(sum(share.get(nm, 0) for nm in NEW), 3),
            **{nm: round(share.get(nm, 0), 4) for nm in NEW}}


def auc(p: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(p) & (y >= 0)
    p, y = p[m], y[m]
    r = np.argsort(np.argsort(p)) + 1
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(out: Path) -> None:
    z = np.load(out / "prices.npz")
    t, yl = z["t"], z["y"]
    n = len(t)
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    ft, fr = load_funding()
    old = [np.load(out / f"pred_base_k{k}.npz")["probs"] for k in KS]
    new = [np.load(out / f"pred_flow_k{k}.npz")["probs"] for k in KS]
    i23, i24 = (int(np.searchsorted(t, E.ts(y))) for y in (2023, 2024))
    iy = int(np.searchsorted(t, t[-1] - YEAR_MS))
    W = {"2023": (i23, i24), "2024~": (i24, n), "1y": (iy, n)}

    def run(li, si, per, sides=(1, -1), k=1.0, keep=False):
        a0, a1 = W[per]
        tr = rest_sim(P, t, li, si, a0, a1, sides, k)
        s = summ_net(tr, ft, fr, (t[a1 - 1] - t[a0]) / F.YEAR)
        return (s, tr) if keep else s

    m23 = t >= E.ts(2023)
    avg_old, avg_new = np.mean(old, 0), np.mean(new, 0)
    res = {"auc_2023~": {"base": auc(avg_old[0][m23], yl[m23]), "flow": auc(avg_new[0][m23], yl[m23])},
           "end": str(np.datetime64(int(t[-1]), "ms"))}
    print("auc", res["auc_2023~"], flush=True)
    rng = np.random.default_rng(1107)
    for r, (q, _) in RULES.items():
        lo, so = rule_masks(avg_old, t, q, 1)
        ln, sn = rule_masks(avg_new, t, q, 1)
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
            xo = run(*rule_masks(po, t, q, 1), "2024~")["x3"]
            xn = run(*rule_masks(pn, t, q, 1), "2024~")["x3"]
            per_seed[f"k{k}"] = {"base": xo, "flow": xn}
            beat += xn > xo
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
        _, tr1y = run(ln, sn, "1y", keep=True)
        res[r] = {"base": base, "flow": cand, "long": long_, "short": short_, "fee1.5": fee,
                  "p": p, "rand_median": float(np.median(rand)), "seeds": per_seed,
                  "seed_beat": int(beat), "crit": crit, "pass": bool(all(crit)),
                  "trades_1y": [list(x) for x in tr1y]}
        print(r, json.dumps({k: v for k, v in res[r].items() if k != "trades_1y"}, default=float),
              flush=True)
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=("predict", "eval", "all"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("predict", "all"):
        predict(out)
    if a.stage in ("eval", "all"):
        evaluate(out)


if __name__ == "__main__":
    main()
