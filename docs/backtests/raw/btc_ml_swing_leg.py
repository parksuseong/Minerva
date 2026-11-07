"""121번: RSI 7 스윙 다리(상승/하락)를 맞히는 XGBoost, 선물 15분봉 매매.
등록: 2026-11-07-btc-ml-swing-leg-hypotheses.md

실행: PYTHONPATH=<pyml>:<numpy>:src:docs/backtests/raw python docs/backtests/raw/btc_ml_swing_leg.py \
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
from btc_scr_entry_stop import summ, with_funding
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, ystart

BAR = 900_000
STOP = 0.01
YEARS = (2022, 2023, 2024, 2025, 2026)
SEEDS = 5
VARIANTS = {"H1": (0.60, 0.40), "H2": (0.80, 0.20)}


def pivots(h, l, r):
    """(확정 봉, 극값 봉, 가격, 종류 +1 저점 / −1 고점). 114번 포팅과 같은 규칙에 극값 봉을 더했다."""
    out = []
    lo_zone = hi_zone = False
    lo_p = hi_p = 0.0
    lo_i = hi_i = 0
    for i in range(1, len(r)):
        if np.isnan(r[i]):
            continue
        if r[i] < 30:
            if not lo_zone or l[i] < lo_p:
                lo_p, lo_i = l[i], i
            lo_zone = True
        elif lo_zone:
            lo_zone = False
            out.append((i, lo_i, lo_p, 1))
        if r[i] > 70:
            if not hi_zone or h[i] > hi_p:
                hi_p, hi_i = h[i], i
            hi_zone = True
        elif hi_zone:
            hi_zone = False
            out.append((i, hi_i, hi_p, -1))
    return out


def zigzag(pv):
    """같은 종류가 이어지면 더 극단인 것만 남긴다. 확정 봉은 남긴 피벗이 확정된 봉."""
    zz = []
    for p in pv:
        if zz and zz[-1][3] == p[3]:
            better = p[2] < zz[-1][2] if p[3] == 1 else p[2] > zz[-1][2]
            if better:
                zz[-1] = p
            continue
        zz.append(p)
    return zz


def leg_labels(n: int, zz) -> tuple:
    """y: 1 상승 다리, 0 하락 다리, −1 모름. reach: 그 다리를 끝내는 피벗이 확정되는 봉."""
    y = np.full(n, -1, np.int64)
    reach = np.full(n, -1, np.int64)
    for k in range(len(zz) - 1):
        a, b = zz[k], zz[k + 1]
        s, e = a[1], b[1]
        y[s:e] = 1 if a[3] == 1 else 0
        # b는 다음 반대 피벗이 확정돼야 더 극단인 같은 종류로 바뀌지 않는다
        reach[s:e] = zz[k + 2][0] if k + 2 < len(zz) else b[0]
    return y, reach


def swing_feats(c, r7, pv, n):
    """봉 마감 때 확정된 스윙만 쓴다."""
    since = np.full(n, np.nan)
    d_hi = np.full(n, np.nan)
    d_lo = np.full(n, np.nan)
    kind = np.full(n, np.nan)
    k, last_i, last_hi, last_lo, last_kind = 0, None, None, None, None
    for i in range(n):
        while k < len(pv) and pv[k][0] <= i:
            ci, _, px, kd = pv[k]
            last_i, last_kind = ci, kd
            if kd == 1:
                last_lo = px
            else:
                last_hi = px
            k += 1
        if last_i is not None:
            since[i] = i - last_i
            kind[i] = last_kind
        if last_hi is not None:
            d_hi[i] = c[i] / last_hi - 1
        if last_lo is not None:
            d_lo[i] = c[i] / last_lo - 1
    return np.stack([r7, np.log1p(since), d_hi, d_lo, kind], 1).astype(np.float32)


def fit(Xtr, ytr, Xte, seed):
    import xgboost as xgb

    prm = {"max_depth": 5, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "nthread": 2,
           "seed": seed, "objective": "binary:logistic"}
    bst = xgb.train(prm, xgb.DMatrix(Xtr, label=ytr), num_boost_round=300)
    return bst.predict(xgb.DMatrix(Xte))


def predict(out: Path) -> None:
    d5 = load5(Path("~/.minerva/data/btc/5m.csv").expanduser())
    B = E.build(d5, 3)
    d, X = B["d"], B["X"]
    t, c, h, l = d["t"], d["c"], d["h"], d["l"]
    n = len(t)
    r7 = rsi(c, 7)
    pv = pivots(h, l, r7)
    zz = zigzag(pv)
    y, reach = leg_labels(n, zz)
    X = np.concatenate([X, swing_feats(c, r7, pv, n)], 1)
    look = np.where(reach >= 0, reach - np.arange(n), 0)
    emb = int(max(2000, look.max()))
    valid = np.isfinite(X).all(1)
    probs = np.full((SEEDS, n), np.nan, np.float32)
    prev = np.full((SEEDS, n), np.nan, np.float32)  # 2022 문턱용: 2022 모델의 2021 예측
    for Y in YEARS:
        te = np.flatnonzero((t >= E.ts(Y)) & (t < E.ts(Y + 1)))
        if not len(te):
            continue
        base = (t >= E.ts(2018)) & (np.arange(n) < int(np.searchsorted(t, E.ts(Y))) - emb) & valid
        i0 = np.flatnonzero(base)
        mu, sd = X[i0].mean(0), X[i0].std(0) + 1e-6
        Xn = ((X - mu) / sd).astype(np.float32)
        Xn[~np.isfinite(Xn)] = 0
        tr = np.flatnonzero(base & (y >= 0))[::3]
        pr = np.flatnonzero((t >= E.ts(Y - 1)) & (t < E.ts(Y))) if Y == YEARS[0] else np.array([], int)
        for s in range(SEEDS):
            t0 = time.time()
            idx = np.r_[te, pr]
            p = fit(Xn[tr], y[tr], Xn[idx], seed=Y + 1000 * s)
            probs[s, te] = p[: len(te)]
            if len(pr):
                prev[s, pr] = p[len(te):]
            print(Y, s, "train", len(tr), "sec", round(time.time() - t0, 1), flush=True)
    np.savez_compressed(out / "pred.npz", probs=probs, prev=prev, y=y, t=t, emb=emb,
                        zz=np.array([(a, b, p_, k) for a, b, p_, k in zz], float))
    print("embargo", emb, "pivots", len(zz), flush=True)


def thresholds(t, p, p_prev, qs):
    """연도 → (위, 아래) 문턱. 전년 표본 밖 확률, 2022는 2022 모델의 2021 예측."""
    out = {}
    for Y in YEARS:
        src = p_prev if Y == YEARS[0] else p
        m = (t >= E.ts(Y - 1)) & (t < E.ts(Y)) & np.isfinite(src)
        out[Y] = (float(np.quantile(src[m], qs[0])), float(np.quantile(src[m], qs[1])))
    return out


def simulate(F, fund, pf, thr_hi, thr_lo, start, cost_side=0.0005):
    """pf: 선물 봉에 맞춘 확률. 새로 넘는 봉 마감에만 신호. 반환 거래 목록(summ 형식)."""
    o, h, l, t = F["o"], F["h"], F["l"], F["t"]
    n = len(o)
    up = np.zeros(n, bool)
    dn = np.zeros(n, bool)
    ok = np.isfinite(pf) & np.r_[False, np.isfinite(pf[:-1])]
    up[1:] = ok[1:] & (pf[1:] >= thr_hi[1:]) & (pf[:-1] < thr_hi[:-1])
    dn[1:] = ok[1:] & (pf[1:] <= thr_lo[1:]) & (pf[:-1] > thr_lo[:-1])
    tr = []
    side, e, px, sp = 0, -1, 0.0, 0.0
    for j in range(start, n - 1):
        if side and j >= e:
            if (side > 0 and l[j] <= sp) or (side < 0 and h[j] >= sp):
                fill = min(o[j], sp) if side > 0 else max(o[j], sp)
                g = with_funding((fill / px - 1) * side, F, e - 1, j, side, fund)
                tr.append((e - 1, side, float(g), j, float(t[e - 1]), True, STOP))
                side = 0
        want = 1 if up[j] else (-1 if dn[j] else 0)
        if want and want != side:
            if side:
                g = with_funding((o[j + 1] / px - 1) * side, F, e - 1, j + 1, side, fund)
                tr.append((e - 1, side, float(g), j + 1, float(t[e - 1]), False, STOP))
            side, e, px = want, j + 1, o[j + 1]
            sp = px * (1 - STOP * side)
    return tr


def block(tr, cost=0.0005):
    return summ(tr, cost)


def report(tr, days):
    old = [z for z in tr if ystart(2023) <= z[4] < ystart(2024)]
    new = [z for z in tr if z[4] >= ystart(2024)]
    g = sorted(z[2] for z in tr)
    return {"all": block(tr), "y2022": block([z for z in tr if z[4] < ystart(2023)]), "y2023": block(old),
            "new": block(new), "new_long": block([z for z in new if z[1] > 0]),
            "new_short": block([z for z in new if z[1] < 0]), "new_c15": block(new, 0.00075),
            "drop5": round((float(np.mean(g[:-5])) - 0.001) * 100, 4) if len(g) > 5 else None,
            "per_day": round(len(tr) / days, 3)}


def trade_stage(out: Path, circ: int) -> None:
    P = np.load(out / "pred.npz")
    ts, y = P["t"], P["y"]
    pm, pp = np.nanmean(P["probs"], 0), np.nanmean(P["prev"], 0)
    F = bars(load5(Path("~/.minerva/data/btc/futures/klines_5m.csv").expanduser()), 0)
    fund = load_funding(Path("~/.minerva/data/btc/futures/funding.csv").expanduser())
    tf = F["t"]
    pos = np.searchsorted(ts, tf)
    hit = (pos < len(ts)) & (ts[np.minimum(pos, len(ts) - 1)] == tf)

    def align(arr):
        a = np.full(len(tf), np.nan)
        a[hit] = arr[pos[hit]]
        return a

    yf = align(y.astype(float))
    start = int(np.searchsorted(tf, E.ts(YEARS[0])))
    days = (tf[-1] - tf[start]) / 86_400_000
    rep: dict = {"embargo_bars": int(P["emb"]), "spot_end": int(ts[-1]), "fut_end": int(tf[-1])}
    from sklearn.metrics import roc_auc_score

    m = np.isfinite(pm) & (y >= 0)
    rep["auc"] = {str(Y): round(float(roc_auc_score(y[m & (ts >= E.ts(Y)) & (ts < E.ts(Y + 1))],
                                                    pm[m & (ts >= E.ts(Y)) & (ts < E.ts(Y + 1))])), 4)
                  for Y in YEARS if (m & (ts >= E.ts(Y))).any()}
    trades = {}
    for name, qs in VARIANTS.items():
        th = thresholds(ts, pm, pp, qs)
        yr = np.array([int(np.datetime64(int(x), "ms").astype("datetime64[Y]").astype(int) + 1970) for x in tf])
        hi = np.array([th.get(v, (np.nan, np.nan))[0] for v in yr])
        lo = np.array([th.get(v, (np.nan, np.nan))[1] for v in yr])
        pf = align(pm)
        tr = simulate(F, fund, pf, hi, lo, start)
        r = report(tr, days)
        r["thresholds"] = {str(k): [round(a, 4), round(b, 4)] for k, (a, b) in th.items()}
        # 닮은 정도: 봉마다 보유 방향
        hold = np.zeros(len(tf))
        for z in tr:
            hold[z[0] + 1:z[3]] = z[1]
        mm = np.isfinite(yf) & (yf >= 0) & (np.arange(len(tf)) >= start)
        r["resemble"] = {"up_long": round(float((hold[mm & (yf == 1)] > 0).mean() * 100), 1),
                         "down_short": round(float((hold[mm & (yf == 0)] < 0).mean() * 100), 1),
                         "agree": round(float(((hold[mm] > 0) & (yf[mm] == 1) | (hold[mm] < 0) & (yf[mm] == 0)).mean() * 100), 1),
                         "flat": round(float((hold[mm] == 0).mean() * 100), 1)}
        seeds = []
        for s in range(SEEDS):
            ths = thresholds(ts, P["probs"][s], P["prev"][s], qs)
            his = np.array([ths.get(v, (np.nan, np.nan))[0] for v in yr])
            los = np.array([ths.get(v, (np.nan, np.nan))[1] for v in yr])
            trs = simulate(F, fund, align(P["probs"][s]), his, los, start)
            seeds.append(block([z for z in trs if z[4] >= ystart(2024)]).get("ev"))
        r["seeds_new_ev"] = seeds
        if circ and tr:
            rng = np.random.default_rng(7)
            actual = r["all"]["ev"]
            span = np.flatnonzero(np.isfinite(pf))
            null = []
            for _ in range(circ):
                k = int(rng.integers(2000, len(span) - 2000))
                pr = pf.copy()
                pr[span] = np.roll(pf[span], k)
                null.append(block(simulate(F, fund, pr, hi, lo, start)).get("ev", 0.0))
            null = np.array(null)
            r["circ_null_ev"] = round(float(null.mean()), 4)
            r["circ_p"] = round(float((null >= actual).mean()), 4)
        crit = [
            all(r[k].get("ev", -1) > 0 and r[k].get("ret3", -1) > 0 for k in ("y2023", "new")),
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
        predict(out)
    if a.stage in ("trade", "all"):
        trade_stage(out, a.circ)


if __name__ == "__main__":
    main()
