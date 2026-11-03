"""넓은 장벽(±2.5%) + 모델 조합 + 시장가 체결. 등록: 2026-11-04-btc-ml-barrier-wide-hypotheses.md

실행: PYTHONPATH=<numpy·sklearn·xgboost·torch 경로>:src:docs/backtests/raw \
      python docs/backtests/raw/btc_ml_wide.py --out <폴더> [--stage predict|trade|all]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import btc_ml_15m as M
import btc_ml_barrier_freq as F
import btc_ml_ensemble as E
from btc_futures_compare import load_funding

BAR = 0.025
COST = 0.0007  # 테이커 0.05% + 밀림 0.02%


def stack_scores(probs, y, t):
    """연도 Y 점수 = Y−1년 예측으로 맞춘 로지스틱 조합. 2022는 단순 평균."""
    yr = np.array([datetime.fromtimestamp(x / 1000, timezone.utc).year for x in t])
    known = np.isfinite(probs).all(0)
    X = np.log(np.clip(probs, 1e-4, 1 - 1e-4) / (1 - np.clip(probs, 1e-4, 1 - 1e-4))).T
    S = np.full(len(t), np.nan)
    A = np.where(known, np.nanmean(np.where(known, probs, 0.5), 0), np.nan)
    for Y in range(2023, 2027):
        tr = known & (yr == Y - 1) & (y >= 0)
        te = known & (yr == Y)
        if not tr.any() or not te.any():
            continue
        Xt, yt = X[tr], y[tr].astype(float)
        w = np.zeros(X.shape[1] + 1)
        Z = np.c_[np.ones(len(Xt)), Xt]
        for _ in range(500):
            p = 1 / (1 + np.exp(-Z @ w))
            w -= 0.5 * (Z.T @ (p - yt) / len(yt) + 1e-3 * np.r_[0, w[1:]])
        S[te] = 1 / (1 + np.exp(-(np.c_[np.ones(te.sum()), X[te]] @ w)))
        print("weights", Y, np.round(w, 3), flush=True)
    return S, A


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage in ("predict", "all"):
        M.TP = M.SL = BAR
        M.EMBARGO = 1000
        M.predict(out)
    if a.stage not in ("trade", "all"):
        return
    z = np.load(out / "pred.npz")
    P = {k: z[k] for k in ("o", "h", "l", "c")}
    t, probs, y = z["t"], z["probs"], z["y"]
    n = len(t)
    F.TP = F.SL = BAR
    F.COST = COST
    ft, fr = load_funding()
    S, A = stack_scores(probs, y, t)
    i23, i24 = (int(np.searchsorted(t, E.ts(v))) for v in (2023, 2024))
    yt = (t[-1] - t[i24]) / F.YEAR
    known = np.isfinite(S)

    def run(score, q=0.90, sides=(1, -1), delay=0, cost=COST, a0=i24):
        F.COST = cost
        hi, lo = F.thresholds(score, t, q)
        with np.errstate(invalid="ignore"):
            li, si = np.isfinite(score) & (score >= hi), np.isfinite(score) & (score <= lo)
        if delay:
            li, si = np.r_[np.zeros(delay, bool), li[:-delay]], np.r_[np.zeros(delay, bool), si[:-delay]]
        r = F.summ(F.simulate(P, t, li, si, a0, n, sides), ft, fr, (t[-1] - t[a0]) / F.YEAR)
        F.COST = COST
        return r

    m24 = known & (y >= 0) & (np.arange(n) >= i24)
    for name, sc in (("S", S), ("A", A)):
        q = np.quantile(sc[m24], 0.9)
        print(name, "top10% hit", round(float(y[m24][sc[m24] >= q].mean() * 100), 1),
              "base", round(float(y[m24].mean() * 100), 1), "be",
              round((BAR + 2 * COST) / (2 * BAR) * 100, 1), flush=True)
    base = run(S)
    print("S 2023", run(S, a0=i23))
    print("S 2024~", base)
    print("A 2024~", run(A))
    lo_, sh_ = run(S, sides=(1,)), run(S, sides=(-1,))
    dl, fee = run(S, delay=1), run(S, cost=COST * 1.5)
    q85, q95 = run(S, q=0.85), run(S, q=0.95)
    for k, v in (("long", lo_), ("short", sh_), ("delay1", dl), ("fee1.5", fee), ("q85", q85),
                 ("q95", q95)):
        print(k, v, flush=True)
    rng = np.random.default_rng(141)
    seg = S[i24:].copy()
    rand = []
    for _ in range(500):
        k = int(rng.integers(96 * 30, len(seg) - 96 * 30))
        s2 = S.copy()
        s2[i24:] = np.roll(seg, k)
        rand.append(run(s2)["x3"])
    rand = np.array(rand)
    p = float((rand >= base["x3"]).mean())
    crit = [base["x3"] > 0 and base["ev3"] > 0, base.get("per_year", 0) >= 180, p < 0.01,
            lo_["ev3"] > 0 and sh_["ev3"] > 0, dl["x3"] > 0, fee["x3"] > 0,
            q85["x3"] > 0 and q95["x3"] > 0]
    print("p", p, "rand median", float(np.median(rand)), "crit", crit, flush=True)
    _ = yt


if __name__ == "__main__":
    main()
