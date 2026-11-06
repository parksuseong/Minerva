"""15분봉 SCR 연속값 × 봉 4개 학습 가중치(능형 회귀), 해마다 다시 학습.

등록: 2026-11-06-btc-scr15-learned-weights-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_learned.py \
    --out <폴더> [--offset 5] [--futures] [--perm]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import cci, load5, resample, rsi, stoch
from btc_scr_entry_stop import summ, walk, with_funding
from btc_scr_gate15 import load_funding

H = 16
STOP = 0.01
Q = 95
YEARS = range(2020, 2027)
UPPER = (12, 48, 288)  # 1시간, 4시간, 일봉(5분봉 개수)
M5 = 300_000


def ystart(y: int) -> float:
    return datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


def six(d: dict) -> np.ndarray:
    s = stoch(d["h"], d["l"], d["c"]) / 100 - 0.5
    c = np.clip(cci(d["h"], d["l"], d["c"]) / 200, -2, 2)
    r = (rsi(d["c"]) - 50) / 50
    v = np.c_[s, c, r]
    return np.c_[v, np.r_[np.full((1, 3), np.nan), np.diff(v, axis=0)]]


def features(d5: dict, d: dict, bar_ms: int) -> np.ndarray:
    """15분봉 d의 각 봉 마감 시각에 알 수 있는 24개 특징."""
    cols = [six(d)]
    close15 = d["t"] + bar_ms
    for k in UPPER:
        u = resample(d5, k)
        f = six(u)
        uc = u["t"] + k * M5
        j = np.searchsorted(uc, close15, side="right") - 1
        out = np.full((len(close15), 6), np.nan)
        ok = j >= 0
        out[ok] = f[j[ok]]
        cols.append(out)
    return np.hstack(cols)


def target(d: dict) -> np.ndarray:
    o = d["o"]
    n = len(o)
    y = np.full(n, np.nan)
    y[: n - 1 - H] = np.log(o[1 + H:] / o[1:n - H])
    return np.clip(y, -0.05, 0.05)


def ridge(X: np.ndarray, y: np.ndarray):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    lam = 0.01 * len(y)
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return mu, sd, w, y.mean()


def predict(m, X: np.ndarray) -> np.ndarray:
    mu, sd, w, b = m
    return ((X - mu) / sd) @ w + b


def trade(d: dict, i: int, side: int):
    """다음 시가 진입, 1% 손절(진입 봉부터, 갭은 시가) 또는 16봉 뒤 시가."""
    o, h, l = d["o"], d["h"], d["l"]
    e = i + 1
    x = e + H
    if x >= len(o):
        return None
    px = o[e]
    stop = px * (1 - STOP * side)
    for j in range(e, x):
        if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
            fill = min(o[j], stop) if side > 0 else max(o[j], stop)
            return (fill / px - 1) * side, j, True
    return (o[x] / px - 1) * side, x, False


def bars(d5: dict, offset: int) -> dict:
    sh = dict(d5)
    sh["t"] = d5["t"] - offset * 60_000
    d = resample(sh, 3)
    d["t"] = d["t"] + offset * 60_000
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--futures", action="store_true")
    ap.add_argument("--perm", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    s5 = load5(base / "5m.csv")
    d = bars(s5, args.offset)
    X = features(s5, d, 900_000)
    y = target(d)
    t = d["t"]
    pred = np.full(len(t), np.nan)
    qs, weights = {}, {}
    for yr in YEARS:
        a, b = ystart(yr), ystart(yr + 1)
        tr = np.flatnonzero((t < a - H * 900_000) & ~np.isnan(X).any(1) & ~np.isnan(y))[::4]
        te = np.flatnonzero((t >= a) & (t < b) & ~np.isnan(X).any(1))
        m = ridge(X[tr], y[tr])
        pred[te] = predict(m, X[te])
        qs[yr] = float(np.percentile(np.abs(predict(m, X[tr])), Q))
        weights[yr] = [round(float(v), 6) for v in m[2]]
    # 매매할 가격: 현물 또는 선물(같은 시각의 예측을 씀)
    fund = None
    if args.futures:
        f5 = load5(base / "futures/klines_5m.csv")
        fund = load_funding(base / "futures/funding.csv")
        dt = bars(f5, args.offset)
        pos = {v: k for k, v in enumerate(t)}
        pr = np.array([pred[pos[v]] if v in pos else np.nan for v in dt["t"]])
    else:
        dt, pr = d, pred
    tt = dt["t"]
    yr_of = np.array([datetime.fromtimestamp(v / 1000, timezone.utc).year for v in tt])
    q_of = np.array([qs.get(v, np.inf) for v in yr_of])
    side = np.where(pr > q_of, 1, np.where(pr < -q_of, -1, 0))
    cands = []
    for i in np.flatnonzero(side != 0):
        res = trade(dt, int(i), int(side[i]))
        if res is None:
            continue
        g, x, stopped = res
        g = with_funding(g, dt, int(i), x, int(side[i]), fund)
        cands.append((int(i), int(side[i]), float(g), int(x), float(tt[i]), stopped, STOP))
    trs = walk(cands)
    rep: dict = {"offset": args.offset, "futures": args.futures, "q": qs, "weights": weights,
                 "all": summ(trs), "all_c10": summ(trs, 0.001),
                 "years": {yr: summ([z for z in trs if ystart(yr) <= z[4] < ystart(yr + 1)]) for yr in YEARS},
                 "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in trs]}
    if args.perm and trs:
        rng = np.random.default_rng(7)
        actual = float(np.mean([z[2] for z in trs]))
        cache: dict = {}
        plan = []
        for yr in YEARS:
            zs = [z for z in trs if ystart(yr) <= z[4] < ystart(yr + 1)]
            pool = np.flatnonzero((tt >= ystart(yr)) & (tt < ystart(yr + 1)))[:-H - 2]
            plan.append((pool, sum(1 for z in zs if z[1] > 0), sum(1 for z in zs if z[1] < 0)))
        hits = 0
        for _ in range(2000):
            picks = []
            for pool, nl, ns in plan:
                if nl:
                    picks += [(int(k), 1) for k in rng.choice(pool, nl, replace=False)]
                if ns:
                    picks += [(int(k), -1) for k in rng.choice(pool, ns, replace=False)]
            picks.sort()
            busy, gs = -1, []
            for i, sd in picks:
                if i <= busy:
                    continue
                if (i, sd) not in cache:
                    r = trade(dt, i, sd)
                    cache[(i, sd)] = None if r is None else (with_funding(r[0], dt, i, r[1], sd, fund), r[1])
                oc = cache[(i, sd)]
                if oc is None:
                    continue
                gs.append(oc[0])
                busy = oc[1]
            if gs and np.mean(gs) >= actual:
                hits += 1
        rep["perm_p"] = round(hits / 2000, 4)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
