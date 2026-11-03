"""고점 근처·저점 근처 구역 판별. 등록: 2026-11-03-btc-zone-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_zone.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import win_view
from btc_scr_litfilter import bars30, load5x, load_fut, roll_mean
from btc_swing_extreme import conditions

H = 48
UP, DOWN = 0.01, 0.02
TOPQ = 0.9
ts = lambda y: datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000  # noqa: E731
USED = [k for k in range(1, 29) if k not in (23, 24, 25)]


def zone(d: dict, side: int) -> np.ndarray:
    c, h, l = d["c"], d["h"], d["l"]  # noqa: E741
    n = len(c)
    out = np.full(n, np.nan)
    fh = np.max(win_view(h[1:], H), 1)
    fl = np.min(win_view(l[1:], H), 1)
    m = n - H
    cc = c[:m]
    if side < 0:
        out[:m] = (fh <= cc * (1 + UP)) & (fl <= cc * (1 - DOWN))
    else:
        out[:m] = (fl >= cc * (1 - UP)) & (fh >= cc * (1 + DOWN))
    return out


def fit(x: np.ndarray, y: np.ndarray, it: int = 3000, lr: float = 0.5, l2: float = 1e-3):
    mu, sd = x.mean(0), x.std(0)
    sd[sd == 0] = 1
    z = np.c_[np.ones(len(x)), (x - mu) / sd]
    w = np.zeros(z.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-z @ w))
        g = z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        w -= lr * g
    return lambda q: np.c_[np.ones(len(q)), (q - mu) / sd] @ w, w


def rate(y, m):
    return float(y[m].mean()) if m.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5x(root / "5m.csv")
    d = bars30(d5, a.offset)
    t = d["t"]
    train = (t >= ts(2022)) & (t < ts(2025))
    lr_ = np.r_[0.0, np.diff(np.log(d["c"]))]
    vol = np.log(np.sqrt(roll_mean(lr_ * lr_, 48 * 7)) + 1e-9)
    conds = conditions(d5, d, load_fut(root), train)
    yr = np.array([datetime.fromtimestamp(x / 1000, timezone.utc).year for x in t])
    rng = np.random.default_rng(5)
    rep: dict = {"offset": a.offset}
    for sd, name in ((-1, "top"), (1, "bottom")):
        y = zone(d, sd)
        X = np.stack([conds[sd][k] for k in USED], 1).astype(float)
        ok = np.isfinite(y) & np.isfinite(vol)
        tr = train & ok
        f, w = fit(X[tr], y[tr])
        fv, _ = fit(vol[tr, None], y[tr])
        s, sv = f(X), fv(vol[:, None])
        cut, cutv = np.quantile(s[tr], TOPQ), np.quantile(sv[tr], TOPQ)
        r = {"coef": {k: round(float(c), 3) for k, c in zip(USED, w[1:])}}
        for per, m in (("train", tr), ("test", ok & (t >= ts(2025))),
                       ("2018~21", ok & (t >= ts(2018)) & (t < ts(2022)))):
            sig = m & (s >= cut)
            base, hit, hitv = rate(y, m), rate(y, sig), rate(y, m & (sv >= cutv))
            idx = np.flatnonzero(m)
            ys, ss = y[idx], s[idx] >= cut
            draws = np.array([ys[np.roll(ss, rng.integers(H * 10, len(idx) - H * 10))].mean()
                              for _ in range(1000)])
            years = {int(k): [int((sig & (yr == k)).sum()), round(rate(y, sig & (yr == k)) * 100, 1),
                              round(rate(y, m & (yr == k)) * 100, 1)] for k in sorted(set(yr[m]))}
            r[per] = {"n": int(sig.sum()), "share": round(float(sig.sum() / m.sum()) * 100, 1),
                      "base": round(base * 100, 1), "hit": round(hit * 100, 1),
                      "vol_only": round(hitv * 100, 1), "p": float((draws >= hit).mean()),
                      "years": years}
            print(name, per, r[per], flush=True)
        r["score_last"] = [[float(t[i]), float(s[i] >= cut)] for i in range(len(t) - 48 * 30, len(t))]
        rep[name] = r
    (out / "result.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
