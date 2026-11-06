"""강제청산 연쇄 뒤 되돌림. 등록: 2026-11-07-btc-fut15-liquidation-reversal-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_fut_liq_rev.py \
    --out <폴더> [--offset 5] [--perm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import btc_flow_feats as FF
import btc_scr_learned as L
import numpy as np
from btc_fractal import load5
from btc_scr_entry_stop import summ, walk, with_funding
from btc_scr_gate15 import load_funding

DAY = 86_400_000
WIN = 90
L.H = 32


def daily_q(x: np.ndarray, t: np.ndarray, q: float) -> np.ndarray:
    """봉마다: 그 봉이 속한 UTC 날 0시 전 90일 값의 q 백분위."""
    day = (t // DAY).astype(np.int64)
    out = np.full(len(x), np.nan)
    for dd in np.unique(day):
        m = (day >= dd - WIN) & (day < dd) & ~np.isnan(x)
        if m.sum() < WIN * 96 * 0.8:
            continue
        out[day == dd] = np.percentile(x[m], q)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--perm", action="store_true")
    ap.add_argument("--follow", action="store_true", help="106번: 방향을 반대로(따라가기)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d = L.bars(load5(base / "futures/klines_5m.csv"), args.offset)
    fund = load_funding(base / "futures/funding.csv")
    t, c = d["t"], d["c"]
    n = len(t)
    r1 = np.r_[np.full(4, np.nan), c[4:] / c[:-4] - 1]
    oi = FF.oi_feats(base / "futures/metrics.csv", d)[:, 0]
    r1[np.isnan(oi)] = np.nan  # 두 변형이 같은 기간을 쓰게
    lo, hi, oil = daily_q(r1, t, 1), daily_q(r1, t, 99), daily_q(oi, t, 10)
    with np.errstate(invalid="ignore"):
        oidrop = oi <= oil
        raw = {"RV": (r1 <= lo, r1 >= hi), "LQ": ((r1 <= lo) & oidrop, (r1 >= hi) & oidrop)}
    rep: dict = {"offset": args.offset, "follow": args.follow}
    years = range(2022, 2027)
    for name, (lv, sv) in raw.items():
        on_l = lv & ~np.r_[False, lv[:-1]]
        on_s = sv & ~np.r_[False, sv[:-1]]
        cands = []
        sgn = -1 if args.follow else 1
        for side, on in ((sgn, on_l), (-sgn, on_s)):
            for i in np.flatnonzero(on):
                res = L.trade(d, int(i), side)
                if res is None:
                    continue
                g, x, stopped = res
                g = with_funding(g, d, int(i), x, side, fund)
                cands.append((int(i), side, float(g), int(x), float(t[i]), stopped, 0.01))
        cands.sort()
        tr = walk(cands)
        r = {"all": summ(tr), "all_c10": summ(tr, 0.001),
             "old": summ([z for z in tr if z[4] < L.ystart(2024)]), "new": summ([z for z in tr if z[4] >= L.ystart(2024)]),
             "long": summ([z for z in tr if z[1] > 0]), "short": summ([z for z in tr if z[1] < 0]),
             "years": {y: summ([z for z in tr if L.ystart(y) <= z[4] < L.ystart(y + 1)]) for y in years},
             "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in tr]}
        if args.perm and tr:
            rng = np.random.default_rng(7)
            actual = float(np.mean([z[2] for z in tr]))
            valid = ~np.isnan(lo) & ~np.isnan(oil)
            cache: dict = {}
            plan = []
            for y in years:
                zs = [z for z in tr if L.ystart(y) <= z[4] < L.ystart(y + 1)]
                pool = np.flatnonzero((t >= L.ystart(y)) & (t < L.ystart(y + 1)) & valid)
                pool = pool[pool < n - L.H - 2]
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
                        q = L.trade(d, i, sd)
                        cache[(i, sd)] = None if q is None else (with_funding(q[0], d, i, q[1], sd, fund), q[1])
                    oc = cache[(i, sd)]
                    if oc is None:
                        continue
                    gs.append(oc[0])
                    busy = oc[1]
                if gs and np.mean(gs) >= actual:
                    hits += 1
            r["perm_p"] = round(hits / 2000, 4)
        rep[name] = r
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
