"""117번: 일봉·4시간·15분 방향이 모두 −1인 날의 양방향 전일 범위 돌파.
등록: 2026-11-07-btc-align-down-breakout-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python3 docs/backtests/raw/btc_fut_align_breakout.py \
    --out <폴더> [--day-shift 8] [--perm 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import btc_fut_daily_regime as R
from btc_fractal import load5
from btc_scr_entry_stop import summ, with_funding
from btc_scr_gate15 import load_funding
from btc_scr_learned import bars, direction, ystart

H = 3_600_000
DAY = 86_400_000
STOP = 0.01


def day_trades(d: dict, i0: int, i1: int, ph: float, pl: float, fund) -> list:
    o, h, l, c, t = d["o"], d["h"], d["l"], d["c"], d["t"]
    out, used, j = [], set(), i0
    while j < i1 - 1:
        side = 1 if c[j] > ph and 1 not in used else (-1 if c[j] < pl and -1 not in used else 0)
        if not side:
            j += 1
            continue
        used.add(side)
        e = j + 1
        px = o[e]
        sp = px * (1 - STOP * side)
        x, g, stopped = i1, None, False
        for k in range(e, i1):
            if (side > 0 and l[k] <= sp) or (side < 0 and h[k] >= sp):
                fill = min(o[k], sp) if side > 0 else max(o[k], sp)
                x, g, stopped = k, (fill / px - 1) * side, True
                break
        if g is None:
            g = (o[i1] / px - 1) * side
        g = with_funding(g, d, j, x, side, fund)
        out.append((int(j), side, float(g), int(x), float(t[j]), stopped, STOP))
        if not stopped:
            break
        j = x + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--day-shift", type=int, default=0)
    ap.add_argument("--perm", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / "futures/klines_5m.csv")
    fund = load_funding(base / "futures/funding.csv")
    d = bars(d5, 0)
    t = d["t"]
    sh = args.day_shift
    dd, _ = R.regime(d5, d, sh)
    d4, _ = R.regime(d5, d, 0, 48)
    d15 = direction(d)
    key = (t - sh * H) // DAY
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    first = int(np.searchsorted(t, t[0] + 60 * DAY))
    days = []  # (신호 여부, 거래 목록)
    for a in range(1, len(starts) - 1):
        p0, i0, i1 = starts[a - 1], starts[a], starts[a + 1]
        if i0 < first or i1 - i0 != 96 or i0 - p0 != 96:
            continue
        sig = dd[i0 - 1] == -1 and d4[i0 - 1] == -1 and d15[i0 - 1] == -1
        ph, pl = float(d["h"][p0:i0].max()), float(d["l"][p0:i0].min())
        days.append((bool(sig), day_trades(d, i0, i1, ph, pl, fund)))
    sig_tr = [z for s, tr in days if s for z in tr]
    ctl_tr = [z for s, tr in days if not s for z in tr]
    g = sorted(z[2] for z in sig_tr)
    rep: dict = {
        "day_shift": sh, "signal_days": int(sum(s for s, _ in days)), "days": len(days),
        "signal": {"all": summ(sig_tr), "old": summ([z for z in sig_tr if z[4] < ystart(2024)]),
                   "new": summ([z for z in sig_tr if z[4] >= ystart(2024)]),
                   "long": summ([z for z in sig_tr if z[1] > 0]), "short": summ([z for z in sig_tr if z[1] < 0]),
                   "drop5": round((float(np.mean(g[:-5])) - 0.001) * 100, 3) if len(g) > 5 else None},
        "control": {"all": summ(ctl_tr), "long": summ([z for z in ctl_tr if z[1] > 0]),
                    "short": summ([z for z in ctl_tr if z[1] < 0])},
        "trades": [[z[0], z[1], round(z[2], 5), z[3], z[5]] for z in sig_tr],
    }
    if args.perm and sig_tr and ctl_tr:
        sums = np.array([sum(z[2] for z in tr) for _, tr in days])
        cnts = np.array([len(tr) for _, tr in days])
        flag = np.array([s for s, _ in days])
        diff = lambda m: sums[m].sum() / max(cnts[m].sum(), 1) - sums[~m].sum() / max(cnts[~m].sum(), 1)  # noqa: E731
        actual = diff(flag)
        rng = np.random.default_rng(7)
        hits = sum(1 for _ in range(args.perm) if diff(rng.permutation(flag)) >= actual)
        rep["diff_pct"] = round(float(actual) * 100, 3)
        rep["perm_p"] = round(hits / args.perm, 4)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
