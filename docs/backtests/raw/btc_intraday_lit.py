"""논문에서 가져온 BTC 단타 규칙 H1~H3. 등록: 2026-11-06-btc-intraday-literature-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_intraday_lit.py \
    --out <폴더> --hyp H1|H2|H3 [--shift <시간>] [--prices spot|futures] [--perm]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import load5
from btc_scr_entry_stop import summ
from btc_scr_gate15 import load_funding

M5 = 300_000
HR = 3_600_000
DAY = 86_400_000
STOP = 0.01


def ts(y: int) -> float:
    return datetime(y, 1, 1, tzinfo=timezone.utc).timestamp() * 1000


def run_trade(d: dict, idx: dict, t_in: float, t_out: float, side: int, fund):
    """t_in 시가 진입, 5분봉으로 1% 손절, t_out 시가 청산. (총손익, 청산 시각, 손절 여부)"""
    a, b = idx.get(t_in), idx.get(t_out)
    if a is None or b is None or b <= a:
        return None
    o, h, l = d["o"], d["h"], d["l"]
    px = o[a]
    stop = px * (1 - STOP * side)
    seg = slice(a, b)
    hit = np.flatnonzero(l[seg] <= stop) if side > 0 else np.flatnonzero(h[seg] >= stop)
    if len(hit):
        j = a + int(hit[0])
        fill = min(o[j], stop) if side > 0 else max(o[j], stop)
        g, tx, st = (fill / px - 1) * side, d["t"][j], True
    else:
        g, tx, st = (o[b] / px - 1) * side, t_out, False
    if fund is not None:
        m = (fund[0] > t_in) & (fund[0] <= tx)
        g -= side * float(fund[1][m].sum())
    return float(g), float(tx), st


def plan(hyp: str, d: dict, shift: float, fund) -> list:
    """(진입 시각, 청산 시각, 방향) 목록. shift는 시간 단위."""
    sh = shift * HR
    days = np.unique((d["t"] // DAY) * DAY)
    out = []
    if hyp == "H1":
        idx = {v: k for k, v in enumerate(d["t"])}
        for d0 in days:
            t0 = d0 + sh
            a, b = idx.get(t0), idx.get(t0 + 30 * 60_000)
            if a is None or b is None:
                continue
            r = d["o"][b] / d["o"][a] - 1
            if r != 0:
                out.append((t0 + 23.5 * HR, t0 + DAY, 1 if r > 0 else -1))
    elif hyp == "H2":
        out = [(d0 + 22 * HR + sh, d0 + DAY + sh, 1) for d0 in days]
    elif hyp == "H3":
        ft, fr = fund
        for k in range(len(ft)):
            past = fr[(ft < ft[k]) & (ft >= ft[k] - 365 * DAY)]
            if ft[k] < ts(2021) or len(past) < 900:
                continue
            side = -1 if fr[k] >= np.percentile(past, 95) else 1 if fr[k] <= np.percentile(past, 5) else 0
            if side:
                t_in = (ft[k] // M5 + 1) * M5 + sh
                out.append((t_in, t_in + DAY, side))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--prices", default="spot")
    ap.add_argument("--perm", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d = load5(base / ("futures/klines_5m.csv" if args.prices == "futures" else "5m.csv"))
    fund_all = load_funding(base / "futures/funding.csv")
    fund = fund_all if args.prices == "futures" else None
    idx = {v: k for k, v in enumerate(d["t"])}
    pl = plan(args.hyp, d, args.shift, fund_all)
    busy, trs = -1.0, []
    for t_in, t_out, side in sorted(pl):
        if t_in < busy:
            continue
        r = run_trade(d, idx, t_in, t_out, side, fund)
        if r is None:
            continue
        g, tx, st = r
        trs.append((t_in, side, g, tx, t_in, st, STOP))
        busy = tx
    split = ts(2024)
    rep = {"hyp": args.hyp, "shift": args.shift, "prices": args.prices,
           "all": summ(trs), "all_c10": summ(trs, 0.001),
           "old": summ([z for z in trs if z[4] < split]), "new": summ([z for z in trs if z[4] >= split]),
           "trades": [[z[0], z[1], round(z[2], 6), z[3], z[5]] for z in trs]}
    if args.perm and trs:
        rng = np.random.default_rng(7)
        hold = np.array([t_out - t_in for t_in, t_out, _ in pl]).mean()
        lo, hi = trs[0][4], trs[-1][4]
        starts = d["t"][(d["t"] >= lo) & (d["t"] <= hi)]
        nl = sum(1 for z in trs if z[1] > 0)
        actual = float(np.mean([z[2] for z in trs]))
        hits = 0
        for _ in range(2000):
            picks = sorted([(float(t), 1) for t in rng.choice(starts, nl, replace=False)]
                           + [(float(t), -1) for t in rng.choice(starts, len(trs) - nl, replace=False)])
            busy, gs = -1.0, []
            for t_in, side in picks:
                if t_in < busy:
                    continue
                r = run_trade(d, idx, t_in, t_in + hold, side, fund)
                if r is None:
                    continue
                gs.append(r[0])
                busy = r[1]
            if gs and np.mean(gs) >= actual:
                hits += 1
        rep["perm_p"] = round(hits / 2000, 4)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
