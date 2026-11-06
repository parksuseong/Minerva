"""15분봉 SCR 3단 + 일봉·4시간봉 SCR 방향 문. 등록: 2026-11-06-btc-scr15-gate-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_gate15.py \
    --out <폴더> [--offset 5] [--futures]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import btc_scr_model as M
from btc_fractal import MIN5, cci, load5, resample, rsi, scr3
from btc_swing_patterns import SPLIT

GATES = ("none", "D", "H4", "both", "not_against")
BAR = 3 * MIN5  # 15분


def higher_dir(d5: dict, factor: int, t_sig: np.ndarray) -> np.ndarray:
    """신호 봉(15분) 마감 때 이미 마감된 상위봉 방향. btc_scr_model.higher_dir의 +30분을 +15분으로."""
    h = resample(d5, factor)
    cc, rr = cci(h["h"], h["l"], h["c"]), rsi(h["c"])
    dr = np.where((cc > 0) & (rr > 50), 1, np.where((cc < 0) & (rr < 50), -1, 0))
    close_t = h["t"] + factor * MIN5
    j = np.searchsorted(close_t, t_sig + BAR, side="right") - 1
    return np.where(j >= 0, dr[np.maximum(j, 0)], 0)


def load_funding(path: Path) -> tuple:
    t, r = [], []
    with path.open() as f:
        rd = csv.reader(f)
        next(rd)
        for row in rd:
            t.append(float(row[0]))
            r.append(float(row[2]))
    return np.array(t), np.array(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--futures", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(args.data)
    d5 = load5(base / ("futures/klines_5m.csv" if args.futures else "5m.csv"))
    fund = load_funding(base / "futures/funding.csv") if args.futures else None
    sh = dict(d5)
    sh["t"] = d5["t"] - args.offset * 60_000
    d = resample(sh, 3)
    d["t"] = d["t"] + args.offset * 60_000
    n = len(d["c"])
    on_l, on_s, *_ = scr3(d)
    h4 = higher_dir(d5, 48, d["t"])
    d1 = higher_dir(d5, 288, d["t"])
    sigs = []
    for side, on, opp in ((1, on_l, on_s), (-1, on_s, on_l)):
        for i in np.flatnonzero(on):
            if i < 6000 or i >= n - 2:
                continue
            res = M.label(d, i, side, opp)
            if res is None:
                continue
            g, x = res
            if fund is not None:  # 진입 봉 시가 ~ 청산 봉 시가 사이 펀딩
                t0, t1 = d["t"][i + 1], d["t"][x]
                m = (fund[0] > t0) & (fund[0] <= t1)
                g -= side * float(fund[1][m].sum())
            sigs.append((int(i), side, float(g), int(x), int(d1[i] * side), int(h4[i] * side)))
    sigs.sort()
    end = d["t"][-1] + BAR
    cut = {"1y": end - 365 * 86_400_000, "6m": end - 183 * 86_400_000, "30d": end - 30 * 86_400_000}

    def ok(gate, dd, hh):
        return {"none": True, "D": dd > 0, "H4": hh > 0, "both": dd > 0 and hh > 0,
                "not_against": dd >= 0 and hh >= 0}[gate]

    def summ(ts, cost=0.0005):
        if not ts:
            return {"n": 0}
        net = np.array([x[2] for x in ts]) - 2 * cost
        eq = np.cumprod(1 + net)
        peak = np.maximum.accumulate(np.r_[1.0, eq])
        eq3 = np.cumprod(1 + np.maximum(3 * net, -1))
        peak3 = np.maximum.accumulate(np.r_[1.0, eq3])
        s = np.sort(net)
        return {"n": len(ts), "win": round(float((net > 0.001).mean() * 100), 1),
                "ev": round(float(net.mean() * 100), 3), "ret": round(float((eq[-1] - 1) * 100), 1),
                "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
                "ret3": round(float((eq3[-1] - 1) * 100), 1),
                "mdd3": round(float((eq3 / peak3[1:] - 1).min() * 100), 1),
                "drop5": round(float(s[:-5].mean() * 100), 3) if len(s) > 5 else None,
                "long": int(sum(1 for x in ts if x[1] > 0))}

    rep: dict = {"offset": args.offset, "futures": args.futures}
    for gate in GATES:
        busy, tr = -1, []
        for i, side, g, x, dd, hh in sigs:
            if i <= busy or not ok(gate, dd, hh):
                continue
            tr.append((i, side, g, x, float(d["t"][i])))
            busy = x
        rep[gate] = {"all": summ(tr), "all_c10": summ(tr, 0.001),
                     "old": summ([z for z in tr if z[4] < SPLIT]),
                     "new": summ([z for z in tr if z[4] >= SPLIT])}
        for k, c in cut.items():
            rep[gate][k] = summ([z for z in tr if z[4] >= c])
        rep[gate]["trades"] = [[z[0], z[1], round(z[2], 5), z[4], z[3]] for z in tr]
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    for gate in GATES:
        r = rep[gate]
        print(gate, r["all"], "c10", r["all_c10"]["ev"], "old", r["old"].get("n"), r["old"].get("ev"),
              "new", r["new"].get("n"), r["new"].get("ev"), "1y", r["1y"], flush=True)


if __name__ == "__main__":
    main()
