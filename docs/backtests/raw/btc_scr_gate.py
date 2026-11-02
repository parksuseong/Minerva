"""30분봉 SCR 3단 + 일봉·4시간봉 SCR 방향 문. 등록: 2026-11-03-btc-scr-gate-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_gate.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import btc_scr_model as M
from btc_fractal import load5, resample, scr3
from btc_swing_patterns import SPLIT

GATES = ("none", "D", "H4", "both", "not_against")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(args.data) / "5m.csv")
    sh = dict(d5)
    sh["t"] = d5["t"] - args.offset * 60_000
    d = resample(sh, 6)
    d["t"] = d["t"] + args.offset * 60_000
    n = len(d["c"])
    on_l, on_s, *_ = scr3(d)
    h4 = M.higher_dir(d5, 48, d["t"])
    d1 = M.higher_dir(d5, 288, d["t"])
    sigs = []
    for side, on, opp in ((1, on_l, on_s), (-1, on_s, on_l)):
        for i in np.flatnonzero(on):
            if i < 3000 or i >= n - 2:
                continue
            res = M.label(d, i, side, opp)
            if res is None:
                continue
            g, x = res
            sigs.append((int(i), side, float(g), int(x), int(d1[i] * side), int(h4[i] * side)))
    sigs.sort()
    end = d["t"][-1] + 1_800_000
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
        s = np.sort(net)
        return {"n": len(ts), "win": round(float((net > 0.001).mean() * 100), 1),
                "ev": round(float(net.mean() * 100), 3), "ret": round(float((eq[-1] - 1) * 100), 1),
                "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
                "drop5": round(float(s[:-5].mean() * 100), 3) if len(s) > 5 else None,
                "long": int(sum(1 for x in ts if x[1] > 0))}

    rep: dict = {"offset": args.offset}
    for gate in GATES:
        busy, tr = -1, []
        for i, side, g, x, dd, hh in sigs:
            if i <= busy or not ok(gate, dd, hh):
                continue
            tr.append((i, side, g, x, float(d["t"][i])))
            busy = x
        t = np.array([z[4] for z in tr])
        rep[gate] = {"all": summ(tr), "all_c10": summ(tr, 0.001),
                     "old": summ([z for z in tr if z[4] < SPLIT]),
                     "new": summ([z for z in tr if z[4] >= SPLIT])}
        for k, c in cut.items():
            rep[gate][k] = summ([z for z in tr if z[4] >= c])
        rep[gate]["years"] = {}
        for z in tr:
            y = int(1970 + z[4] // (365.2425 * 86_400_000))
            rep[gate]["years"].setdefault(y, []).append(z[2] - 0.001)
        rep[gate]["years"] = {str(y): [len(v), round(float((np.prod(1 + np.array(v)) - 1) * 100), 1)]
                              for y, v in sorted(rep[gate]["years"].items())}
        rep[gate]["trades"] = [[z[0], z[1], round(z[2], 5), z[4]] for z in tr]
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    for gate in GATES:
        r = rep[gate]
        print(gate, r["all"], "c10", r["all_c10"]["ev"], "old", r["old"].get("n"), r["old"].get("ev"),
              "new", r["new"].get("n"), r["new"].get("ev"), "1y", r["1y"].get("n"), r["1y"].get("ret"),
              "6m", r["6m"].get("n"), r["6m"].get("ret"), "30d", r["30d"].get("n"), r["30d"].get("ret"),
              flush=True)


if __name__ == "__main__":
    main()
