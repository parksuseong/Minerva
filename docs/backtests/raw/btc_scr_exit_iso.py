"""SCR 3단 청산 규칙만 바꿔 비교: 하루 상한 있음/없음 × 반대 신호 뒤집기 있음/없음.

등록: 2026-11-03-btc-scr-exit-iso.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_exit_iso.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from btc_fractal import load5, resample, scr3
from btc_swing_patterns import SPLIT

STOP = 0.01


def sim(d, on_l, on_s, start, cap, flip):
    o, h, l = d["o"], d["h"], d["l"]
    n = len(o)
    trades = []
    i = start
    pend = None
    while i < n - 2:
        if pend is None:
            if on_l[i] or on_s[i]:
                pend = (i, 1 if on_l[i] else -1)
            else:
                i += 1
                continue
        s_i, side = pend
        pend = None
        e = s_i + 1
        px = o[e]
        stop = px * (1 - STOP * side)
        opp = on_s if side > 0 else on_l
        end = min(e + cap, n - 1) if cap else n - 1
        exit_i, g, why = end, None, "end"
        for j in range(e, end):
            if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
                fill = min(o[j], stop) if side > 0 else max(o[j], stop)
                g, exit_i, why = (fill / px - 1) * side, j, "stop"
                break
            if opp[j]:
                g, exit_i, why = (o[j + 1] / px - 1) * side, j + 1, "opp"
                if flip:
                    pend = (j, -side)
                break
        if g is None:
            g, why = (o[end] / px - 1) * side, "cap" if cap else "end"
        trades.append((s_i, side, g, why, float(d["t"][s_i])))
        i = exit_i if why != "opp" else exit_i
        if why == "stop":
            i = exit_i + 1
    return trades


def summ(ts, cost=0.0005):
    if not ts:
        return {"n": 0}
    net = np.array([x[2] for x in ts]) - 2 * cost
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    return {"n": len(ts), "win": round(float((net > 0.001).mean() * 100), 1),
            "ev": round(float(net.mean() * 100), 3), "ret": round(float((eq[-1] - 1) * 100), 1),
            "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(args.data) / "5m.csv")
    rep = {}
    for off in (0, 15):
        sh = dict(d5)
        sh["t"] = d5["t"] - off * 60_000
        d = resample(sh, 6)
        d["t"] = d["t"] + off * 60_000
        on_l, on_s, *_ = scr3(d)
        end = d["t"][-1] + 1_800_000
        y1 = end - 365 * 86_400_000
        for cap in (0, 48):
            for flip in (False, True):
                tr = sim(d, on_l, on_s, 3000, cap, flip)
                key = f"off{off}_cap{cap}_flip{int(flip)}"
                rep[key] = {"all": summ(tr), "c10": summ(tr, 0.001),
                            "old": summ([x for x in tr if x[4] < SPLIT]),
                            "new": summ([x for x in tr if x[4] >= SPLIT]),
                            "1y": summ([x for x in tr if x[4] >= y1])}
                print(key, rep[key], flush=True)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
