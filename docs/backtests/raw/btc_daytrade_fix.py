"""반전 캔들+거래량(D) 수정판. 등록: 2026-11-02-btc-daytrade-fix-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_daytrade_fix.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import btc_daytrade as T

COSTS = (0.0005, 0.001)
WIN = 0.001
HOLD = 48
VARIANTS = {"D0": (False, False), "D1": (True, False), "D2": (False, True), "D12": (True, True)}


def simulate(d: dict, rl: np.ndarray, rs: np.ndarray, a: int, b: int, use_stop: bool,
             use_hold: bool) -> list:
    """(진입 봉, 청산 봉, 방향, 총수익, 사유). 신호는 i봉 마감, 체결은 i+1봉 시가."""
    o, h, l, c = d["o"], d["h"], d["l"], d["c"]
    out = []
    pos = None  # (진입 봉, 방향, 진입가, 손절가)
    i = a
    while i < b - 1:
        if pos is None:
            if rl[i] or rs[i]:
                side = 1 if rl[i] else -1
                e = i + 1
                stop = l[i] if side > 0 else h[i]
                pos = (e, side, o[e], stop)
                i = e
                continue
            i += 1
            continue
        e, side, px, stop = pos
        # 이 봉(i) 안의 손절. 진입 봉부터 본다
        if use_stop and ((side > 0 and l[i] < stop) or (side < 0 and h[i] > stop)):
            fill = min(o[i], stop) if side > 0 else max(o[i], stop)
            out.append((e, i, side, (fill / px - 1) * side, "stop"))
            pos = None
            i += 1
            continue
        opp = rs[i] if side > 0 else rl[i]
        timeout = use_hold and (i + 1 - e) >= HOLD
        if opp or timeout:
            x = i + 1
            out.append((e, x, side, (o[x] / px - 1) * side, "opp" if opp else "time"))
            pos = None
            if opp:
                ns = -side
                pos = (x, ns, o[x], l[i] if ns > 0 else h[i])
            i = x
            continue
        i += 1
    if pos is not None:
        e, side, px, _ = pos
        out.append((e, b - 1, side, (c[b - 1] / px - 1) * side, "end"))
    return out


def stats(tr: list, lo: int, hi: int, cost: float) -> dict:
    rows = [t for t in tr if lo <= t[0] < hi]
    if not rows:
        return {"n": 0}
    net = np.array([t[3] for t in rows]) - 2 * cost
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])
    hold = np.array([t[1] - t[0] for t in rows]) / 2
    srt = np.sort(net)
    return {
        "n": len(rows),
        "win": round(float((net > WIN).mean() * 100), 1),
        "ret": round(float((eq[-1] - 1) * 100), 1),
        "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
        "worst": round(float(net.min() * 100), 2),
        "best": round(float(net.max() * 100), 2),
        "drop_top3": round(float((np.prod(1 + srt[:-3]) - 1) * 100), 1) if len(srt) > 3 else None,
        "hold_med_h": float(np.median(hold)),
        "hold_max_h": float(hold.max()),
        "long": sum(1 for t in rows if t[2] > 0),
        "reasons": {r: sum(1 for t in rows if t[4] == r) for r in ("opp", "stop", "time", "end")},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    rep: dict = {}
    for tag, d in (("base", T.load(data / "30m.csv")),
                   ("shift", T.load(data / "15m.csv", offset_min=15, merge=True))):
        sig = T.signals(d, T.h4_trend(d, data / "4h.csv"))["D"][0]
        n = len(d["c"])
        a6, a1 = n - T.WINDOWS["6m"], n - T.WINDOWS["30d"]
        for name, (st, hd) in VARIANTS.items():
            tr = simulate(d, sig[0], sig[1], a6, n, st, hd)
            r = rep.setdefault(name, {})
            if tag == "base":
                r["6m"] = stats(tr, a6, n, COSTS[0])
                r["6m_c10"] = stats(tr, a6, n, COSTS[1])
                r["before30"] = stats(tr, a6, a1, COSTS[0])
                r["30d"] = stats(tr, a1, n, COSTS[0])
            else:
                r["shift"] = stats(tr, a6, n, COSTS[0])
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    for k, v in rep.items():
        print(k, v["6m"]["n"], v["6m"]["win"], v["6m"]["ret"], flush=True)


if __name__ == "__main__":
    main()
