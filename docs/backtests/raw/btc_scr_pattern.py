"""SCR 3단을 뼈대로 패턴을 겹친다. 등록: 2026-11-02-btc-scr-pattern-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_pattern.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import btc_book2 as K
import btc_pattern_census as P
import btc_swing_patterns as S
from btc_fractal import load5, resample, scr3

SIZES = (0.02, 0.04)
BREAK = ("DB", "DBP", "TB", "IHS", "ATRI")
WIN6H = 12
STOP = 0.01
CAP = 48


def scr_trade(d: dict, i: int, side: int, opp: np.ndarray) -> tuple:
    """i봉 마감 신호, i+1 시가 진입. 손절 1%, 반대 SCR 원신호 다음 시가, 48봉 상한."""
    o, h, l = d["o"], d["h"], d["l"]
    n = len(o)
    e = i + 1
    px = o[e]
    stop = px * (1 - STOP * side)
    end = min(e + CAP, n - 1)
    for j in range(e, end):
        if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
            fill = min(o[j], stop) if side > 0 else max(o[j], stop)
            return (fill / px - 1) * side, j
        if opp[j] and j + 1 < n:
            return (o[j + 1] / px - 1) * side, j + 1
    return (o[end] / px - 1) * side, end


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
    S.CAP = K.CAP = 48
    n = len(d["c"])
    on_l, on_s, *_ = scr3(d)
    # 패턴 완성 봉(신호 봉 = 진입 봉 − 1) 표시
    pat = {1: np.zeros(n, bool), -1: np.zeros(n, bool)}
    rows = []
    for th in SIZES:
        for side in (1, -1):
            for name, at, g, why, rr in P.rows_for(d, side, th):
                if 1 <= at < n:
                    # 패턴을 안 시점: 돌파형은 신호 봉(at−1) 마감, 지정가형은 체결 봉(at) 마감
                    known = at - 1 if name in BREAK else at
                    pat[side][known] = True
                    # B의 판단 봉: 진입 전에 마감된 마지막 봉(at−1)
                    rows.append((at - 1, side, g, name))
    near_pat = {}
    near_scr = {}
    for side, on in ((1, on_l), (-1, on_s)):
        cp = np.cumsum(np.r_[0, pat[side].astype(int)])
        cs = np.cumsum(np.r_[0, on.astype(int)])
        idx = np.arange(n)
        lo = np.maximum(idx - WIN6H, 0)
        near_pat[side] = (cp[idx + 1] - cp[lo]) > 0  # [i−12, i]에 같은 방향 패턴 완성
        near_scr[side] = (cs[idx + 1] - cs[lo]) > 0  # [i−12, i]에 같은 방향 SCR 3단

    def run_scr(require_pat: bool) -> list:
        sigs = []
        for side, on, opp in ((1, on_l, on_s), (-1, on_s, on_l)):
            for i in np.flatnonzero(on):
                if i < 200 or i >= n - 2:
                    continue
                if require_pat and not near_pat[side][i]:
                    continue
                sigs.append((int(i), side))
        sigs.sort()
        trades, busy = [], -1
        for i, side in sigs:
            if i <= busy:
                continue
            g, x = scr_trade(d, i, side, on_s if side > 0 else on_l)
            trades.append({"i": i, "side": side, "g": float(g), "t": float(d["t"][i])})
            busy = x
        return trades

    def run_pat() -> list:
        sel = sorted((i, side, g, name) for i, side, g, name in rows
                     if i >= 200 and near_scr[side][i])
        trades, busy = [], -1
        for i, side, g, name in sel:
            if i <= busy:
                continue
            trades.append({"i": int(i), "side": side, "g": float(g), "t": float(d["t"][i]),
                           "pat": name})
            busy = i + CAP
        return trades

    end = d["t"][-1] + 1_800_000
    t6, t1 = end - 183 * 86_400_000, end - 30 * 86_400_000

    def summ(ts: list, cost: float = 0.0005) -> dict:
        if not ts:
            return {"n": 0}
        net = np.array([x["g"] for x in ts]) - 2 * cost
        eq = np.cumprod(1 + net)
        peak = np.maximum.accumulate(np.r_[1.0, eq])
        s = np.sort(net)
        return {"n": len(ts), "win": round(float((net > S.WIN).mean() * 100), 1),
                "ev": round(float(net.mean() * 100), 3), "ret": round(float((eq[-1] - 1) * 100), 1),
                "mdd": round(float((eq / peak[1:] - 1).min() * 100), 1),
                "drop5": round(float(s[:-5].mean() * 100), 3) if len(s) > 5 else None,
                "long": int(sum(x["side"] > 0 for x in ts))}

    rep: dict = {"offset": args.offset}
    for name, tr in (("S0_scr", run_scr(False)), ("A_scr_pat", run_scr(True)), ("B_pat_scr", run_pat())):
        rep[name] = {
            "all": summ(tr), "all_c10": summ(tr, 0.001),
            "old": summ([x for x in tr if x["t"] < S.SPLIT]),
            "new": summ([x for x in tr if x["t"] >= S.SPLIT]),
            "6m": summ([x for x in tr if x["t"] >= t6]),
            "30d": summ([x for x in tr if x["t"] >= t1]),
            "trades": tr,
        }
        print(name, {k: v for k, v in rep[name].items() if k != "trades"}, flush=True)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
