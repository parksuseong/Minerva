"""30분봉 SCR 3단 무장 + 짧은 봉 SCR 재확인 진입 + 빠른 청산. 등록: 2026-11-03-btc-scr-wait-fast-hypotheses.md

체결은 5분봉으로 계산한다. 각 봉 크기의 신호는 그 봉이 마감한 시각에 시작하는 5분봉부터 쓸 수 있다.
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_wait_fast.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from btc_fractal import load5, resample, scr3

COST = 0.0005
LIQ = 0.32
ARM = 288  # 5분봉 24시간
START = datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
DAY = 86_400_000
M5 = 300_000


def on_grid(d5: dict, factor: int, offset_min: int) -> tuple:
    """봉 크기 factor(5분 배수)의 SCR 3단 롱·숏 신호를 5분봉 격자(쓸 수 있는 첫 봉)에 옮긴다."""
    if factor == 1:
        f = d5
    else:
        sh = dict(d5)
        sh["t"] = d5["t"] - offset_min * 60_000
        f = resample(sh, factor)
        f["t"] = f["t"] + offset_min * 60_000
    on_l, on_s, *_ = scr3(f)
    close_t = f["t"] + factor * M5
    out = []
    for on in (on_l, on_s):
        g = np.zeros(len(d5["t"]), dtype=bool)
        j = np.searchsorted(d5["t"], close_t[on])
        g[j[j < len(g)]] = True
        out.append(g)
    return tuple(out)


def simulate(d5: dict, a30: tuple, trig: tuple, price_cond: bool, fast: bool, stop: float,
             j0: int) -> list:
    o, h, l, c = d5["o"], d5["h"], d5["l"], d5["c"]  # noqa: E741
    L30, S30 = a30
    TL, TS = trig
    n = len(o)
    tr = []
    side = arm_side = 0
    arm_j = arm_px = px = line = mae = 0.0
    ej = 0
    for j in range(j0, n - 1):
        if side:
            ex = None
            if (S30[j] if side > 0 else L30[j]):
                ex = o[j]
            elif fast and (TS[j] if side > 0 else TL[j]) and (c[j - 1] / px - 1) * side > 0.001:
                ex = o[j]
            if ex is None:
                hit = l[j] <= line if side > 0 else h[j] >= line
                if hit:
                    ex = min(o[j], line) if side > 0 else max(o[j], line)
            if ex is None:
                mae = max(mae, (px - l[j]) / px if side > 0 else (h[j] - px) / px)
                continue
            g = (ex / px - 1) * side
            tr.append((float(d5["t"][ej]), g, max(mae, -g), side, float(d5["t"][j]), float(px)))
            side = 0
            arm_side = 0
            continue
        if L30[j] or S30[j]:
            arm_side = 1 if L30[j] else -1
            arm_j, arm_px = j, c[j - 1]
        if arm_side and j - arm_j > ARM:
            arm_side = 0
        if arm_side and j > arm_j and (TL[j] if arm_side > 0 else TS[j]):
            if price_cond and (o[j] - arm_px) * arm_side >= 0:
                continue
            side, px, ej, mae = arm_side, o[j], j, 0.0
            line = px * (1 - stop * side)
            arm_side = 0
            hit = l[j] <= line if side > 0 else h[j] >= line
            if hit:
                g = (line / px - 1) * side
                tr.append((float(d5["t"][ej]), g, -g, side, float(d5["t"][j]), float(px)))
                side = 0
            else:
                mae = (px - l[j]) / px if side > 0 else (h[j] - px) / px
    return tr


def summ(ts: list) -> dict:
    if not ts:
        return {"n": 0}
    g = np.array([x[1] for x in ts])
    mae = np.array([x[2] for x in ts])
    net1 = g - 2 * COST
    net3 = np.where(mae >= LIQ, -1.0, 3 * g - 6 * COST)
    out = {"n": len(ts), "win": round(float((net1 > 0.001).mean() * 100), 1),
           "ev": round(float(net1.mean() * 100), 3), "worst": round(float(net1.min() * 100), 2),
           "long": int(sum(1 for x in ts if x[3] > 0))}
    for k, net in (("x1", net1), ("x3", net3)):
        eq = np.cumprod(1 + np.maximum(net, -1.0))
        peak = np.maximum.accumulate(np.r_[1.0, eq])
        out[k] = round(float((eq[-1] - 1) * 100), 1)
        out[k + "_mdd"] = round(float((eq / peak[1:] - 1).min() * 100), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(a.data) / "5m.csv")
    a30 = on_grid(d5, 6, a.offset)
    trig = {"5m": on_grid(d5, 1, 0), "15m": on_grid(d5, 3, 0)}
    j0 = int(np.searchsorted(d5["t"], START))
    end = d5["t"][-1] + M5
    cuts = {"30d": end - 30 * DAY, "6m": end - 183 * DAY, "1y": end - 365 * DAY, "2022~": START}
    rep: dict = {"offset": a.offset}
    for tf, pc, fa, st in itertools.product(("5m", "15m"), (False, True), (False, True), (0.01, 0.03)):
        key = f"T{tf}|P{int(pc)}|F{int(fa)}|SL{st * 100:g}%"
        tr = simulate(d5, a30, trig[tf], pc, fa, st, j0)
        r = {k: summ([z for z in tr if z[0] >= c]) for k, c in cuts.items()}
        r["trades"] = [[z[0], round(z[1], 5), z[3], z[4], z[5]] for z in tr if z[0] >= cuts["6m"]]
        rep[key] = r
        print(key, {k: (r[k].get("n"), r[k].get("win"), r[k].get("x1")) for k in cuts}, flush=True)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
