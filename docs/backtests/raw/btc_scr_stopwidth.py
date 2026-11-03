"""30분봉 SCR 3단, 손절 폭 시험. 등록: 2026-11-03-btc-scr-stopwidth-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_stopwidth.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import btc_scr_model as M
from btc_fractal import load5, resample, scr3

STOPS = (0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10, None)
COST = 0.0005
LIQ = 0.32
START = datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
DAY = 86_400_000


def trade(d: dict, i: int, side: int, opp: np.ndarray, stop: float | None) -> tuple | None:
    """(수익, 청산 봉, 최대 역행). 손절이 먼저면 손절, 아니면 반대 신호 다음 시가."""
    o, h, l = d["o"], d["h"], d["l"]  # noqa: E741
    e = i + 1
    px = o[e]
    line = None if stop is None else px * (1 - stop * side)
    mae = 0.0
    for j in range(e, len(o) - 1):
        adverse = (px - l[j]) / px if side > 0 else (h[j] - px) / px
        if line is not None and ((side > 0 and l[j] <= line) or (side < 0 and h[j] >= line)):
            fill = min(o[j], line) if side > 0 else max(o[j], line)
            g = (fill / px - 1) * side
            return g, j, max(mae, -g)
        mae = max(mae, adverse)
        if opp[j]:
            return (o[j + 1] / px - 1) * side, j + 1, mae
    return None


def summ(ts: list) -> dict:
    if not ts:
        return {"n": 0}
    g = np.array([x[1] for x in ts])
    mae = np.array([x[2] for x in ts])
    net1 = g - 2 * COST
    net3 = np.where(mae >= LIQ, -1.0, 3 * g - 6 * COST)
    out = {"n": len(ts), "win": round(float((net1 > 0.001).mean() * 100), 1),
           "long": int(sum(1 for x in ts if x[3] > 0)),
           "worst": round(float(net1.min() * 100), 2), "liq3": int((mae >= LIQ).sum())}
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
    sh = dict(d5)
    sh["t"] = d5["t"] - a.offset * 60_000
    d = resample(sh, 6)
    d["t"] = d["t"] + a.offset * 60_000
    n = len(d["c"])
    on_l, on_s, *_ = scr3(d)
    d1 = M.higher_dir(d5, 288, d["t"])
    end = d["t"][-1] + 1_800_000
    cuts = {"30d": end - 30 * DAY, "6m": end - 183 * DAY, "1y": end - 365 * DAY, "2022~": START}
    sigs = sorted([(int(i), s) for s, on in ((1, on_l), (-1, on_s)) for i in np.flatnonzero(on)
                   if d["t"][i] >= START and i < n - 2])
    rep: dict = {"offset": a.offset, "end": end}
    for gate in ("none", "D"):
        for stop in STOPS:
            key = f"{gate}|{'none' if stop is None else f'{stop * 100:g}%'}"
            busy, tr = -1, []
            for i, side in sigs:
                if i <= busy or (gate == "D" and d1[i] * side <= 0):
                    continue
                res = trade(d, i, side, on_s if side > 0 else on_l, stop)
                if res is None:
                    break
                g, x, mae = res
                tr.append((float(d["t"][i]), g, mae, side))
                busy = x
            r = {k: summ([z for z in tr if z[0] >= c]) for k, c in cuts.items()}
            years: dict = {}
            for z in tr:
                years.setdefault(datetime.fromtimestamp(z[0] / 1000, timezone.utc).year, []).append(z)
            r["years"] = {str(y): summ(v) for y, v in sorted(years.items())}
            rep[key] = r
            print(key, {k: (r[k].get("n"), r[k].get("win"), r[k].get("x1")) for k in cuts}, flush=True)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
