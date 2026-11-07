"""최근 7일 SCR 단독 규칙 표본 안 최적화(사용자 요청 2026-11-07). 판정 시험이 아니다.

문서: 2026-11-07-btc-scr-week-optimize.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_week_opt.py <출력 폴더>
"""

from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from btc_fractal import cci, crossed, load5, recent, resample, rsi, stoch
from btc_scr_gate15 import load_funding

M5 = 300_000
COST = 0.0005
FRAMES = {"15m": 3, "30m": 6, "1h": 12}
ENTRIES = ("SCR3", "SCR3c9", "SCR2", "SC", "R", "REL")
GATES = ("none", "4h", "1d")
TPSL = [(tp, sl) for tp in (0.005, 0.01, 0.015, 0.02, 0.03) for sl in (0.005, 0.01, 0.015, 0.02)]
EXITS = [("OPP",), ("RSIX",)] + [("T", n) for n in (4, 8, 16)] + [("TP", tp, sl) for tp, sl in TPSL]
HARD = 0.01


def ms(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000


W7 = (ms("2026-10-31 05:30"), None)
Y1 = (ms("2025-10-31 05:30"), ms("2026-10-31 05:30"))
Y24 = (ms("2024-01-01 00:00"), None)


def signals(d: dict, kind: str):
    h, l, c = d["h"], d["l"], d["c"]
    s, r = stoch(h, l, c, 5), rsi(c, 14)
    cc = cci(h, l, c, 9 if kind == "SCR3c9" else 20)
    if kind == "REL":
        lv = (25, 75, -80, 80, 35, 65)
    else:
        lv = (20, 80, -100, 100, 30, 70)
    sl_, sh_, cl_, ch_, rl_, rh_ = lv
    a = [recent(crossed(s, sl_, True)), recent(crossed(cc, cl_, True)), recent(crossed(r, rl_, True))]
    b = [recent(crossed(s, sh_, False)), recent(crossed(cc, ch_, False)), recent(crossed(r, rh_, False))]
    if kind in ("SCR3", "SCR3c9", "REL"):
        L, S = a[0] & a[1] & a[2], b[0] & b[1] & b[2]
    elif kind == "SCR2":
        L = (a[0].astype(int) + a[1] + a[2]) >= 2
        S = (b[0].astype(int) + b[1] + b[2]) >= 2
    elif kind == "SC":
        L, S = a[0] & a[1], b[0] & b[1]
    else:
        L, S = crossed(r, 30, True), crossed(r, 70, False)
    L = L & ~np.r_[False, L[:-1]]
    S = S & ~np.r_[False, S[:-1]]
    return L, S, s, cc, r


def direction(u: dict) -> np.ndarray:
    c_, r_ = cci(u["h"], u["l"], u["c"], 20), rsi(u["c"], 14)
    with np.errstate(invalid="ignore"):
        return np.where((c_ > 0) & (r_ > 50), 1, np.where((c_ < 0) & (r_ < 50), -1, 0))


def gate_dir(d5: dict, d: dict, frame: int, k: int) -> np.ndarray:
    u = resample(d5, k)
    close = u["t"] + k * M5
    j = np.searchsorted(close, d["t"] + frame * M5, side="right") - 1
    out = np.zeros(len(d["t"]), int)
    dr = direction(u)
    out[j >= 0] = dr[j[j >= 0]]
    return out


def simulate(d, L, S, r, gate, ex, i0, i1, fund) -> list:
    """한 번에 하나. 진입은 신호 봉 종가. 손익(비용·펀딩 포함), 진입 시각, 청산 시각, 방향, 진입·청산 봉."""
    o, h, l, c, t = d["o"], d["h"], d["l"], d["c"], d["t"]
    sig = np.where(L, 1, np.where(S, -1, 0))
    if gate is not None:
        sig = np.where(sig == gate, sig, 0)
    idx = np.flatnonzero(sig[i0:i1]) + i0
    out, busy = [], -1
    for i in idx:
        if i <= busy or i >= i1 - 1:
            continue
        sd = int(sig[i])
        px = c[i]
        if ex[0] == "TP":
            tp, sl = ex[1], ex[2]
        else:
            tp, sl = None, HARD
        stop = px * (1 - sl * sd)
        tgt = px * (1 + tp * sd) if tp else None
        xi, xp = None, None
        lim = i1 if ex[0] != "T" else min(i1, i + 1 + ex[1])
        for j in range(i + 1, lim):
            if (sd > 0 and l[j] <= stop) or (sd < 0 and h[j] >= stop):
                xp = min(o[j], stop) if sd > 0 else max(o[j], stop)
                xi = j
                break
            if tgt is not None and ((sd > 0 and h[j] >= tgt) or (sd < 0 and l[j] <= tgt)):
                xp = max(o[j], tgt) if sd > 0 else min(o[j], tgt)
                xi = j
                break
            if ex[0] == "OPP" and sig_raw_opp(L, S, j, sd):
                xi, xp = j, c[j]
                break
            if ex[0] == "RSIX" and ((sd > 0 and r[j] >= 70) or (sd < 0 and r[j] <= 30)):
                xi, xp = j, c[j]
                break
        if xi is None:
            if ex[0] == "T" and i + ex[1] < i1:
                xi = i + ex[1]
                xp = c[xi]
            else:
                xi = i1 - 1
                xp = c[xi]
        g = (xp / px - 1) * sd
        m = (fund[0] > t[i]) & (fund[0] <= t[xi])
        g -= sd * float(fund[1][m].sum())
        out.append((g - 2 * COST, float(t[i]), float(t[xi]), sd, int(i), int(xi), float(px), float(xp)))
        busy = xi
    return out


def sig_raw_opp(L, S, j, sd) -> bool:
    return bool(S[j]) if sd > 0 else bool(L[j])


def summ(tr: list) -> dict:
    if not tr:
        return {"n": 0}
    g = np.array([z[0] for z in tr])
    eq = np.cumprod(1 + g)
    pk = np.maximum.accumulate(np.r_[1.0, eq])
    eq3 = np.cumprod(1 + np.maximum(3 * g, -1))
    pk3 = np.maximum.accumulate(np.r_[1.0, eq3])
    return {"n": len(tr), "long": int(sum(z[3] > 0 for z in tr)), "win": round(float((g > 0).mean() * 100), 1),
            "avg": round(float(g.mean() * 100), 3), "best": round(float(g.max() * 100), 2),
            "worst": round(float(g.min() * 100), 2), "ret": round(float((eq[-1] - 1) * 100), 2),
            "ret3": round(float((eq3[-1] - 1) * 100), 2),
            "mdd": round(float((eq / pk[1:] - 1).min() * 100), 2),
            "mdd3": round(float((eq3 / pk3[1:] - 1).min() * 100), 2)}


def rng(d, a, b):
    t = d["t"]
    i0 = int(np.searchsorted(t, a))
    i1 = len(t) if b is None else int(np.searchsorted(t, b))
    return i0, i1


def ename(ex) -> str:
    if ex[0] == "TP":
        return f"TP{ex[1] * 100:g}/SL{ex[2] * 100:g}"
    if ex[0] == "T":
        return f"T{ex[1]}+SL1"
    return {"OPP": "OPP+SL1", "RSIX": "RSIX+SL1"}[ex[0]]


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    base = Path("~/.minerva/data/btc/futures").expanduser()
    d5 = load5(base / "klines_5m.csv")
    keep = d5["t"] >= ms("2023-09-01 00:00")
    d5 = {k: v[keep] for k, v in d5.items()}
    fund = load_funding(base / "funding.csv")
    data = {}
    for fn, k in FRAMES.items():
        d = resample(d5, k)
        g4 = gate_dir(d5, d, k, 48)
        gd = gate_dir(d5, d, k, 288)
        data[fn] = (d, {kind: signals(d, kind) for kind in ENTRIES}, {"none": None, "4h": g4, "1d": gd})
    rows = []
    for fn, kind, gn, ex in itertools.product(FRAMES, ENTRIES, GATES, EXITS):
        d, sg, gt = data[fn]
        L, S, s, cc, r = sg[kind]
        i0, i1 = rng(d, *W7)
        tr = simulate(d, L, S, r, gt[gn], ex, i0, i1, fund)
        rows.append({"frame": fn, "entry": kind, "gate": gn, "exit": ename(ex), "ex": list(ex), "w7": summ(tr)})
    total = len(rows)
    pos = [x for x in rows if x["w7"].get("n", 0) >= 4 and x["w7"]["ret"] > 0]
    for x in pos:
        d, sg, gt = data[x["frame"]]
        L, S, s, cc, r = sg[x["entry"]]
        ex = tuple(x["ex"])
        for key, win in (("y1", Y1), ("y24", Y24)):
            i0, i1 = rng(d, *win)
            x[key] = summ(simulate(d, L, S, r, gt[x["gate"]], ex, i0, i1, fund))
    pos_y1 = sum(1 for x in pos if x["y1"].get("n", 0) and x["y1"]["avg"] > 0)
    ranked = sorted(pos, key=lambda x: -x["w7"]["ret"])
    picks, seen = [], {}
    for x in ranked:
        fam = (x["frame"], x["entry"])
        if seen.get(fam, 0) >= 1 and len(picks) < 8:
            continue
        seen[fam] = seen.get(fam, 0) + 1
        picks.append(x)
        if len(picks) == 8:
            break
    if len(picks) < 8:
        for x in ranked:
            if x not in picks:
                picks.append(x)
            if len(picks) == 8:
                break
    for k, x in enumerate(picks, 1):
        d, sg, gt = data[x["frame"]]
        L, S, s, cc, r = sg[x["entry"]]
        i0, i1 = rng(d, *W7)
        x["rank"] = k
        x["trades7"] = [list(z) for z in simulate(d, L, S, r, gt[x["gate"]], tuple(x["ex"]), i0, i1, fund)]
    rep = {"combos": total, "w7_min4": sum(1 for x in rows if x["w7"].get("n", 0) >= 4),
           "w7_positive": len(pos), "w7_pos_and_y1_pos": pos_y1,
           "window7": [W7[0], float(data["15m"][0]["t"][-1])], "picks": picks,
           "top20": [{k: v for k, v in x.items() if k != "trades7"} for x in ranked[:20]]}
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(total, len(pos), pos_y1)


if __name__ == "__main__":
    main()
