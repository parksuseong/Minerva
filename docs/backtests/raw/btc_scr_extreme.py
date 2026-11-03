"""30분봉 SCR 3단을 고점·저점 근처에서만 받기. 등록: 2026-11-03-btc-scr-extreme-hypotheses.md

실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_scr_extreme.py --out <폴더> [--offset 15]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from btc_fractal import load5, resample, scr3, stoch, win_view
from btc_scr_wait_fast import DAY, M5, START, on_grid, summ

LOOK = 96


def frame30(d5: dict, offset: int) -> dict:
    sh = dict(d5)
    sh["t"] = d5["t"] - offset * 60_000
    d = resample(sh, 6)
    d["t"] = d["t"] + offset * 60_000
    return d


def filters(d5: dict, d: dict) -> dict:
    """30분봉 신호 위치별 필터 통과 여부 (롱 배열, 숏 배열)."""
    on_l, on_s, *_ = scr3(d)
    n = len(d["c"])
    hh = np.full(n, np.nan)
    ll = np.full(n, np.nan)
    hh[LOOK - 1 :] = win_view(d["h"], LOOK).max(1)
    ll[LOOK - 1 :] = win_view(d["l"], LOOK).min(1)
    pos = (d["c"] - ll) / np.where(hh > ll, hh - ll, np.nan)
    h4 = resample(d5, 48)
    st4 = stoch(h4["h"], h4["l"], h4["c"])
    j = np.searchsorted(h4["t"] + 48 * M5, d["t"] + 6 * M5, side="right") - 1
    s4 = np.where(j >= 0, st4[np.maximum(j, 0)], np.nan)
    second = {}
    for side, on in ((1, on_l), (-1, on_s)):
        idx = np.flatnonzero(on)
        ok = np.zeros(n, dtype=bool)
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a <= LOOK and (d["c"][b] - d["c"][a]) * side < 0:
                ok[b] = True
        second[side] = ok
    out = {}
    for name in ("E0", "E1", "E2", "E3", "E4"):
        res = []
        for side, on in ((1, on_l), (-1, on_s)):
            e1 = pos <= 0.2 if side > 0 else pos >= 0.8
            e2 = s4 <= 20 if side > 0 else s4 >= 80
            e3 = second[side]
            m = {"E0": on, "E1": on & e1, "E2": on & e2, "E3": on & e3, "E4": on & e1 & e3}[name]
            res.append(m)
        out[name] = tuple(res)
    return out, (on_l, on_s)


def to_grid(d5: dict, d: dict, mask: np.ndarray) -> np.ndarray:
    g = np.zeros(len(d5["t"]), dtype=bool)
    j = np.searchsorted(d5["t"], d["t"][mask] + 6 * M5)
    g[j[j < len(g)]] = True
    return g


def simulate(d5, ent, ex30, ex5, fast, stop, j0) -> list:
    o, h, l, c, t = d5["o"], d5["h"], d5["l"], d5["c"], d5["t"]  # noqa: E741
    EL, ES = ent
    XL, XS = ex30
    FL, FS = ex5
    tr, side, px, line, mae, ej = [], 0, 0.0, 0.0, 0.0, 0
    for j in range(j0, len(o) - 1):
        if side:
            ex = None
            if XS[j] if side > 0 else XL[j]:
                ex = o[j]
            elif fast and (FS[j] if side > 0 else FL[j]) and (c[j - 1] / px - 1) * side > 0.001:
                ex = o[j]
            if ex is None and (l[j] <= line if side > 0 else h[j] >= line):
                ex = min(o[j], line) if side > 0 else max(o[j], line)
            if ex is None:
                mae = max(mae, (px - l[j]) / px if side > 0 else (h[j] - px) / px)
                continue
            g = (ex / px - 1) * side
            tr.append((float(t[ej]), g, max(mae, -g), side, float(t[j]), float(px)))
            side = 0
            continue
        s = 1 if EL[j] else (-1 if ES[j] else 0)
        if not s:
            continue
        side, px, ej = s, o[j], j
        line = px * (1 - stop * side)
        if l[j] <= line if side > 0 else h[j] >= line:
            g = (line / px - 1) * side
            tr.append((float(t[j]), g, -g, side, float(t[j]), float(px)))
            side = 0
        else:
            mae = (px - l[j]) / px if side > 0 else (h[j] - px) / px
    return tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(a.data) / "5m.csv")
    d = frame30(d5, a.offset)
    flt, raw = filters(d5, d)
    ex30 = tuple(to_grid(d5, d, m) for m in raw)
    ex5 = on_grid(d5, 1, 0)
    j0 = int(np.searchsorted(d5["t"], START))
    end = d5["t"][-1] + M5
    cuts = {"30d": end - 30 * DAY, "6m": end - 183 * DAY, "1y": end - 365 * DAY, "2022~": START}
    rep: dict = {"offset": a.offset}
    for name, st, fa in itertools.product(flt, (0.01, 0.03), (False, True)):
        key = f"{name}|SL{st * 100:g}%|F{int(fa)}"
        ent = tuple(to_grid(d5, d, m) for m in flt[name])
        tr = simulate(d5, ent, ex30, ex5, fa, st, j0)
        r = {k: summ([z for z in tr if z[0] >= c]) for k, c in cuts.items()}
        r["trades"] = [[z[0], round(z[1], 5), z[3], z[4], z[5]] for z in tr if z[0] >= cuts["6m"]]
        rep[key] = r
        print(key, {k: (r[k].get("n"), r[k].get("win"), r[k].get("x1")) for k in cuts}, flush=True)
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
