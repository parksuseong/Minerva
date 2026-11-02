"""SCR·이평·캔들 위치 상황별로 앞으로 하루 안에 어떤 패턴이 완성되는지, 그 패턴의 성과는 어떤지.

등록: 2026-11-02-btc-pattern-context-hypotheses.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_pattern_context.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import btc_book2 as K
import btc_pattern_census as P
import btc_swing_patterns as S
from btc_fractal import cci, load5, resample, rsi, sma, stoch

H = 48
SIZES = (0.02, 0.04)
SCR = ("과매도", "하방", "중립", "상방", "과매수")
MA = ("역배열", "60선아래", "60선위", "정배열")
BB = ("하단밖", "아래절반", "위절반", "상단밖")


def states(d: dict) -> np.ndarray:
    c = d["c"]
    s, cc, r = stoch(d["h"], d["l"], c), cci(d["h"], d["l"], c), rsi(c)
    scr = np.full(len(c), 2)
    scr[(cc > 0) & (r > 50)] = 3
    scr[(cc < 0) & (r < 50)] = 1
    scr[(s < 20) & (cc < -100)] = 0
    scr[(s > 80) & (cc > 100)] = 4
    m20, m60, m120 = sma(c, 20), sma(c, 60), sma(c, 120)
    ma = np.where(c > m60, 2, 1)
    ma[(c > m20) & (m20 > m60) & (m60 > m120)] = 3
    ma[(c < m20) & (m20 < m60) & (m60 < m120)] = 0
    sd = np.full(len(c), np.nan)
    sd[19:] = np.lib.stride_tricks.sliding_window_view(c, 20).std(1)
    up, lo = m20 + 2 * sd, m20 - 2 * sd
    bb = np.where(c >= m20, 2, 1)
    bb[c > up] = 3
    bb[c < lo] = 0
    st = scr * 16 + ma * 4 + bb
    st[~np.isfinite(m120 + sd + cc + r + s)] = -1
    return st


def name_of(code: int) -> str:
    return f"{SCR[code // 16]}·{MA[(code // 4) % 4]}·{BB[code % 4]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = resample(load5(Path(args.data) / "5m.csv"), 6)
    S.CAP = K.CAP = 48
    n = len(d["c"])
    st = states(d)
    split = int(np.searchsorted(d["t"], S.SPLIT))
    pats = []
    for th in SIZES:
        for side in (1, -1):
            for name, at, g, why, rr in P.rows_for(d, side, th):
                pats.append((f"{name}_{'L' if side > 0 else 'S'}", int(at), float(g), float(rr)))
    kinds = sorted({p[0] for p in pats})
    # 패턴이 완성(진입)된 봉 표시. 상황 봉 t 뒤 (t, t+H] 안에 완성이 있으면 1
    hit = {k: np.zeros(n, bool) for k in kinds}
    for k, at, g, rr in pats:
        hit[k][at] = True
    ahead = {}
    for k in kinds:
        cs = np.cumsum(np.r_[0, hit[k].astype(int)])
        idx = np.arange(n)
        hi = np.minimum(idx + H, n - 1)
        ahead[k] = (cs[hi + 1] - cs[idx + 1]) > 0
    valid = (st >= 0) & (np.arange(n) < n - H)
    rep: dict = {"kinds": kinds, "periods": {}}
    for per, mask in (("old", valid & (np.arange(n) < split)), ("new", valid & (np.arange(n) >= split))):
        base = {k: float(ahead[k][mask].mean()) for k in kinds}
        cells = {}
        for code in np.unique(st[mask]):
            m = mask & (st == code)
            if m.sum() < 200:
                continue
            row = {"bars": int(m.sum())}
            for k in kinds:
                p = float(ahead[k][m].mean())
                row[k] = [round(p * 100, 2), round(p / base[k], 2) if base[k] > 0 else None]
            cells[name_of(int(code))] = row
        rep["periods"][per] = {"base": {k: round(v * 100, 2) for k, v in base.items()}, "cells": cells}
    # 안정성: 상황×패턴 배수(lift)가 두 기간에 같은 방향인가
    o, w = rep["periods"]["old"]["cells"], rep["periods"]["new"]["cells"]
    common = [c for c in o if c in w and w[c]["bars"] >= 200]
    xs, ys = [], []
    for c in common:
        for k in kinds:
            a, b = o[c][k][1], w[c][k][1]
            if a and b and o[c]["bars"] * o[c][k][0] / 100 >= 20:
                xs.append(np.log(a))
                ys.append(np.log(b))
    xs, ys = np.array(xs), np.array(ys)
    rng = np.random.default_rng(20261102)
    real = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 10 else None
    null = [float(np.corrcoef(xs, rng.permutation(ys))[0, 1]) for _ in range(1000)] if real else []
    rep["stability"] = {"pairs": int(len(xs)), "corr": round(real, 3) if real else None,
                        "p": round((1 + sum(v >= real for v in null)) / 1001, 4) if real else None,
                        "sign_agree": round(float((np.sign(xs) == np.sign(ys)).mean() * 100), 1)
                        if len(xs) else None}
    # 상황별 패턴 성과(완성 봉의 상황 기준). 같은 기준(30회+, 두 기간 플러스, 0.10% 플러스)
    perf = {}
    for k in kinds:
        rows = [(at, g, rr) for kk, at, g, rr in pats if kk == k and st[at - 1] >= 0]
        for code in sorted({int(st[at - 1]) for at, _, _ in rows}):
            rs = [(at, g, rr) for at, g, rr in rows if st[at - 1] == code]
            g = np.array([x[1] for x in rs])
            new = np.array([x[0] >= split for x in rs])
            if len(rs) < 30:
                continue
            net05, net10 = g - 0.001, g - 0.002
            perf[f"{k}|{name_of(code)}"] = {
                "n": len(rs), "win": round(float((net05 > S.WIN).mean() * 100), 1),
                "rr": round(float(np.median([x[2] for x in rs])), 2),
                "ev": round(float(net05.mean() * 100), 3), "ev10": round(float(net10.mean() * 100), 3),
                "old": [int((~new).sum()), round(float(net05[~new].mean() * 100), 3) if (~new).any() else None],
                "new": [int(new.sum()), round(float(net05[new].mean() * 100), 3) if new.any() else None],
            }
    rep["perf"] = perf
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep["stability"], ensure_ascii=False), len(perf), flush=True)


if __name__ == "__main__":
    main()
