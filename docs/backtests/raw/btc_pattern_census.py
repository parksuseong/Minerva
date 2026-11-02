"""패턴 12종을 5분·15분·30분봉 전체에 대조: 빈도, 나오는 상황, 상황별 승률·손익비.

등록: 2026-11-02-btc-pattern-census.md
실행: PYTHONPATH=<numpy 경로>:src:docs/backtests/raw python docs/backtests/raw/btc_pattern_census.py --out <폴더>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import btc_book2 as K
import btc_swing_patterns as S
from btc_fractal import MIN5, load5, resample

FRAMES = {"5m": 1, "15m": 3, "30m": 6}
NAMES = {"DB": "이중바닥", "DBP": "첫눌림", "TB": "삼중바닥", "IHS": "역머리어깨", "GART": "가틀리",
         "BAT": "배트", "CRAB": "크랩", "BFLY": "버터플라이", "ABCD": "AB=CD", "3DRV": "쓰리드라이브",
         "GP": "골든포켓", "ATRI": "상승삼각"}


def daily_context(d5: dict) -> tuple:
    """일봉 종가가 50일선 위면 상승장. 그날 마감까지 아는 값만, 다음 날부터 적용."""
    dd = resample(d5, 288)
    c = dd["c"]
    s = np.full(len(c), np.nan)
    cs = np.cumsum(np.r_[0.0, c])
    s[49:] = (cs[50:] - cs[:-50]) / 50
    up = c > s
    close_t = dd["t"] + 86_400_000
    return close_t, up, np.isfinite(s)


def rows_for(d: dict, side: int, th: float) -> list:
    out = []
    base = S.zigzag(d["h"], d["l"], th)
    piv = base if side > 0 else [(i, -p, -k, cf) for i, p, k, cf in base]
    src = d if side > 0 else {"c": -d["c"]}
    for kind in ("DB", "DBP", "TB", "IHS"):
        for sig, stop, target, name in S.find(piv, src["c"], kind):
            if sig >= len(d["c"]) - 2:
                continue
            st, tg = (stop, target) if side > 0 else (-stop, -target)
            res = S.outcome(d, sig, side, st, tg)
            if res is None:
                continue
            g, why, px = res
            out.append((name, sig + 1, g, why, abs(tg - px) / abs(px - st)))
    d["c_side"] = d["c"] * side
    for name, start, lim, entry, stop, tgt, inval, how in K.scan(d, piv, side):
        if how == "limit":
            res = K.limit_trade(d, side, start, lim, entry * side, stop * side, tgt * side,
                                inval * side)
        else:
            res = K.next_open_trade(d, side, start, stop * side, tgt * side)
        if res is None:
            continue
        g, why, px, at = res
        st = stop * side
        out.append((name, at, g, why, abs(tgt * side - px) / abs(px - st) if px != st else 0.0))
    return out


def stat(g: np.ndarray, rr: np.ndarray, cost: float = S.COST[0]) -> dict:
    if not len(g):
        return {"n": 0}
    net = g - 2 * cost
    return {"n": int(len(g)), "win": round(float((net > S.WIN).mean() * 100), 1),
            "rr": round(float(np.median(rr)), 2), "ev": round(float(net.mean() * 100), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path("~/.minerva/data/btc").expanduser()))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d5 = load5(Path(args.data) / "5m.csv")
    ct, up, ok = daily_context(d5)
    rep: dict = {}
    for fname, f in FRAMES.items():
        d = resample(d5, f) if f > 1 else {k: v.copy() for k, v in d5.items()}
        S.CAP = K.CAP = 48 * 6 // f  # 하루 상한을 봉 크기에 맞춘다
        n = len(d["c"])
        days = (d["t"][-1] - d["t"][0]) / 86_400_000
        lr = np.r_[0.0, np.diff(np.log(d["c"]))]
        vol = np.full(n, np.nan)
        vol[47:] = np.lib.stride_tricks.sliding_window_view(lr, 48).std(1)
        # 변동성 상하: 직전 30일(봉 수 환산) 중앙값보다 높으면 '큰 변동성'
        win = int(30 * 288 / f)
        med = np.full(n, np.nan)
        step = max(win // 30, 1)
        for i in range(win, n, step):
            med[i:i + step] = np.nanmedian(vol[i - win:i])
        hi_vol = vol > med
        j = np.searchsorted(ct, d["t"], side="right") - 1
        bull = np.where(j >= 0, up[np.maximum(j, 0)], False) & (j >= 0) & ok[np.maximum(j, 0)]
        hour = ((d["t"] // 3_600_000) + 9) % 24  # KST
        fr: dict = {"days": round(float(days), 1)}
        for th in S.THRESH:
            for side in (1, -1):
                for name, at, g, why, rr in rows_for(d, side, th):
                    key = f"{name}_{'L' if side > 0 else 'S'}"
                    fr.setdefault(key, {}).setdefault(th, []).append(
                        (at, g, rr, bool(bull[at]), bool(hi_vol[at]) if np.isfinite(med[at]) else None,
                         int(hour[at]), float(d["t"][at])))
            print(fname, th, flush=True)
        summary = {}
        for key, by_th in fr.items():
            if key == "days":
                continue
            for th, rs in by_th.items():
                at = np.array([r[0] for r in rs])
                g = np.array([r[1] for r in rs])
                rr = np.array([r[2] for r in rs])
                bu = np.array([r[3] for r in rs])
                hv = np.array([r[4] is True for r in rs])
                hrs = np.array([r[5] for r in rs])
                t = np.array([r[6] for r in rs])
                new = t >= S.SPLIT
                summary[f"{key}_{int(th * 100)}"] = {
                    "per_day": round(len(rs) / days, 3),
                    "all": stat(g, rr),
                    "c10": stat(g, rr, S.COST[1]),
                    "old": stat(g[~new], rr[~new]),
                    "new": stat(g[new], rr[new]),
                    "bull_share": round(float(bu.mean() * 100), 1),
                    "hivol_share": round(float(hv.mean() * 100), 1),
                    "bull": stat(g[bu], rr[bu]), "bear": stat(g[~bu], rr[~bu]),
                    "hivol": stat(g[hv], rr[hv]), "lovol": stat(g[~hv], rr[~hv]),
                    "hour_top": [int(x) for x in np.argsort(-np.bincount(hrs, minlength=24))[:3]],
                }
        rep[fname] = {"days": fr["days"], "cells": summary}
        del d
    (out / "result.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
